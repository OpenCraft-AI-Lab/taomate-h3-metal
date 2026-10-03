# Modified for TaoMate-H3 Metal inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Single-process Metal backend for the official Python pipeline.

Uses the model's BF16 torch reference branches, retaining FP32 patch/time/final
parameters. CUDA W8A8/Triton/FA3 acceleration is intentionally not emulated.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .backend import empty_cache, progress, require_mps, synchronize
from .direct import H3Runtime, _distribution_version


class MPSRuntime(H3Runtime):
    def __init__(
        self, *, model_root: Path, adapter_root: Path, output_root: Path, ulysses_degree: int = 1
    ) -> None:
        from ..distributed import ParallelContext
        from ..model.pipeline import MiniMaxH3NativePipeline

        require_mps()
        for key in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ.setdefault(key, "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if ulysses_degree != 1:
            raise ValueError("MPS requires Ulysses1")
        self._worker_started_at = time.perf_counter()
        self._timings = {}
        self.device = torch.device("mps")
        self.rank = self.local_rank = 0
        self.world_size = self.tp_size = self.ulysses_degree = self.expected_world_size = 1
        self.parallel_context = ParallelContext(1, 0, 1, 0, None, None)
        self.model_root = model_root.resolve(strict=True)
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.adapter_root = adapter_root.resolve(strict=True)
        self.pipeline = MiniMaxH3NativePipeline(
            model_root=self.model_root,
            device=self.device,
            parallel_context=self.parallel_context,
            rank=0,
            world_size=1,
            transformer_load_w8a8_policy=None,
        )
        self._text_precompute_receipt = None
        self._static_cache = None
        self.model_resolution = {
            "backend": "pytorch_mps",
            "dependencies": {
                name: _distribution_version(name)
                for name in ("torch", "diffusers", "transformers", "safetensors", "numpy")
            },
            "partition": "FL2VA",
            "model_root": str(self.model_root),
            "precision": "BF16 reference with official FP32 patch/time/final parameters",
            "attention": "torch_sdpa",
            "parallelism": {"tensor_parallel_size": 1, "ulysses_degree": 1, "world_size": 1},
            "acceleration": {"w8a8": False, "triton": False, "flash_attn_3": False},
        }

    def _synchronize(self) -> None:
        synchronize(self.device)

    def barrier(self) -> None:
        return None

    def broadcast(self, value: Any) -> Any:
        return value

    def _collective_error(self, label: str, local_error: str | None) -> str | None:
        return None if local_error is None else f"{label}: {local_error}"

    def precompute_text_prompts(self, prompts: Sequence[str]) -> tuple[tuple[Any, Any], ...]:
        from ..inference.lora_checkpoint import (
            apply_h3_lora_checkpoint,
            materialize_h3_lora_bf16_buffers_,
        )

        encoded = super().precompute_text_prompts(prompts)
        self._timings.update(self.pipeline.component_timings)
        self._timings["text_precompute_seconds"] -= self.pipeline.component_timings[
            "dit_load_seconds"
        ]
        model = self.pipeline.transformer
        self.model_resolution["adapter"] = apply_h3_lora_checkpoint(model, self.adapter_root)
        self.model_resolution["lora_bf16_buffers"] = materialize_h3_lora_bf16_buffers_(model)
        model.enable_inference_static_timestep_cache()
        return encoded

    def reset_dit_peak_memory(self) -> None:
        self._synchronize()

    def dit_peak_memory_receipt(self) -> dict[str, Any]:
        self._synchronize()
        return {
            "backend": "mps",
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "current_allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "note": "MPS API exposes current allocations, not a lifetime peak",
        }

    def release_transformer(self) -> None:
        self._synchronize()
        self._static_cache = self.pipeline.transformer.inference_static_timestep_cache_stats()
        self.pipeline.transformer = None
        progress("dit_released_before_vae")
        gc.collect()
        empty_cache(self.device)

    def decode_video_latents(
        self,
        video_latents: torch.Tensor,
        *,
        output_path: Path,
        video_filter: str | None = None,
        published_frames: int | None = None,
        video_crf: int = 12,
        verify_video_decode: bool = True,
    ) -> Path:
        from ..model.output import save_h3_video_mp4

        start = time.perf_counter()
        progress("video_vae_start", precision="float32_reference_no_autocast")
        decoder = self.pipeline._load_output_decoder()
        frames = decoder.decode_video(video_latents=video_latents, device=self.device)
        result = save_h3_video_mp4(
            output_path,
            frames=frames,
            video_filter=video_filter,
            published_frames=published_frames,
            crf=video_crf,
            verify_decode=verify_video_decode,
        )
        self._video_vae_receipt = dict(decoder.last_video_decode_receipt or {})
        self._video_vae_receipt.update(
            backend="mps",
            precision="float32_reference_no_autocast",
            mode="single_device_spatial_tiling",
            reason="temporal_window_parallelism_disabled_for_mps",
        )
        self._timings["video_decode_seconds"] = time.perf_counter() - start
        # Release video weights before audio is moved to MPS.
        decoder.video_vae = None
        del frames
        gc.collect()
        empty_cache(self.device)
        return result

    def decode_audio_latents(self, audio_latents: torch.Tensor) -> tuple[torch.Tensor, int]:
        start = time.perf_counter()
        progress("audio_vae_start", precision="float32")
        result = self.pipeline._load_output_decoder().decode_audio(
            audio_latents=audio_latents, device=self.device
        )
        self._timings["audio_vae_seconds"] = time.perf_counter() - start
        self.pipeline.output_decoder = None
        gc.collect()
        empty_cache(self.device)
        progress("audio_vae_complete", seconds=self._timings["audio_vae_seconds"])
        return result

    def performance_receipt(self) -> dict[str, Any]:
        return {
            "backend": "mps",
            "timings_seconds": dict(self._timings),
            "worker_cold_e2e_seconds": time.perf_counter() - self._worker_started_at,
            "static_timestep_adaln_cache": self._static_cache,
            "memory": self.dit_peak_memory_receipt(),
        }
