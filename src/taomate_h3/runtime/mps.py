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
from .mps_memory import mps_memory_receipt, reset_mps_peak_memory


class MPSRuntime(H3Runtime):
    def __init__(
        self, *, model_root: Path, adapter_root: Path, output_root: Path, ulysses_degree: int = 1
    ) -> None:
        from ..distributed import ParallelContext
        from ..model.pipeline import MiniMaxH3NativePipeline
        from ..streaming.metal_kv import (
            MAX_KV_TILE_SCRATCH_BYTES,
            kv_tile_scratch_bound,
            parse_kv_tile_tokens,
            parse_kv_allocation,
        )

        self._kv_allocation = parse_kv_allocation(
            os.environ.get("TAOMATE_MPS_KV_ALLOCATION", "per_layer")
        )
        self._kv_tile_tokens = parse_kv_tile_tokens(
            os.environ.get("TAOMATE_MPS_KV_TILE_TOKENS", "128")
        )
        from ..inference.metal_loader import parse_weight_tile_mib

        self._weight_tile_bytes = parse_weight_tile_mib(
            os.environ.get("TAOMATE_MPS_WEIGHT_TILE_MIB", "256")
        )
        require_mps()
        query_chunk = int(os.environ.get("TAOMATE_MPS_ATTENTION_QUERY_CHUNK", "0"))
        if query_chunk < 0:
            raise ValueError("Metal attention query chunk must be nonnegative")
        self._quantization_backend = os.environ.get("TAOMATE_MPS_WEIGHT_QUANTIZATION", "none")
        if self._quantization_backend not in ("none", "dequantize", "native_int8pack"):
            raise ValueError("unknown Metal weight quantization backend")
        self._cpu_adaln = os.environ.get("TAOMATE_MPS_CPU_ADALN", "0") == "1"
        if self._cpu_adaln and os.environ.get("TAOMATE_MPS_RELEASE_ADALN", "0") != "1":
            raise ValueError("CPU AdaLN staging requires fixed-schedule precompute and release")
        self._kv_storage = os.environ.get("TAOMATE_MPS_KV_STORAGE", "bf16")
        if self._kv_storage not in ("bf16", "int8"):
            raise ValueError("unknown Metal KV storage")
        self._kv_tile_scratch_estimated_bytes = None
        if self._kv_storage == "int8":
            from ..model.architecture import MiniMaxH3Architecture

            arch = MiniMaxH3Architecture()
            self._kv_tile_scratch_estimated_bytes = kv_tile_scratch_bound(
                self._kv_tile_tokens, arch.num_attention_heads, arch.attention_head_dim
            )
            if self._kv_tile_scratch_estimated_bytes > MAX_KV_TILE_SCRATCH_BYTES:
                raise ValueError("KV tile scratch estimate exceeds 256 MiB budget")
        self._query_chunk_size = query_chunk or (128 if self._kv_storage == "int8" else None)
        self._empty_cache_phase = os.environ.get("TAOMATE_MPS_EMPTY_CACHE_PHASE", "0") == "1"
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
            transformer_metal_cpu_adaln=self._cpu_adaln,
            transformer_metal_quantization_backend=self._quantization_backend,
            transformer_metal_weight_tile_bytes=self._weight_tile_bytes,
        )
        self._text_precompute_receipt = None
        self._static_cache = None
        self._memory_optimizations = {}
        self.model_resolution = {
            "metal_workspace_options": {
                "inplace_activations": os.environ.get("TAOMATE_MPS_INPLACE_ACTIVATIONS") == "1",
                "attention_query_chunk_size": self._query_chunk_size,
                "empty_cache_between_phases": self._empty_cache_phase,
                "cpu_adaln_staging": self._cpu_adaln,
                "weight_quantization_backend": self._quantization_backend,
                "bf16_weight_tile_budget_bytes": (
                    self._weight_tile_bytes if self._quantization_backend == "dequantize" else None
                ),
                "weight_tile_budget_scope": (
                    "BF16 weights only; excludes FP32 scratch, outputs and matmul workspace"
                ),
                "kv_storage": self._kv_storage,
                "kv_allocation": self._kv_allocation if self._kv_storage == "int8" else None,
                "kv_tile_tokens": self._kv_tile_tokens if self._kv_storage == "int8" else None,
                "kv_tile_scratch_estimated_bytes": self._kv_tile_scratch_estimated_bytes,
                "kv_tile_scratch_budget_bytes": (
                    MAX_KV_TILE_SCRATCH_BYTES if self._kv_storage == "int8" else None
                ),
            },
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
        model._metal_release_adaln = os.environ.get("TAOMATE_MPS_RELEASE_ADALN", "0") == "1"
        model._metal_query_chunk_size = self._query_chunk_size
        model._metal_empty_cache_phase = self._empty_cache_phase
        model._metal_kv_allocation = self._kv_allocation
        model._metal_kv_storage = self._kv_storage
        model._metal_kv_tile_tokens = self._kv_tile_tokens
        model._metal_kv_max_joined_bytes = 4 * 1024**3
        self.model_resolution["metal_weight_loading"] = self.pipeline.dit_weight_receipt
        if self._quantization_backend != "none":
            self.model_resolution["precision"] = (
                "selected W8A16, BF16 LoRA, official FP32 patch/time/final parameters"
            )
        return encoded

    def reset_dit_peak_memory(self) -> None:
        self._synchronize()
        self._dit_peak_reset_succeeded = reset_mps_peak_memory()

    def dit_peak_memory_receipt(self) -> dict[str, Any]:
        self._synchronize()
        return {
            "backend": "mps",
            **mps_memory_receipt(),
            "peak_reset_succeeded": getattr(self, "_dit_peak_reset_succeeded", None),
            "peak_scope": (
                "since_dit_reset" if getattr(self, "_dit_peak_reset_succeeded", False)
                else "allocator_since_external_reset_or_initialization"
            ),
        }

    def release_transformer(self) -> None:
        self._synchronize()
        self._static_cache = self.pipeline.transformer.inference_static_timestep_cache_stats()
        self._memory_optimizations["adaln_release"] = getattr(
            self.pipeline.transformer, "_metal_adaln_release_receipt", None
        )
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
        publication_chunk = 4 if os.environ.get("TAOMATE_MPS_STREAM_PUBLICATION") == "1" else None
        try:
            frames = decoder.decode_video(
                video_latents=video_latents,
                device=self.device,
                stream_publication_frame_chunk_size=publication_chunk,
                offload_after_decode=False,
            )
        finally:
            # One-shot MPS decoding: discard GPU weights directly rather than
            # copying them to CPU immediately before disposal, including failures.
            decoder.video_vae = None
            gc.collect()
            empty_cache(self.device)
        result = save_h3_video_mp4(
            output_path,
            frames=frames,
            video_filter=video_filter,
            published_frames=published_frames,
            crf=video_crf,
            verify_decode=verify_video_decode,
            frame_chunk_size=publication_chunk,
        )
        self._video_vae_receipt = dict(decoder.last_video_decode_receipt or {})
        self._video_vae_receipt.update(
            backend="mps",
            precision="float32_reference_no_autocast",
            mode="single_device_spatial_tiling",
            reason="temporal_window_parallelism_disabled_for_mps",
            stream_publication=publication_chunk is not None,
            publication_frame_chunk_size=publication_chunk,
        )
        self._timings["video_decode_seconds"] = time.perf_counter() - start
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
            "memory_optimizations": dict(self._memory_optimizations),
        }
