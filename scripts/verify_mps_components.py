# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Opt-in real-weight Metal smoke for the official bundled VAEs; no downloads."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import resource
import time
from pathlib import Path

import torch

from taomate_h3.model.output import H3LatentDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    root = args.model_root / "FL2VA"
    receipt = {
        "kind": "official_weight_component_smoke",
        "device": "mps",
        "torch": torch.__version__,
        "video_precision": "float32_reference",
        "video_input_shape": [1, 24, 7, 4, 4],
        "audio_input_shape": [2, 32, 10],
        "phases": [],
    }
    start = time.perf_counter()

    def phase(name, **extra):
        torch.mps.synchronize()
        row = {
            "phase": name,
            "elapsed_seconds": time.perf_counter() - start,
            "allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            **extra,
        }
        receipt["phases"].append(row)
        print(json.dumps(row), flush=True)

    with args.gpu_lock.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            phase("loading_bundled_vaes")
            decoder = H3LatentDecoder(
                video_vae_root=root / "video_vae",
                audio_vae_root=root / "audio_vae",
                video_parallel_group=None,
                video_parallel_rank=0,
                video_parallel_world_size=1,
                load_audio=True,
            )
            decoder.enable_video_vae_batched_tiles()
            decoder.enable_video_vae_cpu_temporal_output()
            phase("loaded_bundled_vaes")
            with torch.inference_mode():
                video = decoder.decode_video(
                    video_latents=torch.zeros(*receipt["video_input_shape"]),
                    device=torch.device("mps"),
                )
            phase("video_decoded", shape=list(video.shape), finite=bool(video.isfinite().all()))
            assert list(video.shape) == [1, 3, 22, 64, 64]
            assert video.isfinite().all()
            decoder.video_vae = None
            del video
            gc.collect()
            torch.mps.empty_cache()
            phase("video_weights_released")
            with torch.inference_mode():
                audio, sample_rate = decoder.decode_audio(
                    audio_latents=torch.zeros(*receipt["audio_input_shape"]),
                    device=torch.device("mps"),
                )
            phase(
                "audio_decoded",
                shape=list(audio.shape),
                sample_rate=sample_rate,
                finite=bool(audio.isfinite().all()),
            )
            assert list(audio.shape[:2]) == [1, 2]
            assert sample_rate == 32000 and audio.isfinite().all()
            receipt["passed"] = True
        except Exception as exc:
            receipt["passed"] = False
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receipt["wall_seconds"] = time.perf_counter() - start
            receipt["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            print(
                json.dumps({"passed": receipt["passed"], "receipt": str(args.receipt)}), flush=True
            )


if __name__ == "__main__":
    main()
