# Documentation

Start with the [English quickstart](../README.md) or [中文说明](../README.zh-CN.md).

| Guide | Scope |
| --- | --- |
| [Metal port](METAL_PORT.md) | Official-source baseline, setup, 5/10/60-second runs and limitations |
| [Memory experiments](METAL_MEMORY_EXPERIMENTS.md) | Options, guarded measurements, W8A16 / KV tradeoffs |
| [Expanded quantization](METAL_EXPANDED_WEIGHT_QUANTIZATION.md) | Native1080 short runs and incomplete long runs |
| [Weight tiles](METAL_WEIGHT_TILES.md) | Bounded BF16 dequantization workspace |
| [Packed KV](METAL_PACKED_KV.md) | Layer-group allocation experiments |
| [KV tiles](METAL_KV_TILES.md) | Bounded history conversion |
| [In-place activations](METAL_INPLACE_ACTIVATIONS.md) | BF16 rounding and storage ownership |
| [Native lifetimes](METAL_NATIVE_LIFETIME_EXPERIMENTS.md) | Projection, clean-cache and VAE lifetimes |
| [Stream publication](METAL_STREAM_PUBLICATION.md) | Bounded CPU frame conversion after the original VAE |
| [CUDA reference](UPSTREAM_CUDA.md) | Pinned TaoLiveAIGC setup, NVIDIA measurements and citation |

## Evidence archive

[`verification/`](verification/) contains historical source hashes, environment
receipts, timings, memory samples, media checks and contact sheets. Successful,
failed and interrupted experiments are preserved. Original machine paths and
hashes are provenance, not portable defaults. Archived QA scripts may require
path adaptation; public `scripts/` and quickstarts use configurable paths.

Weights, generated MP4s and guidance tensors are excluded. Old hashes describe
the recorded run and differ after notice edits. Whole-command time, reused-teacher
Stage3 time, current tensor storage and kernel physical peaks are separate metrics.
See the [release checklist](RELEASE_CHECKLIST.md) for publication verification.
