# Expanded Metal base-weight quantization experiments

The user stopped the allocator-purge experiment and requested additional lower-impact quantization. This is a lossy, explicit MPS-only base-weight experiment; CUDA's original resolver and the default 90 QKV/FC1 selection are unchanged. Original RowParallelLinear/world-one modules, 535 checkpoint keys, BF16 trained LoRA, boundary blocks 0/1/47/48/49, refiners, FP32 heads and FP32 VAE remain protected.

## Real-weight screening

All 90 candidate matrices (blocks 2–46 attention output and FC2) were read one at a time from the verified official weights. Two fixed 257-row BF16 synthetic activation distributions were evaluated on actual MPS, with the real BF16 rank128/alpha128 LoRA. This is candidate screening, not real generation hidden states or perceptual equivalence. Full per-layer metrics and input/output hashes are archived in verification/memory-experiments/expanded-weight-screen.

| Additional coverage | Matrices | Net persistent savings | Mean single-layer output relative L2 with LoRA | Range |
|---|---:|---:|---:|---:|
| Attention output | 45 | 1.614089 GiB | 1.1729% | 1.0436–1.4119% |
| MLP FC2 | 45 | 3.229079 GiB | 1.3637% | 1.1997–1.5380% |
| Both | 90 | 4.843168 GiB | Not additive / not measured jointly | Requires generation |

These percentages are **single-layer output errors, not percentages of image-quality loss**. Storage savings subtract F32 per-output-row scale buffers; they do not predict process peak or eliminate full-width BF16 MLP/LoRA outputs. Screening completed in 28.81s at 1.597 GiB physical peak under a 10 GiB guard.

## Explicit policies and tests

TAOMATE_MPS_WEIGHT_COVERAGE defaults to interior_fc1_qkv_v1. Additional policies are metal_interior_fc1_qkv_out_v1 (135 matrices), metal_interior_fc1_qkv_fc2_v1 (135), and metal_interior_all_linear_v1 (180). None is enabled by default. Expanded policies require original 50-block inventory, single-device original row modules, and an enabled W8A16 backend; no silent fallback occurs.

All 258 CPU/real-MPS tests pass (30.55s), including production indexed safetensors loading, stable parameter identity and 535-key inventory, full 208-target trained-LoRA installation, row-module dispatch against an independent FP64 dot-product oracle, protected dtype/selection, multi-device rejection, default-policy preservation and policy propagation.

## Next complete-generation control

FC2-only expansion is tested first for its 3.23 GiB storage reduction. It is not yet described as low-impact visually. The 480×864/10s candidate reuses the exact prompts, request seeds and original teacher artifacts of int8-packed8-10s. Existing reference weight tile256 MiB/query256/LOW0.3 settings are retained. Boundary-purge and diagnostic trace are explicitly disabled, and row-view mode is disabled for this quality control. Safety remains HIGH0.9, physical105 GiB, reserve6 GiB and swap-growth2 GiB. Video/audio latent errors, full AV decoding and visual contacts will be reported before the native1080 memory test.
