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

## FC2 complete-generation control

The 480×864/10s expanded-FC2 run completed in 606.652s with 55.8707 GiB kernel lifetime physical peak and zero swap growth. All 53 source hashes match the pinned implementation ed1b1d0. Existing reference prompts, seeds, original teacher artifacts and weight256 MiB/query256/LOW0.3 settings were retained; purge, tracing and row views were disabled. Complete strict decoding verified 240 frames, 24 fps, 10 seconds and 32 kHz stereo audio. Teacher creation is excluded from this reused-guidance run time.

Video latent relative L2 is 0.338986, cosine 0.942624 and RMS error 0.336258. Decoded RGB MAE is 8.4474/255 and PSNR 22.4931 dB. These are tensor/pixel differences, **not percentages of perceptual quality loss**. Audio latents and decoded audio samples are exactly equal. The archived comparison lists source/environment differences; the reference predates separately tested exact lifetime changes, so this is not falsely described as identical-source A/B.

Visual inspection of paired frames 0/60/118/123/180/239 finds the same recognizable presenter, backpack and stable scene. Mouth/hand poses and bag details differ. Neither perceptual equivalence nor CUDA parity is established, but no obvious scene collapse appeared in those sampled frames. Contacts and complete metrics are under verification/memory-experiments/expanded-fc2-480-10s.

Next is the native 1088×1920/10s memory validation with this explicit FC2 policy, weight64 MiB/query512/LOW0.1, unchanged HIGH0.9 and 105 GiB physical / 6 GiB reserve / 2 GiB swap-growth limits. Quantization does not remove all full-width BF16 activation allocations; successful native memory behavior remains to be measured.
