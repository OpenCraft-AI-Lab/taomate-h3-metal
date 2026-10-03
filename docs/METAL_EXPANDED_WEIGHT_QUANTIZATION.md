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

## Native 1088×1920 ten-second result

Expanded FC2 completed two requests and all 32 Stage3 forwards (24 noisy, 8 clean), then one FP32 video/audio VAE publication. Guarded wall time was 4075.675s (67m55.7s), including 3130.655s Stage3 and 877.544s video decoding, excluding original cached teacher creation. Kernel lifetime physical peak was 97.399248 GiB. System swap decreased by 24 MiB; no guard or OOM occurred. Allocator active peak was 95,731,546,624 bytes and reserved peak 103,414,759,424 bytes, measured separately from physical footprint and not necessarily simultaneously. The reserved peak leaves little room under unchanged HIGH0.9; longer execution remains unverified.

Strict complete decoding passed 240 H264/yuv420p frames at 1088×1920 and 24 fps, exactly 10.000s video and 32 kHz stereo AAC presentation. Audio and video publication latents are finite. Compared with the original-90 native5 run's first-request publication latents, the first 207 audio latents are identical; first 37 video latents have relative L2 0.279090 and cosine 0.961060. These are lossy tensor differences, not perceptual scores. All candidate source hashes match pinned ed1b1d0; source differences versus the older native5 reference are explicitly listed. Whole-sequence VAE context and global retiming differ, so MP4 first-five-second equality is not claimed.

Sampled frames 0/60/118/121/123/125/128/239 show a stable presenter, backpack and background through the request boundary, without black frames or a gross scene jump. A small bag label remains despite the no-logo prompt; neither exact prompt adherence nor quality equivalence is claimed. Full metrics, contacts and compressed logs are archived under verification/memory-experiments/native1080-expanded-fc2-10s.

The next authorized run is 65 seconds / 13 requests with identical successful flags, original cached Base10 guidance and the official request-13 audio reset, under unchanged 105/6/2 GiB guards and HIGH0.9/LOW0.1. This tests longer-lived allocator state, audio-reset behavior and the large one-shot FP32 VAE output; ten-second success does not establish those outcomes.

## FC2-only 65-second attempt: allocator refusal

The 13-request run failed after 6123.475s (102m03.5s), in request index3 / phase2, while allocating the joined attention V buffer. Three requests had completed; request13 audio reset and the final VAE were not reached, and no video was published. PyTorch reported reserved 95.85 GiB plus a requested 1.16 GiB above the unchanged 96.77 GiB HIGH limit. Failure-point active bytes were not captured; reserved bytes are not equivalent to active tensors.

Physical peak was 97.788759 GiB; swap decreased by 8 MiB. The process exited1 without a guard event, and all 53 source hashes match preflight/receipt. Ten-second success therefore does not establish longer-run safety under the same policy. Logs and failure validation are archived in verification/memory-experiments/native1080-expanded-fc2-65s.

The next quantization candidate adds the already-screened 45 middle-block attention output matrices to FC2 (180 selected base matrices total). Net base-weight saving relative to the original90 policy is 4.843168 GiB, 1.614089 GiB beyond FC2-only; this is not a process-peak guarantee. The combined policy will undergo a separate 480×864/10s real-generation comparison before another native memory test. BF16 LoRA, protected boundaries/heads, original teacher, KV retention and FP32 VAE remain unchanged. No allocator-purge route or increased safety limit is resumed.

## Combined FC2 + attention-output 480×864 quality control

The 180-matrix policy completed the separate 10-second control in 608.646s, at 54.214329 GiB kernel physical peak and zero swap growth. Strict full decoding verified 240 frames / 10.000s video and 32 kHz stereo audio. Original teacher audio latents and decoded PCM are exactly equal to the original-90 reference.

Video latent relative L2 is 0.299279, cosine 0.955519 and RMS error 0.296871. Decoded RGB MAE is 6.5824/255 and PSNR 25.3661 dB. This particular trajectory is numerically closer to the reference than FC2-only; that does not show that adding quantization generally improves quality. The same six paired-frame inspection finds a stable presenter, backpack and scene with altered poses/background details and no obvious collapse. Perceptual or CUDA equivalence remains unverified.

All 53 implementation hashes remain ed1b1d0. Weight256 MiB/query256/LOW0.3 and reference prompts/seeds/teacher were preserved; row views, trace and purge were disabled. Evidence is archived under verification/memory-experiments/expanded-all-480-10s. The next run tests this combined policy at native 1088×1920 for 10 seconds before any repeated 65-second attempt, with the existing unchanged guards.

## Combined policy native ten-second result

The 180-matrix native 1088×1920 run completed in 3915.212s (65m15.2s), with all 32 Stage3 forwards and original FP32 publication. Kernel lifetime physical peak was 97.538057 GiB, swap decreased by 8 MiB, and no guard or OOM occurred. This physical peak is slightly higher than FC2-only; weight payload savings do not guarantee a proportional process-peak reduction. DiT allocator peaks were 93,986,242,048 active bytes and 102,892,568,576 reserved bytes; reserved peak decreased by about 0.486 GiB versus FC2-only, leaving limited HIGH headroom.

Strict complete decoding passed 240 frames / 24 fps / 10.000s at 1088×1920 and 32 kHz stereo AAC. All publication latents are finite. Versus the original-90 native5 first-request video latents, relative L2 is 0.283388 and cosine 0.960005; the audio prefix is identical. A separate full-ten-second same-source comparison against FC2-only finds video relative L2 0.386081 / cosine 0.925747, RGB MAE 10.9142/255 and PSNR 20.1242 dB; all audio latents and decoded PCM remain identical. These are numeric trajectory differences, not percentages of visual-quality loss.

Inspection of frames 0/60/118/121/123/125/128/239 finds recognizable, stable presenter/backpack/background and no sampled black frames or gross scene break. Poses and bag details differ, and the unwanted small bag label persists. This is a bounded visual inspection, not perfect continuity, exact prompt compliance or perceptual/CUDA equivalence. Generation success, strict media validation and this qualitative judgment are recorded separately in verification/memory-experiments/native1080-expanded-all-10s.

The combined policy is now eligible for the separately guarded 65-second / 13-request attempt under the same successful flags. No long-run result is inferred from this ten-second run.
