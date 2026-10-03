# Read-only 1088x1920 memory audit, source 9ec5153

No source edits or GPU execution. Figures below are tensor payload arithmetic,
not measured physical peaks. Use phase-specific current footprint/driver/allocated
samples, not the lifetime maximum, when comparing stages.

## Cross-request DiT/KV

Geometry uses 2040 tokens/video latent, 56 heads x 128, 50 layers.
INT8 K+V plus two F32 scales costs 739200 bytes/token across all layers.
Simulation uses the official direct_5s_plan/canonical_continuation_plan and the
current immutable-segment retention/selector behavior:

| Request index | Largest clean phase | Old physical KV | Old + full staged KV | Single-layer joined BF16 media, excluding text |
|--|--|--|--|--|
| 0 | 3 | 45.186 GiB | 52.248 GiB | 2.023 GiB |
| 1..11 | 2 | ~45.186 GiB | ~59.309 GiB | 2.297 GiB |
| 12 (audio reset) | 2 | 45.187 GiB | 59.309 GiB | 2.297 GiB |

Request 1 introduces a +7.061 GiB transactional payload maximum. Later requests
do not accumulate more video history; audio rounding changes this by tiny amounts.
At the last clean layer, 49 staged layers coexist with old history, a joined BF16
layer and current forward activations. Full 50-layer staging occurs after its
attention returns; the independent maximum columns should not be blindly summed
as an exact peak. Add roughly 27.7 GiB observed active DiT tensor allocation,
BF16 current QKV/MLP/LoRA intermediates, scratch and driver heap retention.

Evidence: streaming/metal_kv.py stage 217–252, commit 254–269, retain 288–296;
attention joined allocation 347–394. streaming/runtime.py 746–756 validates the
append then trims only after a complete clean forward. Audio reset uses selectors,
so old audio backing is not immediately physically freed. The permanent sink's
130 audio rows cost only ~0.0895 GiB; compacting these cannot solve multi-GiB risk.

CPU request outputs are FP32 (runtime/direct.py 293–302). At 13 requests, joined
video latents are only 0.3334 GiB. runner.py 151–157 makes cropped copies and a
joined copy while original segments remain, bringing that part toward ~1 GiB,
not tens of GiB. Prompt features and teacher guidance are CPU and separately
bounded by actual prompt lengths; they do not imply another resident DiT.

## VAE is a separate peak regime

runner.py 339–341 first clears streaming cache/transport/anchor, then releases
DiT with synchronize, gc and empty_cache (runtime/mps.py 183–192). Therefore do
not add 59.3 GiB KV to full decoded video.

CPU assembled FP32 RGB storage (model/output.py 621–636):

| Duration | Native frames | Full CPU RGB |
|--|--|--|
| 5 s | 124 | 2.8949 GiB |
| 10 s | 243 | 5.6731 GiB |
| 60 s | 1433 | 33.4547 GiB |
| 65 s | 1552 | 36.2329 GiB |

Physical pages grow as temporal windows are copied. VAE FP32 weights are about
9.70 GiB (prior verified header inventory), plus audio weights ~0.564 GiB and GPU
spatial/temporal decoder activations. pipeline.py225 enables batched spatial tiles:
fixed official klvae.py278–286 concatenates all local spatial tiles into one
batch; klvae.py399–428 retains decoded tile outputs while building row cats and
final window. This can create a substantial resolution-dependent constant peak.
The current CPU temporal output option removes full-video GPU assembly, not these
per-window activations. Stream publication avoids whole-video normalization/RGB
copies but deliberately retains the one full CPU FP32 decoded storage.

Conditional projection only: if all other VAE allocations are the same, 65s late
VAE adds ~33.338 GiB CPU RGB to the 5s VAE phase footprint. Measure the 5s *VAE
phase* rather than whole-process lifetime peak before using this estimate.

## Minimal semantics-preserving fallback choices

1. Lifetime-only fixes first. output.py676–694 leaves `clip_chunk` as a view of
   the preceding full GPU decoded window even after `del clip_decoded, clip_z`.
   Clear that alias after the final use. Preserve the separate decoded_overlap
   needed for blending. Verify storage weakrefs and byte-exact multi-window output.
2. Avoid redundant VAE offload before immediate disposal. output.py933 moves the
   video VAE to CPU; runtime/mps.py215 encodes before line235 disposes it. Pass
   offload_after_decode=False on this one-shot MPS path, then release model+gc+
   empty_cache immediately after decoded CPU frames are available, before FFmpeg.
   This avoids copying ~9.7 GiB back to CPU only to discard it. Validate frame and
   final MP4 hashes and inject encode failure to check cleanup.
3. output.py631–635 creates host_part before copying into preallocated CPU output.
   Test direct blocking GPU->CPU destination.copy_(part) to remove that temporary;
   preserve dtype and exact bytes, including noncontiguous temporal slices.
4. If DiT stops near a noisy layer, try query64->32 and/or BF16 weight64->32 MiB
   budgets already exposed. They shrink attention/workspace or tile F.linear
   output; they cannot eliminate full MLP output or ~59.3 GiB atomic KV payload.
   Kernel shape changes can change rounding, so compare saved fixed-input outputs.
5. If the late full-video CPU assembly is the limiting factor, move its backing
   to a file-backed FP32 mapping and explicitly reclaim completed pages after
   writes/encoding; or add an emission sink at write_part after overlap blending.
   Both retain official normalization/blend order and final FFmpeg timing, but
   require a separate end-to-end test for tail/pad/overlap and atomic publication.
   This is a larger change than stream publication and should not be slipped in.
6. If atomic old+staged KV is the limiter, disk-backed staged INT8 segments could
   preserve old history until publication, with one-layer reload on later reads.
   Simply freeing old per-layer KV during clean forward would weaken current
   transactional rollback semantics. Do not silently make that change. CPU
   offload alone does not save unified physical memory.

Not recommended as exact fixes: shrinking sink/recent history, changing temporal
window/overlap, FP16 VAE, coarser quantization, or skipping teacher/reset steps.
Changing spatial tile batching keeps mathematical operations but may change
backend accumulation; validate real fixed-latent VAE output before claiming exactness.
