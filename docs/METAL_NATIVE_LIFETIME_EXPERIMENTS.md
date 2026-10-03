# Native 1080 continuation memory experiments

## LOW 0.1, before projection lifetime changes

The unchanged 53-file production source (ffa5c0b) completed request 0 and request 1 phases 0–1, then failed allocating the joined BF16 V tensor during phase 2. MPS reserved 95.64 GiB and rejected a 1.16 GiB allocation under the unchanged 96.77 GiB high watermark. This is allocator enforcement, not a successful 10-second output. No video was delivered.

Measured wall time 2750.902 seconds; kernel lifetime physical peak 98.5833 GiB; postflight swap delta -25165824 bytes. The 105 GiB / 6 GiB available / 2 GiB swap guard and HIGH 0.9 were retained. LOW was 0.1. Teacher artifacts were reused.

Evidence: `verification/memory-experiments/native1080-packed8-low01-10s/`. Full raw memory samples remain under the corresponding outputs directory. Next experiment applies only the reviewed projection and clean-hook temporary lifetime releases; no precision or KV-window changes.

## Reviewed lifetime changes and validation

Production now includes commits 62d4626 and 356340a: release original QKV after the unchanged stack, full FC1 projection after independent contiguous SwiGLU output, and clean-hook temporaries before KV stage. All 217 CPU/MPS tests passed (29.98 seconds). The full native continuation-size synthetic single-layer diagnostic passed under the 10 GiB guard (8.29 seconds, 9.6783 GiB kernel physical peak). Full attention/MLP BF16 output hashes and full INT8 K/V/F32 scales were identical. At clean-stage entry A fell from 4,844,628,480 to 2,444,483,072 bytes; out-projection entry 6,210,968,064 to 5,321,775,616; FC2 entry 3,065,240,064 to 1,857,280,512. R/D at those fixed-order checkpoints did not fall; do not sum these differences or claim measured end-to-end peak reduction. First attempt rejected a shortened source hash before GPU initialization; preserved alongside successful full-hash retry.

## Native 10 seconds with lifetime releases

The exact lifetime changes alone were insufficient. The run failed again in request 1 phase 2 when allocating joined BF16 V: MPS reserved 95.68 GiB, requested 1.16 GiB, unchanged cap 96.77 GiB. Wall 2805.978 seconds; physical peak 98.7244 GiB; swap delta -117440512 bytes. No video delivered and no 65-second run started. All 53 source hashes matched preflight and process receipt. Next isolated candidate removes copies only when selected condition/media rows form a contiguous interval; arbitrary masks must retain the original gather.

## Contiguous-row view candidate

A separate opt-in replaces gathers with verified contiguous slices only for MPS/no-grad contiguous tensors. Noncontiguous masks retain gathers; full-key attention and all numeric operators remain unchanged. All 231 CPU/real-MPS tests passed (28.73s). Actual native continuation-size isolated attention/MLP and full staged INT8 K/V/scales SHA256 are exact between gather/view modes. A synchronized sample before joined-history construction fell from 3,049,323,008 to 2,142,779,904 allocated bytes; strides remained [7168,128,1], while storage offset changed as expected. R/D remained unchanged in this fixed-order isolated control. These are not end-to-end peak claims. Native10 retest uses the unchanged hard limits and teacher hashes.

## Native 10 seconds with contiguous-row views

The run failed in request 1 phase 2 at the FC1 LoRA second linear projection, after progressing beyond the prior joined-V failure site. MPS reserved 95.80 GiB and requested another 1.12 GiB under the unchanged 96.77 GiB cap. Wall 2881.994 seconds, physical peak 98.5405 GiB, swap delta -83886080 bytes. No video was delivered; 65 seconds has not started. All 53 source hashes match preflight and process receipt. Reserved bytes at failure do not establish active-tensor bytes; the next controlled diagnostic adds optional synchronized A/R/D and cache-backing samples at joined-KV, clean-stage, and LoRA boundaries before deciding on further allocation or quantization changes.

## Allocation-boundary diagnostic

Optional scalar-only telemetry now records synchronized A/R/D, history/staged physical backing and requested buffer bytes before/after joined K/V, clean stage, and LoRA updates. It neither collects frames nor retains tensors, invokes GC, or empties the cache. All 237 CPU/real-MPS tests pass (29.59s) under a 10 GiB guard, including exact traced/untraced outputs and cache ownership. Diagnostic synchronization is explicitly disclosed and timing must not be treated as the unchanged uninstrumented workload.
