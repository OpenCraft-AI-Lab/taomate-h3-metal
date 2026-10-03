# Native 1080 continuation memory experiments

## LOW 0.1, before projection lifetime changes

The unchanged 53-file production source (ffa5c0b) completed request 0 and request 1 phases 0–1, then failed allocating the joined BF16 V tensor during phase 2. MPS reserved 95.64 GiB and rejected a 1.16 GiB allocation under the unchanged 96.77 GiB high watermark. This is allocator enforcement, not a successful 10-second output. No video was delivered.

Measured wall time 2750.902 seconds; kernel lifetime physical peak 98.5833 GiB; postflight swap delta -25165824 bytes. The 105 GiB / 6 GiB available / 2 GiB swap guard and HIGH 0.9 were retained. LOW was 0.1. Teacher artifacts were reused.

Evidence: `verification/memory-experiments/native1080-packed8-low01-10s/`. Full raw memory samples remain under the corresponding outputs directory. Next experiment applies only the reviewed projection and clean-hook temporary lifetime releases; no precision or KV-window changes.

## Reviewed lifetime changes and validation

Production now includes commits 62d4626 and 356340a: release original QKV after the unchanged stack, full FC1 projection after independent contiguous SwiGLU output, and clean-hook temporaries before KV stage. All 217 CPU/MPS tests passed (29.98 seconds). The full native continuation-size synthetic single-layer diagnostic passed under the 10 GiB guard (8.29 seconds, 9.6783 GiB kernel physical peak). Full attention/MLP BF16 output hashes and full INT8 K/V/F32 scales were identical. At clean-stage entry A fell from 4,844,628,480 to 2,444,483,072 bytes; out-projection entry 6,210,968,064 to 5,321,775,616; FC2 entry 3,065,240,064 to 1,857,280,512. R/D at those fixed-order checkpoints did not fall; do not sum these differences or claim measured end-to-end peak reduction. First attempt rejected a shortened source hash before GPU initialization; preserved alongside successful full-hash retry.
