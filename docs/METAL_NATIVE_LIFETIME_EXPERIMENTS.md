# Native 1080 continuation memory experiments

## LOW 0.1, before projection lifetime changes

The unchanged 53-file production source (ffa5c0b) completed request 0 and request 1 phases 0–1, then failed allocating the joined BF16 V tensor during phase 2. MPS reserved 95.64 GiB and rejected a 1.16 GiB allocation under the unchanged 96.77 GiB high watermark. This is allocator enforcement, not a successful 10-second output. No video was delivered.

Measured wall time 2750.902 seconds; kernel lifetime physical peak 98.5833 GiB; postflight swap delta -25165824 bytes. The 105 GiB / 6 GiB available / 2 GiB swap guard and HIGH 0.9 were retained. LOW was 0.1. Teacher artifacts were reused.

Evidence: `verification/memory-experiments/native1080-packed8-low01-10s/`. Full raw memory samples remain under the corresponding outputs directory. Next experiment applies only the reviewed projection and clean-hook temporary lifetime releases; no precision or KV-window changes.
