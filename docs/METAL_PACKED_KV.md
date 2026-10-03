# Packed clean KV allocation experiment

`TAOMATE_MPS_KV_STORAGE=int8 TAOMATE_MPS_KV_ALLOCATION=packed8` opts into
allocation grouping. The default remains `per_layer`; BF16 storage and CUDA
are unchanged. This changes allocation ownership, not the existing INT8
quantization formula, BF16 restoration, row selection or full-key softmax.

Each clean segment lazily allocates one INT8 K owner and one V owner per eight
layers, with shape `[min(8, remaining_layers), tokens, heads, 128]`. The final
group contains two layers. Its F32 scales use one shared allocation of shape
`[2, 50, tokens, heads, 1]`, created on first staging. Each layer writes only its
slice. All 50 layers and a successful synchronization are still required before
publication; old history remains live throughout. Rollback discards staging.
Published views retain their owners until segment eviction or clear, including
rows hidden by sink/audio-reset selectors.

Receipts distinguish `staged_payload_bytes` (successfully staged logical views)
from `staged_backing_bytes` (all allocated staging backing, including unfilled
slices). `physical_history_payload_bytes` counts shared storage once. These are
tensor storage bytes, not allocator reserved memory, driver memory or process
footprint. Grouping aims to reduce allocator fragmentation; it does not reduce
logical KV bytes or prove a lower real peak without a measured run.

CPU tests compare both modes bitwise, including scale tensors, full attention,
52 commits against upstream retention/reset metadata, out-of-order staging,
final short group, failures at layers 0/7/49, synchronization failure before
publication and owner lifetime. Matching small MPS tests subsequently passed in the exclusive parent GPU
window, as part of all 198 CPU/real-MPS tests on `8e6b1dd` (29.37 s). These
tests establish arithmetic/ownership behavior at small shapes; native-shape
allocator and whole-model measurements are still required.


A subsequent same-source native-shape allocator control on `ffa5c0b` completed
two phases with 28 GiB resident tensors and real FC1 after every clean layer:
65.716 GiB process peak. Per-layer allocation stopped at the same 75 GiB guard
partway through phase2. At matched phase1/layer44, reserved memory falls from
73.121 to 62.594 GiB; active memory increases from 60.417 to 61.285 GiB due to
unfilled packed slices. This establishes a measured benefit for this synthetic
native geometry, not a guaranteed benefit at other geometries or a completed
full-model video. See the experiment report and diagnostic evidence.
