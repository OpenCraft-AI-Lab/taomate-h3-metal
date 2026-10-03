# Metal memory experiments

These are opt-in experiments on the official TaoMate-H3 Python port. The
successful BF16/FP32 baseline is commit `0053a8a`. The experiment branch is
`codex/metal-memory-experiments`; `h3.c` is an engineering reference only.
The original trained adapter, teacher, Stage3 schedules, CPU noise, retention,
attention visibility and FP32 VAE remain in use.

## Safety and measurement

`scripts/run_guarded.py` holds the shared exclusive GPU lock and samples the
child's macOS physical footprint with `proc_pid_rusage` every 250 ms. It
terminates only the child's process group if the lifetime footprint exceeds
105 GiB, free + inactive memory falls below 6 GiB, system swap grows by more
than 2 GiB, telemetry fails, or the experiment exceeds its wall-time budget.
MPS high/low watermarks are 0.9/0.8; implicit CPU fallback is disabled.
Startup failures and interruptions also clean up descendants. The launch
receipt records source hashes, command, environment, limits and measurements.
Free + inactive is a conservative heuristic, and swap is system-wide.
Current tensor payload, driver allocations and process peak are distinct.
The macOS sampler covers the main Python worker PID; it does not aggregate
FFmpeg child-process memory. The reserve and swap checks are system-wide.
Some early receipts have an unavailable postflight sample (`TimeoutExpired`);
reported swap changes refer to the last valid in-run sample, not a verified
post-exit reading. The guard remained active throughout those runs.

## Experiments completed before quantization

| Experiment | Full-run physical peak | Wall time | Outcome |
| --- | --- | --- | --- |
| Original 480×864, ten seconds | 98.20 GiB | 616.452 s including teacher | Successful baseline |
| Release exact fixed-schedule AdaLN, ten seconds | 97.07 GiB | 329.393 s including teacher, incomplete | Guard stopped after 2.16 GiB system swap growth; no completed video |
| AdaLN + query chunks of 256 + phase cache cleanup | 98.04 GiB | 534.422 s excluding reused teacher | Successful; delivered MP4, all 240 decoded frames and audio exactly equal to baseline |

The second successful experiment releases 24.2877 GiB of AdaLN parameters
and retains 55.5 MiB of exact projected values. Every query chunk attends to
all visible keys through one full softmax; keys are never independently
normalized. Its measured current DiT allocation falls from 63.713 GiB to
39.093 GiB, but driver/temporary allocations still dominate the process peak.
Deleting tensors after first allocating a large GPU model is insufficient.

The exact MP4 SHA-256 for the original and optimized ten-second outputs is
`dba747b56066230fde93cc526c197e4cdd654b826fcd8b44ad72d3a71439d613`.
The A/B harness hashes and reuses the unchanged official teacher artifacts;
those wall times must not be presented as cold end-to-end generation times.

## Final storage allocation and selective W8A16

The original model is constructed on the meta device and allocated once in
its final storage dtype. All 535 checkpoint keys and original module classes
are preserved. AdaLN weights are initially CPU BF16, transferred one original
projection at a time to MPS for exact schedule precomputation, then released.
The original FP32 input/time/final parameters and RoPE buffer remain FP32.

The official interior policy selects exactly 90 matrices: QKV and FC1 in
blocks 2 through 46. Each BF16 CPU checkpoint matrix is reordered/sharded
through the original loader before bounded row-wise INT8 conversion. There
is no full BF16 MPS staging copy of a selected weight. Scales are F32, and
native MPS matmul additionally requires BF16 scales. The original 208 LoRA
targets and 416 trained tensors are installed after checkpoint loading and
retain BF16 compute buffers.

This is W8A16, not CUDA's W8A8: activations remain BF16. `dequantize` uses a
bounded BF16 weight tile and ordinary matmul. `native_int8pack` uses the MPS
`aten::_weight_int8pack_mm` operator, whose arithmetic includes BF16 scale
rounding and is not identical to BF16-rounded dequantized weights. F32 scales
with BF16 native inputs were found to produce invalid numerical results and
are explicitly rejected. Neither path is enabled by default.

A guarded real-shape isolated QKV benchmark (256×5376 input, 21504 output)
measured 2.24 ms BF16 and 7.84 ms native W8A16, with relative L2 distance
0.00924. This is an isolated numerical/performance probe; it is not whole-model
quality or speed validation.

## INT8 clean KV

The optional cache stores independent INT8 K/V and F32 scales per token/head.
Clean chunks are immutable segments, published only after all 50 layers are
staged and synchronized. Retention and audio reset use the original ordered
rows. Failed stage/commit leaves the published history unchanged; the runtime
still invalidates a failed request. A retained diagnostic exception does not
keep joined GPU buffers alive.

Attention constructs at most one layer's joined BF16 K/V, directly filling
condition, dequantized retained history and current rows. It uses the same
full-key softmax and bounded query chunks. Receipts distinguish physically
retained backing storage from logically visible history; selectors can retain
hidden audio backing until the segment is evicted. No whole-cache BF16 copy
is materialized.

The unit suite exercises 52 clean commits, all 50 layer names, both K/V,
retention and the audio reset on CPU and real MPS. Actual full-model tests and
quality measurements are recorded below when completed.

## Reproduction options

All options are disabled by default, except that INT8 KV chooses query chunks
of 128 if no explicit query chunk was supplied.

| Environment variable | Meaning |
| --- | --- |
| `TAOMATE_MPS_RELEASE_ADALN=1` | Exact fixed Stage3 schedule precompute and fail-closed weight release |
| `TAOMATE_MPS_CPU_ADALN=1` | Allocate AdaLN on CPU; requires release option |
| `TAOMATE_MPS_WEIGHT_QUANTIZATION=native_int8pack` | Stream selected weights into INT8 final storage; BF16 activations |
| `TAOMATE_MPS_WEIGHT_QUANTIZATION=dequantize` | Alternative bounded dequantized weight matmul |
| `TAOMATE_MPS_KV_STORAGE=int8` | Lossy clean KV storage with full-softmax BF16 attention |
| `TAOMATE_MPS_KV_ALLOCATION=packed8` | Group eight clean layers into shared INT8 allocations; default per-layer allocation |
| `TAOMATE_MPS_INPLACE_ACTIVATIONS=1` | Reuse eligible detached MPS LoRA/SwiGLU outputs with exact BF16 rounding |
| `TAOMATE_MPS_ATTENTION_QUERY_CHUNK=128` | Bound queries while each query sees all visible keys |
| `TAOMATE_MPS_EMPTY_CACHE_PHASE=1` | Synchronize and trim unused MPS allocations after each phase |
| `TAOMATE_MPS_WEIGHT_TILE_MIB=64` | BF16 weight tile budget (default 256 MiB; validated 1–256) |
| `TAOMATE_MPS_KV_TILE_TOKENS=1024` | Bounded history conversion tile (default 128; scratch budget still enforced) |
| `TAOMATE_MPS_STREAM_PUBLICATION=1` | Convert and encode four CPU frames at a time, preserving the original FP32 VAE |
| `TAOMATE_MPS_SAVE_LATENTS=1` | Save exact FP32 video/audio inputs used for publication, for A/B comparison |

A native 1080p-class canvas is 1088×1920 (or landscape 1920×1088), following
the original alignment. It has 2040 video tokens per latent, compared with 405
at 480×864. At 32 retained video latents plus 228 audio tokens, 50 layers of
BF16 KV contain 87.46 GiB; INT8 plus F32 scales contains 45.10 GiB. These are
retained payload estimates, not measured peaks, and exclude staging, current
activations, attention workspace, driver allocations and decoding.

`scripts/prepare_experiment_guidance.py` invokes the unchanged original
Base10 teacher for a specified geometry. `scripts/run_cached_guidance.py`
then invokes the original runner on that artifact. A 480p artifact cannot be
reused as a 1080p artifact: the official RNG draws and discards full video
noise before audio noise, so geometry affects the audio RNG stream.

`scripts/compare_experiments.py` compares full decoded frames/audio and saved
publication latents without GPU work. Pixel error or latent L2 distance is
not a perceptual, semantic, transcription or lip-sync quality score. Official
CUDA tensor fixtures remain unavailable on this Mac.

## Measured selective-weight result

The guarded native W8A16 ten-second official example completed on source
`677a877`, with all 50 source hashes verified unchanged at completion. The
process peak was **62.82 GiB**, wall time **722.598 s** excluding the reused
teacher, Stage3 **424.285 s**, and FP32 video decoding **234.645 s**. No guard
triggered; the final valid in-run system-wide swap sample was 8 MiB lower. The post-AdaLN current MPS
allocation was **27.670 GiB** and driver allocation **29.331 GiB**. These
model-only readings must not be called the full process peak.

All 240 frames decoded successfully. Compared with the exact BF16 baseline,
mean absolute RGB pixel error was **6.86/255**, PSNR **24.61 dB**, and final
video latent relative L2 distance **0.3151**. Published audio latents and
decoded audio were exact because the same original teacher supplied the
audio. Ten paired visual samples retain the presenter, backpack and studio,
with changes in hand/mouth position and detail. This does not establish
perceptual equivalence, transcription accuracy or lip-sync parity.

Native W8A16 lowered the whole-run peak by about **35.4 GiB**, but Stage3 was
about **1.68× slower** than the bounded BF16 reference (424.285 vs 252.933 s).
This is a memory/quality tradeoff, not an observed speed optimization. See
`verification/memory-experiments/int8-weights-10s/` for receipts, full-frame
comparison and the visual contact sheet.


## Measured weight and KV configurations

The following ten-second tests use the same 480×864 example, unchanged
original audio teacher and trained LoRA. All 240 frames decode successfully.
The launcher wall times exclude teacher generation and include process startup
and cleanup. Source hashes were checked at completion and match the recorded
commits.

| Configuration | Source | Process peak | Wall time | Stage3 | FP32 video VAE |
| --- | --- | --- | --- | --- | --- |
| Native W8A16, BF16 KV | `677a877` | 62.82 GiB | 722.598 s | 424.285 s | 234.645 s |
| Native W8A16, INT8 KV, 128-token tile, bounded publication | `4350b29` | 59.08 GiB | 828.400 s | 548.858 s | 213.954 s |
| Dequantized W8A16, INT8 KV, 1024-token tile, bounded publication | `ca6702f` | 61.08 GiB | 581.132 s | 303.952 s | 212.532 s |

No guard triggered in these three completed runs. The largest in-run swap
increase was zero; their final valid swap samples were 8, 40 and 8 MiB below
preflight respectively. The last configuration changes both the matmul backend
and KV conversion tile, so its speed difference is a combined configuration
result rather than an isolated attribution.

Against the exact BF16 baseline, native W8A16 plus INT8 KV has RGB MAE
6.002/255, PSNR 25.780 dB and video-latent relative L2 0.2704. Dequantized
W8A16 plus INT8 KV has MAE 6.758/255, PSNR 24.781 dB and relative L2 0.3093.
A higher PSNR in one diffusion run does not establish better perceptual
quality. Ten paired samples in each run retain the presenter, backpack and
studio, with different poses and small details. Audio remains exactly equal
because the same unquantized official teacher artifacts are reused; this is
not a parity test of a quantized teacher or audio prediction branch.

The process peak has fallen by about 37 GiB against the original 98.20 GiB
ten-second run. The original trained adapter remains installed. Quantization
is lossy, while exact AdaLN caching, full-key query chunking and bounded
publication preserve their original computation/encoding semantics.

## Isolated large-matrix probes

`scripts/benchmark_metal_linear.py` measures real H3 shapes under the same
watchdog. A combined backend probe with an intentionally small 8 GiB process
cap completed five cases, then was safely terminated at 8.06 GiB while
preparing the largest FC1 case. Its partial log is retained and is not a
completed benchmark.

For QKV with 24,590 input rows, that partial probe measured median BF16
92.31 ms, native W8A16 364.88 ms and tiled BF16 dequantization 101.70 ms.
At 4,990 rows the corresponding times were 17.91, 74.34 and 25.05 ms.
The smaller numerical probes have relative L2 errors around 0.00923 for both
quantized paths. These are isolated linear results, not video quality scores
or full-model throughput.

A separate completed FC1 probe (`9ec5153`, 28,672×5,376 weights) uses a
64 MiB dequantized weight tile and has a measured process peak of 5.113 GiB.
At 4,990 and 24,590 rows its median times are 35.16 and 140.36 ms. It keeps
no resident BF16 baseline GPU matrix and therefore reports timing, not a new
numerical parity comparison. The 64 MiB setting bounds the BF16 weight tile,
not all activations, joined KV, driver memory or the process peak.

## Native 1080p-class experiments

The unchanged original Base10 BF16 audio teacher has prepared 13 requests
at 1088×1920 on source `9ec5153`, including the original request-13 audio
rollover. This teacher-only command completed in 148.124 s with a process
peak of 65.53 GiB and no guard trigger. Its artifacts are hashed and may be
reused by matching 5-, 10-, 60- or 65-second prefix configurations. Reused
Stage3 timings must be reported separately from this 13-request preparation.

Native video experiments proceed from five seconds to ten-second continuation
before a 65-second run. The 13th request additionally exercises the official
12-request audio reset, which a 60-second run does not reach. Initial settings
are dequantized W8A16 with 64 MiB weight tiles, INT8 KV with 1024-token tiles,
64-query full-key attention, exact CPU-staged AdaLN and bounded publication.
The same physical-memory, reserve and swap limits remain enforced.

The first request does not cover the largest continuation cache. Simulating
original retention at this geometry yields approximately 52.25 GiB of old
plus staged INT8 KV in the first request, then 59.31 GiB from the second
request. These are payload estimates, not measured process peaks. The audio
reset changes visibility selectors and does not immediately release all
hidden backing storage. Likewise, the decoder retains a complete CPU FP32
video (about 36.23 GiB for 65 seconds), although the DiT and KV are released
before VAE loading. Published native outcomes will be recorded only after
completion and validation.

Receipts, sampled telemetry, benchmark logs, comparisons and contact sheets
are in `verification/memory-experiments/`. Full raw telemetry and generated
media remain under ignored `outputs/memory-experiments/`.


### First native video attempt

On frozen source `9ec5153`, the 1088×1920 five-second attempt completed two
of four Stage3 phases, then failed during a noisy MLP LoRA merge in phase 2.
MPS refused an additional 1.12 GiB shared-pool allocation: its reported
allocated memory was 96.15 GiB against the enforced 96.77 GiB watermark.
The main Python process lifetime physical peak was 99.42 GiB, below the
105 GiB process watchdog limit. The command exited with code 1 after
1151.987 s, without a guard trigger or observed system swap growth. No
completed video was produced. Neither limit was increased to force progress.

After phase 0, current MPS allocation/driver readings were 45.05/64.58 GiB;
after phase 1 they were 59.60/90.81 GiB. The gap requires separate allocator
and driver diagnostics; it cannot be assigned to attention, graph storage or
fragmentation solely from these two readings. The failure also identifies
large LoRA merge temporaries as a concrete optimization target.

Two exact lifecycle fixes have since been committed: bounded, reliable
postflight watchdog sampling (`a9c569f`) and MPS VAE window-view release plus
model disposal before encoding (`b6179b7`). Their 147 CPU/real-MPS tests pass.
A fixed-latent full decoder comparison and opt-in activation-lifetime work
follow; completed native video outcomes will be documented separately.


### Exact decoder lifecycle validation

The complete saved ten-second BF16 decoder inputs were re-published on
`b6179b7` without teacher or DiT execution. This decoder-only verification
completed in 193.124 s with a main-worker physical peak of 23.46 GiB, valid
postflight telemetry and no guard trigger. Its entire delivered MP4 SHA,
all 240 RGB frames, decoded audio samples and saved latents exactly match the
original baseline. SHA remains
`dba747b56066230fde93cc526c197e4cdd654b826fcd8b44ad72d3a71439d613`.
This validates the lifecycle changes at 480×864; it is not a new generation
or a measured native-1080 decoder peak.

`3b7fc88` adds `TAOMATE_MPS_INPLACE_ACTIVATIONS=1`. On detached MPS BF16
inference with the original single-device linears, a scale-exactly-one LoRA
update reuses its fresh base output, eliminating the separate unit-multiply
and sum outputs. SwiGLU preserves separate BF16 SiLU and multiplication
rounding and contiguous FC2 input. Inputs, adapter GEMMs and nonunit scales
retain their original behavior; grad-enabled, aliased, hooked/custom and
default CUDA paths do not use this optimization. Six new real-MPS cases
exercise bitwise outputs and storage safety; the full CPU/MPS suite passes
161 tests. Whole-model memory savings remain to be measured.


### Allocator diagnostics

Eight independent guarded synthetic diagnostics ran on byte-matched source
`3b7fc88`, Torch 2.14.1 (`5c488690`), with HIGH=0.9 and a separate 75 GiB
process cap. All recorded system swap deltas were zero. These are allocation
and arithmetic controls, not generated videos or whole-model measurements.

Three real FC1 layout variants (28,672×5,376 weight, 20,928 activation rows,
64 MiB weight tiles) match the production path across all 600,047,616 BF16
output elements. Original strided column writes have allocator-reserved peak
3.13 GiB; concatenate and transposed alternatives have 4.26 and 3.26 GiB.
There is no measured memory reason to change this layout.

A control using production INT8 cache, all 50 layers, native geometry, 28 GiB
of uniform touched resident tensors and interleaved real FC1 shows a growing
reserved-minus-active gap. One phase completes at 58.69 GiB process peak
with LOW=0.8 and 57.60 GiB with LOW=0.3. Two phases stop safely at the
fixed 75 GiB cap, after different amounts of completed work. At the matched
second-phase layer-38 FC1 point, both have active 58.677 GiB, while reserved
memory is 73.477 versus 71.012 GiB. LOW=0.3 improves this point by 2.465 GiB;
it is insufficient to complete the controlled interleaving within the cap.

Removing the FC1 workload and its roughly 0.4 GiB of associated weight/input
storage lets the same two-phase cache control finish at 62.63 GiB process
peak, allocator active peak 60.47 GiB and reserved peak 61.01 GiB. In these
controls driver-minus-reserved stays below 0.704 MiB. This supports further
investigation of persistent KV and transient linear allocations sharing heap
space. It does not prove the entire full-model 31 GiB gap has this sole cause.
The native MPS OOM text's “MPS allocated” value is allocator-reserved heaps,
not the active tensor counter.

`scripts/run_guarded.py --mps-low-watermark 0.3` can lower the adaptive
allocation threshold. Accepted finite values are strictly `(0, 0.8]`, default
0.8; the HIGH=0.9 hard cap and the existing process/reserve/swap limits remain.
`dbb6992` records active/reserved current and allocator peak counters through
Torch's generic MPS API, with explicit nulls on older/unsupported APIs.
Allocator peaks remain distinct from the independently sampled physical peak.
The full CPU/real-MPS suite passes 185 tests.

See `verification/memory-experiments/allocator-diagnostics/` for all commands,
source/script hashes, complete and stopped receipts, numerical checks and
per-step active/reserved/driver measurements.


### Complete exact activation/lifecycle regression (E6)

A fresh 480×864 ten-second run on frozen `dbb6992` combines the exact
activation reuse, VAE lifetime fixes and LOW=0.3 with E5's unchanged
dequantized weight path (256 MiB tiles), INT8 KV (1024-token tiles),
256-query full-key attention and the same reused official teacher.
It completes in 567.918 s, with a physical peak of 59.111 GiB, valid
postflight telemetry, no guard trigger and no observed swap growth.
Stage3 takes 299.418 s and FP32 video decoding 208.306 s. Teacher preparation
is excluded from these timings.

The delivered MP4, all 240 RGB frames, decoded audio samples and both
publication latents are byte-exact with E5. MP4 SHA remains
`c89c124f9530bc3649497e82b62d161944162939d55ea3b11d9301b430b97a8b`.
This validates the exact changes against that already quantized result;
E5's quantified difference from the BF16 baseline still applies. E5 and E6
change several lifecycle/allocator settings together, so their 61.083 versus
59.111 GiB process peaks cannot isolate one setting's contribution.

The new counters measure DiT allocator peaks of 41.054 GiB active and
58.625 GiB reserved; final DiT active/reserved are 35.552/56.055 GiB.
A substantial reserve-minus-active gap remains. These allocator readings
are not the independently measured process physical peak.

`8e6b1dd` adds optional packed8 clean-cache ownership. All 198 CPU/real-MPS
tests pass in 29.37 s, including both allocation modes, attention equality,
52-commit retention/reset and transaction failure ownership. `ffa5c0b` fixes
only import ordering, and lint/diff checks pass. Real-shape allocator and
whole-model packed8 measurements remain to be established separately.

See `verification/memory-experiments/int8-inplace-low03-10s/` for the source
hashes, full-output comparison, runtime receipt and sampled physical memory.


### Matched packed8 allocation and full-output SDPA diagnostics

Four more guarded serial controls use byte-identical `ffa5c0b` source (all
53 tracked Python files audited again at completion), LOW=0.3 and HIGH=0.9.
The native-shape two-phase cache/resident28/FC1 experiment completes with
packed8 at 65.716 GiB process peak, versus a same-source per-layer guard stop
at 75.118 GiB after second-phase layer 44. At that matched live-FC1 point,
packed8 active/reserved are 61.285/62.594 GiB, versus 60.417/73.121 GiB:
reserved falls by 10.527 GiB, while active rises by 0.868 GiB because a packed
owner includes not-yet-filled layer slices. The completed first-phase cache
and FC1 samples match, and packed8 passes its independent CPU quantization
samples in both phases. This is an allocator control, not a whole DiT/video.

Two production SDPA controls compare chunks 64/128/256/512, three trials
each, at BF16 NHD/56 heads/D128. Every byte of all 24 outputs matches its
geometry's first chunk 64 result; all are finite, with relative L2 and maximum
absolute distance zero. The production full-key softmax and synchronization
per query chunk remain unchanged.

| SDPA Q/K tokens | Warm median64 | 128 | 256 | 512 | Whole-PID peak | Allocator R peak |
| --- | --- | --- | --- | --- | --- | --- |
| 24610 / 24610 | 1.0363 s | 0.6036 s | 0.4790 s | 0.4000 s | 3.653 GiB | 2.008 GiB |
| 20514 / 86392 | 2.8024 s | 1.6774 s | 1.3440 s | 1.2431 s | 8.204 GiB | 3.320 GiB |

Chunk 512 is 2.59×/2.25× faster than 64 and 1.20×/1.08× faster than 256 in
these operator controls. Reserved peaks are unchanged across chunks in each
process. Fixed chunk order and only three trials limit performance inference;
K=86392 is an approximate later-history size, not an exact count for every
prompt. No full-model speedup or CUDA equality is claimed. All four jobs have
no observed swap growth; the per-layer process guard slightly overshoots its
75 GiB diagnostic budget before termination. Production limits stay unchanged.

Full receipts, per-step counters, script snapshots, artifact hashes and the
source audit are under `verification/memory-experiments/packed-kv-sdpa-diagnostics/`.
Whole-model packed8 and native video outcomes are measured separately.


### Complete packed8 regression (E7)

At 480×864, ten seconds, `ffa5c0b` packed8 with otherwise E6-identical options
completes in 572.580 s. All 240 RGB frames, decoded audio, both publication
latents and the delivered MP4 are byte-exact with E6/E5 (SHA `c89c124f...`).
Stage3 takes 300.431 s and video decode 211.218 s, excluding the same reused
unmodified official teacher. Postflight is valid, no guard triggers and
system swap does not grow.

DiT allocator reserved peak falls from E6's 58.625 to 47.453 GiB, while
active peaks stay near 41.05 GiB. Independently, the main-process physical
peak rises from 59.111 to 64.585 GiB this run. Its high-water mark first
appears at 93.494 s, with sampled current footprint 63.085 GiB and RSS
25.784 GiB; RSS later falls to roughly 2.47 GiB by 125 s. At similar 94 s E6
has footprint 50.514/RSS 9.733 GiB, and by 110 s RSS 0.527 GiB. The identical
AdaLN receipts report release of 24.288 GiB CPU parameters. A read-only
review finds no obvious large packed host copy or retained AdaLN parameter
in the normal returned receipts, but does not distinguish live CPU tensors
from deferred allocator/VM page release. Sparse end-of-phase D/R samples
cannot rule out transient allocations. No universal process-peak reduction
is claimed from the allocator improvement alone; guards remain authoritative.

This complete regression establishes exact delivered values at 480p. Native
1088×1920 tests now use packed8, 64 MiB weight tiles and query 512 based on the
separate controlled measurements; their results must be measured separately.
See `verification/memory-experiments/int8-packed8-10s/` for complete evidence.
