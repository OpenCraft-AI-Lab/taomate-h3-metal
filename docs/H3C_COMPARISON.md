# TaoMate Metal vs h3.c: speed and memory

Two historical 20-second, 1344×768 scene pairs on the same Apple M5 Max / 128 GiB
host. The **h3.c / TaoMate total-time ratio was 4.10–4.20×**; TaoMate used
**about 49% more peak physical process memory** than the tested h3.c workflow. This compares complete
generation workflows with different algorithms and outputs; it does not establish
equal quality, a universal speedup, or a faster PyTorch backend.

## Exact results

Times include model loading, generation, decoding and in-process media output.
TaoMate's total includes a fresh, separately timed Base10 teacher for each scene.
Download, build, preflight and later technical/visual review are excluded.

| Scene | Engine | Fresh teacher, s | Generation + decode, s | Total, s | Teacher peak, GiB | Generation peak, GiB | Workflow peak, GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Glass workshop | TaoMate | 76.003 | 3,092.808 | **3,168.812** | 64.816 | 62.332 | 64.816 |
| Glass workshop | h3.c | — | 13,301.581 | 13,301.581 | — | **43.504** | **43.504** |
| Orbital docking | TaoMate | 76.804 | 2,976.504 | **3,053.308** | 64.867 | 62.511 | 64.867 |
| Orbital docking | h3.c | — | 12,507.288 | 12,507.288 | — | **43.519** | **43.519** |

| Scene | h3.c / TaoMate total time | TaoMate time reduction | TaoMate peak-memory increase |
| --- | ---: | ---: | ---: |
| Glass workshop | 4.197656× | 76.18% | 48.99% |
| Orbital docking | 4.096307× | 75.59% | 49.05% |

Calculations use unrounded seconds and bytes in the
[portable benchmark receipt](verification/h3c-comparison-20261005/benchmark.json).
Physical footprint is `kernel_peak_footprint_bytes / 2**30`, measured with the
same macOS helper. Workflow peak is the **maximum**, not the sum, of the serial
teacher and generation phases. It is neither current tensor allocation nor total
system memory usage. A measured 65 GiB process peak is not a validated 64 GiB Mac
configuration. The default unquantized 480p path still measured 98–100 GiB.

## What was held constant

- Same machine, serial jobs, one sample per engine per scene, fixed order, no rerolls.
- Identical complete UTF-8 prompt text within each pair; seed 8304, 1344×768, 24 fps,
  requested duration 20 seconds. Equal seeds do not imply equal random tensors.
- Official MiniMax H3 files pinned to `42ed227ee7df40d41602854ae760620d6eb651fe`;
  TaoMate adapter pinned to `7d7a51f3e63972138882ec2f0e4d1d47728110cf`.
- One shared GPU lock; physical-footprint guard 105 GiB, minimum available reserve
  6 GiB, maximum system swap growth 2 GiB and wall limit four hours per phase.
  No guard fired. Final swap did not increase; this does not mean the system had
  no existing swap or prove absence of transient paging.

The host's baseline inventory identifies M5 Max and 137,438,953,472 memory bytes
(128 GiB); hardware inventory was not recollected for each phase. The receipt
retains that inventory's digest and the original experiment manifest digest.

## Different workflows, different tradeoffs

| Setting | TaoMate Metal | h3.c |
| --- | --- | --- |
| Recorded source | [`ed1b1d0`](https://github.com/OpenCraft-AI-Lab/taomate-h3-metal/tree/ed1b1d0ee87733f90596bf5248280527dae3b890), 53 frozen Python source files | [`8974cc0`](https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153), isolated source snapshot |
| Conditioning | Same full story repeated over four five-second requests | One full story, whole-clip denoising |
| Sampling | Trained TaoMate LoRA; Stage3, 3 noisy + 1 clean forward per chunk, four chunks/request | 20 steps, 50 layers, reuse 2, core-reuse 1; 11 model evaluations |
| Audio teacher | Original BF16 Base10, 9 forwards/request; fresh for each scene | None |
| Memory policy | Opt-in `metal_interior_all_linear_v1`: 180 W8A16 matrices, dequantized BF16 operator tiles; INT8 packed8 clean KV | Native default quantization; no SSD/BF16 or row-FC2 override |
| Workspaces | Weight tile 256 MiB; KV tile 1,024 tokens; query chunk 256; row views off | Native defaults |
| Video decode | Original FP32 video VAE | Native C/Metal video VAE |
| Presented output | 480 frames, 20.000 s video and audio | 481 frames, 20.041667 s video, 20.050 s audio |

The h3.c snapshot's released frame ceiling was 362. The isolated benchmark changed
**only the ceiling to 481 and its diagnostic message**. The original h3.c checkout
was unchanged; the exact [two-line patch](verification/h3c-comparison-20261005/experimental-481.patch.txt)
and its hash are included. The native frame/audio grids were preserved without
trimming or retiming. The README ratio uses total job wall time, not a ratio
normalized to the slightly different presented durations.

The h3.c patch excerpt retains Salvatore Sanfilippo's attribution and
[MIT license](verification/h3c-comparison-20261005/h3c-LICENSE.txt). TaoMate's own
license remains the MiniMax H3 Community License Agreement.

Most of the timing gap occurred in denoising: 2,211/2,114 s for TaoMate's Stage3
versus 12,674/11,878 s for h3.c. Video decoding favored h3.c: 605/605 s versus
812/792 s for TaoMate. This evidence does not isolate the effect of any one
quantization option. In particular, it does not show that quantization alone
makes TaoMate faster than its BF16 default.

## Quality observations

All four outputs passed strict full audio/video decoding and decoded-audio finite
checks. Sampled image reviews found **different story adherence**:

- Glass workshop: TaoMate transferred the sphere early and showed possible repeated
  grasp/release; h3.c samples more closely followed the single transfer/release order.
- Orbital docking: TaoMate appeared inserted early and moved to a closer framing;
  h3.c samples showed a more gradual approach, although contact was still early.

These samples do not certify continuous motion, physical contact, sound alignment
or equal audiovisual quality. Actual audio listening was pending in the recorded
comparison. The speed headline is a measured completion-time result for these
configurations, not a quality ranking. Full sampled observations and media hashes
are retained in the receipt; generated MP4s and weights are excluded from Git.

## Inputs and replay settings

The portable evidence directory contains:

- [Glass workshop prompt](verification/h3c-comparison-20261005/b-automated-glass-workshop-20s.txt)
  and [TaoMate config](verification/h3c-comparison-20261005/taomate-b-automated-glass-workshop-20s.json).
- [Orbital docking prompt](verification/h3c-comparison-20261005/e-orbital-cargo-docking-20s.txt)
  and [TaoMate config](verification/h3c-comparison-20261005/taomate-e-orbital-cargo-docking-20s.json).
- [Benchmark receipt](verification/h3c-comparison-20261005/benchmark.json): unrounded
  metrics, phase wall times, peak bytes, exit/guard status, source hashes, effective
  environment, model revisions, portable h3.c argv and original receipt hashes.

Configs replace original machine-specific model/output paths with repository-local
paths. Prompt bytes, four request seeds, geometry and parallelism are preserved.
These are extracts of recorded evidence, not newly measured results for the latest
commit. Use the recorded source and pinned model revisions when comparing results.

For TaoMate, use the public [installation instructions](../README.md#install),
the pinned models and a fresh terminal. Choose one config, then export the options
below for **both** teacher and generation. Config paths may be edited to point to
existing model downloads; the output directory must be new.

```sh
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3
export TAOMATE_MPS_RELEASE_ADALN=1
export TAOMATE_MPS_CPU_ADALN=1
export TAOMATE_MPS_WEIGHT_QUANTIZATION=dequantize
export TAOMATE_MPS_WEIGHT_COVERAGE=metal_interior_all_linear_v1
export TAOMATE_MPS_WEIGHT_TILE_MIB=256
export TAOMATE_MPS_KV_STORAGE=int8
export TAOMATE_MPS_KV_TILE_TOKENS=1024
export TAOMATE_MPS_KV_ALLOCATION=packed8
export TAOMATE_MPS_ATTENTION_QUERY_CHUNK=256
export TAOMATE_MPS_EMPTY_CACHE_PHASE=1
export TAOMATE_MPS_SAVE_LATENTS=1
export TAOMATE_MPS_STREAM_PUBLICATION=1
export TAOMATE_MPS_INPLACE_ACTIVATIONS=1
export TAOMATE_MPS_ATTENTION_ROW_VIEWS=0
export TAOMATE_MPS_MEMORY_TRACE=0
export TAOMATE_MPS_EMPTY_CACHE_BOUNDARIES=0

config=docs/verification/h3c-comparison-20261005/taomate-b-automated-glass-workshop-20s.json
comparison_dir=outputs/comparison/b-automated-glass-workshop-20s
mkdir -p .local
cc -O2 -Wall -Wextra scripts/memory_sample.c -o .local/memory-sample

run_phase() {
  phase="$1"
  shift
  mkdir -p "$comparison_dir/$phase"
  python scripts/run_guarded.py \
    --watcher .local/memory-sample --lock .local/heavy-gpu.lock \
    --log "$comparison_dir/$phase/run.log" \
    --memory "$comparison_dir/$phase/memory.jsonl" \
    --receipt "$comparison_dir/$phase/process-receipt.json" \
    --footprint-gib 105 --reserve-gib 6 --swap-growth-gib 2 \
    --max-wall-seconds 14400 --mps-low-watermark 0.3 -- "$@"
}

run_phase teacher python -B -u scripts/prepare_experiment_guidance.py \
  --config "$config" --guidance "$comparison_dir/guidance" && \
run_phase generation python -B -u scripts/run_cached_guidance.py \
  --config "$config" --guidance "$comparison_dir/guidance"
```

Run both phases once and add their receipt `wall_seconds`; do not compare a cached
generation-only time against h3.c's total. For docking, change both `config` and
`comparison_dir` to the corresponding `e-orbital-cargo-docking-20s` paths.

For h3.c, obtain the pinned commit in a separate checkout, apply the included patch
with `git apply`, then build with `make`. Its recorded CLI shape is:

```sh
./h3 -d /absolute/path/to/MiniMax-H3 -p "$(cat /absolute/path/to/scenario-prompt.txt)" \
  -o /absolute/path/to/new-output/video.mp4 \
  --width 1344 --height 768 --seconds 20 --seed 8304 \
  --steps 20 --layers 50 --reuse 2 --core-reuse 1 --profile
```

Use a fresh terminal without `H3_*`/`TAOMATE_MPS_*` overrides, and run the h3.c job
under the same guard and shared lock, with its own output/log/receipt paths.
Do not overlap it with either TaoMate phase. The portable receipt records the
exact argv for each prompt and the effective environment. Separately timed
encoding/muxing and validation durations were not recorded, so no zero-duration
values are invented.
