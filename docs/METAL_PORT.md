# TaoMate-H3 on Metal

This checkout ports the official [TaoMate-H3](https://github.com/TaoLiveAIGC/TaoMate-H3)
Python project to Apple Silicon. The source baseline is
`6b2f9984acf681e2f387188f5d1f3b26e504315b` on branch `codex/metal-backend`.
The official project defines model behavior. `h3.c` is a reference for Apple GPU
engineering, not the application runtime or the algorithm specification.

## Scope

Retain the official model modules, checkpoint loader, trained LoRA, Base10 audio
teacher, Stage3 schedule, CPU noise generation, prompt handling, clean KV cache,
continuation geometry, normalization, and final audio/video publication.
Adapt device execution and memory lifecycle for one Apple GPU. Keep the CUDA
entrypoint available. Do not call the previous `tm` or `h3` executables.

The initial Metal backend uses PyTorch MPS for GPU execution, PyTorch SDPA for
attention, and the upstream unfused tensor operations instead of CUDA-only
Triton kernels. The FP32 input/output projections remain FP32. CUDA's selective
CUTLASS W8A8 is not silently emulated: the initial Metal policy uses the original
BF16 weights and must identify that precision difference in run receipts.
CUDA-vs-MPS equality is not implied by CPU/MPS operator tests.

## Implementation and validation order

1. Add an explicit backend, single-process topology, dependency checks and
   device-aware synchronization, timing and memory reporting.
2. Route the official teacher and Stage3 through MPS, keeping their shared
   model semantics. Preserve condition/media attention visibility and all
   persistent-cache boundaries.
3. Bound unified-memory use: encode text before loading the DiT; finish and
   release teacher resources before Stage3; release DiT/cache before VAE decode.
4. Test CPU and Metal operators, FP32 heads, BF16 operation boundaries,
   transactional cache behavior, CLI selection and absence of NCCL on MPS.
5. Run the official weights through a five-second 480p smoke, then continuation,
   changed prompts and the audio reset boundary. Save real receipts and videos.
6. Compare fixed-input intermediate tensors against an official CUDA run when
   CUDA fixtures are available. Run one minute only after short-run validation.

The earlier h3.c-based minute video belongs to a separate experiment; it is not
validation of this port. Long-video output quality and CUDA tensor parity remain
unverified until measured in this checkout.

## Local setup and provenance

- Mac: M5 Max, 128 GiB unified memory.
- Isolated environment: `.venv`, Python 3.11.15, PyTorch 2.14.1;
  upstream pins `diffusers==0.35.2`, `transformers==4.57.6`, `tokenizers==0.22.1`.
- Existing base: `/Users/supreme-paca/Projects/h3-metal/MiniMax-H3`, revision
  `42ed227ee7df40d41602854ae760620d6eb651fe`.
- Existing adapter: `/Users/supreme-paca/Models/TaoMate-H3`, revision
  `7d7a51f3e63972138882ec2f0e4d1d47728110cf`.
- The 24 bundled VAE Python source files missing from the earlier C-only download
  were retrieved at the same base revision. Total 172,981 bytes; Git blob hashes
  and sizes verified. See `verification/component-source-receipt.json`.
- No model weights were downloaded again. Model files remain outside Git.

FP32, BF16 and FP16 SDPA executed successfully on MPS. The full official
pipeline has also completed five-second, ten-second and one-minute 480p runs; see the
measured results below. These results do not establish CUDA numerical parity
or support for higher resolutions.

## Running on this Mac

Create an isolated Python 3.11 environment and install the project with its test
extra. The verified environment versions are in `verification/environment.json`.
For a new checkout on an Apple Silicon Mac, the environment used here can be
created with these commands (FFmpeg must also be on `PATH`):

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  'torch==2.14.1' 'torchvision==0.29.1' -e '.[test]'
```

Use the complete official FL2VA download, including its bundled VAE Python
source files, and the trained TaoMate-H3 adapter. The base directory happens
to be under the earlier C experiment on this machine; this port reads its
official model files and never calls the C executable.

The existing local environment is ready:

```sh
.venv/bin/python -m taomate_h3 \
  --backend mps \
  --model-root /Users/supreme-paca/Projects/h3-metal/MiniMax-H3 \
  --adapter /Users/supreme-paca/Models/TaoMate-H3 \
  --prompt 'A calm ocean at sunrise, gentle waves washing onto a sandy beach.' \
  --duration 5 --resolution 864x480 --seed 1 \
  --output outputs/new-mps-run
```

The output directory must be new or empty. `--devices` is CUDA-only; MPS always
uses one local process and one Apple GPU. The CUDA backend remains the default.
MPS uses the official CPU random-number streams, not a device RNG substitute.

```sh
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/verify_mps_components.py \
  --model-root /Users/supreme-paca/Projects/h3-metal/MiniMax-H3 \
  --receipt outputs/components.json \
  --gpu-lock /Users/supreme-paca/Projects/h3-metal/outputs/original-baseline-20261002/heavy-gpu.lock
```

The component smoke explicitly loads the real VAE weights. Ordinary pytest
cases use small tensors and do not download or load model weights. Tests that
need Metal skip when the device is unavailable; a CPU-only pass is insufficient
for Metal validation. The macOS execution sandbox can hide MPS device access.

The MPS cache replaces retained rows one layer at a time to avoid holding two
complete copies of the 50-layer KV cache. If that operation fails, the cache
and stream are invalidated and require a new session. This is an explicit
failure policy; a partially updated stream cannot be resumed.

## Validation evidence

- `verification/component-source-receipt.json`: fixed-revision VAE source hashes.
- `verification/component-smoke.json`: both official VAEs executed on MPS with
  real weights, no implicit CPU fallback. Video input `[1,24,7,4,4]` produced
  `[1,3,22,64,64]`; audio input `[2,32,10]` produced `[1,2,8000]` at 32 kHz.
- `verification/component-smoke-invalid-fixture.json`: retained first diagnostic
  attempt with only two temporal latents, which is below the decoder's minimum
  valid window. The corrected seven-latent fixture passed.
- `verification/source-audit/`: independent upstream compatibility and memory
  audit. Its header-based memory estimates are not measured process peaks.

### Completed full-pipeline runs

| Run | Output | Whole command | Process peak physical footprint | Evidence |
| --- | --- | --- | --- | --- |
| Beach smoke | 864×480, 5.000 s, 120 frames | 320.114 s | 97.76 GiB | `verification/smoke-5s/` |
| Official two-prompt example | 480×864, 10.000 s, 240 frames | 616.452 s | 98.20 GiB | `verification/continuation-10s/` |
| Official twelve-prompt example | 480×864, 60.000 s, 1,440 frames | 3,325.502 s | 99.63 GiB | `verification/minute-60s/` |

All three outputs contain 24 fps H.264 video and 32 kHz stereo AAC. All passed
strict FFmpeg decoding and finite decoded-audio checks. The second request
of the ten-second run reused 11,105 history tokens, then updated its prompt.
It performed 24 noisy forwards and eight clean-cache forwards in total.
Its 45 Python source hashes were unchanged from launch to inspection.
The five-second launch has a documented provenance caveat: two backend
files received formatting, annotations and a timing precondition while the
first launch was underway; see its `inspection.json`.

The whole-command durations above include the Base10 teacher. The worker
receipt's `worker_cold_e2e_seconds` starts after teacher preparation and must
not be reported as the full command duration. Process peak physical footprint
comes from macOS `proc_pid_rusage`, including GPU/unified-memory allocations;
RSS or current MPS allocations are different measurements. System-wide swap
increased by about 2.82 GiB during the five-second run and 0.87 GiB during the
ten-second run. The minute run ended with 88 MiB less system-wide swap used
than at launch. These deltas are not attributed exclusively to the model.

Ten one-second samples of the official example show a consistent presenter,
backpack and studio, with hand and mouth movement. This is sampled visual
inspection, not a speech-transcription, lip-sync or CUDA parity evaluation.
The beach sample also introduces two people absent from its simple prompt.

### Regression evidence

`verification/regression-tests.json` records **30 passing tests, no skips**,
on CPU and real MPS with `PYTORCH_ENABLE_MPS_FALLBACK=0`. Coverage includes
FP32 output heads, BF16 operation boundaries, packed-document isolation,
rectangular attention and visibility, official tiny DiT AV/audio-only
forwards, text release before DiT loading, failed-session invalidation, and
50-layer retention with injected failures.

`tests/fixtures/upstream_contracts.json` was exported from an unmodified
upstream checkout after verifying all 43 source files against the pinned
Git commit. It independently covers CPU noise hashes for requests 0, 1 and 12,
both timestep schedules, 13 continuation geometries, and 52 cache commits
including the audio reset before request 13. CPU and MPS cache results match
that fixture. The exact CPU noise hashes are scoped to the recorded Torch/CPU
environment; other environments skip those three hash cases until fixtures
are regenerated from the pinned upstream. This is a state-contract test, not
a 65-second model run.

To regenerate the fixture, extract `src/` from the pinned upstream commit to a
separate directory, then run:

```sh
.venv/bin/python scripts/build_upstream_fixtures.py \
  --source-root /path/to/unmodified-upstream/src \
  --output tests/fixtures/upstream_contracts.json
```

A read-only final review found no blocking changes to the CUDA 4/8-GPU
branches, attention visibility, or FP32 projection boundaries. CUDA hardware
regression remains unperformed on this Mac.

### Minute validation

The official `examples/prompts_60s.json` completed successfully with the same
frozen source, 480×864 resolution and seed 8301. Full wall time was **55 min
25.502 s**, including the teacher and model loading. Stage3 took 1,773.197 s
(29 min 33 s); video decoding took 1,357.303 s (22 min 37 s); audio VAE took
4.053 s. The process peak physical footprint was **106,974,874,048 bytes
(99.63 GiB)**. This validates the existing BF16/FP32 path on this 128 GiB Mac;
it is not real-time playback or a CUDA numerical parity result.

The run used all twelve prompts, 144 noisy forwards and 48 clean-cache
forwards. Requests 2–12 reused prior KV state. The full video latent shape was
`[1,24,422,54,30]`, and the audio shape was `[2,32,2388]`. One video VAE call
processed 84 temporal windows. Following the unchanged upstream publication
policy, 1,433 native frames became exactly 1,440 frames at 24 fps; audio was
exactly 1,920,000 stereo samples at 32 kHz. Final duration is 60.000 seconds.

Strict full-file FFmpeg decoding passed. All 3,840,000 decoded interleaved
audio values were finite. Twenty-four evenly spaced video frames and pairs
around all eleven request boundaries were inspected: the same presenter,
backpack and studio remain visible, with no obvious identity or scene break
in those samples. Speech transcription and lip synchronization were not
independently evaluated. All 45 source hashes matched the launch snapshot.
The final regression suite passed 30 tests with no skips in 2.55 seconds.

Evidence is in `verification/minute-60s/`, including the command, source
hashes, complete log, memory samples, model receipts, media checks and contact
sheets. The generated media remains in the ignored directory
`outputs/metal-official-60s-20261002/video.mp4`. This command ran across
October 2–3 local time; the output directory retains the October 2 task label.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 \
.venv/bin/python -m taomate_h3 \
  --backend mps \
  --model-root /Users/supreme-paca/Projects/h3-metal/MiniMax-H3 \
  --adapter /Users/supreme-paca/Models/TaoMate-H3 \
  --prompt-json examples/prompts_60s.json \
  --duration 60 --resolution 480x864 --seed 8301 \
  --output outputs/new-mps-minute
```

### Precision and validation limits

The current MPS DiT uses the official BF16 reference operations and trained
LoRA. It does not introduce INT4/INT8 model quantization. CUDA's selective
W8A8 acceleration and FlashAttention-3 are replaced by the BF16 path and
PyTorch SDPA. Final projections remain FP32.

Video VAE decoding uses an explicit full-FP32 reference policy because the
bundled model has CUDA-specific autocast protection regions. CUDA normally
uses selective FP16 autocast; claiming identical precision or bitwise output
would be incorrect. The MPS backend preserves the original VAE tiling,
blending and publication logic, with one process and no CUDA fusion.

Remaining validation: fixed-input official CUDA intermediate tensors and
quality comparisons; higher-resolution memory limits; longer runs beyond the
validated minute and real model execution across the 13th-request audio reset.
