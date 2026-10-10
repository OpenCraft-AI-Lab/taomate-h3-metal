# TaoMate-H3 Metal

### Your Mac. Your prompts. A full minute of sound and motion.

**Create video with speech, ambience and movement locally on Apple Silicon.**
Powered by MiniMax H3 and the official TaoMate-H3 three-step LoRA, brought to
Metal/MPS by [OpenCraft AI Lab](https://github.com/OpenCraft-AI-Lab).

[![CI](https://github.com/OpenCraft-AI-Lab/taomate-h3-metal/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenCraft-AI-Lab/taomate-h3-metal/actions/workflows/ci.yml)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Metal%2FMPS-black)](#requirements)
[![Research preview](https://img.shields.io/badge/status-research_preview-blue)](docs/METAL_PORT.md)

**[Get started](#install)** · **[Reproduce the official examples](#reproduce-the-official-prompt-examples-on-metal)** · **[Compare with h3.c](#taomate-metal-vs-h3c)** · [中文](README.zh-CN.md)

| A minute of video | Less waiting at 768p | Runs on your Apple GPU |
| --- | --- | --- |
| **60-second 480p run completed** with the official Mandarin presenter prompts. | **4.1–4.2× faster completion** than the pinned h3.c workflow in two recorded 20-second scene tests. | **Local audio + video generation** through PyTorch MPS, with an inspectable Python runtime. |

![Sampled frames from the recorded 60-second MPS run](docs/verification/minute-60s/contact-sheet.jpg)

*One recorded minute. The presenter, backpack and studio remain recognizable in
these sampled frames. [Make this example on your Mac](#reproduce-the-official-prompt-examples-on-metal).*

## Built for creators who like to experiment

- **Describe the whole scene.** Prompts can include dialogue, action and soundscape;
  the pipeline produces an MP4 with generated audio and video.
- **Keep the story going.** Chain five-second requests with clean KV history;
  the default path has completed 5-, 10- and 60-second 480p outputs.
- **Push the canvas.** Opt-in quantization has completed native 1088×1920 outputs
  at five and ten seconds. [See the recorded runs](docs/METAL_EXPANDED_WEIGHT_QUANTIZATION.md).
- **Own the workflow.** Run locally, inspect configs and metrics, and experiment
  with W8A16 weights, INT8 KV storage and bounded workspaces.

This is an independent, experimental port of
[TaoMate-H3](https://github.com/TaoLiveAIGC/TaoMate-H3). It retains the trained LoRA,
Base10 audio teacher, Stage3 schedule and CPU noise streams; CUDA execution is
also available. Recorded validation used **M5 Max / 128 GiB unified memory**.
Other Macs and smaller capacities are unvalidated. Generation is not real time;
CUDA numerical parity and independently measured lip sync remain unverified.

## TaoMate Metal vs h3.c

**A 20-second 768p clip in about 51–53 minutes, versus 3½–3¾ hours.**
In two recorded scene tests, TaoMate finished in roughly one quarter of the time
of [antirez/h3.c](https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153).

Both engines ran serially on the same M5 Max / 128 GiB, with identical full prompt
text per scene, seed **8304**, **1344×768** and **24 fps**. Time includes TaoMate's
fresh Base10 teacher as well as generation, decoding and in-process media output.
Memory is the **kernel-reported peak physical process footprint**, including the
larger of TaoMate's serial teacher and generation phases.

| 20-second scene | TaoMate total | h3.c total | h3.c / TaoMate time | TaoMate peak | h3.c peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Automated glass workshop | **52 min 49 s** | 3 h 41 min 42 s | **4.20×** | 64.82 GiB | **43.50 GiB** |
| Orbital cargo docking | **50 min 53 s** | 3 h 28 min 27 s | **4.10×** | 64.87 GiB | **43.52 GiB** |

**TaoMate wins on completion time here; h3.c wins on memory.** TaoMate uses about
49% more peak process memory in these tests. Choose the tradeoff that fits your Mac.

These are historical **workflow comparisons using opt-in W8A16 / INT8 KV settings**,
not the default quickstart or equal-quality/backend benchmarks. TaoMate uses four
five-second requests; h3.c denoises one whole sequence. The pinned h3.c snapshot
has a documented two-line frame-limit patch for 20-second output (481 frames vs
TaoMate's 480). One sample per engine per scene; different samplers, quantization
and LoRA can change quality, and sampled story adherence differed. No universal
4× speedup is claimed. [Exact metrics, settings, prompts and methodology](docs/H3C_COMPARISON.md).

## Requirements

- Apple Silicon Mac with MPS support; recorded validation used macOS 26.5 and
  M5 Max / 128 GiB. Other Macs and smaller memory capacities are unvalidated.
- Python 3.11 for the tested Metal quickstart. The inherited CUDA runtime also
  accepts Python 3.10; the pinned Metal constraints target Python 3.11.
- PyTorch 2.14.1 and torchvision 0.29.1 for the tested Metal setup.
- FFmpeg and FFprobe with `libx264` and AAC encoders.
- Complete official MiniMax H3 FL2VA files, including bundled VAE Python sources,
  and the TaoMate-H3 adapter. Allow about 135 GiB for the base files and 2.4 GiB
  for the adapter, plus outputs and temporary data.

The default path measured approximately 98–100 GiB process peak physical footprint
at 480p. Weight/KV experiments can reduce memory but change numerical results;
see [memory measurements](docs/METAL_MEMORY_EXPERIMENTS.md).

## Install

```sh
git clone https://github.com/OpenCraft-AI-Lab/taomate-h3-metal.git
cd taomate-h3-metal
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[metal]' -c requirements/metal-constraints.txt
brew install ffmpeg
python -m taomate_h3 --help
```

The constraints record the direct dependencies used in validation. The import
remains `taomate_h3`; the distribution and optional console command are named
`taomate-h3-metal`. MPS needs no CUDA, Triton, FlashAttention or vLLM installation.

## Download the model files

Review the [MiniMax H3 Community License Agreement](LICENSE) before using or
distributing this project. It contains territorial and use restrictions, including
excluded territories; this is not an MIT- or Apache-licensed project.

```sh
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include 'model_index.json' 'FL2VA/*' \
  --local-dir models/MiniMax-H3
hf download TaoLiveAIGC/TaoMate-H3 \
  --revision 7d7a51f3e63972138882ec2f0e4d1d47728110cf \
  --include 'config.json' 'adapter_config.json' 'adapter_model.safetensors' \
  --local-dir models/TaoMate-H3
```

These revisions match recorded validation. Existing complete downloads can be
reused through `--model-root` and `--adapter`. Include `FL2VA/video_vae/*.py`
and `FL2VA/audio_vae/*.py`, not just safetensors. For gated files, authenticate
with `hf auth login` using an account with access. Omitting `--adapter` lets the
CLI download the official adapter on first use. No weights, cached guidance or
generated videos are bundled in Git or the wheel.

## Generate a first video

Activate the environment and run from the repository root:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m taomate_h3 \
  --backend mps \
  --model-root models/MiniMax-H3 --adapter models/TaoMate-H3 \
  --prompt 'A calm ocean at sunrise, gentle waves washing onto a sandy beach.' \
  --duration 5 --resolution 864x480 --seed 1 \
  --output outputs/ocean-5s
```

The output directory must be new or empty. Final `video.mp4` is accompanied by
`run_config.json`, `result.json` and `latent_streaming.json`. The command prepares
Base10 audio guidance, runs Stage3 and decodes the joined sequence. First final
latent and first playable video are different events. MPS runs in one process;
`--devices` and `--gpus` apply to CUDA. The inherited backend default is `cuda`,
so pass `--backend mps` on a Mac.

## Reproduce the official prompt examples on Metal

Complete installation and the pinned model downloads above first. The bundled
prompt files are unchanged from upstream revision
`6b2f9984acf681e2f387188f5d1f3b26e504315b` and match the recorded MPS runs.
They describe the Mandarin backpack presenter shown in the contact sheet above.

| Example | Prompt file | Five-second requests | Canvas | Seed per request |
| --- | --- | ---: | --- | ---: |
| Official 10-second example | [`prompts_10s.json`](examples/prompts_10s.json) | 2 | 480×864 portrait | 8301 |
| Official 60-second example | [`prompts_60s.json`](examples/prompts_60s.json) | 12 | 480×864 portrait | 8301 |

Use a fresh terminal without `TAOMATE_MPS_*` options left over from experiments.
These commands select the default BF16/FP32 path and the MPS memory watermarks
used in validation. Activate `.venv`, then run from the repository root.

**10-second example:**

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 \
python -m taomate_h3 \
  --backend mps --model-root models/MiniMax-H3 --adapter models/TaoMate-H3 \
  --prompt-json examples/prompts_10s.json \
  --duration 10 --resolution 480x864 --seed 8301 \
  --output outputs/official-example-10s
```

**60-second example:**

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 \
python -m taomate_h3 \
  --backend mps --model-root models/MiniMax-H3 --adapter models/TaoMate-H3 \
  --prompt-json examples/prompts_60s.json \
  --duration 60 --resolution 480x864 --seed 8301 \
  --output outputs/official-example-60s
```

The 10-second JSON explicitly supplies two seeds; the 60-second JSON inherits
`--seed 8301` for all twelve requests. Keep the complete prompts unchanged,
including dialogue and soundscape, when comparing with the recorded examples.
Output is `outputs/official-example-10s/video.mp4` or
`outputs/official-example-60s/video.mp4`; use a new output path for each rerun.

On the recorded M5 Max / 128 GiB, whole-command time was about 10 min 16 s for
ten seconds and 55 min 26 s for one minute, with approximately 98–100 GiB process
peak footprint. Metal reproduction has been checked through complete media
decoding and sampled visual continuity; CUDA pixel/tensor equality remains
unverified. See [the run records](docs/METAL_PORT.md#completed-full-pipeline-runs).

## Custom prompts and canvas

`--prompt-json` accepts `{"prompts": ["first five seconds", "next five seconds"],
"seeds": [8301, 8301]}`. Each prompt describes one five-second request. Duration
must be a positive multiple of five and agree with the prompt count. `--prompt`
and `--prompt-json` are mutually exclusive.

Canvas geometry is `WIDTHxHEIGHT`; both edges must be divisible by 32 and the
short edge must be 480, 768 or 1088. Common landscapes are `864x480`, `1376x768`
and `1920x1088`; reverse them for portrait. Aligned 1088 is the native 1080p-class
edge; crop afterward for exactly 1080 pixels. Parsing a resolution does not
establish that it fits a particular Mac.

## What has been measured

Historical default-path measurements on M5 Max / 128 GiB:

| Output | Whole-command time, including teacher | Process peak physical footprint |
| --- | ---: | ---: |
| 864×480, 5 s | 320.114 s | 97.76 GiB |
| 480×864, 10 s | 616.452 s | 98.20 GiB |
| 480×864, 60 s | 3,325.502 s | 99.63 GiB |

All three passed full FFmpeg decoding and audio finiteness checks. Sampled minute
frames retained a presenter, backpack and studio. These are execution and sampled
visual checks, not CUDA parity, transcript accuracy or lip-sync scores. Measurements
refer to recorded source revisions, not new benchmarks of every later commit.
See [complete evidence and limitations](docs/METAL_PORT.md).

Later opt-in quantization runs completed native 1088×1920 outputs at five and ten
seconds. A 65-second FC2 run failed on an allocator limit; the combined-policy
65-second attempt was stopped before completion. No 65-second native1080 success
or verified request-13 full-model audio reset is claimed. See
[expanded quantization](docs/METAL_EXPANDED_WEIGHT_QUANTIZATION.md).

## Optional memory experiments

Experiments are explicit environment options. The [memory guide](docs/METAL_MEMORY_EXPERIMENTS.md)
records compatible settings, whole-process measurements and comparisons. W8A16
keeps BF16 activations and is distinct from CUDA W8A8; quantized weights/KV are
lossy and can alter detail, poses and composition. They have not shown a general
speed improvement.

For long experiments, compile `scripts/memory_sample.c` and use
[`scripts/run_guarded.py`](scripts/run_guarded.py), which records physical footprint,
memory reserve, system swap and termination reasons. Commands are in
[contributing and verification](CONTRIBUTING.md).

## Development and documentation

```sh
python -m pip install -e '.[metal,test]' -c requirements/metal-constraints.txt
ruff check src scripts tests
python -m pytest -q
```

Ordinary tests use small tensors and download no weights. MPS cases skip on hosts
without MPS; real macOS watchdog helpers compile into pytest's temporary directory.
Two optional publication tests require bundled upstream processor source selected
by `TAOMATE_TEST_MODEL_ROOT`. CI checks CPU contracts, lint and packaging;
it does not establish real Apple GPU generation support.

- [Metal port and baseline validation](docs/METAL_PORT.md)
- [h3.c speed and memory comparison](docs/H3C_COMPARISON.md)
- [Memory experiments](docs/METAL_MEMORY_EXPERIMENTS.md)
- [Expanded quantization results](docs/METAL_EXPANDED_WEIGHT_QUANTIZATION.md)
- [Documentation index and evidence](docs/README.md)
- [Upstream CUDA setup and benchmarks](docs/UPSTREAM_CUDA.md)
- [Release changes](CHANGELOG.md) and [contributing](CONTRIBUTING.md)

## Attribution and license

TaoMate-H3 was developed by the Alibaba TaoLive AIGC Team on MiniMax H3.
OpenCraft AI Lab adds the Apple Silicon backend, memory work and validation.
Upstream is pinned to `6b2f9984acf681e2f387188f5d1f3b26e504315b`.
This independent port lists changes in [MODIFICATIONS.md](MODIFICATIONS.md).

Distribution and use remain subject to [LICENSE](LICENSE) and [NOTICE](NOTICE).
Model weights have their upstream terms. Retain notices in copies. See the
[upstream guide](docs/UPSTREAM_CUDA.md#citation) for the TaoMate paper citation.
