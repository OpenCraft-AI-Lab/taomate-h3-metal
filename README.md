# TaoMate-H3 Metal

[中文](README.zh-CN.md) · [Validation](docs/METAL_PORT.md) · [CUDA reference](docs/UPSTREAM_CUDA.md)

[![CI](https://github.com/OpenCraft-AI-Lab/taomate-h3-metal/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenCraft-AI-Lab/taomate-h3-metal/actions/workflows/ci.yml)

An experimental Apple Silicon port of the official
[TaoMate-H3](https://github.com/TaoLiveAIGC/TaoMate-H3) Python runtime, maintained by
[OpenCraft AI Lab](https://github.com/OpenCraft-AI-Lab). Powered by MiniMax H3.

Generate synchronized audio and video on one Apple GPU through PyTorch Metal/MPS.
The port retains the trained three-step LoRA, Base10 audio teacher, Stage3 schedule,
CPU noise streams and clean KV history. Optional memory experiments add W8A16
base weights, INT8 KV storage and bounded workspaces. CUDA execution remains available.

**Research preview:** validated on M5 Max / 128 GiB unified memory. The default
BF16/FP32 path has completed 5-, 10- and 60-second 480p runs. It is memory intensive
and does not generate in real time on the measured Mac. CUDA numerical parity
and independently measured lip sync remain unverified.

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

For two prompts over ten seconds:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m taomate_h3 \
  --backend mps --model-root models/MiniMax-H3 --adapter models/TaoMate-H3 \
  --prompt-json examples/prompts_10s.json \
  --duration 10 --resolution 480x864 --seed 8301 \
  --output outputs/two-prompts-10s
```

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

![Sampled frames from the recorded 60-second MPS run](docs/verification/minute-60s/contact-sheet.jpg)

Sampled frames from the recorded one-minute MPS run, using the official prompt
example. The contact sheet is historical validation evidence.

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
