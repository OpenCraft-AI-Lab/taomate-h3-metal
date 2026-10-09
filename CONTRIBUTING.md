# Contributing

Use Python 3.11 and an isolated environment. On Apple Silicon:

```sh
python -m pip install -e '.[metal,test]' -c requirements/metal-constraints.txt
ruff check src scripts tests
python -m pytest -q -ra
```

CPU-only CI uses the same PyTorch version. GPU tests skip without MPS; this cannot
establish Apple GPU support. On macOS, pytest builds the memory sampler in its own
temporary directory using clang. Optional upstream processor checks use
`TAOMATE_TEST_MODEL_ROOT` (default `models/MiniMax-H3`), reading Python source only.

Keep original schedules, condition/media visibility, FP32 boundaries, BF16 rounding
and cache transactions covered by meaningful tests. Add memory/precision experiments
as explicit options and record their tradeoffs. CUDA changes need NVIDIA hardware
validation separately.

## Guarded Metal runs

Build the native observer:

```sh
mkdir -p .local/bin
clang -std=c11 -D_DARWIN_C_SOURCE -O2 -Wall -Wextra \
  scripts/memory_sample.c -o .local/bin/memory-sample
```

Use new log/receipt/output paths and one shared lock across all local GPU tasks.
These budgets reproduce the recorded 128 GiB experiment policy; adapt them to the
host. Reserve is a free-plus-inactive heuristic; swap is system-wide. The observer
samples the main worker, not FFmpeg children.

```sh
mkdir -p outputs/guarded-ocean
python scripts/run_guarded.py \
  --watcher .local/bin/memory-sample \
  --lock /tmp/taomate-h3-metal-gpu.lock \
  --log outputs/guarded-ocean/run.log \
  --memory outputs/guarded-ocean/memory.jsonl \
  --receipt outputs/guarded-ocean/process-receipt.json \
  --footprint-gib 105 --reserve-gib 6 --swap-growth-gib 2 \
  --max-wall-seconds 3600 --mps-low-watermark 0.3 -- \
  python -m taomate_h3 --backend mps \
  --model-root models/MiniMax-H3 --adapter models/TaoMate-H3 \
  --prompt 'A calm ocean at sunrise.' --duration 5 --resolution 864x480 \
  --output outputs/guarded-ocean/media
```

The guard terminates its child process group if a limit is exceeded or telemetry
fails. It returns 125 for a guard stop and preserves the cause in the receipt.
Interrupted or failed runs are not completed media validation.

## Packaging

```sh
python -m pip install build twine
python -m build
python -m twine check dist/*
```

The wheel includes Python code, LICENSE, NOTICE and MODIFICATIONS.md. The source
distribution also includes examples, scripts, tests and evidence. Neither contains
weights or videos. A source checkout is recommended for examples and experiments.

## Issues and attribution

Include commit, macOS/chip/memory, Python/PyTorch versions, backend, canvas, duration,
options and failing stage in [issues](https://github.com/OpenCraft-AI-Lab/taomate-h3-metal/issues).
Attach a minimal traceback and redacted receipt, removing credentials/private prompts.

Contributions remain subject to [LICENSE](LICENSE). Keep modification notices and
update [MODIFICATIONS.md](MODIFICATIONS.md) when changing an upstream file.
