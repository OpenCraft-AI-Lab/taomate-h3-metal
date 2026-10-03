# Allocation diagnostics, prepared 2026-10-03

These standalone scripts import the approved TaoMate source without editing it.
No model download, inference video, or quality claim is involved. Every run is a
fresh process under the production heavy-GPU lock and a **75 GiB whole-process
footprint guard**. Default pinned source is `3b7fc88`; all tracked `src/*.py` must
match that commit byte for byte. Documentation-only HEAD changes are permitted.

## Recorded quantities

`steps.jsonl` contains bytes, timestamps and synchronization flags for A (active
allocator buffers), R (PyTorch Metal heaps), D (all driver allocations), R−A and
D−R. A/R high-water marks come from Torch's allocator; D is only sampled. The
outer `process-receipt.json` supplies the kernel's lifetime whole-PID footprint
peak, swap change and watchdog result. A payload total is never a process peak.

Current Torch MPS OOM's "MPS allocated" is R, while "other allocations" is D−R.
R−A alone cannot prove fragmentation, and D−R alone cannot identify MPSGraph.

Each result contains source/script hashes, exact argv, torch version, selected
environment variables, seed, per-step telemetry and completion/error status.
Strict decoder/quality evidence is outside this experiment. The scripts stop at
a sampled D > 73 GiB; the outer independent watcher remains the authoritative
75 GiB physical-footprint stop (also 6 GiB system reserve and 2 GiB swap growth).

## A: real matrix and shape, synthetic activation

Default FC1 is real `blocks.2.mlp.fc1.weight`, `[28672,5376]`, quantized by the
production loader. Activation `[20928,5376]` and BF16 output `[20928,28672]`
correspond to the padded third phase of request zero at 1088×1920. BF16 weight
tiles are 64 MiB. Compare three independently guarded processes with same seed:

* `strided`: production column slice destination, one barrier per tile;
* `cat`: retain contiguous matmul pieces, then one concatenation;
* `transposed`: contiguous row destination from transposed pieces, then one
  canonical contiguous copy.

The latter two intentionally retain a second output's worth of live storage at
their final join. They are candidate layouts, not assumed memory improvements.
All variants are compared over **all output elements** with production
`metal_linear`, executed after measured trials. Three repeats reuse allocator
state; the later reference run is excluded from the variant timing. Instrumented
timing includes telemetry; it is not a pristine end-to-end speed benchmark.

## B: real cache geometry, synthetic K/V, actual production cache

`direct_5s_plan()` supplies `[12,10,10,5]` video latent phases, 2040 tokens per
latent, 56 heads × 128, all 50 layers. Audio/text/padding rows follow their real
counts. Each phase creates **one** BF16 K/V pair reused for every layer, which
allocates its own INT8 storage. Stage/commit/trim/phase-end empty-cache use the
production cache implementation. Full finite checks are production stage checks;
sampled key quantization is independently checked against CPU.

`--resident-gib 28` creates and touches uniform 256 MiB blocks to exercise the
allocator pressure threshold. This is a controlled resident payload, not the
actual model or its allocation distribution. `--interleave-linear fc1` runs the
same real FC1 after each staged layer, using one synthetic activation matrix per
phase. It omits the rest of each DiT block; selected output rows from layers
zero/49 are compared. This interleaving can test allocator behavior but cannot
prove that it caused the full model's failure.

Default is one phase. Two phases may approach the 75 GiB guard at LOW=0.8; guard
termination is a useful diagnostic result, not permission to increase the cap.
Four phases require a lower resident control due to the payload budget check.

Optional `--probe-queries 256 --query-chunks 64,128,256` compares query chunking
for one layer per phase, with **all** condition+visible-history+current keys in
every softmax. CPU reconstruction of the quantized history supplies an independent
layout/assembly reference. This checks quantized-cache assembly, not equality to
unquantized history or CUDA. It adds allocations and should be a separate run
from the allocator-pressure comparison.

## Commands

Run only after parent confirms the GPU window. The guard obtains the shared lock
nonblocking; no attempt is made to steal it. Each unique output directory must
not already exist. HIGH remains 0.9. LOW can only be decreased via the reviewed
guard; no cap is disabled.

```sh
REPO=/Users/supreme-paca/Projects/taomate-h3-metal
PY="$REPO/.venv/bin/python"
DIAG=/private/tmp/taomate-metal-memory-diagnostics
LOCK=/Users/supreme-paca/Projects/h3-metal/outputs/original-baseline-20261002/heavy-gpu.lock
RUN=/private/tmp/taomate-metal-diagnostic-runs/A-strided-low08
"$PY" -B "$REPO/scripts/run_guarded.py" \
  --watcher /private/tmp/taomate-memory-sample --lock "$LOCK" \
  --log "$RUN/run.log" --memory "$RUN/memory.jsonl" --receipt "$RUN/process-receipt.json" \
  --footprint-gib 75 --max-wall-seconds 1200 --mps-low-watermark 0.8 -- \
  "$PY" -B "$DIAG/diagnose_weight_layout.py" --guarded \
  --source-ref 3b7fc88 --output-dir "$RUN/diagnostic" --variant strided
```

Repeat A in fresh output directories with `--variant cat` and `--variant
transposed`. Optional resident-control tests use `--resident-gib 28` and unique
run names.

For B use `diagnose_quantized_cache.py --guarded --source-ref 3b7fc88
--output-dir "$RUN/diagnostic" --phases 1 --resident-gib 28
--interleave-linear fc1`. Run LOW=0.8 and LOW=0.3 as fresh processes, preserving
all other flags/seed. No query probe in this primary pair. If appropriate, repeat
with `--phases 2`, still under 75 GiB. `--plan` is CPU-only and does not initialize
MPS; it prints geometry/payload, not an estimated physical peak.

## Falsifiable outcomes

1. If A's strided case increases R−A while cat/transposed do not, investigate the
   output copy/allocator lifetime path, and inspect actual A/R/D step deltas.
2. If B LOW=0.3 preserves sampled numeric outputs/INT8 values and lowers R−A at
   the same live A/cache payload, the allocator heap-tier pressure hypothesis is
   supported. It does not establish full-model peak or speed until rerun there.
3. If D−R grows while R−A remains small, investigate allocations outside the
   PyTorch heap inventory. Capture/profiling may be needed to identify them.
4. If releasing cache/controls and empty_cache returns R near baseline, retained
   heaps during live allocations are implicated. Failure to return memory is not
   automatically a leak: compare allocator/driver/component lifetimes first.
