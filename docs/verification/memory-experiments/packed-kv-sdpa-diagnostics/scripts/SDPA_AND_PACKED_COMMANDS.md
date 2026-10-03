# SDPA and packed8 diagnostics

Preparation used source `dbb6992`. The four authorized GPU controls have now
completed with pinned source `ffa5c0b`; see
`/private/tmp/taomate-metal-diagnostic-runs/followup-ffa5c0b-results.txt` and its
JSON summary. The diagnostic GPU window has been released. Any future rerun
requires coordination with the parent, which has resumed full-model validation.

All scripts accept `--source-ref`. Every tracked Python source file must match
the chosen commit. Use the parent's approved packed8 commit when that code is
merged; do not silently run a changed tree under an old source label.

## SDPA: two independent 12 GiB guarded processes

Both call production `query_chunked_sdpa` unchanged. Inputs are BF16 contiguous
NHD with 56 heads × 128. K/V transpose/unsqueeze remain views, no extra HND copies
are introduced, and production's synchronize after every query block remains.
Every query block sees all keys in a single softmax. Query sizes 64/128/256/512
are tested in order, three times each. First trials and the median of later
trials are reported separately. CUDA equivalence is not tested.

The first preset uses Q=K=24610. The history preset uses Q=20514, K=86392 as an
explicit approximate late-history count. It does not claim an exact count for
every production prompt. Full outputs are compared with the first chunk64
output: bounded CPU full-element relative L2/max-abs, finite check, and byte
SHA256 for genuine bitwise equality. CPU copies/comparison are outside timing.
Memory is sampled before/after each production call; A/R exact allocator peaks
and the external whole-PID footprint peak cover the call. D is sampled only.

```sh
REPO=/Users/supreme-paca/Projects/taomate-h3-metal
PY="$REPO/.venv/bin/python"
DIAG=/private/tmp/taomate-metal-memory-diagnostics
LOCK=/Users/supreme-paca/Projects/h3-metal/outputs/original-baseline-20261002/heavy-gpu.lock
SRC=ffa5c0b
RUN=/private/tmp/taomate-metal-diagnostic-runs/C-SDPA-first-low03
"$PY" -B "$REPO/scripts/run_guarded.py" \
  --watcher /private/tmp/taomate-memory-sample --lock "$LOCK" \
  --log "$RUN/run.log" --memory "$RUN/memory.jsonl" --receipt "$RUN/process-receipt.json" \
  --footprint-gib 12 --reserve-gib 6 --swap-growth-gib 2 \
  --max-wall-seconds 1200 --mps-low-watermark 0.3 -- \
  "$PY" -B "$DIAG/diagnose_sdpa_chunks.py" --guarded --source-ref "$SRC" \
  --output-dir "$RUN/diagnostic" --case first --repeats 3
```

After this process exits and releases the lock, repeat with a fresh `RUN` ending
`C-SDPA-history-low03` and `--case history`. HIGH stays 0.9. The child also stops
at sampled D>10 GiB; the 12 GiB footprint guard is independent and authoritative.

## B: packed8 versus per_layer

`diagnose_quantized_cache.py` now accepts
`--cache-allocation per_layer|packed8`, passed explicitly to
`QuantizedCleanAVKVCache(..., allocation=...)`. Default remains `per_layer`.
An older source without this constructor parameter can still run per_layer;
requesting packed8 fails before any GPU initialization/control allocation.

Use the same newly approved packed8 source commit for both processes, and keep
the same seed, two phases, 28 GiB touched resident control, actual FC1 interleave,
64 MiB weight tiles, 1024-token cache tiles and LOW=.3. Run sequentially under
75 GiB, preserving source/guard/environment equality. The earlier per_layer
reference is useful context but uses an older source commit, so the clean pair
should use the same new source for both allocation policies.

```sh
# Set SRC to the exact reviewed packed8 commit supplied by the parent.
# RUN must be a new path; use distinct B-packed-pair-{per_layer,packed8}-low03 names.
"$PY" -B "$REPO/scripts/run_guarded.py" \
  --watcher /private/tmp/taomate-memory-sample --lock "$LOCK" \
  --log "$RUN/run.log" --memory "$RUN/memory.jsonl" --receipt "$RUN/process-receipt.json" \
  --footprint-gib 75 --reserve-gib 6 --swap-growth-gib 2 \
  --max-wall-seconds 1200 --mps-low-watermark 0.3 -- \
  "$PY" -B "$DIAG/diagnose_quantized_cache.py" --guarded --source-ref "$SRC" \
  --output-dir "$RUN/diagnostic" --phases 2 --resident-gib 28 \
  --interleave-linear fc1 --weight-tile-mib 64 --kv-tile 1024 \
  --cache-allocation packed8
```

The script retains every field from cache.receipt(), including the new
`staged_backing_bytes`, deduplicated `physical_history_payload_bytes`, logical
`staged_payload_bytes` and `allocation`. This separates the lazy packed owners'
unfilled slices from populated logical layer payload. Sampling the same first/
last-layer key rows, scales and FC1 output SHA checks that the comparison uses
matching values. No full-model/video quality claim follows from these samples.

Preparation checks: all scripts compile; tiny CPU SDPA outputs for all four
chunk sizes match on 33,536 elements; bounded output hashing equals full-byte
hashing. `sdpa-preparation-check.json` records these checks and the two plans.
The subsequent MPS results, numerical checks and matched source audit are in the
followup-ffa5c0b report. The earlier CPU preparation receipt remains historical.
