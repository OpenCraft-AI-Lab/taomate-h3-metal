# OpenCraft modification notice

This independent Metal/MPS port modifies official TaoMate-H3 at
`6b2f9984acf681e2f387188f5d1f3b26e504315b`. Original license and upstream attribution are retained.

Modified code carries an OpenCraft notice. Added code, tests, documentation and
verification reports are OpenCraft modifications under the same LICENSE.
Historical logs and hashes identify their original runs and remain evidence.
The h3.c comparison includes a small external source patch excerpt under its
original MIT license, with attribution and license text in the evidence directory.

## Changes to upstream files

README.md is replaced with a Metal quickstart; the pinned upstream CUDA guide
is retained in docs/UPSTREAM_CUDA.md. pyproject.toml and .gitignore are extended;
NOTICE adds OpenCraft attribution. LICENSE is unchanged.

## Python code, public scripts and tests

| File | Status relative to pinned upstream |
| --- | --- |
| `scripts/benchmark_metal_linear.py` | Added |
| `scripts/build_upstream_fixtures.py` | Added |
| `scripts/compare_experiments.py` | Added |
| `scripts/memory_sample.c` | Added |
| `scripts/prepare_experiment_guidance.py` | Added |
| `scripts/process_watch.c` | Added |
| `scripts/run_cached_guidance.py` | Added |
| `scripts/run_guarded.py` | Added |
| `scripts/run_locked.py` | Added |
| `scripts/verify_mps_components.py` | Added |
| `src/taomate_h3/inference/metal_activations.py` | Added |
| `src/taomate_h3/inference/metal_attention.py` | Added |
| `src/taomate_h3/inference/metal_loader.py` | Added |
| `src/taomate_h3/inference/metal_memory.py` | Added |
| `src/taomate_h3/inference/metal_publication.py` | Added |
| `src/taomate_h3/inference/metal_quantization.py` | Added |
| `src/taomate_h3/runtime/backend.py` | Added |
| `src/taomate_h3/runtime/mps.py` | Added |
| `src/taomate_h3/runtime/mps_memory.py` | Added |
| `src/taomate_h3/streaming/metal_kv.py` | Added |
| `tests/conftest.py` | Added |
| `tests/test_memory_guard.py` | Added |
| `tests/test_metal_activations.py` | Added |
| `tests/test_metal_attention.py` | Added |
| `tests/test_metal_clean_lifetimes.py` | Added |
| `tests/test_metal_expanded_coverage.py` | Added |
| `tests/test_metal_kv.py` | Added |
| `tests/test_metal_kv_packed.py` | Added |
| `tests/test_metal_kv_tiles.py` | Added |
| `tests/test_metal_loader.py` | Added |
| `tests/test_metal_memory.py` | Added |
| `tests/test_metal_memory_trace.py` | Added |
| `tests/test_metal_numerics.py` | Added |
| `tests/test_metal_projection_lifetimes.py` | Added |
| `tests/test_metal_publication.py` | Added |
| `tests/test_metal_quantization.py` | Added |
| `tests/test_metal_row_views.py` | Added |
| `tests/test_metal_vae_lifetimes.py` | Added |
| `tests/test_metal_weight_tiles.py` | Added |
| `tests/test_mps_memory.py` | Added |
| `tests/test_mps_runtime.py` | Added |
| `tests/test_upstream_contracts.py` | Added |
| `src/taomate_h3/cli.py` | Modified |
| `src/taomate_h3/config.py` | Modified |
| `src/taomate_h3/distributed/collectives.py` | Modified |
| `src/taomate_h3/distributed/linear.py` | Modified |
| `src/taomate_h3/distributed/parallel.py` | Modified |
| `src/taomate_h3/distributed/ulysses.py` | Modified |
| `src/taomate_h3/inference/lora_checkpoint.py` | Modified |
| `src/taomate_h3/model/dit.py` | Modified |
| `src/taomate_h3/model/layers.py` | Modified |
| `src/taomate_h3/model/output.py` | Modified |
| `src/taomate_h3/model/pipeline.py` | Modified |
| `src/taomate_h3/model/weight_loading.py` | Modified |
| `src/taomate_h3/runner.py` | Modified |
| `src/taomate_h3/streaming/attention_hook.py` | Modified |
| `src/taomate_h3/streaming/cache.py` | Modified |
| `src/taomate_h3/streaming/runtime.py` | Modified |
| `src/taomate_h3/streaming/session.py` | Modified |
| `src/taomate_h3/teacher.py` | Modified |
