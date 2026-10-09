# Publication verification — 2026-10-09

This research preview packages source commit `46a769d` plus publication preparation.
Inference algorithms and default precision policy are unchanged. Notice edits alter
source hashes; historical receipts retain their recorded identities.

- [x] English/Chinese Metal installation and usage.
- [x] Upstream revision, LICENSE, NOTICE and attribution.
- [x] Weights, environments, credentials and videos excluded.
- [x] Portable optional model-source path and self-built watchdog fixture.
- [x] Package metadata, console command, source manifest and CI workflow.
- [x] Final lint and CPU / real-MPS regression results recorded.
- [x] Source/wheel build, metadata check and clean installation verified.
- [ ] Public repository and main branch pushed and checked.

Full model benchmarks were not repeated for this release. Historical measurements
and limits are in [METAL_PORT.md](METAL_PORT.md).

## Local verification

- Ruff passed in both the original and fresh environments.
- Source checkout: **258 passed, zero skips**, real MPS, 34.80 seconds.
- Fresh environment with the installed wheel: **258 passed, zero skips**,
  real MPS, 41.34 seconds; dependency compatibility check passed.
- Implicit CPU fallback was disabled. Native macOS telemetry and all optional
  bundled-processor tests ran. No model weights were loaded by pytest.
- Isolated source/wheel builds and `twine check` passed. The wheel includes
  LICENSE, NOTICE and MODIFICATIONS.md; the sdist includes examples, scripts,
  tests and constraints. Both exclude weights and generated videos.
- Installed module location was verified inside the fresh environment's
  site-packages. Console help works outside the source checkout.
- 581 historical Git blobs and all publication files passed the common credential
  pattern scan. Public quickstart/documentation links resolve locally.

The machine-readable [local receipt](verification/publication-20261009.json)
records the test source identities and measurements without local machine paths.
