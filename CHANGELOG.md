# Changelog

## 0.1.0 — 2026-10-09 research preview

- Publish the official-source Apple Silicon port under OpenCraft AI Lab with
  upstream provenance and the MiniMax H3 Community License.
- Preserve BF16/FP32 defaults, LoRA, teacher/Stage3 orchestration, clean KV and CUDA.
- Include explicit W8A16, INT8 KV, workspace/lifetime experiments with successful
  and incomplete-run evidence.
- Add English/Chinese quickstarts, dependency constraints, metadata, console
  entrypoint, source packaging and CPU CI.
- Add official 10/60-second reproduction commands and a visual README introduction
  with the historical paired h3.c speed/memory comparison and portable evidence.
- Remove local setup paths and compile macOS watchdog fixtures automatically.

Historical default-path 480p 5/10/60-second and native1080 short quantized runs
are recorded in `docs/verification/`. No new whole-model benchmark is claimed for
this publication release. CUDA parity, smaller-memory Mac support and full-model
request-13 audio-reset validation remain open.
