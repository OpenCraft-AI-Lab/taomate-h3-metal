# Final read-only port review

Reviewed the current official-source Metal adaptation after the 10-second run.
No GPU work or source edits were performed by the reviewer. No blocking issue
was identified.

- CUDA 4/8-GPU paths preserve NCCL, FlashAttention-3, selective W8A8 and
  distributed VAE decoding. This is code inspection, not CUDA hardware testing.
- Teacher, text encoder, Stage3 DiT and VAE resources are released in sequence.
- Condition queries see only condition tokens. Media queries apply a single
  softmax over condition, retained history and current media.
- FP32 patch/time/final parameters remain intact. MPS video VAE uses an explicit
  FP32 policy, with no outer autocast that could bypass protected FP32 regions.
- KV is trimmed every phase; MPS replaces and releases one layer at a time.
  Static timestep cache keys are bounded by the fixed schedule. No unbounded
  accumulation across the 12 requests was found.
- With the default streaming temporal concatenation, VAE windows decode in
  sequence and final frames reside on CPU. The 1,433 native 480p frames of the
  minute example occupy 7,131,525,120 bytes (6.642 GiB) as FP32 RGB, plus
  postprocessing temporaries. At that stage DiT weights have been released.

These findings do not replace the running minute test or establish its peak
physical memory consumption. Source hashes for the reviewed snapshot are in
`../continuation-10s/process-receipt.json`.
