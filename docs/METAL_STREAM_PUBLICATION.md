# Opt-in bounded CPU video publication

Enable `TAOMATE_MPS_STREAM_PUBLICATION=1` for the Python MPS runtime. Default and
CUDA paths keep their existing behavior. This does not make the VAE incremental:
the complete decoded FP32 video still resides in CPU memory before publication.

The MPS path calls the unchanged bundled `processor.revert_tensor` on four-frame
slices and copies each result back into the original decoded storage. It avoids
an additional whole-video normalization/contiguous allocation. Input must be
CPU FP32 `[1,3,T,H,W]`, with nonoverlapping storage; dtype and arithmetic are not
changed. Four 1920x1080 FP32 RGB frames occupy about 95 MiB per temporary instead
of about 33 GiB for a minute. The processor can own several such temporary blocks.

`save_h3_video_mp4(..., frame_chunk_size=4)` then performs the original NumPy
clip, multiply by 255, round and uint8 conversion on each block, writing directly
to FFmpeg stdin. Stderr is spooled to a temporary file to avoid a blocked pipe;
error reporting reads at most 1 MiB. Encoder cleanup, strict decode validation
and atomic destination replacement remain in place. Temporary disk exhaustion or
an encoder failure does not publish an incomplete destination.

The MPS video receipt records `stream_publication` and
`publication_frame_chunk_size`. No full-model speed, peak-memory or quality claim
is made by the CPU tests. The optional upstream-processor tests read local official
source from `TAOMATE_TEST_MODEL_ROOT` and never modify or download it.

Validation in the isolated worktree:

- Original bundled processor: exact FP32 pixels for contiguous and transposed input.
- RGB byte identity at clipping and half-integer rounding boundaries, including
  noncontiguous layouts and uint8 inputs; each conversion block is bounded.
- Actual FFmpeg: MP4 bytes identical for original and chunked input, including a
  frame filter; both outputs pass the existing full decode check.
- Encoder failure preserves an existing destination; large stderr cannot deadlock.
- Explicit runtime switch/receipt and MPS-only postprocessing enforcement.

Run with the worktree source selected explicitly:

```sh
PYTHONPATH=/private/tmp/taomate-video-memory/src \
  /Users/supreme-paca/Projects/taomate-h3-metal/.venv/bin/python \
  -m pytest tests/test_metal_publication.py -q
```
