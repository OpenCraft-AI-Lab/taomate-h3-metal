# Opt-in INT8 KV tile size

For MPS INT8 KV storage, `TAOMATE_MPS_KV_TILE_TOKENS=1024` changes the token
block used by finite validation, clean-KV quantization and history restoration.
The default remains 128. The quantization formula, scale precision, clean-commit
transactions, history retention and full-key softmax are unchanged. Each tile
still synchronizes, so larger tiles reduce host synchronization count without
queuing all 50 layers' temporaries. BF16 KV storage is unaffected.

The option accepts ASCII decimal integers 1..2048. A separate conservative
scratch estimate must fit 256 MiB:

```
tile_tokens * (20 * heads * head_dim + 32 * heads + 8)
```

The estimate includes five FP32-equivalent tiles for arithmetic intermediates,
replacement overlap, scales, index tensors and finite validation workspace.
It excludes joined BF16 K/V, persistent/staged INT8 storage, attention workspace,
model activations and driver allocator caching. This is a bounding design estimate,
not a measured process peak. At 56 heads/dim128, tile1024 estimates 141.76 MiB;
tile2048 is rejected by the 256 MiB limit. The MPS runtime rejects over-budget
settings before loading models, and the cache validates again against its actual
head count. Small-head tests may use 2048 within budget.

Model metadata records the active token tile, estimate and budget. The cache
receipt records the actual tile and estimate alongside logical/physical/staged
payload bytes. No speed or real-model quality improvement is claimed by tests.

Tests compare every K/V INT8 value and F32 scale for 50-layer clean commits across
128/1024 boundaries, then require bitwise identical complete attention outputs.
They also reject NaN/Inf at both tile boundaries and verify early budget rejection.
