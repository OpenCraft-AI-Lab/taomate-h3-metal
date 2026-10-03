# Opt-in BF16 weight tile budget for W8A16

`TAOMATE_MPS_WEIGHT_TILE_MIB=64` lowers the temporary dequantized BF16 weight
block used by `dequantize` W8A16. Default is still 256 MiB. The environment option
accepts ASCII decimal integers 1..256 and is validated before model loading.
MPSRuntime passes bytes through the pipeline and DiT allocation into each selected
original linear's `_metal_temporary_bytes`. The original 90-projection policy,
INT8/F32-scale formula, BF16 LoRA and 535 checkpoint keys do not change.

This is **only a BF16 weight block budget**, not a process-memory cap:

- Row count is floor(budget / (input_features * 2)); a partial tail is supported.
- Dequantization has additional F32 cast/product temporaries. Conservatively allow
  up to five times the BF16 weight block budget for explicit conversion tensors.
- The full BF16 result `[input_rows, output_features]` is allocated once.
- Each `F.linear` also returns a BF16 `[input_rows, block_output_features]` tensor
  before it is copied into the full result. Both exist together briefly.
- Matmul workspace, input activations, INT8 storage, LoRA, driver allocations and
  allocator caches are additional. Each block still synchronizes before release.

The runtime and allocation receipt record `bf16_weight_tile_budget_bytes` only
for `dequantize`; it is null for `native_int8pack` and `none`. Native INT8pack
execution does not use this tile budget. Original CUDA/default paths retain their
existing behavior. Smaller GEMMs may change accumulation details or throughput;
this option does not claim bitwise matmul equivalence or better model quality.

CPU tests use the real original MergedColumnParallelLinear dispatch with a
1031x512 matrix, crossing a 1 MiB block's 1024-row boundary. Both 1/2 MiB results
are checked against independent FP64 products with explicit BF16 weight rounding;
quantized weights/scales remain identical. Tests also verify option validation,
runtime/pipeline budget propagation and truthful receipts for other backends.
Real MPS units are included and run only when the shared GPU lock is available.
