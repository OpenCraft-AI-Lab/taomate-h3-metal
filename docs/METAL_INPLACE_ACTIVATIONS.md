# Opt-in fresh activation reuse

`TAOMATE_MPS_INPLACE_ACTIVATIONS=1` enables two inference-only BF16 storage
optimizations on MPS. Default, CUDA, grad-enabled execution and tensors requiring
gradients keep the original path. Model receipts record the flag.

## Unit-scale LoRA

For scale exactly 1, the original two BF16 `F.linear` adapter calls are unchanged.
The freshly allocated base output is updated with `output.add_(update)`. This
avoids both the complete `update * 1` allocation and the complete new sum output.
Nonunit scales keep the original separate BF16 multiply and add, with no fused
alpha or rounding-order change. The current official rank128/alpha128 case is
covered; the condition is mathematically unit scale, not an assumption about rank.

Ownership is restricted to original single-device merged-column/row modules,
without output observer hooks, custom instance forwards or compiled overrides.
Input/output storage aliasing disables reuse. Adapter updates are not mutated.
No weight, KV, quantization policy, attention or final FP32/VAE operation changes.

## MLP SwiGLU

The freshly allocated FC1 projection is split into gate/up views. The gate uses
`F.silu(..., inplace=True)` and then a separate `mul_(up)`, retaining the original
BF16 rounding after SiLU and after multiplication. The up half and caller input
are unchanged. A final `contiguous()` preserves the original FC2 input layout;
therefore this optimization still allocates a half-width output in general. It
avoids the separate SiLU intermediate, rather than claiming zero temporary storage.
It deliberately does not fuse the sigmoid, multiplication or downstream GEMM.

CPU tests check exact BF16 bit patterns (including signed zeros/subnormals),
original LoRA wrapper and MLP outputs, input/update immutability, storage reuse,
nonunit/grad/CPU fallbacks and rejected unsafe aliases. Matching real MPS tests
are included; the parent runs them only after the active full-size job finishes.
A successful small test does not establish full-model physical-peak reduction.
