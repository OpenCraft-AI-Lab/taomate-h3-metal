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

## Projection lifetime trimming

The same opt-in also permits dropping two obsolete local references, with no
operator, stride, rounding or KV-window change:

- In the non-fused, single-Ulysses-rank attention path, the existing Q/K/V stack
  remains unchanged. Once all three returned tensors own storage distinct from
  the original projection, its `qkv` local is released before attention.
- After the existing in-place SwiGLU produces the contiguous FC2 input, the
  full-width `projected` local is released before FC2 only if the result does not
  alias its storage. A single-row contiguous view may alias and is retained.

These checks require MPS BF16 no-grad inference and the opt-in flag. Default,
CUDA and gradient paths retain the prior lifetime. Estimated obsolete buffers
are about 0.822 GiB QKV and 1.096 GiB FC1 for 20,514 native media rows, excluding
text/padding. They exist at different points; this is not an additive or measured
process-peak reduction. FC1's own calculation peak is unaffected.

CPU tests simulate only the device eligibility gate for complete official tiny
attention/MLP+LoRA calls, compare exact BF16 output bits and staged INT8 K/V/F32
scales, and inspect weak references at downstream projection entry. They also
cover real production gating, unchanged noncontiguous inputs, and single-row
alias fallback. Corresponding small MPS cases are provided for an exclusive GPU
window; they were not executed during this implementation.

The same gate now trims clean-commit hook temporaries after the final document
`torch.cat` and before `cache.stage`: document output lists, condition/media
selection copies, and their attention outputs. Original Q/K/V remain live and
are passed unchanged to staging. Every temporary local is initialized before
the document loop, so cleanup is valid for arbitrary live-document position,
padding documents and noncontiguous masks. No gather becomes a view and noisy
forward lifetime is unchanged. CPU tests inspect weak references at stage entry
and compare final attention output plus INT8 K/V and F32 scales, including full
tiny official LoRA attention. MPS variants await an exclusive test window.
These released buffers overlap other lifetime optimizations; theoretical byte
counts must not be added together as a measured peak saving.

## Contiguous attention row views (separate opt-in)

`TAOMATE_MPS_ATTENTION_ROW_VIEWS=1` permits MPS/no-grad hook selections to use a
slice only when cached selected indices are verified to equal an ascending
contiguous range and the source tensor is contiguous. This preserves the
original gather's shape, strides, dtype and values. Noncontiguous masks, reordered
indices, strided inputs, default/CUDA and gradients retain `index_select`.
Condition/media visibility, the original full-key softmax, KV quantization and
retention are unchanged. Views share immutable caller storage; no consumer may
mutate them. Activation-lifetime cleanup still drops temporary views before
clean staging. The process receipt records the actual option in its environment.

This candidate follows the native ten-second failure that persisted after pure
lifetime releases; its aim is removing three current-media gather copies during
attention. Savings are not yet an established end-to-end process-peak reduction.
Tests cover storage sharing, exact output/stride, all fallback paths, arbitrary
masks and padding documents, complete LoRA attention and staged INT8 K/V/scales.
