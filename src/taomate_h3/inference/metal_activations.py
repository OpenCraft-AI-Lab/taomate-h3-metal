# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
# Modified for TaoMate-H3 Metal inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Opt-in reuse of fresh BF16 activation storage, never caller input storage."""

from __future__ import annotations

import os

import torch
from torch.nn import functional as F


def _aliases(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage()._cdata == right.untyped_storage()._cdata


def _eligible(output: torch.Tensor, inputs: torch.Tensor) -> bool:
    return (
        os.environ.get("TAOMATE_MPS_INPLACE_ACTIVATIONS") == "1"
        and output.device.type == "mps"
        and not torch.is_grad_enabled()
        and not output.requires_grad
        and not inputs.requires_grad
        and output.dtype == torch.bfloat16
        and not _aliases(output, inputs)
    )


def _known_fresh_linear(module: object) -> bool:
    from ..distributed.linear import MergedColumnParallelLinear, RowParallelLinear

    # Forward hooks can retain outputs; custom/replaced modules have no ownership
    # guarantee. Restrict this optimization to the original single-device linears.
    if (
        getattr(module, "_forward_hooks", None)
        or getattr(module, "_forward_pre_hooks", None)
        or "forward" in getattr(module, "__dict__", {})
        or getattr(module, "_compiled_call_impl", None) is not None
    ):
        return False
    if getattr(module, "_taomate_h3_inference_lora", False):
        module = module.base
    return (
        type(module) in (MergedColumnParallelLinear, RowParallelLinear)
        and module.tp_size == 1
        and not module._forward_hooks
        and not module._forward_pre_hooks
        and "forward" not in module.__dict__
        and getattr(module, "_compiled_call_impl", None) is None
    )


def _unit_add_owned_(
    output: torch.Tensor, update: torch.Tensor, inputs: torch.Tensor
) -> torch.Tensor:
    """CPU-testable primitive; caller proves output is freshly allocated."""
    if (
        torch.is_grad_enabled()
        or any(t.requires_grad for t in (output, update, inputs))
        or output.dtype != torch.bfloat16
        or update.dtype != output.dtype
        or output.shape != update.shape
        or output.device != update.device
        or _aliases(output, inputs)
        or _aliases(output, update)
    ):
        raise ValueError("unit LoRA add requires detached, nonaliasing BF16 output/update")
    # Multiplication by exactly one introduces no intended quantization change.
    # Nonunit scales deliberately keep the original separate multiply and add.
    return output.add_(update)


def try_inplace_lora(
    module: object,
    output: torch.Tensor,
    update: torch.Tensor,
    inputs: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor | None:
    if (
        scale != 1.0
        or not _eligible(output, inputs)
        or not _known_fresh_linear(module)
        or update.requires_grad
        or update.dtype != output.dtype
        or update.shape != output.shape
        or update.device != output.device
        or _aliases(output, update)
    ):
        return None
    return _unit_add_owned_(output, update, inputs)


def _swiglu_owned_(projected: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Preserve two distinct BF16 rounding boundaries, then contiguous FC2 input."""
    if (
        torch.is_grad_enabled()
        or projected.requires_grad
        or inputs.requires_grad
        or projected.dtype != torch.bfloat16
        or projected.ndim < 1
        or projected.shape[-1] == 0
        or projected.shape[-1] % 2
        or not projected.is_contiguous()
        or _aliases(projected, inputs)
    ):
        raise ValueError("SwiGLU requires detached fresh contiguous BF16 projection")
    gate, up = projected.chunk(2, dim=-1)
    F.silu(gate, inplace=True)  # round to BF16 before multiplication, as upstream
    gate.mul_(up)  # separate BF16 multiplication, not fused activation
    # Original F.silu(gate) * up produces contiguous FC2 input. Preserve that
    # layout rather than changing downstream GEMM kernel selection/accumulation.
    return gate.contiguous()


def try_inplace_swiglu(
    module: object, projected: torch.Tensor, inputs: torch.Tensor
) -> torch.Tensor | None:
    if (
        not _eligible(projected, inputs)
        or not _known_fresh_linear(module)
        or not projected.is_contiguous()
        or projected.ndim < 1
        or projected.shape[-1] == 0
        or projected.shape[-1] % 2
    ):
        return None
    return _swiglu_owned_(projected, inputs)


def can_release_projection(
    projected: torch.Tensor, inputs: torch.Tensor, *consumers: torch.Tensor
) -> bool:
    """Release only a no-grad Metal projection whose replacement owns other storage.

    This predicate never mutates tensors or changes their layout. Checking every
    surviving consumer also handles single-row contiguous() returning an alias.
    """
    return (
        _eligible(projected, inputs)
        and not _aliases(projected, inputs)
        and bool(consumers)
        and all(
            not consumer.requires_grad
            and consumer.device == projected.device
            and not _aliases(projected, consumer)
            for consumer in consumers
        )
    )
