# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
# Modified for TaoMate-H3 Metal inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Opt-in W8A16 experiments; neither backend quantizes activations.

``dequantize`` uses bounded BF16 weight staging and F32 per-row scales.
``native_int8pack`` uses PyTorch's private MPS operator and BF16 scales;
its fused arithmetic differs from rounding every dequantized weight to BF16.
No CUDA W8A8 equivalence or quality parity is implied. Not wired into the CLI.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .w8a8 import INTERIOR_FC1_QKV_POLICY, resolve_h3_cutlass_w8a8_policy


class MetalW8A16Linear(nn.Module):
    """Inference-only weight storage with no persistent dequantized weights.

    The budget covers BF16 weight staging. Dequantization additionally uses two
    F32 row-block temporaries, so its explicit tensor bound is five times that
    budget, excluding output/activation and the MPS allocator's cached blocks.
    MPS synchronization after each block prevents queued blocks accumulating.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        backend: str = "dequantize",
        temporary_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        super().__init__()
        if backend not in ("dequantize", "native_int8pack"):
            raise ValueError("unknown W8A16 backend")
        if weight.ndim != 2 or weight.dtype != torch.int8 or not all(weight.shape):
            raise ValueError("weight must be a nonempty INT8 matrix")
        if scale.shape != (weight.shape[0], 1) or scale.dtype != torch.float32:
            raise ValueError("scale must be F32 [out_features, 1]")
        if scale.device != weight.device or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("scales must be positive finite and colocated")
        if temporary_bytes < weight.shape[1] * 2:
            raise ValueError("temporary budget cannot fit one BF16 weight row")
        if backend == "native_int8pack":
            if weight.device.type != "mps" or any(n % 32 for n in weight.shape):
                raise ValueError(
                    "experimental native backend requires MPS and 32-aligned dimensions"
                )
        self.register_buffer("weight", weight.contiguous())
        self.register_buffer("scale", scale.contiguous())
        self.register_buffer(
            "native_scale",
            scale.flatten().to(torch.bfloat16) if backend == "native_int8pack" else None,
        )
        self.backend = backend
        self.temporary_bytes = int(temporary_bytes)
        self.bias = None
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    @classmethod
    @torch.no_grad()
    def from_weight(
        cls,
        source: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        backend: str = "dequantize",
        temporary_bytes: int = 256 * 1024 * 1024,
        load_rows: int = 128,
    ) -> MetalW8A16Linear:
        """Row-stream quantization; source should be CPU BF16 checkpoint storage.

        CPU quantization matches the official per-row formula, including BF16
        source rounding. No complete BF16 target-device copy is materialized.
        """
        if source.ndim != 2 or source.dtype != torch.bfloat16 or not all(source.shape):
            raise ValueError("source must be a nonempty BF16 matrix")
        if temporary_bytes < source.shape[1] * 2:
            raise ValueError("temporary budget cannot fit one BF16 weight row")
        if load_rows <= 0:
            raise ValueError("load_rows must be positive")
        target = source.device if device is None else torch.device(device)
        quantized = torch.empty(source.shape, dtype=torch.int8, device=target)
        scales = torch.empty((source.shape[0], 1), dtype=torch.float32, device=target)
        for start in range(0, source.shape[0], load_rows):
            rows = source[start : start + load_rows].detach().to("cpu")
            if not torch.isfinite(rows).all():
                raise ValueError("source weight is not finite")
            scale = (rows.abs().amax(dim=1, keepdim=True).float() / 127).clamp_min(1e-12)
            values = (rows / scale).round().clamp(-127, 127).to(torch.int8)
            quantized[start : start + load_rows].copy_(values)
            scales[start : start + load_rows].copy_(scale)
            if target.type == "mps":
                torch.mps.synchronize()
        return cls(quantized, scales, backend=backend, temporary_bytes=temporary_bytes)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dtype != torch.bfloat16 or inputs.device != self.weight.device:
            raise ValueError("W8A16 requires colocated BF16 activations")
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError("activation width differs")
        flat = inputs.reshape(-1, self.in_features).contiguous()
        if flat.shape[0] == 0:
            return inputs.new_empty((*inputs.shape[:-1], self.out_features))
        if self.backend == "native_int8pack":
            # This private kernel interprets scales as the activation dtype;
            # F32 scales with BF16 input can silently produce invalid results.
            result = torch.ops.aten._weight_int8pack_mm(flat, self.weight, self.native_scale)
        else:
            result = torch.empty(
                (flat.shape[0], self.out_features), dtype=inputs.dtype, device=inputs.device
            )
            rows = max(1, self.temporary_bytes // (self.in_features * 2))
            for start in range(0, self.out_features, rows):
                stop = min(start + rows, self.out_features)
                temporary = (self.weight[start:stop].float() * self.scale[start:stop]).to(
                    inputs.dtype
                )
                result[:, start:stop] = F.linear(flat, temporary)
                if inputs.device.type == "mps":
                    torch.mps.synchronize()
                del temporary
        return result.reshape(*inputs.shape[:-1], self.out_features)

    def receipt(self) -> dict[str, Any]:
        return {
            "scheme": "W8A16",
            "backend": self.backend,
            "activation_dtype": "bfloat16",
            "lora_dtype": "unchanged",
            "scale_compute_dtype": "bfloat16" if self.native_scale is not None else "float32",
            "persistent_bytes": sum(t.numel() * t.element_size() for t in self.buffers()),
            "bf16_staging_budget_bytes": self.temporary_bytes,
            "persistent_dequantized_weights": False,
        }


@torch.no_grad()
def quantize_interior_linears_(
    model: nn.Module, *, backend: str = "dequantize", temporary_bytes: int = 256 * 1024 * 1024
) -> dict[str, Any]:
    """Experimental post-load conversion. Apply AFTER official LoRA attachment.

    Only selected base projections are replaced; the LoRA wrapper and its A/B
    buffers remain untouched. This is not a low-peak checkpoint loader: the
    caller already owns the BF16 model. Conversion failures require discarding
    the partially converted model. Single-device inference only.
    """
    modules = dict(model.named_modules())
    selection = resolve_h3_cutlass_w8a8_policy(list(modules), policy=INTERIOR_FC1_QKV_POLICY)
    targets = []
    for name in selection["quantized_linear_names"]:
        wrapper = modules[name]
        base = getattr(wrapper, "base", wrapper)
        if getattr(base, "tp_size", 1) != 1 or getattr(base, "bias", None) is not None:
            raise ValueError(f"{name}: expected single-device bias-free projection")
        if not hasattr(base, "weight") or base.weight.dtype != torch.bfloat16:
            raise ValueError(f"{name}: expected BF16 base weights")
        targets.append((name, wrapper, base))
    for name, wrapper, base in targets:
        replacement = MetalW8A16Linear.from_weight(
            base.weight, backend=backend, temporary_bytes=temporary_bytes
        )
        if wrapper is not base:
            wrapper.base = replacement
        else:
            parent_name, leaf = name.rsplit(".", 1)
            setattr(modules[parent_name], leaf, replacement)
        # Do not retain source references for all layers during conversion.
        base.weight = None
    return {
        **selection,
        "scheme": "W8A16",
        "backend": backend,
        "policy": INTERIOR_FC1_QKV_POLICY,
        "lora": "unchanged BF16",
        "bf16_staging_budget_bytes": temporary_bytes,
    }
