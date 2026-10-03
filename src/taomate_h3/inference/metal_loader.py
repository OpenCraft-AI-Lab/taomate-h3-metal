# Modified for TaoMate-H3 Metal inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Allocate official meta modules once in their final storage dtype/device.

CPU AdaLN storage and selective W8A16 are explicit experiments. Original module
classes and checkpoint keys remain intact for the official loader and LoRA.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from taomate_h3.distributed.linear import MergedColumnParallelLinear, RowParallelLinear

from .w8a8 import INTERIOR_FC1_QKV_POLICY, resolve_h3_cutlass_w8a8_policy

METAL_EXPANDED_WEIGHT_POLICIES = {
    "metal_interior_fc1_qkv_out_v1": ("attn.out_proj",),
    "metal_interior_fc1_qkv_fc2_v1": ("mlp.fc2",),
    "metal_interior_all_linear_v1": ("attn.out_proj", "mlp.fc2"),
}


def parse_metal_weight_policy(value: str) -> str:
    if value != INTERIOR_FC1_QKV_POLICY and value not in METAL_EXPANDED_WEIGHT_POLICIES:
        raise ValueError("unknown TAOMATE_MPS_WEIGHT_COVERAGE policy")
    return value


def resolve_metal_weight_policy(names: list[str], *, policy: str) -> dict[str, Any]:
    """Metal-only experimental expansion; the official CUDA resolver is untouched.

    Extra base weights may change quality. Boundary blocks, refiners, FP32 heads
    and trained LoRA buffers are never selected by the additional policies.
    """
    parse_metal_weight_policy(policy)
    original = resolve_h3_cutlass_w8a8_policy(names, policy=INTERIOR_FC1_QKV_POLICY)
    base = list(original["quantized_linear_names"])
    extras = []
    if policy != INTERIOR_FC1_QKV_POLICY:
        expected_base = {
            f"blocks.{block}.{suffix}"
            for block in range(2, 47)
            for suffix in ("attn.qkv_proj", "mlp.fc1")
        }
        if set(base) != expected_base or original["bf16_boundary_block_indices"] != [
            0,
            1,
            47,
            48,
            49,
        ]:
            raise ValueError("expanded Metal policy requires the original 50-block architecture")
        extras = [
            f"blocks.{block}.{suffix}"
            for block in range(2, 47)
            for suffix in METAL_EXPANDED_WEIGHT_POLICIES[policy]
        ]
        if any(name not in names for name in extras):
            raise ValueError("expanded Metal policy projection inventory differs")
    return {
        **original,
        "quantized_linear_names": sorted(base + extras),
        "bf16_protected_linear_names": [
            name for name in original["bf16_protected_linear_names"] if name not in extras
        ],
        "additional_quantized_linear_names": sorted(extras),
        "policy": policy,
    }


def parse_weight_tile_mib(value: str) -> int:
    """Parse the opt-in BF16 dequantized weight budget, returning bytes."""
    if (
        not value
        or not value.isascii()
        or not value.isdecimal()
        or len(value) > 3
        or not 1 <= int(value) <= 256
    ):
        raise ValueError("TAOMATE_MPS_WEIGHT_TILE_MIB must be an integer in 1..256")
    return int(value) * 1024**2


def allocate_metal_storage_(
    model_meta: nn.Module,
    device: str | torch.device = "mps",
    *,
    cpu_adaln: bool = True,
    quantization_backend: str = "none",
    quantization_policy: str = INTERIOR_FC1_QKV_POLICY,
    temporary_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Materialize meta parameters/buffers without BF16 GPU staging copies.

    CPU is supported for deterministic tests. No parameter data are initialized;
    the caller must completely load the checkpoint before executing the model.
    Allocation failure requires discarding the partially materialized model.
    """
    target = torch.device(device)
    if target.type not in ("cpu", "mps"):
        raise ValueError("Metal allocation supports CPU tests or MPS")
    if quantization_backend not in ("none", "dequantize", "native_int8pack"):
        raise ValueError("unsupported Metal quantization backend")
    if (
        isinstance(temporary_bytes, bool)
        or not isinstance(temporary_bytes, int)
        or not 0 < temporary_bytes <= 256 * 1024**2
    ):
        raise ValueError("temporary_bytes must be a positive integer at most 256 MiB")
    parse_metal_weight_policy(quantization_policy)
    if quantization_backend == "none" and quantization_policy != INTERIOR_FC1_QKV_POLICY:
        raise ValueError("expanded Metal coverage requires an enabled quantization backend")
    modules = dict(model_meta.named_modules())
    original_keys = tuple(model_meta.state_dict())
    selection = (
        resolve_metal_weight_policy(list(modules), policy=quantization_policy)
        if quantization_backend != "none"
        else {"quantized_linear_names": [], "additional_quantized_linear_names": []}
    )
    selected = set(selection["quantized_linear_names"])
    plans = []
    alias_specs: dict[int, tuple] = {}
    for name, module in modules.items():
        if name in selected:
            expected_type = (
                RowParallelLinear
                if name in selection["additional_quantized_linear_names"]
                else MergedColumnParallelLinear
            )
            if (
                type(module) is not expected_type
                or module.tp_size != 1
                or module.parallel_context.tp_world_size != 1
                or module.parallel_context.ulysses_world_size != 1
            ):
                raise ValueError(
                    f"{name}: quantization requires original single-device {expected_type.__name__}"
                )
            if module.bias is not None or module.weight.dtype != torch.bfloat16:
                raise ValueError(f"{name}: quantization requires bias-free BF16 weight")
            if (
                quantization_backend == "dequantize"
                and temporary_bytes < module.weight.shape[1] * 2
            ):
                raise ValueError(f"{name}: budget cannot fit one BF16 row")
            if quantization_backend == "native_int8pack" and any(
                n % 32 for n in module.weight.shape
            ):
                raise ValueError(f"{name}: native dimensions must be 32-aligned")
        for kind, items in (("parameter", module._parameters), ("buffer", module._buffers)):
            for leaf, tensor in items.items():
                if tensor is None:
                    continue
                if tensor.device.type != "meta":
                    raise ValueError(f"{name}.{leaf}: all storage must initially be meta")
                host_adaln = (
                    cpu_adaln and name.endswith("adaln_proj.linear") and leaf in ("weight", "bias")
                )
                if host_adaln and tensor.dtype != torch.bfloat16:
                    raise ValueError(f"{name}.{leaf}: CPU AdaLN must be BF16")
                destination = torch.device("cpu") if host_adaln else target
                quantized = name in selected and leaf == "weight" and kind == "parameter"
                dtype = torch.int8 if quantized else tensor.dtype
                spec = (destination, dtype, kind)
                if id(tensor) in alias_specs and alias_specs[id(tensor)] != spec:
                    raise ValueError("shared storage has conflicting allocation policies")
                alias_specs[id(tensor)] = spec
                plans.append((module, leaf, tensor, destination, dtype, kind))
    allocated = {}
    for module, leaf, tensor, destination, dtype, kind in plans:
        if id(tensor) not in allocated:
            value = torch.empty_like(tensor, device=destination, dtype=dtype)
            allocated[id(tensor)] = (
                nn.Parameter(
                    value, requires_grad=tensor.requires_grad if dtype != torch.int8 else False
                )
                if kind == "parameter"
                else value
            )
        if kind == "parameter":
            module._parameters[leaf] = allocated[id(tensor)]
        else:
            module._buffers[leaf] = allocated[id(tensor)]
    for name in selected:
        module = modules[name]
        rows = module.weight.shape[0]
        module.register_buffer(
            "_metal_weight_scale",
            torch.empty((rows, 1), dtype=torch.float32, device=target),
            persistent=False,
        )
        module.register_buffer(
            "_metal_native_scale",
            torch.empty(rows, dtype=torch.bfloat16, device=target)
            if quantization_backend == "native_int8pack"
            else None,
            persistent=False,
        )
        module._metal_stream_load_pending = True
        module._metal_stream_loaded = False
        module._metal_quantization_backend = quantization_backend
        module._metal_trace_target = name
        module._metal_temporary_bytes = temporary_bytes
    if tuple(model_meta.state_dict()) != original_keys:
        raise RuntimeError("Metal allocation changed checkpoint keys")
    return {
        "quantization": "W8A16" if selected else "none",
        "backend": quantization_backend,
        "policy": quantization_policy,
        "additional_quantized_linear_names": selection["additional_quantized_linear_names"],
        "additional_weight_storage_saved_bytes": sum(
            modules[name].weight.numel()
            - modules[name].weight.shape[0]
            * (6 if quantization_backend == "native_int8pack" else 4)
            for name in selection["additional_quantized_linear_names"]
        ),
        "additional_coverage_note": (
            "Lossy base-weight expansion; storage savings are not process peak savings "
            "or quality equivalence."
        ),
        "quantized_linear_names": sorted(selected),
        "cpu_adaln": cpu_adaln,
        "checkpoint_key_count": len(original_keys),
        "weight_tile_budget_scope": (
            "BF16 weights only; excludes FP32 scratch, outputs and matmul workspace"
        ),
        "bf16_weight_tile_budget_bytes": (
            temporary_bytes if quantization_backend == "dequantize" else None
        ),
    }


@torch.no_grad()
def load_metal_stream_weight_(
    owner: nn.Module,
    source_BF16_CPU: torch.Tensor,
    *,
    maximum_temporary_bytes: int = 8 * 1024 * 1024,
) -> None:
    """Write row-quantized source into final INT8 storage, preserving Parameter identity.

    CPU scratch follows the official F32 scale formula. Its intermediate tensors
    can use ~five times the BF16 row budget; no BF16 device weight is created.
    Pending remains true on any failure, so incomplete weights cannot execute.
    """
    source = source_BF16_CPU
    if getattr(owner, "_metal_stream_load_pending", False) is not True:
        raise RuntimeError("no pending Metal streamed weight")
    if source.device.type != "cpu" or source.dtype != torch.bfloat16 or source.ndim != 2:
        raise ValueError("source must be CPU BF16 matrix")
    if owner.weight.dtype != torch.int8 or source.shape != owner.weight.shape:
        raise ValueError("streamed weight shape/dtype differs")
    row_bytes = source.shape[1] * 2
    if (
        isinstance(maximum_temporary_bytes, bool)
        or not isinstance(maximum_temporary_bytes, int)
        or maximum_temporary_bytes < row_bytes
    ):
        raise ValueError("load budget must fit one BF16 row")
    rows = maximum_temporary_bytes // row_bytes
    for start in range(0, source.shape[0], rows):
        part = source[start : start + rows].detach()
        if not torch.isfinite(part).all():
            raise ValueError("nonfinite source weight")
        scale = (part.abs().amax(1, keepdim=True).float() / 127).clamp_min(1e-12)
        values = (part / scale).round().clamp(-127, 127).to(torch.int8)
        owner.weight[start : start + rows].copy_(values)
        owner._metal_weight_scale[start : start + rows].copy_(scale)
        if owner._metal_native_scale is not None:
            owner._metal_native_scale[start : start + rows].copy_(scale.flatten().bfloat16())
        if owner.weight.device.type == "mps":
            torch.mps.synchronize()
    owner._metal_stream_loaded = True
    owner._metal_stream_load_pending = False


@torch.no_grad()
def metal_linear(inputs: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """W8A16 using owner buffers directly; no registered helper or BF16 cache."""
    if getattr(module, "_metal_stream_load_pending", True) or not getattr(
        module, "_metal_stream_loaded", False
    ):
        raise RuntimeError("Metal weight loading incomplete")
    if inputs.dtype != torch.bfloat16 or inputs.device != module.weight.device:
        raise ValueError("Metal W8A16 requires colocated BF16 inputs")
    if inputs.ndim < 1 or inputs.shape[-1] != module.weight.shape[1] or module.bias is not None:
        raise ValueError("Metal linear shape/bias differs")
    flat = inputs.reshape(-1, inputs.shape[-1]).contiguous()
    shape = (*inputs.shape[:-1], module.weight.shape[0])
    if not len(flat):
        return inputs.new_empty(shape)
    if module._metal_quantization_backend == "native_int8pack":
        if inputs.device.type != "mps" or module._metal_native_scale.dtype != torch.bfloat16:
            raise ValueError("native INT8pack requires MPS and BF16 scales")
        result = torch.ops.aten._weight_int8pack_mm(flat, module.weight, module._metal_native_scale)
    elif module._metal_quantization_backend == "dequantize":
        from ..runtime.mps_memory import memory_trace_enabled, trace_mps_memory

        if memory_trace_enabled(inputs.device.type):
            trace_mps_memory(
                "before_weight_linear_output",
                layer=getattr(module, "_metal_trace_target", "unknown"),
                rows=len(flat),
                input_columns=module.weight.shape[1],
                output_columns=module.weight.shape[0],
                requested_output_bytes=len(flat) * module.weight.shape[0] * inputs.element_size(),
            )
        result = inputs.new_empty((len(flat), module.weight.shape[0]))
        rows = module._metal_temporary_bytes // (module.weight.shape[1] * 2)
        for start in range(0, module.weight.shape[0], rows):
            temporary = (
                module.weight[start : start + rows].float()
                * module._metal_weight_scale[start : start + rows]
            ).bfloat16()
            result[:, start : start + rows] = F.linear(flat, temporary)
            if inputs.device.type == "mps":
                torch.mps.synchronize()
            del temporary
    else:
        raise ValueError("unknown Metal linear backend")
    return result.reshape(shape)
