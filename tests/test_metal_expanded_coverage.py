"""Explicit lossy base-weight policies preserve the official architecture and LoRA."""

import pytest
import torch
from test_metal_loader import model as model

from taomate_h3.distributed import ParallelContext
from taomate_h3.distributed.linear import RowParallelLinear
from taomate_h3.inference.metal_loader import (
    METAL_EXPANDED_WEIGHT_POLICIES,
    allocate_metal_storage_,
    parse_metal_weight_policy,
    resolve_metal_weight_policy,
)
from taomate_h3.inference.w8a8 import INTERIOR_FC1_QKV_POLICY, resolve_h3_cutlass_w8a8_policy


@pytest.mark.parametrize("policy", [INTERIOR_FC1_QKV_POLICY, *METAL_EXPANDED_WEIGHT_POLICIES])
def test_exact_inventory_protection_keys_and_bytes(model, policy):
    before = {k: (v.shape, v.dtype) for k, v in model.state_dict().items()}
    assert len(before) == 535
    names = list(dict(model.named_modules()))
    official = resolve_h3_cutlass_w8a8_policy(names, policy=INTERIOR_FC1_QKV_POLICY)
    selection = resolve_metal_weight_policy(names, policy=policy)
    assert not set(selection["quantized_linear_names"]) & set(
        selection["bf16_protected_linear_names"]
    )
    if policy == INTERIOR_FC1_QKV_POLICY:
        assert set(selection["quantized_linear_names"]) == set(official["quantized_linear_names"])
    extra = {
        f"blocks.{i}.{suffix}"
        for i in range(2, 47)
        for suffix in METAL_EXPANDED_WEIGHT_POLICIES.get(policy, ())
    }
    receipt = allocate_metal_storage_(
        model, "cpu", quantization_backend="dequantize", quantization_policy=policy
    )
    assert set(model.state_dict()) == set(before)
    assert set(receipt["additional_quantized_linear_names"]) == extra
    assert len(receipt["quantized_linear_names"]) == 90 + len(extra)
    selected = set(receipt["quantized_linear_names"])
    saved = 0
    for key, value in model.state_dict().items():
        owner = key.rsplit(".", 1)[0]
        expected = torch.int8 if owner in selected and key.endswith(".weight") else before[key][1]
        assert value.dtype == expected and value.shape == before[key][0]
        if owner in extra:
            module = model.get_submodule(owner)
            assert type(module) is RowParallelLinear and not module.weight.requires_grad
            assert module._metal_weight_scale.dtype == torch.float32
            saved += value.numel() - value.shape[0] * 4
    assert receipt["additional_weight_storage_saved_bytes"] == saved
    assert all(2 <= int(name.split(".")[1]) <= 46 for name in selected)
    for name in [
        "blocks.0.attn.out_proj",
        "blocks.1.mlp.fc2",
        "blocks.47.mlp.fc2",
        "blocks.49.attn.out_proj",
    ]:
        assert model.get_submodule(name).weight.dtype == torch.bfloat16
    # Parsing remains Metal-specific; the original CUDA resolver rejects expansion.
    if policy != INTERIOR_FC1_QKV_POLICY:
        with pytest.raises(ValueError, match="unsupported H3"):
            resolve_h3_cutlass_w8a8_policy(names, policy=policy)


@pytest.mark.parametrize("value", ["", "all", "interior_fc2", "INTERIOR_FC1_QKV_V1"])
def test_invalid_policy_fails_before_allocation(model, value):
    with pytest.raises(ValueError, match="COVERAGE"):
        allocate_metal_storage_(
            model, "cpu", quantization_backend="dequantize", quantization_policy=value
        )
    assert all(p.device.type == "meta" for p in model.parameters())


def test_disabled_quantization_cannot_silently_accept_expansion(model):
    with pytest.raises(ValueError, match="enabled"):
        allocate_metal_storage_(model, "cpu", quantization_policy="metal_interior_all_linear_v1")
    assert all(p.device.type == "meta" for p in model.parameters())


@pytest.mark.parametrize("axis", ["tp", "ulysses"])
def test_expanded_row_requires_world_one_before_allocating(model, axis):
    owner = model.blocks[2].mlp.fc2
    owner.parallel_context = (
        ParallelContext(2, 0, 1, 0, object(), None)
        if axis == "tp"
        else ParallelContext(1, 0, 2, 0, None, object())
    )
    with pytest.raises(ValueError, match="single-device RowParallelLinear"):
        allocate_metal_storage_(
            model,
            "cpu",
            quantization_backend="dequantize",
            quantization_policy="metal_interior_fc1_qkv_fc2_v1",
        )
    assert all(p.device.type == "meta" for p in model.parameters())


@pytest.mark.parametrize("policy", [INTERIOR_FC1_QKV_POLICY, *METAL_EXPANDED_WEIGHT_POLICIES])
def test_runtime_propagates_explicit_policy_and_receipt(monkeypatch, tmp_path, policy):
    import taomate_h3.model.pipeline as pipeline
    import taomate_h3.runtime.mps as mps

    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(mps, "require_mps", lambda: None)
    monkeypatch.setattr(pipeline, "MiniMaxH3NativePipeline", fake)
    monkeypatch.setenv("TAOMATE_MPS_WEIGHT_QUANTIZATION", "dequantize")
    monkeypatch.setenv("TAOMATE_MPS_WEIGHT_COVERAGE", policy)
    monkeypatch.setenv("TAOMATE_MPS_CPU_ADALN", "0")
    runtime = mps.MPSRuntime(model_root=tmp_path, adapter_root=tmp_path, output_root=tmp_path)
    assert captured["transformer_metal_quantization_policy"] == policy
    assert (
        runtime.model_resolution["metal_workspace_options"]["weight_quantization_policy"] == policy
    )
    assert parse_metal_weight_policy(policy) == policy
