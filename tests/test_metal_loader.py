"""CPU/meta tests of final storage, streamed quantization and official LoRA."""

import pytest
import torch

from taomate_h3.distributed import ParallelContext
from taomate_h3.distributed.linear import MergedColumnParallelLinear
from taomate_h3.inference.metal_loader import (
    allocate_metal_storage_,
    load_metal_stream_weight_,
    metal_linear,
)


@pytest.fixture
def model(monkeypatch):
    import taomate_h3.model.dit as dit
    from taomate_h3.model.architecture import MiniMaxH3Architecture

    arch = MiniMaxH3Architecture(
        num_layers=50,
        token_refiner_num_layers=2,
        hidden_size=128,
        num_attention_heads=1,
        ffn_hidden_size=128,
        text_dim=16,
        timestep_input_dim=16,
        time_embed_hidden_size=128,
        time_embed_dim=64,
        adaln_out_features=18 * 128,
        final_adaln_out_features=2 * 128,
    )
    monkeypatch.setattr(dit, "MiniMaxH3Architecture", lambda: arch)
    with torch.device("meta"):
        return dit.MiniMaxH3DiT(parallel_context=ParallelContext(1, 0, 1, 0, None, None))


@pytest.mark.parametrize("backend", ["none", "dequantize", "native_int8pack"])
def test_original_535_keys_dtype_and_90_targets(model, backend):
    original = {name: (p.shape, p.dtype) for name, p in model.state_dict().items()}
    assert len(original) == 535
    receipt = allocate_metal_storage_(model, device="cpu", quantization_backend=backend)
    assert set(model.state_dict()) == set(original)
    names = receipt["quantized_linear_names"]
    assert len(names) == (0 if backend == "none" else 90)
    adaln = 0
    for name, p in model.state_dict().items():
        owner = name.rsplit(".", 1)[0]
        expected = torch.int8 if owner in names and name.endswith(".weight") else original[name][1]
        assert p.shape == original[name][0] and p.dtype == expected and p.device.type == "cpu"
        if owner.endswith("adaln_proj.linear") and name.endswith(".weight"):
            adaln += 1
    assert adaln == 51
    if backend != "none":
        assert all(2 <= int(name.split(".")[1]) <= 46 for name in names)
        for name in names:
            owner = model.get_submodule(name)
            assert type(owner) is MergedColumnParallelLinear
            assert not owner.weight.requires_grad and owner._metal_stream_load_pending
            assert owner._metal_weight_scale.dtype == torch.float32


def test_stream_formula_identity_forward_and_pending(model):
    allocate_metal_storage_(model, "cpu", quantization_backend="dequantize", temporary_bytes=256)
    owner = model.blocks[2].mlp.fc1
    identity = owner.weight
    source = (
        torch.arange(owner.weight.numel())
        .reshape(owner.weight.shape)
        .remainder(255)
        .sub(127)
        .bfloat16()
    )
    x = torch.ones(3, owner.input_size, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="incomplete"):
        metal_linear(x, owner)
    load_metal_stream_weight_(owner, source, maximum_temporary_bytes=256)
    expected_scale = (source.abs().amax(1, keepdim=True).float() / 127).clamp_min(1e-12)
    expected_q = (source / expected_scale).round().clamp(-127, 127).to(torch.int8)
    assert owner.weight is identity and not owner._metal_stream_load_pending
    assert torch.equal(owner.weight, expected_q)
    assert torch.equal(owner._metal_weight_scale, expected_scale)
    expected = torch.nn.functional.linear(x, (expected_q.float() * expected_scale).bfloat16())
    assert torch.equal(metal_linear(x, owner), expected)
    with pytest.raises(RuntimeError):
        load_metal_stream_weight_(owner, source)


def test_failed_load_pending_retry(model):
    allocate_metal_storage_(model, "cpu", quantization_backend="dequantize")
    owner = model.blocks[2].mlp.fc1
    source = torch.ones(owner.weight.shape, dtype=torch.bfloat16)
    source[-1, -1] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        load_metal_stream_weight_(owner, source, maximum_temporary_bytes=256)
    assert owner._metal_stream_load_pending and not owner._metal_stream_loaded
    source[-1, -1] = 1
    load_metal_stream_weight_(owner, source)
    assert owner._metal_stream_loaded


def test_actual_official_lora_installation(model, monkeypatch):
    import taomate_h3.distributed.linear as linear
    import taomate_h3.inference.lora_checkpoint as lora

    allocate_metal_storage_(model, "cpu", quantization_backend="dequantize")
    state = {}
    for name in lora.canonical_h3_lora_targets():
        owner = model.get_submodule(name)
        state[name + ".lora_a"] = torch.ones(2, owner.input_size) / 128
        state[name + ".lora_b"] = torch.ones(owner.output_size, 2) / 128
    loaded = lora.LoadedH3LoRA(rank=2, alpha=2, state=state, loaded_tensor_count=416)
    monkeypatch.setattr(lora, "load_h3_lora_checkpoint", lambda _: loaded)
    receipt = lora.apply_h3_lora_checkpoint(model, "unused")
    assert receipt["target_count"] == 208
    lora.materialize_h3_lora_bf16_buffers_(model)
    wrapper = model.blocks[2].attn.qkv_proj
    assert type(wrapper.base) is MergedColumnParallelLinear
    assert wrapper.lora_a.dtype == wrapper.lora_b.dtype == torch.bfloat16
    load_metal_stream_weight_(
        wrapper.base, torch.ones(wrapper.base.weight.shape, dtype=torch.bfloat16)
    )
    old = linear._module_linear
    monkeypatch.setattr(
        linear,
        "_module_linear",
        lambda module, inputs, bias: (
            metal_linear(inputs, module)
            if hasattr(module, "_metal_weight_scale")
            else old(module, inputs, bias)
        ),
    )
    x = torch.ones(2, wrapper.base.input_size, dtype=torch.bfloat16)
    expected = metal_linear(x, wrapper.base) + torch.nn.functional.linear(
        torch.nn.functional.linear(x, wrapper.lora_a), wrapper.lora_b
    )
    assert torch.equal(wrapper(x), expected)


def test_alias_preserved_and_no_materialized_model_accepted():
    model = torch.nn.Module()
    with torch.device("meta"):
        model.a = torch.nn.Parameter(torch.empty(3))
        model.b = model.a
    allocate_metal_storage_(model, "cpu", cpu_adaln=False)
    assert model.a is model.b
    with pytest.raises(ValueError, match="meta"):
        allocate_metal_storage_(model, "cpu")
