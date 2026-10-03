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


@pytest.mark.parametrize("backend", ["dequantize", "native_int8pack"])
@torch.no_grad()
def test_mps_integrated_safetensors_loader_and_official_lora(model, tmp_path, backend):
    """Actual allocate/load/LoRA/dispatch on MPS; independent CPU FP64 oracle."""
    if not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    import gc
    import json

    from safetensors.torch import save_file

    from taomate_h3.inference.lora_checkpoint import (
        apply_h3_lora_checkpoint,
        canonical_h3_lora_targets,
        materialize_h3_lora_bf16_buffers_,
    )
    from taomate_h3.model.weight_loading import load_minimax_h3_dit_weights

    generator = torch.Generator().manual_seed(831)
    state = {
        name: (torch.randn(t.shape, generator=generator) * 0.025).to(t.dtype)
        for name, t in model.state_dict().items()
    }
    assert len(state) == 535
    # Two real indexed shards exercise the production inventory/streaming reader.
    root = tmp_path / "checkpoint"
    root.mkdir()
    names = list(state)
    shards = (names[::2], names[1::2])
    weight_map = {}
    for index, keys in enumerate(shards):
        filename = f"model-{index}.safetensors"
        save_file({key: state[key] for key in keys}, str(root / filename))
        weight_map.update({key: filename for key in keys})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    actual = (
        type(model)
        .allocate(
            "mps",
            parallel_context=model.parallel_context,
            metal_cpu_adaln=True,
            metal_quantization_backend=backend,
        )
        .eval()
    )
    before = {name: id(p) for name, p in actual.named_parameters()}
    receipt = load_minimax_h3_dit_weights(actual, root)
    assert receipt["tensor_count"] == 535 and receipt["shard_count"] == 2
    assert receipt["metal_streamed_w8a16_linear_count"] == 90
    assert before == {name: id(p) for name, p in actual.named_parameters()}
    selected = set(actual._metal_allocation_receipt["quantized_linear_names"])
    for name, tensor in actual.state_dict().items():
        owner = name.rsplit(".", 1)[0]
        if owner in selected:
            assert tensor.dtype == torch.int8 and tensor.device.type == "mps"
        else:
            # One head makes checkpoint QKV grouping the identity permutation.
            assert torch.equal(tensor.cpu(), state[name])
            assert tensor.device.type == ("cpu" if owner.endswith("adaln_proj.linear") else "mps")
    assert sum(p.numel() * p.element_size() for p in actual.parameters()) < 128 * 1024**2
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "config.json").write_text(json.dumps({"rank": 2, "alpha": 3}))
    adapter_state = {}
    for name in canonical_h3_lora_targets():
        owner = actual.get_submodule(name)
        adapter_state[name + ".lora_a"] = (
            torch.randn(2, owner.input_size, generator=generator) * 0.01
        )
        adapter_state[name + ".lora_b"] = (
            torch.randn(owner.output_size, 2, generator=generator) * 0.01
        )
    save_file(adapter_state, str(adapter / "adapter_model.safetensors"))
    assert apply_h3_lora_checkpoint(actual, adapter)["target_count"] == 208
    materialize_h3_lora_bf16_buffers_(actual)
    # Full-key QKV and FC1 both go through unmocked original module dispatch.
    for target in ("blocks.2.attn.qkv_proj", "blocks.46.mlp.fc1"):
        wrapper = actual.get_submodule(target)
        x = torch.randn(3, wrapper.base.input_size, generator=generator).bfloat16()
        output = wrapper(x.to("mps")).cpu().float()
        w = state[target + ".weight"].double()
        scales = (w.abs().amax(1, keepdim=True) / 127).float()
        q = (w / scales.double()).round().clamp(-127, 127)
        if backend == "native_int8pack":
            reference = ((x.double() @ q.T) * scales.bfloat16().double().flatten()).bfloat16()
        else:
            restored = (q * scales.double()).bfloat16()
            reference = (x.double() @ restored.double().T).bfloat16()
        a = adapter_state[target + ".lora_a"].bfloat16()
        b = adapter_state[target + ".lora_b"].bfloat16()
        update = ((x.double() @ a.double().T).bfloat16().double() @ b.double().T).bfloat16()
        reference = (reference + (update * 1.5).bfloat16()).float()
        relative = torch.linalg.vector_norm(output - reference) / torch.linalg.vector_norm(
            reference
        )
        assert torch.isfinite(output).all() and relative < 0.012
        assert (output - reference).abs().max() < 0.02
    del actual, wrapper
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()


@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.no_grad()
def test_boundary_cache_purge_real_quantized_lora_dispatch(model, monkeypatch, capsys, device):
    import json

    from taomate_h3.inference import lora_checkpoint as lora

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    allocate_metal_storage_(model, device, quantization_backend="dequantize", temporary_bytes=768)
    owner = model.blocks[2].mlp.fc1
    source = torch.linspace(-1, 1, owner.weight.numel()).reshape(owner.weight.shape).bfloat16()
    load_metal_stream_weight_(owner, source, maximum_temporary_bytes=768)
    target = "blocks.2.mlp.fc1"
    g = torch.Generator().manual_seed(813)
    state = {
        target + ".lora_a": torch.randn(128, owner.input_size, generator=g) * 0.01,
        target + ".lora_b": torch.randn(owner.output_size, 128, generator=g) * 0.01,
    }
    monkeypatch.setattr(lora, "canonical_h3_lora_targets", lambda: (target,))
    monkeypatch.setattr(
        lora, "load_h3_lora_checkpoint", lambda _: lora.LoadedH3LoRA(128, 128, state, 2)
    )
    lora.apply_h3_lora_checkpoint(model, "synthetic")
    lora.materialize_h3_lora_bf16_buffers_(model)
    projection = model.blocks[2].mlp.fc1
    inputs = (
        torch.linspace(-1, 1, owner.input_size * 26)
        .reshape(26, owner.input_size)
        .bfloat16()
        .to(device)[::2]
    )
    before = inputs.clone()
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "1")
    monkeypatch.delenv("TAOMATE_MPS_MEMORY_TRACE", raising=False)
    baseline = projection(inputs)
    monkeypatch.setenv("TAOMATE_MPS_MEMORY_TRACE", "1")
    monkeypatch.setenv("TAOMATE_MPS_EMPTY_CACHE_BOUNDARIES", "1")
    candidate = projection(inputs)
    assert torch.equal(baseline.view(torch.int16), candidate.view(torch.int16))
    assert torch.equal(before, inputs)
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    if device == "cpu":
        assert events == []
    else:
        assert [e["point"] for e in events] == [
            "before_weight_linear_output",
            "before_lora_update",
            "after_lora_update",
        ]
        assert [e["cache_purge_applied"] for e in events] == [True, True, False]
        assert events[0]["requested_output_bytes"] == 13 * owner.output_size * 2
        for event in events[:2]:
            assert event["pre_purge_current_allocated_bytes"] == event["current_allocated_bytes"]
            assert event["current_reserved_bytes"] <= event["pre_purge_current_reserved_bytes"]
