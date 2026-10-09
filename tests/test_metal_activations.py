# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Exact BF16 boundaries and ownership for opt-in activation reuse."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from taomate_h3.distributed import ParallelContext
from taomate_h3.distributed.linear import MergedColumnParallelLinear
from taomate_h3.inference import metal_activations as activation


def available(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")


@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.no_grad()
def test_unit_add_primitive_bitwise_storage_and_input(device):
    available(device)
    g = torch.Generator().manual_seed(227)
    inputs = torch.randn(7, 128, generator=g).bfloat16().to(device)
    output = torch.randn(7, 256, generator=g).bfloat16().to(device)
    update = torch.randn(7, 256, generator=g).bfloat16().to(device)
    # Exact zero, extrema and BF16 subnormal cases alongside real random values.
    output[0, :4] = torch.tensor([0.0, -0.0, 1.0, torch.finfo(torch.bfloat16).tiny], device=device)
    update[0, :4] = torch.tensor(
        [-0.0, 0.0, -1.0, torch.finfo(torch.bfloat16).tiny / 128], device=device
    )
    before_input = inputs.clone()
    before_update = update.clone()
    expected = output + update * 1.0
    pointer = output.untyped_storage()._cdata
    actual = activation._unit_add_owned_(output, update, inputs)
    assert actual is output and actual.untyped_storage()._cdata == pointer
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(inputs, before_input) and torch.equal(update, before_update)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.no_grad()
def test_swiglu_preserves_bf16_rounding_and_contiguous_fc2_input(device):
    available(device)
    g = torch.Generator().manual_seed(431)
    inputs = torch.randn(9, 128, generator=g).bfloat16().to(device)
    projected = (torch.randn(9, 256, generator=g) * 4).bfloat16().to(device)
    before_input = inputs.clone()
    before_up = projected[:, 128:].clone()
    gate, up = projected.chunk(2, -1)
    expected = F.silu(gate) * up
    actual = activation._swiglu_owned_(projected, inputs)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert actual.is_contiguous()
    assert torch.equal(projected[:, 128:], before_up) and torch.equal(inputs, before_input)


def make_lora(monkeypatch, device):
    import taomate_h3.inference.lora_checkpoint as lora

    context = ParallelContext(1, 0, 1, 0, None, None)
    model = torch.nn.Module()
    model.projection = MergedColumnParallelLinear(
        128, [128, 128], bias=False, gather_output=False, dtype=torch.bfloat16, context=context
    )
    g = torch.Generator().manual_seed(112)
    with torch.no_grad():
        model.projection.weight.copy_(torch.randn(256, 128, generator=g) * 0.02)
    state = {
        "projection.lora_a": torch.randn(128, 128, generator=g) * 0.01,
        "projection.lora_b": torch.randn(256, 128, generator=g) * 0.01,
    }
    monkeypatch.setattr(lora, "canonical_h3_lora_targets", lambda: ("projection",))
    monkeypatch.setattr(
        lora, "load_h3_lora_checkpoint", lambda _: lora.LoadedH3LoRA(128, 128, state, 2)
    )
    model.to(device).eval()
    lora.apply_h3_lora_checkpoint(model, "synthetic")
    lora.materialize_h3_lora_bf16_buffers_(model)
    return model.projection


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("scale", [1.0, 0.5, 1.5])
@torch.no_grad()
def test_real_lora_wrapper_gating_bitwise_and_fresh_output(monkeypatch, device, scale):
    available(device)
    wrapper = make_lora(monkeypatch, device)
    wrapper.scale = scale
    inputs = torch.linspace(-1, 1, 7 * 128).reshape(7, 128).bfloat16().to(device)
    snapshot = inputs.clone()
    monkeypatch.delenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", raising=False)
    expected = wrapper(inputs)
    calls = []
    original = activation._unit_add_owned_

    def spy(output, update, value):
        calls.append(output.untyped_storage()._cdata)
        return original(output, update, value)

    monkeypatch.setattr(activation, "_unit_add_owned_", spy)
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "1")
    actual = wrapper(inputs)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(inputs, snapshot)
    if device == "mps" and scale == 1:
        assert calls == [actual.untyped_storage()._cdata]
    else:
        assert calls == []


@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.no_grad()
def test_original_mlp_complete_output_bitwise(monkeypatch, device):
    available(device)
    from taomate_h3.model.layers import MiniMaxH3MLP

    mlp = (
        MiniMaxH3MLP(
            SimpleNamespace(hidden_size=128, ffn_hidden_size=128),
            context=ParallelContext(1, 0, 1, 0, None, None),
        )
        .to(device)
        .eval()
    )
    g = torch.Generator().manual_seed(87)
    for p in mlp.parameters():
        p.copy_((torch.randn(p.shape, generator=g) * 0.02).to(device))
    inputs = torch.randn(13, 128, generator=g).bfloat16().to(device)
    snapshot = inputs.clone()
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "0")
    expected = mlp(inputs)
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "1")
    actual = mlp(inputs)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(inputs, snapshot)


def test_grad_and_unsafe_aliases_are_never_mutated(monkeypatch):
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "1")
    wrapper = make_lora(monkeypatch, "cpu")
    inputs = torch.ones(2, 128, dtype=torch.bfloat16, requires_grad=True)
    wrapper(inputs).float().sum().backward()
    assert inputs.grad is not None and inputs.detach().eq(1).all()
    with torch.no_grad():
        x = torch.ones(2, 128, dtype=torch.bfloat16)
        before = x.clone()
        with pytest.raises(ValueError):
            activation._unit_add_owned_(x, x.clone(), x)
        with pytest.raises(ValueError):
            activation._swiglu_owned_(x, x)
        with pytest.raises(ValueError):
            activation._swiglu_owned_(x[:, :127].contiguous(), x)
        assert torch.equal(x, before)
    with pytest.raises(ValueError):
        activation._unit_add_owned_(x, x.clone(), x.clone())


@torch.no_grad()
def test_unknown_module_and_observer_hooks_disable_reuse():
    context = ParallelContext(1, 0, 1, 0, None, None)
    known = MergedColumnParallelLinear(
        4, [4, 4], bias=False, gather_output=False, dtype=torch.bfloat16, context=context
    )
    assert activation._known_fresh_linear(known)
    handle = known.register_forward_hook(lambda *args: None)
    assert not activation._known_fresh_linear(known)
    handle.remove()
    assert not activation._known_fresh_linear(torch.nn.Identity())
