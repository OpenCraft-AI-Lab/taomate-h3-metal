# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Projection owners die before downstream work without changing arithmetic."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from taomate_h3.distributed import ParallelContext
from taomate_h3.inference import metal_activations as activation
from taomate_h3.model import layers


def make_modules(monkeypatch, device):
    from taomate_h3.inference import lora_checkpoint as lora

    arch = SimpleNamespace(
        hidden_size=128,
        ffn_hidden_size=128,
        num_attention_heads=1,
        attention_head_dim=128,
        qk_norm_eps=1e-6,
    )
    context = ParallelContext(1, 0, 1, 0, None, None)
    model = torch.nn.Module()
    model.attn = layers.MiniMaxH3Attention(arch, context=context)
    model.mlp = layers.MiniMaxH3MLP(arch, context=context)
    generator = torch.Generator().manual_seed(814)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=generator) * 0.02)
    targets = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    state = {}
    for name in targets:
        module = model.get_submodule(name)
        state[name + ".lora_a"] = torch.randn(128, module.input_size, generator=generator) * 0.01
        state[name + ".lora_b"] = torch.randn(module.output_size, 128, generator=generator) * 0.01
    monkeypatch.setattr(lora, "canonical_h3_lora_targets", lambda: targets)
    monkeypatch.setattr(
        lora, "load_h3_lora_checkpoint", lambda _: lora.LoadedH3LoRA(128, 128, state, len(state))
    )
    model.to(device).eval()
    lora.apply_h3_lora_checkpoint(model, "synthetic")
    lora.materialize_h3_lora_bf16_buffers_(model)
    return model


def test_release_predicate_default_grad_and_alias_fallback(monkeypatch):
    x = torch.ones(3, 4, dtype=torch.bfloat16)
    projected = x.clone()
    replacement = x.clone()
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "1")
    with torch.no_grad():
        assert not activation.can_release_projection(projected, x, replacement)  # CPU default
        monkeypatch.setattr(activation, "_eligible", lambda *args: True)
        assert activation.can_release_projection(projected, x, replacement)
        assert not activation.can_release_projection(projected, x, projected[:1])
        assert not activation.can_release_projection(x, x, replacement)
        assert not activation.can_release_projection(projected, x)
        replacement.requires_grad_(True)
        assert not activation.can_release_projection(projected, x, replacement)
    # Actual eligibility must also reject grad-enabled execution, even if no leaf requires grad.
    monkeypatch.undo()
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "1")
    assert not activation.can_release_projection(projected, x, replacement)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("rows", [1, 13])
@torch.no_grad()
def test_full_lora_mlp_owner_and_bitwise(monkeypatch, device, rows):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Exclusive MPS window required")
    model = make_modules(monkeypatch, device)
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "1")
    if device == "cpu":
        # Only the device gate is simulated; production operators/modules run unchanged.
        monkeypatch.setattr(activation, "_eligible", lambda *args: True)
    x = torch.linspace(-2, 2, rows * 128).reshape(rows, 128).bfloat16().to(device)
    before = x.clone()
    predicate = activation.can_release_projection
    monkeypatch.setattr(activation, "can_release_projection", lambda *args: False)
    expected = model.mlp(x)
    owners = []
    decisions = []

    def capture(projected, inputs, *consumers):
        owners.append(weakref.ref(projected))
        decision = predicate(projected, inputs, *consumers)
        decisions.append(decision)
        return decision

    monkeypatch.setattr(activation, "can_release_projection", capture)
    observed = []
    handle = model.mlp.fc2.register_forward_pre_hook(
        lambda *_: observed.append(owners[-1]() is None)
    )
    actual = model.mlp(x)
    handle.remove()
    assert observed == decisions
    assert decisions == [rows > 1]  # single-row contiguous() aliases original projection
    assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
    assert torch.equal(x, before)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("trim_clean", [False, True])
@pytest.mark.parametrize("row_views", [False, True])
@torch.no_grad()
def test_full_lora_attention_owner_output_and_clean_quantized_kv(
    monkeypatch, device, trim_clean, row_views
):
    from taomate_h3.streaming import attention_hook as hooks
    from taomate_h3.streaming.attention_hook import H3StreamingAttentionHook, HookMode
    from taomate_h3.streaming.cache import KVContract
    from taomate_h3.streaming.metal_kv import QuantizedCleanAVKVCache

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Exclusive MPS window required")
    model = make_modules(monkeypatch, device)
    model.attn.layer_name = "blocks.0"
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "1")
    if device == "cpu":
        monkeypatch.setattr(activation, "_eligible", lambda *args: True)
    # Noncontiguous source input and a separate padding document.
    x = torch.linspace(-1, 1, 14 * 256).reshape(14, 256).bfloat16().to(device)[:, ::2]
    before = x.clone()
    tags = torch.tensor([1, 1, 2, 2] + [0] * 9 + [-1], device=device)
    mask = (tags == 0) | (tags == 2)
    positions = torch.arange(14, device=device)
    rope = (torch.linspace(-0.7, 0.9, 14 * 128).reshape(14, 128).bfloat16().to(device), positions)
    original = activation.can_release_projection
    clean_gate = hooks._release_clean_temporaries_enabled
    row_gate = hooks._row_views_enabled
    refs = []

    def predicate(projected, inputs, *consumers):
        refs.append(weakref.ref(projected))
        assert original(projected, inputs, *consumers)
        return True

    outputs, pairs = [], []
    for release in (False, True):
        monkeypatch.setenv("TAOMATE_MPS_ATTENTION_ROW_VIEWS", "1" if row_views and release else "0")
        monkeypatch.setattr(
            hooks,
            "_row_views_enabled",
            row_gate if device == "mps" else lambda _: bool(row_views and release),
        )
        monkeypatch.setattr(
            hooks,
            "_release_clean_temporaries_enabled",
            clean_gate
            if device == "mps" and trim_clean and release
            else lambda *args: bool(device == "cpu" and trim_clean and release),
        )
        cache = QuantizedCleanAVKVCache(
            KVContract(local_heads=1, device_type=device), allocation="packed8", tile_tokens=4
        )
        cache.max_joined_bytes = 1024**2
        hook = H3StreamingAttentionHook(cache, backend="torch_sdpa", query_chunk_size=3)
        cache.begin_clean_commit(0)
        hook.activate(HookMode.CLEAN_COMMIT)
        monkeypatch.setattr(
            activation, "can_release_projection", predicate if release else lambda *a: False
        )
        observed = []
        handle = model.attn.out_proj.register_forward_pre_hook(
            lambda *_: observed.append(refs[-1]() is None) if release else None
        )
        layers.set_native_stream_attention_hook(hook)
        try:
            with layers.native_stream_attention_metadata(tags, mask):
                outputs.append(
                    model.attn(
                        x,
                        rope_cache=rope,
                        cu_seqlens_host=(0, 13, 14),
                        max_seqlen=13,
                        ulysses_active=True,
                    )
                )
            pair = cache._staged["blocks.0.attn"]
            pairs.append(
                tuple(
                    getattr(pair, name).clone()
                    for name in ("key", "value", "key_scale", "value_scale")
                )
            )
        finally:
            handle.remove()
            layers.set_native_stream_attention_hook(None)
            hook.deactivate()
            cache.rollback()
        if release:
            assert observed == [True]
    assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
    assert all(torch.equal(a, b) for a, b in zip(*pairs))
    assert torch.equal(x, before)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_production_release_gate_requires_opt_in_no_grad(monkeypatch, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Exclusive MPS window required")
    x = torch.ones(3, 8, dtype=torch.bfloat16, device=device)
    projected, replacement = x.clone(), x.clone()
    monkeypatch.delenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", raising=False)
    with torch.no_grad():
        assert not activation.can_release_projection(projected, x, replacement)
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "1")
    assert not activation.can_release_projection(projected, x, replacement)
    with torch.no_grad():
        assert activation.can_release_projection(projected, x, replacement) == (device == "mps")
        projected.requires_grad_(True)
        assert not activation.can_release_projection(projected, x, replacement)
