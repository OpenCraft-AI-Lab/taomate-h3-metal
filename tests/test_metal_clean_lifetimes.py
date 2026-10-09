# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Temporary attention copies must not survive into clean cache staging."""

import weakref

import pytest
import torch

from taomate_h3.streaming import attention_hook as hooks
from taomate_h3.streaming.cache import KVContract
from taomate_h3.streaming.metal_kv import QuantizedCleanAVKVCache


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize(
    "points,mask_values",
    [
        ((0, 7, 9), [False, False, True, True, True, True, True, False, False]),
        ((0, 2, 7, 9), [False, False, True, False, True, False, True, False, False]),
        ((0, 2, 9), [False, False, False, True, False, True, True, False, True]),
    ],
)
@pytest.mark.parametrize("row_views", [False, True])
@torch.no_grad()
def test_stage_observes_released_copies_and_identical_results(
    monkeypatch, device, points, mask_values, row_views
):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Exclusive MPS window required")
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "1")
    generator = torch.Generator().manual_seed(723)
    q, k, v = [torch.randn(9, 1, 128, generator=generator).bfloat16().to(device) for _ in range(3)]
    snapshot = [t.clone() for t in (q, k, v)]
    mask = torch.tensor(mask_values, device=device)
    tags = torch.ones(9, dtype=torch.long, device=device)
    indices = torch.nonzero(mask).flatten()
    tags[indices] = 0
    tags[indices[0]] = 2
    outputs, states = [], []
    actual_gate = hooks._release_clean_temporaries_enabled
    row_gate = hooks._row_views_enabled
    for enabled in (False, True):
        monkeypatch.setenv("TAOMATE_MPS_ATTENTION_ROW_VIEWS", "1" if row_views and enabled else "0")
        monkeypatch.setattr(
            hooks,
            "_row_views_enabled",
            row_gate if device == "mps" else lambda _: bool(row_views and enabled),
        )
        refs = []
        cache = QuantizedCleanAVKVCache(
            KVContract(local_heads=1, device_type=device), allocation="packed8"
        )
        cache.max_joined_bytes = 1024**2
        hook = hooks.H3StreamingAttentionHook(cache, backend="torch_sdpa", query_chunk_size=3)
        cache.begin_clean_commit(0)
        hook.activate(hooks.HookMode.CLEAN_COMMIT)
        monkeypatch.setattr(
            hooks,
            "_release_clean_temporaries_enabled",
            (lambda *args: enabled) if device == "cpu" or not enabled else actual_gate,
        )
        original_attention = hook._attention

        def attention(query, parts, **kwargs):
            result = original_attention(query, parts, **kwargs)
            refs.append(weakref.ref(result))
            return result

        monkeypatch.setattr(hook, "_attention", attention)
        original_history = cache.attention_with_history

        def history(name, query, condition, current, **kwargs):
            refs.extend(
                weakref.ref(t)
                for t in (query, condition.key, condition.value, current.key, current.value)
            )
            result = original_history(name, query, condition, current, **kwargs)
            refs.append(weakref.ref(result))
            return result

        monkeypatch.setattr(cache, "attention_with_history", history)
        original_stage = cache.stage
        observed = []

        def stage(name, key, value, token_tags, commit_mask):
            observed.append(all(ref() is None for ref in refs))
            assert key is k and value is v
            original_stage(name, key, value, token_tags, commit_mask)

        monkeypatch.setattr(cache, "stage", stage)
        outputs.append(
            hook(
                attention=type("Attention", (), {"softmax_scale": 128**-0.5})(),
                layer_name="blocks.0.attn",
                query=q,
                key=k,
                value=v,
                token_tags=tags,
                commit_mask=mask,
                cu_seqlens_host=points,
            )
        )
        assert observed == [enabled]
        pair = cache._staged["blocks.0.attn"]
        states.append(
            tuple(
                getattr(pair, name).clone() for name in ("key", "value", "key_scale", "value_scale")
            )
        )
        hook.deactivate()
        cache.rollback()
    assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
    assert all(torch.equal(a, b) for a, b in zip(*states))
    assert all(torch.equal(a, b) for a, b in zip(snapshot, (q, k, v)))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_cleanup_gate(monkeypatch, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Exclusive MPS window required")
    x = torch.ones(2, 1, 128, dtype=torch.bfloat16, device=device)
    monkeypatch.delenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", raising=False)
    with torch.no_grad():
        assert not hooks._release_clean_temporaries_enabled(x, x, x)
    monkeypatch.setenv("TAOMATE_MPS_INPLACE_ACTIVATIONS", "1")
    assert not hooks._release_clean_temporaries_enabled(x, x, x)
    with torch.no_grad():
        assert hooks._release_clean_temporaries_enabled(x, x, x) == (device == "mps")
        x.requires_grad_(True)
        assert not hooks._release_clean_temporaries_enabled(x, x, x)
