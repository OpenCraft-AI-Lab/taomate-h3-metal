"""Scalar-only allocation diagnostics; no model-frame or tensor ownership escapes."""

import gc
import json
import weakref

import pytest
import torch

from taomate_h3.runtime import mps_memory as memory
from taomate_h3.streaming import metal_kv
from taomate_h3.streaming.cache import AVKV, MAIN_LAYER_NAMES, KVContract


def test_disabled_trace_never_queries_or_synchronizes(monkeypatch):
    monkeypatch.delenv("TAOMATE_MPS_MEMORY_TRACE", raising=False)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: pytest.fail("unexpected sync"))
    monkeypatch.setattr(memory, "mps_memory_receipt", lambda: pytest.fail("unexpected receipt"))
    memory.trace_mps_memory("disabled", ignored=torch.empty(1))
    monkeypatch.setenv("TAOMATE_MPS_MEMORY_TRACE", "1")
    assert memory.memory_trace_enabled("mps")
    assert not memory.memory_trace_enabled("cpu")
    assert not memory.memory_trace_enabled("cuda")


def test_trace_scalar_validation_and_counter_mapping(monkeypatch, capsys):
    monkeypatch.setenv("TAOMATE_MPS_MEMORY_TRACE", "1")
    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(
        memory,
        "mps_memory_receipt",
        lambda: {
            "current_allocated_bytes": 11,
            "current_reserved_bytes": 37,
            "driver_allocated_bytes": 41,
            "peak_allocated_bytes": None,
        },
    )
    memory.trace_mps_memory("before_joined_key", requested_buffer_bytes=23)
    result = json.loads(capsys.readouterr().err)
    assert calls == ["sync"]
    assert result["current_allocated_bytes"] == 11
    assert result["current_reserved_bytes"] == 37
    assert result["driver_allocated_bytes"] == 41
    assert result["peak_allocated_bytes"] is None
    assert result["requested_buffer_bytes"] == 23
    with pytest.raises(TypeError, match="scalar"):
        memory.trace_mps_memory("reject", tensor=torch.empty(1))
    assert calls == ["sync"]


@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.no_grad()
def test_cache_trace_exact_outputs_backing_and_ownership(monkeypatch, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    x = torch.arange(7 * 128).reshape(7, 1, 128).remainder(19).bfloat16().to(device)
    tags = torch.tensor([1, 2, 2, 0, 0, 0, 3], device=device)
    mask = torch.tensor([False, True, True, True, True, True, False], device=device)
    q = x[1:6].contiguous()
    samples = []
    baseline = None
    for enabled in (False, True):
        cache = metal_kv.QuantizedCleanAVKVCache(
            KVContract(local_heads=1, device_type=device), allocation="packed8"
        )
        monkeypatch.setattr(metal_kv, "memory_trace_enabled", lambda _: enabled)

        def record(point, **fields):
            assert all(type(v) in (int, str, bool, type(None)) for v in fields.values())
            if device == "mps":
                torch.mps.synchronize()
            samples.append((point, fields))

        monkeypatch.setattr(metal_kv, "trace_mps_memory", record)
        cache.begin_clean_commit(0)
        for name in MAIN_LAYER_NAMES:
            cache.stage(name, x, -x, tags, mask)
        cache.commit()
        output = cache.attention_with_history(
            MAIN_LAYER_NAMES[0],
            q,
            AVKV(x[:1], -x[:1]),
            AVKV(q, -q),
            scale=128**-0.5,
            query_chunk_size=3,
            max_joined_bytes=100000,
        )
        pair = cache._segments[0].layers[MAIN_LAYER_NAMES[0]]
        actual = [
            output.cpu(),
            pair.key.cpu().clone(),
            pair.value.cpu().clone(),
            pair.key_scale.cpu().clone(),
            pair.value_scale.cpu().clone(),
        ]
        if baseline is None:
            baseline = actual
        else:
            assert all(torch.equal(a, b) for a, b in zip(actual, baseline))
        ref = weakref.ref(pair.key)
        del pair
        cache.clear()
        gc.collect()
        assert ref() is None
    assert len(samples) == 103
    assert [p for p, _ in samples[-3:]] == [
        "before_joined_key",
        "after_joined_key",
        "after_joined_value",
    ]
    before, after = samples[:2]
    assert before[1]["staged_backing_bytes"] == 0
    assert after[1]["staged_backing_bytes"] > after[1]["staged_payload_bytes"]
    assert after[1]["staged_layers"] == 1
    assert samples[-3][1]["requested_buffer_bytes"] == 11 * 128 * 2


@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.no_grad()
def test_real_lora_trace_preserves_outputs_inputs_and_keys(monkeypatch, device):
    from test_metal_activations import make_lora

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    wrapper = make_lora(monkeypatch, device)
    x = torch.linspace(-1, 1, 7 * 128).reshape(7, 128).bfloat16().to(device)
    original = x.clone()
    keys = tuple(wrapper.state_dict())
    monkeypatch.setattr(memory, "memory_trace_enabled", lambda _: False)
    expected = wrapper(x)
    samples = []
    monkeypatch.setattr(memory, "memory_trace_enabled", lambda _: True)

    def record(point, **fields):
        assert all(type(v) in (int, str, bool, type(None)) for v in fields.values())
        if device == "mps":
            torch.mps.synchronize()
        samples.append((point, fields))

    monkeypatch.setattr(memory, "trace_mps_memory", record)
    actual = wrapper(x)
    assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
    assert torch.equal(x, original)
    assert keys == tuple(wrapper.state_dict())
    assert [p for p, _ in samples] == ["before_lora_update", "after_lora_update"]
    assert samples[0][1] == dict(
        layer="projection", rows=7, output_columns=256, requested_update_bytes=7 * 256 * 2, rank=128
    )


def test_purge_requires_trace_and_only_before_points(monkeypatch, capsys):
    calls = []
    monkeypatch.setenv("TAOMATE_MPS_EMPTY_CACHE_BOUNDARIES", "1")
    monkeypatch.delenv("TAOMATE_MPS_MEMORY_TRACE", raising=False)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(
        memory,
        "mps_memory_receipt",
        lambda: {
            "current_allocated_bytes": 11,
            "current_reserved_bytes": 17 if "empty" in calls else 37,
            "driver_allocated_bytes": 19 if "empty" in calls else 39,
        },
    )
    memory.trace_mps_memory("before_test")
    assert calls == []
    monkeypatch.setenv("TAOMATE_MPS_MEMORY_TRACE", "1")
    memory.trace_mps_memory("before_test")
    result = json.loads(capsys.readouterr().err)
    assert calls == ["sync", "empty"]
    assert result["cache_purge_applied"]
    assert result["pre_purge_current_allocated_bytes"] == result["current_allocated_bytes"] == 11
    assert (result["pre_purge_current_reserved_bytes"], result["current_reserved_bytes"]) == (
        37,
        17,
    )
    memory.trace_mps_memory("after_test")
    assert calls == ["sync", "empty", "sync"]
    assert not json.loads(capsys.readouterr().err)["cache_purge_applied"]


@torch.no_grad()
def test_mps_purge_keeps_complete_cache_and_attention_bits(monkeypatch, capsys):
    if not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    x = torch.arange(7 * 128).reshape(7, 1, 128).remainder(17).bfloat16().to("mps")
    tags = torch.tensor([1, 2, 2, 0, 0, 0, 3], device="mps")
    mask = torch.tensor([False, True, True, True, True, True, False], device="mps")
    baseline = None
    for enabled in (False, True):
        monkeypatch.setenv("TAOMATE_MPS_MEMORY_TRACE", "1" if enabled else "0")
        monkeypatch.setenv("TAOMATE_MPS_EMPTY_CACHE_BOUNDARIES", "1" if enabled else "0")
        cache = metal_kv.QuantizedCleanAVKVCache(
            KVContract(local_heads=1, device_type="mps"), allocation="packed8"
        )
        cache.begin_clean_commit(0)
        for name in MAIN_LAYER_NAMES:
            cache.stage(name, x, -x, tags, mask)
        cache.commit()
        output = cache.attention_with_history(
            MAIN_LAYER_NAMES[0],
            x[1:6],
            AVKV(x[:1], -x[:1]),
            AVKV(x[1:6], -x[1:6]),
            scale=128**-0.5,
            query_chunk_size=3,
            max_joined_bytes=100000,
        )
        values = [output.cpu()]
        for pair in cache._segments[0].layers.values():
            values.extend(t.cpu() for t in (pair.key, pair.value, pair.key_scale, pair.value_scale))
        if baseline is None:
            baseline = values
        else:
            assert all(torch.equal(a, b) for a, b in zip(values, baseline))
        cache.clear()
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert len(events) == 103
    assert sum(e["cache_purge_applied"] for e in events) == 51
    assert all(
        e["current_allocated_bytes"] == e["pre_purge_current_allocated_bytes"]
        for e in events
        if e["cache_purge_applied"]
    )
