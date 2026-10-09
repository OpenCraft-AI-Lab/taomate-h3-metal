# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Storage grouping must leave quantization, publication and attention unchanged."""

import gc
import weakref

import pytest
import torch

from taomate_h3.streaming.cache import AVKV, MAIN_LAYER_NAMES, KVContract
from taomate_h3.streaming.metal_kv import QuantizedCleanAVKVCache, parse_kv_allocation


def fixture(device):
    x = (
        torch.arange(11 * 128)
        .reshape(11, 1, 128)
        .remainder(71)
        .float()
        .div(13)
        .bfloat16()
        .to(device)
    )
    tags = torch.tensor([1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 3], device=device)
    mask = (tags == 0) | (tags == 2)
    return x, tags, mask


@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.no_grad()
def test_packed_bitwise_quantization_and_full_attention(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS window required")
    caches = [
        QuantizedCleanAVKVCache(
            KVContract(local_heads=1, device_type=device), tile_tokens=4, allocation=allocation
        )
        for allocation in ("per_layer", "packed8")
    ]
    x, tags, mask = fixture(device)
    for cache in caches:
        cache.begin_clean_commit(0)
        # Out-of-order staging also exercises lazy final short group.
        for name in reversed(MAIN_LAYER_NAMES):
            cache.stage(name, x, -x, tags, mask)
        cache.commit()
    for name in MAIN_LAYER_NAMES:
        pairs = [c._segments[0].layers[name] for c in caches]
        for attr in ("key", "value", "key_scale", "value_scale"):
            assert torch.equal(getattr(pairs[0], attr), getattr(pairs[1], attr))
    outputs = [
        c.attention_with_history(
            MAIN_LAYER_NAMES[8],
            x[:3],
            AVKV(x[:1], -x[:1]),
            AVKV(x[1:4], -x[1:4]),
            scale=128**-0.5,
            query_chunk_size=2,
            max_joined_bytes=1024**2,
        )
        for c in caches
    ]
    assert torch.equal(*outputs)
    assert (
        caches[0].receipt()["physical_history_payload_bytes"]
        == caches[1].receipt()["physical_history_payload_bytes"]
    )
    pairs = caches[1]._segments[0].layers
    assert (
        pairs[MAIN_LAYER_NAMES[0]].key.untyped_storage().data_ptr()
        == pairs[MAIN_LAYER_NAMES[7]].key.untyped_storage().data_ptr()
    )
    assert (
        pairs[MAIN_LAYER_NAMES[8]].key.untyped_storage().data_ptr()
        != pairs[MAIN_LAYER_NAMES[7]].key.untyped_storage().data_ptr()
    )
    assert pairs[MAIN_LAYER_NAMES[49]].key.untyped_storage().nbytes() == 2 * 9 * 128


@pytest.mark.parametrize("failure_layer", [0, 7, 49])
def test_failed_quantization_preserves_history_and_releases_owners(monkeypatch, failure_layer):
    cache = QuantizedCleanAVKVCache(
        KVContract(local_heads=1, device_type="cpu"), allocation="packed8"
    )
    x, tags, mask = fixture("cpu")
    cache.begin_clean_commit(0)
    for name in MAIN_LAYER_NAMES:
        cache.stage(name, x, x, tags, mask)
    cache.commit()
    old = cache._segments
    cache.begin_clean_commit(1)
    for name in MAIN_LAYER_NAMES[:failure_layer]:
        cache.stage(name, x, x, tags, mask)

    def fail(*args, **kwargs):
        raise RuntimeError("injected quantization failure")

    monkeypatch.setattr(cache, "_quantize_selected", fail)
    with pytest.raises(RuntimeError, match="injected"):
        cache.stage(MAIN_LAYER_NAMES[failure_layer], x, x, tags, mask)
    assert cache._segments is old and cache.committed_blocks == 1
    receipt = cache.receipt()
    assert receipt["staged_backing_bytes"] > receipt["staged_payload_bytes"]
    refs = [weakref.ref(t) for group in cache._packed_groups.values() for t in group]
    refs.append(weakref.ref(cache._packed_scales))
    cache.rollback()
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert cache.receipt()["staged_backing_bytes"] == 0
    assert cache._segments is old


def test_preallocation_receipt_counts_shared_scales_once():
    cache = QuantizedCleanAVKVCache(
        KVContract(local_heads=1, device_type="cpu"), allocation="packed8"
    )
    x, tags, mask = fixture("cpu")
    cache.begin_clean_commit(0)
    cache.stage(MAIN_LAYER_NAMES[0], x, x, tags, mask)
    assert cache.receipt()["staged_backing_bytes"] == 2 * 8 * 9 * 128 + 2 * 50 * 9 * 4
    assert cache.receipt()["staged_payload_bytes"] == 2 * 9 * (128 + 4)
    cache.stage(MAIN_LAYER_NAMES[7], x, x, tags, mask)
    assert cache.receipt()["staged_backing_bytes"] == 2 * 8 * 9 * 128 + 2 * 50 * 9 * 4


@pytest.mark.parametrize("value", ["", "packed", "packed16", "PACKED8"])
def test_invalid_allocation(value):
    with pytest.raises(ValueError):
        parse_kv_allocation(value)


def test_commit_sync_failure_is_atomic_and_clear_releases_storage(monkeypatch):
    cache = QuantizedCleanAVKVCache(
        KVContract(local_heads=1, device_type="cpu"), allocation="packed8"
    )
    x, tags, mask = fixture("cpu")
    cache.begin_clean_commit(0)
    for name in MAIN_LAYER_NAMES:
        cache.stage(name, x, x, tags, mask)
    refs = [weakref.ref(t) for group in cache._packed_groups.values() for t in group]
    refs.append(weakref.ref(cache._packed_scales))

    def fail():
        raise RuntimeError("commit synchronization")

    monkeypatch.setattr(cache, "_sync", fail)
    with pytest.raises(RuntimeError, match="commit synchronization"):
        cache.commit()
    assert cache.committed_blocks == cache.history_tokens == 0
    assert cache.clean_commit_active
    monkeypatch.setattr(cache, "_sync", lambda: None)
    cache.commit()
    gc.collect()
    assert all(ref() is not None for ref in refs)
    assert not cache._packed_groups and cache._packed_scales is None
    cache.clear()
    gc.collect()
    assert all(ref() is None for ref in refs)
