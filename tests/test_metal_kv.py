# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
import weakref

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from taomate_h3.streaming import metal_kv
from taomate_h3.streaming.cache import (
    AVKV,
    MAIN_LAYER_NAMES,
    CleanAVKVCache,
    KVContract,
)
from taomate_h3.streaming.metal_kv import QuantizedCleanAVKVCache


def fixture():
    x = (
        torch.arange(7 * 128, dtype=torch.float32)
        .reshape(7, 1, 128)
        .remainder(19)
        .bfloat16()
    )
    # text, stereo audio, video, padding: official packed row-order convention.
    tags = torch.tensor([1, 2, 2, 0, 0, 0, 3])
    mask = torch.tensor([False, True, True, True, True, True, False])
    return x, tags, mask


def add(cache):
    x, tags, mask = fixture()
    cache.begin_clean_commit(cache.committed_blocks)
    for name in MAIN_LAYER_NAMES:
        cache.stage(name, x, x, tags, mask)
    cache.commit()
    cache.retain_sink_and_recent_commits()


@pytest.mark.parametrize("device", ("cpu", "mps"))
@pytest.mark.parametrize("allocation", ("per_layer", "packed8"))
@torch.no_grad()
def test_52_commits_match_upstream_retention_and_reset(device, allocation):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    contract = KVContract(local_heads=1, device_type=device)
    cache = QuantizedCleanAVKVCache(contract, allocation=allocation)
    original = CleanAVKVCache(contract)
    _, tags, mask = fixture()
    tags, mask = tags.to(device), mask.to(device)
    for block in range(52):
        if block == 48:
            assert cache.drop_audio_history() == original.drop_audio_history()
        for active in (cache, original):
            active.begin_clean_commit(block)
        ordinals = torch.arange(7) + block * 7
        # Distinct exactly representable BF16 values, and one zero column per
        # layer, reveal swapped rows, commits or transformer layers.
        values = torch.where(ordinals < 256, ordinals, 256 + 2*(ordinals-256))
        x = values[:, None, None].expand(7, 1, 128).bfloat16().clone()
        for layer, name in enumerate(MAIN_LAYER_NAMES):
            keyed = x.clone()
            keyed[:, :, layer] = 0
            keyed = keyed.to(device)
            for active in (cache, original):
                active.stage(name, keyed, -keyed, tags, mask)
        for active in (cache, original):
            active.commit()
            active.retain_sink_and_recent_commits()
        for name in ("committed_blocks", "history_tokens", "history_video_tokens",
                     "history_audio_tokens"):
            assert getattr(cache, name) == getattr(original, name)
        for name in MAIN_LAYER_NAMES:
            restored = []
            for segment in cache._segments:
                pair = segment.layers[name]
                index = torch.tensor(segment.rows, dtype=torch.long, device=device)
                restored.append(tuple(
                    (q.index_select(0, index).float()*scale.index_select(0, index))
                    .bfloat16().cpu()
                    for q, scale in ((pair.key, pair.key_scale), (pair.value, pair.value_scale))
                ))
            dense = original.history(name)
            assert torch.equal(torch.cat([p[0] for p in restored]), dense.key.cpu())
            assert torch.equal(torch.cat([p[1] for p in restored]), dense.value.cpu())
    assert cache.committed_blocks == 52
    assert (cache.receipt()["physical_history_payload_bytes"]
            > cache.receipt()["visible_history_payload_bytes"])
    assert cache.receipt()["staged_payload_bytes"] == 0
    cache.clear()
    assert cache.committed_blocks == cache.history_tokens == 0


@pytest.mark.parametrize("layer_index", [0, 7, 49])
def test_failure_keeps_published_history_and_retry(layer_index, monkeypatch):
    cache = QuantizedCleanAVKVCache(KVContract(local_heads=1, device_type="cpu"))
    add(cache)
    old = cache._segments
    cache.begin_clean_commit(1)
    x, tags, mask = fixture()
    for i, name in enumerate(MAIN_LAYER_NAMES):
        if i == layer_index:
            bad = x.clone()
            bad[2, 0, 0] = float("nan")
            with pytest.raises(ValueError, match="nonfinite"):
                cache.stage(name, bad, x, tags, mask)
            break
        cache.stage(name, x, x, tags, mask)
    with pytest.raises(RuntimeError):
        cache.commit()
    cache.rollback()
    assert cache._segments is old
    add(cache)
    assert cache.committed_blocks == 2


def test_commit_sync_failure_is_atomic(monkeypatch):
    cache = QuantizedCleanAVKVCache(KVContract(local_heads=1, device_type="cpu"))
    add(cache)
    old = cache._segments
    cache.begin_clean_commit(1)
    x, tags, mask = fixture()
    for name in MAIN_LAYER_NAMES:
        cache.stage(name, x, x, tags, mask)
    with monkeypatch.context() as m:
        m.setattr(
            cache, "_sync", lambda: (_ for _ in ()).throw(RuntimeError("injected sync"))
        )
        with pytest.raises(RuntimeError, match="injected"):
            cache.commit()
    assert cache._segments is old and cache.committed_blocks == 1
    cache.rollback()
    add(cache)


def test_full_key_softmax_error_budget_and_ownership(monkeypatch):
    cache = QuantizedCleanAVKVCache(
        KVContract(local_heads=1, device_type="cpu"), tile_tokens=2
    )
    add(cache)
    g = torch.Generator().manual_seed(9)
    query = torch.randn(3, 1, 128, generator=g).bfloat16()
    condition = AVKV(
        torch.randn(2, 1, 128, generator=g).bfloat16(),
        torch.ones(2, 1, 128).bfloat16() * 100,
    )
    current = AVKV(
        torch.randn(4, 1, 128, generator=g).bfloat16(),
        torch.ones(4, 1, 128).bfloat16() * -100,
    )
    p = cache._segments[0].layers[MAIN_LAYER_NAMES[0]]
    key = torch.cat(
        [condition.key, (p.key.float() * p.key_scale).bfloat16(), current.key]
    )
    value = torch.cat(
        [condition.value, (p.value.float() * p.value_scale).bfloat16(), current.value]
    )
    ref = scaled_dot_product_attention(
        query.double().transpose(0, 1),
        key.double().transpose(0, 1),
        value.double().transpose(0, 1),
    ).transpose(0, 1)
    refs = []
    original = metal_kv.query_chunked_sdpa

    def spy(q, k, v, **kwargs):
        refs.extend([weakref.ref(k), weakref.ref(v)])
        with pytest.raises(RuntimeError, match="already active"):
            cache.clear()
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(metal_kv, "query_chunked_sdpa", spy)
    got = cache.attention_with_history(
        MAIN_LAYER_NAMES[0],
        query,
        condition,
        current,
        scale=128**-0.5,
        query_chunk_size=2,
        max_joined_bytes=10000,
    )
    assert (
        torch.linalg.vector_norm(got.double() - ref) / torch.linalg.vector_norm(ref)
        < 0.008
    )
    assert all(r() is None for r in refs)
    with monkeypatch.context() as m:
        m.setattr(
            torch,
            "empty",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("allocated before budget")
            ),
        )
        with pytest.raises(MemoryError):
            cache.attention_with_history(
                MAIN_LAYER_NAMES[0],
                query,
                condition,
                current,
                scale=1,
                query_chunk_size=2,
                max_joined_bytes=1,
            )


def test_metadata_dtype_finite_and_history_escape():
    cache = QuantizedCleanAVKVCache(KVContract(local_heads=1, device_type="cpu"))
    x, tags, mask = fixture()
    cache.begin_clean_commit(0)
    cache.stage(MAIN_LAYER_NAMES[0], x, x, tags, mask)
    wrong = tags.clone()
    wrong[0] = 9
    with pytest.raises(ValueError, match="metadata differs"):
        cache.stage(MAIN_LAYER_NAMES[1], x, x, wrong, mask)
    with pytest.raises(ValueError):
        cache.stage(MAIN_LAYER_NAMES[1], x.float(), x, tags, mask)
    with pytest.raises(RuntimeError):
        cache.history(MAIN_LAYER_NAMES[0])
    cache.rollback()
    assert cache.receipt()["staged_payload_bytes"] == 0


def test_eviction_releases_storage():
    cache = QuantizedCleanAVKVCache(KVContract(local_heads=1, device_type="cpu"))
    add(cache)
    add(cache)
    refs = [weakref.ref(p.key) for p in cache._segments[1].layers.values()]
    add(cache)
    add(cache)
    assert all(r() is None for r in refs)


def test_no_history_uses_same_layout():
    cache = QuantizedCleanAVKVCache(KVContract(local_heads=1, device_type="cpu"))
    q = torch.ones(1, 1, 128).bfloat16()
    condition = AVKV(q, q * 3)
    current = AVKV(q, q * 7)
    result = cache.attention_with_history(
        MAIN_LAYER_NAMES[0],
        q,
        condition,
        current,
        scale=1,
        query_chunk_size=1,
        max_joined_bytes=1024,
    )
    assert torch.equal(result, q * 5)


@pytest.mark.parametrize("layer_index", [0, 7, 49])
def test_allocation_failure_during_value_quantization(layer_index, monkeypatch):
    cache = QuantizedCleanAVKVCache(KVContract(local_heads=1, device_type="cpu"))
    add(cache)
    old = cache._segments
    cache.begin_clean_commit(1)
    x, tags, mask = fixture()
    for name in MAIN_LAYER_NAMES[:layer_index]:
        cache.stage(name, x, x, tags, mask)
    actual = cache._quantize_selected
    refs = []

    def fail_value(tensor, indices):
        if refs:
            raise MemoryError("injected value allocation")
        q, s = actual(tensor, indices)
        refs.extend([weakref.ref(q), weakref.ref(s)])
        return q, s

    with monkeypatch.context() as m:
        m.setattr(cache, "_quantize_selected", fail_value)
        with pytest.raises(MemoryError):
            cache.stage(MAIN_LAYER_NAMES[layer_index], x, x, tags, mask)
    assert all(r() is None for r in refs)
    assert cache._segments is old
    cache.rollback()
    add(cache)


def test_per_head_quantization_error_bound():
    cache = QuantizedCleanAVKVCache(
        KVContract(local_heads=3, device_type="cpu"), tile_tokens=2
    )
    g = torch.Generator().manual_seed(17)
    x = torch.randn(9, 3, 128, generator=g).bfloat16()
    q, s = cache._quantize_selected(x, torch.arange(9))
    assert ((q.double() * s.double() - x.double()).abs() <= s.double() * 0.501).all()


def test_failed_attention_releases_joined_even_if_exception_is_retained(monkeypatch):
    import gc

    cache = QuantizedCleanAVKVCache(KVContract(local_heads=1, device_type="cpu"))
    add(cache)
    query = torch.ones(2, 1, 128).bfloat16()
    pair = AVKV(query, query)
    references = []

    def fail(q, k, v, **kwargs):
        references.extend([weakref.ref(k), weakref.ref(v)])
        raise RuntimeError("injected attention failure")

    monkeypatch.setattr(metal_kv, "query_chunked_sdpa", fail)
    caught = None
    try:
        cache.attention_with_history(
            MAIN_LAYER_NAMES[0], query, pair, pair, scale=128**-0.5,
            query_chunk_size=1, max_joined_bytes=10000,
        )
    except RuntimeError as error:
        caught = error
    assert caught is not None
    cache.clear()
    gc.collect()
    assert all(reference() is None for reference in references)
