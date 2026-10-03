"""Tile-size invariance, unchanged full-key attention and explicit scratch bounds."""

import gc

import pytest
import torch

from taomate_h3.streaming.cache import AVKV, MAIN_LAYER_NAMES, KVContract
from taomate_h3.streaming.metal_kv import (
    MAX_KV_TILE_SCRATCH_BYTES,
    QuantizedCleanAVKVCache,
    kv_tile_scratch_bound,
    parse_kv_tile_tokens,
)


@pytest.mark.parametrize(
    "text", ["", "0", "-1", "+128", " 128", "128 ", "1.0", "2049", "１２８", "1" * 10000]
)
def test_strict_environment_rejects_invalid(text):
    with pytest.raises(ValueError):
        parse_kv_tile_tokens(text)


def test_default_and_scratch_budget():
    contract = KVContract(local_heads=56, device_type="cpu")
    cache = QuantizedCleanAVKVCache(contract)
    assert cache.tile_tokens == parse_kv_tile_tokens("128") == 128
    large = QuantizedCleanAVKVCache(contract, tile_tokens=1024)
    assert large.receipt()["tile_scratch_estimated_bytes"] == kv_tile_scratch_bound(1024, 56)
    assert large.receipt()["tile_scratch_budget_bytes"] == 256 * 1024**2
    assert large.tile_scratch_estimated_bytes < MAX_KV_TILE_SCRATCH_BYTES
    with pytest.raises(ValueError, match="scratch"):
        QuantizedCleanAVKVCache(contract, tile_tokens=2048)
    assert QuantizedCleanAVKVCache(KVContract(local_heads=1, device_type="cpu"), tile_tokens=2048)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.no_grad()
def test_128_and_1024_exact_quantization_and_full_attention(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    g = torch.Generator().manual_seed(114)
    # Not a multiple of either tile; different K/V and per-head extrema.
    key = torch.randn(1031, 2, 128, generator=g).bfloat16().to(device)
    value = (torch.randn(1031, 2, 128, generator=g) * 2).bfloat16().to(device)
    tags = torch.tensor([1] + [2] * 330 + [0] * 700, device=device)
    mask = tags != 1
    caches = [
        QuantizedCleanAVKVCache(KVContract(local_heads=2, device_type=device), tile_tokens=t)
        for t in (128, 1024)
    ]
    for cache in caches:
        cache.begin_clean_commit(0)
        for layer in MAIN_LAYER_NAMES:
            cache.stage(layer, key, value, tags, mask)
        cache.commit()
        assert cache.committed_blocks == 1 and cache.history_tokens == 1030
    for layer in MAIN_LAYER_NAMES:
        a, b = [c._segments[0].layers[layer] for c in caches]
        for field in ("key", "value", "key_scale", "value_scale"):
            assert torch.equal(getattr(a, field), getattr(b, field))
    condition = AVKV(key[:3], value[:3])
    current = AVKV(key[-7:], value[-7:])
    query = key[127:136]
    results = [
        c.attention_with_history(
            MAIN_LAYER_NAMES[0],
            query,
            condition,
            current,
            scale=128**-0.5,
            query_chunk_size=4,
            max_joined_bytes=2 * 1024**2,
        ).cpu()
        for c in caches
    ]
    assert torch.equal(*results)
    for cache in caches:
        cache.clear()
    del a, b, caches, cache, results, key, value
    gc.collect()
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("position", [127, 128, 1023, 1024, 1030])
def test_nonfinite_across_tile_boundaries(device, position):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    x = torch.zeros(1031, 1, 128, dtype=torch.bfloat16, device=device)
    x[position, 0, 127] = float("nan") if position % 2 else float("inf")
    tags = torch.tensor([2] * 331 + [0] * 700, device=device)
    mask = torch.ones(1031, dtype=torch.bool, device=device)
    for tile in (128, 1024):
        cache = QuantizedCleanAVKVCache(
            KVContract(local_heads=1, device_type=device), tile_tokens=tile
        )
        cache.begin_clean_commit(0)
        with pytest.raises(ValueError, match="nonfinite"):
            cache.stage(MAIN_LAYER_NAMES[0], x, x, tags, mask)
        assert cache.receipt()["staged_payload_bytes"] == 0 and cache.committed_blocks == 0
        cache.rollback()


@pytest.mark.parametrize("requested", [None, "1024"])
def test_runtime_option_receipt_without_model_loading(monkeypatch, tmp_path, requested):
    import taomate_h3.model.pipeline as pipeline
    import taomate_h3.runtime.mps as mps

    monkeypatch.setattr(mps, "require_mps", lambda: None)
    monkeypatch.setattr(pipeline, "MiniMaxH3NativePipeline", lambda **kw: object())
    monkeypatch.setenv("TAOMATE_MPS_KV_STORAGE", "int8")
    monkeypatch.setenv("TAOMATE_MPS_CPU_ADALN", "0")
    if requested is None:
        monkeypatch.delenv("TAOMATE_MPS_KV_TILE_TOKENS", raising=False)
    else:
        monkeypatch.setenv("TAOMATE_MPS_KV_TILE_TOKENS", requested)
    runtime = mps.MPSRuntime(model_root=tmp_path, adapter_root=tmp_path, output_root=tmp_path)
    receipt = runtime.model_resolution["metal_workspace_options"]
    assert receipt["kv_tile_tokens"] == (128 if requested is None else 1024)
    assert receipt["kv_tile_scratch_estimated_bytes"] == kv_tile_scratch_bound(
        receipt["kv_tile_tokens"], 56
    )


def test_runtime_rejects_scratch_overrun_before_loading(monkeypatch, tmp_path):
    import taomate_h3.model.pipeline as pipeline
    import taomate_h3.runtime.mps as mps

    monkeypatch.setattr(mps, "require_mps", lambda: None)

    def forbidden(**kw):
        raise AssertionError("loaded before validation")

    monkeypatch.setattr(pipeline, "MiniMaxH3NativePipeline", forbidden)
    monkeypatch.setenv("TAOMATE_MPS_KV_STORAGE", "int8")
    monkeypatch.setenv("TAOMATE_MPS_CPU_ADALN", "0")
    monkeypatch.setenv("TAOMATE_MPS_KV_TILE_TOKENS", "2048")
    with pytest.raises(ValueError, match="scratch"):
        mps.MPSRuntime(model_root=tmp_path, adapter_root=tmp_path, output_root=tmp_path)
