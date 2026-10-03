"""Contiguous row views preserve gather layout and arbitrary-mask fallback."""

import pytest
import torch

from taomate_h3.streaming import attention_hook as hooks


@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.no_grad()
def test_selection_bitwise_alias_and_fallback(monkeypatch, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS window required")
    monkeypatch.setenv("TAOMATE_MPS_ATTENTION_ROW_VIEWS", "1")
    x = torch.arange(13 * 256).reshape(13, 2, 128).bfloat16().to(device)
    before = x.clone()
    for numbers in ([0, 1, 2], [3, 4, 5], [0, 2, 4], [2, 1, 0], [0, 0, 2], []):
        index = torch.tensor(numbers, dtype=torch.long, device=device)
        span = hooks._contiguous_span(index)
        assert (span is not None) == (numbers in ([0, 1, 2], [3, 4, 5]))
        expected = x.index_select(0, index)
        if device == "cpu":
            monkeypatch.setattr(hooks, "_row_views_enabled", lambda _: True)
        result = hooks._select_attention_rows(x, index, span)
        assert torch.equal(result.view(torch.int16), expected.view(torch.int16))
        assert result.stride() == expected.stride()
        if span is not None:
            assert result.untyped_storage()._cdata == x.untyped_storage()._cdata
        # Preserve original gather layout even if source is noncontiguous.
        strided = x.transpose(0, 1)
        small = torch.tensor([0, 1], device=device)
        actual = hooks._select_attention_rows(strided, small, (0, 2))
        assert torch.equal(actual, strided.index_select(0, small))
        assert actual.is_contiguous()
        assert actual.untyped_storage()._cdata != strided.untyped_storage()._cdata
    assert torch.equal(x, before)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_gate_default_grad_and_device(monkeypatch, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS window required")
    x = torch.ones(2, 1, 128, dtype=torch.bfloat16, device=device)
    monkeypatch.delenv("TAOMATE_MPS_ATTENTION_ROW_VIEWS", raising=False)
    with torch.no_grad():
        assert not hooks._row_views_enabled(x)
    monkeypatch.setenv("TAOMATE_MPS_ATTENTION_ROW_VIEWS", "1")
    assert not hooks._row_views_enabled(x)
    with torch.no_grad():
        assert hooks._row_views_enabled(x) == (device == "mps")
        x.requires_grad_(True)
        assert not hooks._row_views_enabled(x)
