"""Independent precision/visibility checks for upstream layers on CPU and Metal.

These are small operator tests, not full-model or CUDA parity claims.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from taomate_h3.distributed import ParallelContext
from taomate_h3.model.architecture import MiniMaxH3Architecture
from taomate_h3.model.layers import (
    MiniMaxH3FinalLayer,
    _packed_sdpa,
    modulate_gate,
    modulate_scale_shift,
)


@pytest.fixture(params=("cpu", "mps"))
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal device unavailable in this process")
    return torch.device(request.param)


def _context():
    return ParallelContext(1, 0, 1, 0, None, None)


@pytest.mark.parametrize("modality", ("video", "audio"))
@torch.inference_mode()
def test_final_velocity_preserves_f32_parameters_and_output(device, modality):
    arch = MiniMaxH3Architecture(
        hidden_size=16, time_embed_dim=8, final_adaln_out_features=32,
    )
    layer = MiniMaxH3FinalLayer(arch, context=_context()).eval()
    # Isolate the projection contract while exercising the official final path.
    # Other tests exercise BF16 modulation separately.
    layer.norm = nn.Identity()
    layer.adaln_proj.linear.weight.zero_()
    layer.adaln_proj.linear.bias.zero_()
    projection = getattr(layer, modality + "_out")
    projection.weight.copy_(
        torch.arange(projection.weight.numel()).reshape_as(projection.weight)
        * 0.000103 + 0.0000173
    )
    projection.bias.copy_(torch.arange(projection.bias.numel()) * 0.0000197)
    hidden = (torch.arange(48).reshape(3, 16).float() / 17 - 1).bfloat16()
    expected = nn.functional.linear(hidden.float(), projection.weight, projection.bias)
    layer.to(device)
    output = getattr(layer, "forward_" + modality + "_only")(
        hidden.to(device),
        adaln_input=torch.zeros(1, 8, dtype=torch.bfloat16, device=device),
        inverse_indices=torch.zeros(3, dtype=torch.long, device=device),
    ).cpu()
    assert projection.weight.dtype == torch.float32
    assert projection.bias.dtype == torch.float32
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=2e-6)
    # A BF16-rounded output would still have FP32 dtype after a cast; reject it.
    assert (output - output.bfloat16().float()).abs().max() > 1e-5


@torch.inference_mode()
def test_gate_and_adaln_keep_each_bf16_rounding_boundary(device):
    generator = torch.Generator().manual_seed(7103)
    value, gate, other, shift, scale = (
        torch.randn(5, 128, generator=generator).bfloat16() for _ in range(5)
    )
    indices = torch.tensor([4, 2, 0, 3, 1], dtype=torch.long)
    selected_gate = gate.index_select(0, indices).float()
    expected_gate = (
        value.float() + (selected_gate * other.float()).bfloat16().float()
    ).bfloat16()
    selected_scale = scale.index_select(0, indices).float()
    one_plus_scale = (1.0 + selected_scale).bfloat16().float()
    product = (value.float() * one_plus_scale).bfloat16().float()
    expected_modulation = (
        product + shift.index_select(0, indices).float()
    ).bfloat16()
    actual_gate = modulate_gate(
        value.to(device), gate.to(device), other.to(device), indices.to(device),
        dtype=torch.bfloat16,
    ).cpu()
    actual_modulation = modulate_scale_shift(
        value.to(device), shift.to(device), scale.to(device), indices.to(device),
        dtype=torch.bfloat16,
    ).cpu()
    torch.testing.assert_close(actual_gate, expected_gate, rtol=0, atol=0)
    torch.testing.assert_close(actual_modulation, expected_modulation, rtol=0, atol=0)


@torch.inference_mode()
def test_packed_attention_matches_independent_dense_reference(device):
    generator = torch.Generator().manual_seed(1923)
    q, k, v = [torch.randn(7, 2, 128, generator=generator).bfloat16() for _ in range(3)]
    scale = 1 / math.sqrt(128)
    expected_segments = []
    for start, stop in ((0, 3), (3, 7)):
        qh, kh, vh = [t[start:stop].double().transpose(0, 1) for t in (q, k, v)]
        scores = qh @ kh.transpose(-1, -2) * scale
        expected_segments.append((scores.softmax(-1) @ vh).transpose(0, 1))
    expected = torch.cat(expected_segments).float()
    actual = _packed_sdpa(
        q.to(device), k.to(device), v.to(device),
        cu_seqlens_host=(0, 3, 3, 7), max_seqlen=4, scale=scale,
    ).float().cpu()
    # BF16 rounding scale; this tolerance is for this small attention fixture.
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.008)
    changed_v = v.clone()
    changed_v[3:] = 64
    changed = _packed_sdpa(
        q.to(device), k.to(device), changed_v.to(device),
        cu_seqlens_host=(0, 3, 3, 7), max_seqlen=4, scale=scale,
    ).float().cpu()
    torch.testing.assert_close(changed[:3], actual[:3], rtol=0, atol=0)
    assert (changed[3:] - actual[3:]).abs().max() > 1
