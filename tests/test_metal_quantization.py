"""Synthetic quantization contracts; no model weights or quality assertions."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from taomate_h3.inference.metal_quantization import MetalW8A16Linear, quantize_interior_linears_


def test_row_quantization_golden_and_bounded_linear():
    weight = torch.tensor([[0, 0, 0], [127, -127, 1], [63.5, -63.5, 0.5]], dtype=torch.bfloat16)
    layer = MetalW8A16Linear.from_weight(weight, load_rows=1, temporary_bytes=6)
    assert layer.weight.tolist() == [[0, 0, 0], [127, -127, 1], [127, -127, 1]]
    assert layer.scale[:, 0].tolist() == pytest.approx([1e-12, 1, 0.5])
    inputs = torch.tensor([[[1, 2, 3], [3, 2, 1]]], dtype=torch.bfloat16)
    assert torch.equal(layer(inputs), F.linear(inputs, weight))
    assert not any(t.dtype == torch.bfloat16 for t in layer.buffers())


def test_quantization_error_and_chunk_partition():
    g = torch.Generator().manual_seed(7)
    weight = torch.randn(65, 96, generator=g).bfloat16()
    inputs = torch.randn(2, 7, 96, generator=g).bfloat16()
    a = MetalW8A16Linear.from_weight(weight, load_rows=1, temporary_bytes=192)
    b = MetalW8A16Linear.from_weight(weight, load_rows=17)
    assert torch.equal(a.weight, b.weight)
    assert torch.equal(a.scale, b.scale)
    dequant = (a.weight.float() * a.scale).bfloat16()
    torch.testing.assert_close(a(inputs), F.linear(inputs, dequant), rtol=0.01, atol=0.125)
    reference = F.linear(inputs.float(), weight.float())
    relative = torch.linalg.vector_norm(a(inputs).float() - reference) / torch.linalg.vector_norm(
        reference
    )
    assert relative < 0.015
    assert b.receipt()["persistent_bytes"] == weight.numel() + 65 * 4


class LoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        self.register_buffer("lora_a", torch.ones(2, 32, dtype=torch.bfloat16) / 32)
        self.register_buffer("lora_b", torch.ones(32, 2, dtype=torch.bfloat16) / 32)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b)


def test_official_selection_protects_boundaries_and_lora():
    model = nn.Module()
    model.blocks = nn.ModuleList()
    for _ in range(50):
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.qkv_proj = LoRA()
        block.mlp = nn.Module()
        block.mlp.fc1 = LoRA()
        block.mlp.fc2 = LoRA()
        model.blocks.append(block)
    model.final = nn.Linear(32, 32, dtype=torch.float32)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    receipt = quantize_interior_linears_(model, temporary_bytes=2048)
    assert len(receipt["quantized_linear_names"]) == 90
    assert receipt["bf16_boundary_block_indices"] == [0, 1, 47, 48, 49]
    for index, block in enumerate(model.blocks):
        for name, wrapper in [
            ("attn.qkv_proj", block.attn.qkv_proj),
            ("mlp.fc1", block.mlp.fc1),
            ("mlp.fc2", block.mlp.fc2),
        ]:
            selected = 2 <= index <= 46 and name != "mlp.fc2"
            assert isinstance(wrapper.base, MetalW8A16Linear) == selected
            prefix = f"blocks.{index}.{name}"
            assert torch.equal(wrapper.lora_a, original[prefix + ".lora_a"])
            assert torch.equal(wrapper.lora_b, original[prefix + ".lora_b"])
            if not selected:
                assert torch.equal(wrapper.base.weight, original[prefix + ".base.weight"])
    assert torch.equal(model.final.weight, original["final.weight"])
    x = torch.ones(2, 32, dtype=torch.bfloat16)
    wrapper = model.blocks[3].mlp.fc1
    expected = wrapper.base(x) + F.linear(F.linear(x, wrapper.lora_a), wrapper.lora_b)
    assert torch.equal(wrapper(x), expected)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_weight_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        MetalW8A16Linear.from_weight(torch.tensor([[bad]], dtype=torch.bfloat16))


def test_invalid_dtype_and_budget():
    with pytest.raises(ValueError):
        MetalW8A16Linear.from_weight(torch.ones(2, 3))
    with pytest.raises(ValueError, match="budget"):
        MetalW8A16Linear.from_weight(torch.ones(2, 3).bfloat16(), temporary_bytes=5)
    layer = MetalW8A16Linear.from_weight(torch.ones(2, 3).bfloat16())
    with pytest.raises(ValueError, match="BF16"):
        layer(torch.ones(2, 3))


@pytest.mark.parametrize("rows", [1, 4, 17])
@pytest.mark.parametrize("backend", ["dequantize", "native_int8pack"])
def test_mps_small_matrix(rows, backend):
    if not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    g = torch.Generator().manual_seed(41)
    weight = torch.randn(64, 128, generator=g).bfloat16()
    x = torch.randn(rows, 128, generator=g).bfloat16()
    layer = MetalW8A16Linear.from_weight(
        weight, device="mps", backend=backend, temporary_bytes=8192
    )
    actual = layer(x.to("mps")).cpu().float()
    q = layer.weight.cpu().double()
    scale = layer.scale.cpu().double()
    if backend == "native_int8pack":
        scale = scale.bfloat16().double()
        reference = (x.double() @ q.T) * scale.flatten()
    else:
        reference = x.double() @ (q * scale).bfloat16().double().T
    relative = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference)
    assert torch.isfinite(actual).all()
    assert relative < 0.008
    assert (actual - reference).abs().max() < 0.4
