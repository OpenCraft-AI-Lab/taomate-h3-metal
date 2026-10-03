"""Weight-tile configuration, real module dispatch and independent matmul oracle."""

import pytest
import torch
from torch import nn

from taomate_h3.distributed import ParallelContext
from taomate_h3.distributed.linear import MergedColumnParallelLinear
from taomate_h3.inference.metal_loader import (
    allocate_metal_storage_,
    load_metal_stream_weight_,
    parse_weight_tile_mib,
)


@pytest.mark.parametrize(
    "value", ["", "0", "-1", "+64", "64.0", " 64", "64 ", "257", "６４", "1" * 10000]
)
def test_invalid_env_budget(value):
    with pytest.raises(ValueError):
        parse_weight_tile_mib(value)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.no_grad()
def test_different_tiles_real_dispatch_crosses_tail(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    # 1031 output rows at 512 BF16 inputs cross the 1 MiB boundary (1024 rows).
    context = ParallelContext(1, 0, 1, 0, None, None)
    results = []
    g = torch.Generator().manual_seed(927)
    source = (torch.randn(1031, 512, generator=g) * 0.02).bfloat16()
    inputs = torch.randn(7, 512, generator=g).bfloat16()
    quantized = []
    for mib in (1, 2):
        with torch.device("meta"):
            model = nn.Module()
            model.blocks = nn.ModuleList()
            for _ in range(6):
                block = nn.Module()
                block.mlp = nn.Module()
                block.mlp.fc1 = MergedColumnParallelLinear(
                    512,
                    (515, 516),
                    bias=False,
                    gather_output=False,
                    dtype=torch.bfloat16,
                    context=context,
                )
                model.blocks.append(block)
        keys = tuple(model.state_dict())
        receipt = allocate_metal_storage_(
            model,
            device,
            quantization_backend="dequantize",
            temporary_bytes=parse_weight_tile_mib(str(mib)),
        )
        assert tuple(model.state_dict()) == keys
        assert receipt["bf16_weight_tile_budget_bytes"] == mib * 1024**2
        owner = model.blocks[2].mlp.fc1
        assert type(owner) is MergedColumnParallelLinear
        load_metal_stream_weight_(owner, source)
        quantized.append((owner.weight.cpu().clone(), owner._metal_weight_scale.cpu().clone()))
        results.append(owner(inputs.to(device)).cpu())
    assert torch.equal(quantized[0][0], quantized[1][0])
    assert torch.equal(quantized[0][1], quantized[1][1])
    # Different GEMM shapes can change backend accumulation; assess both against
    # an independent CPU FP64 dot product using exactly the stored BF16 weights.
    q, scale = quantized[0]
    weight = (q.double() * scale.double()).bfloat16()
    reference = inputs.double() @ weight.double().T
    for output in results:
        relative = torch.linalg.vector_norm(output.double() - reference) / torch.linalg.vector_norm(
            reference
        )
        assert torch.isfinite(output).all() and relative < 0.008
        assert (output.double() - reference).abs().max() < 0.025
    torch.testing.assert_close(*results, rtol=0.015, atol=0.015)
    if device == "mps":
        torch.mps.synchronize()


@pytest.mark.parametrize("backend", ["none", "native_int8pack", "dequantize"])
@pytest.mark.parametrize("setting", [None, "64"])
def test_runtime_to_pipeline_budget_and_receipt(monkeypatch, tmp_path, backend, setting):
    import taomate_h3.model.pipeline as pipeline
    import taomate_h3.runtime.mps as mps

    captured = {}

    def fake_pipeline(**kw):
        captured.update(kw)
        return object()

    monkeypatch.setattr(mps, "require_mps", lambda: None)
    monkeypatch.setattr(pipeline, "MiniMaxH3NativePipeline", fake_pipeline)
    monkeypatch.setenv("TAOMATE_MPS_WEIGHT_QUANTIZATION", backend)
    monkeypatch.setenv("TAOMATE_MPS_CPU_ADALN", "0")
    monkeypatch.setenv("TAOMATE_MPS_KV_STORAGE", "bf16")
    if setting is None:
        monkeypatch.delenv("TAOMATE_MPS_WEIGHT_TILE_MIB", raising=False)
    else:
        monkeypatch.setenv("TAOMATE_MPS_WEIGHT_TILE_MIB", setting)
    runtime = mps.MPSRuntime(model_root=tmp_path, adapter_root=tmp_path, output_root=tmp_path)
    expected = (256 if setting is None else 64) * 1024**2
    assert captured["transformer_metal_weight_tile_bytes"] == expected
    assert runtime.model_resolution["metal_workspace_options"]["bf16_weight_tile_budget_bytes"] == (
        expected if backend == "dequantize" else None
    )


def test_pipeline_forwards_budget_to_allocate(monkeypatch, tmp_path):
    import taomate_h3.model.pipeline as pipeline

    captured = {}

    def allocate(*args, **kwargs):
        captured.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(pipeline.MiniMaxH3DiT, "allocate", allocate)
    monkeypatch.setattr(pipeline, "load_minimax_h3_dit_weights", lambda *args: {})
    instance = pipeline.MiniMaxH3NativePipeline.__new__(pipeline.MiniMaxH3NativePipeline)
    instance.__dict__.update(
        transformer=None,
        device=torch.device("mps"),
        parallel_context=None,
        transformer_load_w8a8_policy=None,
        transformer_metal_cpu_adaln=True,
        transformer_metal_quantization_backend="dequantize",
        transformer_metal_quantization_policy="interior_fc1_qkv_v1",
        transformer_metal_weight_tile_bytes=64 * 1024**2,
        partition_root=tmp_path,
        component_timings={},
    )
    instance._load_transformer()
    assert captured["metal_weight_tile_bytes"] == 64 * 1024**2
