"""Guarded isolated real-shape W8A16 benchmark; not full-model throughput."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from taomate_h3.inference.metal_loader import load_metal_stream_weight_, metal_linear


def execute(inputs, owner, baseline_weight, backend):
    if backend == "bf16":
        return torch.nn.functional.linear(inputs, baseline_weight)
    return metal_linear(inputs, owner)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[256, 4990, 24590])
    parser.add_argument("--output", required=True)
    parser.add_argument("--outputs", type=int, nargs="+", default=[21504, 28672])
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["bf16", "native_int8pack", "dequantize"],
        default=["bf16", "native_int8pack", "dequantize"],
    )
    parser.add_argument("--tile-mib", type=int, choices=range(1, 257), default=256)
    args = parser.parse_args()
    torch.manual_seed(41)
    results = []
    for output_features in args.outputs:
        source = (torch.randn(output_features, 5376) * 0.02).bfloat16()
        owner = torch.nn.Module()
        owner.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(source.shape, device="mps", dtype=torch.int8), requires_grad=False
            ),
        )
        owner.register_parameter("bias", None)
        owner.register_buffer("_metal_weight_scale", torch.empty(output_features, 1, device="mps"))
        owner.register_buffer(
            "_metal_native_scale", torch.empty(output_features, device="mps", dtype=torch.bfloat16)
        )
        owner._metal_stream_load_pending = True
        owner._metal_stream_loaded = False
        owner._metal_temporary_bytes = args.tile_mib * 1024**2
        load_metal_stream_weight_(owner, source)
        baseline_weight = source.to("mps") if "bf16" in args.backends else None
        del source
        for rows in args.rows:
            inputs = torch.randn(rows, 5376, device="mps", dtype=torch.bfloat16)
            item = {"input_rows": rows, "input_features": 5376, "output_features": output_features}
            for backend in args.backends:
                owner._metal_quantization_backend = backend
                for _ in range(2):
                    result = execute(inputs, owner, baseline_weight, backend)
                    torch.mps.synchronize()
                    del result
                timings = []
                for _ in range(3):
                    start = time.perf_counter()
                    result = execute(inputs, owner, baseline_weight, backend)
                    torch.mps.synchronize()
                    timings.append(time.perf_counter() - start)
                    del result
                item[backend] = {"seconds": sorted(timings)[1], "repeat_seconds": timings}
            if rows == min(args.rows) and baseline_weight is not None:
                reference = torch.nn.functional.linear(inputs, baseline_weight).float()
                for backend in args.backends:
                    if backend == "bf16":
                        continue
                    owner._metal_quantization_backend = backend
                    output = metal_linear(inputs, owner).float()
                    item[backend]["relative_l2_vs_bf16"] = float(
                        (
                            torch.linalg.vector_norm(output - reference)
                            / torch.linalg.vector_norm(reference)
                        ).item()
                    )
                    item[backend]["finite"] = bool(torch.isfinite(output).all().item())
                    del output
                del reference
            print(json.dumps(item), flush=True)
            results.append(item)
            del inputs
            gc.collect()
            torch.mps.synchronize()
            torch.mps.empty_cache()
        del owner, baseline_weight
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
    Path(args.output).write_text(
        json.dumps(
            {
                "isolated_linear_benchmark": results,
                "temporary_weight_bf16_budget_bytes": args.tile_mib * 1024**2,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
