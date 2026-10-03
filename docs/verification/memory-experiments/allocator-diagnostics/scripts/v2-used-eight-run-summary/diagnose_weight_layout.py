"""Real-shape, real-weight dequantized FC1/QKV output-layout memory diagnostic."""
from __future__ import annotations
import argparse
import json
import time
from diag_common import (MIB, Recorder, common_arguments, initialize, resident_control,
    cpu_normal_bf16, load_real_linear, compare_cpu, release_cache)


def layout_linear(torch, recorder, inputs, owner, variant, repeat):
    from torch.nn import functional as F
    rows = owner._metal_temporary_bytes // (owner.weight.shape[1] * 2)
    shape = (len(inputs), owner.weight.shape[0])
    output = None
    pieces = []
    if variant == 'strided':
        output = inputs.new_empty(shape)
    elif variant == 'transposed':
        output = inputs.new_empty((shape[1], shape[0]))
    recorder.memory(torch, 'output_destination_created', repeat=repeat, variant=variant)
    for tile, start in enumerate(range(0, shape[1], rows)):
        end = min(start+rows, shape[1])
        temporary = (owner.weight[start:end].float() * owner._metal_weight_scale[start:end]).bfloat16()
        recorder.memory(torch, 'dequantized_tile', synchronize=False, repeat=repeat, tile=tile, rows=end-start)
        piece = F.linear(inputs, temporary)
        recorder.memory(torch, 'matmul_piece_live', synchronize=False, repeat=repeat, tile=tile,
                        piece_bytes=piece.numel()*piece.element_size())
        if variant == 'strided':
            output[:,start:end].copy_(piece)
        elif variant == 'transposed':
            output[start:end].copy_(piece.T)
        elif variant == 'cat':
            pieces.append(piece)
        else:
            raise ValueError(variant)
        del piece, temporary
        # Same one completion barrier per tile as the production dequantize path.
        recorder.memory(torch, 'tile_finished', repeat=repeat, tile=tile)
    if variant == 'cat':
        output = torch.cat(pieces, dim=1)
        recorder.memory(torch, 'cat_with_pieces_live', repeat=repeat)
        pieces.clear()
    elif variant == 'transposed':
        final = output.T.contiguous()
        recorder.memory(torch, 'canonical_copy_with_transposed_live', repeat=repeat)
        del output
        output = final
    recorder.memory(torch, 'canonical_output_ready', repeat=repeat, strides=list(output.stride()))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument('--kind', choices=('fc1','qkv'), default='fc1')
    parser.add_argument('--rows', type=int, default=20928)
    parser.add_argument('--tile-mib', type=int, default=64)
    parser.add_argument('--variant', choices=('strided','cat','transposed'), default='strided')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.rows <= 25024 or not 1 <= args.tile_mib <= 256 or not 1 <= args.repeats <= 10:
        parser.error('rows=1..25024, tile-mib=1..256, repeats=1..10 required')
    output_columns = 28672 if args.kind == 'fc1' else 21504
    plan = {'case':'A_weight_layout', 'input':[args.rows,5376], 'weight':[output_columns,5376],
            'output':[args.rows,output_columns], 'output_payload_bytes':args.rows*output_columns*2,
            'tile_bf16_weight_budget_bytes':args.tile_mib*MIB, 'variant':args.variant,
            'repeat_count':args.repeats, 'resident_control_gib':args.resident_gib,
            'seed':args.seed, 'input_kind':'deterministic CPU Gaussian F32 -> BF16; same input every repeat/variant',
            'reference':'Production metal_linear on the same real quantized weight, run AFTER measured layout trials',
            'timing_scope':'Instrumented layout only; excludes load, RNG, CPU copy/comparison and later reference. Includes per-tile stats bookkeeping/barriers.',
            'interpretation':'Single matrix; no LoRA, other DiT weights, KV, teacher or VAE. Resident control is synthetic uniform 256 MiB blocks.'}
    if args.plan:
        print(json.dumps(plan,indent=2)); return
    rec = Recorder(args, plan)
    try:
        torch = initialize(args, rec)
        with torch.inference_mode():
            resident = resident_control(torch, rec, args.resident_gib)
            owner = load_real_linear(torch, rec, args.weights, args.kind, args.tile_mib*MIB)
            cpu_input = cpu_normal_bf16(torch, (args.rows,5376), args.seed)
            inputs = cpu_input.to('mps')
            del cpu_input
            rec.memory(torch, 'input_ready')
            trials, last_cpu, first_cpu = [], None, None
            for repeat in range(args.repeats):
                torch.accelerator.reset_peak_memory_stats('mps')
                rec.memory(torch, 'trial_begin', repeat=repeat)
                began = time.monotonic()
                output = layout_linear(torch, rec, inputs, owner, args.variant, repeat)
                elapsed = time.monotonic()-began
                last_cpu = output.cpu()
                del output
                rec.memory(torch, 'trial_output_released', repeat=repeat)
                if first_cpu is None:
                    first_cpu = last_cpu
                comparison = compare_cpu(torch, last_cpu, first_cpu)
                if not comparison['finite']:
                    raise RuntimeError('Nonfinite layout output')
                trials.append({'repeat':repeat,'instrumented_seconds':elapsed,'repeat_vs_first':comparison})
                # No empty_cache between repeats: exposes heap reuse/accumulation.
            rec.receipt['layout_trials'] = trials
            rec.memory(torch, 'layout_trials_complete_before_reference')
            release_cache(torch, rec, 'layout_outputs_released')
            from taomate_h3.inference.metal_loader import metal_linear
            began = time.monotonic()
            reference = metal_linear(inputs, owner)
            torch.mps.synchronize()
            reference_seconds = time.monotonic()-began
            reference_cpu = reference.cpu()
            del reference
            rec.memory(torch, 'production_reference_output_released')
            difference = compare_cpu(torch, last_cpu, reference_cpu)
            rec.event('comparison', comparison=difference)
            if not difference['finite']:
                raise RuntimeError('Nonfinite reference or variant output')
            del inputs, owner, resident, last_cpu, first_cpu, reference_cpu
            release_cache(torch, rec, 'all_controls_released')
            rec.finish(layout_trials=trials, variant_vs_production=difference,
                       production_reference_seconds=reference_seconds)
    except BaseException as error:
        rec.finish(error=error)
        raise


if __name__ == '__main__':
    main()
