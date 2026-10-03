"""Query chunk sweep of unchanged production SDPA, with synthetic BF16 NHD inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
import time

from diag_common import (GIB, Recorder, common_arguments, initialize,
                         cpu_normal_bf16, compare_cpu, release_cache)

CASES = {'first': (24610,24610), 'history': (20514,86392)}


def cpu_digest(torch,value):
    """Hash already-copied CPU output with bounded host scratch."""
    if value.device.type!='cpu':
        raise ValueError('Output digest must run on CPU')
    digest=hashlib.sha256()
    for start in range(0,len(value),128):
        block=value[start:start+128].contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(block))
    return digest.hexdigest()


def make_plan(args):
    query_tokens,key_tokens=CASES[args.case]
    chunks=[int(x) for x in args.query_chunks.split(',')]
    if (not chunks or chunks[0]!=64 or chunks!=sorted(set(chunks))
            or not set(chunks)<={64,128,256,512,1024,2048,4096}):
        raise ValueError('Query chunks must be unique ascending choices from 64,128,256,512,1024,2048,4096, beginning with 64')
    expected=getattr(args,'expected_reference_sha256',None)
    if expected is not None and (len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected)):
        raise ValueError('Expected reference SHA256 must be 64 lowercase hex characters')
    strict=getattr(args,'require_bitwise',False) or expected is not None
    return {'case':'C_SDPA_query_chunks','shape_case':args.case,'query_shape':[query_tokens,56,128],
            'key_value_shape':[key_tokens,56,128],'dtype':'BF16','input_layout':'contiguous NHD',
            'attention_layout':'production transpose/unsqueeze views into BHND; no pre-contiguous K/V copies',
            'input_payload_bytes':(query_tokens+2*key_tokens)*56*128*2,
            'output_payload_bytes':query_tokens*56*128*2,'query_chunks':chunks,
            'dispatch_counts':[math.ceil(query_tokens/chunk) for chunk in chunks],
            'seed':args.seed,'repeats':args.repeats,'scale':1/math.sqrt(128),
            'softmax':'Every query block sees the entire unchanged key axis; noncausal, dropout=0',
            'reference':'First chunk64 production output; compare all output elements and CPU-byte SHA256',
            'expected_previous_reference_cpu_sha256':expected,
            'require_bitwise':strict,
            'timing_scope':'Unmodified production function only, including its per-query synchronize. Excludes CPU RNG, H2D, memory snapshots, output D2H, hashing/comparison.',
            'warmth':'First trial per chunk reported separately; median of subsequent trials. Same process/inputs, ascending chunk order; no empty_cache between trials/chunks.',
            'outer_footprint_limit_gib':12,'internal_sampled_driver_limit_gib':10,
            'limitations':['Synthetic operator inputs; no teacher/model/LoRA/VAE/KV allocation interleaving.',
                'history K=86392 is an explicit approximate late-history count, not a claim of exact prompt-dependent full-model key count.',
                'first uses Q=K=24610 as requested; production condition tokens may add keys.',
                'Chunk64 numerical reference is not an independent CUDA/FP32 oracle.',
                'A/R peaks are exact allocator counters; D only sampled at operation boundaries. Outer guard records whole-PID kernel peak.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.set_defaults(resident_gib=0)
    parser.add_argument('--case',choices=CASES,default='first')
    parser.add_argument('--query-chunks',default='64,128,256,512')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--expected-reference-sha256',help='Require every full output to match this earlier CPU-byte SHA256; implies strict bitwise equality.')
    parser.add_argument('--require-bitwise',action='store_true',help='Fail if any output differs bitwise from the first query64 output.')
    args=parser.parse_args()
    if args.resident_gib!=0 or not 2<=args.repeats<=5:
        parser.error('SDPA sweep requires resident-gib=0 and repeats within 2..5')
    try:
        plan=make_plan(args)
    except ValueError as error:
        parser.error(str(error))
    if args.plan:
        print(json.dumps(plan,indent=2));return
    rec=Recorder(args,plan,driver_limit_gib=10)
    try:
        torch=initialize(args,rec)
        from taomate_h3.inference.metal_attention import query_chunked_sdpa
        query_tokens,key_tokens=CASES[args.case]
        with torch.inference_mode():
            query=cpu_normal_bf16(torch,(query_tokens,56,128),args.seed).to('mps')
            rec.memory(torch,'query_live')
            key=cpu_normal_bf16(torch,(key_tokens,56,128),args.seed+1).to('mps')
            rec.memory(torch,'key_live')
            value=cpu_normal_bf16(torch,(key_tokens,56,128),args.seed+2).to('mps')
            rec.memory(torch,'all_inputs_live',query_stride=list(query.stride()),
                       key_stride=list(key.stride()),value_stride=list(value.stride()),
                       key_bhnd_view_stride=list(key.transpose(0,1).unsqueeze(0).stride()))
            results=[]
            reference_cpu=None
            reference_sha=None
            for chunk in plan['query_chunks']:
                trials=[]
                for repeat in range(args.repeats):
                    torch.accelerator.reset_peak_memory_stats('mps')
                    rec.memory(torch,'trial_begin',query_chunk=chunk,repeat=repeat)
                    started=time.monotonic()
                    output=query_chunked_sdpa(query,key,value,scale=plan['scale'],query_chunk_size=chunk)
                    # The production function already synchronizes after every
                    # query chunk, including its final chunk. Do not replace or
                    # instrument its internals with another copy/sync policy.
                    seconds=time.monotonic()-started
                    memory=rec.memory(torch,'trial_output_live',synchronize=False,
                                      query_chunk=chunk,repeat=repeat,operator_seconds=seconds)
                    cpu=output.cpu()
                    del output
                    rec.memory(torch,'trial_output_released',query_chunk=chunk,repeat=repeat)
                    digest=cpu_digest(torch,cpu)
                    if reference_cpu is None:
                        reference_cpu=cpu
                        reference_sha=digest
                    distance=compare_cpu(torch,cpu,reference_cpu)
                    distance['bitwise_equal']=digest==reference_sha
                    trial={'repeat':repeat,'operator_seconds':seconds,'output_cpu_sha256':digest,
                           'vs_chunk64_reference':distance,
                           'matches_previous_reference':None if plan['expected_previous_reference_cpu_sha256'] is None else digest==plan['expected_previous_reference_cpu_sha256'],
                           'allocated_peak_bytes':memory['allocated_peak_bytes'],
                           'reserved_peak_bytes':memory['reserved_peak_bytes'],
                           'driver_bytes_at_output':memory['driver_bytes']}
                    trials.append(trial)
                    rec.event('trial_checked',query_chunk=chunk,**trial)
                    if not distance['finite']:
                        raise RuntimeError('Nonfinite SDPA output/reference')
                    if plan['require_bitwise'] and not distance['bitwise_equal']:
                        raise RuntimeError(f'Query chunk {chunk} differs bitwise from chunk64; exact-output acceptance failed')
                    if trial['matches_previous_reference'] is False:
                        raise RuntimeError(f'Query chunk {chunk} does not match the pinned earlier output SHA256')
                    del cpu
                result={'query_chunk':chunk,'trials':trials,'cold_or_first_operator_seconds':trials[0]['operator_seconds'],
                        'later_operator_median_seconds':statistics.median(t['operator_seconds'] for t in trials[1:])}
                results.append(result)
                rec.receipt['chunk_results']=results
                rec.flush()
            del query,key,value,reference_cpu
            release_cache(torch,rec,'all_sdpa_inputs_released')
            rec.finish(chunk_results=results,chunk64_reference_cpu_sha256=reference_sha)
    except BaseException as error:
        rec.finish(error=error)
        raise


if __name__=='__main__':
    main()
