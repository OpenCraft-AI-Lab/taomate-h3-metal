"""Production INT8 KV allocation/retention with real phase shapes; no model generation."""
from __future__ import annotations
import argparse
import gc
import inspect
import json
import math
import sys
import time
from diag_common import (GIB, MIB, Recorder, common_arguments, initialize, resident_control,
    cpu_normal_bf16, load_real_linear, compare_cpu, release_cache, sha)


def tensor_digest(torch, value):
    if value.device.type != 'cpu':
        raise ValueError('Digest only already-collected CPU samples')
    return sha(value.contiguous().view(torch.uint8).numpy().tobytes())


def geometry(repository, phases, condition_tokens):
    # Pure Python official plan, imported only after argv parsing; no torch/MPS.
    sys.path.insert(0, str(repository/'src'))
    from taomate_h3.streaming.geometry import direct_5s_plan
    result = []
    for phase in direct_5s_plan().phases[:phases]:
        video = (phase.video_latent_stop-phase.video_latent_start)*2040
        audio = 2*(phase.audio_latent_stop-phase.audio_latent_start)
        unpadded = condition_tokens+video+audio
        result.append({'phase':phase.index,'video_tokens':video,'audio_tokens':audio,
                       'condition_tokens':condition_tokens,'unpadded_tokens':unpadded,
                       'packed_tokens':((unpadded+63)//64)*64,
                       'physical_commit_payload_bytes':(video+audio)*56*(128+4)*2*50})
    return result


def independent_dense_cpu(torch, cache, layer_name, condition, current):
    """Independent host reconstruction; no full history BF16 GPU retained."""
    count = len(condition.key)+cache.history_tokens+len(current.key)
    host = []
    for which in range(2):
        out = torch.empty((count,56,128),dtype=torch.bfloat16,device='cpu')
        prefix = condition.key if which == 0 else condition.value
        suffix = current.key if which == 0 else current.value
        out[:len(prefix)].copy_(prefix.cpu())
        position = len(prefix)
        for segment in cache._segments:
            pair = segment.layers[layer_name]
            q, s = (pair.key,pair.key_scale) if which == 0 else (pair.value,pair.value_scale)
            # CPU copies are per segment, one of K/V at a time; no GPU index_select.
            qc, sc = q.cpu(), s.cpu()
            for start in range(0,len(segment.rows),1024):
                idx = torch.tensor(segment.rows[start:start+1024],dtype=torch.long)
                restored = (qc.index_select(0,idx).float()*sc.index_select(0,idx)).bfloat16()
                out[position:position+len(idx)].copy_(restored)
                position += len(idx)
            del qc, sc, restored, idx
        out[position:].copy_(suffix.cpu())
        host.append(out)
    return host


def attention_probe(torch, rec, cache, condition, current, query, chunk_sizes, phase):
    from taomate_h3.inference.metal_attention import query_chunked_sdpa
    layer_name = 'blocks.0.attn'
    results = []
    candidates = []
    scale = 1/math.sqrt(128)
    # All query variants see exactly the same condition/history/current keys.
    for chunk in chunk_sizes:
        began = time.monotonic()
        out = cache.attention_with_history(layer_name,query,condition,current,scale=scale,
                query_chunk_size=chunk,max_joined_bytes=3*GIB)
        torch.mps.synchronize()
        elapsed = time.monotonic()-began
        rec.memory(torch,'cache_attention_output_live',phase=phase,query_chunk=chunk,
                   complete_key_count=len(condition.key)+cache.history_tokens+len(current.key))
        candidates.append(out.cpu())
        del out
        rec.memory(torch,'cache_attention_output_released',phase=phase,query_chunk=chunk)
        results.append({'chunk':chunk,'instrumented_seconds':elapsed})
    # This is a layout/assembly oracle for quantized-history semantics, not an
    # unquantized BF16 cache quality comparison. Full keys remain in every softmax.
    cpu_key, cpu_value = independent_dense_cpu(torch,cache,layer_name,condition,current)
    key, value = cpu_key.to('mps'), cpu_value.to('mps')
    del cpu_key, cpu_value
    rec.memory(torch,'independent_joined_reference_live',phase=phase)
    reference = query_chunked_sdpa(query,key,value,scale=scale,query_chunk_size=chunk_sizes[0])
    expected = reference.cpu()
    del reference, key, value
    for result, candidate in zip(results,candidates):
        result['vs_independent_dense_quantized_reference'] = compare_cpu(torch,candidate,expected)
        if not result['vs_independent_dense_quantized_reference']['finite']:
            raise RuntimeError('Nonfinite attention output')
    rec.memory(torch,'attention_probe_reference_released',phase=phase)
    return results


def quantization_sample(torch, cache, packed_cpu, shape, layer_name):
    pair = cache._staged[layer_name]
    n = shape['video_tokens']+shape['audio_tokens']
    picks = sorted(set([0,1,shape['audio_tokens']-1,shape['audio_tokens'],n//2,n-1]))
    idx = torch.tensor(picks,dtype=torch.long)
    source = packed_cpu[shape['condition_tokens']:shape['unpadded_tokens']].index_select(0,idx).float()
    scale = (source.abs().amax(-1,keepdim=True)/127).clamp_min(1e-12)
    quantized = (source/scale).round().clamp(-127,127).to(torch.int8)
    actual_q = pair.key.index_select(0,idx.to('mps')).cpu()
    actual_s = pair.key_scale.index_select(0,idx.to('mps')).cpu()
    return {'rows':picks,'scope':'sampled key rows; all staged input values separately checked finite by production stage()',
            'int8_sha256':tensor_digest(torch,actual_q),'scale_sha256':tensor_digest(torch,actual_s),
            'int8_exact':bool(torch.equal(actual_q,quantized)),
            'scales':compare_cpu(torch,actual_s,scale),
            'restored_vs_original':compare_cpu(torch,(actual_q.float()*actual_s).bfloat16(),source.bfloat16())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument('--phases',type=int,default=1,choices=range(1,5))
    parser.add_argument('--kv-tile',type=int,default=1024)
    parser.add_argument('--cache-allocation',choices=('per_layer','packed8'),default='per_layer',
                        help='Production allocation policy; packed8 requires its explicitly pinned source commit.')
    parser.add_argument('--condition-tokens',type=int,default=370)
    parser.add_argument('--interleave-linear',choices=('none','fc1','qkv'),default='none')
    parser.add_argument('--weight-tile-mib',type=int,default=64)
    parser.add_argument('--probe-queries',type=int,default=0,help='0 disables attention; 1..256 checks all full keys for a limited query slice, one layer/phase.')
    parser.add_argument('--query-chunks',default='64,128,256')
    args = parser.parse_args()
    chunks = [int(x) for x in args.query_chunks.split(',')]
    if not all(1<=x<=256 for x in chunks) or not 0<=args.probe_queries<=256:
        parser.error('query chunks 1..256, probe-queries 0..256')
    if not 1<=args.kv_tile<=1024 or not 1<=args.weight_tile_mib<=256 or not 0<=args.condition_tokens<=1024:
        parser.error('kv-tile 1..1024, weight-tile-mib 1..256, condition-tokens 0..1024')
    shapes = geometry(args.repository,args.phases,args.condition_tokens)
    plan = {'case':'B_quantized_cache', 'canvas':[1088,1920], 'heads':56,'head_dim':128,'layers':50,
            'shapes':shapes,'kv_tile_tokens':args.kv_tile,'cache_allocation':args.cache_allocation,
            'interleave_linear':args.interleave_linear,
            'resident_control_gib':args.resident_gib,'seed':args.seed,'probe_queries':args.probe_queries,'query_chunks':chunks,
            'input_kind':'one deterministic synthetic BF16 packed K/V pair per phase; values reused across all 50 layers, which own distinct INT8 storage',
            'interleave_scope':'optional single real matrix after each stage; omits QKV/MLP/LoRA/attention remainder. Allocation stress diagnostic, not a full DiT forward.',
            'attention_scope':'optional first-layer limited queries; full condition+visible history+current keys in one softmax; CPU independently rebuilt quantized-history reference',
            'limits':'75 GiB whole-PID external guard; 73 GiB sampled driver stop; cache payload arithmetic is not process peak'}
    if args.plan:
        print(json.dumps(plan,indent=2)); return
    # Estimate lower-bound payload before allocating; resident weight control plus
    # complete staged phases can alone exceed the guard. Leave 8 GiB working room.
    if sum(x['physical_commit_payload_bytes'] for x in shapes)+args.resident_gib*GIB>67*GIB:
        parser.error('Planned resident + pre-trim physical KV exceeds 67 GiB; lower resident control or phase count.')
    rec = Recorder(args,plan)
    try:
        # Import/inspect only; do not allocate GPU control/weights on an unsupported API.
        sys.path.insert(0,str(args.repository/'src'))
        from taomate_h3.streaming.metal_kv import QuantizedCleanAVKVCache
        supports_allocation='allocation' in inspect.signature(QuantizedCleanAVKVCache).parameters
        if args.cache_allocation!='per_layer' and not supports_allocation:
            raise RuntimeError('Pinned cache constructor does not support allocation=packed8')
        torch = initialize(args,rec)
        from taomate_h3.streaming.cache import KVContract, AVKV, MAIN_LAYER_NAMES
        from taomate_h3.inference.metal_loader import metal_linear
        with torch.inference_mode():
            resident = resident_control(torch,rec,args.resident_gib)
            owner = None if args.interleave_linear=='none' else load_real_linear(torch,rec,args.weights,args.interleave_linear,args.weight_tile_mib*MIB)
            kwargs={'tile_tokens':args.kv_tile}
            if supports_allocation:
                kwargs['allocation']=args.cache_allocation
            cache = QuantizedCleanAVKVCache(KVContract(56,128,torch.bfloat16,'mps'),**kwargs)
            rec.event('cache_constructed',class_name=type(cache).__name__,constructor_kwargs=kwargs,cache=cache.receipt())
            phases = []
            for shape in shapes:
                phase = shape['phase']
                torch.accelerator.reset_peak_memory_stats('mps')
                began = time.monotonic()
                key_cpu = cpu_normal_bf16(torch,(shape['packed_tokens'],56,128),args.seed+phase*10)
                value_cpu = cpu_normal_bf16(torch,(shape['packed_tokens'],56,128),args.seed+phase*10+1)
                key,value = key_cpu.to('mps'),value_cpu.to('mps')
                del value_cpu
                tags_cpu = torch.full((shape['packed_tokens'],),3,dtype=torch.int8)
                tags_cpu[:args.condition_tokens]=1
                tags_cpu[args.condition_tokens:args.condition_tokens+shape['audio_tokens']]=2
                tags_cpu[args.condition_tokens+shape['audio_tokens']:shape['unpadded_tokens']]=0
                tags = tags_cpu.to('mps')
                mask = ((tags_cpu==0)|(tags_cpu==2)).to('mps')
                linear_input = None
                if owner is not None:
                    linear_input = cpu_normal_bf16(torch,(shape['packed_tokens'],5376),args.seed+phase*10+2).to('mps')
                rec.memory(torch,'phase_inputs_live',phase=phase,cache=cache.receipt())
                probes = []
                if args.probe_queries:
                    condition = AVKV(key[:args.condition_tokens],value[:args.condition_tokens])
                    current = AVKV(key[args.condition_tokens:shape['unpadded_tokens']],value[args.condition_tokens:shape['unpadded_tokens']])
                    query = cpu_normal_bf16(torch,(args.probe_queries,56,128),args.seed+phase*10+3).to('mps')
                    probes = attention_probe(torch,rec,cache,condition,current,query,chunks,phase)
                    del condition,current,query
                cache.begin_clean_commit(phase)
                qchecks = []
                first_linear = last_linear = None
                for layer, name in enumerate(MAIN_LAYER_NAMES):
                    layer_started=time.monotonic()
                    cache.stage(name,key,value,tags,mask)
                    rec.memory(torch,'layer_staged',phase=phase,layer=layer,cache=cache.receipt(),stage_seconds=time.monotonic()-layer_started)
                    if layer in (0,49):
                        check=quantization_sample(torch,cache,key_cpu,shape,name)
                        check['layer']=layer
                        qchecks.append(check)
                        if not check['int8_exact'] or not check['scales']['finite']:
                            raise RuntimeError('Sampled real-shape key quantization mismatch/nonfinite')
                    if owner is not None:
                        output=metal_linear(linear_input,owner)
                        rec.memory(torch,'interleaved_linear_output_live',phase=phase,layer=layer)
                        if layer in (0,49):
                            # Only sampled output distances; memory study does not
                            # claim full-output or official-CUDA numerical parity.
                            selected=torch.linspace(0,len(output)-1,64,dtype=torch.long,device='cpu')
                            sampled=output.index_select(0,selected.to('mps')).cpu()
                            if layer==0: first_linear=sampled
                            else: last_linear=sampled
                            del sampled,selected
                        del output
                        rec.memory(torch,'interleaved_linear_output_released',phase=phase,layer=layer)
                cache.commit()
                rec.memory(torch,'phase_committed_before_trim',phase=phase,cache=cache.receipt())
                cache.retain_sink_and_recent_commits()
                rec.memory(torch,'phase_trimmed',phase=phase,cache=cache.receipt())
                sample_difference=None if owner is None else compare_cpu(torch,last_linear,first_linear)
                if sample_difference is not None and not sample_difference['finite']:
                    raise RuntimeError('Nonfinite interleaved linear sample')
                phase_result={'phase':phase,'elapsed_seconds_including_input_and_validation':time.monotonic()-began,
                              'cache':cache.receipt(),'quantization_samples':qchecks,'attention_probe':probes,
                              'linear_first_64_row_sample_sha256':None if first_linear is None else tensor_digest(torch,first_linear),
                              'linear_last_64_row_sample_sha256':None if last_linear is None else tensor_digest(torch,last_linear),
                              'linear_last_vs_first_64_row_sample':sample_difference}
                phases.append(phase_result)
                rec.receipt['phases']=phases
                rec.flush()
                del key,value,key_cpu,tags,mask,tags_cpu,linear_input,first_linear,last_linear
                # Match production phase-end sync+empty_cache. Persisted cache remains.
                release_cache(torch,rec,f'phase_{phase}_inputs_released')
            cache.clear()
            del cache,owner,resident
            release_cache(torch,rec,'all_cache_and_controls_released')
            rec.finish(phases=phases)
    except BaseException as error:
        rec.finish(error=error)
        raise


if __name__=='__main__':
    main()
