"""CPU-only report builder for the four ffa5c0b allocation/SDPA controls."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/private/tmp/taomate-metal-diagnostic-runs')
REPO=Path('/Users/supreme-paca/Projects/taomate-h3-metal')
REF='ffa5c0b800d4777782e82b2d892f175524f696c5'
GIB=1024**3
NAMES=(
    'B-phase2-packed8-fc1-resident28-low03',
    'B-phase2-per-layer-paired-fc1-resident28-low03',
    'C-SDPA-first-low03',
    'C-SDPA-history-low03',
)


def main():
    names=subprocess.check_output(['git','-C',str(REPO),'ls-tree','-r','--name-only',REF,'src']).decode().splitlines()
    expected={}
    for name in names:
        if name.endswith('.py'):
            pinned=subprocess.check_output(['git','-C',str(REPO),'show',f'{REF}:{name}'])
            actual=(REPO/name).read_bytes()
            assert actual==pinned,(name,'current source differs')
            expected[name]=hashlib.sha256(actual).hexdigest()
    results=[]
    events={}
    for name in NAMES:
        p=ROOT/name
        rec=json.loads((p/'diagnostic/receipt.json').read_text())
        guard=json.loads((p/'process-receipt.json').read_text())
        lines=[json.loads(line) for line in (p/'diagnostic/steps.jsonl').read_text().splitlines()]
        memory=[x for x in lines if 'allocated_bytes' in x]
        assert rec['source_ref']==REF
        assert rec['src_sha256']==guard['source_sha256']==expected
        assert guard['environment']['PYTORCH_MPS_LOW_WATERMARK_RATIO']=='0.3'
        assert guard['environment']['PYTORCH_MPS_HIGH_WATERMARK_RATIO']=='0.9'
        assert guard['environment']['PYTORCH_ENABLE_MPS_FALLBACK']=='0'
        cap=12 if name.startswith('C-') else 75
        assert guard['limits']['footprint_bytes']==cap*GIB
        swap=guard['postflight']['system_swap_bytes']-guard['preflight']['system_swap_bytes']
        assert swap<=0
        events[name]=memory
        result={'name':name,'path':str(p),'complete':rec['complete'],
                'source_ref':rec['source_ref'],'head_at_launch':rec['head'],
                'all_53_source_hashes_match_pinned_and_completion':True,
                'source_sha256':rec['src_sha256'],'diagnostic_sha256':rec['diagnostic_sha256'],
                'torch_version':rec['torch_version'],'torch_git_version':rec['torch_git_version'],
                'guard_environment':guard['environment'],'limits':guard['limits'],
                'guard_returncode':guard['returncode'],'guard_reason':guard['guard_reason'],
                'wall_seconds':guard['wall_seconds'],'kernel_peak_footprint_bytes':guard['kernel_peak_footprint_bytes'],
                'swap_delta_bytes':swap,'allocated_peak_bytes':max(x['allocated_peak_bytes'] for x in memory),
                'reserved_peak_bytes':max(x['reserved_peak_bytes'] for x in memory),
                'driver_sampled_peak_bytes':max(x['driver_bytes'] for x in memory),
                'maximum_sampled_driver_minus_reserved_bytes':max(x['driver_minus_reserved_bytes'] for x in memory),
                'plan':rec['plan'],'last_memory':memory[-1],
                'phases':rec.get('phases'),'chunk_results':rec.get('chunk_results')}
        if result['chunk_results']:
            chunks=result['chunk_results']
            assert all(t['vs_chunk64_reference']['finite'] and t['vs_chunk64_reference']['bitwise_equal']
                       and t['vs_chunk64_reference']['rel_l2']==0 for c in chunks for t in c['trials'])
            result['chunk64_over_chunk512_measured_speed_ratio']=chunks[0]['later_operator_median_seconds']/chunks[-1]['later_operator_median_seconds']
            result['chunk256_over_chunk512_measured_speed_ratio']=chunks[2]['later_operator_median_seconds']/chunks[-1]['later_operator_median_seconds']
            result['full_output_elements_each_trial']=chunks[0]['trials'][0]['vs_chunk64_reference']['compared_elements']
        results.append(result)
    packed,per_layer,first,history=results
    assert packed['complete'] and not per_layer['complete'] and first['complete'] and history['complete']
    # The only benchmark argument differing in the B pair is allocation/output-dir.
    plan_keys=set(packed['plan'])|set(per_layer['plan'])
    assert all(packed['plan'].get(k)==per_layer['plan'].get(k) for k in plan_keys-{'cache_allocation'})
    assert packed['diagnostic_sha256']==per_layer['diagnostic_sha256']
    p0=packed['phases'][0]
    b0=per_layer['phases'][0]
    assert p0['linear_first_64_row_sample_sha256']==b0['linear_first_64_row_sample_sha256']
    assert p0['linear_last_64_row_sample_sha256']==b0['linear_last_64_row_sample_sha256']
    quant_pair=[]
    for p,b in zip(p0['quantization_samples'],b0['quantization_samples']):
        same=(p['rows']==b['rows'] and p['int8_sha256']==b['int8_sha256'] and p['scale_sha256']==b['scale_sha256'])
        assert same
        quant_pair.append({'phase':0,'layer':p['layer'],'same_selected_rows':same,
                           'int8_sha256':p['int8_sha256'],'scale_sha256':p['scale_sha256']})
    def index(rows):
        return {(r['event'],r.get('phase'),r.get('layer')):r for r in rows if 'layer' in r or r['event']=='phase_committed_before_trim'}
    pi,bi=index(events[NAMES[0]]),index(events[NAMES[1]])
    matches=[]
    for key in pi.keys() & bi.keys():
        p,b=pi[key],bi[key]
        matches.append({'event':key[0],'phase':key[1],'layer':key[2],
            'packed8':{k:p[k] for k in ('allocated_bytes','allocated_peak_bytes','reserved_bytes','reserved_peak_bytes','driver_bytes','reserved_minus_allocated_bytes','driver_minus_reserved_bytes')},
            'per_layer':{k:b[k] for k in ('allocated_bytes','allocated_peak_bytes','reserved_bytes','reserved_peak_bytes','driver_bytes','reserved_minus_allocated_bytes','driver_minus_reserved_bytes')},
            'packed8_cache':p.get('cache'),'per_layer_cache':b.get('cache'),
            'reserved_reduction_bytes':b['reserved_bytes']-p['reserved_bytes']})
    matches.sort(key=lambda x:(x['phase'],-1 if x['layer'] is None else x['layer'],x['event']))
    summary={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source_ref':REF,'head_at_completion_check':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD']).decode().strip(),
             'source_file_count':len(expected),'all_receipt_current_and_pinned_source_hashes_equal':True,
             'gpu_jobs_complete_no_more_started':True,'runs':results,'matched_B_steps':matches,
             'matched_B_phase0_quantization_samples':quant_pair,
             'B_phase0_FC1_sample_sha256':p0['linear_first_64_row_sample_sha256'],
             'limits':['Synthetic operator/allocation experiments; no full-model inference or quality equivalence claim.',
                       'Packed8 completes more work than the guarded per_layer run, so compare matched phase/layer samples as well as whole-job peaks.',
                       'Matched packed8 A can be higher because its current group owns unfilled slices; receipts separate logical staged bytes and physical backing.',
                       'B numeric checks cover sampled key/scale/FC1 rows; no full-B-layer BF16/INT8 output identity claim.',
                       'SDPA compares every output element and bytes to chunk64, preserving full keys and per-query sync; it is not an independent CUDA oracle.',
                       'SDPA warm medians use two later trials in one fixed ascending-chunk order; they do not establish full-model speedup.',
                       'A/R are allocator counters; D is boundary-sampled; whole-process kernel footprint is from the independent guard.']}
    target=ROOT/'followup-ffa5c0b-summary.json'
    target.write_text(json.dumps(summary,indent=2)+'\n')
    print('source files',len(expected),'all four hashes/guard/outputs checked')
    for r in results:
        print(r['name'],r['complete'],round(r['wall_seconds'],4),
              {k:round(r[k]/GIB,6) for k in ('kernel_peak_footprint_bytes','allocated_peak_bytes','reserved_peak_bytes','driver_sampled_peak_bytes')})
    for m in matches:
        if m['event']=='interleaved_linear_output_live' and (m['phase'],m['layer']) in ((0,49),(1,44)):
            print('matched',m['phase'],m['layer'],{kind:{k:round(v/GIB,6) for k,v in m[kind].items()} for kind in ('packed8','per_layer')})


if __name__=='__main__':main()
