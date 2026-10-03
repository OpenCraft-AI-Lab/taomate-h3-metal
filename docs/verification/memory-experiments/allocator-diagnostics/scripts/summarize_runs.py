"""CPU-only aggregation of the guarded allocation diagnostics."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path('/private/tmp/taomate-metal-diagnostic-runs')
GIB=1024**3


def main():
    runs=[]
    events={}
    for p in sorted(ROOT.iterdir()):
        if not p.is_dir() or not (p/'process-receipt.json').exists():
            continue
        guard=json.loads((p/'process-receipt.json').read_text())
        rec=json.loads((p/'diagnostic/receipt.json').read_text())
        rows=[json.loads(s) for s in (p/'diagnostic/steps.jsonl').read_text().splitlines()]
        memory=[r for r in rows if 'allocated_bytes' in r]
        events[p.name]=memory
        before=guard.get('preflight',{}).get('system_swap_bytes')
        after=guard.get('postflight',{}).get('system_swap_bytes')
        result={'name':p.name,'path':str(p),'complete':rec['complete'],
                'returncode':guard['returncode'],'guard_reason':guard['guard_reason'],
                'error':rec.get('error'),'source_ref':rec['source_ref'],
                'torch_version':rec.get('torch_version'),'torch_git_version':rec.get('torch_git_version'),
                'guard_environment':guard['environment'],'limits':guard['limits'],
                'wall_seconds':guard['wall_seconds'],
                'kernel_peak_footprint_bytes':guard['kernel_peak_footprint_bytes'],
                'swap_delta_bytes':None if before is None or after is None else after-before,
                'allocated_peak_bytes':max(x['allocated_peak_bytes'] for x in memory),
                'reserved_peak_bytes':max(x['reserved_peak_bytes'] for x in memory),
                'driver_sampled_peak_bytes':max(x['driver_bytes'] for x in memory),
                'max_sampled_driver_minus_reserved_bytes':max(x['driver_minus_reserved_bytes'] for x in memory),
                'plan':rec['plan'],
                'last_memory':memory[-1],
                'phase_end_samples':[x for x in memory if x['event'] in ('phase_committed_before_trim','phase_trimmed') or x['event'].endswith('_inputs_released_after_empty_cache')],
                'layout_trials':rec.get('layout_trials'),'variant_vs_production':rec.get('variant_vs_production'),
                'phases':rec.get('phases')}
        runs.append(result)
    matched=[]
    for prefix in ('B-phase1-fc1-resident28','B-phase2-fc1-resident28'):
        left,right=events.get(prefix+'-low08',[]),events.get(prefix+'-low03',[])
        def index(xs):
            return {(x['event'],x.get('phase'),x.get('layer')):x for x in xs
                    if 'layer' in x or x['event']=='phase_committed_before_trim'}
        one,two=index(left),index(right)
        for key in one.keys() & two.keys():
            a,b=one[key],two[key]
            matched.append({'pair':prefix,'event':key[0],'phase':key[1],'layer':key[2],
                            'allocated_equal':a['allocated_bytes']==b['allocated_bytes'],
                            'low08_allocated_bytes':a['allocated_bytes'],'low03_allocated_bytes':b['allocated_bytes'],
                            'low08_reserved_bytes':a['reserved_bytes'],'low03_reserved_bytes':b['reserved_bytes'],
                            'reserved_saving_bytes':a['reserved_bytes']-b['reserved_bytes'],
                            'low08_driver_bytes':a['driver_bytes'],'low03_driver_bytes':b['driver_bytes']})
    matched.sort(key=lambda x:(x['pair'],x['phase'] if x['phase'] is not None else -1,x['layer'] if x['layer'] is not None else -1,x['event']))
    result={'runs':runs,'matched_steps':matched,
            'limits':['Synthetic controlled allocation diagnostics, not full-model inference.',
                'Uniform dummy layout is not the model parameter layout.',
                'A/R are allocator counters; D peak is sampled; whole-PID footprint peak comes from the external guard.',
                'Failed two-phase runs cannot be compared as equal completed workloads by their peaks alone; use matched_steps.',
                'First five runs use the preserved v1 scripts; subsequent cache scripts add CPU sample digests only.',
                'B audio/video cache tensors are synthetic; numerical distances do not measure generative quality.']}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in runs:
        print(r['name'],json.dumps({'complete':r['complete'],'wall_seconds':r['wall_seconds'],
            'pid_peak_GiB':r['kernel_peak_footprint_bytes']/GIB,
            'A_peak_GiB':r['allocated_peak_bytes']/GIB,'R_peak_GiB':r['reserved_peak_bytes']/GIB,
            'D_minus_R_max_MiB':r['max_sampled_driver_minus_reserved_bytes']/1024**2,'swap_delta':r['swap_delta_bytes']}))


if __name__=='__main__':
    main()
