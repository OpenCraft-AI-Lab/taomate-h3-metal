#!/usr/bin/env python3
"""Read only safetensors JSON headers; no tensor payloads/dependencies/GPU."""
import collections
import json
import math
import re
import struct
from fractions import Fraction
from pathlib import Path

BASE = Path('/Users/supreme-paca/Projects/h3-metal/MiniMax-H3/FL2VA')
ADAPTER = Path('/Users/supreme-paca/Models/TaoMate-H3/adapter_model.safetensors')
DTYPES = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}

def read_header(path):
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        assert 0 < length < 16 * 1024 * 1024
        data = json.loads(f.read(length))
    return {k: v for k, v in data.items() if k != '__metadata__'}, length + 8

def selected_text(name):
    if name == 'model.language_model.embed_tokens.weight': return True
    m = re.match(r'^model\.language_model\.layers\.(\d+)\.', name)
    return bool(m and int(m.group(1)) < 50)

def inventory(paths, keep=lambda name: True):
    kinds = collections.defaultdict(lambda: {'tensors': 0, 'elements': 0, 'bytes': 0})
    tensors, largest, read_bytes = {}, [], 0
    for path in paths:
        header, consumed = read_header(path)
        read_bytes += consumed
        for name, value in header.items():
            if not keep(name): continue
            assert name not in tensors, (path, name)
            shape, dtype = value['shape'], value['dtype']
            elements = math.prod(shape)
            count = elements * DTYPES[dtype]
            assert value['data_offsets'][1] - value['data_offsets'][0] == count
            kinds[dtype]['tensors'] += 1
            kinds[dtype]['elements'] += elements
            kinds[dtype]['bytes'] += count
            tensors[name] = {'dtype': dtype, 'shape': shape, 'bytes': count}
            largest.append({'name': name, 'shape': shape, 'dtype': dtype, 'bytes': count})
    return {'tensor_count': len(tensors), 'tensor_bytes': sum(t['bytes'] for t in tensors.values()),
            'tensor_gib': sum(t['bytes'] for t in tensors.values()) / 2**30,
            'by_dtype': dict(kinds), 'header_bytes_read': read_bytes,
            'largest': sorted(largest, key=lambda x: x['bytes'], reverse=True)[:8]}, tensors

def main():
    result = {'scope': 'Only safetensors 8-byte prefix and JSON header read. No tensor data loaded. Memory arithmetic is a lower-bound estimate, not observed resident/peak usage.'}
    records = {}
    for name, paths, keep in [
        ('transformer', sorted((BASE/'transformer').glob('*.safetensors')), lambda n: True),
        ('text_selected_50_layers', sorted((BASE/'text_encoder').glob('*.safetensors')), selected_text),
        ('video_vae', [BASE/'video_vae/source/model.safetensors'], lambda n: True),
        ('audio_vae', [BASE/'audio_vae/model.safetensors'], lambda n: True),
        ('adapter_f32_checkpoint', [ADAPTER], lambda n: True),
    ]:
        result[name], records[name] = inventory(paths, keep)
    tb = result['transformer']['tensor_bytes']
    xb = result['text_selected_50_layers']['tensor_bytes']
    ab = result['adapter_f32_checkpoint']['tensor_bytes']//2
    result['arithmetic'] = {
        'text_plus_transformer_gib': (tb+xb)/2**30,
        'adapter_bf16_gib': ab/2**30,
        'transformer_plus_bf16_adapter_gib': (tb+ab)/2**30,
        'all_f32_transformer_gib_if_upcast': sum(math.prod(t['shape'])*4 for t in records['transformer'].values())/2**30,
    }
    result['transformer_f32_tensors'] = {n: v for n,v in records['transformer'].items() if v['dtype']=='F32'}
    frame_rows = 405
    # 50 layers x K/V x 56 heads x 128 head_dim x BF16.
    bytes_per_token = 50*2*56*128*2
    commits = []
    cache = []
    for request in range(12):
        if request > 0 and request % 12 == 0:
            cache = [(v,0) for v,a in cache]
        global_start = 0 if request == 0 else 124+(request-1)*119
        frame = 0
        for phase, groups in enumerate((2,2,2,1)):
            frames = groups*17 + (5 if request == 0 and phase == 0 else 0)
            videos = groups*5 + (2 if request == 0 and phase == 0 else 0)
            start = round(Fraction((global_start+frame)*40,24))
            stop = round(Fraction((global_start+frame+frames)*40,24))
            audio = 2*(stop-start)
            frame += frames
            staged = videos*frame_rows + audio
            pre = sum(v+a for v,a in cache)
            cache.append((videos*frame_rows,audio))
            committed = pre + staged
            if len(cache)>2:
                retained = [(cache[0][0],0)] + cache[-2:]
            else: retained = cache[:]
            retained_count = sum(v+a for v,a in retained)
            commits.append({'request':request,'phase':phase,'history_before_tokens':pre,'staged_tokens':staged,
                'history_after_commit_tokens':committed,'history_retained_tokens':retained_count,
                'commit_plus_staging_gib':committed*bytes_per_token/2**30,
                'trim_old_plus_new_gib':(committed+retained_count)*bytes_per_token/2**30})
            cache = retained
    result['kv_480p'] = {'frame_rows':frame_rows,'bytes_per_history_token_all_50_layers':bytes_per_token,
         'max_retained_gib':max(c['history_retained_tokens'] for c in commits)*bytes_per_token/2**30,
         'max_pretrim_gib':max(c['commit_plus_staging_gib'] for c in commits),
         'max_old_plus_new_trim_gib':max(c['trim_old_plus_new_gib'] for c in commits),
         'worst_trim':max(commits,key=lambda c:c['trim_old_plus_new_gib']),
         'note':'Derived from official cache.py:273–282 keeping old dict until all retained tensors built; ignores allocator/workspaces and input temporaries.'}
    destination = Path(__file__).with_name('header-inventory.json')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'result':str(destination),'components':{k:{p:result[k][p] for p in ('tensor_count','tensor_bytes','tensor_gib','by_dtype')} for k in records},'arithmetic':result['arithmetic'],'kv_480p':result['kv_480p']},indent=2))

if __name__ == '__main__': main()
