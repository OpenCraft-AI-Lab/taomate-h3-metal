"""Read-only TaoMate MPS diagnostics. This file never initializes MPS at import."""
from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

GIB = 1024**3
MIB = 1024**2
DEFAULT_REPO = Path('/Users/supreme-paca/Projects/taomate-h3-metal')
DEFAULT_WEIGHTS = Path('/Users/supreme-paca/Projects/h3-metal/MiniMax-H3/FL2VA/transformer')
DEFAULT_LOCK = Path('/Users/supreme-paca/Projects/h3-metal/outputs/original-baseline-20261002/heavy-gpu.lock')
DEFAULT_REF = 'dbb6992'


def common_arguments(parser):
    parser.add_argument('--repository', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--source-ref', default=DEFAULT_REF)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--resident-gib', type=float, default=0)
    parser.add_argument('--guarded', action='store_true', help='Required: launch with scripts/run_guarded.py, footprint <=75 GiB.')
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--plan', action='store_true', help='Print CPU-only plan; does not import torch or initialize MPS.')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_inventory(repository, ref):
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(repository), *arguments])
    resolved = git('rev-parse', ref).decode().strip()
    expected = git('ls-tree', '-r', '--name-only', resolved, 'src').decode().splitlines()
    hashes = {}
    for name in expected:
        if not name.endswith('.py'):
            continue
        expected_hash = sha(git('show', f'{resolved}:{name}'))
        current_hash = sha((repository / name).read_bytes())
        if expected_hash != current_hash:
            raise RuntimeError(f'Source differs from pinned {resolved}: {name}')
        hashes[name] = current_hash
    scripts = {p.name: sha(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}
    return {'source_ref': resolved, 'head': git('rev-parse', 'HEAD').decode().strip(),
            'src_sha256': hashes, 'diagnostic_sha256': scripts}


def require_guard(args):
    # The official launcher owns the lock and starts this child in a new session.
    # These checks are defense in depth, not proof of another process's ownership.
    if not args.guarded or os.getpgrp() != os.getpid():
        raise RuntimeError('Run via the official run_guarded.py with --guarded, not directly.')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise RuntimeError('CPU fallback must be disabled by the guard.')
    high = float(os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'nan'))
    low = float(os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO', 'nan'))
    if high != 0.9 or not 0 < low <= 0.8:
        raise RuntimeError('Expected HIGH=0.9 and 0<LOW<=0.8 from guard.')
    with args.lock.open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
            raise RuntimeError('Shared heavy-GPU lock is not held by launcher.')
    if not 0 <= args.resident_gib <= 30:
        raise ValueError('Synthetic resident control must be within 0..30 GiB.')


class Recorder:
    def __init__(self, args, plan, *, driver_limit_gib=73):
        if args.output_dir is None:
            raise ValueError('--output-dir required for execution')
        self.path = args.output_dir.resolve()
        if not self.path.is_relative_to(Path('/private/tmp')):
            raise ValueError('Diagnostic output must remain under /private/tmp')
        self.path.mkdir(parents=True, exist_ok=False)
        self.stream = (self.path / 'steps.jsonl').open('x', buffering=1)
        self.started = time.monotonic()
        self.driver_limit_bytes = int(driver_limit_gib*GIB)
        self.peak_samples = {'allocated_bytes': 0, 'reserved_bytes': 0, 'driver_bytes': 0}
        self.receipt = {'complete': False, 'plan': plan, 'argv': sys.argv,
                        'command': [sys.executable, *sys.argv], 'pid': os.getpid(),
                        'platform': platform.platform(), 'internal_sampled_driver_limit_bytes':self.driver_limit_bytes,
                        'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                        'environment': {k: v for k,v in os.environ.items() if k.startswith(('TAOMATE_MPS_', 'PYTORCH_MPS_')) or k == 'PYTORCH_ENABLE_MPS_FALLBACK'},
                        'memory_meanings': {'A': 'active PyTorch allocator buffers', 'R': 'all PyTorch MTL heaps',
                            'D': 'all Metal driver allocations', 'R-A': 'unused space in PyTorch heaps; not alone proof of fragmentation',
                            'D-R': 'outside PyTorch heap inventory; not alone proof of MPSGraph',
                            'peaks': 'A/R exact allocator high-water marks; D sampled only; whole-PID kernel peak is outer guard process-receipt.json'},
                        **source_inventory(args.repository, args.source_ref)}
        self.flush()

    def flush(self):
        target = self.path / 'receipt.json'
        temp = target.with_suffix('.json.tmp')
        temp.write_text(json.dumps(self.receipt, indent=2, allow_nan=False) + '\n')
        temp.replace(target)

    def event(self, label, **data):
        row = {'event': label, 'elapsed_seconds': time.monotonic() - self.started, **data}
        self.stream.write(json.dumps(row, allow_nan=False) + '\n')
        return row

    def memory(self, torch, label, *, synchronize=True, **data):
        if synchronize:
            torch.mps.synchronize()
        stats = torch.accelerator.memory_stats('mps')
        required = ('allocated_bytes.all.current', 'allocated_bytes.all.peak',
                    'reserved_bytes.all.current', 'reserved_bytes.all.peak')
        missing = [k for k in required if k not in stats]
        if missing:
            raise RuntimeError(f'MPS A/R statistics unavailable: {missing}')
        values = {'allocated_bytes': int(stats[required[0]]),
                  'allocated_peak_bytes': int(stats[required[1]]),
                  'reserved_bytes': int(stats[required[2]]),
                  'reserved_peak_bytes': int(stats[required[3]]),
                  'driver_bytes': int(torch.mps.driver_allocated_memory()),
                  'mps_current_allocated_bytes': int(torch.mps.current_allocated_memory())}
        values['reserved_minus_allocated_bytes'] = values['reserved_bytes'] - values['allocated_bytes']
        values['driver_minus_reserved_bytes'] = values['driver_bytes'] - values['reserved_bytes']
        for k in self.peak_samples:
            self.peak_samples[k] = max(self.peak_samples[k], values[k])
        self.receipt['sampled_maxima_bytes'] = self.peak_samples.copy()
        self.receipt['latest_memory'] = values
        row = self.event(label, synchronized=synchronize, **values, **data)
        self.flush()
        if values['driver_bytes'] > self.driver_limit_bytes:
            raise RuntimeError(f'Diagnostic driver allocation exceeded {self.driver_limit_bytes/GIB:g} GiB; external footprint guard remains active.')
        return row

    def finish(self, *, error=None, **data):
        self.receipt.update(complete=error is None, elapsed_seconds=time.monotonic()-self.started, **data)
        if error is not None:
            self.receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
        self.flush()
        self.stream.close()


def initialize(args, recorder):
    require_guard(args)
    sys.path.insert(0, str(args.repository / 'src'))
    import torch
    torch.set_num_threads(4)
    if torch.__version__ != '2.14.1':
        raise RuntimeError(f'Expected reviewed torch 2.14.1; got {torch.__version__}')
    recorder.receipt.update(torch_version=torch.__version__, torch_git_version=torch.version.git_version)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    tiny = torch.empty(1, device='mps')
    del tiny
    torch.mps.synchronize()
    torch.mps.empty_cache()
    recorder.memory(torch, 'initialized')
    return torch


def resident_control(torch, recorder, gib):
    # A controlled live payload, not an imitation of the model's parameter layout.
    resident = []
    remaining = int(gib * GIB)
    while remaining:
        size = min(256*MIB, remaining)
        block = torch.empty(size, dtype=torch.uint8, device='mps')
        block.zero_()  # Touch pages so outer footprint observes the control.
        resident.append(block)
        remaining -= size
        if len(resident) % 16 == 0:
            recorder.memory(torch, 'resident_control_growing', blocks=len(resident))
    recorder.memory(torch, 'resident_control_ready', payload_bytes=int(gib*GIB), blocks=len(resident))
    return resident


def cpu_normal_bf16(torch, shape, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    # CPU F32 source and final CPU BF16 tensor. Never uses a device RNG.
    return torch.randn(shape, generator=generator, dtype=torch.float32).bfloat16()


def load_real_linear(torch, recorder, weights, kind, tile_bytes):
    from safetensors import safe_open
    from taomate_h3.distributed.linear import MergedColumnParallelLinear
    from taomate_h3.distributed.parallel import ParallelContext
    from taomate_h3.inference.metal_loader import load_metal_stream_weight_
    from taomate_h3.model.weight_loading import reorder_grouped_qkv_to_qkv
    key = 'blocks.2.mlp.fc1.weight' if kind == 'fc1' else 'blocks.2.attn.qkv_proj.weight'
    sizes = (14336, 14336) if kind == 'fc1' else (7168, 7168, 7168)
    index_path = weights / 'model.safetensors.index.json'
    index_bytes = index_path.read_bytes()
    shard = weights / json.loads(index_bytes)['weight_map'][key]
    with safe_open(shard, framework='pt', device='cpu') as file:
        source = file.get_tensor(key)
        if source.dtype != torch.bfloat16 or tuple(source.shape) != (sum(sizes), 5376):
            raise RuntimeError('Real checkpoint header shape/dtype differs')
        with torch.device('meta'):
            owner = MergedColumnParallelLinear(5376, sizes, bias=False, gather_output=False,
                    dtype=torch.bfloat16, context=ParallelContext(1,0,1,0,None,None))
        if kind == 'qkv':
            source = reorder_grouped_qkv_to_qkv(source, num_query_groups=56, heads_per_group=1, head_dim=128)
        source = owner.full_weight_slice(source)
        owner.weight = torch.nn.Parameter(torch.empty(source.shape, dtype=torch.int8, device='mps'), requires_grad=False)
        owner.register_buffer('_metal_weight_scale', torch.empty((source.shape[0],1), dtype=torch.float32,device='mps'),persistent=False)
        owner.register_buffer('_metal_native_scale', None, persistent=False)
        owner._metal_stream_load_pending = True
        owner._metal_stream_loaded = False
        owner._metal_quantization_backend = 'dequantize'
        owner._metal_temporary_bytes = tile_bytes
        load_metal_stream_weight_(owner, source)
        del source
    recorder.event('real_weight_loaded', key=key, shard=str(shard), index_sha256=sha(index_bytes),
                   shape=list(owner.weight.shape), stored_dtype='INT8 + F32 row scale',
                   qkv_reordered=(kind=='qkv'),
                   provenance='Local already-verified checkpoint. This script hashes the index, not the full shard again.')
    recorder.memory(torch, 'weight_loaded')
    return owner


def compare_cpu(torch, actual, expected):
    if actual.device.type != 'cpu' or expected.device.type != 'cpu' or actual.shape != expected.shape:
        raise ValueError('Comparison requires same-shaped CPU tensors')
    a = actual.reshape(actual.shape[0], -1)
    b = expected.reshape(expected.shape[0], -1)
    maximum, delta2, reference2, count, unequal = 0.0, 0.0, 0.0, 0, 0
    finite = True
    for start in range(0, len(a), 32):
        x, y = a[start:start+32].double(), b[start:start+32].double()
        good = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        finite = finite and good
        if not good:
            continue
        diff = x-y
        maximum = max(maximum, float(diff.abs().max()))
        delta2 += float((diff*diff).sum())
        reference2 += float((y*y).sum())
        unequal += int((x != y).sum())
        count += x.numel()
    return {'finite': finite, 'max_abs': maximum if finite else None,
            'rel_l2': math.sqrt(delta2/max(reference2,1e-300)) if finite else None,
            'exact': finite and unequal == 0, 'unequal_elements': unequal, 'compared_elements': count}


def release_cache(torch, recorder, label):
    gc.collect()
    recorder.memory(torch, label + '_before_empty_cache')
    torch.mps.empty_cache()
    recorder.memory(torch, label + '_after_empty_cache')
