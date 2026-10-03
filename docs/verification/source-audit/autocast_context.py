#!/usr/bin/env python3
"""Inspect autocast TLS flags only. No tensors, model load, or GPU operations."""
import json
from pathlib import Path
import torch

result = {'torch': torch.__version__, 'before_mps_autocast': torch.is_autocast_enabled('mps')}
with torch.autocast('mps', dtype=torch.float16):
    result['outer_mps_autocast'] = torch.is_autocast_enabled('mps')
    with torch.autocast('cuda', enabled=False):
        result['inside_cuda_disabled_mps_autocast'] = torch.is_autocast_enabled('mps')
        result['inside_cuda_disabled_cuda_autocast'] = torch.is_autocast_enabled('cuda')
result['scope'] = 'Autocast context flags only; no tensors/GPU operations/model loading'
assert result['inside_cuda_disabled_mps_autocast'] is True
path = Path(__file__).with_name('autocast-context.json')
path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
