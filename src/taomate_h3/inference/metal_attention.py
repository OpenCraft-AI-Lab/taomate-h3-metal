# Modified for TaoMate-H3 Metal inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Bound attention workspace by slicing queries, retaining every key per softmax."""
from __future__ import annotations

import torch
from torch.nn.functional import scaled_dot_product_attention


@torch.no_grad()
def query_chunked_sdpa(query, key, value, *, scale, query_chunk_size):
    if isinstance(query_chunk_size, bool) or query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("query and KV must be NHD tensors")
    if query.shape[1:] != key.shape[1:]:
        raise ValueError("query and KV head shapes differ")
    output = torch.empty_like(query)
    for start in range(0, query.shape[0], query_chunk_size):
        stop = min(start + query_chunk_size, query.shape[0])
        chunk = scaled_dot_product_attention(
            query[start:stop].transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0), value.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0, is_causal=False, scale=scale,
        )
        output[start:stop].copy_(chunk[0].transpose(0, 1))
        if query.device.type == "mps":
            # Complete each query tile before its workspace can be reused.
            torch.mps.synchronize()
        del chunk
    return output
