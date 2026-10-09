# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Independent full-key softmax oracle for bounded query workspace."""
import math

import pytest
import torch

from taomate_h3.inference.metal_attention import query_chunked_sdpa


@pytest.mark.parametrize("device", ("cpu", "mps"))
@pytest.mark.parametrize("query_chunk_size", (1, 7, 23))
@torch.no_grad()
def test_every_query_tile_sees_all_keys(device, query_chunk_size):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    generator = torch.Generator().manual_seed(1904)
    q = torch.randn(23, 2, 128, generator=generator).bfloat16()
    k, v = [torch.randn(37, 2, 128, generator=generator).bfloat16() for _ in range(2)]
    scale = 1 / math.sqrt(128)
    scores = torch.einsum("qhd,khd->hqk", q.double(), k.double()) * scale
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.double()).float()
    actual = query_chunked_sdpa(q.to(device), k.to(device), v.to(device),
                               scale=scale, query_chunk_size=query_chunk_size).cpu().float()
    torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.008)
    # A strong value in the final key must affect even the first query tile.
    changed = v.clone()
    changed[-1] = 1000
    second = query_chunked_sdpa(q.to(device), k.to(device), changed.to(device),
                               scale=scale, query_chunk_size=query_chunk_size).cpu().float()
    assert (second[:1] - actual[:1]).abs().max() > 1
