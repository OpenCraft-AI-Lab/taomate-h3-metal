# Modified for TaoMate-H3 Metal inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Opt-in INT8 clean KV storage; BF16 query/current and full-key softmax.

Immutable segments preserve atomic clean publication without copying 50 layers.
Row selectors retain hidden audio backing until segment eviction; receipts report
both physical and logically visible payload. This is lossy storage, not W8A8.
"""

from __future__ import annotations

import math
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch

from taomate_h3.inference.metal_attention import query_chunked_sdpa
from taomate_h3.streaming.cache import (
    AUDIO_TOKEN_TAG,
    AVKV,
    MAIN_LAYER_NAMES,
    VIDEO_TOKEN_TAG,
    CleanAVKVCache,
    KVContract,
)


@dataclass(frozen=True)
class _QuantizedPair:
    key: torch.Tensor
    value: torch.Tensor
    key_scale: torch.Tensor
    value_scale: torch.Tensor

    def bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.key, self.value, self.key_scale, self.value_scale)
        )


@dataclass(frozen=True)
class _Segment:
    layers: dict[str, _QuantizedPair]
    tags: tuple[int, ...]
    rows: tuple[int, ...]

    def video_only(self) -> _Segment:
        return _Segment(
            self.layers,
            self.tags,
            tuple(r for r in self.rows if self.tags[r] == VIDEO_TOKEN_TAG),
        )


MAX_KV_TILE_TOKENS = 2048
MAX_KV_TILE_SCRATCH_BYTES = 256 * 1024**2


def parse_kv_tile_tokens(value: str) -> int:
    """Strict environment option, independent of device availability/model loading."""
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError("TAOMATE_MPS_KV_TILE_TOKENS must be an integer in 1..2048")
    # Bound string length before int conversion, including pathological env input.
    if len(value) > 4 or not 1 <= int(value) <= MAX_KV_TILE_TOKENS:
        raise ValueError("TAOMATE_MPS_KV_TILE_TOKENS must be an integer in 1..2048")
    return int(value)


def kv_tile_scratch_bound(tile_tokens: int, heads: int, head_dim: int = 128) -> int:
    """Conservative live-tensor estimate, NOT measured process/driver peak.

    Five F32-equivalent tiles cover selected/quantization intermediates and
    replacement overlap; add scales, indices and validation workspace. Joined
    BF16 KV, persistent/staged INT8 payload and allocator caches are separate.
    Each tile still synchronizes before the next tile is enqueued.
    """
    return tile_tokens * (20 * heads * head_dim + 32 * heads + 8)


def parse_kv_allocation(value: str) -> str:
    if value not in ("per_layer", "packed8"):
        raise ValueError("TAOMATE_MPS_KV_ALLOCATION must be per_layer or packed8")
    return value


def _backing_bytes(tensors) -> int:
    # Views share storage; count each allocation once, including unused slices.
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def _pair_tensors(pairs):
    for pair in pairs:
        yield from (pair.key, pair.value, pair.key_scale, pair.value_scale)


class QuantizedCleanAVKVCache(CleanAVKVCache):
    """Single-owner inference cache with non-reentrant mutation/attention leases.

    A failed stage/commit leaves published history unchanged. Whole-request
    recovery remains the runtime's responsibility (MPS currently fails closed).
    GPU staging is bounded to one joined BF16 K/V layer plus token-tile scratch;
    no BF16 history is returned or stored. Clear cannot resurrect failed runtime.
    """

    def __init__(
        self, contract: KVContract, *, tile_tokens: int = 128, allocation: str = "per_layer"
    ) -> None:
        if contract.dtype != torch.bfloat16 or contract.head_dim != 128:
            raise ValueError("INT8 KV requires BF16 computation and head_dim=128")
        if contract.local_heads <= 0 or contract.device_type not in ("cpu", "mps"):
            raise ValueError("INT8 KV supports positive head count on CPU/MPS only")
        if (
            isinstance(tile_tokens, bool)
            or not isinstance(tile_tokens, int)
            or not 1 <= tile_tokens <= MAX_KV_TILE_TOKENS
        ):
            raise ValueError("tile_tokens must be an integer in 1..2048")
        scratch_bound = kv_tile_scratch_bound(tile_tokens, contract.local_heads, contract.head_dim)
        if scratch_bound > MAX_KV_TILE_SCRATCH_BYTES:
            raise ValueError("KV tile scratch estimate exceeds 256 MiB budget")
        self.allocation = parse_kv_allocation(allocation)
        self._packed_groups: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._packed_scales: torch.Tensor | None = None
        self.contract = contract
        self.tile_tokens = tile_tokens
        self.tile_scratch_estimated_bytes = scratch_bound
        self._segments: tuple[_Segment, ...] = ()
        self._staged: dict[str, _QuantizedPair] = {}
        self._staged_block: int | None = None
        self._metadata: tuple[tuple[int, ...], tuple[bool, ...]] | None = None
        self._tags: tuple[int, ...] = ()
        self._block_index = 0
        self._lease = threading.Lock()

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if not self._lease.acquire(blocking=False):
            raise RuntimeError("KV mutation/attention already active")
        try:
            yield
        finally:
            self._lease.release()

    def _sync(self) -> None:
        if self.contract.device_type == "mps":
            torch.mps.synchronize()

    @property
    def committed_blocks(self) -> int:
        return self._block_index

    @property
    def clean_commit_active(self) -> bool:
        return self._staged_block is not None

    @property
    def history_tokens(self) -> int:
        return sum(len(s.rows) for s in self._segments)

    @property
    def history_audio_tokens(self) -> int:
        return sum(s.tags[r] == AUDIO_TOKEN_TAG for s in self._segments for r in s.rows)

    @property
    def history_video_tokens(self) -> int:
        return sum(s.tags[r] == VIDEO_TOKEN_TAG for s in self._segments for r in s.rows)

    def history(self, layer_name: str) -> None:
        raise RuntimeError("quantized history must use attention_with_history()")

    def _validate_tensor(self, tensor: torch.Tensor, label: str) -> None:
        if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (
            self.contract.local_heads,
            128,
        ):
            raise ValueError(f"{label}: expected NHD tensor matching contract")
        if tensor.dtype != torch.bfloat16 or tensor.device.type != self.contract.device_type:
            raise ValueError(f"{label}: dtype/device differs from contract")
        for start in range(0, len(tensor), self.tile_tokens):
            if not torch.isfinite(tensor[start : start + self.tile_tokens]).all().item():
                raise ValueError(f"{label}: nonfinite values")

    def begin_clean_commit(self, block_index: int) -> None:
        with self._exclusive():
            if isinstance(block_index, bool) or not isinstance(block_index, int):
                raise TypeError("block_index must be an integer")
            if self.clean_commit_active or block_index != self._block_index:
                raise RuntimeError("invalid clean block sequence")
            self._staged_block = block_index
            self._staged = {}
            self._metadata = None
            self._tags = ()

    def _packed_destination(self, layer_name, count, device):
        layer = MAIN_LAYER_NAMES.index(layer_name)
        group, offset = divmod(layer, 8)
        if self._packed_scales is None:
            self._packed_scales = torch.empty(
                (2, len(MAIN_LAYER_NAMES), count, self.contract.local_heads, 1),
                dtype=torch.float32,
                device=device,
            )
        if group not in self._packed_groups:
            shape = (
                min(8, len(MAIN_LAYER_NAMES) - group * 8),
                count,
                self.contract.local_heads,
                128,
            )
            # Retain a successfully allocated key if value allocation fails:
            # local exception ownership is transient; published history is untouched.
            key = torch.empty(shape, dtype=torch.int8, device=device)
            value = torch.empty_like(key)
            self._packed_groups[group] = (key, value)
        key, value = self._packed_groups[group]
        return (
            key[offset],
            value[offset],
            self._packed_scales[0, layer],
            self._packed_scales[1, layer],
        )

    def _quantize_selected(self, tensor: torch.Tensor, indices: torch.Tensor, destination=None):
        shape = (len(indices), self.contract.local_heads, 128)
        if destination is None:
            quantized = torch.empty(shape, dtype=torch.int8, device=tensor.device)
            scales = torch.empty((*shape[:-1], 1), dtype=torch.float32, device=tensor.device)
        else:
            quantized, scales = destination
        for start in range(0, len(indices), self.tile_tokens):
            selected = tensor.index_select(0, indices[start : start + self.tile_tokens]).float()
            scale = (selected.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
            quantized[start : start + self.tile_tokens] = (
                (selected / scale).round().clamp(-127, 127).to(torch.int8)
            )
            scales[start : start + self.tile_tokens] = scale
            self._sync()
        return quantized, scales

    @torch.no_grad()
    def stage(
        self,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        token_tags: torch.Tensor,
        commit_mask: torch.Tensor,
    ) -> None:
        with self._exclusive():
            self._validate_layer_name(layer_name)
            if not self.clean_commit_active or layer_name in self._staged:
                raise RuntimeError("inactive or duplicate clean staging")
            self._validate_tensor(key, "key")
            self._validate_tensor(value, "value")
            if key.shape != value.shape or key.device != value.device:
                raise ValueError("key/value shape/device differs")
            if (
                token_tags.ndim != 1
                or commit_mask.ndim != 1
                or len(token_tags) != len(key)
                or len(commit_mask) != len(key)
                or commit_mask.dtype != torch.bool
                or token_tags.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
                or token_tags.device != key.device
                or commit_mask.device != key.device
            ):
                raise ValueError("packed metadata shape/dtype/device differs")
            metadata = (
                tuple(token_tags.cpu().tolist()),
                tuple(commit_mask.cpu().tolist()),
            )
            tags = tuple(t for t, keep in zip(*metadata) if keep)
            if set(tags) != {VIDEO_TOKEN_TAG, AUDIO_TOKEN_TAG}:
                raise ValueError("clean chunk must contain exactly audio/video media tags")
            if self._metadata is not None and metadata != self._metadata:
                raise ValueError("packed metadata differs between layers")
            self._metadata, self._tags = metadata, tags
            indices = torch.nonzero(commit_mask, as_tuple=False).flatten()
            if self.allocation == "packed8":
                qk, qv, sk, sv = self._packed_destination(layer_name, len(indices), key.device)
                self._quantize_selected(key, indices, (qk, sk))
                self._quantize_selected(value, indices, (qv, sv))
            else:
                qk, sk = self._quantize_selected(key, indices)
                qv, sv = self._quantize_selected(value, indices)
            pair = _QuantizedPair(qk, qv, sk, sv)
            self._sync()
            self._staged[layer_name] = pair
            self._metadata, self._tags = metadata, tags

    def commit(self) -> None:
        with self._exclusive():
            if not self.clean_commit_active or set(self._staged) != set(MAIN_LAYER_NAMES):
                raise RuntimeError("incomplete clean commit")
            segment = _Segment(dict(self._staged), self._tags, tuple(range(len(self._tags))))
            next_segments = self._segments + (segment,)
            self._sync()  # any asynchronous failure precedes publication
            self._segments = next_segments
            self._block_index += 1
            self._clear_staging()

    def _clear_staging(self) -> None:
        self._packed_groups = {}
        self._packed_scales = None
        self._staged = {}
        self._staged_block = None
        self._metadata = None
        self._tags = ()

    def rollback(self) -> None:
        with self._exclusive():
            self._clear_staging()

    def clear(self) -> None:
        with self._exclusive():
            self._segments = ()
            self._block_index = 0
            self._clear_staging()

    def retain_sink_and_recent_commits(self) -> None:
        with self._exclusive():
            if self.clean_commit_active:
                raise RuntimeError("cannot retain during clean staging")
            if len(self._segments) >= 3:
                retained = (self._segments[0].video_only(), *self._segments[-2:])
                self._segments = retained

    def drop_audio_history(self) -> int:
        with self._exclusive():
            if self.clean_commit_active:
                raise RuntimeError("cannot reset audio during clean staging")
            removed = self.history_audio_tokens
            retained = tuple(s.video_only() for s in self._segments)
            self._segments = retained
            return removed

    def receipt(self) -> dict[str, int | str]:
        per_token = self.contract.local_heads * (128 + 4) * 2
        return {
            "storage": "int8_per_token_per_head_f32_scale",
            "allocation": self.allocation,
            "visible_history_payload_bytes": self.history_tokens * per_token * 50,
            "physical_history_payload_bytes": _backing_bytes(
                _pair_tensors(p for s in self._segments for p in s.layers.values())
            ),
            "staged_backing_bytes": _backing_bytes(
                [
                    *_pair_tensors(self._staged.values()),
                    *(t for group in self._packed_groups.values() for t in group),
                    *([] if self._packed_scales is None else [self._packed_scales]),
                ]
            ),
            "staged_payload_bytes": sum(p.bytes() for p in self._staged.values()),
            "dequant_tile_tokens": self.tile_tokens,
            "tile_scratch_estimated_bytes": self.tile_scratch_estimated_bytes,
            "tile_scratch_budget_bytes": MAX_KV_TILE_SCRATCH_BYTES,
        }

    @torch.no_grad()
    def attention_with_history(
        self,
        layer_name: str,
        query: torch.Tensor,
        condition: AVKV,
        current: AVKV,
        *,
        scale: float,
        query_chunk_size: int,
        max_joined_bytes: int,
    ) -> torch.Tensor:
        with self._exclusive():
            self._validate_layer_name(layer_name)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("scale must be positive finite")
            for number, label in (
                (query_chunk_size, "query_chunk_size"),
                (max_joined_bytes, "max_joined_bytes"),
            ):
                if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                    raise ValueError(f"{label} must be a positive integer")
            # Shape/dtype checks and budget precede any joined allocation.
            for label, t in (
                ("query", query),
                ("condition key", condition.key),
                ("condition value", condition.value),
                ("current key", current.key),
                ("current value", current.value),
            ):
                self._validate_tensor(t, label)
                if t.device != query.device:
                    raise ValueError("attention devices differ")
            if (
                condition.key.shape != condition.value.shape
                or current.key.shape != current.value.shape
            ):
                raise ValueError("attention pair shape differs")
            count = len(condition.key) + self.history_tokens + len(current.key)
            if count == 0 or 4 * count * self.contract.local_heads * 128 > max_joined_bytes:
                raise MemoryError("joined BF16 KV exceeds explicit budget or is empty")
            joined_key = joined_value = out = restored = indices = None
            try:
                joined_key = torch.empty(
                    (count, self.contract.local_heads, 128),
                    dtype=torch.bfloat16,
                    device=query.device,
                )
                joined_value = torch.empty_like(joined_key)
                for which, out in enumerate((joined_key, joined_value)):
                    prefix = condition.key if which == 0 else condition.value
                    suffix = current.key if which == 0 else current.value
                    position = len(prefix)
                    out[:position].copy_(prefix)
                    for segment in self._segments:
                        pair = segment.layers[layer_name]
                        q, s = (
                            (pair.key, pair.key_scale)
                            if which == 0
                            else (pair.value, pair.value_scale)
                        )
                        for start in range(0, len(segment.rows), self.tile_tokens):
                            indices = torch.tensor(
                                segment.rows[start : start + self.tile_tokens],
                                dtype=torch.long,
                                device=query.device,
                            )
                            restored = (
                                q.index_select(0, indices).float() * s.index_select(0, indices)
                            ).bfloat16()
                            out[position : position + len(indices)].copy_(restored)
                            position += len(indices)
                            self._sync()
                            restored = indices = None
                    out[position:].copy_(suffix)
                out = None  # loop alias must not outlive the owning joined tensors
                output = query_chunked_sdpa(
                    query,
                    joined_key,
                    joined_value,
                    scale=scale,
                    query_chunk_size=query_chunk_size,
                )
                self._sync()
                return output
            except BaseException as error:
                # An exception retained by diagnostics must not own the joined
                # GPU buffers through an exited attention helper's traceback.
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                try:
                    self._sync()
                finally:
                    joined_key = joined_value = out = restored = indices = None
