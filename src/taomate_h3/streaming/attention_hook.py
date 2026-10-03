# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Dense FlashAttention-3 hook for direct Stage3 streaming."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch

from .cache import AVKV, CleanAVKVCache


class HookMode(str, Enum):
    IDLE = "idle"
    NOISY = "noisy"
    CLEAN_COMMIT = "clean_commit"


@dataclass(frozen=True)
class _KVPart:
    key: torch.Tensor
    value: torch.Tensor


def _live_documents_from_mask(
    commit_mask: torch.Tensor,
    boundaries: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    live: list[int] = []
    for document_index, (start, stop) in enumerate(boundaries):
        if bool(commit_mask[start:stop].any().item()):
            live.append(document_index)
    return tuple(live)


class H3StreamingAttentionHook:
    """Condition current media on text, persistent clean AV KV, and current KV."""

    def __init__(
        self, cache: CleanAVKVCache, *, backend: str = "flash_attn_3",
        query_chunk_size: int | None = None,
    ) -> None:
        if backend not in ("flash_attn_3", "torch_sdpa"):
            raise ValueError("unknown streaming attention backend")
        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError("query chunk size must be positive")
        if query_chunk_size is not None and backend != "torch_sdpa":
            raise ValueError("query chunking requires torch_sdpa")
        self.query_chunk_size = query_chunk_size
        self.backend = backend
        self.cache = cache
        self._flash_attention = None
        if backend == "flash_attn_3":
            from flash_attn_interface import flash_attn_func

            self._flash_attention = flash_attn_func
        self.kernel_calls = 0
        self.query_tokens = 0
        self.key_tokens = 0
        self.mode = HookMode.IDLE
        self._live_documents: tuple[int, ...] | None = None
        self._live_document_rows: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def receipt(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "tensor_layout": "NHD",
            "is_causal": False,
            "query_chunk_size": self.query_chunk_size,
            "softmax_scope": "all_visible_keys_per_query",
            "kernel_calls": self.kernel_calls,
            "query_tokens": self.query_tokens,
            "key_tokens": self.key_tokens,
        }

    def _attention(
        self,
        query: torch.Tensor,
        parts: Sequence[_KVPart],
        *,
        scale: float,
    ) -> torch.Tensor:
        if not parts:
            raise RuntimeError("attention requires at least one KV part")
        key = (
            parts[0].key if len(parts) == 1 else torch.cat(tuple(part.key for part in parts), dim=0)
        )
        value = (
            parts[0].value
            if len(parts) == 1
            else torch.cat(tuple(part.value for part in parts), dim=0)
        )
        if self.query_chunk_size is not None:
            from ..inference.metal_attention import query_chunked_sdpa

            output = query_chunked_sdpa(
                query, key, value, scale=scale, query_chunk_size=self.query_chunk_size,
            ).unsqueeze(0)
        elif self.backend == "torch_sdpa":
            from torch.nn.functional import scaled_dot_product_attention

            output = (
                scaled_dot_product_attention(
                    query.transpose(0, 1).unsqueeze(0),
                    key.transpose(0, 1).unsqueeze(0),
                    value.transpose(0, 1).unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                )
                .squeeze(0)
                .transpose(0, 1)
                .unsqueeze(0)
            )
        else:
            output = self._flash_attention(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                softmax_scale=scale,
                causal=False,
                num_splits=1,
            )
        if isinstance(output, tuple):
            output = output[0]
        self.kernel_calls += (
            (int(query.shape[0]) + self.query_chunk_size - 1) // self.query_chunk_size
            if self.query_chunk_size is not None else 1
        )
        self.query_tokens += int(query.shape[0])
        self.key_tokens += int(key.shape[0])
        return output.squeeze(0)

    @property
    def active(self) -> bool:
        return self.mode is not HookMode.IDLE

    def activate(self, mode: HookMode) -> None:
        if self.active:
            raise RuntimeError("streaming attention hook is already active")
        if mode not in (HookMode.NOISY, HookMode.CLEAN_COMMIT):
            raise ValueError(f"unsupported streaming hook mode: {mode}")
        self.mode = mode
        self._live_documents = None
        self._live_document_rows.clear()

    def deactivate(self) -> None:
        self.mode = HookMode.IDLE
        self._live_documents = None
        self._live_document_rows.clear()

    def __call__(
        self,
        *,
        attention: object,
        layer_name: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_tags: torch.Tensor,
        commit_mask: torch.Tensor,
        cu_seqlens_host: Sequence[int],
    ) -> torch.Tensor:
        if not self.active:
            raise RuntimeError("streaming attention hook was called while inactive")
        self._validate_qkv(layer_name, query, key, value)
        if commit_mask.ndim != 1 or commit_mask.shape[0] != query.shape[0]:
            raise RuntimeError(f"{layer_name}: commit_mask does not match packed QKV rows")
        if token_tags.ndim != 1 or token_tags.shape[0] != query.shape[0]:
            raise RuntimeError(f"{layer_name}: token_tags does not match packed QKV rows")

        points = tuple(int(point) for point in cu_seqlens_host)
        if (
            len(points) < 2
            or points[0] != 0
            or points[-1] != query.shape[0]
            or any(left > right for left, right in zip(points, points[1:]))
        ):
            raise RuntimeError(f"{layer_name}: invalid packed sequence boundaries")
        boundaries = tuple(zip(points[:-1], points[1:]))

        if self._live_documents is None:
            self._live_documents = _live_documents_from_mask(
                commit_mask,
                boundaries,
            )
            if len(self._live_documents) != 1:
                raise RuntimeError(
                    "direct streaming expects exactly one live document, got "
                    f"{self._live_documents}"
                )
        live_document = self._live_documents[0]
        scale = float(getattr(attention, "softmax_scale"))
        persistent: AVKV | None = self.cache.history(layer_name)
        outputs: list[torch.Tensor] = []

        for document_index, (start, stop) in enumerate(boundaries):
            if start == stop:
                continue
            q_document = query[start:stop]
            k_document = key[start:stop]
            v_document = value[start:stop]
            if document_index != live_document:
                outputs.append(
                    self._attention(
                        q_document,
                        (_KVPart(k_document, v_document),),
                        scale=scale,
                    )
                )
                continue

            rows = self._live_document_rows.get(document_index)
            if rows is None:
                media_mask = commit_mask[start:stop].to(torch.bool)
                condition_indices = torch.nonzero(~media_mask, as_tuple=False).flatten()
                media_indices = torch.nonzero(media_mask, as_tuple=False).flatten()
                rows = (condition_indices, media_indices)
                self._live_document_rows[document_index] = rows
            condition_indices, media_indices = rows
            if condition_indices.numel() == 0 or media_indices.numel() == 0:
                raise RuntimeError("live streaming document must contain condition and media rows")

            condition_key = k_document.index_select(0, condition_indices)
            condition_value = v_document.index_select(0, condition_indices)
            condition_output = self._attention(
                q_document.index_select(0, condition_indices),
                (_KVPart(condition_key, condition_value),),
                scale=scale,
            )

            media_parts = [_KVPart(condition_key, condition_value)]
            if persistent is not None:
                media_parts.append(_KVPart(persistent.key, persistent.value))
            media_parts.append(
                _KVPart(
                    k_document.index_select(0, media_indices),
                    v_document.index_select(0, media_indices),
                )
            )
            media_output = self._attention(
                q_document.index_select(0, media_indices),
                media_parts,
                scale=scale,
            )

            document_output = torch.zeros_like(q_document)
            document_output.index_copy_(0, condition_indices, condition_output)
            document_output.index_copy_(0, media_indices, media_output)
            outputs.append(document_output)

        output = torch.cat(outputs, dim=0)
        if self.mode is HookMode.CLEAN_COMMIT:
            self.cache.stage(layer_name, key, value, token_tags, commit_mask)
        return output

    def _validate_qkv(
        self,
        layer_name: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        contract = self.cache.contract
        expected_tail = (contract.local_heads, contract.head_dim)
        if query.shape != key.shape or value.shape != key.shape or key.ndim != 3:
            raise RuntimeError(f"{layer_name}: query/key/value must have matching rank-3 shapes")
        if tuple(key.shape[1:]) != expected_tail:
            raise RuntimeError(
                f"{layer_name}: expected local attention tail {expected_tail}, "
                f"got {tuple(key.shape[1:])}"
            )
        if (
            query.dtype != contract.dtype
            or key.dtype != contract.dtype
            or value.dtype != contract.dtype
        ):
            raise RuntimeError(f"{layer_name}: streaming attention requires {contract.dtype} QKV")
        if (
            query.device.type != contract.device_type
            or key.device.type != contract.device_type
            or value.device.type != contract.device_type
        ):
            raise RuntimeError(
                f"{layer_name}: streaming attention requires {contract.device_type} QKV"
            )
