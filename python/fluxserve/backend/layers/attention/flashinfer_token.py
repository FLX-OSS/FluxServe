# Copyright (c) 2026 FLUX-OSS
# SPDX-License-Identifier: MIT

"""Token-causal / bidirectional paged attention using public FlashInfer APIs.

Unlike the LLaDA block-extend adapters, this path has one task per request.
Queries must be the suffix of the visible KV sequence. Native bottom-right
causality then needs neither absolute offsets nor a quadratic custom mask.
"""

from __future__ import annotations

import torch

from fluxserve.backend.layers.attention.metadata import PagedAttentionMetadata
from fluxserve.backend.layers.attention.utils import get_flashinfer_workspace_size


def require_flashinfer_token_paged():
    try:
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    except ImportError as exc:
        raise RuntimeError(
            "Nemotron FlashInfer requires BatchPrefillWithPagedKVCacheWrapper "
            "from flashinfer-python; no DLLM block-extend extension is required."
        ) from exc
    return BatchPrefillWithPagedKVCacheWrapper


class FlashInferTokenPagedState:
    def __init__(self, device, *, wrapper_factory=None):
        factory = wrapper_factory or require_flashinfer_token_paged()
        self.workspace = torch.empty(
            get_flashinfer_workspace_size(), dtype=torch.uint8, device=device
        )
        # An explicit backend avoids auto-selection becoming fixed by the first
        # (causal) prefill when later denoising requires non-causal attention.
        self.wrapper = factory(self.workspace, kv_layout="NHD", backend="fa2")
        self.metadata = None
        self.signature = None

    def run(self, q, cache, metadata, config):
        signature = (
            q.dtype, cache[0].dtype, config.num_heads, config.num_kv_heads,
            config.head_dim, config.scale,
        )
        # Eager runners create fresh immutable metadata for every forward.
        # Keep only the last plan, shared across layers on the same device.
        # Identity includes causality and physical page ownership; equal shapes
        # alone must never reuse a denoise plan for a causal commit.
        if self.metadata is not metadata or self.signature != signature:
            self.metadata = None
            self.signature = None
            lengths = [
                offset + length for offset, length in zip(
                    metadata.q_offsets_cpu, metadata.q_lens_cpu, strict=True
                )
            ]
            page_counts = [
                (length + metadata.page_size - 1) // metadata.page_size
                for length in lengths
            ]
            table = metadata.page_table.detach().cpu()
            indices = torch.cat([
                table[row, :count] for row, count in enumerate(page_counts)
            ]).to(torch.int32)
            indptr = torch.tensor([0, *page_counts], dtype=torch.int32).cumsum(
                0, dtype=torch.int32
            )
            last_page_len = torch.tensor([
                (length - 1) % metadata.page_size + 1 for length in lengths
            ], dtype=torch.int32)
            self.wrapper.plan(
                metadata.qo_indptr.detach().cpu(), indptr, indices, last_page_len,
                num_qo_heads=config.num_heads, num_kv_heads=config.num_kv_heads,
                head_dim_qk=config.head_dim, page_size=metadata.page_size,
                causal=metadata.causal, q_data_type=q.dtype,
                kv_data_type=cache[0].dtype, sm_scale=config.scale,
            )
            self.metadata = metadata
            self.signature = signature
        return self.wrapper.run(q, cache)


_STATES: dict[torch.device, FlashInferTokenPagedState] = {}


class FlashInferTokenPagedAttention:
    def __init__(self, config, *, state=None):
        self.config = config
        self._state = state

    def can_run(self, q, k, v, cache, attention_mask, forward_batch):
        metadata = getattr(forward_batch, "paged_attention_metadata", None)
        return bool(
            isinstance(metadata, PagedAttentionMetadata)
            and metadata.backend == "flashinfer"
            and attention_mask is None
            and isinstance(cache, (tuple, list)) and len(cache) == 2
            and q.dtype in (torch.float16, torch.bfloat16)
            and (q.is_cuda or self._state is not None)
        )

    @torch.compiler.disable(recursive=False)
    def forward(self, q, k, v, cache, metadata):
        config = self.config
        expected_q = (
            metadata.batch_size, config.num_heads,
            metadata.max_input_len, config.head_dim,
        )
        expected_kv = (
            metadata.batch_size, config.num_kv_heads,
            metadata.max_input_len, config.head_dim,
        )
        if tuple(q.shape) != expected_q or any(
            tuple(t.shape) != expected_kv for t in (k, v)
        ):
            raise RuntimeError("FlashInfer Q/K/V shapes do not match paged metadata")
        if any(t.dtype != q.dtype or t.device != q.device for t in (k, v, *cache)):
            raise RuntimeError("FlashInfer Q/K/V and cache must share dtype and device")
        tail = (metadata.page_size, config.num_kv_heads, config.head_dim)
        if cache[0].shape != cache[1].shape or any(
            tuple(t.shape[1:]) != tail or not t.is_contiguous() for t in cache
        ):
            raise RuntimeError("FlashInfer requires contiguous NHD paged K/V caches")
        if config.num_heads % config.num_kv_heads:
            raise RuntimeError("FlashInfer Q heads must be divisible by KV heads")
        if metadata.num_tasks != metadata.batch_size:
            raise RuntimeError("Token paged attention requires one task per request")

        def pack(tensor, heads):
            tokens = tensor.transpose(1, 2).reshape(-1, heads, config.head_dim)
            if metadata.is_identity_mapping:
                return tokens.contiguous()
            return tokens.index_select(0, metadata.q_token_indices)

        packed_q = pack(q, config.num_heads)
        for target, values in zip(cache, (k, v), strict=True):
            target.view(-1, config.num_kv_heads, config.head_dim).index_copy_(
                0, metadata.slot_mapping, pack(values, config.num_kv_heads)
            )
        state = self._state or _STATES.get(q.device)
        if state is None:
            state = _STATES[q.device] = FlashInferTokenPagedState(q.device)
        output = state.run(packed_q, cache, metadata, config)
        if output.shape != packed_q.shape:
            raise RuntimeError("FlashInfer returned an unexpected output shape")
        if metadata.is_identity_mapping:
            padded = output
        else:
            padded = torch.zeros_like(q.transpose(1, 2)).reshape(
                -1, config.num_heads, config.head_dim
            )
            padded.index_copy_(0, metadata.q_token_indices, output)
        return padded.view(
            metadata.batch_size, metadata.max_input_len,
            config.num_heads, config.head_dim,
        ).transpose(1, 2)
