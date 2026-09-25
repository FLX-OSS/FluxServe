# Copyright (c) 2026 FLUX-OSS
# SPDX-License-Identifier: MIT

"""Token-causal / bidirectional paged attention using public FlashInfer APIs.

Unlike the LLaDA block-extend adapters, this path has one task per request.
Queries must be the suffix of the visible KV sequence. Native bottom-right
causality then needs neither absolute offsets nor a quadratic custom mask.
"""

from __future__ import annotations

from contextlib import contextmanager

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


def _page_plan(metadata, kv_lengths):
    """Host-side page plan for one forward, shared by the eager and graph states."""
    page_size = metadata.page_size
    page_counts = [
        (int(length) + page_size - 1) // page_size for length in kv_lengths
    ]
    table = metadata.page_table.detach().cpu()
    indices = torch.cat([
        table[row, :count] for row, count in enumerate(page_counts)
    ]).to(torch.int32)
    indptr = torch.tensor([0, *page_counts], dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    last_page_len = torch.tensor([
        (int(length) - 1) % page_size + 1 for length in kv_lengths
    ], dtype=torch.int32)
    return indptr, indices, last_page_len


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
            indptr, indices, last_page_len = _page_plan(metadata, lengths)
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


class FlashInferTokenPagedGraphState:
    """A plan/run split that survives CUDA graph capture.

    The eager state plans lazily inside ``run``, which a graph cannot record:
    ``plan`` reads the page table on the host and schedules on the CPU, so a
    recorded plan would never re-run and a recorded ``run`` would read a stale
    schedule.

    Here the wrapper owns persistent CSR buffers and is planned **once**, with
    split-KV disabled so the schedule does not depend on the KV lengths. Each
    replay then only rewrites those buffers -- with device work alone, no host
    sync -- and the captured region is just the kernel launch. This is the
    pattern ``_GemmaAttentionGraphState`` uses for Diffusion-Gemma, generalised
    to rows whose page counts differ: Gemma can hold every row at a fixed page
    span, Nemotron cannot, because its visible KV grows block by block.

    Measured on 14B, disabling split-KV costs nothing systematic: -0.6% at
    concurrency 1, -4.3% at 4, +5.4% at 8, against re-planning on the host.
    """

    def __init__(self, device, *, batch_size, pages_per_sequence,
                 wrapper_factory=None):
        factory = wrapper_factory or require_flashinfer_token_paged()
        batch_size = int(batch_size)
        pages_per_sequence = int(pages_per_sequence)
        if batch_size < 1 or pages_per_sequence < 1:
            raise ValueError("graph state needs a positive bucket and page span")
        device = torch.device(device)
        self.workspace = torch.empty(
            get_flashinfer_workspace_size(), dtype=torch.uint8, device=device
        )
        total_pages = batch_size * pages_per_sequence
        self.qo_indptr_buf = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        self.kv_indptr_buf = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        # One slot past the real page list absorbs the scattered writes for
        # pages a row does not own, so the compaction needs no host-visible
        # count and therefore no synchronisation.
        self.kv_indices_buf = torch.zeros(
            total_pages + 1, dtype=torch.int32, device=device
        )
        self.last_page_len_buf = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )
        self.wrapper = factory(
            self.workspace, kv_layout="NHD", backend="fa2",
            use_cuda_graph=True,
            qo_indptr_buf=self.qo_indptr_buf,
            paged_kv_indptr_buf=self.kv_indptr_buf,
            paged_kv_indices_buf=self.kv_indices_buf,
            paged_kv_last_page_len_buf=self.last_page_len_buf,
        )
        self.batch_size = batch_size
        self.pages_per_sequence = pages_per_sequence
        self._page_index = torch.arange(
            pages_per_sequence, dtype=torch.int32, device=device
        ).expand(batch_size, pages_per_sequence)
        self._trash_slot = total_pages
        self.planned = False
        # Learned from the first run, which happens during warmup: the layer
        # config and dtypes the single plan needs.
        self._signature = None

    def run(self, q, cache, metadata, config):
        if not self.planned:
            self._signature = (q.dtype, cache[0].dtype, config)
            self._plan_once(metadata)
        return self.wrapper.run(q, cache)

    def freeze(self) -> None:
        """Assert the warmup actually reached this state before capture."""
        if not self.planned:
            raise RuntimeError(
                "freeze a FlashInfer graph state only after a warmup run has "
                "planned it; an unplanned state means the capture never routed "
                "attention here"
            )

    def _plan_once(self, metadata) -> None:
        """Plan the widest layout this bucket can see. Never inside a graph."""
        q_dtype, kv_dtype, config = self._signature
        pages = self.pages_per_sequence
        counts = [pages] * self.batch_size
        indptr = torch.tensor([0, *counts], dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        indices = metadata.page_table[:, :pages].reshape(-1).to(torch.int32).cpu()
        last_page_len = torch.full(
            (self.batch_size,), metadata.page_size, dtype=torch.int32
        )
        self.wrapper.plan(
            metadata.qo_indptr.detach().cpu(), indptr, indices, last_page_len,
            num_qo_heads=config.num_heads, num_kv_heads=config.num_kv_heads,
            head_dim_qk=config.head_dim, page_size=metadata.page_size,
            causal=metadata.causal, q_data_type=q_dtype,
            kv_data_type=kv_dtype, sm_scale=config.scale,
            # The schedule must not depend on the KV lengths, because only the
            # buffer contents change from replay to replay.
            disable_split_kv=True,
        )
        self.planned = True

    @torch.inference_mode()
    def refresh(self, metadata) -> None:
        """Rewrite the CSR buffers for this replay's KV lengths, on device.

        Called before ``graph.replay()``. Every step is a device op: a host
        readback here would cost a synchronisation per block step, which is the
        whole reason the plan is issued only once.
        """
        if not self.planned:
            raise RuntimeError("refresh a FlashInfer graph state only after planning")
        page_size = int(metadata.page_size)
        kv_lens = metadata.kv_lens.to(torch.int32)
        counts = (kv_lens + (page_size - 1)) // page_size
        self.kv_indptr_buf[0].zero_()
        self.kv_indptr_buf[1:].copy_(counts.cumsum(0, dtype=torch.int32))
        self.last_page_len_buf.copy_((kv_lens - 1) % page_size + 1)
        owned = self._page_index < counts[:, None]
        destination = torch.where(
            owned,
            self.kv_indptr_buf[:-1, None] + self._page_index,
            torch.tensor(self._trash_slot, dtype=torch.int32,
                         device=self.kv_indices_buf.device),
        )
        self.kv_indices_buf.scatter_(
            0,
            destination.reshape(-1).to(torch.long),
            metadata.page_table[:, :self.pages_per_sequence]
            .reshape(-1).to(torch.int32),
        )


_STATES: dict[torch.device, FlashInferTokenPagedState] = {}
_OVERRIDES: dict[torch.device, FlashInferTokenPagedGraphState] = {}


def _state_key(device) -> torch.device:
    """One key per physical device.

    A cache built for ``cuda`` and a tensor that reports ``cuda:0`` are the same
    device but not equal as ``torch.device`` values, so an override registered
    under one spelling would be invisible to the other.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


@contextmanager
def override_token_paged_state(device, state):
    """Point this device's token-paged attention at a specific state.

    CUDA graph capture needs the graph-safe state to serve every layer of the
    forward it records, without disturbing the eager state the rest of the
    server shares.
    """
    device = _state_key(device)
    previous = _OVERRIDES.get(device)
    _OVERRIDES[device] = state
    try:
        yield state
    finally:
        if previous is None:
            _OVERRIDES.pop(device, None)
        else:
            _OVERRIDES[device] = previous


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
        key = _state_key(q.device)
        state = self._state or _OVERRIDES.get(key) or _STATES.get(key)
        if state is None:
            state = _STATES[key] = FlashInferTokenPagedState(q.device)
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
