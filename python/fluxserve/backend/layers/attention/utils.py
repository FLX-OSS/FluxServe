# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import os
from typing import Any, Optional

import torch


_FLASHINFER_STATES: dict[tuple[str, int, int, str], Any] = {}

# FlashInfer's planning workspace is shared by the wrappers on each device.
# Keep this configurable because larger online batches may require more than
# the historical 128 MiB allocation (TokenSpeed defaults to 384 MiB).
DEFAULT_FLASHINFER_WORKSPACE_SIZE = 384 * 1024 * 1024


def get_flashinfer_workspace_size() -> int:
    """Return the per-device FlashInfer workspace size in bytes."""
    value = int(os.environ.get("FLASHINFER_WORKSPACE_SIZE", DEFAULT_FLASHINFER_WORKSPACE_SIZE))
    if value <= 0:
        raise ValueError(f"FLASHINFER_WORKSPACE_SIZE must be positive, got {value}")
    return value


def _require_flashinfer_dllm():
    try:
        from flashinfer.dllm import BatchBlockExtendRaggedOffsetWrapper
    except Exception as exc:
        raise RuntimeError(
            "attention_backend='flashinfer' requires a flashinfer-python build "
            "with flashinfer.dllm.BatchBlockExtendRaggedOffsetWrapper. "
            "Run with attention_backend='sdpa' or use the rebuilt container."
        ) from exc
    return BatchBlockExtendRaggedOffsetWrapper


def _require_flashinfer_paged_prefill():
    try:
        import flashinfer

        wrapper_cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper
        segment_packbits = flashinfer.segment_packbits
        append_paged_kv_cache = flashinfer.append_paged_kv_cache
        get_batch_indices_positions = flashinfer.get_batch_indices_positions
    except Exception as exc:
        raise RuntimeError(
            "flashinfer_cache_mode='paged' requires a flashinfer-python build "
            "with BatchPrefillWithPagedKVCacheWrapper and segment_packbits."
        ) from exc
    try:
        import inspect
        params = inspect.signature(wrapper_cls.__init__).parameters
        if "block_extend" not in params or "q_offsets_buf" not in params:
            raise TypeError("paged wrapper lacks native block-extend constructor")
        plan_params = inspect.signature(wrapper_cls.plan).parameters
        if "q_offsets" not in plan_params or "kv_offsets" not in plan_params:
            raise TypeError("paged wrapper lacks offset-aware plan()")
    except Exception as exc:
        raise RuntimeError(
            "flashinfer_cache_mode='paged' requires FlashInfer 0.6.18-compatible "
            "native block-extend paged attention, offset-aware plan(), "
            "segment_packbits, and append_paged_kv_cache."
        ) from exc
    return (
        wrapper_cls,
        segment_packbits,
        append_paged_kv_cache,
        get_batch_indices_positions,
    )


class BatchBlockExtendRaggedOffsetWrapper:
    """FluxServe-compatible wrapper over upstream flashinfer.dllm."""

    def __init__(
        self,
        workspace: torch.Tensor,
        *,
        kv_layout: str = "NHD",
        dllm_block_size: int,
        backend: str = "auto",
    ):
        if dllm_block_size <= 0 or dllm_block_size & (dllm_block_size - 1):
            raise ValueError(
                "dllm_block_size must be a positive power of 2, "
                f"got {dllm_block_size}."
            )
        wrapper_cls = _require_flashinfer_dllm()
        self.wrapper = wrapper_cls(
            workspace,
            kv_layout=kv_layout,
            dllm_block_size=int(dllm_block_size),
            backend=backend,
        )
        self.plan_key: tuple[Any, ...] | None = None
        self.sm_scale = None

    def plan(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_data_type: torch.dtype,
        sm_scale: Optional[float] = None,
        q_offsets: torch.Tensor,
        kv_offsets: Optional[torch.Tensor] = None,
        plan_key: Optional[tuple[Any, ...]] = None,
    ):
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim)

        effective_plan_key = plan_key or (
            tuple(qo_indptr.detach().cpu().tolist()),
            tuple(kv_indptr.detach().cpu().tolist()),
            tuple(q_offsets.detach().cpu().tolist()),
            None
            if kv_offsets is None
            else tuple(kv_offsets.detach().cpu().tolist()),
            q_data_type,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            qo_indptr.device.index,
        )
        if self.plan_key == effective_plan_key:
            self.sm_scale = sm_scale
            return

        self.wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_data_type=q_data_type,
            q_offsets=q_offsets,
            kv_offsets=kv_offsets,
        )
        self.plan_key = effective_plan_key
        self.sm_scale = sm_scale

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.plan_key is None:
            raise RuntimeError("BatchBlockExtendRaggedOffsetWrapper.plan() was not called.")
        return self.wrapper.run(q, k, v)


class FlashInferDLLMBlockExtendState:
    def __init__(
        self,
        device: torch.device,
        *,
        block_length: int,
        backend: str = "auto",
        workspace_mb: int | None = None,
    ):
        self.device = torch.device(device)
        self.block_length = int(block_length)
        workspace_bytes = get_flashinfer_workspace_size() if workspace_mb is None else int(workspace_mb) * 1024 * 1024
        self.workspace = torch.empty(
            workspace_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        self.wrapper = BatchBlockExtendRaggedOffsetWrapper(
            self.workspace,
            kv_layout="NHD",
            dllm_block_size=self.block_length,
            backend=backend,
        )

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        q_offsets: torch.Tensor,
        kv_offsets: Optional[torch.Tensor],
        plan_key: tuple[Any, ...],
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
    ) -> torch.Tensor:
        self.wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_data_type=q.dtype,
            sm_scale=sm_scale,
            q_offsets=q_offsets,
            kv_offsets=kv_offsets,
            plan_key=plan_key,
        )
        return self.wrapper.run(q, k, v)


def get_flashinfer_dllm_state(
    device: torch.device,
    *,
    block_length: int,
    backend: str = "auto",
) -> FlashInferDLLMBlockExtendState:
    device = torch.device(device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device.type, device_index, int(block_length), backend)
    state = _FLASHINFER_STATES.get(key)
    if state is None:
        state = FlashInferDLLMBlockExtendState(
            torch.device(device.type, device_index),
            block_length=block_length,
            backend=backend,
        )
        _FLASHINFER_STATES[key] = state
    return state


class FlashInferPagedBlockExtendState:
    def __init__(
        self,
        device: torch.device,
        *,
        workspace_mb: int | None = None,
        backend: str = "auto",
    ):
        self.device = torch.device(device)
        workspace_bytes = get_flashinfer_workspace_size() if workspace_mb is None else int(workspace_mb) * 1024 * 1024
        self.workspace = torch.empty(
            workspace_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        wrapper_cls, _, _, _ = _require_flashinfer_paged_prefill()
        self.wrapper = wrapper_cls(
            self.workspace, kv_layout="NHD", backend=backend,
            block_extend=True, block_size=1,
        )
        self.block_size = 1
        self.plan_key: tuple[Any, ...] | None = None

    def needs_plan(self, plan_key: tuple[Any, ...]) -> bool:
        return self.plan_key != plan_key

    def make_mask(
        self,
        *,
        q_offsets: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_lens: torch.Tensor,
        block_length: int,
    ) -> torch.Tensor:
        bool_parts = []
        q_offsets_cpu = q_offsets.detach().cpu().tolist()
        kv_lens_cpu = kv_lens.detach().cpu().tolist()
        qo_indptr_cpu = qo_indptr.detach().cpu().tolist()
        for idx, (q_offset, kv_len) in enumerate(
            zip(q_offsets_cpu, kv_lens_cpu, strict=True)
        ):
            q_len = int(qo_indptr_cpu[idx + 1] - qo_indptr_cpu[idx])
            q_pos = torch.arange(q_len, device=self.device, dtype=torch.long) + int(q_offset)
            k_pos = torch.arange(int(kv_len), device=self.device, dtype=torch.long)
            mask = (q_pos[:, None] // int(block_length)) >= (
                k_pos[None, :] // int(block_length)
            )
            bool_parts.append(mask.reshape(-1))
        return torch.cat(bool_parts, dim=0)

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        last_page_len: torch.Tensor,
        kv_lens: torch.Tensor,
        q_offsets: torch.Tensor,
        kv_offsets: Optional[torch.Tensor],
        custom_mask: Optional[torch.Tensor],
        plan_key: tuple[Any, ...],
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        block_length: int,
        sm_scale: float,
    ) -> torch.Tensor:
        def plan():
            if self.plan_key == plan_key:
                return
            # Recreate the wrapper when block size changes; upstream captures it
            # at construction time.
            if getattr(self, "block_size", None) != int(block_length):
                wrapper_cls, _, _, _ = _require_flashinfer_paged_prefill()
                self.wrapper = wrapper_cls(
                    self.workspace, kv_layout="NHD", backend="auto",
                    block_extend=True, block_size=int(block_length),
                )
                self.block_size = int(block_length)
            self.wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                custom_mask=custom_mask,
                causal=False,
                q_data_type=q.dtype,
                kv_data_type=paged_kv_cache[0].dtype,
                sm_scale=sm_scale,
                q_offsets=q_offsets,
                kv_offsets=kv_offsets,
            )
            self.plan_key = plan_key

        plan()
        return self.wrapper.run(q, paged_kv_cache)


def get_flashinfer_paged_block_extend_state(
    device: torch.device,
    *,
    backend: str = "auto",
) -> FlashInferPagedBlockExtendState:
    device = torch.device(device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device.type, device_index, -1, f"paged:{backend}")
    state = _FLASHINFER_STATES.get(key)
    if state is None:
        state = FlashInferPagedBlockExtendState(
            torch.device(device.type, device_index),
            backend=backend,
        )
        _FLASHINFER_STATES[key] = state
    return state
