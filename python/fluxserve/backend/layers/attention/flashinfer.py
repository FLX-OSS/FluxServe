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

from __future__ import annotations

from typing import Optional

import torch

from fluxserve.backend.execution.forward_batch_info import ForwardBatch
from fluxserve.backend.layers.attention.base import AttentionForwardConfig
from fluxserve.backend.layers.attention.utils import (
    get_flashinfer_dllm_state,
    get_flashinfer_paged_block_extend_state,
)


class FlashInferRaggedPrefillAttention:
    def __init__(self, config: AttentionForwardConfig):
        self.config = config

    def can_run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch],
    ) -> bool:
        return bool(
            forward_batch is not None
            and getattr(forward_batch, "use_flashinfer_prefill", False)
            and attention_mask is None
            and q.is_cuda
            and k.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and k.dtype in (torch.float16, torch.bfloat16)
            and q.shape[2] > 0
            and q.shape[2] == k.shape[2]
        )

    @torch.compiler.disable(recursive=False)
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        bsz, _, max_q_len, _ = q.shape
        prefill_lens = forward_batch.flashinfer_prefill_lens_cpu
        if len(prefill_lens) != bsz:
            raise RuntimeError(
                "FlashInfer ragged prefill metadata batch size mismatch: "
                f"got {len(prefill_lens)} lengths for batch size {bsz}."
            )
        if forward_batch.flashinfer_qo_indptr is None:
            raise RuntimeError("FlashInfer ragged prefill requires qo_indptr.")
        if forward_batch.flashinfer_kv_indptr is None:
            raise RuntimeError("FlashInfer ragged prefill requires kv_indptr.")
        if forward_batch.flashinfer_q_offsets is None:
            raise RuntimeError("FlashInfer ragged prefill requires q_offsets.")
        if forward_batch.flashinfer_kv_offsets is None:
            raise RuntimeError("FlashInfer ragged prefill requires kv_offsets.")
        block_length = int(forward_batch.flashinfer_block_length or max_q_len)
        for prefill_len in prefill_lens:
            if int(prefill_len) % block_length != 0:
                raise RuntimeError(
                    "FlashInfer ragged prefill requires block-aligned lengths: "
                    f"got prefill_len={int(prefill_len)}, "
                    f"block_length={block_length}."
                )

        def pack_qkv():
            q_parts = []
            k_parts = []
            v_parts = []
            for batch_idx, prefill_len in enumerate(prefill_lens):
                prefill_len = int(prefill_len)
                if prefill_len <= 0 or prefill_len > max_q_len:
                    raise RuntimeError(
                        "Invalid FlashInfer ragged prefill length "
                        f"{prefill_len} for max_q_len={max_q_len}."
                    )
                q_parts.append(
                    q[batch_idx, :, :prefill_len, :]
                    .transpose(0, 1)
                    .contiguous()
                )
                k_parts.append(
                    k[batch_idx, :, :prefill_len, :]
                    .transpose(0, 1)
                    .contiguous()
                )
                v_parts.append(
                    v[batch_idx, :, :prefill_len, :]
                    .transpose(0, 1)
                    .contiguous()
                )
            return (
                torch.cat(q_parts, dim=0),
                torch.cat(k_parts, dim=0),
                torch.cat(v_parts, dim=0),
            )

        q_packed, k_packed, v_packed = pack_qkv()
        plan_key = (
            "prefill",
            tuple(prefill_lens),
            tuple(forward_batch.flashinfer_qo_indptr_cpu),
            tuple(forward_batch.flashinfer_kv_indptr_cpu),
            tuple(forward_batch.flashinfer_q_offsets_cpu),
            tuple(forward_batch.flashinfer_kv_offsets_cpu),
            q.dtype,
            k.dtype,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            max_q_len,
            block_length,
            q.device.index,
        )
        state = get_flashinfer_dllm_state(q.device, block_length=block_length)
        output = state.run(
            q_packed,
            k_packed,
            v_packed,
            forward_batch.flashinfer_qo_indptr,
            forward_batch.flashinfer_kv_indptr,
            forward_batch.flashinfer_q_offsets,
            forward_batch.flashinfer_kv_offsets,
            plan_key,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            self.config.scale,
        )

        padded = q.new_zeros(
            bsz,
            max_q_len,
            self.config.num_heads,
            self.config.head_dim,
        )
        cursor = 0
        for batch_idx, prefill_len in enumerate(prefill_lens):
            prefill_len = int(prefill_len)
            padded[batch_idx, :prefill_len] = output[cursor : cursor + prefill_len]
            cursor += prefill_len
        return padded.transpose(1, 2).contiguous()


class FlashInferRaggedAttention:
    def __init__(self, config: AttentionForwardConfig):
        self.config = config

    def can_run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        past_key_values,
        attention_mask: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch],
    ) -> bool:
        return bool(
            forward_batch is not None
            and getattr(forward_batch, "use_flashinfer_decode", False)
            and attention_mask is None
            and past_key_values is not None
            and q.is_cuda
            and k.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and k.dtype in (torch.float16, torch.bfloat16)
            and q.shape[2] > 0
            and q.shape[2] <= k.shape[2]
        )

    @torch.compiler.disable(recursive=False)
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        bsz, _, q_len, _ = q.shape
        cache_length = k.shape[2]
        kv_lens = forward_batch.flashinfer_kv_lens_cpu
        if len(kv_lens) != bsz:
            raise RuntimeError(
                "FlashInfer decode metadata batch size mismatch: "
                f"got {len(kv_lens)} kv lengths for batch size {bsz}."
            )
        q_offsets = forward_batch.flashinfer_q_offsets
        q_offsets_cpu = forward_batch.flashinfer_q_offsets_cpu
        block_length = int(forward_batch.flashinfer_block_length or q_len)
        if q_offsets is None or len(q_offsets_cpu) != bsz:
            q_offsets_cpu = tuple(int(kv_len) - q_len for kv_len in kv_lens)
            q_offsets = torch.tensor(q_offsets_cpu, dtype=torch.int32, device=q.device)

        def pack_qkv():
            q_packed = q.transpose(1, 2).contiguous().view(
                bsz * q_len,
                self.config.num_heads,
                self.config.head_dim,
            )
            k_parts = []
            v_parts = []
            for batch_idx, kv_len in enumerate(kv_lens):
                prefix_len = int(kv_len) - q_len
                if prefix_len < 0 or prefix_len + q_len > cache_length:
                    raise RuntimeError(
                        "Invalid FlashInfer decode KV length "
                        f"{kv_len} for q_len={q_len} and cache_length={cache_length}."
                    )
                k_item = torch.cat(
                    (
                        k[batch_idx, :, :prefix_len, :],
                        k[batch_idx, :, cache_length - q_len : cache_length, :],
                    ),
                    dim=1,
                )
                v_item = torch.cat(
                    (
                        v[batch_idx, :, :prefix_len, :],
                        v[batch_idx, :, cache_length - q_len : cache_length, :],
                    ),
                    dim=1,
                )
                k_parts.append(k_item.transpose(0, 1).contiguous())
                v_parts.append(v_item.transpose(0, 1).contiguous())
            return q_packed, torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

        q_packed, k_packed, v_packed = pack_qkv()
        plan_key = (
            tuple(kv_lens),
            tuple(q_offsets_cpu),
            q.dtype,
            k.dtype,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            q_len,
            block_length,
            q.device.index,
        )
        state = get_flashinfer_dllm_state(q.device, block_length=block_length)
        output = state.run(
            q_packed,
            k_packed,
            v_packed,
            forward_batch.flashinfer_qo_indptr,
            forward_batch.flashinfer_kv_indptr,
            q_offsets,
            None,
            plan_key,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            self.config.scale,
        )
        return output.view(
            bsz,
            q_len,
            self.config.num_heads,
            self.config.head_dim,
        ).transpose(1, 2).contiguous()


class FlashInferPagedPrefillAttention:
    def __init__(self, config: AttentionForwardConfig):
        self.config = config

    def can_run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        past_key_values,
        attention_mask: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch],
    ) -> bool:
        return bool(
            forward_batch is not None
            and getattr(forward_batch, "use_flashinfer_paged_prefill", False)
            and attention_mask is None
            and past_key_values is not None
            and q.is_cuda
            and k.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and k.dtype in (torch.float16, torch.bfloat16)
            and q.shape[2] > 0
        )

    @torch.compiler.disable(recursive=False)
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_values,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        bsz, _, max_q_len, _ = q.shape
        prefill_lens = forward_batch.flashinfer_prefill_lens_cpu
        if len(prefill_lens) != bsz:
            raise RuntimeError(
                "FlashInfer paged prefill metadata batch size mismatch: "
                f"got {len(prefill_lens)} lengths for batch size {bsz}."
            )
        if forward_batch.flashinfer_slot_mapping is None:
            raise RuntimeError("FlashInfer paged prefill requires slot_mapping.")
        if forward_batch.flashinfer_qo_indptr is None:
            raise RuntimeError("FlashInfer paged prefill requires qo_indptr.")
        if forward_batch.flashinfer_kv_indptr is None:
            raise RuntimeError("FlashInfer paged prefill requires kv_indptr.")
        if forward_batch.flashinfer_paged_kv_indices is None:
            raise RuntimeError("FlashInfer paged prefill requires kv indices.")
        if forward_batch.flashinfer_paged_kv_last_page_len is None:
            raise RuntimeError("FlashInfer paged prefill requires last_page_len.")
        q_offsets = forward_batch.flashinfer_q_offsets
        if q_offsets is None:
            q_offsets = torch.zeros(
                len(prefill_lens),
                dtype=torch.int32,
                device=q.device,
            )

        block_length = int(forward_batch.flashinfer_block_length or max_q_len)
        page_size = int(forward_batch.flashinfer_page_size or block_length)
        self._write_prefill_kv(
            past_key_values,
            forward_batch.flashinfer_slot_mapping,
            k,
            v,
            prefill_lens,
            page_size,
        )
        q_packed = self._pack_q(q, prefill_lens)
        plan_key = (
            "paged_prefill",
            tuple(prefill_lens),
            tuple(forward_batch.flashinfer_q_offsets_cpu),
            tuple(forward_batch.flashinfer_qo_indptr_cpu),
            tuple(forward_batch.flashinfer_kv_indptr_cpu),
            tuple(forward_batch.flashinfer_paged_kv_indices_cpu),
            tuple(forward_batch.flashinfer_paged_kv_last_page_len_cpu),
            q.dtype,
            past_key_values[0].dtype,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            max_q_len,
            block_length,
            page_size,
            q.device.index,
        )
        state = get_flashinfer_paged_block_extend_state(q.device)
        if state.needs_plan(plan_key) and forward_batch.flashinfer_custom_mask is None:
            forward_batch.flashinfer_custom_mask = state.make_mask(
                q_offsets=q_offsets,
                qo_indptr=forward_batch.flashinfer_qo_indptr,
                kv_lens=forward_batch.flashinfer_kv_lens,
                block_length=block_length,
            )
        output = state.run(
            q_packed,
            past_key_values,
            forward_batch.flashinfer_qo_indptr,
            forward_batch.flashinfer_kv_indptr,
            forward_batch.flashinfer_paged_kv_indices,
            forward_batch.flashinfer_paged_kv_last_page_len,
            forward_batch.flashinfer_kv_lens,
            q_offsets,
            forward_batch.flashinfer_custom_mask,
            plan_key,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            page_size,
            block_length,
            self.config.scale,
        )
        return self._pad_output(output, q, prefill_lens)

    def _pack_q(self, q: torch.Tensor, prefill_lens: tuple[int, ...]) -> torch.Tensor:
        parts = []
        for batch_idx, prefill_len in enumerate(prefill_lens):
            parts.append(
                q[batch_idx, :, : int(prefill_len), :]
                .transpose(0, 1)
                .contiguous()
            )
        return torch.cat(parts, dim=0)

    def _write_prefill_kv(
        self,
        past_key_values,
        slot_mapping: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prefill_lens: tuple[int, ...],
        page_size: int,
    ) -> None:
        k_cache, v_cache = past_key_values
        k_parts = []
        v_parts = []
        for batch_idx, prefill_len in enumerate(prefill_lens):
            k_parts.append(
                k[batch_idx, :, : int(prefill_len), :]
                .transpose(0, 1)
                .contiguous()
            )
            v_parts.append(
                v[batch_idx, :, : int(prefill_len), :]
                .transpose(0, 1)
                .contiguous()
            )
        flat_slots = []
        for batch_idx, prefill_len in enumerate(prefill_lens):
            flat_slots.append(slot_mapping[batch_idx, : int(prefill_len)])
        flat_slots = torch.cat(flat_slots, dim=0).to(device=k.device, dtype=torch.long)
        pages = flat_slots // int(page_size)
        offsets = flat_slots % int(page_size)
        k_cache[pages, offsets] = torch.cat(k_parts, dim=0)
        v_cache[pages, offsets] = torch.cat(v_parts, dim=0)

    def _pad_output(
        self,
        output: torch.Tensor,
        q: torch.Tensor,
        prefill_lens: tuple[int, ...],
    ) -> torch.Tensor:
        padded = q.new_zeros(
            q.shape[0],
            q.shape[2],
            self.config.num_heads,
            self.config.head_dim,
        )
        cursor = 0
        for batch_idx, prefill_len in enumerate(prefill_lens):
            prefill_len = int(prefill_len)
            padded[batch_idx, :prefill_len] = output[cursor : cursor + prefill_len]
            cursor += prefill_len
        return padded.transpose(1, 2).contiguous()


class FlashInferPagedAttention:
    def __init__(self, config: AttentionForwardConfig):
        self.config = config

    def can_run(
        self,
        q: torch.Tensor,
        past_key_values,
        attention_mask: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch],
    ) -> bool:
        return bool(
            forward_batch is not None
            and getattr(forward_batch, "use_flashinfer_paged_decode", False)
            and attention_mask is None
            and past_key_values is not None
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and q.shape[2] > 0
        )

    @torch.compiler.disable(recursive=False)
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_values,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        bsz, _, q_len, _ = q.shape
        kv_lens = forward_batch.flashinfer_kv_lens_cpu
        if len(kv_lens) != bsz:
            raise RuntimeError(
                "FlashInfer paged metadata batch size mismatch: "
                f"got {len(kv_lens)} kv lengths for batch size {bsz}."
            )
        q_offsets = forward_batch.flashinfer_q_offsets
        if q_offsets is None:
            raise RuntimeError("FlashInfer paged decode requires q_offsets.")
        if forward_batch.flashinfer_paged_kv_indices is None:
            raise RuntimeError("FlashInfer paged decode requires kv indices.")
        if forward_batch.flashinfer_paged_kv_last_page_len is None:
            raise RuntimeError("FlashInfer paged decode requires last_page_len.")
        if forward_batch.flashinfer_qo_indptr is None:
            raise RuntimeError("FlashInfer paged decode requires qo_indptr.")
        if forward_batch.flashinfer_kv_indptr is None:
            raise RuntimeError("FlashInfer paged decode requires kv_indptr.")
        if forward_batch.flashinfer_seq_ids is None:
            raise RuntimeError("FlashInfer paged decode requires seq_ids.")
        if forward_batch.flashinfer_slot_mapping is None:
            raise RuntimeError("FlashInfer paged decode requires slot_mapping.")

        block_length = int(forward_batch.flashinfer_block_length or q_len)
        page_size = int(forward_batch.flashinfer_page_size or block_length)
        self._write_current_block(
            past_key_values,
            forward_batch.flashinfer_slot_mapping,
            k,
            v,
            page_size,
        )
        q_packed = q.transpose(1, 2).contiguous().view(
            bsz * q_len,
            self.config.num_heads,
            self.config.head_dim,
        )
        plan_key = (
            "paged_block_extend",
            tuple(kv_lens),
            tuple(forward_batch.flashinfer_q_offsets_cpu),
            tuple(forward_batch.flashinfer_qo_indptr_cpu),
            tuple(forward_batch.flashinfer_kv_indptr_cpu),
            tuple(forward_batch.flashinfer_paged_kv_indices_cpu),
            tuple(forward_batch.flashinfer_paged_kv_last_page_len_cpu),
            q.dtype,
            past_key_values[0].dtype,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            q_len,
            block_length,
            page_size,
            q.device.index,
        )
        state = get_flashinfer_paged_block_extend_state(q.device)
        if state.needs_plan(plan_key) and forward_batch.flashinfer_custom_mask is None:
            forward_batch.flashinfer_custom_mask = state.make_mask(
                q_offsets=q_offsets,
                qo_indptr=forward_batch.flashinfer_qo_indptr,
                kv_lens=forward_batch.flashinfer_kv_lens,
                block_length=block_length,
            )
        output = state.run(
            q_packed,
            past_key_values,
            forward_batch.flashinfer_qo_indptr,
            forward_batch.flashinfer_kv_indptr,
            forward_batch.flashinfer_paged_kv_indices,
            forward_batch.flashinfer_paged_kv_last_page_len,
            forward_batch.flashinfer_kv_lens,
            q_offsets,
            forward_batch.flashinfer_custom_mask,
            plan_key,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            page_size,
            block_length,
            self.config.scale,
        )
        return output.view(
            bsz,
            q_len,
            self.config.num_heads,
            self.config.head_dim,
        ).transpose(1, 2).contiguous()

    def _write_current_block(
        self,
        past_key_values,
        slot_mapping: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        page_size: int,
    ) -> None:
        k_cache, v_cache = past_key_values
        flat_slots = slot_mapping.reshape(-1).to(device=k.device, dtype=torch.long)
        pages = flat_slots // int(page_size)
        offsets = flat_slots % int(page_size)
        k_values = k.transpose(1, 2).contiguous().view(
            -1,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        v_values = v.transpose(1, 2).contiguous().view_as(k_values)
        k_cache[pages, offsets] = k_values
        v_cache[pages, offsets] = v_values
