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

"""
    FlashInfer paged attention for Diffusion-Gemma.

With ``sliding_backend="fa4"`` the sliding-window layers (head_dim 256) run on
FA4's SM100 hd256 kernel instead. The full-attention layers (head_dim 512,
K != V) stay on FlashInfer unless FA4 MLA is enabled for them
(``DiffusionGemmaPagedAttention.enable_fa4_mla``): their K is the rotary of
a scalar multiple of V on 128 of the 512 dims, so they cache only those dims
of K (k_pe) next to V and run as MLA with the scalar as softmax scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Any, Callable

import torch

from fluxserve.backend.layers.attention.diffusion_gemma_window import (
    fa4_window_kwargs,
    sliding_window_mask,
    validate_sliding_window_mode,
)

# FA4's SM100 head_dim-256 kernel loads paged KV with TMA, one 128-token page per tile.
FA4_SLIDING_PAGE_SIZE = 128


@lru_cache(maxsize=1)
def require_diffusion_gemma_flashinfer():
    try:
        import flashinfer

        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper
        packbits = flashinfer.segment_packbits
        append = flashinfer.append_paged_kv_cache
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Diffusion-Gemma FlashInfer paged attention requires upstream "
            "BatchPrefillWithPagedKVCacheWrapper, segment_packbits, and "
            "append_paged_kv_cache."
        ) from exc
    return flashinfer, wrapper, packbits, append


@dataclass(frozen=True)
class DiffusionGemmaLayerGeometry:
    num_kv_heads: int
    head_dim: int
    # K cache head dim when it differs from V's (FA4 MLA full layers cache only k_pe).
    key_head_dim: int | None = None


@dataclass
class DiffusionGemmaFA4Metadata:
    """Paged varlen view of a Diffusion-Gemma batch for FA4.

    Shapes stay fixed for a batch bucket, so CUDA graphs update them in place:
    ``kv_lens`` (batch,) int32 visible keys per sequence (``seqused_k``),
    ``page_table`` (batch, pages_per_sequence) int32 cache rows,
    ``slot_mapping`` (tokens,) int64 flat cache slot of each new token, and
    ``max_kv_len`` the per-sequence capacity, not the batch maximum.
    """

    kv_lens: torch.Tensor
    page_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_kv_len: int


@dataclass
class DiffusionGemmaAttentionMetadata:
    phase: str
    seq_ids: tuple[int, ...]
    q_offsets: tuple[int, ...]
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    kv_indices_signature: tuple[int, ...]
    max_q_len: int
    page_size: int
    gather_indices: torch.Tensor
    append_batch_indices: torch.Tensor
    append_positions: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    last_page_len: torch.Tensor
    _masks: dict[int | None, torch.Tensor] = field(default_factory=dict)
    sliding_window_mode: str = "per_query"
    sliding_window_left: int | None = None
    fa4: DiffusionGemmaFA4Metadata | None = None

    @property
    def batch_size(self) -> int:
        return len(self.q_lens)

    @property
    def plan_signature(self) -> tuple[Any, ...]:
        return (
            self.phase,
            self.seq_ids,
            self.q_offsets,
            self.q_lens,
            self.kv_lens,
            self.kv_indices_signature,
            self.page_size,
            self.sliding_window_mode,
            self.sliding_window_left,
        )

    def mask(self, sliding_window: int | None) -> torch.Tensor:
        cached = self._masks.get(sliding_window)
        if cached is not None:
            return cached
        parts = []
        device = self.qo_indptr.device
        for q_offset, q_len, kv_len in zip(
            self.q_offsets, self.q_lens, self.kv_lens, strict=True
        ):
            q_pos = torch.arange(q_len, device=device, dtype=torch.long) + q_offset
            k_pos = torch.arange(kv_len, device=device, dtype=torch.long)
            if self.phase == "denoise":
                mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
            elif self.phase in {"prefill", "commit"}:
                mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            else:
                raise ValueError(f"unknown Diffusion-Gemma phase {self.phase!r}")
            if sliding_window is not None:
                mask &= sliding_window_mask(
                    q_pos,
                    k_pos,
                    phase=self.phase,
                    sliding_window=int(sliding_window),
                    mode=self.sliding_window_mode,
                    window_left=self.sliding_window_left,
                    q_start=q_offset,
                )
            parts.append(mask.reshape(-1))
        result = (
            torch.cat(parts)
            if parts
            else torch.empty(0, dtype=torch.bool, device=device)
        )
        self._masks[sliding_window] = result
        return result


class DiffusionGemmaPagedKVCache:
    """Batched paged KV storage with heterogeneous per-layer geometry."""

    is_diffusion_gemma_paged_cache = True

    def __init__(
        self,
        *,
        layer_geometries: list[DiffusionGemmaLayerGeometry],
        max_length: int,
        page_size: int,
        dtype: torch.dtype,
        device: str | torch.device,
        batch_size: int = 1,
        sliding_backend: str = "flashinfer",
        sliding_window_mode: str = "per_query",
        sliding_window_left: int | None = None,
    ):
        if max_length <= 0 or page_size <= 0 or batch_size <= 0:
            raise ValueError("max_length, page_size, and batch_size must be positive")
        if sliding_backend not in ("flashinfer", "fa4"):
            raise ValueError(
                f"sliding_backend must be 'flashinfer' or 'fa4', got {sliding_backend!r}"
            )
        if sliding_backend == "fa4" and int(page_size) != FA4_SLIDING_PAGE_SIZE:
            raise ValueError(
                "Diffusion-Gemma FA4 sliding layers require page_size="
                f"{FA4_SLIDING_PAGE_SIZE} (the SM100 hd256 TMA page), got {page_size}"
            )
        validate_sliding_window_mode(sliding_window_mode, sliding_window_left)
        self.sliding_backend = sliding_backend
        self.sliding_window_mode = sliding_window_mode
        self.sliding_window_left = (
            None if sliding_window_left is None else int(sliding_window_left)
        )
        self.layer_geometries = tuple(layer_geometries)
        self.max_length = int(max_length)
        self.page_size = int(page_size)
        self.batch_size = int(batch_size)
        self.pages_per_sequence = math.ceil(self.max_length / self.page_size)
        self.num_pages = self.batch_size * self.pages_per_sequence
        self.device = torch.device(device)
        self.page_table = torch.arange(
            self.num_pages, dtype=torch.int32, device=self.device
        ).reshape(self.batch_size, self.pages_per_sequence)
        self.layers = []
        for geometry in self.layer_geometries:
            shape = (self.num_pages, self.page_size, int(geometry.num_kv_heads))
            key_dim = geometry.key_head_dim or geometry.head_dim
            self.layers.append(
                (
                    torch.zeros((*shape, int(key_dim)), dtype=dtype, device=self.device),
                    torch.zeros((*shape, int(geometry.head_dim)), dtype=dtype, device=self.device),
                )
            )

    def layer_paged_kv(self, layer_id: int):
        return self.layers[int(layer_id)]

    def metadata(self, kv_len: int):
        metadata = self.build_metadata(
            phase="prefill",
            seq_ids=(0,),
            q_offsets=(0,),
            q_lens=(int(kv_len),),
            kv_lens=(int(kv_len),),
            max_q_len=int(kv_len),
        )
        return metadata.kv_indptr, metadata.kv_indices, metadata.last_page_len

    def build_metadata(
        self,
        *,
        phase: str,
        seq_ids: tuple[int, ...] | list[int],
        q_offsets: tuple[int, ...] | list[int],
        q_lens: tuple[int, ...] | list[int],
        kv_lens: tuple[int, ...] | list[int],
        max_q_len: int,
    ) -> DiffusionGemmaAttentionMetadata:
        seq_ids = tuple(int(x) for x in seq_ids)
        q_offsets = tuple(int(x) for x in q_offsets)
        q_lens = tuple(int(x) for x in q_lens)
        kv_lens = tuple(int(x) for x in kv_lens)
        size = len(seq_ids)
        if not (size == len(q_offsets) == len(q_lens) == len(kv_lens)):
            raise ValueError("Diffusion-Gemma metadata fields must have matching sizes")
        if size <= 0 or max_q_len <= 0:
            raise ValueError("Diffusion-Gemma metadata requires a non-empty batch")
        if any(seq_id < 0 or seq_id >= self.batch_size for seq_id in seq_ids):
            raise ValueError("Diffusion-Gemma sequence id exceeds cache batch capacity")
        if any(q_len <= 0 or q_len > max_q_len for q_len in q_lens):
            raise ValueError("invalid Diffusion-Gemma query length")
        if any(
            offset < 0
            or kv_len <= 0
            or offset + q_len > kv_len
            or kv_len > self.max_length
            for offset, q_len, kv_len in zip(q_offsets, q_lens, kv_lens, strict=True)
        ):
            raise ValueError("Diffusion-Gemma positions exceed cache capacity")

        qo_values = [0]
        kv_values = [0]
        page_ids: list[int] = []
        last_page_lens: list[int] = []
        gather: list[int] = []
        batch_indices: list[int] = []
        append_positions: list[int] = []
        slots: list[int] = []
        for local_idx, (seq_id, offset, q_len, kv_len) in enumerate(
            zip(seq_ids, q_offsets, q_lens, kv_lens, strict=True)
        ):
            qo_values.append(qo_values[-1] + q_len)
            page_count = math.ceil(kv_len / self.page_size)
            kv_values.append(kv_values[-1] + page_count)
            base_page = seq_id * self.pages_per_sequence
            page_ids.extend(range(base_page, base_page + page_count))
            last_page_lens.append((kv_len - 1) % self.page_size + 1)
            gather.extend(local_idx * max_q_len + pos for pos in range(q_len))
            batch_indices.extend([local_idx] * q_len)
            append_positions.extend(range(offset, offset + q_len))
            # Row seq_id owns pages [seq_id * pps, (seq_id + 1) * pps), so a position's
            # flat slot is seq_id * pps * page_size + pos.
            row_base = seq_id * self.pages_per_sequence * self.page_size
            slots.extend(range(row_base + offset, row_base + offset + q_len))

        device = self.device
        fa4 = None
        if self.sliding_backend == "fa4":
            seq_index = torch.tensor(seq_ids, dtype=torch.long, device=device)
            fa4 = DiffusionGemmaFA4Metadata(
                kv_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
                page_table=self.page_table.index_select(0, seq_index).contiguous(),
                slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
                max_kv_len=self.pages_per_sequence * self.page_size,
            )
        return DiffusionGemmaAttentionMetadata(
            phase=phase,
            seq_ids=seq_ids,
            q_offsets=q_offsets,
            q_lens=q_lens,
            kv_lens=kv_lens,
            kv_indices_signature=tuple(page_ids),
            max_q_len=int(max_q_len),
            page_size=self.page_size,
            gather_indices=torch.tensor(gather, dtype=torch.long, device=device),
            append_batch_indices=torch.tensor(
                batch_indices, dtype=torch.int32, device=device
            ),
            append_positions=torch.tensor(
                append_positions, dtype=torch.int32, device=device
            ),
            qo_indptr=torch.tensor(qo_values, dtype=torch.int32, device=device),
            kv_indptr=torch.tensor(kv_values, dtype=torch.int32, device=device),
            kv_indices=torch.tensor(page_ids, dtype=torch.int32, device=device),
            last_page_len=torch.tensor(
                last_page_lens, dtype=torch.int32, device=device
            ),
            sliding_window_mode=self.sliding_window_mode,
            sliding_window_left=self.sliding_window_left,
            fa4=fa4,
        )

    def reset(self) -> None:
        for k, v in self.layers:
            k.zero_()
            v.zero_()


class _WrapperState:
    def __init__(self, device: torch.device, backend: str):
        _, wrapper_cls, _, _ = require_diffusion_gemma_flashinfer()
        self.workspace = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.wrapper = wrapper_cls(self.workspace, "NHD", backend=backend)
        self.plan_key: tuple[Any, ...] | None = None


_STATES: dict[tuple[Any, ...], _WrapperState] = {}


class DiffusionGemmaPagedAttention:
    def __init__(
        self,
        *,
        layer_id: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int | None,
        backend: str = "auto",
        fa4_kernel: Callable | None = None,
    ):
        self.layer_id = int(layer_id)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.scale = float(scale)
        self.sliding_window = sliding_window
        self.backend = backend
        self.fa4_kernel = fa4_kernel
        # FA4 MLA for a full-attention layer (enable_fa4_mla): softmax scale and rotary dims.
        self.mla_scale: float | None = None
        self.mla_rope_dims: torch.Tensor | None = None

    def enable_fa4_mla(self, *, scale: float, rope_dims: torch.Tensor) -> None:
        """Run this full-attention layer as FA4 MLA. The caller passes K as RoPE(V) (no k_norm:
        its weight is the scalar ``scale``); ``rope_dims`` are the rotated dims of the head."""
        if self.sliding_window is not None:
            raise ValueError("FA4 MLA is for full-attention layers")
        self.mla_scale = float(scale)
        self.mla_rope_dims = rope_dims.to(torch.long)

    def _state(self, device: torch.device) -> _WrapperState:
        index = device.index if device.index is not None else torch.cuda.current_device()
        key = (
            device.type,
            index,
            self.backend,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.sliding_window,
        )
        state = _STATES.get(key)
        if state is None:
            state = _WrapperState(torch.device(device.type, index), self.backend)
            _STATES[key] = state
        return state

    def _mask(
        self,
        positions: torch.Tensor,
        kv_len: int,
        phase: str,
        sliding_window_mode: str = "per_query",
        sliding_window_left: int | None = None,
    ) -> torch.Tensor:
        q_pos = positions.reshape(-1).long()
        k_pos = torch.arange(kv_len, device=positions.device, dtype=torch.long)
        mask = (
            torch.ones(q_pos.numel(), kv_len, dtype=torch.bool, device=positions.device)
            if phase == "denoise"
            else k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        )
        sliding_window = self.sliding_window
        if sliding_window is not None:
            mask &= sliding_window_mask(
                q_pos,
                k_pos,
                phase=phase,
                sliding_window=int(sliding_window),
                mode=sliding_window_mode,
                window_left=sliding_window_left,
                q_start=int(positions.reshape(-1)[0]) if q_pos.numel() else 0,
            )
        return mask

    @staticmethod
    def _write_slots(packed_k, packed_v, k_cache, v_cache, slot_mapping) -> None:
        k_cache.view(-1, *k_cache.shape[-2:]).index_copy_(0, slot_mapping, packed_k)
        v_cache.view(-1, *v_cache.shape[-2:]).index_copy_(0, slot_mapping, packed_v)

    def _fa4_kernel(self):
        if self.fa4_kernel is not None:
            return self.fa4_kernel
        from fluxserve.backend.layers.attention.fa4 import load_fa4_varlen_func

        return load_fa4_varlen_func()

    def _fa4_mla_forward(
        self,
        packed_q: torch.Tensor,
        packed_k: torch.Tensor,
        packed_v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: DiffusionGemmaAttentionMetadata,
    ) -> torch.Tensor:
        """Full-attention layer as FA4 MLA: S = scale * (q[R] . k_pe + q[notR] . V), O = P V.

        ``packed_k`` is RoPE(V); only its rotated dims R are cached (k_pe) next to V.
        """
        fa4 = metadata.fa4
        rope_dims = self.mla_rope_dims
        if rope_dims.device != packed_q.device:
            rope_dims = self.mla_rope_dims = rope_dims.to(packed_q.device)
        self._write_slots(
            packed_k.index_select(-1, rope_dims), packed_v, k_cache, v_cache, fa4.slot_mapping
        )
        result = self._fa4_kernel()(
            packed_q.index_select(-1, rope_dims),
            k_cache,
            v_cache,
            qv=packed_q.index_fill(-1, rope_dims, 0.0),
            cu_seqlens_q=metadata.qo_indptr,
            cu_seqlens_k=None,
            seqused_k=fa4.kv_lens,
            max_seqlen_q=metadata.max_q_len,
            max_seqlen_k=fa4.max_kv_len,
            page_table=fa4.page_table,
            softmax_scale=self.mla_scale,
            causal=metadata.phase != "denoise",
            return_lse=False,
        )
        return result[0] if isinstance(result, tuple) else result

    def _fa4_forward(
        self,
        packed_q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: DiffusionGemmaAttentionMetadata,
    ) -> torch.Tensor:
        """Sliding-window layer on FA4: paged varlen, seqused_k, bottom-right windows."""
        fa4 = metadata.fa4
        window = fa4_window_kwargs(
            phase=metadata.phase,
            sliding_window=int(self.sliding_window),
            mode=metadata.sliding_window_mode,
            window_left=metadata.sliding_window_left,
        )
        if window["window_left_anchor"] == "query":
            # The default; keeps FA4 builds without per-row windows working.
            del window["window_left_anchor"]
        result = self._fa4_kernel()(
            packed_q,
            k_cache,
            v_cache,
            cu_seqlens_q=metadata.qo_indptr,
            cu_seqlens_k=None,
            seqused_k=fa4.kv_lens,
            max_seqlen_q=metadata.max_q_len,
            max_seqlen_k=fa4.max_kv_len,
            page_table=fa4.page_table,
            softmax_scale=self.scale,
            return_lse=False,
            **window,
        )
        return result[0] if isinstance(result, tuple) else result

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cache: DiffusionGemmaPagedKVCache,
        metadata: DiffusionGemmaAttentionMetadata,
        graph_runner=None,
    ) -> torch.Tensor:
        bsz, _, q_len, _ = q.shape
        if bsz != metadata.batch_size or q_len != metadata.max_q_len:
            raise RuntimeError("Diffusion-Gemma attention metadata does not match Q shape")
        gather = metadata.gather_indices
        packed_q = q.transpose(1, 2).contiguous().view(
            bsz * q_len, self.num_heads, self.head_dim
        ).index_select(0, gather)
        packed_k = k.transpose(1, 2).contiguous().view(
            bsz * q_len, self.num_kv_heads, self.head_dim
        ).index_select(0, gather)
        packed_v = v.transpose(1, 2).contiguous().view(
            bsz * q_len, self.num_kv_heads, self.head_dim
        ).index_select(0, gather)

        k_cache, v_cache = cache.layer_paged_kv(self.layer_id)
        # FA4 layers (also inside the denoise CUDA graph, whose static buffers hold the FA4
        # metadata) write their own pages by flat slot; FlashInfer layers append.
        if metadata.fa4 is not None and self.mla_scale is not None:
            packed_output = self._fa4_mla_forward(
                packed_q, packed_k, packed_v, k_cache, v_cache, metadata
            )
        elif metadata.fa4 is not None and self.sliding_window is not None:
            self._write_slots(
                packed_k, packed_v, k_cache, v_cache, metadata.fa4.slot_mapping
            )
            packed_output = self._fa4_forward(packed_q, k_cache, v_cache, metadata)
        else:
            _, _, _, append_paged_kv_cache = require_diffusion_gemma_flashinfer()
            append_paged_kv_cache(
                packed_k,
                packed_v,
                metadata.append_batch_indices,
                metadata.append_positions,
                (k_cache, v_cache),
                metadata.kv_indices,
                metadata.kv_indptr,
                metadata.last_page_len,
                kv_layout="NHD",
            )
            packed_output = self._flashinfer_forward(
                q, packed_q, k_cache, v_cache, cache, metadata, graph_runner
            )
        flat_output = q.new_zeros(bsz * q_len, self.num_heads, self.head_dim)
        flat_output.index_copy_(0, gather, packed_output)
        return (
            flat_output.view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _flashinfer_forward(
        self, q, packed_q, k_cache, v_cache, cache, metadata, graph_runner
    ) -> torch.Tensor:
        if graph_runner is not None:
            return graph_runner.run_gemma_attention(
                q=packed_q,
                paged_kv_cache=(k_cache, v_cache),
                cache=cache,
                metadata=metadata,
                layer_id=self.layer_id,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                sliding_window=self.sliding_window,
                sm_scale=self.scale,
            )
        mask = metadata.mask(self.sliding_window)
        plan_key = (
            metadata.plan_signature,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.sliding_window,
            q.dtype,
            k_cache.dtype,
        )
        state = self._state(q.device)
        if state.plan_key != plan_key:
            state.wrapper.plan(
                metadata.qo_indptr,
                metadata.kv_indptr,
                metadata.kv_indices,
                metadata.last_page_len,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                page_size=cache.page_size,
                custom_mask=mask,
                causal=False,
                q_data_type=q.dtype,
                kv_data_type=k_cache.dtype,
                sm_scale=self.scale,
            )
            state.plan_key = plan_key
        return state.wrapper.run(packed_q, (k_cache, v_cache))


def probe_diffusion_gemma_flashinfer(
    geometries: list[tuple[int, int, int]], device: str | torch.device
) -> None:
    device = torch.device(device)
    for num_heads, num_kv_heads, head_dim in sorted(set(geometries)):
        cache = DiffusionGemmaPagedKVCache(
            layer_geometries=[DiffusionGemmaLayerGeometry(num_kv_heads, head_dim)],
            max_length=8,
            page_size=8,
            dtype=torch.bfloat16,
            device=device,
        )
        attention = DiffusionGemmaPagedAttention(
            layer_id=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=1.0,
            sliding_window=None,
        )
        q = torch.zeros(1, num_heads, 1, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=torch.bfloat16, device=device)
        metadata = cache.build_metadata(
            phase="prefill",
            seq_ids=(0,),
            q_offsets=(0,),
            q_lens=(1,),
            kv_lens=(1,),
            max_q_len=1,
        )
        attention.forward(q, k, k, cache=cache, metadata=metadata)


def probe_diffusion_gemma_fa4(
    geometries: list[tuple[int, int, int, int]],
    device: str | torch.device,
    *,
    sliding_window_mode: str = "per_query",
    sliding_window_left: int | None = None,
) -> None:
    """Compile the FA4 sliding-layer kernels before serving.

    ``geometries`` holds ``(num_heads, num_kv_heads, head_dim, sliding_window)``.
    One causal call (prefill and commit share it) and one denoise call per
    geometry; FA4 compiles per variant, not per length.
    """
    device = torch.device(device)
    for num_heads, num_kv_heads, head_dim, sliding_window in sorted(set(geometries)):
        cache = DiffusionGemmaPagedKVCache(
            layer_geometries=[DiffusionGemmaLayerGeometry(num_kv_heads, head_dim)],
            max_length=FA4_SLIDING_PAGE_SIZE,
            page_size=FA4_SLIDING_PAGE_SIZE,
            dtype=torch.bfloat16,
            device=device,
            sliding_backend="fa4",
            sliding_window_mode=sliding_window_mode,
            sliding_window_left=sliding_window_left,
        )
        attention = DiffusionGemmaPagedAttention(
            layer_id=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=1.0,
            sliding_window=sliding_window,
        )
        q = torch.zeros(1, num_heads, 1, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=torch.bfloat16, device=device)
        for phase in ("prefill", "denoise"):
            metadata = cache.build_metadata(
                phase=phase,
                seq_ids=(0,),
                q_offsets=(0,),
                q_lens=(1,),
                kv_lens=(1,),
                max_q_len=1,
            )
            attention.forward(q, k, k, cache=cache, metadata=metadata)


def probe_diffusion_gemma_fa4_mla(
    geometries: list[tuple[int, int, int, tuple[int, ...]]],
    device: str | torch.device,
) -> None:
    """Compile the FA4 MLA kernels of the full-attention layers before serving.

    ``geometries`` holds ``(num_heads, num_kv_heads, head_dim, rope_dims)``; one causal
    (prefill / commit) and one denoise call each.
    """
    device = torch.device(device)
    for num_heads, num_kv_heads, head_dim, rope_dims in sorted(set(geometries)):
        cache = DiffusionGemmaPagedKVCache(
            layer_geometries=[
                DiffusionGemmaLayerGeometry(num_kv_heads, head_dim, key_head_dim=len(rope_dims))
            ],
            max_length=FA4_SLIDING_PAGE_SIZE,
            page_size=FA4_SLIDING_PAGE_SIZE,
            dtype=torch.bfloat16,
            device=device,
            sliding_backend="fa4",
        )
        attention = DiffusionGemmaPagedAttention(
            layer_id=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=1.0,
            sliding_window=None,
        )
        attention.enable_fa4_mla(scale=1.0, rope_dims=torch.tensor(rope_dims, device=device))
        q = torch.zeros(1, num_heads, 1, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=torch.bfloat16, device=device)
        for phase in ("prefill", "denoise"):
            metadata = cache.build_metadata(
                phase=phase,
                seq_ids=(0,),
                q_offsets=(0,),
                q_lens=(1,),
                kv_lens=(1,),
                max_q_len=1,
            )
            attention.forward(q, k, k, cache=cache, metadata=metadata)


__all__ = [
    "FA4_SLIDING_PAGE_SIZE",
    "DiffusionGemmaAttentionMetadata",
    "DiffusionGemmaFA4Metadata",
    "DiffusionGemmaLayerGeometry",
    "DiffusionGemmaPagedAttention",
    "DiffusionGemmaPagedKVCache",
    "probe_diffusion_gemma_fa4",
    "probe_diffusion_gemma_fa4_mla",
    "probe_diffusion_gemma_flashinfer",
    "require_diffusion_gemma_flashinfer",
]
