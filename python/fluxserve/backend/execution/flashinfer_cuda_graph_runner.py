from __future__ import annotations

import bisect
import gc
import logging
import time
from dataclasses import dataclass

import torch

from fluxserve.backend.execution.cuda_graph_runner import model_capture_mode
from fluxserve.backend.execution.forward_batch_info import ForwardBatch, ForwardMode
from fluxserve.backend.layers.dp_attention import get_attention_tp_size

logger = logging.getLogger(__name__)


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    kv_indices: torch.Tensor
    slot_mapping: torch.Tensor
    packed_mask: torch.Tensor
    dummy_slot_mapping: torch.Tensor
    wrapper: object
    last_mask_length: int | None = None


@dataclass
class _DecodeGraphEntry:
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    last_page_len: torch.Tensor
    slot_mapping: torch.Tensor
    wrapper: object
    hidden_states: torch.Tensor


class FlashInferCudaGraphRunner:
    """One full-model CUDA graph per FlashInfer paged prefill bucket."""

    DEFAULT_CAPTURE_SIZES = (64, 128, 256, 512, 1024)

    def __init__(
        self,
        device: str | torch.device,
        capture_sizes=DEFAULT_CAPTURE_SIZES,
        log_callback=None,
        num_layers: int | None = None,
        decode_capture_batch_sizes=(1, 2, 4, 8),
    ):
        self.device = torch.device(device)
        self.capture_sizes = tuple(sorted(set(int(x) for x in capture_sizes)))
        self.decode_capture_batch_sizes = tuple(
            sorted(set(int(x) for x in decode_capture_batch_sizes))
        )
        if not self.decode_capture_batch_sizes or any(
            size <= 0 or size & (size - 1)
            for size in self.decode_capture_batch_sizes
        ):
            raise ValueError(
                "decode_capture_batch_sizes must contain positive powers of two"
            )
        self._log_callback = log_callback
        self._num_layers = int(num_layers) if num_layers is not None else None
        self._graphs: dict[tuple, _GraphEntry] = {}
        self._decode_graphs: dict[tuple, _DecodeGraphEntry] = {}
        self._packed_masks: dict[tuple[int, int, int], torch.Tensor] = {}
        self._active_entry: _GraphEntry | None = None
        self._workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self._capture_stream = torch.cuda.Stream(device=self.device)
        self._graph_pool = torch.cuda.graph_pool_handle()
        self.capture_count = 0
        self.replay_count = 0
        self.decode_capture_count = 0
        self.decode_replay_count = 0
        self.decode_decomposed_plan_count = 0
        self.decode_component_replay_count = 0
        self._decode_capture_counts_by_bs = {
            size: 0 for size in self.decode_capture_batch_sizes
        }
        self._decode_replay_counts_by_bs = {
            size: 0 for size in self.decode_capture_batch_sizes
        }
        self.invalidation_count = 0
        self.prefill_fallback_count = 0
        self.decode_fallback_count = 0
        self.capture_time_s = 0.0
        self.capture_memory_allocated_bytes = 0
        self.capture_memory_reserved_bytes = 0

    def log(self, message: str, *args) -> None:
        if self._log_callback is not None:
            self._log_callback(message % args)
        else:
            logger.warning(message, *args)

    def invalidate(self, reason: str = "unspecified", *, log: bool = True) -> None:
        had_graphs = bool(self._graphs or self._decode_graphs)
        if had_graphs:
            self.invalidation_count += 1
        if had_graphs and log:
            self.log(
                "Invalidating CUDA graphs: prefill=%d decode=%d reason=%s",
                len(self._graphs), len(self._decode_graphs), reason,
            )
        self._active_entry = None
        self._graphs.clear()
        self._decode_graphs.clear()

    def reset_serving_counts(self) -> None:
        self.replay_count = 0
        self.decode_replay_count = 0
        self.decode_decomposed_plan_count = 0
        self.decode_component_replay_count = 0
        for batch_size in self._decode_replay_counts_by_bs:
            self._decode_replay_counts_by_bs[batch_size] = 0
        self.prefill_fallback_count = 0
        self.decode_fallback_count = 0

    def shutdown(self, process_group=None, *, log: bool = True) -> None:
        """Release captured collectives before their NCCL process group."""
        import torch.distributed as dist

        torch.cuda.synchronize(self.device)
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=process_group)
        self.invalidate(log=log)
        self._packed_masks.clear()
        self._workspace = None
        self._capture_stream = None
        self._graph_pool = None
        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=process_group)

    def bucket(self, length: int) -> int | None:
        index = bisect.bisect_left(self.capture_sizes, int(length))
        return self.capture_sizes[index] if index < len(self.capture_sizes) else None

    def can_run(self, *, q_len: int, q_offset: int, page_size: int) -> bool:
        return bool(
            page_size == 64
            and q_offset == 0
            and self.bucket(q_len) is not None
        )

    def replay(
        self,
        *,
        runner,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        q_len = int(input_ids.shape[1])
        bucket = self.bucket(q_len)
        if bucket is None:
            raise RuntimeError("Full-prefill CUDA graph received unsupported length")
        cache = runner.past_key_values
        key = (bucket, cache.data.data_ptr(), input_ids.dtype)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._capture(runner, bucket, key)

        entry.input_ids.fill_(int(runner.decoder.mask_id))
        entry.input_ids[:, :q_len].copy_(input_ids)
        actual_pages = forward_batch.flashinfer_paged_kv_indices
        entry.kv_indices.fill_(int(cache.dummy_page_id))
        entry.kv_indices[: actual_pages.numel()].copy_(actual_pages)

        real_slots = forward_batch.flashinfer_slot_mapping.reshape(-1)
        entry.slot_mapping[:q_len].copy_(real_slots[:q_len])
        entry.slot_mapping[q_len:].copy_(entry.dummy_slot_mapping[q_len:])
        mask_key = (q_len, bucket, int(runner.block_length))
        if entry.last_mask_length != q_len:
            entry.wrapper._custom_mask_buf.copy_(self._packed_masks[mask_key])
            entry.last_mask_length = q_len
        entry.graph.replay()
        self.replay_count += 1

    def run_attention(
        self,
        *,
        q: torch.Tensor,
        paged_kv_cache,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
    ) -> torch.Tensor:
        del num_q_heads, num_kv_heads, head_dim, sm_scale
        if self._active_entry is None:
            raise RuntimeError("FlashInfer full-prefill graph is not active")
        return self._active_entry.wrapper.run(
            q, paged_kv_cache, enable_pdl=False
        )

    def can_run_decode(self, *, batch_size: int, q_len: int, kv_len: int) -> bool:
        return bool(
            batch_size in self.decode_capture_batch_sizes
            and q_len == 64
            and kv_len >= q_len
            and kv_len % q_len == 0
        )

    @staticmethod
    def decompose_batch_size(
        batch_size: int, capture_batch_sizes=(1, 2, 4, 8)
    ) -> tuple[int, ...]:
        """Decompose a batch into exact power-of-two graphs without padding."""
        if batch_size <= 0:
            return ()
        parts = []
        remaining = int(batch_size)
        for bucket in reversed(tuple(capture_batch_sizes)):
            while remaining >= bucket:
                parts.append(bucket)
                remaining -= bucket
        return tuple(parts)

    @staticmethod
    def capture_batch_sizes(max_batch_size: int) -> tuple[int, ...]:
        """Return graph sizes reachable up to the configured online limit."""
        if max_batch_size <= 0:
            return ()
        return tuple(
            1 << power for power in range(int(max_batch_size).bit_length())
        )

    def record_capture_memory(self, allocated_before: int, reserved_before: int) -> None:
        """Record and log allocator growth across an eager graph capture phase."""
        torch.cuda.synchronize(self.device)
        self.capture_memory_allocated_bytes = max(
            0, int(torch.cuda.memory_allocated(self.device)) - int(allocated_before)
        )
        self.capture_memory_reserved_bytes = max(
            0, int(torch.cuda.memory_reserved(self.device)) - int(reserved_before)
        )
        mib = 1024 * 1024
        self.log(
            "CUDA graph memory: allocated_delta=%d bytes (%.2f MiB), "
            "reserved_delta=%d bytes (%.2f MiB)",
            self.capture_memory_allocated_bytes,
            self.capture_memory_allocated_bytes / mib,
            self.capture_memory_reserved_bytes,
            self.capture_memory_reserved_bytes / mib,
        )

    def record_decode_decomposition(self, component_count: int) -> None:
        if int(component_count) > 1:
            self.decode_decomposed_plan_count += 1

    def capture_decode_batch_sizes(self, runner, batch_sizes=None) -> None:
        started = time.perf_counter()
        cache = runner.past_key_values
        if batch_sizes is None:
            batch_sizes = self.decode_capture_batch_sizes
        for batch_size in sorted(set(map(int, batch_sizes)), reverse=True):
            key = (batch_size, cache.data.data_ptr(), torch.long)
            if key not in self._decode_graphs:
                self._capture_decode(runner, batch_size, key)
        self.capture_time_s += time.perf_counter() - started

    def stats(self) -> dict[str, int | float]:
        stats = {
            "prefill_capture_count": self.capture_count,
            "prefill_replay_count": self.replay_count,
            "prefill_fallback_count": self.prefill_fallback_count,
            "decode_capture_count": self.decode_capture_count,
            "decode_replay_count": self.decode_replay_count,
            "decode_decomposed_plan_count": self.decode_decomposed_plan_count,
            "decode_component_replay_count": self.decode_component_replay_count,
            "decode_fallback_count": self.decode_fallback_count,
            "invalidation_count": self.invalidation_count,
            "capture_time_s": self.capture_time_s,
            "memory_allocated_bytes": torch.cuda.memory_allocated(self.device),
            "memory_reserved_bytes": torch.cuda.memory_reserved(self.device),
        }
        for batch_size in self.decode_capture_batch_sizes:
            stats[f"decode_capture_bs_{batch_size}"] = (
                self._decode_capture_counts_by_bs[batch_size]
            )
            stats[f"decode_replay_bs_{batch_size}"] = (
                self._decode_replay_counts_by_bs[batch_size]
            )
        return stats

    def replay_decode(
        self,
        *,
        runner,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        actual_batch_size = int(input_ids.shape[0])
        batch_size = actual_batch_size
        if getattr(runner.runner_config, "decode_cuda_graph_mode", "decomposed") == "padded":
            batch_size = next(
                (size for size in self.decode_capture_batch_sizes if size >= actual_batch_size),
                actual_batch_size,
            )
        cache = runner.past_key_values
        key = (batch_size, cache.data.data_ptr(), input_ids.dtype)
        entry = self._decode_graphs.get(key)
        if entry is None:
            entry = self._capture_decode(runner, batch_size, key)
        if batch_size != actual_batch_size:
            # Use model mask tokens and dedicated dummy pages for padding. The
            # padded rows are needed only for collective/graph shape and their
            # outputs are discarded; they must not alias a live request's KV.
            pad_rows = batch_size - actual_batch_size
            dummy_ids = torch.full(
                (pad_rows, input_ids.shape[1]), int(runner.decoder.mask_id),
                dtype=input_ids.dtype, device=input_ids.device,
            )
            dummy_pos = torch.zeros(
                (pad_rows, position_ids.shape[1]),
                dtype=position_ids.dtype, device=position_ids.device,
            )
            input_ids = torch.cat((input_ids, dummy_ids), dim=0).contiguous()
            position_ids = torch.cat((position_ids, dummy_pos), dim=0).contiguous()
            kv_indptr = forward_batch.flashinfer_kv_indptr
            kv_indices = forward_batch.flashinfer_paged_kv_indices
            last_page_len = forward_batch.flashinfer_paged_kv_last_page_len
            slots = forward_batch.flashinfer_slot_mapping.reshape(-1, forward_batch.flashinfer_slot_mapping.shape[-1])
            # Build padded CSR metadata by repeating the last request's pages.
            lengths = kv_indptr[1:] - kv_indptr[:-1]
            pad_len = int(lengths[-1].item()) if lengths.numel() else 1
            kv_indptr = torch.cat((kv_indptr, kv_indptr[-1] + torch.arange(1, batch_size - actual_batch_size + 1, device=kv_indptr.device, dtype=kv_indptr.dtype) * pad_len))
            dummy_pages = torch.full(
                (pad_rows * pad_len,), int(cache.dummy_page_id),
                dtype=kv_indices.dtype, device=kv_indices.device,
            )
            kv_indices = torch.cat((kv_indices, dummy_pages))
            last_page_len = torch.cat((last_page_len, torch.full(
                (pad_rows,), int(cache.page_size), dtype=last_page_len.dtype,
                device=last_page_len.device,
            )))
            dummy_slots = torch.arange(
                int(cache.dummy_page_id) * int(cache.page_size),
                int(cache.dummy_page_id) * int(cache.page_size) + pad_rows * slots.shape[1],
                dtype=slots.dtype, device=slots.device,
            ).view(pad_rows, slots.shape[1])
            slots = torch.cat((slots, dummy_slots), dim=0)
            padded_kv_indptr, padded_kv_indices = kv_indptr, kv_indices
            padded_last_page_len, padded_slots = last_page_len, slots
        else:
            padded_kv_indptr = forward_batch.flashinfer_kv_indptr
            padded_kv_indices = forward_batch.flashinfer_paged_kv_indices
            padded_last_page_len = forward_batch.flashinfer_paged_kv_last_page_len
            padded_slots = forward_batch.flashinfer_slot_mapping
        entry.input_ids.copy_(input_ids)
        entry.position_ids.copy_(position_ids)
        actual_indices = padded_kv_indices
        if actual_indices.numel() > entry.kv_indices.numel():
            raise RuntimeError(
                "Dynamic decode metadata exceeds captured page capacity: "
                f"actual={actual_indices.numel()}, capacity={entry.kv_indices.numel()}"
            )
        entry.kv_indptr.copy_(padded_kv_indptr)
        entry.kv_indices.fill_(int(cache.dummy_page_id))
        entry.kv_indices[: actual_indices.numel()].copy_(actual_indices)
        entry.last_page_len.copy_(padded_last_page_len)
        entry.slot_mapping.copy_(padded_slots.reshape(-1))
        entry.graph.replay()
        self.decode_replay_count += 1
        self.decode_component_replay_count += 1
        self._decode_replay_counts_by_bs[batch_size] += 1
        return entry.hidden_states[:actual_batch_size]

    def run_decode_attention(self, q: torch.Tensor, paged_kv_cache) -> torch.Tensor:
        if not isinstance(self._active_entry, _DecodeGraphEntry):
            raise RuntimeError("FlashInfer full-decode graph is not active")
        output = self._active_entry.wrapper.run(q, paged_kv_cache)
        bsz = int(self._active_entry.input_ids.shape[0])
        q_len = int(self._active_entry.input_ids.shape[1])
        return output.view(bsz, q_len, q.shape[1], q.shape[2]).transpose(1, 2).contiguous()

    def _capture_decode(self, runner, batch_size: int, key: tuple) -> _DecodeGraphEntry:
        import flashinfer

        cache = runner.past_key_values
        block_length = int(runner.block_length)
        page_size = int(cache.page_size)
        max_kv_len = (int(runner.max_length) // block_length) * block_length
        max_kv_len = max(block_length, max_kv_len)
        pages_per_row = (max_kv_len + page_size - 1) // page_size
        num_pages = batch_size * pages_per_row
        q_offset = max_kv_len - block_length
        qo_indptr = torch.arange(
            batch_size + 1, dtype=torch.int32, device=self.device
        ) * block_length
        kv_indptr = torch.arange(
            batch_size + 1, dtype=torch.int32, device=self.device
        ) * pages_per_row
        if int(cache.num_dummy_pages) < batch_size:
            raise RuntimeError(
                "Dynamic decode capture requires one dummy page per batch row: "
                f"need={batch_size}, reserved={cache.num_dummy_pages}"
            )
        dummy_page_ids = torch.arange(
            int(cache.dummy_page_id),
            int(cache.dummy_page_id) + batch_size,
            dtype=torch.int32,
            device=self.device,
        )
        kv_indices = dummy_page_ids.repeat_interleave(pages_per_row)
        last_page_len = torch.full(
            (batch_size,), page_size, dtype=torch.int32, device=self.device
        )
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD", use_cuda_graph=True,
            qo_indptr_buf=qo_indptr,
            paged_kv_indptr_buf=kv_indptr,
            paged_kv_indices_buf=kv_indices,
            paged_kv_last_page_len_buf=last_page_len,
            backend="fa2",
        )
        input_ids = torch.full(
            (batch_size, block_length), int(runner.decoder.mask_id), dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.arange(
            q_offset, max_kv_len, dtype=torch.long, device=self.device
        ).unsqueeze(0).repeat(batch_size, 1)
        slot_mapping = (
            dummy_page_ids.to(torch.long).unsqueeze(1) * page_size
            + torch.arange(block_length, dtype=torch.long, device=self.device)
        ).reshape(-1)
        kv_lens = torch.full(
            (batch_size,), max_kv_len, dtype=torch.int32, device=self.device
        )
        q_offsets = torch.full(
            (batch_size,), q_offset, dtype=torch.int32, device=self.device
        )
        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, last_page_len,
            num_qo_heads=runner.model.model.config.num_attention_heads
            // get_attention_tp_size(),
            num_kv_heads=max(
                1, runner.model.model.config.num_key_value_heads
                // get_attention_tp_size(),
            ),
            head_dim_qk=runner.model.model.config.hidden_size
            // runner.model.model.config.num_attention_heads,
            page_size=page_size,
            causal=False,
            q_data_type=torch.bfloat16,
            kv_data_type=cache.data.dtype,
            disable_split_kv=True,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            use_flashinfer_paged_decode=True,
            flashinfer_kv_lens_cpu=(max_kv_len,) * batch_size,
            flashinfer_kv_lens=kv_lens,
            flashinfer_q_offsets_cpu=(q_offset,) * batch_size,
            flashinfer_q_offsets=q_offsets,
            flashinfer_qo_indptr_cpu=tuple(qo_indptr.cpu().tolist()),
            flashinfer_qo_indptr=qo_indptr,
            flashinfer_kv_indptr_cpu=tuple(kv_indptr.cpu().tolist()),
            flashinfer_kv_indptr=kv_indptr,
            flashinfer_paged_kv_indices=kv_indices,
            flashinfer_paged_kv_last_page_len=last_page_len,
            flashinfer_seq_ids=torch.arange(batch_size, dtype=torch.long, device=self.device),
            flashinfer_slot_mapping=slot_mapping.view(batch_size, block_length),
            flashinfer_block_length=block_length,
            flashinfer_page_size=page_size,
            flashinfer_cuda_graph_runner=self,
            flashinfer_full_decode_graph=True,
        )
        past_key_values = [
            cache.layer_paged_kv(layer_id) for layer_id in range(cache.num_layers)
        ]

        def run_model():
            hidden_states, _ = runner.model.model(
                input_ids, position_ids, past_key_values,
                use_cache=False, attention_mask=None, forward_batch=forward_batch,
            )
            return hidden_states

        self.log(
            "CUDA graph capturing: dynamic paged decode batch_size=%d max_kv_len=%d",
            batch_size, max_kv_len,
        )
        placeholder = torch.empty(
            batch_size, block_length, runner.model.model.config.hidden_size,
            dtype=torch.bfloat16, device=self.device,
        )
        entry = _DecodeGraphEntry(
            torch.cuda.CUDAGraph(), input_ids, position_ids, kv_indptr, kv_indices,
            last_page_len, slot_mapping, wrapper, placeholder,
        )
        self._active_entry = entry
        try:
            for _ in range(2):
                run_model()
            torch.cuda.synchronize(self.device)
            runner.tp_group.barrier()
            with model_capture_mode(), torch.cuda.graph(
                entry.graph, pool=self._graph_pool, stream=self._capture_stream
            ):
                entry.hidden_states = run_model()
            torch.cuda.synchronize(self.device)
            runner.tp_group.barrier()
        finally:
            self._active_entry = None
        self._decode_graphs[key] = entry
        self.decode_capture_count += 1
        self._decode_capture_counts_by_bs[batch_size] = (
            self._decode_capture_counts_by_bs.get(batch_size, 0) + 1
        )
        self.log(
            "CUDA graph captured: dynamic paged decode batch_size=%d layers=%d",
            batch_size, cache.num_layers,
        )
        return entry

    def _capture(self, runner, bucket: int, key: tuple) -> _GraphEntry:
        import flashinfer
        from flashinfer import segment_packbits

        cache = runner.past_key_values
        page_size = int(cache.page_size)
        bucket_pages = bucket // page_size
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD", backend="fa2"
        )
        input_ids = torch.full(
            (1, bucket), int(runner.decoder.mask_id), dtype=torch.long, device=self.device
        )
        position_ids = torch.arange(bucket, device=self.device).unsqueeze(0)
        kv_indices = torch.full(
            (bucket_pages,), int(cache.dummy_page_id), dtype=torch.int32, device=self.device
        )
        slot_mapping = torch.arange(
            cache.dummy_page_id * page_size,
            cache.dummy_page_id * page_size + bucket,
            dtype=torch.long,
            device=self.device,
        )
        qo_indptr = torch.tensor([0, bucket], dtype=torch.int32, device=self.device)
        kv_indptr = torch.tensor([0, bucket_pages], dtype=torch.int32, device=self.device)
        last_page_len = torch.tensor([page_size], dtype=torch.int32, device=self.device)
        q_pos = torch.arange(bucket, device=self.device)
        k_pos = torch.arange(bucket, device=self.device)
        mask = (q_pos[:, None] // runner.block_length) >= (
            k_pos[None, :] // runner.block_length
        )
        mask_indptr = torch.tensor([0, bucket * bucket], dtype=torch.int32, device=self.device)
        packed_mask, _ = segment_packbits(mask.flatten(), mask_indptr, bitorder="little")
        previous_bucket = max(
            (size for size in self.capture_sizes if size < bucket),
            default=0,
        )
        for actual_length in range(
            max(1, previous_bucket + 1),
            bucket + 1,
        ):
            actual_mask = mask & (k_pos[None, :] < actual_length)
            actual_mask[actual_length:] = False
            actual_packed_mask, _ = segment_packbits(
                actual_mask.flatten(), mask_indptr, bitorder="little"
            )
            self._packed_masks[
                (actual_length, bucket, int(runner.block_length))
            ] = actual_packed_mask
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads=runner.model.model.config.num_attention_heads
            // get_attention_tp_size(),
            num_kv_heads=max(
                1,
                runner.model.model.config.num_key_value_heads
                // get_attention_tp_size(),
            ),
            head_dim_qk=runner.model.model.config.hidden_size
            // runner.model.model.config.num_attention_heads,
            page_size=page_size,
            custom_mask=mask.flatten(),
            causal=False,
            q_data_type=torch.bfloat16,
            kv_data_type=cache.data.dtype,
            disable_split_kv=True,
        )
        wrapper._custom_mask_buf.copy_(packed_mask)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            use_flashinfer_paged_prefill=True,
            flashinfer_prefill_lens_cpu=(bucket,),
            flashinfer_kv_lens_cpu=(bucket,),
            flashinfer_q_offsets=torch.zeros(1, dtype=torch.int32, device=self.device),
            flashinfer_q_offsets_cpu=(0,),
            flashinfer_qo_indptr=qo_indptr,
            flashinfer_qo_indptr_cpu=(0, bucket),
            flashinfer_kv_indptr=kv_indptr,
            flashinfer_kv_indptr_cpu=(0, bucket_pages),
            flashinfer_paged_kv_indices=kv_indices,
            flashinfer_paged_kv_indices_cpu=tuple([cache.dummy_page_id] * bucket_pages),
            flashinfer_paged_kv_last_page_len=last_page_len,
            flashinfer_paged_kv_last_page_len_cpu=(page_size,),
            flashinfer_slot_mapping=slot_mapping.view(1, bucket),
            flashinfer_block_length=int(runner.block_length),
            flashinfer_page_size=page_size,
            flashinfer_cuda_graph_runner=self,
            flashinfer_cuda_graph_dummy_page=int(cache.dummy_page_id),
            flashinfer_full_prefill_graph=True,
        )
        past_key_values = [
            cache.layer_paged_kv(layer_id)
            for layer_id in range(cache.num_layers)
        ]

        def run_model():
            return runner.model.model(
                input_ids,
                position_ids,
                past_key_values,
                use_cache=False,
                attention_mask=None,
                forward_batch=forward_batch,
            )

        entry = _GraphEntry(
            graph=torch.cuda.CUDAGraph(),
            input_ids=input_ids,
            position_ids=position_ids,
            kv_indices=kv_indices,
            slot_mapping=slot_mapping,
            packed_mask=packed_mask,
            dummy_slot_mapping=slot_mapping.clone(),
            wrapper=wrapper,
            last_mask_length=bucket,
        )
        self.log("CUDA graph capturing: full paged prefill bucket=%d", bucket)
        self._active_entry = entry
        try:
            for _ in range(2):
                run_model()
            torch.cuda.synchronize(self.device)
            runner.tp_group.barrier()
            with model_capture_mode(), torch.cuda.graph(
                entry.graph,
                pool=self._graph_pool,
                stream=self._capture_stream,
            ):
                run_model()
            torch.cuda.synchronize(self.device)
            runner.tp_group.barrier()
        finally:
            self._active_entry = None
        self._graphs[key] = entry
        self.capture_count += 1
        self.log(
            "CUDA graph captured: full paged prefill bucket=%d layers=%d",
            bucket,
            cache.num_layers,
        )
        return entry

    @staticmethod
    def _pack_mask(mask: torch.Tensor, mask_indptr: torch.Tensor):
        from flashinfer import segment_packbits

        return segment_packbits(mask.flatten(), mask_indptr, bitorder="little")
