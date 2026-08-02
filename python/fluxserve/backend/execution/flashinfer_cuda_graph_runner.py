from __future__ import annotations

import bisect
import gc
import logging
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
    kv_indices: torch.Tensor
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
    ):
        self.device = torch.device(device)
        self.capture_sizes = tuple(sorted(set(int(x) for x in capture_sizes)))
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

    def log(self, message: str, *args) -> None:
        if self._log_callback is not None:
            self._log_callback(message % args)
        else:
            logger.info(message, *args)

    def invalidate(self) -> None:
        if self._graphs:
            logger.info("Invalidating %d full-prefill CUDA graphs", len(self._graphs))
        self._active_entry = None
        self._graphs.clear()
        self._decode_graphs.clear()

    def shutdown(self, process_group=None) -> None:
        """Release captured collectives before their NCCL process group."""
        import torch.distributed as dist

        torch.cuda.synchronize(self.device)
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=process_group)
        self.invalidate()
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
            batch_size == 1
            and q_len == 64
            and kv_len >= q_len
            and kv_len % q_len == 0
        )

    def capture_decode_lengths(self, runner, lengths) -> None:
        cache = runner.past_key_values
        for kv_len in sorted(set(map(int, lengths)), reverse=True):
            key = (kv_len, cache.data.data_ptr(), torch.long)
            if key not in self._decode_graphs:
                self._capture_decode(runner, kv_len, key)

    def replay_decode(
        self,
        *,
        runner,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        kv_len = int(forward_batch.flashinfer_kv_lens_cpu[0])
        cache = runner.past_key_values
        key = (kv_len, cache.data.data_ptr(), input_ids.dtype)
        entry = self._decode_graphs.get(key)
        if entry is None:
            entry = self._capture_decode(runner, kv_len, key)
        entry.input_ids.copy_(input_ids)
        entry.position_ids.copy_(position_ids)
        entry.kv_indices.copy_(forward_batch.flashinfer_paged_kv_indices)
        entry.slot_mapping.copy_(forward_batch.flashinfer_slot_mapping.reshape(-1))
        entry.graph.replay()
        self.decode_replay_count += 1
        return entry.hidden_states

    def run_decode_attention(self, q: torch.Tensor, paged_kv_cache) -> torch.Tensor:
        if not isinstance(self._active_entry, _DecodeGraphEntry):
            raise RuntimeError("FlashInfer full-decode graph is not active")
        output = self._active_entry.wrapper.run(q, paged_kv_cache)
        bsz = 1
        q_len = int(self._active_entry.input_ids.shape[1])
        return output.view(bsz, q_len, q.shape[1], q.shape[2]).transpose(1, 2).contiguous()

    def _capture_decode(self, runner, kv_len: int, key: tuple) -> _DecodeGraphEntry:
        import flashinfer

        cache = runner.past_key_values
        block_length = int(runner.block_length)
        page_size = int(cache.page_size)
        num_pages = (kv_len + page_size - 1) // page_size
        q_offset = kv_len - block_length
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD", backend="fa2"
        )
        input_ids = torch.full(
            (1, block_length), int(runner.decoder.mask_id), dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.arange(
            q_offset, kv_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        kv_indices = torch.full(
            (num_pages,), int(cache.dummy_page_id), dtype=torch.int32,
            device=self.device,
        )
        slot_mapping = torch.arange(
            cache.dummy_page_id * page_size,
            cache.dummy_page_id * page_size + block_length,
            dtype=torch.long,
            device=self.device,
        )
        qo_indptr = torch.tensor([0, block_length], dtype=torch.int32, device=self.device)
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=self.device)
        last_page_len = torch.tensor(
            [((kv_len - 1) % page_size) + 1], dtype=torch.int32, device=self.device
        )
        kv_lens = torch.tensor([kv_len], dtype=torch.int32, device=self.device)
        q_offsets = torch.tensor([q_offset], dtype=torch.int32, device=self.device)
        q_pos = torch.arange(q_offset, kv_len, device=self.device)
        k_pos = torch.arange(kv_len, device=self.device)
        mask = (q_pos[:, None] // block_length) >= (k_pos[None, :] // block_length)
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
            custom_mask=mask.flatten(),
            causal=False,
            q_data_type=torch.bfloat16,
            kv_data_type=cache.data.dtype,
            disable_split_kv=True,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            use_flashinfer_paged_decode=True,
            flashinfer_kv_lens_cpu=(kv_len,),
            flashinfer_kv_lens=kv_lens,
            flashinfer_q_offsets_cpu=(q_offset,),
            flashinfer_q_offsets=q_offsets,
            flashinfer_qo_indptr_cpu=(0, block_length),
            flashinfer_qo_indptr=qo_indptr,
            flashinfer_kv_indptr_cpu=(0, num_pages),
            flashinfer_kv_indptr=kv_indptr,
            flashinfer_paged_kv_indices=kv_indices,
            flashinfer_paged_kv_last_page_len=last_page_len,
            flashinfer_seq_ids=torch.zeros(1, dtype=torch.long, device=self.device),
            flashinfer_slot_mapping=slot_mapping.view(1, block_length),
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

        self.log("CUDA graph capturing: full paged decode kv_len=%d", kv_len)
        placeholder = torch.empty(
            1, block_length, runner.model.model.config.hidden_size,
            dtype=torch.bfloat16, device=self.device,
        )
        entry = _DecodeGraphEntry(
            torch.cuda.CUDAGraph(), input_ids, position_ids, kv_indices,
            slot_mapping, wrapper, placeholder,
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
        self.log("CUDA graph captured: full paged decode kv_len=%d layers=%d", kv_len, cache.num_layers)
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
            max(int(runner.block_length), previous_bucket + int(runner.block_length)),
            bucket + 1,
            int(runner.block_length),
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
