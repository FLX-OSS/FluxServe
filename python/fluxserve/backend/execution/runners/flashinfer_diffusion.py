import logging
import os
import time

import torch

from fluxserve.backend.execution.decoders.utils import broadcast_if_needed
from fluxserve.backend.execution.forward_batch_info import ForwardBatch, ForwardMode
from fluxserve.backend.engine.request import RequestState
from fluxserve.backend.execution.runners.block_diffusion import BlockDiffusionRunner
from fluxserve.backend.execution.runners.utils import (
    align_exp2,
    gather_blocks,
    select_batch_sequences_by_mask_number,
)
from fluxserve.backend.layers.dp_attention import get_attention_tp_size
from fluxserve.backend.managers.kvcache import PagedKVCache
from fluxserve.backend.execution.flashinfer_cuda_graph_runner import (
    FlashInferCudaGraphRunner,
)
from fluxserve.backend.layers.attention.utils import _require_flashinfer_paged_prefill

logger = logging.getLogger(__name__)


class FlashInferDiffusionRunner(BlockDiffusionRunner):
    """Block diffusion runner with FlashInfer-specific decode batching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, _allow_flashinfer=True)
        if self.runner_config.attention_backend != "flashinfer":
            raise ValueError(
                "FlashInferDiffusionRunner requires attention_backend='flashinfer'."
            )
        if self.runner_config.kv_cache_layout == "paged":
            _require_flashinfer_paged_prefill()
        self._paged_request_slots: dict[str, int] = {}
        self.flashinfer_graph_runner = (
            FlashInferCudaGraphRunner(
                self.device,
                self.runner_config.cuda_graph_capture_sizes,
                self.runner_config.cuda_graph_log_callback,
                self.model.model.config.num_hidden_layers,
                decode_capture_batch_sizes=(
                    self.runner_config.cuda_graph_capture_batch_sizes
                    or self.runner_config.supported_batch_sizes
                ),
                model_family="llada2",
            )
            if self.enable_flashinfer_attention_graph
            else None
        )
        if self.flashinfer_graph_runner is not None:
            if self.runner_config.enable_prefill_cuda_graph:
                self.flashinfer_graph_runner.log(
                    "CUDA graph enabled: FlashInfer paged prefill "
                    "(batch_size=1, buckets=%s)",
                    self.flashinfer_graph_runner.capture_sizes,
                )
            if self.runner_config.enable_decode_cuda_graph:
                self.flashinfer_graph_runner.log(
                    "CUDA graph enabled: FlashInfer dynamic paged decode "
                    "(batch_sizes=%s, block_length=%d)",
                    ",".join(
                        map(
                            str,
                            self.flashinfer_graph_runner.decode_capture_batch_sizes,
                        )
                    ),
                    self.block_length,
                )

    def _paged_slot(self, request_id: str) -> int:
        existing = self._paged_request_slots.get(request_id)
        if existing is not None:
            return existing

        used_slots = set(self._paged_request_slots.values())
        for slot in range(int(self.server_args.max_num_seqs)):
            if slot not in used_slots:
                self._paged_request_slots[request_id] = slot
                return slot
        raise RuntimeError(
            "paged scheduled more concurrent requests than max_num_seqs="
            f"{self.server_args.max_num_seqs}"
        )

    def shutdown_cuda_graphs(self, *, log: bool = True) -> None:
        if self.flashinfer_graph_runner is not None:
            self.flashinfer_graph_runner.shutdown_llada2(
                self.tp_group.device_group, log=log
            )

    def _release_paged_slot(self, request_id: str) -> None:
        self._paged_request_slots.pop(request_id, None)

    def release_paged_requests(self, request_ids) -> None:
        for request_id in request_ids:
            self._release_paged_slot(str(request_id))

    def allocate_kv_cache(self, batch_size):
        if self.runner_config.kv_cache_layout == "paged":
            if self.flashinfer_graph_runner is None:
                return super().allocate_kv_cache(batch_size)
            config = self.model.model.config
            existing = getattr(self, "past_key_values", None)
            if (
                isinstance(existing, PagedKVCache)
                and not existing.uses_external_page_table
                and existing.batch_size >= int(batch_size)
                and existing.max_length >= self.max_length
                and existing.page_size == int(self.runner_config.page_size)
                and existing.local_kv_heads
                == max(1, config.num_key_value_heads // get_attention_tp_size())
            ):
                return existing
            if isinstance(existing, PagedKVCache):
                self.flashinfer_graph_runner.invalidate_llada2(
                    "KV cache allocation changed"
                )
            return PagedKVCache(
                num_layers=config.num_hidden_layers,
                batch_size=batch_size,
                local_kv_heads=max(
                    1, config.num_key_value_heads // get_attention_tp_size()
                ),
                max_length=self.max_length,
                head_dim=config.hidden_size // config.num_attention_heads,
                page_size=int(self.runner_config.page_size),
                reserve_dummy_page=max(
                    max(self.runner_config.supported_batch_sizes)
                    if self.runner_config.enable_decode_cuda_graph else 0,
                    max(self.runner_config.cuda_graph_capture_sizes)
                    // int(self.runner_config.page_size),
                ),
                dtype=torch.bfloat16,
                device=self.device,
            )
        config = self.model.model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        tp_size = get_attention_tp_size()
        local_kv_heads = max(1, num_kv_heads // tp_size)
        return torch.zeros(
            (
                num_layers,
                2,
                batch_size,
                local_kv_heads,
                self.max_length,
                head_dim,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _use_flashinfer_paged_cache(self) -> bool:
        return (
            self.runner_config.attention_backend == "flashinfer"
            and self.runner_config.kv_cache_layout == "paged"
            and self.runner_config.flashinfer_cache_mode == "paged"
        )

    def _use_flashinfer_paged_prefill(self) -> bool:
        return (
            self._use_flashinfer_paged_cache()
            and self.runner_config.flashinfer_prefill_mode == "paged"
        )

    def _attach_flashinfer_graph(self, forward_batch: ForwardBatch) -> ForwardBatch:
        if self.flashinfer_graph_runner is not None:
            forward_batch.flashinfer_cuda_graph_runner = self.flashinfer_graph_runner
            forward_batch.flashinfer_cuda_graph_dummy_page = int(
                self.past_key_values.dummy_page_id
            )
        return forward_batch

    def _attach_flashinfer_append_metadata(
        self,
        forward_batch: ForwardBatch,
        *,
        q_lens: torch.Tensor,
        kv_lens: torch.Tensor,
    ) -> ForwardBatch:
        """Build compact logical append coordinates once for all layers."""
        import flashinfer

        q_lens = q_lens.to(device=self.device, dtype=torch.int32).contiguous()
        kv_lens = kv_lens.to(device=self.device, dtype=torch.int32).contiguous()
        if q_lens.numel() != kv_lens.numel():
            raise RuntimeError(
                "FlashInfer append metadata requires one q_len per kv_len."
            )
        append_indptr = torch.empty(
            q_lens.numel() + 1, dtype=torch.int32, device=self.device
        )
        append_indptr[0] = 0
        append_indptr[1:] = torch.cumsum(q_lens, dim=0)
        nnz = int(sum(int(x) for x in q_lens.detach().cpu().tolist()))
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            append_indptr, kv_lens, nnz
        )
        forward_batch.flashinfer_append_indptr = append_indptr
        forward_batch.flashinfer_append_batch_indices = batch_indices.contiguous()
        forward_batch.flashinfer_append_positions = positions.contiguous()
        return forward_batch

    def ensure_paged_kv_cache(self, *, num_device_pages: int) -> None:
        if not self._use_flashinfer_paged_prefill():
            raise RuntimeError(
                "paged KV-page execution requires flashinfer paged cache and "
                "flashinfer paged prefill."
            )
        config = self.model.model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        tp_size = get_attention_tp_size()
        local_kv_heads = max(1, num_kv_heads // tp_size)
        max_num_seqs = int(self.server_args.max_num_seqs)
        if (
            isinstance(getattr(self, "past_key_values", None), PagedKVCache)
            and self.past_key_values.num_pages >= int(num_device_pages)
            and self.past_key_values.batch_size >= max_num_seqs
        ):
            return
        if self.flashinfer_graph_runner is not None:
            self.flashinfer_graph_runner.invalidate_llada2(
                "scheduler KV pool allocation changed"
            )
        scheduler_pages = int(num_device_pages)
        self.past_key_values = PagedKVCache(
            num_layers=num_layers,
            batch_size=max_num_seqs,
            local_kv_heads=local_kv_heads,
            max_length=self.max_length,
            head_dim=head_dim,
            page_size=int(self.runner_config.page_size),
            num_pages=scheduler_pages,
            reserve_dummy_page=max(
                max(self.runner_config.supported_batch_sizes)
                if self.runner_config.enable_decode_cuda_graph else 0,
                max(self.runner_config.cuda_graph_capture_sizes)
                // int(self.runner_config.page_size),
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.past_key_values.scheduler_num_pages = scheduler_pages

    def prepare_online_cuda_graphs(self) -> dict[str, int | float]:
        graph_runner = self.flashinfer_graph_runner
        if graph_runner is None:
            return {}
        num_pages = int(self.server_args.scheduler_num_device_pages)
        if num_pages <= 0:
            raise RuntimeError("CUDA graph startup requires final scheduler_num_device_pages")
        started = time.perf_counter()
        self.ensure_paged_kv_cache(num_device_pages=num_pages)
        # The scheduler KV pool is persistent serving state, not graph memory.
        torch.cuda.synchronize(self.device)
        allocated_before = torch.cuda.memory_allocated(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        if self.runner_config.enable_prefill_cuda_graph:
            class _WarmupPrefill:
                extend_prefix_lens = [0]

                def __init__(self, length: int, mask_id: int):
                    self.input_lengths = [length]
                    self.input_ids = [int(mask_id)] * length

            max_pages = max(graph_runner.capture_sizes) // int(self.past_key_values.page_size)
            self.past_key_values.set_page_tables(
                [0], [list(range(1, max_pages + 1))]
            )
            for bucket in sorted(graph_runner.capture_sizes, reverse=True):
                warmup = _WarmupPrefill(int(bucket), int(self.decoder.mask_id))
                self._execute_paged_prefill(warmup, 1, [0])
        if self.runner_config.enable_decode_cuda_graph:
            decode_batch_sizes = graph_runner.decode_capture_batch_sizes
            graph_runner.capture_decode_batch_sizes(self, decode_batch_sizes)
        graph_runner.capture_time_s = time.perf_counter() - started
        graph_runner.record_capture_memory(allocated_before, reserved_before)
        stats = graph_runner.stats()
        graph_runner.log("CUDA graph online startup complete: %s", stats)
        graph_runner.reset_serving_counts()
        return graph_runner.stats()

    def cuda_graph_stats(self) -> dict[str, int | float]:
        if self.flashinfer_graph_runner is None:
            return {}
        return self.flashinfer_graph_runner.stats()

    async def execute_paged_forward_plan(
        self,
        op,
        states_by_id: dict[str, RequestState],
        tokenizer,
    ):
        from fluxserve.backend.engine.executor import ForwardStepResult

        if not self._use_flashinfer_paged_prefill():
            raise RuntimeError(
                "paged currently supports only FlashInfer paged KV execution "
                "with flashinfer_prefill_mode='paged'."
            )
        max_page_id = 0
        for pages in op.occupied_pages:
            if pages:
                max_page_id = max(max_page_id, max(int(p) for p in pages))
        configured_pages = int(getattr(self.server_args, "scheduler_num_device_pages", 0) or 0)
        if configured_pages <= 0:
            raise RuntimeError("paged execution requires scheduler_num_device_pages")
        if max_page_id >= configured_pages:
            raise RuntimeError(
                "scheduler page id exceeds the fixed CUDA-graph KV pool: "
                f"page_id={max_page_id}, num_device_pages={configured_pages}"
            )
        self.ensure_paged_kv_cache(num_device_pages=configured_pages)

        request_ids = list(op.request_ids)
        slot_indices = [self._paged_slot(rid) for rid in request_ids]
        try:
            occupied_pages = [list(map(int, pages)) for pages in op.occupied_pages]
            self.past_key_values.set_page_tables(slot_indices, occupied_pages)

            num_prefill = int(op.num_extends())
            results: list[ForwardStepResult] = []
            if num_prefill:
                self._execute_paged_prefill(op, num_prefill, slot_indices)
                for rid in request_ids[:num_prefill]:
                    if rid in states_by_id:
                        states_by_id[rid].plan_prefill_done = True
                        results.append(ForwardStepResult(rid=rid, token_ids=[], text=""))

            decode_start_row = num_prefill
            if decode_start_row < len(request_ids):
                results.extend(
                    self._execute_paged_decode(
                        op,
                        decode_start_row,
                        slot_indices,
                        states_by_id,
                        tokenizer,
                    )
                )
        except Exception:
            for rid in request_ids:
                self._release_paged_slot(rid)
            raise
        for result in results:
            if result.finished:
                self._release_paged_slot(result.rid)
        return results

    def _execute_paged_prefill(self, op, num_prefill: int, slot_indices: list[int]) -> None:
        input_lengths = [int(x) for x in op.input_lengths[:num_prefill]]
        extend_prefix_lens = [int(x) for x in op.extend_prefix_lens]
        if len(extend_prefix_lens) != num_prefill:
            raise RuntimeError(
                "paged prefill metadata mismatch: "
                f"num_extends={num_prefill}, extend_prefix_lens={len(extend_prefix_lens)}"
            )
        input_ids = list(map(int, op.input_ids))
        chunks = []
        cursor = 0
        for length in input_lengths:
            chunks.append(input_ids[cursor : cursor + length])
            cursor += length
        if cursor != len(input_ids):
            raise RuntimeError(
                "paged prefill input_ids length mismatch: "
                f"consumed={cursor}, total={len(input_ids)}"
            )
        if not any(input_lengths):
            return
        if any(length <= 0 for length in input_lengths):
            raise RuntimeError(
                "mixed zero-length and non-empty paged prefill is unsupported; "
                "the scheduler must schedule zero-length prefill separately"
            )
        max_q_len = max(input_lengths)
        prefill_tokens = torch.full(
            (num_prefill, max_q_len),
            int(self.decoder.mask_id),
            dtype=torch.long,
            device=self.device,
        )
        for row, chunk in enumerate(chunks):
            prefill_tokens[row, : len(chunk)] = torch.tensor(
                chunk,
                dtype=torch.long,
                device=self.device,
            )
        seq_ids = torch.tensor(slot_indices[:num_prefill], dtype=torch.long, device=self.device)
        q_offsets = torch.tensor(extend_prefix_lens, dtype=torch.long, device=self.device)
        q_lens = torch.tensor(input_lengths, dtype=torch.long, device=self.device)
        kv_lens = q_offsets + q_lens
        position_ids = q_offsets.unsqueeze(1) + torch.arange(
            max_q_len,
            dtype=torch.long,
            device=self.device,
        )
        forward_batch = self._make_forward_batch(
            num_prefill * max_q_len,
            is_prefill=True,
        )
        forward_batch = self._make_scheduler_paged_batch(
            seq_ids=seq_ids,
            q_offsets=q_offsets,
            q_lens=q_lens,
            kv_lens=kv_lens,
            forward_batch=forward_batch,
            is_prefill=True,
        )
        num_layers = self.model.model.config.num_hidden_layers
        use_prefill_graph = bool(
            self.flashinfer_graph_runner is not None
            and self.runner_config.enable_prefill_cuda_graph
            and num_prefill == 1
            and int(q_offsets[0]) == 0
            and self.flashinfer_graph_runner.can_run(
                q_len=max_q_len,
                q_offset=0,
                page_size=int(self.past_key_values.page_size),
            )
        )
        if use_prefill_graph:
            self.flashinfer_graph_runner.replay(
                runner=self,
                input_ids=prefill_tokens,
                position_ids=position_ids,
                forward_batch=forward_batch,
            )
        else:
            if self.flashinfer_graph_runner is not None:
                self.flashinfer_graph_runner.prefill_fallback_count += 1
            self.model(
                prefill_tokens,
                use_cache=True,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=[
                    self.past_key_values.layer_paged_kv(layer_id)
                    for layer_id in range(num_layers)
                ],
                forward_batch=forward_batch,
            )
        self.num_forwards += 1

    def _execute_paged_decode(
        self,
        op,
        decode_start_row: int,
        slot_indices: list[int],
        states_by_id: dict[str, RequestState],
        tokenizer,
    ):
        from fluxserve.backend.engine.executor import ForwardStepResult

        request_ids = list(op.request_ids)
        active_rows = [
            row
            for row in range(decode_start_row, len(request_ids))
            if request_ids[row] in states_by_id and not states_by_id[request_ids[row]].finished
        ]
        if not active_rows:
            return []

        block_length = int(self.block_length)
        seq_ids = torch.tensor(
            [slot_indices[row] for row in active_rows],
            dtype=torch.long,
            device=self.device,
        )
        block_starts = []
        total_lengths = []
        for row in active_rows:
            state = states_by_id[request_ids[row]]
            first_block_start = state.aligned_prefill_length(block_length)
            block_start = first_block_start + state.current_decode_block * block_length
            remaining = max(0, state.max_new_tokens - len(state.output_ids))
            block_starts.append(block_start)
            total_lengths.append(block_start + min(block_length, remaining))
        max_total_len = max(start + block_length for start in block_starts)
        decode_tokens = torch.full(
            (int(self.server_args.max_num_seqs), max_total_len),
            int(self.decoder.mask_id),
            dtype=torch.long,
            device=self.device,
        )
        for row in active_rows:
            state = states_by_id[request_ids[row]]
            tokens = state.input_ids + state.output_ids
            seq_id = slot_indices[row]
            decode_tokens[seq_id, : len(tokens)] = torch.tensor(
                tokens,
                dtype=torch.long,
                device=self.device,
            )
        block_start_tensor = torch.zeros(
            decode_tokens.shape[0],
            dtype=torch.long,
            device=self.device,
        )
        for seq_id, block_start in zip(seq_ids.tolist(), block_starts, strict=True):
            block_start_tensor[int(seq_id)] = int(block_start)

        class _PlanTokenArray:
            def __init__(self, data, mask_id, eos_id):
                self.data = data
                self.mask_id = mask_id
                self.eos_id = eos_id

            def select_seqs(self, idx):
                return _PlanTokenArray(self.data[idx].clone(), self.mask_id, self.eos_id)

            def __getitem__(self, idx):
                return self.data[idx]

            def __setitem__(self, idx, vals):
                self.data[idx] = vals

        x = _PlanTokenArray(decode_tokens, self.decoder.mask_id, self.decoder.eos_id)
        num_layers = self.model.model.config.num_hidden_layers
        total_length = max_total_len
        # Prompt boundary per slot, from input_ids alone: output_ids may have
        # had mask/eos tokens filtered, so its length is not a position.
        prompt_lengths = torch.zeros(
            decode_tokens.shape[0], dtype=torch.long, device=self.device
        )
        for row in active_rows:
            prompt_lengths[slot_indices[row]] = len(
                states_by_id[request_ids[row]].input_ids
            )
        edit_budget, row_state = self._make_decode_loop_state(
            decode_tokens.shape[0]
        )
        if getattr(self.decoder, "needs_row_state", False):
            max_decode_iters = self.decoder.max_block_iters
        elif getattr(self.decoder, "needs_editing_inputs", False):
            max_decode_iters = (
                block_length
                + int(getattr(self.runner_config, "max_post_steps", 16))
                + 1
            )
        else:
            max_decode_iters = block_length + 1
        pending_seq_ids = seq_ids
        for _ in range(max_decode_iters):
            before = block_start_tensor[pending_seq_ids].clone()
            current_cache_length = int(torch.max(before + block_length).item())
            self._decode_selected_batch(
                x,
                pending_seq_ids,
                block_start_tensor,
                current_cache_length,
                total_length,
                None,
                num_layers,
                prompt_lengths=prompt_lengths,
                edit_budget=edit_budget,
                row_state=row_state,
            )
            after = block_start_tensor[pending_seq_ids]
            unfinished = after == before
            if not torch.any(unfinished):
                break
            pending_seq_ids = pending_seq_ids[unfinished]
        else:
            stuck_rows = [
                request_ids[row]
                for row in active_rows
                if slot_indices[row] in pending_seq_ids.tolist()
            ]
            raise RuntimeError(
                "paged decode block did not finish within "
                f"{max_decode_iters} iterations for request(s) {stuck_rows} "
                f"(post_steps={edit_budget.post_steps[pending_seq_ids].tolist()})."
            )

        results = []
        eos_id = int(self.decoder.eos_id)
        mask_id = int(self.decoder.mask_id)
        for local_idx, row in enumerate(active_rows):
            rid = request_ids[row]
            state = states_by_id[rid]
            block_start = block_starts[local_idx]
            remaining = max(0, state.max_new_tokens - len(state.output_ids))
            # The first aligned block may start inside the prompt. Those fixed
            # prompt tokens participate in attention but are not completion.
            generated_start = max(block_start, len(state.input_ids))
            raw = x.data[
                slot_indices[row], generated_start : block_start + block_length
            ]
            generated = raw.detach().cpu().tolist()[:remaining]
            finish_reason = None
            if not state.ignore_eos and eos_id in generated:
                generated = generated[: generated.index(eos_id)]
                finish_reason = "stop"
            if state.ignore_eos:
                generated = [tok for tok in generated if tok != mask_id]
            else:
                generated = [tok for tok in generated if tok != mask_id and tok != eos_id]
            projected = len(state.output_ids) + len(generated)
            finished = finish_reason == "stop" or projected >= state.max_new_tokens
            if finished and finish_reason is None:
                finish_reason = "length"
            text = tokenizer.decode(generated, skip_special_tokens=True)
            results.append(
                ForwardStepResult(
                    rid=rid,
                    token_ids=generated,
                    text=text,
                    finished=finished,
                    finish_reason=finish_reason,
                    reserve_tokens=0 if finished else block_length,
                    decode_block_completed=True,
                    trajectory_metrics={
                        "block_length": block_length,
                        "transferred_tokens": len(generated),
                        "remaining_masks": max(0, block_length - len(generated)),
                        "progress": len(generated) / max(block_length, 1),
                    },
                )
            )
        return results

    def _make_scheduler_paged_batch(
        self,
        *,
        seq_ids: torch.Tensor,
        q_offsets: torch.Tensor,
        q_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        forward_batch: ForwardBatch | None,
        is_prefill: bool,
    ) -> ForwardBatch:
        if forward_batch is None:
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE
            )
        self._attach_flashinfer_graph(forward_batch)
        qo_values = [0]
        for q_len in q_lens.detach().cpu().tolist():
            qo_values.append(qo_values[-1] + int(q_len))
        kv_indptr, kv_indices, last_page_len = self.past_key_values.flashinfer_paged_metadata(
            seq_ids=seq_ids,
            lengths=kv_lens,
        )
        max_q_len = int(torch.max(q_lens).item())
        positions = q_offsets.unsqueeze(1) + torch.arange(
            max_q_len,
            dtype=torch.long,
            device=self.device,
        )
        # Padded prefill columns must still reference an allocated page. The
        # query lengths tell FlashInfer to ignore these repeated positions.
        last_valid_positions = (q_offsets + q_lens - 1).unsqueeze(1)
        positions = torch.minimum(positions, last_valid_positions)
        forward_batch.flashinfer_seq_ids = seq_ids
        forward_batch.flashinfer_slot_mapping = self.past_key_values.slot_mapping(
            seq_ids,
            positions,
        )
        forward_batch.flashinfer_kv_lens_cpu = tuple(
            int(x) for x in kv_lens.detach().cpu().tolist()
        )
        forward_batch.flashinfer_kv_lens = kv_lens.to(torch.int32)
        forward_batch.flashinfer_q_offsets_cpu = tuple(
            int(x) for x in q_offsets.detach().cpu().tolist()
        )
        forward_batch.flashinfer_q_offsets = q_offsets.to(torch.int32)
        forward_batch.flashinfer_qo_indptr_cpu = tuple(qo_values)
        forward_batch.flashinfer_qo_indptr = torch.tensor(
            qo_values,
            dtype=torch.int32,
            device=self.device,
        )
        forward_batch.flashinfer_kv_indptr = kv_indptr
        forward_batch.flashinfer_kv_indptr_cpu = tuple(
            int(x) for x in kv_indptr.detach().cpu().tolist()
        )
        forward_batch.flashinfer_paged_kv_indices = kv_indices
        forward_batch.flashinfer_paged_kv_indices_cpu = tuple(
            int(x) for x in kv_indices.detach().cpu().tolist()
        )
        forward_batch.flashinfer_paged_kv_last_page_len = last_page_len
        forward_batch.flashinfer_paged_kv_last_page_len_cpu = tuple(
            int(x) for x in last_page_len.detach().cpu().tolist()
        )
        forward_batch.flashinfer_block_length = int(self.block_length)
        forward_batch.flashinfer_page_size = int(self.past_key_values.page_size)
        if is_prefill:
            forward_batch.use_flashinfer_paged_prefill = True
            forward_batch.flashinfer_prefill_lens_cpu = tuple(
                int(x) for x in q_lens.detach().cpu().tolist()
            )
        else:
            forward_batch.use_flashinfer_decode = False
            forward_batch.use_flashinfer_paged_decode = True
        self._attach_flashinfer_append_metadata(
            forward_batch, q_lens=q_lens, kv_lens=kv_lens
        )
        return self._attach_flashinfer_graph(forward_batch)

    def _make_flashinfer_decode_batch(
        self,
        seq_ids: torch.Tensor,
        decoding_start: torch.Tensor,
        forward_batch: ForwardBatch | None,
    ) -> ForwardBatch:
        if forward_batch is None:
            forward_batch = ForwardBatch(forward_mode=ForwardMode.DECODE)

        kv_lens_cpu = tuple(
            int(length)
            for length in (
                decoding_start[seq_ids] + self.block_length
            ).detach().cpu().tolist()
        )
        q_offsets_cpu = tuple(
            int(offset) for offset in decoding_start[seq_ids].detach().cpu().tolist()
        )
        q_lens_cpu = (self.block_length,) * len(kv_lens_cpu)
        qo_values = [0]
        kv_values = [0]
        for q_len, kv_len in zip(q_lens_cpu, kv_lens_cpu, strict=True):
            qo_values.append(qo_values[-1] + q_len)
            kv_values.append(kv_values[-1] + kv_len)

        forward_batch.use_flashinfer_decode = True
        forward_batch.flashinfer_kv_lens_cpu = kv_lens_cpu
        forward_batch.flashinfer_qo_indptr = torch.tensor(
            qo_values, dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_kv_indptr = torch.tensor(
            kv_values, dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_kv_lens = torch.tensor(
            kv_lens_cpu, dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_q_offsets_cpu = q_offsets_cpu
        forward_batch.flashinfer_q_offsets = torch.tensor(
            q_offsets_cpu, dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_kv_offsets_cpu = (0,) * len(kv_lens_cpu)
        forward_batch.flashinfer_kv_offsets = torch.zeros(
            len(kv_lens_cpu), dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_block_length = self.block_length
        return forward_batch

    def _make_flashinfer_prefill_batch(
        self,
        *,
        seq_ids: torch.Tensor,
        prefilling_lengths: torch.Tensor,
        prefilling_length: int,
        forward_batch: ForwardBatch | None,
    ) -> ForwardBatch:
        if forward_batch is None:
            forward_batch = ForwardBatch(forward_mode=ForwardMode.EXTEND)

        prefill_lens_cpu = tuple(
            int(length)
            for length in prefilling_lengths[seq_ids].detach().cpu().tolist()
        )
        block_length = int(self.block_length)
        for prefill_len in prefill_lens_cpu:
            if prefill_len % block_length != 0:
                raise RuntimeError(
                    "FlashInfer ragged prefill requires block-aligned lengths: "
                    f"got prefill_len={prefill_len}, block_length={block_length}."
                )
        indptr_values = [0]
        for prefill_len in prefill_lens_cpu:
            indptr_values.append(indptr_values[-1] + int(prefill_len))
        q_offsets_cpu = (0,) * len(prefill_lens_cpu)
        kv_offsets_cpu = (0,) * len(prefill_lens_cpu)

        forward_batch.use_flashinfer_prefill = True
        forward_batch.flashinfer_prefill_lens_cpu = prefill_lens_cpu
        forward_batch.flashinfer_kv_lens_cpu = prefill_lens_cpu
        forward_batch.flashinfer_qo_indptr_cpu = tuple(indptr_values)
        forward_batch.flashinfer_qo_indptr = torch.tensor(
            indptr_values, dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_kv_indptr = torch.tensor(
            indptr_values, dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_kv_indptr_cpu = tuple(indptr_values)
        forward_batch.flashinfer_kv_lens = torch.tensor(
            prefill_lens_cpu, dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_q_offsets_cpu = q_offsets_cpu
        forward_batch.flashinfer_q_offsets = torch.tensor(
            q_offsets_cpu, dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_kv_offsets_cpu = kv_offsets_cpu
        forward_batch.flashinfer_kv_offsets = torch.tensor(
            kv_offsets_cpu, dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_block_length = block_length
        return forward_batch

    def _make_flashinfer_paged_prefill_batch(
        self,
        *,
        seq_ids: torch.Tensor,
        prefilling_lengths: torch.Tensor,
        forward_batch: ForwardBatch | None,
    ) -> ForwardBatch:
        if forward_batch is None:
            forward_batch = ForwardBatch(forward_mode=ForwardMode.EXTEND)
        if not isinstance(self.past_key_values, PagedKVCache):
            raise RuntimeError(
                "FlashInfer paged prefill requires PagedKVCache as past_key_values."
            )
        prefill_lens_cpu = tuple(
            int(length)
            for length in prefilling_lengths[seq_ids].detach().cpu().tolist()
        )
        for prefill_len in prefill_lens_cpu:
            if prefill_len % self.block_length != 0:
                raise RuntimeError(
                    "FlashInfer paged prefill requires block-aligned lengths: "
                    f"got prefill_len={prefill_len}, block_length={self.block_length}."
                )
        qo_values = [0]
        for prefill_len in prefill_lens_cpu:
            qo_values.append(qo_values[-1] + int(prefill_len))
        kv_indptr, kv_indices, last_page_len = (
            self.past_key_values.flashinfer_paged_metadata(
                seq_ids=seq_ids,
                lengths=prefilling_lengths[seq_ids],
            )
        )
        positions = torch.arange(
            max(prefill_lens_cpu),
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0).repeat(seq_ids.shape[0], 1)
        forward_batch.use_flashinfer_paged_prefill = True
        forward_batch.flashinfer_seq_ids = seq_ids
        forward_batch.flashinfer_slot_mapping = self.past_key_values.slot_mapping(
            seq_ids,
            positions,
        )
        forward_batch.flashinfer_prefill_lens_cpu = prefill_lens_cpu
        forward_batch.flashinfer_kv_lens_cpu = prefill_lens_cpu
        forward_batch.flashinfer_kv_lens = torch.tensor(
            prefill_lens_cpu,
            dtype=torch.int32,
            device=self.device,
        )
        forward_batch.flashinfer_q_offsets_cpu = (0,) * len(prefill_lens_cpu)
        forward_batch.flashinfer_q_offsets = torch.zeros(
            len(prefill_lens_cpu), dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_kv_offsets_cpu = (0,) * len(prefill_lens_cpu)
        forward_batch.flashinfer_kv_offsets = torch.zeros(
            len(prefill_lens_cpu), dtype=torch.int32, device=self.device
        )
        forward_batch.flashinfer_qo_indptr_cpu = tuple(qo_values)
        forward_batch.flashinfer_qo_indptr = torch.tensor(
            qo_values,
            dtype=torch.int32,
            device=self.device,
        )
        forward_batch.flashinfer_kv_indptr = kv_indptr
        forward_batch.flashinfer_kv_indptr_cpu = tuple(
            int(x) for x in kv_indptr.detach().cpu().tolist()
        )
        forward_batch.flashinfer_paged_kv_indices = kv_indices
        forward_batch.flashinfer_paged_kv_indices_cpu = tuple(
            int(x) for x in kv_indices.detach().cpu().tolist()
        )
        forward_batch.flashinfer_paged_kv_last_page_len = last_page_len
        forward_batch.flashinfer_paged_kv_last_page_len_cpu = tuple(
            int(x) for x in last_page_len.detach().cpu().tolist()
        )
        forward_batch.flashinfer_block_length = int(self.block_length)
        forward_batch.flashinfer_page_size = int(self.past_key_values.page_size)
        self._attach_flashinfer_append_metadata(
            forward_batch,
            q_lens=torch.tensor(prefill_lens_cpu, device=self.device),
            kv_lens=forward_batch.flashinfer_kv_lens,
        )
        return self._attach_flashinfer_graph(forward_batch)

    def _prefill_batches(
        self,
        x,
        prefilling_lengths,
        non_mask_number,
        attention_mask,
        pos_ids,
        num_layers,
        mini_batch_size,
    ):
        if self._use_flashinfer_paged_prefill():
            return self._prefill_batches_paged(
                x,
                prefilling_lengths,
                pos_ids,
                num_layers,
                mini_batch_size,
            )
        if self.runner_config.flashinfer_prefill_mode != "ragged":
            return super()._prefill_batches(
                x,
                prefilling_lengths,
                non_mask_number,
                attention_mask,
                pos_ids,
                num_layers,
                mini_batch_size,
            )

        prefilling_flag = prefilling_lengths > 0
        while torch.any(prefilling_flag):
            seq_ids = select_batch_sequences_by_mask_number(
                x, prefilling_flag, self.decoder.mask_id, mini_batch_size
            )
            prefilling_length = int(torch.max(prefilling_lengths[seq_ids]).item())
            prefilling_x = x.select_seqs(seq_ids)
            forward_batch = self._make_forward_batch(
                len(seq_ids) * prefilling_length, is_prefill=True
            )
            forward_batch = self._make_flashinfer_prefill_batch(
                seq_ids=seq_ids,
                prefilling_lengths=prefilling_lengths,
                prefilling_length=prefilling_length,
                forward_batch=forward_batch,
            )

            output = self.model(
                prefilling_x[:, :prefilling_length].contiguous(),
                use_cache=True,
                attention_mask=None,
                position_ids=pos_ids[seq_ids, :prefilling_length].contiguous(),
                forward_batch=forward_batch,
            )

            inner_shape = output.past_key_values[0].shape
            prefilling_kv = torch.stack(output.past_key_values, dim=0).reshape(
                num_layers, 2, *inner_shape
            )
            for local_idx, sample_len in enumerate(prefilling_lengths[seq_ids]):
                self._write_prefill_kv_cache(
                    global_idx=seq_ids[local_idx],
                    local_idx=local_idx,
                    sample_len=int(sample_len.item()),
                    prefilling_kv=prefilling_kv,
                )
            self.num_forwards += 1
            prefilling_flag[seq_ids] = False

    def _prefill_batches_paged(
        self,
        x,
        prefilling_lengths,
        pos_ids,
        num_layers,
        mini_batch_size,
    ):
        prefilling_flag = prefilling_lengths > 0
        while torch.any(prefilling_flag):
            seq_ids = select_batch_sequences_by_mask_number(
                x, prefilling_flag, self.decoder.mask_id, mini_batch_size
            )
            prefilling_length = int(torch.max(prefilling_lengths[seq_ids]).item())
            prefilling_x = x.select_seqs(seq_ids)
            forward_batch = self._make_forward_batch(
                len(seq_ids) * prefilling_length,
                is_prefill=True,
            )
            forward_batch = self._make_flashinfer_paged_prefill_batch(
                seq_ids=seq_ids,
                prefilling_lengths=prefilling_lengths,
                forward_batch=forward_batch,
            )
            input_ids = prefilling_x[:, :prefilling_length].contiguous()
            position_ids = pos_ids[seq_ids, :prefilling_length].contiguous()
            if (
                self.flashinfer_graph_runner is not None
                and self.runner_config.enable_prefill_cuda_graph
                and len(seq_ids) == 1
                and self.flashinfer_graph_runner.can_run(
                    q_len=prefilling_length,
                    q_offset=0,
                    page_size=int(self.past_key_values.page_size),
                )
            ):
                self.flashinfer_graph_runner.replay(
                    runner=self,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    forward_batch=forward_batch,
                )
            else:
                if self.flashinfer_graph_runner is not None:
                    self.flashinfer_graph_runner.prefill_fallback_count += 1
                self.model(
                    input_ids,
                    use_cache=True,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=[
                        self.past_key_values.layer_paged_kv(layer_id)
                        for layer_id in range(num_layers)
                    ],
                    forward_batch=forward_batch,
                )

            # Paged prefill attention writes K/V into PagedKVCache per layer.
            self.num_forwards += 1
            prefilling_flag[seq_ids] = False

    def _make_decode_forward_batch(
        self,
        seq_ids: torch.Tensor,
        decoding_start: torch.Tensor,
    ) -> ForwardBatch:
        forward_batch = self._make_forward_batch(
            len(seq_ids) * self.block_length,
            is_prefill=False,
        )
        forward_batch = self._make_flashinfer_decode_batch(
            seq_ids,
            decoding_start,
            forward_batch,
        )
        if not self._use_flashinfer_paged_cache():
            return forward_batch
        if not isinstance(self.past_key_values, PagedKVCache):
            raise RuntimeError(
                "FlashInfer paged cache mode requires PagedKVCache as past_key_values."
            )
        kv_lens = decoding_start[seq_ids] + self.block_length
        kv_indptr, kv_indices, last_page_len = (
            self.past_key_values.flashinfer_paged_metadata(
                seq_ids=seq_ids,
                lengths=kv_lens,
            )
        )
        forward_batch.use_flashinfer_decode = False
        forward_batch.use_flashinfer_paged_decode = True
        forward_batch.flashinfer_seq_ids = seq_ids
        positions = decoding_start[seq_ids].unsqueeze(1) + torch.arange(
            self.block_length,
            device=self.device,
            dtype=torch.long,
        )
        forward_batch.flashinfer_slot_mapping = self.past_key_values.slot_mapping(
            seq_ids,
            positions,
        )
        forward_batch.flashinfer_kv_indptr = kv_indptr
        forward_batch.flashinfer_kv_indptr_cpu = tuple(
            int(x) for x in kv_indptr.detach().cpu().tolist()
        )
        forward_batch.flashinfer_paged_kv_indices = kv_indices
        forward_batch.flashinfer_paged_kv_indices_cpu = tuple(
            int(x) for x in kv_indices.detach().cpu().tolist()
        )
        forward_batch.flashinfer_paged_kv_last_page_len = last_page_len
        forward_batch.flashinfer_paged_kv_last_page_len_cpu = tuple(
            int(x) for x in last_page_len.detach().cpu().tolist()
        )
        forward_batch.flashinfer_page_size = int(self.past_key_values.page_size)
        self._attach_flashinfer_append_metadata(
            forward_batch,
            q_lens=torch.full(
                (seq_ids.numel(),), self.block_length, device=self.device
            ),
            kv_lens=kv_lens,
        )
        return self._attach_flashinfer_graph(forward_batch)

    def _decode_batches(
        self,
        x,
        decoding_start,
        total_length,
        pos_ids,
        num_layers,
        mini_batch_size,
        prompt_lengths=None,
    ):
        edit_budget, row_state = self._make_decode_loop_state(
            decoding_start.shape[0]
        )
        if self.runner_config.flashinfer_decode_batch_mode == "default":
            return self._decode_batches_default(
                x,
                decoding_start,
                total_length,
                pos_ids,
                num_layers,
                mini_batch_size,
                prompt_lengths=prompt_lengths,
                edit_budget=edit_budget,
                row_state=row_state,
            )
        return self._decode_batches_max_batch(
            x,
            decoding_start,
            total_length,
            pos_ids,
            num_layers,
            mini_batch_size,
            prompt_lengths=prompt_lengths,
            edit_budget=edit_budget,
            row_state=row_state,
        )

    def _decode_batches_default(
        self,
        x,
        decoding_start,
        total_length,
        pos_ids,
        num_layers,
        mini_batch_size,
        prompt_lengths=None,
        edit_budget=None,
        row_state=None,
    ):
        decoding_flag = (decoding_start + self.block_length) <= total_length
        while torch.any(decoding_flag):
            current_cache_length = max(
                self.runner_config.max_cache_length_align,
                align_exp2(
                    int(torch.min(decoding_start[decoding_flag]).item())
                    + self.block_length
                ),
            )
            current_cache_length = min(current_cache_length, self.max_length)
            current_cache_flag = decoding_flag & (
                (decoding_start + self.block_length) <= current_cache_length
            )
            while torch.any(current_cache_flag):
                seq_ids = select_batch_sequences_by_mask_number(
                    x, current_cache_flag, self.decoder.mask_id, mini_batch_size
                )
                self._decode_selected_batch(
                    x,
                    seq_ids,
                    decoding_start,
                    current_cache_length,
                    total_length,
                    pos_ids,
                    num_layers,
                    prompt_lengths=prompt_lengths,
                    edit_budget=edit_budget,
                    row_state=row_state,
                )
                decoding_flag = decoding_flag & (
                    (decoding_start + self.block_length) <= total_length
                )
                current_cache_flag = decoding_flag & (
                    (decoding_start + self.block_length) <= current_cache_length
                )

    def _decode_batches_max_batch(
        self,
        x,
        decoding_start,
        total_length,
        pos_ids,
        num_layers,
        mini_batch_size,
        prompt_lengths=None,
        edit_budget=None,
        row_state=None,
    ):
        decoding_flag = (decoding_start + self.block_length) <= total_length
        while torch.any(decoding_flag):
            seq_ids = select_batch_sequences_by_mask_number(
                x, decoding_flag, self.decoder.mask_id, mini_batch_size
            )
            current_cache_length = int(
                torch.max(decoding_start[seq_ids] + self.block_length).item()
            )
            self._decode_selected_batch(
                x,
                seq_ids,
                decoding_start,
                current_cache_length,
                total_length,
                pos_ids,
                num_layers,
                prompt_lengths=prompt_lengths,
                edit_budget=edit_budget,
                row_state=row_state,
            )
            decoding_flag = decoding_flag & (
                (decoding_start + self.block_length) <= total_length
            )

    def _decode_selected_batch(
        self,
        x,
        seq_ids,
        decoding_start,
        current_cache_length,
        total_length,
        pos_ids,
        num_layers,
        prompt_lengths=None,
        edit_budget=None,
        row_state=None,
    ):
        actual_batch_size = len(seq_ids)
        # CUDA graphs have exact power-of-two batch shapes. Split irregular
        # batches instead of padding them: padded rows would still enter the
        # MoE router, experts, shared experts, and distributed collectives.
        if (
            self.flashinfer_graph_runner is not None
            and self.runner_config.enable_decode_cuda_graph
            and self._use_flashinfer_paged_cache()
        ):
            parts = self.flashinfer_graph_runner.decompose_batch_size(
                len(seq_ids),
                self.flashinfer_graph_runner.decode_capture_batch_sizes,
            )
            if self.runner_config.decode_cuda_graph_mode != "padded" and len(parts) > 1:
                self.flashinfer_graph_runner.record_decode_decomposition(len(parts))
                start = 0
                for part in parts:
                    component_ids = seq_ids[start : start + part]
                    component_cache_length = int(
                        torch.max(
                            decoding_start[component_ids] + self.block_length
                        ).item()
                    )
                    self._decode_selected_batch(
                        x,
                        component_ids,
                        decoding_start,
                        component_cache_length,
                        total_length,
                        pos_ids,
                        num_layers,
                        prompt_lengths=prompt_lengths,
                        edit_budget=edit_budget,
                        row_state=row_state,
                    )
                    start += part
                return
        decoding_x = x.select_seqs(seq_ids)
        decoding_block = gather_blocks(
            decoding_x.data, decoding_start[seq_ids], self.block_length
        )
        if self._use_flashinfer_paged_cache():
            decoding_past_key_values = None
        else:
            decoding_past_key_values = self.past_key_values[
                :, :, seq_ids, :, :current_cache_length
            ]
        decoding_pos_ids = torch.arange(
            self.block_length, device=self.device, dtype=torch.long
        ).unsqueeze(0).repeat(seq_ids.shape[0], 1)
        decoding_pos_ids = decoding_pos_ids + decoding_start[seq_ids].unsqueeze(1)
        forward_batch = self._make_decode_forward_batch(seq_ids, decoding_start)
        if self._use_flashinfer_paged_cache():
            past_key_values = [
                self.past_key_values.layer_paged_kv(layer_id)
                for layer_id in range(num_layers)
            ]
        else:
            past_key_values = decoding_past_key_values
        use_decode_graph = bool(
            self.flashinfer_graph_runner is not None
            and self.runner_config.enable_decode_cuda_graph
            and self._use_flashinfer_paged_cache()
            and self.flashinfer_graph_runner.can_run_decode(
                batch_size=len(seq_ids),
                q_len=self.block_length,
                kv_len=current_cache_length,
                mode=self.runner_config.decode_cuda_graph_mode,
            )
        )
        fused_decoder = bool(getattr(self.decoder, "graph_fused_step", False))
        fused_replay = None
        if use_decode_graph:
            graph_prompt_positions = None
            graph_allow_edit = None
            if fused_decoder:
                if prompt_lengths is None or edit_budget is None:
                    raise RuntimeError(
                        "joint_threshold decoding requires prompt_lengths and "
                        "an edit budget; the decode loop was not given them."
                    )
                block_offsets = torch.arange(
                    self.block_length, device=self.device
                ).unsqueeze(0) + decoding_start[seq_ids].unsqueeze(1)
                graph_prompt_positions = block_offsets < prompt_lengths[
                    seq_ids
                ].unsqueeze(1)
                graph_allow_edit = edit_budget.allow_edit(seq_ids)
            replayed = self.flashinfer_graph_runner.replay_decode(
                runner=self,
                input_ids=decoding_block,
                position_ids=decoding_pos_ids,
                forward_batch=forward_batch,
                prompt_positions=graph_prompt_positions,
                allow_edit=graph_allow_edit,
            )
            output = None
            if fused_decoder and replayed.x_updated is not None:
                fused_replay = replayed
            else:
                logits = self.model._get_logits(replayed.hidden_states)
        else:
            if self.flashinfer_graph_runner is not None:
                self.flashinfer_graph_runner.decode_fallback_count += 1
            output = self.model(
                decoding_block,
                use_cache=True,
                position_ids=decoding_pos_ids,
                past_key_values=past_key_values,
                forward_batch=forward_batch,
            )
            logits = output.logits[: len(seq_ids)]

        if fused_replay is not None:
            # The lm_head, joint selection, and block-finished predicate all
            # ran inside the replayed graph; only the scatter into the token
            # array (dynamic total length) stays eager.
            T = decoding_x.data.shape[1]
            flat_idx = block_offsets.clamp(max=T - 1) + torch.arange(
                actual_batch_size, device=self.device
            ).unsqueeze(1) * T
            decoding_x.data.view(-1)[flat_idx] = fused_replay.x_updated
            broadcast_if_needed(decoding_x.data)
            had_mask = fused_replay.had_mask
            changed = fused_replay.changed
            block_finished = fused_replay.block_finished
        else:
            decoder_kwargs = self._decoder_editing_kwargs(
                seq_ids, prompt_lengths, edit_budget, row_state
            )
            step_stats = self.decoder.batch_decode(
                logits,
                decoding_start[seq_ids],
                decoding_x,
                self.block_length,
                **decoder_kwargs,
            )

            # A block is finished on an iteration whose pre-update tokens had
            # no masks AND that applied no update. Equivalent to the old `no
            # masks` test for the 2.0 decoders (they only write masked
            # positions), and the per-row vectorization of the LLaDA2.1
            # reference's stability break. `after` is gathered from x.data
            # after the decoder's broadcast, so every rank computes identical
            # predicates.
            after = gather_blocks(
                decoding_x.data, decoding_start[seq_ids], self.block_length
            )
            had_mask = (decoding_block == self.decoder.mask_id).any(dim=1)
            changed = (after != decoding_block).any(dim=1)
            block_finished = (~had_mask) & (~changed)
        if fused_replay is not None:
            step_stats = None
        self._record_block_profile(
            seq_ids,
            decoding_start[seq_ids],
            step_stats,
            block_finished,
        )
        if edit_budget is not None:
            edit_budget.update(seq_ids, had_mask, changed, block_finished)
        if output is not None:
            self._update_finished_kv_cache(
                output,
                seq_ids,
                decoding_start,
                block_finished,
                current_cache_length,
                num_layers,
            )

        decoding_start[seq_ids] += block_finished.long() * self.block_length
        x[seq_ids] = decoding_x.data

        if self.early_stop:
            # Branchless on purpose: an `if eos_mask.any()` here forces a
            # GPU->CPU sync every decode iteration and stalls the pipeline.
            eos_mask = torch.any(x[seq_ids] == self.decoder.eos_id, dim=1) & block_finished
            decoding_start[seq_ids] = torch.where(
                eos_mask,
                torch.full_like(decoding_start[seq_ids], total_length),
                decoding_start[seq_ids],
            )

        self.num_forwards += 1

    def _update_finished_kv_cache(
        self,
        output,
        seq_ids,
        decoding_start,
        block_finished,
        current_cache_length,
        num_layers,
    ):
        if not self._use_flashinfer_paged_cache():
            return super()._update_finished_kv_cache(
                output,
                seq_ids,
                decoding_start,
                block_finished,
                current_cache_length,
                num_layers,
            )
        if not torch.any(block_finished):
            return
        decoding_kv = torch.stack(output.past_key_values, dim=0).reshape(
            num_layers,
            2,
            *output.past_key_values[0].shape,
        )
        for local_idx in block_finished.nonzero(as_tuple=True)[0].tolist():
            self.past_key_values.write_range(
                seq_id=seq_ids[local_idx],
                start=int(decoding_start[seq_ids[local_idx]].item()),
                kv=decoding_kv[:, :, local_idx],
            )
