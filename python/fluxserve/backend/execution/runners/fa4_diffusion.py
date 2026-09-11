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

import torch

from fluxserve.backend.engine.request import RequestState
from fluxserve.backend.execution.forward_batch_info import ForwardBatch, ForwardMode
from fluxserve.backend.execution.runners.block_diffusion import BlockDiffusionRunner
from fluxserve.backend.execution.runners.utils import (
    gather_blocks,
    select_batch_sequences_by_mask_number,
)
from fluxserve.backend.layers.attention.fa4 import validate_fa4_runtime
from fluxserve.backend.layers.attention.metadata import (
    build_block_diffusion_paged_metadata,
)
from fluxserve.backend.layers.dp_attention import get_attention_tp_size
from fluxserve.backend.managers.kvcache import PagedKVCache


class FA4DiffusionRunner(BlockDiffusionRunner):
    """LLaDA 2.x runner backed directly by standalone FlashAttention-4."""

    def __init__(self, *args, **kwargs):
        runner_config = kwargs.get("runner_config")
        if runner_config is None and len(args) >= 3:
            runner_config = args[2]
        device = kwargs.get("device", args[3] if len(args) >= 4 else "cuda")
        model_config = kwargs.get("model_config", args[0] if args else None)
        if runner_config is None or runner_config.attention_backend != "fa4":
            raise ValueError("FA4DiffusionRunner requires attention_backend='fa4'.")
        if runner_config.kv_cache_layout != "paged":
            raise ValueError("FA4DiffusionRunner requires kv_cache_layout='paged'.")
        if runner_config.enable_prefill_cuda_graph:
            raise ValueError("FA4 supports --use-decode-cuda-graph; prefill graph is unsupported")
        if runner_config.enable_decode_cuda_graph:
            server_args = kwargs.get("server_args", args[1] if len(args) > 1 else None)
            if any(int(getattr(server_args, name, 1)) != 1 for name in ("tp_size", "ep_size", "dp_size", "pp_size")):
                raise ValueError("FA4 decode CUDA graph currently requires TP/EP/DP/PP=1")
            if runner_config.decode_cuda_graph_mode != "padded":
                raise ValueError("FA4 decode CUDA graph requires --cuda-graph-decode-mode padded")
            if int(runner_config.page_size or runner_config.block_length) != int(runner_config.block_length):
                raise ValueError("FA4 decode graph requires page_size == block_length")
            sizes = runner_config.cuda_graph_capture_batch_sizes or runner_config.supported_batch_sizes
            if max(sizes) < int(server_args.max_num_seqs):
                raise ValueError("FA4 decode graph buckets must cover max_num_seqs")
        if int(runner_config.page_size or runner_config.block_length) % 16 != 0:
            raise ValueError("FA4 page_size must be a multiple of 16.")
        architecture_names = " ".join(
            str(name).lower()
            for name in (getattr(model_config, "architectures", ()) or ())
        )
        model_type = str(getattr(model_config, "model_type", "")).lower()
        if "llada2" not in architecture_names and "llada2" not in model_type:
            raise ValueError("FA4DiffusionRunner currently supports LLaDA 2.x only.")
        validate_fa4_runtime(device)
        super().__init__(*args, **kwargs)
        self._paged_request_slots: dict[str, int] = {}
        self.fa4_graph_runner = None
        if runner_config.enable_decode_cuda_graph:
            from fluxserve.backend.execution.fa4_cuda_graph_runner import FA4CudaGraphRunner
            self.fa4_graph_runner = FA4CudaGraphRunner(
                runner_config.cuda_graph_capture_batch_sizes or runner_config.supported_batch_sizes
            )

    def _use_unbounded_paged_prefill(self) -> bool:
        return True

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

    def _release_paged_slot(self, request_id: str) -> None:
        self._paged_request_slots.pop(request_id, None)

    def release_paged_requests(self, request_ids) -> None:
        for request_id in request_ids:
            self._release_paged_slot(str(request_id))

    def _make_paged_batch(
        self,
        *,
        seq_ids: torch.Tensor,
        q_offsets: torch.Tensor,
        q_lens: torch.Tensor,
        max_input_len: int,
        is_prefill: bool,
        forward_batch: ForwardBatch | None = None,
    ) -> ForwardBatch:
        if not isinstance(self.past_key_values, PagedKVCache):
            raise RuntimeError("FA4 requires FluxServe PagedKVCache.")
        if forward_batch is None:
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE
            )
        selected_page_table = self.past_key_values.page_table.index_select(
            0, seq_ids.to(device=self.past_key_values.device, dtype=torch.long)
        )
        forward_batch.paged_attention_metadata = build_block_diffusion_paged_metadata(
            phase="prefill" if is_prefill else "decode",
            q_offsets=q_offsets,
            q_lens=q_lens,
            page_table=selected_page_table,
            max_input_len=max_input_len,
            block_length=int(self.block_length),
            page_size=int(self.past_key_values.page_size),
        )
        return forward_batch

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
        del non_mask_number, attention_mask
        prefilling_flag = prefilling_lengths > 0
        while torch.any(prefilling_flag):
            seq_ids = select_batch_sequences_by_mask_number(
                x, prefilling_flag, self.decoder.mask_id, mini_batch_size
            )
            q_lens = prefilling_lengths[seq_ids].to(torch.long)
            max_q_len = int(torch.max(q_lens).item())
            prefilling_x = x.select_seqs(seq_ids)
            forward_batch = self._make_forward_batch(
                len(seq_ids) * max_q_len,
                is_prefill=True,
            )
            forward_batch = self._make_paged_batch(
                seq_ids=seq_ids,
                q_offsets=torch.zeros_like(q_lens),
                q_lens=q_lens,
                max_input_len=max_q_len,
                is_prefill=True,
                forward_batch=forward_batch,
            )
            self.model(
                prefilling_x[:, :max_q_len].contiguous(),
                use_cache=True,
                attention_mask=None,
                position_ids=pos_ids[seq_ids, :max_q_len].contiguous(),
                past_key_values=[
                    self.past_key_values.layer_paged_kv(layer_id)
                    for layer_id in range(num_layers)
                ],
                forward_batch=forward_batch,
            )
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
        q_offsets = decoding_start[seq_ids].to(torch.long)
        q_lens = torch.full_like(q_offsets, int(self.block_length))
        return self._make_paged_batch(
            seq_ids=seq_ids,
            q_offsets=q_offsets,
            q_lens=q_lens,
            max_input_len=int(self.block_length),
            is_prefill=False,
            forward_batch=forward_batch,
        )

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
        edit_budget, row_state = self._make_decode_loop_state(decoding_start.shape[0])
        decoding_flag = (decoding_start + self.block_length) <= total_length
        while torch.any(decoding_flag):
            seq_ids = select_batch_sequences_by_mask_number(
                x, decoding_flag, self.decoder.mask_id, mini_batch_size
            )
            self._decode_selected_batch(
                x,
                seq_ids,
                decoding_start,
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
        total_length,
        pos_ids,
        num_layers,
        prompt_lengths=None,
        edit_budget=None,
        row_state=None,
    ):
        decoder_kwargs = self._decoder_editing_kwargs(
            seq_ids, prompt_lengths, edit_budget, row_state
        )
        decoding_x = x.select_seqs(seq_ids)
        decoding_block = gather_blocks(
            decoding_x.data, decoding_start[seq_ids], self.block_length
        )
        decoding_pos_ids = (
            torch.arange(self.block_length, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(seq_ids.shape[0], 1)
        )
        decoding_pos_ids += decoding_start[seq_ids].unsqueeze(1)
        graph_runner = getattr(self, "fa4_graph_runner", None)
        fused_step = None
        if graph_runner is not None:
            prompt_positions = allow_edit = None
            if getattr(self.decoder, "graph_fused_step", False):
                if prompt_lengths is None or edit_budget is None:
                    raise RuntimeError("Joint decode graph requires prompt_lengths and edit_budget")
                prompt_positions = decoding_pos_ids < prompt_lengths[seq_ids, None]
                allow_edit = edit_budget.allow_edit(seq_ids)
            replay = graph_runner.replay(
                self, decoding_block, decoding_pos_ids, seq_ids, prompt_positions, allow_edit
            )
            logits, fused_step = replay.logits, replay.step
        else:
            forward_batch = self._make_decode_forward_batch(seq_ids, decoding_start)
            output = self.model(
                decoding_block,
                use_cache=True,
                position_ids=decoding_pos_ids,
                past_key_values=[
                    self.past_key_values.layer_paged_kv(layer_id)
                    for layer_id in range(num_layers)
                ],
                forward_batch=forward_batch,
            )
            logits = output.logits[: len(seq_ids)]
        if fused_step is not None:
            updated, had_mask, changed, block_finished = fused_step
            decoding_x.data.scatter_(1, decoding_pos_ids, updated)
        elif callable(getattr(self.decoder, "batch_decode", None)):
            self.decoder.batch_decode(
                logits, decoding_start[seq_ids], decoding_x, self.block_length,
                **decoder_kwargs,
            )
        else:
            # HierarchyDecoder exposes a single-request decode API. Keep its
            # B=1 semantics even when the model forward batches requests.
            for row, start in enumerate(decoding_start[seq_ids].tolist()):
                single = decoding_x.select_seqs(slice(row, row + 1))
                self.decoder.decode(
                    logits[row : row + 1], start, start + self.block_length, single
                )
                decoding_x[row : row + 1] = single.data

        if fused_step is None:
            after = gather_blocks(decoding_x.data, decoding_start[seq_ids], self.block_length)
            had_mask = (decoding_block == self.decoder.mask_id).any(dim=1)
            changed = (after != decoding_block).any(dim=1)
            # Commit KV only after a forward on the final, unedited tokens.
            block_finished = (~had_mask) & (~changed)
        if edit_budget is not None:
            edit_budget.update(seq_ids, had_mask, changed, block_finished)
        decoding_start[seq_ids] += block_finished.long() * self.block_length
        x[seq_ids] = decoding_x.data
        if self.early_stop:
            eos_mask = (
                torch.any(x[seq_ids] == self.decoder.eos_id, dim=1) & block_finished
            )
            if eos_mask.any():
                stop_seq_ids = seq_ids[eos_mask.nonzero(as_tuple=True)[0]]
                decoding_start[stop_seq_ids] = total_length
        self.num_forwards += 1

    def ensure_paged_kv_cache(self, *, num_device_pages: int) -> None:
        config = self.model.model.config
        max_num_seqs = int(self.server_args.max_num_seqs)
        if (
            isinstance(getattr(self, "past_key_values", None), PagedKVCache)
            and getattr(self.past_key_values, "scheduler_num_pages", 0) >= int(num_device_pages)
            and self.past_key_values.batch_size >= max_num_seqs
        ):
            return
        num_heads = int(config.num_attention_heads)
        head_dim = int(getattr(config, "head_dim", config.hidden_size // num_heads))
        graph_runner = getattr(self, "fa4_graph_runner", None)
        if graph_runner is not None:
            graph_runner.invalidate()
        self.past_key_values = PagedKVCache(
            num_layers=int(config.num_hidden_layers),
            batch_size=max_num_seqs,
            local_kv_heads=max(
                1, int(config.num_key_value_heads) // get_attention_tp_size()
            ),
            max_length=self.max_length,
            head_dim=head_dim,
            page_size=int(self.runner_config.page_size),
            num_pages=int(num_device_pages),
            reserve_dummy_page=max(graph_runner.batch_sizes) if graph_runner is not None else 0,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.past_key_values.scheduler_num_pages = int(num_device_pages)

    def prepare_online_cuda_graphs(self) -> dict[str, int | float]:
        if self.fa4_graph_runner is None:
            return {}
        self.ensure_paged_kv_cache(num_device_pages=int(self.server_args.scheduler_num_device_pages))
        self.fa4_graph_runner.capture(self)
        return self.cuda_graph_stats()

    def cuda_graph_stats(self) -> dict[str, int | float]:
        graph_runner = getattr(self, "fa4_graph_runner", None)
        return graph_runner.stats() if graph_runner is not None else {}

    def shutdown_cuda_graphs(self, *, log: bool = True) -> None:
        del log
        graph_runner = getattr(self, "fa4_graph_runner", None)
        if graph_runner is not None:
            graph_runner.invalidate()

    async def execute_paged_forward_plan(
        self,
        op,
        states_by_id: dict[str, RequestState],
        tokenizer,
    ):
        from fluxserve.backend.engine.executor import ForwardStepResult

        configured_pages = int(
            getattr(self.server_args, "scheduler_num_device_pages", 0) or 0
        )
        if configured_pages <= 0:
            raise RuntimeError("paged execution requires scheduler_num_device_pages")
        max_page_id = max(
            (max(map(int, pages)) for pages in op.occupied_pages if pages),
            default=0,
        )
        if max_page_id >= configured_pages:
            raise RuntimeError(
                "scheduler page id exceeds the FA4 KV pool: "
                f"page_id={max_page_id}, num_device_pages={configured_pages}"
            )
        self.ensure_paged_kv_cache(num_device_pages=configured_pages)

        request_ids = list(op.request_ids)
        slot_indices = [self._paged_slot(rid) for rid in request_ids]
        try:
            self.past_key_values.set_page_tables(
                slot_indices,
                [list(map(int, pages)) for pages in op.occupied_pages],
            )
            num_prefill = int(op.num_extends())
            results: list[ForwardStepResult] = []
            if num_prefill:
                self._execute_paged_prefill(op, num_prefill, slot_indices)
                for rid in request_ids[:num_prefill]:
                    if rid in states_by_id:
                        states_by_id[rid].plan_prefill_done = True
                        results.append(
                            ForwardStepResult(rid=rid, token_ids=[], text="")
                        )
            if num_prefill < len(request_ids):
                results.extend(
                    self._execute_paged_decode(
                        op,
                        num_prefill,
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

    @torch.no_grad()
    def _execute_paged_prefill(
        self,
        op,
        num_prefill: int,
        slot_indices: list[int],
    ) -> None:
        input_lengths = [int(x) for x in op.input_lengths[:num_prefill]]
        q_offsets_cpu = [int(x) for x in op.extend_prefix_lens]
        if len(q_offsets_cpu) != num_prefill:
            raise RuntimeError("paged prefill metadata does not match num_extends")
        if not input_lengths or any(length <= 0 for length in input_lengths):
            raise RuntimeError("FA4 paged prefill requires positive query lengths")
        input_ids = list(map(int, op.input_ids))
        chunks = []
        cursor = 0
        for length in input_lengths:
            chunks.append(input_ids[cursor : cursor + length])
            cursor += length
        if cursor != len(input_ids):
            raise RuntimeError("paged prefill input_ids length mismatch")

        max_q_len = max(input_lengths)
        tokens = torch.full(
            (num_prefill, max_q_len),
            int(self.decoder.mask_id),
            dtype=torch.long,
            device=self.device,
        )
        for row, chunk in enumerate(chunks):
            tokens[row, : len(chunk)] = torch.tensor(
                chunk, dtype=torch.long, device=self.device
            )
        seq_ids = torch.tensor(
            slot_indices[:num_prefill], dtype=torch.long, device=self.device
        )
        q_offsets = torch.tensor(q_offsets_cpu, dtype=torch.long, device=self.device)
        q_lens = torch.tensor(input_lengths, dtype=torch.long, device=self.device)
        positions = q_offsets.unsqueeze(1) + torch.arange(
            max_q_len, dtype=torch.long, device=self.device
        )
        last_positions = (q_offsets + q_lens - 1).unsqueeze(1)
        positions = torch.minimum(positions, last_positions)
        forward_batch = self._make_forward_batch(
            num_prefill * max_q_len, is_prefill=True
        )
        forward_batch = self._make_paged_batch(
            seq_ids=seq_ids,
            q_offsets=q_offsets,
            q_lens=q_lens,
            max_input_len=max_q_len,
            is_prefill=True,
            forward_batch=forward_batch,
        )
        num_layers = self.model.model.config.num_hidden_layers
        self.model(
            tokens,
            use_cache=True,
            attention_mask=None,
            position_ids=positions,
            past_key_values=[
                self.past_key_values.layer_paged_kv(layer_id)
                for layer_id in range(num_layers)
            ],
            forward_batch=forward_batch,
        )
        self.num_forwards += 1

    @torch.no_grad()
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
            if request_ids[row] in states_by_id
            and not states_by_id[request_ids[row]].finished
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
        for row in active_rows:
            state = states_by_id[request_ids[row]]
            block_starts.append(
                state.aligned_prefill_length(block_length)
                + state.current_decode_block * block_length
            )
        max_total_len = max(start + block_length for start in block_starts)
        decode_tokens = torch.full(
            (int(self.server_args.max_num_seqs), max_total_len),
            int(self.decoder.mask_id),
            dtype=torch.long,
            device=self.device,
        )
        for row in active_rows:
            state = states_by_id[request_ids[row]]
            seq_id = slot_indices[row]
            values = state.input_ids + state.output_ids
            decode_tokens[seq_id, : len(values)] = torch.tensor(
                values, dtype=torch.long, device=self.device
            )
        decoding_start = torch.zeros(
            decode_tokens.shape[0], dtype=torch.long, device=self.device
        )
        for seq_id, start in zip(seq_ids.tolist(), block_starts, strict=True):
            decoding_start[int(seq_id)] = int(start)

        class _PlanTokenArray:
            def __init__(self, data, mask_id, eos_id):
                self.data = data
                self.mask_id = mask_id
                self.eos_id = eos_id

            def select_seqs(self, idx):
                return _PlanTokenArray(
                    self.data[idx].clone(), self.mask_id, self.eos_id
                )

            def __getitem__(self, idx):
                return self.data[idx]

            def __setitem__(self, idx, values):
                self.data[idx] = values

        x = _PlanTokenArray(decode_tokens, self.decoder.mask_id, self.decoder.eos_id)
        num_layers = self.model.model.config.num_hidden_layers
        prompt_lengths = torch.zeros_like(decoding_start)
        for row in active_rows:
            prompt_lengths[slot_indices[row]] = len(states_by_id[request_ids[row]].input_ids)
        edit_budget, row_state = self._make_decode_loop_state(decode_tokens.shape[0])
        pending = seq_ids
        for _ in range(edit_budget.max_block_iters):
            before = decoding_start[pending].clone()
            self._decode_selected_batch(
                x,
                pending,
                decoding_start,
                max_total_len,
                None,
                num_layers,
                prompt_lengths=prompt_lengths,
                edit_budget=edit_budget,
                row_state=row_state,
            )
            unfinished = decoding_start[pending] == before
            if not torch.any(unfinished):
                break
            pending = pending[unfinished]
        else:
            raise RuntimeError("FA4 paged decode block did not finish")

        results = []
        eos_id = int(self.decoder.eos_id)
        mask_id = int(self.decoder.mask_id)
        for local_idx, row in enumerate(active_rows):
            rid = request_ids[row]
            state = states_by_id[rid]
            block_start = block_starts[local_idx]
            remaining = max(0, state.max_new_tokens - len(state.output_ids))
            generated_start = max(block_start, len(state.input_ids))
            generated = (
                x.data[slot_indices[row], generated_start : block_start + block_length]
                .detach()
                .cpu()
                .tolist()[:remaining]
            )
            finish_reason = None
            if not state.ignore_eos and eos_id in generated:
                generated = generated[: generated.index(eos_id)]
                finish_reason = "stop"
            if state.ignore_eos:
                generated = [token for token in generated if token != mask_id]
            else:
                generated = [
                    token for token in generated if token != mask_id and token != eos_id
                ]
            finished = (
                finish_reason == "stop"
                or len(state.output_ids) + len(generated) >= state.max_new_tokens
            )
            if finished and finish_reason is None:
                finish_reason = "length"
            results.append(
                ForwardStepResult(
                    rid=rid,
                    token_ids=generated,
                    text=tokenizer.decode(generated, skip_special_tokens=True),
                    finished=finished,
                    finish_reason=finish_reason,
                    reserve_tokens=0 if finished else block_length,
                    decode_block_completed=True,
                )
            )
        return results
