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

"""Paged FlashAttention-4 runner for Nemotron-Labs-Diffusion.

Reuses ``FA4DiffusionRunner``'s paging, slot bookkeeping and metadata, and
replaces only the block contract. Two things make FA4 a natural fit:

* The kernel takes ``causal`` as a first-class argument next to ``page_table``,
  and aligns its causal mask bottom-right. A block appended at the end of a
  request's visible keys therefore gets full prefix visibility plus causality
  inside the block -- exactly the commit forward's semantics -- with no mask
  tensor. The same metadata under ``causal=False`` is the denoising forward.
* Every forward writes its keys and values into the request's own pages before
  reading them, per layer. A denoising forward's provisional keys therefore
  live only in slots this request owns, and the commit forward overwrites them
  before anything can read them back, because the committed prefix length does
  not advance until the commit completes.

Rows in a batch resolve at different denoising steps, so this runner does not
run a block synchronously across the batch. Each row carries its own state and
each iteration issues at most one denoise launch and one commit launch over the
rows that are ready for each. A resolved row never pays for another denoising
forward just because a neighbour is still working.

Decode CUDA graphs capture a denoise and a commit variant per batch bucket;
see ``NemotronCudaGraphRunner``. Configuration is documented in
``docs/serving/nemotron/nemotron-labs-diffusion-14B.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from fluxserve.backend.execution.decoders.nemotron import (
    NemotronThresholdDecoder,
    load_thinking_budget,
)
from fluxserve.backend.execution.runners.fa4_diffusion import FA4DiffusionRunner
from fluxserve.backend.execution.runners.nemotron_diffusion import (
    BlockStats,
    NemotronBlockBudgetExceeded,
)
from fluxserve.backend.layers.attention.metadata import PagedAttentionMetadata
from fluxserve.backend.managers.kvcache import PagedKVCache
from fluxserve.backend.execution.nemotron_sampling import (
    NemotronSamplingMixin, sample_tokens,
)

logger = logging.getLogger(__name__)

DENOISE = "denoise"
COMMIT = "commit"
DONE = "done"


def build_nemotron_paged_metadata(
    *,
    phase: str,
    q_offsets: torch.Tensor,
    q_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_input_len: int,
    block_length: int,
    page_size: int,
    causal: bool,
) -> PagedAttentionMetadata:
    """One varlen task per request, rather than one per block.

    ``build_block_diffusion_paged_metadata`` decomposes each request into
    block-sized virtual sequences, which is how LLaDA gets block-causal
    semantics out of a non-causal kernel call. It therefore requires
    block-aligned query offsets and lengths.

    Neither holds here. A Nemotron block starts at the prompt length, which is
    an arbitrary number, and the causal prefill covers the whole prompt rather
    than a whole number of blocks. The decomposition is also unnecessary:
    ``causal`` gives token-level causality inside one task directly, and a
    single non-causal task over prefix-plus-block is exactly the denoising
    attention pattern. So Nemotron builds one task per request and lets the
    kernel flag carry the semantics.
    """
    if q_offsets.ndim != 1 or q_lens.ndim != 1:
        raise ValueError("q_offsets and q_lens must be one-dimensional")
    if q_offsets.numel() != q_lens.numel():
        raise ValueError("q_offsets and q_lens must contain one value per request")
    if page_size % 16:
        raise ValueError(f"paged FA4 requires page_size % 16 == 0, got {page_size}")
    batch_size = int(q_lens.numel())
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError(
            "page_table must have one row per request, got "
            f"{tuple(page_table.shape)} for batch_size={batch_size}"
        )

    q_lens_cpu = tuple(int(value) for value in q_lens.detach().cpu().tolist())
    q_offsets_cpu = tuple(int(value) for value in q_offsets.detach().cpu().tolist())
    if any(length <= 0 for length in q_lens_cpu):
        raise ValueError(f"query lengths must be positive, got {q_lens_cpu}")
    if any(length > max_input_len for length in q_lens_cpu):
        raise ValueError(
            f"query length exceeds max_input_len={max_input_len}: {q_lens_cpu}"
        )
    if any(offset < 0 for offset in q_offsets_cpu):
        raise ValueError(f"query offsets must be non-negative: {q_offsets_cpu}")

    device = page_table.device
    selected = page_table.to(device=device, dtype=torch.int32).contiguous()
    q_indices: list[int] = []
    qo_values = [0]
    kv_lens: list[int] = []
    slot_parts: list[torch.Tensor] = []

    for row, (offset, length) in enumerate(zip(q_offsets_cpu, q_lens_cpu, strict=True)):
        kv_len = offset + length
        required_pages = (kv_len + page_size - 1) // page_size
        if required_pages > selected.shape[1]:
            raise ValueError(
                f"request {row} needs {required_pages} pages for kv_len={kv_len}, "
                f"but page_table has {selected.shape[1]} columns"
            )
        if torch.any(selected[row, :required_pages] < 0):
            raise ValueError(f"request {row} contains an invalid negative page id")
        positions = torch.arange(offset, kv_len, device=device, dtype=torch.long)
        physical = selected[row, positions // page_size].long()
        slot_parts.append(physical * page_size + positions % page_size)
        q_indices.extend(range(row * max_input_len, row * max_input_len + length))
        qo_values.append(qo_values[-1] + length)
        kv_lens.append(kv_len)

    max_pages = (max(kv_lens) + page_size - 1) // page_size
    return PagedAttentionMetadata(
        phase="prefill" if phase == "prefill" else "decode",
        batch_size=batch_size,
        max_input_len=int(max_input_len),
        block_length=int(block_length),
        page_size=int(page_size),
        q_lens_cpu=q_lens_cpu,
        q_offsets_cpu=q_offsets_cpu,
        q_token_indices=torch.tensor(q_indices, device=device, dtype=torch.long),
        qo_indptr=torch.tensor(qo_values, device=device, dtype=torch.int32),
        kv_lens=torch.tensor(kv_lens, device=device, dtype=torch.int32),
        page_table=selected[:, :max_pages].contiguous(),
        slot_mapping=torch.cat(slot_parts).contiguous(),
        max_q_len=max(q_lens_cpu),
        max_kv_len=max(kv_lens),
        causal=bool(causal),
    )


@dataclass
class RowState:
    """Per-request block state. Nothing here is shared across rows."""

    index: int
    seq_id: int
    prompt_length: int
    generation_length: int
    padded_generation: int
    prefix_length: int
    block_budget: int
    block: torch.Tensor
    seed: int
    state: str = DENOISE
    denoise_steps: int = 0
    terminal: bool = False
    emitted: list = None
    stats: BlockStats = None
    sampling: object = None
    stop_on_eos: bool = True

    def __post_init__(self):
        if self.emitted is None:
            self.emitted = []
        if self.stats is None:
            self.stats = BlockStats()


class NemotronFA4DiffusionRunner(NemotronSamplingMixin, FA4DiffusionRunner):
    """Nemotron's block contract on FA4 paging."""

    supported_architecture_keys = ("nemotron_labs_diffusion", "nemotronlabsdiffusion")
    supported_architecture_label = "Nemotron-Labs-Diffusion"
    requires_prompt_lengths = True
    paged_attention_backend = "fa4"

    def _init_paged_backend(self, *args, **kwargs):
        """Initialize backend resources separately from the shared block loop."""
        super().__init__(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        runner_config = kwargs.get("runner_config")
        if runner_config is None and len(args) >= 3:
            runner_config = args[2]
        self._init_paged_backend(*args, **kwargs)
        if self.fa4_graph_runner is not None:
            # The base constructor builds a LLaDA graph runner; Nemotron needs
            # a denoise and a commit variant per bucket. Both names point at the
            # same object so the inherited capture/stats/dummy-page plumbing
            # keeps working unchanged.
            from fluxserve.backend.execution.nemotron_cuda_graph_runner import (
                NemotronCudaGraphRunner,
            )

            buckets = (
                self.runner_config.cuda_graph_capture_batch_sizes
                or self.runner_config.supported_batch_sizes
            )
            self.fa4_graph_runner = NemotronCudaGraphRunner(buckets)
        self.nemotron_graph_runner = self.fa4_graph_runner
        steps = int(getattr(self.runner_config, "steps", 0) or 0)
        self.max_denoise_steps = steps if steps > 0 else int(self.block_length)
        self.last_stats: list[dict] = []
        # Seeds are per request and outlive one plan call: a block's first
        # position comes from the previous causal forward.
        self.thinking_budget = load_thinking_budget(self.runner_config)
        self._request_seeds: dict[str, int] = {}
        self._last_graph_seed = None

    def init_decoder(self):
        self.decoder = NemotronThresholdDecoder(
            threshold=self.runner_config.threshold,
            mask_id=self.runner_config.mask_id,
            eos_ids=(
                tuple(self.runner_config.eos_ids)
                or (int(self.runner_config.eos_id),)
            ),
        )

    def _apply_thinking_budget(self, block, emitted=None, *, produced_tokens=None):
        """Place the end-of-thinking marker in ``block``, if it is due.

        ``emitted`` is the offline path's list of committed block tensors;
        ``produced_tokens`` is the online path's already-published tokens. Both
        answer the same question -- how many tokens came before this block, and
        did any of them carry the marker.
        """
        budget = self.thinking_budget
        if not budget.enabled:
            return
        if produced_tokens is None:
            produced_tokens = [
                int(value) for item in (emitted or []) for value in item.tolist()
            ]
        if budget.satisfied(produced_tokens):
            return
        offset = budget.block_injection_offset(
            len(produced_tokens), int(block.shape[-1])
        )
        if offset is None:
            return
        block[offset] = budget.end_think_token_id

    # -- forwards ----------------------------------------------------------

    def _paged_forward(self, *, seq_ids, tokens, q_offsets, causal: bool,
                       is_prefill: bool, max_input_len: int, q_lens=None,
                       positions=None):
        """One launch over a set of rows sharing a phase."""
        if q_lens is None:
            q_lens = torch.full_like(q_offsets, max_input_len)
        forward_batch = self._make_forward_batch(
            len(seq_ids) * max_input_len, is_prefill=is_prefill
        )
        if not isinstance(self.past_key_values, PagedKVCache):
            raise RuntimeError("Nemotron paged execution requires PagedKVCache.")
        selected_page_table = self.past_key_values.page_table.index_select(
            0, seq_ids.to(device=self.past_key_values.device, dtype=torch.long)
        )
        forward_batch.paged_attention_metadata = build_nemotron_paged_metadata(
            phase="prefill" if is_prefill else "decode",
            q_offsets=q_offsets,
            q_lens=q_lens,
            page_table=selected_page_table,
            max_input_len=max_input_len,
            block_length=int(self.block_length),
            page_size=int(self.past_key_values.page_size),
            causal=causal,
        )
        if self.paged_attention_backend != "fa4":
            from dataclasses import replace

            forward_batch.paged_attention_metadata = replace(
                forward_batch.paged_attention_metadata,
                backend=self.paged_attention_backend,
            )
        num_layers = self.model.model.config.num_hidden_layers
        if positions is None:
            positions = (
                torch.arange(max_input_len, device=self.device).unsqueeze(0)
                + q_offsets.unsqueeze(1)
            )
        self.num_forwards += 1
        output = self.model(
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
        return output.logits

    def _graph_replay(self, *, seq_ids, tokens, positions, causal: bool):
        """Replay a captured block graph, or return ``None`` to run eagerly."""
        graph_runner = getattr(self, "nemotron_graph_runner", None)
        if graph_runner is None:
            return None
        if not graph_runner.can_replay(
            self,
            batch_size=int(tokens.shape[0]),
            length=int(tokens.shape[1]),
            causal=causal,
        ):
            return None
        self.num_forwards += 1
        return graph_runner.replay(
            self,
            input_ids=tokens,
            position_ids=positions,
            seq_ids=seq_ids,
            causal=causal,
        )

    # -- continuously scheduled paged serving ------------------------------
    #
    # Replaces rather than extends ``FA4DiffusionRunner``'s plan, which
    # implements the LLaDA block contract: it ends a block on the
    # ``(~had_mask) & (~changed)`` predicate and lets a denoising forward supply
    # the committed KV. Inheriting it would serve a different model rather than
    # fail. One plan call advances every active request by exactly one block.

    def _release_paged_slot(self, request_id: str) -> None:
        super()._release_paged_slot(request_id)
        self._request_seeds.pop(str(request_id), None)
        getattr(self, "_request_sampling", {}).pop(str(request_id), None)

    async def execute_paged_forward_plan(self, op, states_by_id, tokenizer):
        from fluxserve.backend.engine.executor import ForwardStepResult

        configured_pages = int(
            getattr(self.server_args, "scheduler_num_device_pages", 0) or 0
        )
        if configured_pages <= 0:
            raise RuntimeError("paged execution requires scheduler_num_device_pages")
        max_page_id = max(
            (max(map(int, pages)) for pages in op.occupied_pages if pages), default=0
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
                self._plan_prefill(op, num_prefill, slot_indices, states_by_id)
                for rid in request_ids[:num_prefill]:
                    if rid in states_by_id:
                        states_by_id[rid].plan_prefill_done = True
                        results.append(
                            ForwardStepResult(rid=rid, token_ids=[], text="")
                        )
            if num_prefill < len(request_ids):
                results.extend(
                    self._plan_decode(
                        op, num_prefill, slot_indices, states_by_id, tokenizer
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
    def _plan_prefill(self, op, num_prefill, slot_indices, states_by_id) -> None:
        """Causal prefill over the whole prompt, recording each request's seed.

        The scheduler is configured to prefill the full prompt for this model
        rather than its block-aligned floor, because a Nemotron block must
        begin after the entire prompt with its first position seeded. A chunked
        prefill only yields a seed on the chunk that reaches the prompt's end.
        """
        request_ids = list(op.request_ids)[:num_prefill]
        input_lengths = [int(value) for value in op.input_lengths[:num_prefill]]
        q_offsets_cpu = [int(value) for value in op.extend_prefix_lens]
        if len(q_offsets_cpu) != num_prefill:
            raise RuntimeError("paged prefill metadata does not match num_extends")
        if not input_lengths or any(length <= 0 for length in input_lengths):
            raise RuntimeError("Nemotron paged prefill requires positive lengths")

        input_ids = list(map(int, op.input_ids))
        chunks, cursor = [], 0
        for length in input_lengths:
            chunks.append(input_ids[cursor : cursor + length])
            cursor += length
        if cursor != len(input_ids):
            raise RuntimeError("paged prefill input_ids length mismatch")

        max_q_len = max(input_lengths)
        tokens = torch.full(
            (num_prefill, max_q_len), int(self.decoder.mask_id),
            dtype=torch.long, device=self.device,
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
        positions = torch.minimum(positions, (q_offsets + q_lens - 1).unsqueeze(1))

        logits = self._paged_forward(
            seq_ids=seq_ids, tokens=tokens, q_offsets=q_offsets,
            causal=True, is_prefill=True, max_input_len=max_q_len,
            q_lens=q_lens, positions=positions,
        )
        for row, rid in enumerate(request_ids):
            state = states_by_id.get(rid)
            if state is None:
                continue
            if q_offsets_cpu[row] + input_lengths[row] < len(state.input_ids):
                continue  # an intermediate chunk carries no seed
            last = input_lengths[row] - 1
            self._request_seeds[str(rid)] = int(sample_tokens(
                logits[row, last], self._sampling_for_request(state)
            ))

    @torch.no_grad()
    def _plan_decode(self, op, decode_start_row, slot_indices, states_by_id,
                     tokenizer):
        from fluxserve.backend.engine.executor import ForwardStepResult

        request_ids = list(op.request_ids)
        block_length = int(self.block_length)
        rows: list[RowState] = []
        for row in range(decode_start_row, len(request_ids)):
            rid = request_ids[row]
            state = states_by_id.get(rid)
            if state is None or state.finished:
                continue
            seed = self._request_seeds.get(str(rid))
            if seed is None:
                raise RuntimeError(
                    f"request {rid} reached decode without a seed; its causal "
                    "prefill did not complete"
                )
            prefix_length = (
                len(state.input_ids) + state.current_decode_block * block_length
            )
            block = torch.full(
                (block_length,), self.decoder.mask_id,
                dtype=torch.long, device=self.device,
            )
            block[0] = seed
            self._apply_thinking_budget(
                block, produced_tokens=list(state.output_ids)
            )
            candidate = RowState(
                index=row, seq_id=slot_indices[row],
                prompt_length=len(state.input_ids),
                generation_length=state.max_new_tokens,
                padded_generation=block_length,
                prefix_length=prefix_length,
                block_budget=1,
                block=block, seed=seed, sampling=self._sampling_for_request(state),
                stop_on_eos=not state.ignore_eos,
            )
            if candidate.stop_on_eos and bool(self.decoder.eos_is_terminal(block.unsqueeze(0)).all()):
                candidate.terminal = True
                candidate.state = COMMIT
                candidate.stats.denoise_per_block.append(0)
            rows.append(candidate)

        if not rows:
            return []
        self._run_block_loop(rows)

        results = []
        eos_ids = frozenset(int(value) for value in self.decoder.eos_ids)
        mask_id = int(self.decoder.mask_id)
        for row in rows:
            rid = request_ids[row.index]
            state = states_by_id[rid]
            self._request_seeds[str(rid)] = row.seed
            remaining = max(0, state.max_new_tokens - len(state.output_ids))
            generated = row.emitted[0].detach().cpu().tolist()[:remaining]
            finish_reason = None
            if not state.ignore_eos:
                stop = next(
                    (i for i, token in enumerate(generated) if token in eos_ids), None
                )
                if stop is not None:
                    generated = generated[:stop]
                    finish_reason = "stop"
            if state.ignore_eos:
                generated = [token for token in generated if token != mask_id]
            else:
                generated = [
                    token for token in generated
                    if token != mask_id and token not in eos_ids
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

    # -- batch entry point -------------------------------------------------

    @torch.no_grad()
    def generate(self, prompts, prompt_lengths=None, generation_lengths=None, sampling_params=None):
        from fluxserve.backend.model_loader.nemotron import (
            check_nemotron_context_limit,
        )

        batch_size, padded_prompt_len = prompts.shape
        self._offline_sampling = self._sampling_batch(batch_size, sampling_params)
        self._offline_stop_on_eos = self._stop_on_eos_batch(batch_size, sampling_params)
        mask_id = self.decoder.mask_id
        if prompt_lengths is None:
            prompt_lengths = (prompts != mask_id).sum(dim=-1).tolist()
        prompt_lengths = [int(value) for value in prompt_lengths]
        if generation_lengths is None:
            generation_lengths = [int(self.runner_config.gen_length)] * batch_size
        generation_lengths = [int(value) for value in generation_lengths]
        if len(prompt_lengths) != batch_size or len(generation_lengths) != batch_size:
            raise ValueError("prompt/generation lengths must have one value per row")
        if any(value < 0 for value in generation_lengths):
            raise ValueError("generation_lengths must be non-negative")

        block_length = int(self.block_length)
        padded = [
            ((value + block_length - 1) // block_length) * block_length
            for value in generation_lengths
        ]
        total = max(
            (prompt + generated for prompt, generated in zip(prompt_lengths, padded)),
            default=0,
        )
        check_nemotron_context_limit(
            total, getattr(self, "model_config", None),
            serving_limit=getattr(getattr(self, "server_args", None), "max_model_len", None),
        )
        if total > self.max_length:
            self.max_length = total
        self.past_key_values = self.allocate_kv_cache(batch_size)

        rows = self._prefill_rows(prompts, prompt_lengths, generation_lengths, padded)
        self._run_block_loop(rows)

        output = self._assemble(rows, prompts, prompt_lengths, generation_lengths,
                                padded_prompt_len)
        self.last_stats = [row.stats.as_dict() for row in rows]
        logger.info(
            "Nemotron paged FA4: %d request(s), denoise/commit=%s",
            batch_size,
            [(s["denoise_calls"], s["commit_calls"]) for s in self.last_stats],
        )
        return output

    def _prefill_group(self, prompts, seq_ids, length):
        chunk_size = int(getattr(self.runner_config, "nemotron_prefill_chunk_size", 1024))
        calls = 0
        for start in range(0, length, chunk_size):
            end = min(length, start + chunk_size)
            logits = self._paged_forward(
                seq_ids=seq_ids, tokens=prompts[seq_ids, start:end].contiguous(),
                q_offsets=torch.full_like(seq_ids, start), causal=True,
                is_prefill=True, max_input_len=end - start,
            )
            calls += 1
        return logits, calls

    def _prefill_rows(self, prompts, prompt_lengths, generation_lengths, padded):
        """Causal prefill, one launch per distinct prompt length."""
        rows: list[RowState] = []
        by_length: dict[int, list[int]] = {}
        for index, length in enumerate(prompt_lengths):
            by_length.setdefault(length, []).append(index)

        seeds: dict[int, int] = {}
        prefill_calls = {}
        for length, indices in sorted(by_length.items()):
            if length == 0:
                raise ValueError("Nemotron paged prefill requires a non-empty prompt")
            seq_ids = torch.tensor(indices, device=self.device, dtype=torch.long)
            logits, calls = self._prefill_group(prompts, seq_ids, length)
            for position, index in enumerate(indices):
                sampling = getattr(self, "_offline_sampling", [None] * len(prompt_lengths))[index]
                seeds[index] = int(sample_tokens(logits[position, -1], sampling))
                prefill_calls[index] = calls

        block_length = int(self.block_length)
        for index, length in enumerate(prompt_lengths):
            block = torch.full(
                (block_length,), self.decoder.mask_id,
                dtype=prompts.dtype, device=self.device,
            )
            block[0] = seeds[index]
            self._apply_thinking_budget(block, [])
            row = RowState(
                index=index,
                seq_id=index,
                prompt_length=length,
                generation_length=generation_lengths[index],
                padded_generation=padded[index],
                prefix_length=length,
                block_budget=padded[index] // block_length,
                block=block,
                seed=seeds[index],
                sampling=getattr(self, "_offline_sampling", [None] * len(prompt_lengths))[index],
                stop_on_eos=getattr(self, "_offline_stop_on_eos", [self.early_stop] * len(prompt_lengths))[index],
            )
            row.stats.prefill_calls = prefill_calls[index]
            if row.padded_generation == 0:
                row.state = DONE
            elif row.stop_on_eos and bool(self.decoder.eos_is_terminal(block.unsqueeze(0)).all()):
                # EOS as the seed: skip straight to the commit, as the dense
                # runner does, and record the zero-denoise block.
                row.terminal = True
                row.state = COMMIT
                row.stats.denoise_per_block.append(0)
            rows.append(row)
        return rows

    def _run_block_loop(self, rows: list[RowState]) -> None:
        block_length = int(self.block_length)
        while True:
            denoising = [row for row in rows if row.state == DENOISE]
            committing = [row for row in rows if row.state == COMMIT]
            if not denoising and not committing:
                break

            if denoising:
                logits = self._launch(denoising, causal=False)
                for position, row in enumerate(denoising):
                    row.stats.denoise_calls += 1
                    row.denoise_steps += 1
                    self.decoder.step(
                        logits[position : position + 1], row.block.unsqueeze(0),
                        sampling=row.sampling,
                    )
                    resolved = not bool(
                        self.decoder.has_masks(row.block.unsqueeze(0)).any()
                    )
                    terminal = row.stop_on_eos and bool(
                        self.decoder.eos_is_terminal(row.block.unsqueeze(0)).all()
                    )
                    if resolved or terminal:
                        row.terminal = terminal
                        row.stats.denoise_per_block.append(row.denoise_steps)
                        row.state = COMMIT
                    elif row.denoise_steps >= self.max_denoise_steps:
                        remaining = int(
                            (row.block == self.decoder.mask_id).sum()
                        )
                        raise NemotronBlockBudgetExceeded(
                            f"request row {row.index}: {remaining} masked "
                            f"position(s) remain after {self.max_denoise_steps} "
                            f"denoising steps at prefix length "
                            f"{row.prefix_length}"
                        )

            if committing:
                logits = self._launch(committing, causal=True)
                for position, row in enumerate(committing):
                    row.stats.commit_calls += 1
                    row.stats.blocks += 1
                    row.emitted.append(row.block.clone())
                    row.prefix_length += block_length
                    # Sampling stays outside graph capture. A captured argmax
                    # must not replace a nonzero-temperature seed.
                    row.seed = int(sample_tokens(logits[position, -1], row.sampling))
                    row.denoise_steps = 0
                    if row.terminal or (
                        len(row.emitted) >= row.block_budget
                    ):
                        row.state = DONE
                        continue
                    row.block = torch.full(
                        (block_length,), self.decoder.mask_id,
                        dtype=row.block.dtype, device=self.device,
                    )
                    row.block[0] = row.seed
                    self._apply_thinking_budget(row.block, row.emitted)
                    if row.stop_on_eos and bool(
                        self.decoder.eos_is_terminal(row.block.unsqueeze(0)).all()
                    ):
                        row.terminal = True
                        row.state = COMMIT
                        row.stats.denoise_per_block.append(0)
                    else:
                        row.state = DENOISE

    def _launch(self, rows: list[RowState], *, causal: bool) -> torch.Tensor:
        seq_ids = torch.tensor(
            [row.seq_id for row in rows], device=self.device, dtype=torch.long
        )
        q_offsets = torch.tensor(
            [row.prefix_length for row in rows], device=self.device, dtype=torch.long
        )
        tokens = torch.stack([row.block for row in rows], dim=0)
        positions = (
            torch.arange(int(self.block_length), device=self.device).unsqueeze(0)
            + q_offsets.unsqueeze(1)
        )
        replay = self._graph_replay(
            seq_ids=seq_ids, tokens=tokens, positions=positions, causal=causal
        )
        if replay is not None:
            self._last_graph_seed = replay.seed
            return replay.logits
        self._last_graph_seed = None
        return self._paged_forward(
            seq_ids=seq_ids,
            tokens=tokens,
            q_offsets=q_offsets,
            causal=causal,
            is_prefill=False,
            max_input_len=int(self.block_length),
            positions=positions,
        )

    def _assemble(self, rows, prompts, prompt_lengths, generation_lengths,
                  padded_prompt_len):
        mask_id = self.decoder.mask_id
        width = padded_prompt_len + max(generation_lengths, default=0)
        output = torch.full(
            (len(rows), width), mask_id, dtype=prompts.dtype, device=prompts.device
        )
        for row in rows:
            generated = (
                torch.cat(row.emitted, dim=0)
                if row.emitted
                else prompts.new_empty((0,))
            )
            row.stats.internal_tokens = int(generated.shape[0])
            generated = generated[: row.generation_length]
            if row.stop_on_eos:
                cut = int(self.decoder.first_eos_index(generated.unsqueeze(0))[0])
                generated = generated[: min(cut + 1, generated.shape[0])]
            row.stats.returned_tokens = int(generated.shape[0])
            output[row.index, : row.prompt_length] = prompts[
                row.index, : row.prompt_length
            ]
            output[
                row.index, padded_prompt_len : padded_prompt_len + generated.shape[0]
            ] = generated
        return output


__all__ = ["NemotronFA4DiffusionRunner", "RowState"]
