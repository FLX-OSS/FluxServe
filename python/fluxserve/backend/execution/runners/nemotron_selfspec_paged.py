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

"""Paged, batched linear self-speculation.

Offline `PagedKVCache` preallocates ``batch_size x pages_per_sequence`` pages.
Continuous scheduling reserves each provisional block before its forward.
Rolling back a rejected speculative tail uses the same rule in both: leave the
committed prefix length where it was. The verify forward's keys past the
accepted end stay in slots this request already owns, and the next iteration
overwrites them.

Batching is what actually makes this different from the dense runner. Rows
accept different numbers of tokens per iteration, so their committed prefixes
diverge immediately and a block-synchronous loop would be wrong. Each row
carries its own state and each iteration issues at most one draft launch and one
verify launch over the rows ready for each -- the same shape as the paged
diffusion runner, with DRAFT and VERIFY in place of DENOISE and COMMIT.

The one-task-per-request metadata is what allows it: every row brings its own
``q_offset``, so a single launch can cover rows whose prefixes have drifted
apart.

Continuous scheduling reports only accepted tokens; reserved tail pages remain
owned by the request until they are reused or the request finishes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from fluxserve.backend.execution.nemotron_sampling import sample_tokens

from fluxserve.backend.execution.decoders.nemotron import (
    NemotronThresholdDecoder,
    load_thinking_budget,
)
from fluxserve.backend.execution.runners.nemotron_fa4 import (
    NemotronFA4DiffusionRunner,
)
from fluxserve.backend.execution.runners.nemotron_selfspec import (
    NemotronSelfSpecRunner,
    SelfSpecStats,
)

logger = logging.getLogger(__name__)

DRAFT = "draft"
VERIFY = "verify"
DONE = "done"


@dataclass
class SpecRow:
    """Per-request speculation state. Prefixes diverge, so nothing is shared."""

    index: int
    seq_id: int
    prompt_length: int
    generation_length: int
    prefix_length: int
    block: torch.Tensor
    seed: int
    state: str = DRAFT
    draft_steps: int = 0
    # None means "run until the budget or EOS"; the online path sets 1, because
    # one plan call advances a request by one speculation iteration.
    max_iterations: int | None = None
    terminal: bool = False
    emitted: list = field(default_factory=list)
    stats: SelfSpecStats = field(default_factory=SelfSpecStats)
    sampling: object = None
    stop_on_eos: bool = True


class NemotronSelfSpecPagedRunner(NemotronFA4DiffusionRunner):
    """Self-speculation over FA4 paging, several requests at a time."""

    requires_prompt_lengths = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draft_threshold = float(
            getattr(self.runner_config, "draft_threshold", 0.0) or 0.0
        )
        self.draft_decoder = NemotronThresholdDecoder(
            threshold=self.draft_threshold,
            mask_id=self.decoder.mask_id,
            eos_ids=self.decoder.eos_ids,
            draft=True,
        )
        self.thinking_budget = load_thinking_budget(self.runner_config)
        self.lora = None
        self.last_stats: list[dict] = []
        # A speculation iteration produces a variable number of tokens, so the
        # committed prefix cannot be derived from a block counter.
        self._request_prefix: dict[str, int] = {}

    def load_draft_adapter(self, path=None) -> None:
        from fluxserve.backend.model_loader.nemotron import load_nemotron_lora

        # Captured kernels retain the adapter's weight addresses. Destroy them
        # before replacing either weight set, and load against base weights.
        graph_runner = getattr(self, "nemotron_graph_runner", None)
        if graph_runner is not None:
            graph_runner.invalidate()
        if self.lora is not None:
            self.lora.apply(False)
        self.lora = load_nemotron_lora(self.model, self.model_config, path)
        # Say so either way. A silent `None` is indistinguishable from a
        # loaded adapter in the log, and acceptance length depends on it.
        if self.lora is None:
            logger.info("Nemotron draft adapter not present; drafting with base weights")
        else:
            logger.info(
                "Nemotron draft adapter loaded for %d o_proj layers",
                len(self.lora.layers),
            )

    def graph_capture_context(self, *, causal: bool):
        return self._adapters(enabled=not causal)

    def _adapters(self, *, enabled: bool):
        return NemotronSelfSpecRunner._adapters(self, enabled=enabled)

    # -- continuously scheduled paged serving ------------------------------
    #
    # A speculation iteration produces between one and `block_length` tokens, so
    # a request's committed prefix does not advance by a fixed stride and cannot
    # be derived from `current_decode_block`. This runner therefore keeps the
    # prefix per request alongside the seed, and reports only the accepted
    # tokens to the scheduler, which is what keeps the scheduler's notion of the
    # request length in step with the cache.
    #
    # The initial decode reserves one full block. Each later iteration extends
    # that high-water mark only by the accepted length: the rejected tail is
    # already reserved and is overwritten by the next draft/verify pair.

    def _release_paged_slot(self, request_id: str) -> None:
        super()._release_paged_slot(request_id)
        self._request_prefix.pop(str(request_id), None)

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
                with self._adapters(enabled=False):
                    self._plan_prefill(op, num_prefill, slot_indices, states_by_id)
                for rid in request_ids[:num_prefill]:
                    state = states_by_id.get(rid)
                    if state is None:
                        continue
                    state.plan_prefill_done = True
                    if str(rid) in self._request_seeds:
                        # The prefill that reached the prompt's end also fixes
                        # where this request's first block starts.
                        self._request_prefix[str(rid)] = len(state.input_ids)
                    results.append(ForwardStepResult(rid=rid, token_ids=[], text=""))
            if num_prefill < len(request_ids):
                results.extend(
                    self._plan_speculate(
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
    def _plan_speculate(self, op, decode_start_row, slot_indices, states_by_id,
                        tokenizer):
        from fluxserve.backend.engine.executor import ForwardStepResult

        request_ids = list(op.request_ids)
        block_length = int(self.block_length)
        rows: list[SpecRow] = []
        for row in range(decode_start_row, len(request_ids)):
            rid = request_ids[row]
            state = states_by_id.get(rid)
            if state is None or state.finished:
                continue
            seed = self._request_seeds.get(str(rid))
            prefix = self._request_prefix.get(str(rid))
            if seed is None or prefix is None:
                raise RuntimeError(
                    f"request {rid} reached speculation without a seed and "
                    "prefix; its causal prefill did not complete"
                )
            block = torch.full(
                (block_length,), self.decoder.mask_id,
                dtype=torch.long, device=self.device,
            )
            block[0] = seed
            budget = self.thinking_budget
            if budget.force_next_seed(len(state.output_ids)) and not budget.satisfied(
                list(state.output_ids)
            ):
                block[0] = budget.end_think_token_id
            rows.append(
                SpecRow(
                    index=row,
                    sampling=self._sampling_for_request(state),
                    seq_id=slot_indices[row],
                    prompt_length=len(state.input_ids),
                    generation_length=max(
                        1, state.max_new_tokens - len(state.output_ids)
                    ),
                    prefix_length=prefix,
                    block=block,
                    seed=int(block[0]),
                    max_iterations=1,
                    stop_on_eos=not state.ignore_eos,
                )
            )
            candidate = rows[-1]
            if prefix == len(state.input_ids) and not state.output_ids:
                # The prefill seed is the first generated token, just as in
                # offline generation. Verification predicts the token after it.
                candidate.emitted.append(seed)
                if candidate.stop_on_eos and seed in self.decoder.eos_ids:
                    candidate.terminal = True
                    candidate.state = DONE
                elif candidate.generation_length <= 1:
                    candidate.state = DONE

        if not rows:
            return []
        self._run_speculation(rows)

        results = []
        eos_ids = frozenset(int(value) for value in self.decoder.eos_ids)
        mask_id = int(self.decoder.mask_id)
        for row in rows:
            rid = request_ids[row.index]
            state = states_by_id[rid]
            accepted = row.prefix_length - self._request_prefix[str(rid)]
            self._request_seeds[str(rid)] = row.seed
            self._request_prefix[str(rid)] = row.prefix_length
            remaining = max(0, state.max_new_tokens - len(state.output_ids))
            generated = list(row.emitted)[:remaining]
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
                    # Acquire() adds to the previous reservation; it does not
                    # replace it. Only the accepted prefix needs new capacity.
                    reserve_tokens=0 if finished else accepted,
                    # Deliberately False: `current_decode_block` counts fixed
                    # strides, and a speculation iteration does not advance by
                    # one. The prefix is tracked here instead.
                    decode_block_completed=False,
                )
            )
        return results

    # -- launches ----------------------------------------------------------

    def _spec_launch(self, rows: list[SpecRow], *, causal: bool) -> torch.Tensor:
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
        # Only model forwards are captured. Acceptance, rollback and sampling
        # stay outside the graph; replay refreshes each row's actual prefix and
        # page table, including unaligned blocks after partial acceptance.
        replay = self._graph_replay(
            seq_ids=seq_ids, tokens=tokens, positions=positions, causal=causal
        )
        if replay is not None:
            return replay.logits
        return self._paged_forward(
            seq_ids=seq_ids,
            tokens=tokens,
            q_offsets=q_offsets,
            causal=causal,
            is_prefill=False,
            max_input_len=int(self.block_length),
            positions=positions,
        )

    # -- the loop ----------------------------------------------------------

    def _run_speculation(self, rows: list[SpecRow]) -> None:
        block_length = int(self.block_length)
        while True:
            drafting = [row for row in rows if row.state == DRAFT]
            verifying = [row for row in rows if row.state == VERIFY]
            if not drafting and not verifying:
                break

            if drafting:
                with self._adapters(enabled=True):
                    logits = self._spec_launch(drafting, causal=False)
                for position, row in enumerate(drafting):
                    row.stats.draft_calls += 1
                    row.draft_steps += 1
                    self.draft_decoder.step(
                        logits[position : position + 1], row.block.unsqueeze(0),
                        sampling=row.sampling,
                    )
                    if not bool(
                        self.draft_decoder.has_masks(row.block.unsqueeze(0)).any()
                    ):
                        row.state = VERIFY
                    elif row.draft_steps >= self.max_denoise_steps:
                        remaining = int(
                            (row.block == self.draft_decoder.mask_id).sum()
                        )
                        raise RuntimeError(
                            f"request row {row.index}: draft left {remaining} "
                            f"masked position(s) after {row.draft_steps} "
                            f"forward(s) at prefix length {row.prefix_length}"
                        )

            if verifying:
                with self._adapters(enabled=False):
                    logits = self._spec_launch(verifying, causal=True)
                for position, row in enumerate(verifying):
                    verified = sample_tokens(logits[position], row.sampling)
                    row.stats.verify_calls += 1
                    accepted = NemotronSelfSpecRunner.accepted_length(
                        verified, row.block
                    )
                    tokens = [int(v) for v in verified[:accepted]]
                    row.stats.iterations += 1
                    row.stats.accepted_per_iteration.append(accepted)
                    row.stats.draft_calls_per_iteration.append(row.draft_steps)
                    row.draft_steps = 0

                    # Roll back: only the accepted prefix joins the committed
                    # region. The rejected tail stays in pages this request
                    # owns and is overwritten next iteration.
                    row.prefix_length += accepted
                    row.seed = tokens[-1]
                    for token in tokens:
                        row.emitted.append(token)
                        if row.stop_on_eos and token in self.decoder.eos_ids:
                            row.terminal = True
                            break
                    reached_iteration_budget = (
                        row.max_iterations is not None
                        and row.stats.iterations >= row.max_iterations
                    )
                    if (
                        row.terminal
                        or len(row.emitted) >= row.generation_length
                        or reached_iteration_budget
                    ):
                        row.state = DONE
                        continue
                    budget = self.thinking_budget
                    if budget.force_next_seed(
                        len(row.emitted)
                    ) and not budget.satisfied(row.emitted):
                        row.seed = budget.end_think_token_id
                    row.block = torch.full(
                        (block_length,), self.decoder.mask_id,
                        dtype=row.block.dtype, device=self.device,
                    )
                    row.block[0] = row.seed
                    row.state = DRAFT

    # -- entry point -------------------------------------------------------

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
        # One speculative block beyond the budget, since a verify forward writes
        # a whole block before the accepted length is known.
        total = max(
            (
                prompt + generated + block_length
                for prompt, generated in zip(prompt_lengths, generation_lengths)
            ),
            default=0,
        )
        check_nemotron_context_limit(
            total, getattr(self, "model_config", None),
            serving_limit=getattr(getattr(self, "server_args", None), "max_model_len", None),
        )
        if total > self.max_length:
            self.max_length = total
        self.past_key_values = self.allocate_kv_cache(batch_size)

        rows = self._prefill_spec_rows(prompts, prompt_lengths, generation_lengths)
        self._run_speculation(rows)

        self.last_stats = []
        output = torch.full(
            (batch_size, padded_prompt_len + max(generation_lengths, default=0)),
            mask_id, dtype=prompts.dtype, device=prompts.device,
        )
        for row in rows:
            generated = torch.tensor(
                [row.emitted], dtype=prompts.dtype, device=self.device
            )[:, : row.generation_length]
            if row.stop_on_eos:
                cut = int(self.decoder.first_eos_index(generated)[0])
                generated = generated[:, : min(cut + 1, generated.shape[1])]
            row.stats.returned_tokens = int(generated.shape[1])
            self.last_stats.append(row.stats.as_dict())
            output[row.index, : row.prompt_length] = prompts[
                row.index, : row.prompt_length
            ]
            output[
                row.index,
                padded_prompt_len : padded_prompt_len + generated.shape[1],
            ] = generated[0]
        logger.info(
            "Nemotron paged self-speculation: %d request(s), mean acceptance=%s",
            batch_size,
            [round(stats["mean_acceptance"], 2) for stats in self.last_stats],
        )
        return output

    def _prefill_spec_rows(self, prompts, prompt_lengths, generation_lengths):
        rows: list[SpecRow] = []
        by_length: dict[int, list[int]] = {}
        for index, length in enumerate(prompt_lengths):
            by_length.setdefault(length, []).append(index)

        seeds: dict[int, int] = {}
        prefill_calls: dict[int, int] = {}
        with self._adapters(enabled=False):
            for length, indices in sorted(by_length.items()):
                if length == 0:
                    raise ValueError("self-speculation requires a non-empty prompt")
                seq_ids = torch.tensor(indices, device=self.device, dtype=torch.long)
                logits, calls = self._prefill_group(prompts, seq_ids, length)
                for position, index in enumerate(indices):
                    sampling = getattr(self, "_offline_sampling", [None] * len(prompt_lengths))[index]
                    seeds[index] = int(sample_tokens(logits[position, -1], sampling))
                    prefill_calls[index] = calls

        block_length = int(self.block_length)
        for index, length in enumerate(prompt_lengths):
            stats = SelfSpecStats(prefill_calls=prefill_calls.get(index, 0))
            block = torch.full(
                (block_length,), self.decoder.mask_id,
                dtype=prompts.dtype, device=self.device,
            )
            block[0] = seeds[index]
            row = SpecRow(
                index=index,
                seq_id=index,
                prompt_length=length,
                generation_length=generation_lengths[index],
                prefix_length=length,
                block=block,
                seed=seeds[index],
                stats=stats,
                sampling=getattr(self, "_offline_sampling", [None] * len(prompt_lengths))[index],
                stop_on_eos=getattr(self, "_offline_stop_on_eos", [self.early_stop] * len(prompt_lengths))[index],
            )
            # The prefill's own seed already counts as a generated token, the
            # way the dense runner counts it.
            row.emitted.append(seeds[index])
            if row.stop_on_eos and seeds[index] in self.decoder.eos_ids:
                row.terminal = True
                row.state = DONE
            elif generation_lengths[index] <= 1:
                row.state = DONE
            rows.append(row)
        return rows


__all__ = ["NemotronSelfSpecPagedRunner", "SpecRow"]
