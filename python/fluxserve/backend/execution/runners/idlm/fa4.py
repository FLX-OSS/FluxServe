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
    I-DLM FA4 paged ISD runner.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from fluxserve.backend.execution.decoders.idlm import ISDDecoder, make_sampling
from fluxserve.backend.execution.decoders.nemotron import sample_tokens
from fluxserve.backend.execution.fa4_cuda_graph_runner import NEMOTRON_COMMIT
from fluxserve.backend.execution.runners.nemotron.fa4 import NemotronFA4DiffusionRunner

logger = logging.getLogger(__name__)

# The paper's recommended stride (``idlm_blockN4_config.yaml``) for both sizes.
DEFAULT_GEN_BLOCK_SIZE = 4


def idlm_gen_block_size(runner_config) -> int:
    value = getattr(runner_config, "idlm_gen_block_size", None)
    return DEFAULT_GEN_BLOCK_SIZE if value is None else int(value)


@dataclass
class ISDStats:
    """Per-request accounting.

    Tokens per step is the number that explains I-DLM's throughput: every step
    is one forward, and a step emits between one and ``N`` tokens.
    """

    prefill_calls: int = 0
    steps: int = 0
    cold_steps: int = 0
    verify_steps: int = 0
    accepted_specs: int = 0
    rejections: int = 0
    tokens_per_step: list[int] = field(default_factory=list)
    returned_tokens: int = 0

    @property
    def total_calls(self) -> int:
        return self.prefill_calls + self.steps

    def as_dict(self) -> dict:
        emitted = sum(self.tokens_per_step)
        verified = self.accepted_specs + self.rejections
        return {
            "prefill_calls": self.prefill_calls,
            "steps": self.steps,
            "total_calls": self.total_calls,
            "cold_steps": self.cold_steps,
            "verify_steps": self.verify_steps,
            "accepted_specs": self.accepted_specs,
            "rejections": self.rejections,
            "tokens_per_step": list(self.tokens_per_step),
            # Prefill counts, as in the reference: its seed is emitted by the
            # first step, so autoregressive decoding (N=1) is exactly 1.0.
            "tokens_per_forward": emitted / self.total_calls if self.total_calls else 0.0,
            "spec_acceptance": self.accepted_specs / verified if verified else 0.0,
            "returned_tokens": self.returned_tokens,
        }


@dataclass
class ISDRow:
    """Per-request ISD state. It outlives a plan call; prefixes diverge.

    ``prefix`` is the committed KV length and ``anchor`` the token at that
    position. Only the prefill seed is an anchor that has not been emitted.
    """

    index: int
    seq_id: int
    prompt_length: int
    generation_length: int
    prefix: int
    anchor: int
    anchor_emitted: bool = False
    specs: list = field(default_factory=list)
    draft_logits: torch.Tensor | None = None
    sampling: object = None
    stop_on_eos: bool = True
    # None means "until the budget or EOS"; the online path sets 1, because one
    # plan call advances a request by one ISD step.
    max_steps: int | None = None
    steps_this_call: int = 0
    advance_this_call: int = 0
    emitted: list = field(default_factory=list)
    terminal: bool = False
    done: bool = False
    stats: ISDStats = field(default_factory=ISDStats)


class IDLMPagedRunner(NemotronFA4DiffusionRunner):
    """I-DLM Introspective Strided Decoding over FA4 paging.

    Inherits Nemotron's paged plumbing, which already speaks causal attention
    at arbitrary offsets: the full-prompt causal prefill that samples a seed,
    one varlen task per request, slots, and decode-graph replay. What differs is
    the decode step: one causal forward of ``2N - 1`` tokens per request, which
    both verifies the previous drafts and drafts the next ones.
    """

    # Qwen3 is accepted as the AR base of a gated LoRA (R-ISD) only.
    supported_architecture_keys = ("sdar", "qwen3")
    supported_architecture_label = "I-DLM (SDAR architecture, or a Qwen3 R-ISD base)"
    requires_prompt_lengths = True
    paged_attention_backend = "fa4"

    def __init__(self, *args, **kwargs):
        # Reject before the base constructor loads the weights.
        runner_config = kwargs.get("runner_config", args[2] if len(args) > 2 else None)
        if runner_config is not None:
            gen_block_size = idlm_gen_block_size(runner_config)
            page_size = int(runner_config.page_size or runner_config.block_length)
            if 2 * gen_block_size - 1 > page_size:
                raise ValueError(
                    f"--idlm-gen-block-size {gen_block_size} needs a "
                    f"{2 * gen_block_size - 1}-token forward, wider than "
                    f"page_size={page_size}; use a larger --block-length/"
                    "--page-size or a smaller stride"
                )
        super().__init__(*args, **kwargs)
        width = int(self.decoder.width)
        page_size = int(self.runner_config.page_size or self.block_length)
        # The decode graph captures the ISD width, and only causal forwards.
        self.graph_input_length = width
        if self.nemotron_graph_runner is not None:
            from fluxserve.backend.execution.fa4_cuda_graph_runner import (
                NemotronCudaGraphRunner,
            )

            self.fa4_graph_runner = NemotronCudaGraphRunner(
                self.nemotron_graph_runner.batch_sizes,
                backend=self.paged_attention_backend,
                variants=(NEMOTRON_COMMIT,),
            )
            self.nemotron_graph_runner = self.fa4_graph_runner
        self._isd_rows: dict[str, ISDRow] = {}
        self.lora = None
        logger.info(
            "I-DLM ISD: gen_block_size=%d, forward width=%d, page_size=%d",
            self.decoder.gen_block_size, width, page_size,
        )

    def init_decoder(self):
        self.decoder = ISDDecoder(
            gen_block_size=idlm_gen_block_size(self.runner_config),
            mask_id=self.runner_config.mask_id,
            eos_ids=(
                tuple(self.runner_config.eos_ids)
                or (int(self.runner_config.eos_id),)
            ),
        )

    # -- R-ISD gated LoRA --------------------------------------------------

    def load_lora(self, path) -> None:
        """Attach the gated adapter; MASK positions draft with it (R-ISD)."""
        from fluxserve.backend.model_loader.loader import load_idlm_lora

        # Captured kernels retain weight addresses; recapture after loading.
        graph_runner = getattr(self, "nemotron_graph_runner", None)
        if graph_runner is not None:
            graph_runner.invalidate()
        self.lora = load_idlm_lora(self.model, path, mask_id=self.decoder.mask_id)

    @contextmanager
    def _lora_scope(self, active: bool):
        lora = getattr(self, "lora", None)
        if lora is None:
            yield
            return
        previous = lora.active
        lora.active = active
        try:
            yield
        finally:
            lora.active = previous

    def graph_capture_context(self, *, causal: bool):
        # Decode graphs are ISD steps, which draft through the adapter.
        del causal
        return self._lora_scope(True)

    # -- sampling ----------------------------------------------------------

    def _sampling_batch(self, batch_size, sampling_params=None):
        if sampling_params is None or isinstance(sampling_params, dict):
            sampling_params = [sampling_params] * batch_size
        if len(sampling_params) != batch_size:
            raise ValueError("sampling_params must contain one entry per request")
        return [make_sampling(self.runner_config, params) for params in sampling_params]

    def _sampling_for_request(self, request):
        if not hasattr(self, "_request_sampling"):
            self._request_sampling = {}
        rid = str(request.rid)
        if rid not in self._request_sampling:
            self._request_sampling[rid] = make_sampling(
                self.runner_config, getattr(request, "sampling_params", {})
            )
        return self._request_sampling[rid]

    # -- continuously scheduled paged serving ------------------------------
    #
    # As in self-speculation, a step advances the committed prefix by a
    # variable amount, so the prefix is tracked per request and only the
    # advance is reserved: the scheduler then keeps two pages of capacity past
    # the prefix (one from prefill, one from the first decode schedule), which
    # covers a forward of up to one page.

    def _release_paged_slot(self, request_id: str) -> None:
        super()._release_paged_slot(request_id)
        self._isd_rows.pop(str(request_id), None)

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
                for row, rid in enumerate(request_ids[:num_prefill]):
                    state = states_by_id.get(rid)
                    if state is None:
                        continue
                    state.plan_prefill_done = True
                    seed = self._request_seeds.get(str(rid))
                    if seed is not None and str(rid) not in self._isd_rows:
                        # The prefill that reached the prompt's end fixes the
                        # first anchor and where the first step starts.
                        self._isd_rows[str(rid)] = ISDRow(
                            index=row,
                            seq_id=slot_indices[row],
                            prompt_length=len(state.input_ids),
                            generation_length=state.max_new_tokens,
                            prefix=len(state.input_ids),
                            anchor=int(seed),
                            sampling=self._sampling_for_request(state),
                            stop_on_eos=not state.ignore_eos,
                        )
                    results.append(ForwardStepResult(rid=rid, token_ids=[], text=""))
            if num_prefill < len(request_ids):
                results.extend(
                    self._plan_isd(op, num_prefill, slot_indices, states_by_id, tokenizer)
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
    def _plan_isd(self, op, decode_start_row, slot_indices, states_by_id, tokenizer):
        from fluxserve.backend.engine.executor import ForwardStepResult

        request_ids = list(op.request_ids)
        rows: list[ISDRow] = []
        for row in range(decode_start_row, len(request_ids)):
            rid = request_ids[row]
            state = states_by_id.get(rid)
            if state is None or state.finished:
                continue
            isd = self._isd_rows.get(str(rid))
            if isd is None:
                raise RuntimeError(
                    f"request {rid} reached ISD decode without a seed; its "
                    "causal prefill did not complete"
                )
            isd.index = row
            isd.seq_id = slot_indices[row]
            isd.generation_length = max(1, state.max_new_tokens - len(state.output_ids))
            isd.max_steps = 1
            isd.steps_this_call = 0
            isd.advance_this_call = 0
            isd.emitted = []
            isd.done = False
            rows.append(isd)

        if not rows:
            return []
        self._run_isd(rows)

        results = []
        eos_ids = frozenset(int(value) for value in self.decoder.eos_ids)
        mask_id = int(self.decoder.mask_id)
        for row in rows:
            rid = request_ids[row.index]
            state = states_by_id[rid]
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
                generated = [
                    token for token in generated
                    if token != mask_id and token not in eos_ids
                ]
            else:
                generated = [token for token in generated if token != mask_id]
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
                    # Acquire() adds to the previous reservation; only the
                    # committed advance needs new capacity.
                    reserve_tokens=0 if finished else max(1, row.advance_this_call),
                    # The prefix is tracked here, not as fixed block strides.
                    decode_block_completed=False,
                )
            )
        return results

    # -- the loop ----------------------------------------------------------

    def _isd_launch(self, rows: list[ISDRow]) -> torch.Tensor:
        width = int(self.decoder.width)
        seq_ids = torch.tensor(
            [row.seq_id for row in rows], device=self.device, dtype=torch.long
        )
        q_offsets = torch.tensor(
            [row.prefix for row in rows], device=self.device, dtype=torch.long
        )
        tokens = torch.tensor(
            [self.decoder.block(row.anchor, row.specs) for row in rows],
            device=self.device, dtype=torch.long,
        )
        positions = (
            torch.arange(width, device=self.device).unsqueeze(0)
            + q_offsets.unsqueeze(1)
        )
        # Only the forward is captured; verification, sampling and rollback
        # stay outside the graph, which is refreshed from each row's prefix.
        # A gated adapter drafts at the MASK positions of a step; prefill
        # never enables it, so the prompt's KV is the base model's.
        with self._lora_scope(True):
            replay = self._graph_replay(
                seq_ids=seq_ids, tokens=tokens, positions=positions, causal=True
            )
            if replay is not None:
                return replay.logits
            return self._paged_forward(
                seq_ids=seq_ids,
                tokens=tokens,
                q_offsets=q_offsets,
                causal=True,
                is_prefill=False,
                max_input_len=width,
                positions=positions,
            )

    def _emit(self, row: ISDRow, tokens) -> None:
        eos_ids = self.decoder.eos_ids
        for token in tokens:
            row.emitted.append(int(token))
            if row.stop_on_eos and int(token) in eos_ids:
                row.terminal = True
                break

    def _run_isd(self, rows: list[ISDRow]) -> None:
        # A seed that already ends the request needs no forward: EOS, or a
        # budget of one token.
        for row in rows:
            if not row.anchor_emitted and (
                (row.stop_on_eos and row.anchor in self.decoder.eos_ids)
                or row.generation_length <= 1
            ):
                row.anchor_emitted = True
                self._emit(row, [row.anchor])
                row.done = True

        while True:
            active = [row for row in rows if not row.done]
            if not active:
                break
            logits = self._isd_launch(active)
            outcomes = self.decoder.resolve(
                logits,
                [row.specs for row in active],
                [row.draft_logits for row in active],
                [row.sampling for row in active],
            )
            totals = self._decode_totals()
            for row, outcome in zip(active, outcomes, strict=True):
                tokens = ([] if row.anchor_emitted else [row.anchor]) + outcome.emitted
                row.stats.steps += 1
                if row.specs:
                    row.stats.verify_steps += 1
                else:
                    row.stats.cold_steps += 1
                row.stats.accepted_specs += outcome.accepted
                row.stats.rejections += int(outcome.rejected)
                row.stats.tokens_per_step.append(len(tokens))
                totals["row_steps"] += 1
                totals["step_tokens"] += len(outcome.emitted)
                totals["accepted_specs"] += outcome.accepted
                totals["rejections"] += int(outcome.rejected)

                # Roll back by advancing only past the anchor and the accepted
                # specs; everything this forward wrote beyond is overwritten
                # by the next step, which starts there.
                row.prefix += outcome.advance
                row.advance_this_call += outcome.advance
                row.steps_this_call += 1
                row.anchor = outcome.next_anchor
                row.anchor_emitted = True
                row.specs = outcome.next_specs
                row.draft_logits = outcome.next_draft_logits
                self._emit(row, tokens)
                if (
                    row.terminal
                    or len(row.emitted) >= row.generation_length
                    or (row.max_steps is not None and row.steps_this_call >= row.max_steps)
                ):
                    row.done = True

    def _decode_totals(self) -> dict[str, int]:
        if "_isd_totals" not in self.__dict__:
            self._isd_totals = {
                "row_steps": 0, "step_tokens": 0, "accepted_specs": 0, "rejections": 0,
            }
        return self._isd_totals

    def decode_stats(self) -> dict[str, int | float]:
        """Cumulative ISD accounting, served on ``/metrics``.

        ``idlm_tokens_per_step`` is the mean number of tokens one decode step
        commits for one request (1 to ``N``; the prefill seed is not counted),
        the number that explains I-DLM's throughput.
        """
        totals = self._decode_totals()
        verified = totals["accepted_specs"] + totals["rejections"]
        return {
            **{f"idlm_{name}": value for name, value in totals.items()},
            "idlm_tokens_per_step": (
                totals["step_tokens"] / totals["row_steps"] if totals["row_steps"] else 0.0
            ),
            "idlm_spec_acceptance": totals["accepted_specs"] / verified if verified else 0.0,
            # Gauges, not counters: requests the runner holds right now. Both
            # return to zero once every request has finished or been aborted.
            "idlm_active_rows": len(self._isd_rows),
            "idlm_slots_in_use": len(getattr(self, "_paged_request_slots", {})),
        }

    # -- entry point -------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompts, prompt_lengths=None, generation_lengths=None, sampling_params=None):
        from fluxserve.backend.models.idlm import check_idlm_context_limit

        batch_size, padded_prompt_len = prompts.shape
        samplings = self._sampling_batch(batch_size, sampling_params)
        stop_on_eos = self._stop_on_eos_batch(batch_size, sampling_params)
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

        # Every step writes a whole ISD block past the committed prefix before
        # the advance is known.
        width = int(self.decoder.width)
        total = max(
            (
                prompt + generated + width
                for prompt, generated in zip(prompt_lengths, generation_lengths)
            ),
            default=0,
        )
        check_idlm_context_limit(
            total, getattr(self, "model_config", None),
            serving_limit=getattr(getattr(self, "server_args", None), "max_model_len", None),
        )
        if total > self.max_length:
            self.max_length = total
        self.past_key_values = self.allocate_kv_cache(batch_size)

        rows = self._prefill_isd_rows(
            prompts, prompt_lengths, generation_lengths, samplings, stop_on_eos
        )
        self._run_isd([row for row in rows if row.generation_length > 0])

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
            output[row.index, : row.prompt_length] = prompts[row.index, : row.prompt_length]
            output[
                row.index, padded_prompt_len : padded_prompt_len + generated.shape[1]
            ] = generated[0]
        logger.info(
            "I-DLM ISD: %d request(s), tokens/forward=%s",
            batch_size,
            [round(stats["tokens_per_forward"], 2) for stats in self.last_stats],
        )
        return output

    def _prefill_isd_rows(self, prompts, prompt_lengths, generation_lengths,
                          samplings, stop_on_eos):
        by_length: dict[int, list[int]] = {}
        for index, length in enumerate(prompt_lengths):
            by_length.setdefault(length, []).append(index)

        seeds: dict[int, int] = {}
        prefill_calls: dict[int, int] = {}
        for length, indices in sorted(by_length.items()):
            if length == 0:
                raise ValueError("I-DLM requires a non-empty prompt")
            seq_ids = torch.tensor(indices, device=self.device, dtype=torch.long)
            logits, calls = self._prefill_group(prompts, seq_ids, length)
            for position, index in enumerate(indices):
                seeds[index] = int(sample_tokens(logits[position, -1], samplings[index]))
                prefill_calls[index] = calls

        rows = []
        for index, length in enumerate(prompt_lengths):
            row = ISDRow(
                index=index,
                seq_id=index,
                prompt_length=length,
                generation_length=generation_lengths[index],
                prefix=length,
                anchor=seeds[index],
                sampling=samplings[index],
                stop_on_eos=stop_on_eos[index],
            )
            row.stats.prefill_calls = prefill_calls[index]
            rows.append(row)
        return rows


__all__ = [
    "DEFAULT_GEN_BLOCK_SIZE",
    "IDLMPagedRunner",
    "ISDRow",
    "ISDStats",
    "idlm_gen_block_size",
]
