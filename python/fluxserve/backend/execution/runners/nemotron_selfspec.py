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

"""Linear self-speculation for Nemotron-Labs-Diffusion (dense path).

One set of weights plays both roles against one KV cache: the diffusion mode
drafts a block under bidirectional attention, the autoregressive mode verifies
it under causal attention, and the longest matching prefix plus one bonus token
is accepted. See
``docs/serving/nemotron/nemotron-labs-diffusion-14B.md`` for configuration.

Two properties carry over from diffusion mode rather than fighting it. The
committed KV always comes from the causal verify forward, which is the rule G1
imposes everywhere in this model. And the emitted tokens are the *verifier's*,
never the draft's. At temperature zero, accepting `k` tokens is exactly what
`k` greedy autoregressive steps would have produced. Positive temperatures use
the checkpoint's categorical draft/verify rule, with request-owned randomness.

What is new is the rollback. The verify forward writes a whole block's keys and
values, and only the accepted prefix is kept. On this dense path that is a
prefix length: the cache buffer is preallocated and the next iteration
overwrites from the accepted end. The paged runner retains reserved pages and
uses the same logical-prefix rule under continuous scheduling.
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
from fluxserve.backend.execution.runners.nemotron_diffusion import (
    NemotronDiffusionRunner,
)

logger = logging.getLogger(__name__)


@dataclass
class SelfSpecStats:
    """Per-request accounting.

    Acceptance length is the number that decides whether self-speculation is
    worth anything: at one accepted token per iteration it is strictly worse
    than autoregressive decoding, because it pays for a draft forward too.
    """

    prefill_calls: int = 0
    draft_calls: int = 0
    verify_calls: int = 0
    iterations: int = 0
    accepted_per_iteration: list[int] = field(default_factory=list)
    draft_calls_per_iteration: list[int] = field(default_factory=list)
    returned_tokens: int = 0

    @property
    def total_calls(self) -> int:
        return self.prefill_calls + self.draft_calls + self.verify_calls

    @property
    def mean_acceptance(self) -> float:
        if not self.accepted_per_iteration:
            return 0.0
        return sum(self.accepted_per_iteration) / len(self.accepted_per_iteration)

    def as_dict(self) -> dict:
        return {
            "prefill_calls": self.prefill_calls,
            "draft_calls": self.draft_calls,
            "verify_calls": self.verify_calls,
            "total_calls": self.total_calls,
            "iterations": self.iterations,
            "accepted_per_iteration": list(self.accepted_per_iteration),
            "draft_calls_per_iteration": list(self.draft_calls_per_iteration),
            "mean_acceptance": self.mean_acceptance,
            "returned_tokens": self.returned_tokens,
            "tokens_per_forward": (
                self.returned_tokens / self.total_calls if self.total_calls else 0.0
            ),
        }


class NemotronSelfSpecRunner(NemotronDiffusionRunner):
    """Diffusion draft, autoregressive verify, one KV cache."""

    requires_prompt_lengths = True

    def __init__(self, model_config, server_args, runner_config=None, device="cuda"):
        super().__init__(model_config, server_args, runner_config, device)
        # The reference's own default: fill every masked position in one
        # forward. A positive threshold makes the draft iterate, which costs
        # forwards for a possibly better acceptance length.
        self.draft_threshold = float(
            getattr(self.runner_config, "draft_threshold", 0.0) or 0.0
        )
        # A second decoder instance, because the draft's threshold is not the
        # serving threshold: the reference drafts at 0 by default, filling every
        # masked position in one forward.
        self.draft_decoder = NemotronThresholdDecoder(
            threshold=self.draft_threshold,
            mask_id=self.decoder.mask_id,
            eos_ids=self.decoder.eos_ids,
            draft=True,
        )
        self.thinking_budget = load_thinking_budget(self.runner_config)
        self.lora = None

    # -- draft selection ---------------------------------------------------
    #
    # The draft reuses the diffusion decoder rather than reimplementing the
    # inline rule in `linear_spec_generate`, because at batch size one the two
    # are the same function. The reference commits every masked position at or
    # above the threshold and falls back to the argmax only when nothing
    # clears; the decoder commits the same set plus the argmax unconditionally.
    # Whenever anything clears the threshold the argmax clears it too, being
    # the maximum, so the extra term is a no-op; when nothing clears, both
    # commit exactly the argmax. `threshold = 0` also coincides: every
    # confidence is a probability, so the whole masked set is admitted, which
    # is the reference's one-forward fill. A test checks this exhaustively over
    # a confidence grid rather than resting on the argument.
    #
    # The one place they could diverge is the reference's flattened
    # `conf.view(-1).argmax()`, which would pick a single position across an
    # entire batch. Self-speculation is batch size one, so it cannot arise
    # here; a batched implementation must not copy that line.

    def _draft_block(self, block: torch.Tensor, prefix_len: int,
                     cache: torch.Tensor, stats: SelfSpecStats) -> int:
        calls = 0
        for _ in range(self.max_denoise_steps):
            if not bool(self.draft_decoder.has_masks(block).any()):
                break
            calls += 1
            stats.draft_calls += 1
            logits = self._denoise(block, prefix_len, cache)
            self.draft_decoder.step(logits, block, sampling=getattr(self, "_sampling", None))
        if bool(self.draft_decoder.has_masks(block).any()):
            remaining = int((block == self.draft_decoder.mask_id).sum())
            raise RuntimeError(
                f"draft left {remaining} masked position(s) after {calls} "
                f"forward(s) at prefix length {prefix_len}"
            )
        return calls

    # -- acceptance --------------------------------------------------------

    @staticmethod
    def accepted_length(verified: torch.Tensor, drafted: torch.Tensor) -> int:
        """Longest matching prefix plus the verifier's bonus token.

        ``verified[i]`` is the autoregressive prediction made *from* position
        ``i``, so it is compared against the draft at ``i + 1``. Even a total
        mismatch accepts one token, because ``verified[0]`` is a correct
        autoregressive step regardless of what the draft guessed.
        """
        length = int(drafted.shape[-1])
        accepted = 0
        for index in range(length - 1):
            if int(verified[index]) != int(drafted[index + 1]):
                break
            accepted += 1
        return accepted + 1

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def _generate_one(self, prompt: torch.Tensor, generation_length: int):
        device = self.device
        prompt = prompt.unsqueeze(0)
        prompt_len = prompt.shape[1]
        block_length = self.block_length
        stats = SelfSpecStats()
        if generation_length <= 0:
            return prompt.new_empty((1, 0)), stats

        # Each iteration can extend the cache by a whole block before rolling
        # back, so the buffer must hold one speculative block beyond the budget.
        total_length = prompt_len + generation_length + block_length
        self._check_context(total_length)
        cache = self._allocate_prefix_cache(total_length)

        with self._adapters(enabled=False):
            stats.prefill_calls += 1
            logits = self._prefill(prompt, cache)
        next_token = self._seed_from(logits)
        stats.prefill_calls = getattr(self, "_last_prefill_calls", 1)

        prefix_len = prompt_len
        emitted: list[int] = [next_token]
        terminal = self.early_stop and next_token in self.decoder.eos_ids

        while len(emitted) < generation_length and not terminal:
            block = torch.full(
                (1, block_length), self.decoder.mask_id,
                dtype=prompt.dtype, device=device,
            )
            block[0, 0] = next_token

            with self._adapters(enabled=True):
                draft_calls = self._draft_block(block, prefix_len, cache, stats)

            with self._adapters(enabled=False):
                stats.verify_calls += 1
                verify_logits = self._commit(block, prefix_len, cache)
            verified = sample_tokens(verify_logits[0], getattr(self, "_sampling", None))

            accepted = self.accepted_length(verified, block[0])
            accepted_tokens = [int(value) for value in verified[:accepted]]

            stats.iterations += 1
            stats.accepted_per_iteration.append(accepted)
            stats.draft_calls_per_iteration.append(draft_calls)

            # Roll back: the verify forward wrote a whole block, but only the
            # accepted prefix is real. Advancing the prefix by `accepted` leaves
            # the rest to be overwritten by the next iteration.
            prefix_len += accepted
            next_token = accepted_tokens[-1]

            for token in accepted_tokens:
                emitted.append(token)
                if self.early_stop and token in self.decoder.eos_ids:
                    terminal = True
                    break
            if len(emitted) >= generation_length:
                break
            # A speculative block is drafted and verified whole, so the budget
            # is enforced on the next seed rather than inside a block.
            budget = self.thinking_budget
            if budget.force_next_seed(len(emitted)) and not budget.satisfied(
                emitted
            ):
                next_token = budget.end_think_token_id

        generated = torch.tensor(
            [emitted], dtype=prompt.dtype, device=device
        )[:, :generation_length]
        if self.early_stop:
            cut = int(self.decoder.first_eos_index(generated)[0])
            generated = generated[:, : min(cut + 1, generated.shape[1])]
        stats.returned_tokens = int(generated.shape[1])
        return generated, stats

    # -- LoRA --------------------------------------------------------------

    def _adapters(self, *, enabled: bool):
        """Context manager toggling the draft adapter, a no-op without one."""
        runner = self

        class _Toggle:
            def __enter__(self):
                if runner.lora is not None:
                    runner.lora.apply(enabled)
                return self

            def __exit__(self, *exc):
                if runner.lora is not None:
                    runner.lora.apply(False)
                return False

        return _Toggle()

    def load_draft_adapter(self, path=None) -> None:
        """Attach the ``linear_spec_lora`` draft adapter, if present."""
        from fluxserve.backend.model_loader.nemotron import load_nemotron_lora

        self.lora = load_nemotron_lora(self.model, self.model_config, path)
        if self.lora is not None:
            logger.info(
                "Nemotron draft adapter loaded for %d o_proj layers",
                len(self.lora.layers),
            )

    # -- batch entry point -------------------------------------------------

    @torch.no_grad()
    def generate(self, prompts, prompt_lengths=None, generation_lengths=None, sampling_params=None):
        if prompts.shape[0] != 1 and not getattr(
            self.server_args, "allow_selfspec_batching", False
        ):
            # The reference is batch size one and acceptance lengths differ per
            # row, so a batched implementation is a separate design question.
            raise ValueError(
                "Nemotron self-speculation runs one request at a time; got "
                f"batch size {prompts.shape[0]}"
            )
        return super().generate(prompts, prompt_lengths, generation_lengths, sampling_params)


__all__ = ["NemotronSelfSpecRunner", "SelfSpecStats"]
