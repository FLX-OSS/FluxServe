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

"""Threshold denoising for Nemotron-Labs-Diffusion.

This is deliberately a separate decoder rather than a configuration of
:class:`ThresholdParallelDecoder`. The two implement the same broad strategy,
but differ in three places that change which positions get committed, and the
LLaDA defaults must not move:

======================  =========================  ========================
Behaviour               LLaDA threshold decoder    Reference / here
======================  =========================  ========================
Confidence dtype        float32 softmax            softmax in the logits'
                                                   own dtype (bfloat16)
Guaranteed progress     every position within      exactly one position,
                        1e-5 of the maximum        the highest-confidence
Mask-token prediction   refused (``rm_mask``)      allowed
======================  =========================  ========================

The reference helper is ``_get_transfer_index`` in
``modeling_nemotron_labs_diffusion.py``: it ranks every masked position by
confidence, always commits the top-ranked one, and commits the rest only where
confidence reaches the threshold. Because ``topk`` returns a descending run,
that is the same set as "the argmax, plus everything at or above the
threshold", which is what this decoder computes without a per-row loop.

Nonzero-temperature diffusion uses the reference's float64 Gumbel transform.
Self-speculation drafts instead use categorical sampling and scaled confidence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import normalize_eos_ids
from fluxserve.backend.execution.nemotron_sampling import NemotronSampling


class NemotronThresholdDecoder:
    """Commit masked positions whose confidence reaches ``threshold``.

    ``block`` tensors are ``[batch, block_length]`` token ids and are updated in
    place; ``logits`` are ``[batch, block_length, vocab]`` for the same block.
    """

    def __init__(
        self,
        *,
        threshold: float,
        mask_id: int,
        eos_ids,
        temperature: float = 0.0,
        confidence_dtype: torch.dtype | None = None,
        allow_mask_prediction: bool = True,
        draft: bool = False,
    ):
        self._default_sampling = NemotronSampling(temperature=temperature)
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
        self.temperature = temperature
        self.draft = draft
        self.threshold = float(threshold)
        self.mask_id = int(mask_id)
        self.eos_ids = normalize_eos_ids(eos_ids)
        self.eos_id = self.eos_ids[0]
        # None keeps the reference's behaviour: softmax in the logits' dtype.
        self.confidence_dtype = confidence_dtype
        self.allow_mask_prediction = bool(allow_mask_prediction)

    # -- selection ---------------------------------------------------------

    def select(self, logits: torch.Tensor, block: torch.Tensor, *, sampling=None):
        """Return ``(predicted, transfer)`` for one denoising step.

        ``predicted`` is the selected token at every position; ``transfer`` marks
        the masked positions to commit this step.
        """
        if logits.shape[:2] != block.shape:
            raise ValueError(
                f"logits {tuple(logits.shape)} do not match block "
                f"{tuple(block.shape)}"
            )
        masked = block == self.mask_id
        sampling = sampling or self._default_sampling
        predicted = (sampling.categorical(logits) if self.draft
                     else sampling.diffusion(logits))
        confidence_logits = (
            logits / sampling.temperature
            if self.draft and sampling.temperature > 0 else logits
        )

        probabilities = F.softmax(
            confidence_logits if self.confidence_dtype is None
            else confidence_logits.to(self.confidence_dtype),
            dim=-1,
        )
        confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
        if not self.allow_mask_prediction:
            masked = masked & (predicted != self.mask_id)
        confidence = torch.where(
            masked, confidence, torch.full_like(confidence, -torch.inf)
        )

        transfer = masked & (confidence >= self.threshold)
        # Always commit the single highest-confidence masked position, so a
        # block whose every candidate is below the threshold still progresses.
        # `scatter_` rather than `confidence >= max` so exact ties commit one
        # position, as the reference's top-k scan does.
        best = confidence.argmax(dim=-1, keepdim=True)
        transfer = transfer.scatter(-1, best, True) & masked
        return predicted, transfer

    def step(self, logits: torch.Tensor, block: torch.Tensor, *, sampling=None) -> torch.Tensor:
        """Apply one denoising step to ``block`` in place; return it."""
        predicted, transfer = self.select(logits, block, sampling=sampling)
        block[transfer] = predicted[transfer]
        return block

    # -- block state -------------------------------------------------------

    def has_masks(self, block: torch.Tensor) -> torch.Tensor:
        return (block == self.mask_id).any(dim=-1)

    def eos_positions(self, block: torch.Tensor) -> torch.Tensor:
        hit = torch.zeros_like(block, dtype=torch.bool)
        for eos_id in self.eos_ids:
            hit |= block == eos_id
        return hit

    def eos_is_terminal(self, block: torch.Tensor) -> torch.Tensor:
        """Per row: an EOS exists and nothing before it is still masked.

        An EOS sitting behind an unresolved mask is not terminal, because the
        masked position could still resolve to something that belongs before
        it. Truncating there would drop a position that was never decided.
        """
        eos_hit = self.eos_positions(block)
        has_eos = eos_hit.any(dim=-1)
        at_or_after_eos = eos_hit.cumsum(dim=-1).bool()
        masked_before_eos = ((block == self.mask_id) & ~at_or_after_eos).any(dim=-1)
        return has_eos & ~masked_before_eos

    def first_eos_index(self, tokens: torch.Tensor) -> torch.Tensor:
        """Index of the first EOS per row, or ``tokens.shape[-1]`` if none."""
        hit = self.eos_positions(tokens)
        length = tokens.shape[-1]
        indices = torch.arange(length, device=tokens.device).expand_as(tokens)
        return torch.where(hit, indices, torch.full_like(indices, length)).min(dim=-1).values


class ThinkingBudget:
    """Force an end-of-thinking marker once the thinking allowance is spent.

    The checkpoint's generate paths accept ``max_thinking_tokens`` and
    ``end_think_token_id`` and enforce them differently per mode, so the policy
    lives here rather than being reimplemented in each runner:

    * **Block diffusion** injects the marker into the block that would carry the
      budget past its limit, at the position where the limit falls, *before*
      denoising. The position then holds a resolved token, so the decoder never
      writes over it.
    * **Self-speculation** forces the next block's seed instead, because a
      speculative block is drafted and verified as a whole.

    Both are no-ops once the model has produced the marker on its own, and the
    whole thing is disabled unless both settings are given.
    """

    def __init__(self, *, max_thinking_tokens=None, end_think_token_id=None):
        if (max_thinking_tokens is None) != (end_think_token_id is None):
            raise ValueError(
                "max_thinking_tokens and end_think_token_id must be set "
                "together or not at all"
            )
        if max_thinking_tokens is not None and int(max_thinking_tokens) < 0:
            raise ValueError(
                f"max_thinking_tokens must be non-negative, got {max_thinking_tokens!r}"
            )
        self.max_thinking_tokens = (
            None if max_thinking_tokens is None else int(max_thinking_tokens)
        )
        self.end_think_token_id = (
            None if end_think_token_id is None else int(end_think_token_id)
        )

    @property
    def enabled(self) -> bool:
        return self.max_thinking_tokens is not None

    def satisfied(self, generated) -> bool:
        """Has the marker already been produced? Then nothing is forced."""
        if not self.enabled:
            return True
        if isinstance(generated, torch.Tensor):
            if generated.numel() == 0:
                return False
            return bool((generated == self.end_think_token_id).any())
        return self.end_think_token_id in generated

    def block_injection_offset(self, tokens_before: int, block_length: int):
        """Offset inside the next block for the marker, or ``None``.

        ``tokens_before`` counts generated tokens before this block. The marker
        goes in the block that carries the budget past its limit, at the
        position where the limit falls; a budget already spent puts it at
        offset zero, overwriting that block's seed, which is what the reference
        does.
        """
        if not self.enabled:
            return None
        if tokens_before + int(block_length) <= self.max_thinking_tokens:
            return None
        return max(0, self.max_thinking_tokens - int(tokens_before))

    def force_next_seed(self, generated_count: int) -> bool:
        """Should the next seed be the marker rather than a sampled token?"""
        return self.enabled and int(generated_count) > self.max_thinking_tokens


def load_thinking_budget(config) -> ThinkingBudget:
    return ThinkingBudget(
        max_thinking_tokens=getattr(config, "max_thinking_tokens", None),
        end_think_token_id=getattr(config, "end_think_token_id", None),
    )


def load_nemotron_decoder(config) -> NemotronThresholdDecoder:
    """Build the decoder from a ``RunnerConfig``-shaped object."""
    return NemotronThresholdDecoder(
        threshold=getattr(config, "threshold", 0.9),
        mask_id=getattr(config, "mask_id"),
        eos_ids=(
            getattr(config, "eos_ids", None) or (getattr(config, "eos_id"),)
        ),
        temperature=getattr(config, "temperature", 0.0) or 0.0,
    )


__all__ = [
    "NemotronThresholdDecoder",
    "ThinkingBudget",
    "load_nemotron_decoder",
    "load_thinking_budget",
]
