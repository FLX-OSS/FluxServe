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

"""Dense block-diffusion runner for Nemotron-Labs-Diffusion.

The LLaDA loop commits a block's KV opportunistically: whichever denoising
forward happens to see the block's final tokens is the forward whose keys and
values are kept, under the block-diffusion attention pattern. Nemotron cannot
reuse that. Its committed prefix has to be produced by a *causal* forward,
because every layer above the first consumes hidden states that depend on the
attention pattern, so bidirectional-forward KV is not the KV the model expects
to read back. That causal forward also produces the next block's seed token.

One block therefore costs ``D + 1`` forwards, where ``D`` counts denoising
forwards that actually had masked positions to resolve:

======== ============================== ====================================
State    Input and attention            Cache
======== ============================== ====================================
PREFILL  prompt, strictly causal        commit prompt KV; last logit seeds
DENOISE  block, bidirectional over      ``use_cache=False`` -- the dense path
         committed prefix plus block    splices out of place, so the
                                        committed prefix cannot be touched
COMMIT   block, causal within block     write the block's KV; last logit
                                        seeds the next block
======== ============================== ====================================

Transition to COMMIT is decided on the block *after* the decoder update, so a
resolved block does not pay for an extra no-op denoising forward.

See ``docs/serving/nemotron/nemotron-labs-diffusion-14B.md`` for configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from fluxserve.backend.execution.decoders.nemotron import (
    NemotronThresholdDecoder,
    load_thinking_budget,
)
from fluxserve.backend.execution.runners.block_diffusion import BlockDiffusionRunner
from fluxserve.backend.layers.dp_attention import get_attention_tp_size
from fluxserve.backend.models.nemotron_diffusion import nemotron_head_dim
from fluxserve.backend.execution.nemotron_sampling import (
    NemotronSamplingMixin, sample_tokens,
)

logger = logging.getLogger(__name__)


class NemotronBlockBudgetExceeded(RuntimeError):
    """A block still held masked positions when its denoise budget ran out."""


@dataclass
class BlockStats:
    """Forward accounting, kept separate per phase on purpose.

    A single number hides the thing worth watching: the commit forward is not
    optional overhead, and comparing a Nemotron block's total against a LLaDA
    block's total without splitting it is not a like-for-like comparison.
    """

    prefill_calls: int = 0
    denoise_calls: int = 0
    commit_calls: int = 0
    blocks: int = 0
    denoise_per_block: list[int] = field(default_factory=list)
    internal_tokens: int = 0
    returned_tokens: int = 0

    @property
    def total_calls(self) -> int:
        return self.prefill_calls + self.denoise_calls + self.commit_calls

    def as_dict(self) -> dict:
        return {
            "prefill_calls": self.prefill_calls,
            "denoise_calls": self.denoise_calls,
            "commit_calls": self.commit_calls,
            "total_calls": self.total_calls,
            "blocks": self.blocks,
            "denoise_per_block": list(self.denoise_per_block),
            "internal_tokens": self.internal_tokens,
            "returned_tokens": self.returned_tokens,
        }


class NemotronDiffusionRunner(NemotronSamplingMixin, BlockDiffusionRunner):
    """Block diffusion with causal prefill, seeded blocks and causal commit."""

    requires_prompt_lengths = True

    def __init__(self, model_config, server_args, runner_config=None, device="cuda"):
        # Reject before the base constructor loads 27 GB of weights.
        backend = getattr(runner_config, "attention_backend", "sdpa")
        if backend != "sdpa":
            raise ValueError(
                "NemotronDiffusionRunner implements the dense path only; "
                f"got attention_backend={backend!r}. Paged FA4 serving uses "
                "NemotronFA4DiffusionRunner."
            )
        super().__init__(model_config, server_args, runner_config, device)
        self.block_length = int(self.runner_config.block_length)
        steps = int(getattr(self.runner_config, "steps", 0) or 0)
        self.max_denoise_steps = steps if steps > 0 else self.block_length
        self.thinking_budget = load_thinking_budget(self.runner_config)
        self.last_stats: list[dict] = []

    # -- construction hooks ------------------------------------------------

    def init_decoder(self):
        self.decoder = NemotronThresholdDecoder(
            threshold=self.runner_config.threshold,
            mask_id=self.runner_config.mask_id,
            eos_ids=(
                tuple(self.runner_config.eos_ids)
                or (int(self.runner_config.eos_id),)
            ),
        )

    # -- masks -------------------------------------------------------------

    @staticmethod
    def _causal_mask(query_len: int, prefix_len: int, device) -> torch.Tensor:
        query = torch.arange(query_len, device=device).unsqueeze(1)
        keys = torch.arange(prefix_len + query_len, device=device).unsqueeze(0)
        return (keys <= prefix_len + query).unsqueeze(0)

    @staticmethod
    def _bidirectional_mask(query_len: int, prefix_len: int, device) -> torch.Tensor:
        return torch.ones(
            1, query_len, prefix_len + query_len, dtype=torch.bool, device=device
        )

    # -- dense prefix cache ------------------------------------------------

    def _cache_geometry(self) -> tuple[int, int, int]:
        config = self.model.model.config
        num_layers = int(config.num_hidden_layers)
        try:
            tp_size = get_attention_tp_size()
        except AssertionError:  # direct construction, outside a served process
            tp_size = 1
        local_kv_heads = max(1, int(config.num_key_value_heads) // tp_size)
        # Read head_dim from the config: this checkpoint's 128 is not
        # hidden_size // num_attention_heads, which would give 160.
        return num_layers, local_kv_heads, nemotron_head_dim(config)

    def _allocate_prefix_cache(self, length: int) -> torch.Tensor:
        num_layers, local_kv_heads, head_dim = self._cache_geometry()
        return torch.zeros(
            num_layers,
            2,
            1,
            local_kv_heads,
            length,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _commit_kv(self, cache: torch.Tensor, present, start: int, end: int) -> None:
        """Write only the newly produced slots back into the prefix cache."""
        num_layers, local_kv_heads, head_dim = self._cache_geometry()
        # Copy only new slices; stacking all layers would duplicate the full
        # long-context KV cache just to commit one chunk.
        for layer in range(num_layers):
            for kind in range(2):
                values = present[2 * layer + kind].reshape(1, local_kv_heads, end, head_dim)
                cache[layer, kind, :, :, start:end] = values[:, :, start:end]

    # -- forwards ----------------------------------------------------------

    def _forward(self, **kwargs):
        self.num_forwards += 1
        return self.model(**kwargs)

    def _prefill(self, prompt: torch.Tensor, cache: torch.Tensor):
        length = prompt.shape[1]
        if length <= 0:
            raise ValueError("Nemotron requires a non-empty prompt")
        chunk_size = int(getattr(self.runner_config, "nemotron_prefill_chunk_size", 1024))
        self._last_prefill_calls = 0
        for start in range(0, length, chunk_size):
            end = min(length, start + chunk_size)
            positions = torch.arange(start, end, device=self.device).unsqueeze(0)
            output = self._forward(
                input_ids=prompt[:, start:end], position_ids=positions,
                past_key_values=cache[:, :, :, :, :end], use_cache=True,
                attention_mask=self._causal_mask(end - start, start, self.device),
            )
            self._commit_kv(cache, output.past_key_values, start, end)
            self._last_prefill_calls += 1
            logits = output.logits
            del output
        return logits

    def _denoise(self, block: torch.Tensor, prefix_len: int, cache: torch.Tensor):
        block_length = block.shape[1]
        positions = torch.arange(
            prefix_len, prefix_len + block_length, device=self.device
        ).unsqueeze(0)
        # use_cache=False is load-bearing, not an optimisation: the dense
        # attention path splices the block's keys out of place, so with no
        # cache returned there is no way for a denoising forward to reach the
        # committed prefix.
        output = self._forward(
            input_ids=block,
            position_ids=positions,
            past_key_values=cache[:, :, :, :, : prefix_len + block_length],
            use_cache=False,
            attention_mask=self._bidirectional_mask(
                block_length, prefix_len, self.device
            ),
        )
        return output.logits

    def _commit(self, block: torch.Tensor, prefix_len: int, cache: torch.Tensor):
        block_length = block.shape[1]
        end = prefix_len + block_length
        positions = torch.arange(
            prefix_len, end, device=self.device
        ).unsqueeze(0)
        output = self._forward(
            input_ids=block,
            position_ids=positions,
            past_key_values=cache[:, :, :, :, :end],
            use_cache=True,
            attention_mask=self._causal_mask(block_length, prefix_len, self.device),
        )
        self._commit_kv(cache, output.past_key_values, prefix_len, end)
        return output.logits

    # -- block loop --------------------------------------------------------

    def _seed_from(self, logits: torch.Tensor) -> int:
        return int(sample_tokens(logits[:, -1], getattr(self, "_sampling", None)))

    def _denoise_block(self, block: torch.Tensor, prefix_len: int,
                       cache: torch.Tensor, stats: BlockStats) -> int:
        """Run denoising until the block resolves; return the call count."""
        calls = 0
        for _ in range(self.max_denoise_steps):
            if not bool(self.decoder.has_masks(block).any()):
                break
            calls += 1
            stats.denoise_calls += 1
            logits = self._denoise(block, prefix_len, cache)
            self.decoder.step(logits, block, sampling=getattr(self, "_sampling", None))
            # Decide on the updated block: a resolved block must not pay for
            # another forward just to notice it is resolved.
            if not bool(self.decoder.has_masks(block).any()):
                break
            if self.early_stop and bool(self.decoder.eos_is_terminal(block).all()):
                break
        if bool(self.decoder.has_masks(block).any()) and not (
            self.early_stop and bool(self.decoder.eos_is_terminal(block).all())
        ):
            remaining = int((block == self.decoder.mask_id).sum())
            raise NemotronBlockBudgetExceeded(
                f"{remaining} masked position(s) remain after "
                f"{self.max_denoise_steps} denoising steps at prefix length "
                f"{prefix_len} (threshold={self.decoder.threshold}); "
                "block tokens: " + str(block[0].tolist())
            )
        return calls

    @torch.no_grad()
    def _generate_one(self, prompt: torch.Tensor, generation_length: int):
        device = self.device
        prompt = prompt.unsqueeze(0)
        prompt_len = prompt.shape[1]
        block_length = self.block_length
        stats = BlockStats()
        if generation_length <= 0:
            return prompt.new_empty((1, 0)), stats

        # The reference requires a budget divisible by the block length. Serve
        # arbitrary budgets by running whole blocks and truncating the result;
        # the extra internal work is recorded rather than hidden.
        padded_generation = (
            (generation_length + block_length - 1) // block_length
        ) * block_length
        total_length = prompt_len + padded_generation
        self._check_context(total_length)

        cache = self._allocate_prefix_cache(total_length)
        stats.prefill_calls += 1
        seed = self._seed_from(self._prefill(prompt, cache))
        stats.prefill_calls = getattr(self, "_last_prefill_calls", 1)

        prefix_len = prompt_len
        emitted: list[torch.Tensor] = []
        while len(emitted) * block_length < padded_generation:
            block = torch.full(
                (1, block_length), self.decoder.mask_id,
                dtype=prompt.dtype, device=device,
            )
            block[0, 0] = seed
            self._apply_thinking_budget(block, emitted)

            if self.early_stop and bool(self.decoder.eos_is_terminal(block).all()):
                # EOS arrived as the seed. The reference would still execute a
                # denoising iteration before its own EOS check; skipping it is
                # a serving policy choice, recorded so call counts stay
                # comparable.
                calls, terminal = 0, True
            else:
                calls = self._denoise_block(block, prefix_len, cache, stats)
                terminal = bool(self.decoder.eos_is_terminal(block).all())
            stats.denoise_per_block.append(calls)

            # The terminal block is still committed, as the reference does.
            stats.commit_calls += 1
            seed = self._seed_from(self._commit(block, prefix_len, cache))
            stats.blocks += 1
            prefix_len += block_length
            emitted.append(block)
            if self.early_stop and terminal:
                break

        generated = torch.cat(emitted, dim=1) if emitted else prompt.new_empty((1, 0))
        stats.internal_tokens = int(generated.shape[1])
        generated = generated[:, :generation_length]
        if self.early_stop:
            cut = int(self.decoder.first_eos_index(generated)[0])
            generated = generated[:, : min(cut + 1, generated.shape[1])]
        stats.returned_tokens = int(generated.shape[1])
        return generated, stats

    def _apply_thinking_budget(self, block: torch.Tensor, emitted: list) -> None:
        """Place the end-of-thinking marker in this block, if it is due.

        Injected before denoising, so the position holds a resolved token and
        the decoder never writes over it.
        """
        budget = self.thinking_budget
        if not budget.enabled:
            return
        produced = [int(value) for item in emitted for value in item[0].tolist()]
        if budget.satisfied(produced):
            return
        offset = budget.block_injection_offset(len(produced), block.shape[1])
        if offset is None:
            return
        block[0, offset] = budget.end_think_token_id

    def _check_context(self, total_length: int) -> None:
        from fluxserve.backend.model_loader.nemotron import (
            check_nemotron_context_limit,
        )

        check_nemotron_context_limit(
            total_length, getattr(self, "model_config", None),
            serving_limit=getattr(getattr(self, "server_args", None), "max_model_len", None),
        )

    # -- batch entry point -------------------------------------------------

    @torch.no_grad()
    def generate(self, prompts, prompt_lengths=None, generation_lengths=None, sampling_params=None):
        """Generate row by row, returning the executor's padded layout.

        Rows are independent requests here; batching across requests belongs to
        the paged path, not to this correctness-first dense one.
        """
        batch_size, padded_prompt_len = prompts.shape
        sampling = self._sampling_batch(batch_size, sampling_params)
        stop_on_eos = self._stop_on_eos_batch(batch_size, sampling_params)
        mask_id = self.decoder.mask_id
        if prompt_lengths is None:
            prompt_lengths = (prompts != mask_id).sum(dim=-1).tolist()
        prompt_lengths = [int(value) for value in prompt_lengths]
        if len(prompt_lengths) != batch_size:
            raise ValueError("prompt_lengths must contain one value per batch row")
        if generation_lengths is None:
            generation_lengths = [int(self.runner_config.gen_length)] * batch_size
        generation_lengths = [int(value) for value in generation_lengths]
        if len(generation_lengths) != batch_size:
            raise ValueError("generation_lengths must contain one value per batch row")
        if any(value < 0 for value in generation_lengths):
            raise ValueError("generation_lengths must be non-negative")

        padded_generation_len = max(generation_lengths, default=0)
        self.last_stats = []
        rows = []
        for index in range(batch_size):
            prompt = prompts[index, : prompt_lengths[index]]
            self._sampling = sampling[index]
            original_early_stop = self.early_stop
            self.early_stop = stop_on_eos[index]
            try:
                generated, stats = self._generate_one(prompt, generation_lengths[index])
            finally:
                self._sampling = None
                self.early_stop = original_early_stop
            self.last_stats.append(stats.as_dict())
            row = torch.full(
                (1, padded_prompt_len + padded_generation_len),
                mask_id,
                dtype=prompts.dtype,
                device=prompts.device,
            )
            row[0, : prompt_lengths[index]] = prompt
            row[0, padded_prompt_len : padded_prompt_len + generated.shape[1]] = (
                generated
            )
            rows.append(row)
        logger.info(
            "Nemotron diffusion: %d request(s), forwards=%s",
            batch_size,
            [stats["total_calls"] for stats in self.last_stats],
        )
        return torch.cat(rows, dim=0)


__all__ = [
    "BlockStats",
    "NemotronBlockBudgetExceeded",
    "NemotronDiffusionRunner",
]
