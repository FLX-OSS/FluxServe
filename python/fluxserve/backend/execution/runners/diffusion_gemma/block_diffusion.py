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
    Diffusion-Gemma runner.
"""

from __future__ import annotations

import logging

import torch
from transformers import GenerationConfig

from fluxserve.backend.execution.decoders.diffusion_gemma import (
    DiffusionGemmaDecoder,
    DiffusionGemmaSamplingConfig,
    normalize_eos_ids,
)
from fluxserve.backend.execution.runners.base import BlockDiffusionRunner

logger = logging.getLogger(__name__)


class DiffusionGemmaRunner(BlockDiffusionRunner):
    """Correctness-first text runner for Diffusion-Gemma."""

    _supports_flashinfer = False

    requires_prompt_lengths = True

    def __init__(self, model_config, server_args, runner_config=None, device="cuda"):
        use_flashinfer = bool(
            runner_config is not None
            and runner_config.attention_backend == "flashinfer"
        )
        if use_flashinfer and not self._supports_flashinfer:
            raise ValueError("Use DiffusionGemmaFlashInferRunner for FlashInfer")
        if runner_config is not None and runner_config.enable_prefill_cuda_graph:
            raise ValueError(
                "Diffusion-Gemma supports decode CUDA graphs only; disable "
                "prefill CUDA graphs and use --use-decode-cuda-graph."
            )
        if server_args.pp_size != 1 or server_args.dp_size != 1:
            raise ValueError("Diffusion-Gemma supports single-DP and no PP")
        if server_args.enable_dp_attention:
            raise ValueError("Diffusion-Gemma does not support DP attention yet")
        if server_args.tp_size != server_args.ep_size:
            raise ValueError(
                "Diffusion-Gemma requires TP and EP to use the same world size; "
                f"got tp_size={server_args.tp_size}, ep_size={server_args.ep_size}"
            )
        super().__init__(
            model_config,
            server_args,
            runner_config,
            device,
        )
        self.use_flashinfer_paged = use_flashinfer
        self.last_denoising_steps = []

    def init_decoder(self):
        try:
            generation = GenerationConfig.from_pretrained(
                self.server_args.model_name,
                trust_remote_code=self.server_args.trust_remote_code,
            ).to_dict()
        except Exception:
            logger.warning(
                "Could not load Diffusion-Gemma generation_config.json; using "
                "documented defaults.",
                exc_info=True,
            )
            generation = {}
        sampler = generation.get("sampler_config") or {}
        entropy_bound = sampler.get("entropy_bound", generation.get("entropy_bound", 0.1))
        text = self.model.text_config
        canvas_length = int(
            getattr(self.runner_config, "canvas_length", None)
            or getattr(self.model_config, "canvas_length", None)
            or self.runner_config.block_length
        )
        self.runner_config.block_length = canvas_length
        self.block_length = canvas_length
        eos_value = generation.get("eos_token_id", getattr(text, "eos_token_id", 1))
        config = DiffusionGemmaSamplingConfig(
            canvas_length=canvas_length,
            max_denoising_steps=int(
                getattr(self.runner_config, "max_denoising_steps", None)
                or generation.get("max_denoising_steps", 48)
            ),
            t_min=float(
                getattr(self.runner_config, "t_min", None)
                if getattr(self.runner_config, "t_min", None) is not None
                else generation.get("t_min", 0.0)
            ),
            t_max=float(
                getattr(self.runner_config, "t_max", None)
                if getattr(self.runner_config, "t_max", None) is not None
                else generation.get("t_max", 1.0)
            ),
            entropy_bound=float(
                getattr(self.runner_config, "entropy_bound", None) or entropy_bound
            ),
            confidence_threshold=float(
                getattr(self.runner_config, "confidence_threshold", None)
                or generation.get("confidence_threshold", 0.1)
            ),
            stability_threshold=int(
                getattr(self.runner_config, "stability_threshold", None)
                or generation.get("stability_threshold", 1)
                or 1
            )
            + 1,
            vocab_size=int(text.vocab_size),
            eos_ids=normalize_eos_ids(eos_value),
            pad_id=int(getattr(text, "pad_token_id", 0) or 0),
        )
        self.decoder = DiffusionGemmaDecoder(config)

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

    def _forward_model(self, **kwargs):
        self.num_forwards += 1
        return self.model(**kwargs)

    def _generate_one(self, prompt: torch.Tensor, generation_length: int) -> torch.Tensor:
        device = prompt.device
        prompt = prompt.unsqueeze(0)
        prompt_len = prompt.shape[1]
        canvas_length = self.decoder.config.canvas_length
        padded_generation = (
            (generation_length + canvas_length - 1) // canvas_length
        ) * canvas_length
        prefix_cache = (
            self._paged_cache(prompt_len + padded_generation)
            if self.use_flashinfer_paged
            else None
        )
        prompt_positions = torch.arange(prompt_len, device=device).unsqueeze(0)
        prefill = self._forward_model(
            input_ids=prompt,
            position_ids=prompt_positions,
            past_key_values=prefix_cache,
            use_cache=True,
            attention_mask=(
                None
                if self.use_flashinfer_paged
                else self._causal_mask(prompt_len, 0, device)
            ),
            forward_batch=(
                self._paged_batch(
                    prefix_cache.build_metadata(
                        phase="prefill",
                        seq_ids=(0,),
                        q_offsets=(0,),
                        q_lens=(prompt_len,),
                        kv_lens=(prompt_len,),
                        max_q_len=prompt_len,
                    ),
                    prefill=True,
                )
                if self.use_flashinfer_paged
                else None
            ),
        )
        prefix_cache = prefill.past_key_values
        emitted = []
        denoising_steps = 0
        remaining = generation_length
        embed = self.model.model.embed_tokens
        shard = embed.shard_indices

        while remaining > 0:
            state = self.decoder.new_state(device)
            prefix_len = prompt_len + sum(item.shape[1] for item in emitted)
            positions = torch.arange(
                prefix_len,
                prefix_len + self.decoder.config.canvas_length,
                device=device,
            ).unsqueeze(0)
            canvas_steps = 0
            for _ in range(self.decoder.config.max_denoising_steps):
                canvas_steps += 1
                inputs_embeds = self.model.embed_with_self_conditioning(
                    state.canvas, state.soft_embeds
                )
                denoise = self._forward_model(
                    input_ids=state.canvas,
                    inputs_embeds=inputs_embeds,
                    position_ids=positions,
                    past_key_values=prefix_cache,
                    use_cache=True,
                    attention_mask=(
                        None
                        if self.use_flashinfer_paged
                        else self._bidirectional_mask(
                            self.decoder.config.canvas_length, prefix_len, device
                        )
                    ),
                    forward_batch=(
                        self._paged_batch(
                            prefix_cache.build_metadata(
                                phase="denoise",
                                seq_ids=(0,),
                                q_offsets=(prefix_len,),
                                q_lens=(self.decoder.config.canvas_length,),
                                kv_lens=(prefix_len + self.decoder.config.canvas_length,),
                                max_q_len=self.decoder.config.canvas_length,
                            ),
                            prefill=False,
                        )
                        if self.use_flashinfer_paged
                        else None
                    ),
                )
                converged = self.decoder.step(
                    denoise.logits,
                    state,
                    embed.weight,
                    shard.org_vocab_start_index,
                    shard.org_vocab_end_index,
                    self.model.model.normalizer,
                )
                if converged:
                    break
            denoising_steps += canvas_steps

            commit = self._forward_model(
                input_ids=state.argmax_canvas,
                position_ids=positions,
                past_key_values=prefix_cache,
                use_cache=True,
                attention_mask=(
                    None
                    if self.use_flashinfer_paged
                    else self._causal_mask(
                        self.decoder.config.canvas_length, prefix_len, device
                    )
                ),
                forward_batch=(
                    self._paged_batch(
                        prefix_cache.build_metadata(
                            phase="commit",
                            seq_ids=(0,),
                            q_offsets=(prefix_len,),
                            q_lens=(self.decoder.config.canvas_length,),
                            kv_lens=(prefix_len + self.decoder.config.canvas_length,),
                            max_q_len=self.decoder.config.canvas_length,
                        ),
                        prefill=False,
                    )
                    if self.use_flashinfer_paged
                    else None
                ),
            )
            prefix_cache = commit.past_key_values
            take = min(remaining, self.decoder.config.canvas_length)
            emitted.append(state.argmax_canvas[:, :take])
            remaining -= take
            if self.early_stop and self.decoder.contains_eos(
                state.argmax_canvas[:, :take]
            ):
                break
        self._current_denoising_steps = denoising_steps
        return torch.cat(emitted, dim=1) if emitted else prompt[:, :0]



    @torch.no_grad()
    def generate(self, prompts, prompt_lengths=None, generation_lengths=None):
        batch_size, padded_prompt_len = prompts.shape
        if prompt_lengths is None:
            prompt_lengths = (prompts != self.decoder.pad_id).sum(dim=-1).tolist()
        else:
            prompt_lengths = [int(item) for item in prompt_lengths]
        if len(prompt_lengths) != batch_size:
            raise ValueError("prompt_lengths must contain one value per batch row")
        if generation_lengths is None:
            generation_lengths = [int(self.runner_config.gen_length)] * batch_size
        else:
            generation_lengths = [int(item) for item in generation_lengths]
        if len(generation_lengths) != batch_size:
            raise ValueError("generation_lengths must contain one value per batch row")
        if any(length < 0 for length in generation_lengths):
            raise ValueError("generation_lengths must be non-negative")
        if getattr(self, "use_flashinfer_paged", False):
            return self._generate_flashinfer_batch(
                prompts, prompt_lengths, generation_lengths
            )
        padded_generation_len = max(generation_lengths, default=0)
        rows = []
        self.last_denoising_steps = []
        for index in range(batch_size):
            prompt = prompts[index, : prompt_lengths[index]]
            generated = self._generate_one(prompt, generation_lengths[index])
            self.last_denoising_steps.append(self._current_denoising_steps)
            row = torch.full(
                (1, padded_prompt_len + padded_generation_len),
                self.decoder.pad_id,
                dtype=prompts.dtype,
                device=prompts.device,
            )
            row[0, : prompt_lengths[index]] = prompt
            row[0, padded_prompt_len : padded_prompt_len + generated.shape[1]] = generated
            rows.append(row)
        return torch.cat(rows, dim=0)


__all__ = ["DiffusionGemmaRunner"]
