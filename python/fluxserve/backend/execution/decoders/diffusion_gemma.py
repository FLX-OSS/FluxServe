from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from fluxserve.backend.distributed import tensor_model_parallel_all_reduce


def normalize_eos_ids(value: int | Sequence[int]) -> tuple[int, ...]:
    values = (value,) if isinstance(value, int) else tuple(value)
    eos_ids = tuple(dict.fromkeys(int(item) for item in values))
    if not eos_ids:
        raise ValueError("Diffusion-Gemma requires at least one EOS token ID")
    return eos_ids


@dataclass(frozen=True)
class DiffusionGemmaSamplingConfig:
    canvas_length: int
    max_denoising_steps: int
    t_min: float
    t_max: float
    entropy_bound: float
    confidence_threshold: float
    stability_threshold: int
    vocab_size: int
    eos_ids: tuple[int, ...]
    pad_id: int


@dataclass
class DiffusionGemmaSamplingState:
    canvas: torch.Tensor
    argmax_canvas: torch.Tensor
    soft_embeds: torch.Tensor | None = None
    step: int = 0
    history: list[torch.Tensor] | None = None

    def __post_init__(self):
        if self.history is None:
            self.history = []


class DiffusionGemmaDecoder:
    """Entropy-bound accept/renoise decoder used by Diffusion-Gemma."""

    def __init__(self, config: DiffusionGemmaSamplingConfig):
        if config.entropy_bound <= 0:
            raise ValueError("Diffusion-Gemma entropy_bound must be positive")
        self.config = config
        # Keep the common executor contract without giving the token mask semantics.
        self.mask_id = config.pad_id
        self.pad_id = config.pad_id
        self.eos_ids = normalize_eos_ids(config.eos_ids)
        self.eos_id = self.eos_ids[0]

    def eos_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for eos_id in self.eos_ids:
            mask |= token_ids == eos_id
        return mask

    def contains_eos(self, token_ids: torch.Tensor) -> bool:
        return bool(self.eos_mask(token_ids).any())

    def first_eos_index(self, token_ids: torch.Tensor) -> int | None:
        indices = self.eos_mask(token_ids).flatten().nonzero(as_tuple=True)[0]
        return int(indices[0].item()) if indices.numel() else None

    def new_state(self, device: torch.device) -> DiffusionGemmaSamplingState:
        canvas = torch.randint(
            0,
            self.config.vocab_size,
            (1, self.config.canvas_length),
            dtype=torch.long,
            device=device,
        )
        return DiffusionGemmaSamplingState(canvas=canvas, argmax_canvas=canvas.clone())

    def step(
        self,
        logits: torch.Tensor,
        state: DiffusionGemmaSamplingState,
        embed_weight: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
        normalizer: float,
    ) -> bool:
        cfg = self.config
        remaining = max(cfg.max_denoising_steps - state.step, 1)
        temperature = cfg.t_min + (cfg.t_max - cfg.t_min) * (
            remaining / cfg.max_denoising_steps
        )
        scaled = logits.float() / max(temperature, 1e-10)
        argmax = scaled.argmax(dim=-1)

        log_probs = scaled.log_softmax(dim=-1)
        probs = log_probs.exp()
        sampled = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]), num_samples=1
        ).reshape(state.canvas.shape)
        entropy = -(probs * log_probs).sum(dim=-1)
        sorted_entropy, sorted_indices = entropy.sort(dim=-1)
        cumulative = sorted_entropy.cumsum(dim=-1)
        # Match Transformers: each position is accepted when the entropy of
        # all lower-entropy positions is within the sampler budget.
        sorted_accept = (cumulative - sorted_entropy) <= cfg.entropy_bound
        accept = torch.zeros_like(sorted_accept)
        accept.scatter_(1, sorted_indices, sorted_accept)
        random_tokens = torch.randint(
            0,
            cfg.vocab_size,
            state.canvas.shape,
            dtype=state.canvas.dtype,
            device=state.canvas.device,
        )
        state.canvas = torch.where(accept, sampled, random_tokens)
        state.argmax_canvas = argmax
        state.step += 1

        state.history.append(argmax.clone())
        if len(state.history) > cfg.stability_threshold:
            state.history.pop(0)
        stable = len(state.history) >= cfg.stability_threshold and all(
            torch.equal(state.history[0], item) for item in state.history[1:]
        )
        confident = bool(entropy.mean().item() < cfg.confidence_threshold)
        converged = (stable and confident) or state.step >= cfg.max_denoising_steps

        local_probs = probs[..., vocab_start:vocab_end].to(embed_weight.dtype)
        local_weight = embed_weight[: vocab_end - vocab_start]
        soft = local_probs @ local_weight
        state.soft_embeds = tensor_model_parallel_all_reduce(soft) * normalizer
        if converged:
            state.canvas = state.argmax_canvas.clone()
        return converged


__all__ = [
    "DiffusionGemmaDecoder",
    "DiffusionGemmaSamplingConfig",
    "DiffusionGemmaSamplingState",
    "normalize_eos_ids",
]
