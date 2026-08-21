"""Transformers config registration for Diffusion-Gemma."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig


class DiffusionGemmaTextConfig(Gemma4TextConfig):
    model_type = "diffusion_gemma_text"

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("attention_k_eq_v", True)
        # Diffusion-Gemma's text checkpoint omits Gemma4 per-layer embeddings
        # (PLE). Keep the feature opt-in when a model explicitly configures it.
        kwargs.setdefault("hidden_size_per_layer_input", 0)
        if kwargs.get("num_experts"):
            kwargs.setdefault("enable_moe_block", True)
        super().__init__(**kwargs)


class DiffusionGemmaConfig(PretrainedConfig):
    model_type = "diffusion_gemma"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        canvas_length: int = 256,
        self_conditioning_size: int | None = None,
        vision_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.text_config = DiffusionGemmaTextConfig(**(text_config or {}))
        self.canvas_length = canvas_length
        self.self_conditioning_size = self_conditioning_size
        self.vision_config = vision_config
        self.audio_config = None
        super().__init__(**kwargs)
        self.dtype = self.text_config.dtype

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size


def register_diffusion_gemma_config() -> None:
    try:
        AutoConfig.register(DiffusionGemmaConfig.model_type, DiffusionGemmaConfig)
    except ValueError as error:
        if "already used" not in str(error):
            raise


register_diffusion_gemma_config()

__all__ = [
    "DiffusionGemmaConfig",
    "DiffusionGemmaTextConfig",
    "register_diffusion_gemma_config",
]
