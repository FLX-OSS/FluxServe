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
    Transformers config for Nemotron-Labs-Diffusion.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig


class NemotronLabsDiffusionConfig(PretrainedConfig):
    """Native counterpart of the checkpoint's remote-code config.

    Once registered, Transformers resolves ``nemotron_labs_diffusion`` to this
    class with or without ``--trust-remote-code``, so it must produce the same
    fields as the checkpoint's own class.
    """

    model_type = "nemotron_labs_diffusion"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 34,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-05,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_parameters: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        sliding_window: int | None = None,
        attn_implementation: str = "sdpa",
        mask_token_id: int = -1,
        dlm_paradigm: str = "bidirectional",
        block_size: int = 32,
        **kwargs: Any,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.rope_theta = (rope_parameters or {}).get("rope_theta", rope_theta)
        self.rope_scaling = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.sliding_window = sliding_window
        self.attn_implementation = attn_implementation
        self.mask_token_id = mask_token_id
        self.dlm_paradigm = dlm_paradigm
        self.block_size = block_size
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def register_nemotron_diffusion_config() -> None:
    try:
        AutoConfig.register(
            NemotronLabsDiffusionConfig.model_type, NemotronLabsDiffusionConfig
        )
    except ValueError as error:
        if "already used" not in str(error):
            raise


register_nemotron_diffusion_config()

__all__ = [
    "NemotronLabsDiffusionConfig",
    "register_nemotron_diffusion_config",
]
