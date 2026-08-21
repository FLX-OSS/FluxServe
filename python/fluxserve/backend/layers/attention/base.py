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

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from fluxserve.backend.execution.cuda_graph_runner import get_is_capture_mode
logger = logging.getLogger(__name__)
_FLEX_ATTENTION = None
_CREATE_BLOCK_MASK = None


@dataclass
class AttentionForwardConfig:
    layer_id: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_key_value_groups: int
    scale: float
    alt_stream: Optional[torch.cuda.Stream] = None


def _load_flex_attention():
    global _FLEX_ATTENTION, _CREATE_BLOCK_MASK
    if _FLEX_ATTENTION is not None:
        return _FLEX_ATTENTION, _CREATE_BLOCK_MASK
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except ImportError as exc:
        raise RuntimeError("FlexAttention is unavailable in this PyTorch build.") from exc
    if torch.cuda.is_available():
        flex_attention = torch.compile(
            flex_attention,
            mode="max-autotune",
            fullgraph=True,
        )
    _FLEX_ATTENTION = flex_attention
    _CREATE_BLOCK_MASK = create_block_mask
    return _FLEX_ATTENTION, _CREATE_BLOCK_MASK


def apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    query_layernorm,
    key_layernorm,
    head_dim: int,
    alt_stream: Optional[torch.cuda.Stream],
) -> tuple[torch.Tensor, torch.Tensor]:
    def apply_flux_qk_norm(q_tensor, k_tensor):
        if (
            not q_tensor.is_cuda
            or not k_tensor.is_cuda
            or q_tensor.dtype != k_tensor.dtype
            or not hasattr(query_layernorm, "weight")
            or not hasattr(key_layernorm, "weight")
            or query_layernorm.weight.shape[0] != head_dim
            or key_layernorm.weight.shape[0] != head_dim
        ):
            return None

        try:
            from flux_kernel.ops import qk_rmsnorm
        except Exception:
            return None

        try:
            q_out, k_out = qk_rmsnorm(
                q_tensor,
                k_tensor,
                query_layernorm.weight,
                key_layernorm.weight,
                query_layernorm.variance_epsilon,
            )
            return q_out.view(q_tensor.shape), k_out.view(k_tensor.shape)
        except Exception:
            logger.debug(
                "flux_kernel QK RMSNorm failed; falling back to separate RMSNorms",
                exc_info=True,
            )
            return None

    def apply_q_norm(q_tensor):
        q_by_head = q_tensor.reshape(-1, head_dim)
        q_by_head = query_layernorm(q_by_head)
        return q_by_head.view(q_tensor.shape)

    def apply_k_norm(k_tensor):
        k_by_head = k_tensor.reshape(-1, head_dim)
        k_by_head = key_layernorm(k_by_head)
        return k_by_head.view(k_tensor.shape)

    if alt_stream is not None and get_is_capture_mode():
        current_stream = torch.cuda.current_stream()
        alt_stream.wait_stream(current_stream)
        q = apply_q_norm(q)
        with torch.cuda.stream(alt_stream):
            k = apply_k_norm(k)
        current_stream.wait_stream(alt_stream)
    else:
        flux_kernel_output = apply_flux_qk_norm(q, k)
        if flux_kernel_output is not None:
            return flux_kernel_output
        q = apply_q_norm(q)
        k = apply_k_norm(k)
    return q, k


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        slen,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DenseAttention:
    def __init__(self, config: AttentionForwardConfig):
        self.config = config

    def splice_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_values,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past_key_values is None:
            return k, v
        cache_k = past_key_values[0]
        cache_v = past_key_values[1]
        cache_length = cache_k.shape[2]
        block_length = k.shape[2]
        k = cache_k.slice_scatter(
            k,
            dim=2,
            start=cache_length - block_length,
            end=cache_length,
        )
        v = cache_v.slice_scatter(
            v,
            dim=2,
            start=cache_length - block_length,
            end=cache_length,
        )
        return k, v

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        k, v = self._apply_repeat(k, v)
        return self._run_attention(q, k, v, attention_mask)

    def _run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.config.scale,
            )

        if attention_mask.__class__.__name__ == "BlockMask":
            flex_attention, _ = _load_flex_attention()
            return flex_attention(
                q,
                k,
                v,
                block_mask=attention_mask,
                scale=self.config.scale,
            )

        if len(attention_mask.shape) == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.config.scale,
        )

    @torch.compiler.disable(recursive=False)
    def _apply_repeat(self, k, v):
        if self.config.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.config.alt_stream.wait_stream(current_stream)
            k = repeat_kv(k, self.config.num_key_value_groups)
            with torch.cuda.stream(self.config.alt_stream):
                v = repeat_kv(v, self.config.num_key_value_groups)
            current_stream.wait_stream(self.config.alt_stream)
        else:
            k = repeat_kv(k, self.config.num_key_value_groups)
            v = repeat_kv(v, self.config.num_key_value_groups)
        return k, v
