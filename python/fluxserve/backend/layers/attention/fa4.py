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

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Optional

import torch

from fluxserve.backend.execution.forward_batch_info import ForwardBatch
from fluxserve.backend.layers.attention.base import AttentionForwardConfig
from fluxserve.backend.layers.attention.metadata import PagedAttentionMetadata


@lru_cache(maxsize=1)
def load_fa4_varlen_func() -> Callable:
    """Load the standalone upstream FA4 kernel, never a serving framework shim."""

    try:
        from flash_attn.cute import flash_attn_varlen_func
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "attention_backend='fa4' requires the standalone flash-attn-4 "
            "package. Install the pinned FluxServe environment."
        ) from exc
    return flash_attn_varlen_func


def fa4_package_version() -> str | None:
    try:
        return version("flash-attn-4")
    except PackageNotFoundError:
        return None


def validate_fa4_runtime(device: str | torch.device) -> None:
    load_fa4_varlen_func()
    device_text = str(device)
    resolved = torch.device(
        f"cuda:{device_text}" if device_text.isdigit() else device_text
    )
    if resolved.type != "cuda":
        raise RuntimeError("FA4 requires an NVIDIA CUDA device")
    device_index = resolved.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    if major not in (9, 10, 11):
        raise RuntimeError(
            "FA4 paged KV requires a supported Hopper or Blackwell GPU; got "
            f"compute capability {major}.{minor}"
        )


class FA4PagedAttention:
    """Standalone FlashAttention-4 adapter for LLaDA block attention."""

    def __init__(
        self,
        config: AttentionForwardConfig,
        kernel: Optional[Callable] = None,
    ):
        self.config = config
        self._kernel = kernel

    def can_run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_values,
        attention_mask: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch],
    ) -> bool:
        metadata = (
            getattr(forward_batch, "paged_attention_metadata", None)
            if forward_batch is not None
            else None
        )
        return bool(
            isinstance(metadata, PagedAttentionMetadata)
            and attention_mask is None
            and isinstance(past_key_values, (tuple, list))
            and len(past_key_values) == 2
            and q.dtype in (torch.float16, torch.bfloat16)
            and q.dtype == k.dtype == v.dtype
            and q.shape[-1] == k.shape[-1] == v.shape[-1]
            and q.shape[-1] % 8 == 0
            and (self._kernel is not None or q.is_cuda)
        )

    @torch.compiler.disable(recursive=False)
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_values,
        metadata: PagedAttentionMetadata,
    ) -> torch.Tensor:
        self._validate_shapes(q, k, v, past_key_values, metadata)
        k_cache, v_cache = past_key_values
        q_tokens = (
            q.transpose(1, 2)
            .contiguous()
            .view(
                metadata.batch_size * metadata.max_input_len,
                self.config.num_heads,
                self.config.head_dim,
            )
        )
        k_tokens = (
            k.transpose(1, 2)
            .contiguous()
            .view(
                metadata.batch_size * metadata.max_input_len,
                self.config.num_kv_heads,
                self.config.head_dim,
            )
        )
        v_tokens = v.transpose(1, 2).contiguous().view_as(k_tokens)

        if metadata.is_identity_mapping:
            packed_q, packed_k, packed_v = q_tokens, k_tokens, v_tokens
        else:
            packed_q = q_tokens.index_select(0, metadata.q_token_indices)
            packed_k = k_tokens.index_select(0, metadata.q_token_indices)
            packed_v = v_tokens.index_select(0, metadata.q_token_indices)
        # Contiguous NHD pages can be addressed directly with flat token slots.
        # Avoid two per-layer division/remainder kernels and 2-D scatter indices.
        k_cache.view(-1, self.config.num_kv_heads, self.config.head_dim).index_copy_(
            0, metadata.slot_mapping, packed_k
        )
        v_cache.view(-1, self.config.num_kv_heads, self.config.head_dim).index_copy_(
            0, metadata.slot_mapping, packed_v
        )

        kernel = self._kernel or load_fa4_varlen_func()
        result = kernel(
            packed_q,
            k_cache,
            v_cache,
            cu_seqlens_q=metadata.qo_indptr,
            cu_seqlens_k=None,
            seqused_k=metadata.kv_lens,
            max_seqlen_q=metadata.max_q_len,
            max_seqlen_k=metadata.max_kv_len,
            page_table=metadata.page_table,
            softmax_scale=self.config.scale,
            causal=False,
            return_lse=False,
        )
        output = result[0] if isinstance(result, tuple) else result
        if output.shape != packed_q.shape:
            raise RuntimeError(
                "FA4 returned an unexpected output shape: "
                f"got {tuple(output.shape)}, expected {tuple(packed_q.shape)}"
            )

        if metadata.is_identity_mapping:
            padded = output
        else:
            padded = q.new_zeros(
                metadata.batch_size * metadata.max_input_len,
                self.config.num_heads,
                self.config.head_dim,
            )
            padded.index_copy_(0, metadata.q_token_indices, output)
        return (
            padded.view(
                metadata.batch_size,
                metadata.max_input_len,
                self.config.num_heads,
                self.config.head_dim,
            )
            .transpose(1, 2)
        )

    def _validate_shapes(self, q, k, v, past_key_values, metadata) -> None:
        expected_q = (
            metadata.batch_size,
            self.config.num_heads,
            metadata.max_input_len,
            self.config.head_dim,
        )
        expected_kv = (
            metadata.batch_size,
            self.config.num_kv_heads,
            metadata.max_input_len,
            self.config.head_dim,
        )
        if tuple(q.shape) != expected_q:
            raise RuntimeError(f"FA4 Q shape {tuple(q.shape)} != {expected_q}")
        if tuple(k.shape) != expected_kv or tuple(v.shape) != expected_kv:
            raise RuntimeError(
                f"FA4 K/V shapes must both be {expected_kv}, got "
                f"{tuple(k.shape)} and {tuple(v.shape)}"
            )
        k_cache, v_cache = past_key_values
        expected_cache_tail = (
            metadata.page_size,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        if tuple(k_cache.shape[1:]) != expected_cache_tail:
            raise RuntimeError(
                f"FA4 K cache tail {tuple(k_cache.shape[1:])} != {expected_cache_tail}"
            )
        if v_cache.shape != k_cache.shape:
            raise RuntimeError("FA4 K and V caches must have identical shapes")
        if any(t.device != q.device for t in (k, v, k_cache, v_cache)):
            raise RuntimeError("FA4 Q/K/V and paged cache must be on the same device")
        if any(t.dtype != q.dtype for t in (k, v, k_cache, v_cache)):
            raise RuntimeError("FA4 Q/K/V and paged cache must use the same dtype")
        if self.config.num_heads % self.config.num_kv_heads != 0:
            raise RuntimeError("FA4 requires num_heads to be divisible by num_kv_heads")
        if q.is_cuda:
            major, _ = torch.cuda.get_device_capability(q.device)
            max_head_dim = 256 if major == 9 else 128
            if not 8 <= self.config.head_dim <= max_head_dim:
                raise RuntimeError(
                    f"FA4 paged varlen head_dim must be in [8, {max_head_dim}] "
                    f"on compute capability {major}.x, got {self.config.head_dim}"
                )
        if not all(tensor.is_contiguous() for tensor in (k_cache, v_cache)):
            raise RuntimeError("FA4 paged K/V caches must be contiguous")
