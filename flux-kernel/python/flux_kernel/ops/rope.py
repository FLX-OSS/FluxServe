# Copyright (c) 2026 FLUX-OSS

"""Rotary positional embedding with an SGL-compatible interface."""

from __future__ import annotations

import torch

from flux_kernel.cuda.rope import load_rope


def _rotate_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    left, right = x[..., :half], x[..., half:]
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


def _rotate_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return rotated.flatten(-2)


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    fused_set_kv_buffer_arg=None,
    output_q_rope: torch.Tensor | None = None,
    output_k_rope: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> None:
    """Apply cached RoPE to Q and K, in-place unless output tensors are given."""
    del enable_pdl
    if fused_set_kv_buffer_arg is not None:
        raise NotImplementedError("fused KV-cache scatter is not supported")
    if head_size not in (64, 128, 256, 512):
        raise ValueError("head_size must be one of 64, 128, 256, or 512")
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache must have dtype float32")
    if positions.ndim != 1 or positions.shape[0] != query.shape[0]:
        raise ValueError("positions must be 1-D with one entry per token")
    if key.shape[0] != query.shape[0]:
        raise ValueError("query and key must have the same token dimension")
    if query.shape[-1] % head_size or key.shape[-1] % head_size:
        raise ValueError("query and key widths must be divisible by head_size")
    if query.device != key.device or cos_sin_cache.device != query.device:
        raise ValueError("query, key, and cos_sin_cache must be on the same device")

    rotary_dim = cos_sin_cache.shape[-1]
    if rotary_dim <= 0 or rotary_dim > head_size or rotary_dim % 2:
        raise ValueError("cache width must be an even rotary dimension <= head_size")

    query_out = query if output_q_rope is None else output_q_rope
    key_out = key if output_k_rope is None else output_k_rope
    for source, target in ((query, query_out), (key, key_out)):
        if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
            raise ValueError("RoPE output must match its input's shape, dtype, and device")
    load_rope()
    torch.ops.flux_kernel.apply_rope(
        positions.to(torch.int64),
        query,
        key,
        query_out,
        key_out,
        head_size,
        cos_sin_cache,
        is_neox,
    )


__all__ = ["apply_rope_with_cos_sin_cache_inplace"]
