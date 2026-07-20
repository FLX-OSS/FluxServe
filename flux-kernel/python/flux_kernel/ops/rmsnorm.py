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

"""RMSNorm triton kernels"""

from __future__ import annotations

import os

import torch

_triton_cache_dir = os.environ.get("TRITON_CACHE_DIR", "")
if not _triton_cache_dir or _triton_cache_dir == "/work" or _triton_cache_dir.startswith(
    "/work/"
):
    os.environ["TRITON_CACHE_DIR"] = "/tmp/fluxserve-triton-cache"

import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        x += residual
        tl.store(residual_out_ptr + row_offsets, x, mask=mask)

    variance = tl.sum(x * x, axis=0) / n_cols
    x *= tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row_offsets, x * weight, mask=mask)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm using the copied TokenSpeed Triton implementation."""
    if x.shape[0] == 0:
        if residual is None:
            return x if out is None else out
        return (x if out is None else out), residual
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size {x.shape[-1]}"
        )
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match input shape {tuple(x.shape)}"
        )

    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None and not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    hidden_size = x.shape[-1]
    x_2d = x.view(-1, hidden_size)
    out = torch.empty_like(x) if out is None else out
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_2d = out.view(-1, hidden_size)

    residual_out = torch.empty_like(x) if residual is not None else None
    block = triton.next_power_of_2(hidden_size)
    _rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d,
        residual,
        weight,
        out_2d,
        residual_out,
        hidden_size,
        eps,
        BLOCK=block,
        HAS_RESIDUAL=residual is not None,
    )
    if residual is None:
        return out
    return out, residual_out


@triton.jit
def _qk_rmsnorm_kernel(
    q_in_ptr,
    k_in_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    q_in_token_stride,
    k_in_token_stride,
    q_out_token_stride,
    k_out_token_stride,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    offsets = tl.arange(0, BLOCK)
    mask = offsets < head_dim

    if is_k:
        in_addrs = (
            k_in_ptr + token * k_in_token_stride + local_head * head_dim + offsets
        )
        out_addrs = (
            k_out_ptr + token * k_out_token_stride + local_head * head_dim + offsets
        )
        weight_addrs = k_weight_ptr + offsets
    else:
        in_addrs = (
            q_in_ptr + token * q_in_token_stride + local_head * head_dim + offsets
        )
        out_addrs = (
            q_out_ptr + token * q_out_token_stride + local_head * head_dim + offsets
        )
        weight_addrs = q_weight_ptr + offsets

    x = tl.load(in_addrs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / head_dim
    x *= tl.rsqrt(variance + eps)
    weight = tl.load(weight_addrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_addrs, x * weight, mask=mask)


def _flattenable_token_stride(x: torch.Tensor) -> int | None:
    if x.ndim == 1:
        return x.shape[-1]

    token_stride = x.stride(-2)
    expected_stride = token_stride
    for dim in range(x.ndim - 2, -1, -1):
        if x.stride(dim) != expected_stride:
            return None
        expected_stride *= x.shape[dim]
    return token_stride


def _prepare_qk_input(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if x.ndim == 0:
        raise ValueError("q and k must have at least one dimension")
    if x.shape[-1] <= 0:
        raise ValueError("q and k must have a non-empty last dimension")
    if x.stride(-1) != 1:
        x = x.reshape(-1, x.shape[-1]).contiguous()
        return x, x.shape[0], x.stride(0)

    num_tokens = x.numel() // x.shape[-1]
    token_stride = _flattenable_token_stride(x)
    if token_stride is None:
        x = x.reshape(num_tokens, x.shape[-1]).contiguous()
        token_stride = x.stride(0)
    return x, num_tokens, token_stride


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm of Q and K in one Triton kernel launch.

    Q and K may be regular strided views from a packed QKV projection. Outputs
    are fresh contiguous 2-D tensors shaped ``[num_tokens, packed_head_dim]``.
    """
    if not q.is_cuda or not k.is_cuda:
        raise ValueError("qk_rmsnorm requires CUDA q and k tensors")
    if q.device != k.device or q_weight.device != q.device or k_weight.device != q.device:
        raise ValueError("q, k, q_weight, and k_weight must be on the same device")
    if q.dtype != k.dtype:
        raise ValueError(f"q/k dtype mismatch: q={q.dtype}, k={k.dtype}")
    if q_weight.ndim != 1 or k_weight.ndim != 1:
        raise ValueError("q_weight and k_weight must be 1-D tensors")
    if q_weight.shape[0] != k_weight.shape[0]:
        raise ValueError(
            "q_weight and k_weight must have the same head dimension, "
            f"got {q_weight.shape[0]} and {k_weight.shape[0]}"
        )

    head_dim = q_weight.shape[0]
    if q.shape[-1] % head_dim != 0:
        raise ValueError(
            f"q last dimension {q.shape[-1]} must be divisible by head_dim {head_dim}"
        )
    if k.shape[-1] % head_dim != 0:
        raise ValueError(
            f"k last dimension {k.shape[-1]} must be divisible by head_dim {head_dim}"
        )

    q, q_num_tokens, q_token_stride = _prepare_qk_input(q)
    k, k_num_tokens, k_token_stride = _prepare_qk_input(k)
    if q_num_tokens != k_num_tokens:
        raise ValueError(
            f"q and k must have the same number of tokens, "
            f"got {q_num_tokens} and {k_num_tokens}"
        )

    q_out = torch.empty(
        (q_num_tokens, q.shape[-1]),
        dtype=q.dtype,
        device=q.device,
    )
    k_out = torch.empty(
        (k_num_tokens, k.shape[-1]),
        dtype=k.dtype,
        device=k.device,
    )
    if q_num_tokens == 0:
        return q_out, k_out

    if not q_weight.is_contiguous():
        q_weight = q_weight.contiguous()
    if not k_weight.is_contiguous():
        k_weight = k_weight.contiguous()

    num_q_heads = q.shape[-1] // head_dim
    num_kv_heads = k.shape[-1] // head_dim
    block = triton.next_power_of_2(head_dim)
    _qk_rmsnorm_kernel[(q_num_tokens, num_q_heads + num_kv_heads)](
        q,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        q_token_stride,
        k_token_stride,
        q_out.stride(0),
        k_out.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        eps,
        BLOCK=block,
    )
    return q_out, k_out


__all__ = ["rmsnorm", "qk_rmsnorm"]
