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
    MoE routing and dispatch helpers with SGL-compatible APIs.
"""

from __future__ import annotations

import torch

from flux_kernel.cuda.moe import load_moe


def moe_fused_gate(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select grouped, biased experts using the TokenSpeed reference semantics."""
    if logits.ndim != 2 or correction_bias.ndim != 1:
        raise ValueError("logits must be 2-D and correction_bias must be 1-D")
    num_tokens, num_experts = logits.shape
    if correction_bias.shape[0] != num_experts:
        raise ValueError("correction_bias length must match the number of experts")
    if num_experts % num_expert_group:
        raise ValueError("num_experts must be divisible by num_expert_group")
    if not 0 < topk_group <= num_expert_group:
        raise ValueError("topk_group must be in [1, num_expert_group]")
    if not 0 < topk <= topk_group * (num_experts // num_expert_group):
        raise ValueError("topk exceeds the experts available in selected groups")
    if num_fused_shared_experts != 0:
        raise NotImplementedError("fused shared experts are not supported")

    load_moe()
    weights, ids = torch.ops.flux_kernel.moe_fused_gate(
        logits,
        correction_bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
    return weights, ids


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool,
) -> None:
    """Group flattened token slots by expert and pad groups to ``block_size``."""
    if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be a 2-D int32 or int64 tensor")
    actual_num_experts = num_experts - 1
    if actual_num_experts < 0 or block_size <= 0:
        raise ValueError("num_experts and block_size must be positive")
    if cumsum_buffer.numel() < num_experts + 1:
        raise ValueError("cumsum_buffer is too small")

    pad_id = topk_ids.numel()
    max_padded = pad_id + num_experts * (block_size - 1)
    if sorted_ids.numel() < max_padded:
        raise ValueError("sorted_ids is too small")
    max_blocks = (max_padded + block_size - 1) // block_size
    if expert_ids.numel() < max_blocks or num_tokens_post_pad.numel() != 1:
        raise ValueError("expert_ids or num_tokens_post_pad has an invalid size")
    if not pad_sorted_token_ids:
        sorted_ids.fill_(pad_id)
    load_moe()
    torch.ops.flux_kernel.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )


__all__ = ["moe_align_block_size", "moe_fused_gate"]
