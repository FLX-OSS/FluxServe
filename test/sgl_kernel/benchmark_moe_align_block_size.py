#!/usr/bin/env python3
"""Benchmark the EP-local MoE token alignment kernel."""

import argparse
import math

import torch
from sgl_kernel import moe_align_block_size

from benchmark_utils import add_common_args, benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--tokens", type=int, default=23)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--local-experts", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    topk_ids = torch.randint(0, args.local_experts, (args.tokens, args.topk), device=args.device, dtype=torch.int64)
    max_padded = topk_ids.numel() + (args.local_experts + 1) * (args.block_size - 1)
    sorted_ids = torch.empty(max_padded, device=args.device, dtype=torch.int32)
    expert_ids = torch.empty(math.ceil(max_padded / args.block_size), device=args.device, dtype=torch.int32)
    post_padded = torch.empty(1, device=args.device, dtype=torch.int32)
    cumsum = torch.empty(args.local_experts + 2, device=args.device, dtype=torch.int32)
    fused_padding = max_padded <= 4096

    def op():
        moe_align_block_size(topk_ids, args.local_experts + 1, args.block_size, sorted_ids, expert_ids, post_padded, cumsum, fused_padding)

    op()
    torch.cuda.synchronize(args.device)
    assert 0 <= post_padded.item() <= max_padded
    benchmark("sgl_kernel.moe_align_block_size", op, args, {
        "topk_ids_shape": list(topk_ids.shape), "local_experts": args.local_experts,
        "block_size": args.block_size, "max_padded_tokens": max_padded,
        "fused_sorted_ids_padding": fused_padding,
    })


if __name__ == "__main__":
    main()
