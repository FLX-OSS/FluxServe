#!/usr/bin/env python3
"""Benchmark the LLaDA2 grouped top-k routing kernel."""

import argparse

import torch
from sgl_kernel import moe_fused_gate

from benchmark_utils import add_common_args, benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--tokens", type=int, default=23)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--expert-groups", type=int, default=8)
    parser.add_argument("--topk-groups", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--routed-scaling-factor", type=float, default=2.5)
    args = parser.parse_args()
    logits = torch.randn(args.tokens, args.experts, device=args.device, dtype=torch.float32)
    bias = torch.randn(args.experts, device=args.device, dtype=torch.float32)

    def op():
        return moe_fused_gate(logits, bias, args.expert_groups, args.topk_groups, args.topk, 0, args.routed_scaling_factor, False)

    weights, ids = op()
    assert weights.shape == ids.shape == (args.tokens, args.topk)
    assert torch.isfinite(weights).all()
    benchmark("sgl_kernel.moe_fused_gate", op, args, {
        "logits_shape": list(logits.shape), "topk": args.topk,
        "expert_groups": args.expert_groups, "topk_groups": args.topk_groups,
        "routed_scaling_factor": args.routed_scaling_factor,
    })


if __name__ == "__main__":
    main()
