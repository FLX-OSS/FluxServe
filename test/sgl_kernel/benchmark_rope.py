#!/usr/bin/env python3
"""Benchmark LLaDA2 rotary embedding through sgl-kernel."""

import argparse

import torch
from sgl_kernel import apply_rope_with_cos_sin_cache_inplace

from benchmark_utils import add_common_args, benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--tokens", type=int, default=23)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--max-position", type=int, default=32768)
    args = parser.parse_args()
    positions = torch.arange(args.tokens, device=args.device, dtype=torch.int64)
    query = torch.randn(args.tokens, args.query_heads * args.head_size, device=args.device, dtype=torch.bfloat16)
    key = torch.randn(args.tokens, args.kv_heads * args.head_size, device=args.device, dtype=torch.bfloat16)
    # LLaDA2 rotates half of each 128-wide head; the cache packs cos and sin.
    frequencies = torch.randn(args.max_position, args.head_size // 4, device=args.device, dtype=torch.float32)
    cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)

    def op():
        apply_rope_with_cos_sin_cache_inplace(positions=positions, query=query, key=key, head_size=args.head_size, cos_sin_cache=cache, is_neox=True)

    op()
    assert torch.isfinite(query).all() and torch.isfinite(key).all()
    benchmark("sgl_kernel.apply_rope_with_cos_sin_cache_inplace", op, args, {
        "positions_shape": list(positions.shape), "query_shape": list(query.shape),
        "key_shape": list(key.shape), "cache_shape": list(cache.shape),
        "dtype": str(query.dtype), "is_neox": True,
    })


if __name__ == "__main__":
    main()
