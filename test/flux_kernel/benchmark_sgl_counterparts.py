#!/usr/bin/env python3
"""Compare Flux Kernel counterparts with SGL Kernel at audited shapes."""

from __future__ import annotations

import json
import statistics

import torch

import flux_kernel
import sgl_kernel


def latency_us(op, warmup: int = 100, rounds: int = 10, iterations: int = 1000):
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            op()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    ordered = sorted(samples)
    return {
        "p10": ordered[max(0, int(0.1 * len(ordered)) - 1)],
        "median": statistics.median(samples),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
    }


def main() -> None:
    x = torch.randn(23, 2560, device="cuda", dtype=torch.bfloat16)
    silu_out = torch.empty(23, 1280, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(23, 256, device="cuda", dtype=torch.float32)
    bias = torch.randn(256, device="cuda", dtype=torch.float32)
    topk_ids = torch.randint(0, 64, (23, 8), device="cuda", dtype=torch.int64)
    max_padded = topk_ids.numel() + 65 * 15
    align_args = (
        topk_ids,
        65,
        16,
        torch.empty(max_padded, device="cuda", dtype=torch.int32),
        torch.empty((max_padded + 15) // 16, device="cuda", dtype=torch.int32),
        torch.empty(1, device="cuda", dtype=torch.int32),
        torch.empty(66, device="cuda", dtype=torch.int32),
        True,
    )
    positions = torch.arange(23, device="cuda", dtype=torch.int64)
    query = torch.randn(23, 512, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(23, 128, device="cuda", dtype=torch.bfloat16)
    frequencies = torch.randn(32768, 32, device="cuda", dtype=torch.float32)
    cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)

    cases = {
        "silu_and_mul": (
            lambda: flux_kernel.silu_and_mul(x, silu_out),
            lambda: sgl_kernel.silu_and_mul(x, silu_out),
        ),
        "moe_fused_gate": (
            lambda: flux_kernel.moe_fused_gate(logits, bias, 8, 4, 8, 0, 2.5, False),
            lambda: sgl_kernel.moe_fused_gate(logits, bias, 8, 4, 8, 0, 2.5, False),
        ),
        "moe_align_block_size": (
            lambda: flux_kernel.moe_align_block_size(*align_args),
            lambda: sgl_kernel.moe_align_block_size(*align_args),
        ),
        "rope": (
            lambda: flux_kernel.apply_rope_with_cos_sin_cache_inplace(
                positions, query, key, 128, cache, True
            ),
            lambda: sgl_kernel.apply_rope_with_cos_sin_cache_inplace(
                positions, query, key, 128, cache, True
            ),
        ),
    }
    report = {}
    for name, (flux_op, sgl_op) in cases.items():
        flux_latency = latency_us(flux_op)
        sgl_latency = latency_us(sgl_op)
        report[name] = {
            "flux_kernel_us": flux_latency,
            "sgl_kernel_us": sgl_latency,
            "median_ratio": flux_latency["median"] / sgl_latency["median"],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
