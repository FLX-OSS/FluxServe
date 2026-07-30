"""Shared utilities for standalone sgl-kernel CUDA microbenchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-output", type=Path)


def benchmark(name: str, op, args: argparse.Namespace, metadata: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    for _ in range(args.warmup):
        op()
    torch.cuda.synchronize(device)

    samples_us = []
    for _ in range(args.rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            op()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000 / args.iterations)

    ordered = sorted(samples_us)
    result = {
        "kernel": name,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "warmup": args.warmup,
        "iterations_per_round": args.iterations,
        "rounds": args.rounds,
        "latency_us": {
            "min": min(samples_us),
            "median": statistics.median(samples_us),
            "mean": statistics.mean(samples_us),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "max": max(samples_us),
        },
        "configuration": metadata,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return result
