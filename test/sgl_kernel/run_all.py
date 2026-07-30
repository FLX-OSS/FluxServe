#!/usr/bin/env python3
"""Run all request-critical sgl-kernel microbenchmarks."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(".sgl-kernel-benchmarks"))
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    root = Path(__file__).parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for script in ("benchmark_silu_and_mul.py", "benchmark_moe_fused_gate.py", "benchmark_moe_align_block_size.py", "benchmark_rope.py"):
        subprocess.run([
            sys.executable, str(root / script), "--warmup", str(args.warmup),
            "--iterations", str(args.iterations), "--rounds", str(args.rounds),
            "--json-output", str(args.output_dir / script.replace(".py", ".json")),
        ], check=True)


if __name__ == "__main__":
    main()
