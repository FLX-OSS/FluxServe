#!/usr/bin/env python3
"""Benchmark the production sgl_kernel.silu_and_mul call."""

import argparse

import torch
from sgl_kernel import silu_and_mul

from benchmark_utils import add_common_args, benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--tokens", type=int, default=23)
    parser.add_argument("--intermediate-size", type=int, default=1280)
    args = parser.parse_args()
    x = torch.randn(args.tokens, 2 * args.intermediate_size, device=args.device, dtype=torch.bfloat16)
    output = torch.empty(args.tokens, args.intermediate_size, device=args.device, dtype=torch.bfloat16)

    def op():
        silu_and_mul(x, output)

    op()
    expected = torch.nn.functional.silu(x[:, : args.intermediate_size].float()) * x[:, args.intermediate_size :].float()
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)
    benchmark("sgl_kernel.silu_and_mul", op, args, {"input_shape": list(x.shape), "dtype": str(x.dtype)})


if __name__ == "__main__":
    main()
