#!/usr/bin/env python3
"""
    Benchmark FlashAttention-4 variable-length prefill against FlashInfer.
"""

# import cutlass.cute as cute

# # Compatibility with older FA4 / Quack - Monkey Patch
# for name in ["ThrMma", "ThrCopy"]:
#     if not hasattr(cute.core, name) and hasattr(cute, name):
#         setattr(cute.core, name, getattr(cute, name))

# if not hasattr(cute, "make_fragment") and hasattr(cute, "make_rmem_tensor"):
#     cute.make_fragment = cute.make_rmem_tensor


import argparse
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Case:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int


def parse_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers") from exc
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be positive")
    return lengths


def make_indptr(lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    cumulative = torch.cumsum(torch.tensor(lengths), dim=0).tolist()
    return torch.tensor([0, *cumulative], dtype=torch.int32, device=device)


def load_fa4() -> Callable:
    try:
        from flash_attn.cute import flash_attn_varlen_func
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "flash-attn-4 import failed; check CUTLASS/Quack versions "
            "(missing ThrMma is a known mismatch)"
        ) from exc
    return flash_attn_varlen_func


def make_case(args: argparse.Namespace) -> Case:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    total_tokens = sum(args.seq_lens)
    q = torch.randn(total_tokens, args.q_heads, args.head_dim, device=device, dtype=args.dtype)
    k = torch.randn(total_tokens, args.kv_heads, args.head_dim, device=device, dtype=args.dtype)
    return Case(q, k, torch.randn_like(k), make_indptr(args.seq_lens, device), max(args.seq_lens))


def time_callable(fn: Callable, warmup: int, iterations: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    output = None
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return output, samples


def attention_output(result):
    """Extract the attention output when an implementation returns auxiliaries."""
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (tuple, list)):
        for value in result:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"attention implementation returned unsupported result: {type(result)!r}")


def build_flashinfer_runner(args: argparse.Namespace, case: Case):
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

    workspace = torch.empty(
        args.workspace_mb * 1024 * 1024,
        dtype=torch.uint8,
        device=args.device,
    )
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        case.cu_seqlens,
        case.cu_seqlens,
        num_qo_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        head_dim_qk=args.head_dim,
        head_dim_vo=args.head_dim,
        q_data_type=args.dtype,
        kv_data_type=args.dtype,
        causal=args.causal,
        sm_scale=args.scale,
    )

    def run():
        return wrapper.run(
            case.q, case.k, case.v
        )

    return run


def build_fa4_runner(args: argparse.Namespace, case: Case):
    flash_attn_varlen_func = load_fa4()

    def run():
        return flash_attn_varlen_func(
            case.q,
            case.k,
            case.v,
            cu_seqlens_q=case.cu_seqlens,
            cu_seqlens_k=case.cu_seqlens,
            max_seqlen_q=case.max_seqlen,
            max_seqlen_k=case.max_seqlen,
            causal=args.causal,
            softmax_scale=args.scale,
        )

    return run


def report_timing(name: str, samples: list[float], total_tokens: int) -> float:
    mean_ms = sum(samples) / len(samples)
    p50_ms = sorted(samples)[len(samples) // 2]
    tokens_per_second = total_tokens / (mean_ms / 1000.0)
    print(
        f"{name}: mean_ms={mean_ms:.3f} p50_ms={p50_ms:.3f} "
        f"tokens_per_s={tokens_per_second:.1f}"
    )
    return mean_ms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=parse_lengths, default=(128, 256, 512, 1024, 2048, 4096, 8192, 16384))
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace-mb", type=int, default=512)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    return parser


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")

    args.causal = True
    args.dtype = getattr(torch, args.dtype)
    case = make_case(args)
    fa4_result, fa4_samples = time_callable(
        build_fa4_runner(args, case), args.warmup, args.iters
    )
    flashinfer_result, flashinfer_samples = time_callable(
        build_flashinfer_runner(args, case), args.warmup, args.iters
    )
    fa4_output = attention_output(fa4_result)
    flashinfer_output = attention_output(flashinfer_result)

    if not args.skip_correctness:
        torch.testing.assert_close(
            fa4_output, flashinfer_output, rtol=args.rtol, atol=args.atol
        )
        print("correctness: PASS")

    total_tokens = sum(args.seq_lens)
    fa4_ms = report_timing("fa4", fa4_samples, total_tokens)
    flashinfer_ms = report_timing("flashinfer", flashinfer_samples, total_tokens)
    print(f"speedup_fa4_over_flashinfer={flashinfer_ms / fa4_ms:.3f}")
    print(
        f"seq_lens={args.seq_lens} q_heads={args.q_heads} kv_heads={args.kv_heads} "
        f"head_dim={args.head_dim} dtype={args.dtype} causal={args.causal}"
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
