"""Fair FA4 vs BatchPrefillBlockExtend benchmark for LLaDA2.x.

The timed region measures eager attention API calls or single-call graph replay,
including GPU work and host submission gaps. Eager includes output allocation;
graph capture/allocation is excluded. JIT compilation, FlashInfer
planning, metadata construction, input allocation, and KV-cache population are
outside the timed region. Both backends consume the same Q tensor and physical K/V cache,
compute the same block-causal attention pairs, and return the same token order.

The program always runs a cross-backend correctness gate before benchmarking or
profiling. It exits without entering the profiler range if that gate fails.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from fluxserve.backend.layers.attention.fa4 import load_fa4_varlen_func


BLOCK_LENGTH = 64
PAGE_SIZE = 64
Q_HEADS = 16
KV_HEADS = 4
HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5
PREFILL_LENS = (128, 256, 512, 1024)
DECODE_KV_LENS = (
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    2560,
    3072,
    3584,
    4096,
    5120,
    6144,
    7168,
    8192,
)


@dataclass
class Workload:
    phase: str
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    q_offsets: tuple[int, ...]
    request_page_table: torch.Tensor
    fa4_qo_indptr: torch.Tensor
    fa4_kv_lens: torch.Tensor
    fa4_page_table: torch.Tensor
    batch_qo_indptr: torch.Tensor
    batch_kv_indptr: torch.Tensor
    batch_kv_indices: torch.Tensor
    batch_last_page_len: torch.Tensor
    batch_q_offsets: torch.Tensor
    batch_kv_offsets: torch.Tensor

    @property
    def query_tokens(self) -> int:
        return sum(self.q_lens)


def _indptr(lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=device)


def _build_workload(
    phase: str, seed: int, dtype: torch.dtype = torch.bfloat16,
    batch_size: int | None = None, kv_length: int | None = None,
) -> Workload:
    device = torch.device("cuda")
    if (batch_size is None) != (kv_length is None):
        raise ValueError("batch-size and kv-length must be specified together")
    if batch_size is not None and (
        batch_size <= 0 or kv_length < BLOCK_LENGTH or kv_length % BLOCK_LENGTH
    ):
        raise ValueError("batch-size must be positive; kv-length must be a positive block multiple")
    lengths = (kv_length,) * batch_size if batch_size is not None else None
    if phase == "prefill":
        q_lens = lengths if lengths is not None else PREFILL_LENS
        kv_lens = q_lens
        q_offsets = (0,) * len(q_lens)
    else:
        kv_lens = lengths if lengths is not None else DECODE_KV_LENS
        q_lens = (BLOCK_LENGTH,) * len(kv_lens)
        q_offsets = tuple(length - BLOCK_LENGTH for length in kv_lens)

    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        sum(q_lens), Q_HEADS, HEAD_DIM,
        dtype=dtype, device=device, generator=generator,
    )
    page_counts = tuple(math.ceil(length / PAGE_SIZE) for length in kv_lens)
    total_pages = sum(page_counts)
    k_cache = torch.randn(
        total_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM,
        dtype=dtype, device=device, generator=generator,
    )
    v_cache = torch.randn(
        total_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM,
        dtype=dtype, device=device, generator=generator,
    )

    # Each request owns distinct pages, but their physical IDs are shuffled.
    permutation = torch.randperm(total_pages, device=device, generator=generator)
    max_pages = max(page_counts)
    request_page_table = torch.full(
        (len(q_lens), max_pages), -1, dtype=torch.int32, device=device
    )
    cursor = 0
    for row, count in enumerate(page_counts):
        request_page_table[row, :count] = permutation[cursor : cursor + count]
        cursor += count

    batch_kv_indices = torch.cat(
        [request_page_table[row, :count] for row, count in enumerate(page_counts)]
    ).contiguous()
    batch_last_page_len = torch.tensor(
        tuple((length - 1) % PAGE_SIZE + 1 for length in kv_lens),
        dtype=torch.int32,
        device=device,
    )

    fa4_task_rows: list[int] = []
    fa4_task_kv_lens: list[int] = []
    fa4_task_q_lens: list[int] = []
    for row, (q_len, q_offset) in enumerate(zip(q_lens, q_offsets, strict=True)):
        for local_start in range(0, q_len, BLOCK_LENGTH):
            fa4_task_rows.append(row)
            fa4_task_q_lens.append(BLOCK_LENGTH)
            fa4_task_kv_lens.append(q_offset + local_start + BLOCK_LENGTH)
    row_indices = torch.tensor(fa4_task_rows, dtype=torch.long, device=device)

    return Workload(
        phase=phase,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        q_lens=q_lens,
        kv_lens=kv_lens,
        q_offsets=q_offsets,
        request_page_table=request_page_table,
        fa4_qo_indptr=_indptr(tuple(fa4_task_q_lens), device),
        fa4_kv_lens=torch.tensor(
            fa4_task_kv_lens, dtype=torch.int32, device=device
        ),
        fa4_page_table=request_page_table.index_select(0, row_indices).contiguous(),
        batch_qo_indptr=_indptr(q_lens, device),
        batch_kv_indptr=_indptr(page_counts, device),
        batch_kv_indices=batch_kv_indices,
        batch_last_page_len=batch_last_page_len,
        batch_q_offsets=torch.tensor(q_offsets, dtype=torch.int32, device=device),
        batch_kv_offsets=torch.zeros(len(q_lens), dtype=torch.int32, device=device),
    )


def _make_fa4(workload: Workload) -> Callable[[], torch.Tensor]:
    kernel = load_fa4_varlen_func()

    def run() -> torch.Tensor:
        result = kernel(
            workload.q,
            workload.k_cache,
            workload.v_cache,
            cu_seqlens_q=workload.fa4_qo_indptr,
            cu_seqlens_k=None,
            seqused_k=workload.fa4_kv_lens,
            max_seqlen_q=BLOCK_LENGTH,
            max_seqlen_k=max(workload.kv_lens),
            page_table=workload.fa4_page_table,
            softmax_scale=SCALE,
            causal=False,
            return_lse=False,
        )
        return result[0] if isinstance(result, tuple) else result

    return run


def _make_batch_prefill(workload: Workload, workspace_mib: int) -> Callable[[], torch.Tensor]:
    import flashinfer

    workspace = torch.empty(
        workspace_mib * 1024**2, dtype=torch.uint8, device=workload.q.device
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        backend="auto",
        block_extend=True,
        block_size=BLOCK_LENGTH,
    )
    wrapper.plan(
        workload.batch_qo_indptr,
        workload.batch_kv_indptr,
        workload.batch_kv_indices,
        workload.batch_last_page_len,
        num_qo_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim_qk=HEAD_DIM,
        page_size=PAGE_SIZE,
        custom_mask=None,
        causal=False,
        q_data_type=workload.q.dtype,
        kv_data_type=workload.k_cache.dtype,
        sm_scale=SCALE,
        q_offsets=workload.batch_q_offsets,
        kv_offsets=workload.batch_kv_offsets,
    )

    def run() -> torch.Tensor:
        return wrapper.run(
            workload.q,
            (workload.k_cache, workload.v_cache),
            return_lse=False,
        )

    # Keep the wrapper and workspace alive in the closure.
    run.wrapper = wrapper  # type: ignore[attr-defined]
    run.workspace = workspace  # type: ignore[attr-defined]
    return run


def _cross_backend_gate(
    runners: dict[str, Callable[[], torch.Tensor]], atol: float, rtol: float,
    workload: Workload,
) -> dict[str, object]:
    fa4 = runners["fa4"]()
    batch = runners["batchprefill"]()
    torch.cuda.synchronize()
    error = (fa4.float() - batch.float()).abs()
    metrics = {
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
    }
    torch.testing.assert_close(fa4, batch, atol=atol, rtol=rtol)
    # Independent FP32 math SDPA checks every valid query, not just agreement
    # between two optimized backends. All reference work is outside timing.
    references = []
    cursor = 0
    with sdpa_kernel(SDPBackend.MATH):
        for row, (length, offset, kv_length) in enumerate(zip(
            workload.q_lens, workload.q_offsets, workload.kv_lens, strict=True
        )):
            pages = workload.request_page_table[row, :math.ceil(kv_length / PAGE_SIZE)].long()
            k = workload.k_cache[pages].flatten(0, 1)[:kv_length]
            v = workload.v_cache[pages].flatten(0, 1)[:kv_length]
            k = k.transpose(0, 1).repeat_interleave(Q_HEADS // KV_HEADS, dim=0).float()
            v = v.transpose(0, 1).repeat_interleave(Q_HEADS // KV_HEADS, dim=0).float()
            for start in range(0, length, BLOCK_LENGTH):
                end = start + BLOCK_LENGTH
                references.append(F.scaled_dot_product_attention(
                    workload.q[cursor + start:cursor + end].transpose(0, 1).float(),
                    k[:, :offset + end], v[:, :offset + end],
                    dropout_p=0.0, is_causal=False, scale=SCALE,
                ).transpose(0, 1))
            cursor += length
    reference = torch.cat(references)
    for name, output in (("fa4", fa4), ("batchprefill", batch)):
        diff = (output.float() - reference).abs()
        metrics[name + "_vs_fp32_sdpa"] = {
            "max_abs_error": float(diff.max()), "mean_abs_error": float(diff.mean()),
        }
        torch.testing.assert_close(output.float(), reference, atol=atol, rtol=rtol)
    metrics.update(atol=atol, rtol=rtol)
    return metrics


def _event_average_ms(fn: Callable[[], torch.Tensor], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def _summarize(samples: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "std_ms": statistics.pstdev(samples),
    }


def _profile(
    name: str,
    phase: str,
    runner: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
) -> None:
    for _ in range(warmup):
        runner()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for iteration in range(iterations):
        torch.cuda.nvtx.range_push(
            f"fluxserve::llada2::{phase}::{name}::iteration_{iteration}"
        )
        runner()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def _capture_runner(runner, workload):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            runner()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner()

    def replay():
        graph.replay()
        return output

    # Detect stale captured inputs, not just agreement on the capture input.
    saved_q = workload.q.clone()
    for factor in (0.75, -0.5):
        workload.q.copy_(saved_q * factor)
        expected = runner().clone()
        actual = replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    workload.q.copy_(saved_q)
    replay()
    torch.cuda.synchronize()
    return replay


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("this benchmark is calibrated for GH200/Hopper (SM90)")
    if args.warmup < 0 or args.warmup_seconds < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("invalid warmup/iterations/repeats")

    torch.manual_seed(args.seed)
    workload = _build_workload(
        args.phase, args.seed, getattr(torch, args.dtype), args.batch_size, args.kv_length,
    )
    runners = {
        "fa4": _make_fa4(workload),
        "batchprefill": _make_batch_prefill(workload, args.workspace_mib),
    }
    # This also forces both JIT paths to compile before any timed/profiled range.
    correctness = _cross_backend_gate(runners, args.atol, args.rtol, workload)
    eager_runners = runners
    if args.execution != "eager":
        graph_runners = {name: _capture_runner(fn, workload) for name, fn in runners.items()}
        _cross_backend_gate(graph_runners, args.atol, args.rtol, workload)
        runners = graph_runners if args.execution == "graph" else {
            **{name + "_eager": fn for name, fn in eager_runners.items()},
            **{name + "_graph": fn for name, fn in graph_runners.items()},
        }
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "phase": args.phase,
        "dtype": args.dtype,
        "host": socket.gethostname(),
        "torch_version": torch.__version__,
        "measurement": {"mode": args.execution, "warmup": args.warmup,
                        "warmup_seconds_per_runner": args.warmup_seconds,
                        "iterations": args.iterations, "repeats": args.repeats},
        "graph_parity": "bitwise, two changed Q inputs plus FP32 SDPA" if args.execution != "eager" else None,
        "geometry": {
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "block_length": BLOCK_LENGTH,
            "page_size": PAGE_SIZE,
        },
        "q_lens": workload.q_lens,
        "kv_lens": workload.kv_lens,
        "query_tokens": workload.query_tokens,
        "attention_pairs_per_head": sum(
            BLOCK_LENGTH * (offset + end)
            for length, offset in zip(workload.q_lens, workload.q_offsets, strict=True)
            for end in range(BLOCK_LENGTH, length + 1, BLOCK_LENGTH)
        ),
        "seed": args.seed,
        "correctness_gate": {"passed": True, **correctness},
        "fairness": {
            "same_q_storage": True,
            "same_physical_kv_cache_storage": True,
            "same_physical_page_ids": True,
            "same_scale_and_attention_pairs": True,
            "same_output_token_order": True,
            "metadata_planning_jit_input_allocation_and_kv_population_timed": False,
            "scope": "attention API (not full adapter); graph excludes capture and input updates",
            "graph_calls_per_replay": 1,
        },
    }

    if args.profile_only:
        _profile(
            args.profile_only + "_" + args.execution,
            args.phase,
            runners[args.profile_only],
            args.profile_warmup,
            args.profile_iterations,
        )
        result["profile"] = {
            "backend": args.profile_only,
            "warmup": args.profile_warmup,
            "iterations": args.profile_iterations,
            "range": "cudaProfilerApi plus per-iteration NVTX",
        }
        return result

    for _ in range(args.warmup):
        for runner in runners.values():
            runner()
    torch.cuda.synchronize()
    # Bounded synchronized warmup avoids queuing seconds of host submissions
    # that could represent minutes of GPU work for a large captured kernel.
    for runner in runners.values():
        deadline = time.perf_counter() + args.warmup_seconds
        while time.perf_counter() < deadline:
            runner()
            torch.cuda.synchronize()
    samples = {name: [] for name in runners}
    names = tuple(runners)
    # Rotate positions to balance all four backend/execution combinations.
    for repeat in range(args.repeats):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            samples[name].append(_event_average_ms(runners[name], args.iterations))
    timing = {name: _summarize(values) for name, values in samples.items()}
    result["timing"] = timing
    if args.execution == "both":
        result["comparison"] = {
            mode + "_fa4_speedup": timing["batchprefill_" + mode]["mean_ms"] / timing["fa4_" + mode]["mean_ms"]
            for mode in ("eager", "graph")
        }
        result["comparison"].update({
            backend + "_graph_speedup": timing[backend + "_eager"]["mean_ms"] / timing[backend + "_graph"]["mean_ms"]
            for backend in ("fa4", "batchprefill")
        })
        return result
    result["comparison"] = {
        "speedup_fa4_over_batchprefill": (
            timing["batchprefill"]["mean_ms"] / timing["fa4"]["mean_ms"]
        ),
        "latency_reduction_percent": 100.0 * (
            1.0 - timing["fa4"]["mean_ms"] / timing["batchprefill"]["mean_ms"]
        ),
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--execution", choices=("eager", "graph", "both"), default="eager")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, help="Uniform workload batch size; requires --kv-length")
    parser.add_argument("--kv-length", type=int, help="Uniform total KV length (block multiple)")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--profile-only", choices=("fa4", "batchprefill"))
    parser.add_argument("--profile-warmup", type=int, default=50)
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.profile_only and args.execution == "both":
        raise ValueError("profile one execution mode per process")
    result = run(args)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
