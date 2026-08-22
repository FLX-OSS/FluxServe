"""Compare eager and CUDA graph latency for LLaDA2-mini paged attention."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from test_flashinfer_paged_cuda_graph_correctness import (
    ATOL,
    CASES,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    PAGE_SIZE,
    RTOL,
    _block_causal_mask,
    _make_wrapper,
    _metadata,
    _plan,
)


def _barrier(distributed: bool) -> None:
    if distributed:
        torch.distributed.barrier()


def _measure(
    op, *, warmup: int, iterations: int, rounds: int, device, distributed: bool
):
    for _ in range(warmup):
        op()
    torch.cuda.synchronize(device)
    _barrier(distributed)

    event_samples_us = []
    host_samples_us = []
    for _ in range(rounds):
        _barrier(distributed)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        start.record()
        for _ in range(iterations):
            op()
        end.record()
        end.synchronize()
        host_samples_us.append(
            (time.perf_counter() - host_start) * 1_000_000 / iterations
        )
        event_samples_us.append(start.elapsed_time(end) * 1000 / iterations)
    return {
        "cuda_event_us": _stats(event_samples_us),
        "host_synchronized_us": _stats(host_samples_us),
    }


def _stats(samples):
    ordered = sorted(samples)
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": max(samples),
    }


def _run_case(flashinfer, case, args, device, rank, distributed):
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    local_q_heads = NUM_Q_HEADS // args.tp_size
    local_kv_heads = NUM_KV_HEADS // args.tp_size
    num_pages = (case.kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    page_ids = torch.arange(num_pages, dtype=torch.int32, device=device)
    metadata = _metadata(case, page_ids, device)
    mask = _block_causal_mask(case, device)
    q = torch.randn(
        case.q_len,
        local_q_heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.randn(
        num_pages,
        2,
        PAGE_SIZE,
        local_kv_heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    eager_output = torch.empty_like(q)
    graph_output = torch.empty_like(q)

    eager_wrapper = _make_wrapper(flashinfer, device)
    graph_wrapper = _make_wrapper(flashinfer, device)
    _plan(
        eager_wrapper,
        case,
        metadata,
        mask,
        num_q_heads=local_q_heads,
        num_kv_heads=local_kv_heads,
    )
    _plan(
        graph_wrapper,
        case,
        metadata,
        mask,
        num_q_heads=local_q_heads,
        num_kv_heads=local_kv_heads,
    )

    def eager_op():
        eager_wrapper.run(q, kv_cache, out=eager_output, enable_pdl=False)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        for _ in range(args.warmup):
            graph_wrapper.run(q, kv_cache, out=graph_output, enable_pdl=False)
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    torch.cuda.synchronize(device)
    _barrier(distributed)

    graph = torch.cuda.CUDAGraph()
    capture_start = time.perf_counter()
    with torch.cuda.graph(graph):
        graph_wrapper.run(q, kv_cache, out=graph_output, enable_pdl=False)
    torch.cuda.synchronize(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000

    eager_op()
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(graph_output, eager_output, rtol=RTOL, atol=ATOL)

    eager = _measure(
        eager_op,
        warmup=args.warmup,
        iterations=args.iterations,
        rounds=args.rounds,
        device=device,
        distributed=distributed,
    )
    captured = _measure(
        graph.replay,
        warmup=args.warmup,
        iterations=args.iterations,
        rounds=args.rounds,
        device=device,
        distributed=distributed,
    )
    event_speedup = (
        eager["cuda_event_us"]["median"]
        / captured["cuda_event_us"]["median"]
    )
    host_speedup = (
        eager["host_synchronized_us"]["median"]
        / captured["host_synchronized_us"]["median"]
    )
    saved_us = (
        eager["host_synchronized_us"]["median"]
        - captured["host_synchronized_us"]["median"]
    )
    return {
        "case": case.name,
        "rank": rank,
        "q_len": case.q_len,
        "kv_len": case.kv_len,
        "capture_ms": capture_ms,
        "eager": eager,
        "cuda_graph": captured,
        "speedup": {
            "cuda_event": event_speedup,
            "host_synchronized": host_speedup,
        },
        "capture_break_even_replays": (
            capture_ms * 1000 / saved_us if saved_us > 0 else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("all", *(case.name for case in CASES)), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tp-size", type=int, choices=(1, 4), default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-output")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.rounds < 1:
        parser.error("warmup, iterations, and rounds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import flashinfer

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if world_size != args.tp_size:
        parser.error(
            f"WORLD_SIZE={world_size} must equal --tp-size={args.tp_size}; "
            "use torchrun for TP > 1"
        )
    if NUM_Q_HEADS % args.tp_size or NUM_KV_HEADS % args.tp_size:
        parser.error("LLaDA2-mini attention heads must divide evenly across TP ranks")
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend="nccl", device_id=device)
        rank = torch.distributed.get_rank()
    else:
        rank = 0
        device = torch.device(args.device)
    torch.cuda.set_device(device)
    selected = CASES if args.case == "all" else tuple(
        case for case in CASES if case.name == args.case
    )
    local_results = [
        _run_case(flashinfer, case, args, device, rank, distributed)
        for case in selected
    ]
    gathered = [None] * world_size if rank == 0 else None
    if distributed:
        torch.distributed.gather_object(local_results, gathered, dst=0)
    else:
        gathered = [local_results]
    if rank != 0:
        torch.distributed.destroy_process_group()
        return

    per_rank_results = [entry for rank_results in gathered for entry in rank_results]
    aggregate_results = []
    for case in selected:
        ranks = [entry for entry in per_rank_results if entry["case"] == case.name]
        eager_us = max(entry["eager"]["host_synchronized_us"]["median"] for entry in ranks)
        graph_us = max(entry["cuda_graph"]["host_synchronized_us"]["median"] for entry in ranks)
        aggregate_results.append(
            {
                "case": case.name,
                "tp_critical_path_eager_us": eager_us,
                "tp_critical_path_cuda_graph_us": graph_us,
                "tp_critical_path_speedup": eager_us / graph_us,
            }
        )
    result = {
        "benchmark": "llada2_mini_flashinfer_paged_cuda_graph",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
        "dtype": "bfloat16",
        "tp_size": args.tp_size,
        "global_num_q_heads": NUM_Q_HEADS,
        "global_num_kv_heads": NUM_KV_HEADS,
        "local_num_q_heads": NUM_Q_HEADS // args.tp_size,
        "local_num_kv_heads": NUM_KV_HEADS // args.tp_size,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "aggregate_results": aggregate_results,
        "per_rank_results": per_rank_results,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
