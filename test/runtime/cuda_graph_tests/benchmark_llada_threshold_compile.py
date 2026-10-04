"""Report eager versus compiled LLaDA threshold-tail latency on CUDA.

Run: python test/runtime/cuda_graph_tests/benchmark_llada_threshold_compile.py
This measures a synthetic decoder tail, not complete FlashInfer inference.
"""

import argparse
import json
import statistics
import time

import torch

from threshold_tail_probe import threshold_tail


def measure(op, iterations, rounds, device):
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
    return statistics.median(samples)


def capture(op, device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            op()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    start = time.perf_counter()
    with torch.cuda.graph(graph, stream=stream):
        output = op()
    torch.cuda.synchronize(device)
    return graph, output, (time.perf_counter() - start) * 1000


def run_case(batch_size, args, device):
    logits = torch.randn(
        batch_size, args.block_length, args.vocab_size,
        device=device, dtype=torch.float32,
    )
    # Mix high-confidence transfers with the below-threshold progress case.
    logits[:, ::4, 1] = 20
    block = torch.full(
        (batch_size, args.block_length), args.mask_id,
        device=device, dtype=torch.long,
    )
    block[:, 0] = 1

    def eager():
        return threshold_tail(logits, block, args.mask_id, args.threshold)

    compiled = torch.compile(threshold_tail, fullgraph=True, dynamic=False)
    expected = eager()
    start = time.perf_counter()
    actual = compiled(logits, block, args.mask_id, args.threshold)
    torch.cuda.synchronize(device)
    first_compiled_call_ms = (time.perf_counter() - start) * 1000
    for result, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, target, rtol=0, atol=0)

    def compiled_op():
        return compiled(logits, block, args.mask_id, args.threshold)

    for _ in range(args.warmup):
        eager()
        compiled_op()
    torch.cuda.synchronize(device)

    eager_graph, eager_output, eager_capture_ms = capture(eager, device)
    compiled_graph, compiled_output, compiled_capture_ms = capture(compiled_op, device)
    # Replay must read updated input buffers, not just reproduce capture values.
    logits[:, 1::4, 2] = 21
    block[:, 1] = 2
    expected = eager()
    eager_graph.replay()
    compiled_graph.replay()
    torch.cuda.synchronize(device)
    for output in (eager_output, compiled_output):
        for result, target in zip(output, expected, strict=True):
            torch.testing.assert_close(result, target, rtol=0, atol=0)
    operations = {
        "eager": eager,
        "compiled": compiled_op,
        "eager_graph": eager_graph.replay,
        "compiled_graph": compiled_graph.replay,
    }
    timings = {
        name: measure(op, args.iterations, args.rounds, device)
        for name, op in operations.items()
    }
    return {
        "batch_size": batch_size,
        "block_length": args.block_length,
        "vocab_size": args.vocab_size,
        "first_compiled_call_ms": first_compiled_call_ms,
        "capture_ms": {"eager": eager_capture_ms, "compiled": compiled_capture_ms},
        "median_cuda_us": timings,
        "direct_speedup": timings["eager"] / timings["compiled"],
        "graph_speedup": timings["eager_graph"] / timings["compiled_graph"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=160000)
    parser.add_argument("--mask-id", type=int, default=156895)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json-output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if any(size < 1 for size in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if min(args.block_length, args.vocab_size, args.warmup, args.iterations, args.rounds) < 1:
        parser.error("shape and timing options must be positive")
    if not 0 <= args.mask_id < args.vocab_size:
        parser.error("mask-id must be in the vocabulary")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(7)
    with torch.inference_mode():
        results = [run_case(size, args, device) for size in args.batch_sizes]
    report = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "workload": "synthetic LLaDA threshold tail; no model or attention",
        "results": results,
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            handle.write(output + "\n")


if __name__ == "__main__":
    main()
