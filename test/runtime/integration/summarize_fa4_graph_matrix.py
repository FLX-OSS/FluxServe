"""Publish only a complete four-backend, three-round GSM8K matrix."""
import argparse
import json
from pathlib import Path
from statistics import median


REPRODUCTION = r"""
### Reproduce the full GSM8K concurrency-16 runs

Run inside a single-GH200 allocation. Omitting `--limit` runs all 1,319
requests; each output directory must not already exist.

```bash
cd /u/dzhu8/workspace/FluxServe
OUTPUT_ROOT=test/runtime/integration/profiles/llada21_c16_reproduce
mkdir -p "$OUTPUT_ROOT"

COMMON_ARGS=(
  --dataset data/gsm8k.jsonl --concurrency 16 --repeats 3
  --max-num-seqs 16 --mini-batch-size 4 --max-model-len 65536
  --max-scheduled-tokens 2048 --scheduler-num-device-pages 0
  --gpu-memory-utilization 0.8 --capture-bs 1 2 4 8 10 12 16
)

bash test/runtime/integration/run_fa4_graph_benchmark.sh \
  python test/runtime/integration/bench_llada21_quality.py \
  --backend fa4 "${COMMON_ARGS[@]}" --output "$OUTPUT_ROOT/fa4_eager_c16"
bash test/runtime/integration/run_fa4_graph_benchmark.sh \
  python test/runtime/integration/bench_llada21_quality.py \
  --backend fa4 --graph "${COMMON_ARGS[@]}" --output "$OUTPUT_ROOT/fa4_graph_c16"
bash test/runtime/integration/run_fa4_graph_benchmark.sh \
  python test/runtime/integration/bench_llada21_quality.py \
  --backend flashinfer "${COMMON_ARGS[@]}" --output "$OUTPUT_ROOT/batchprefill_eager_c16"
bash test/runtime/integration/run_fa4_graph_benchmark.sh \
  python test/runtime/integration/bench_llada21_quality.py \
  --backend flashinfer --graph "${COMMON_ARGS[@]}" --output "$OUTPUT_ROOT/batchprefill_graph_c16"
```

For accuracy, set `GSM8K_TEST` to the official 1,319-row GSM8K test JSONL:

```bash
GSM8K_TEST=/path/to/official/gsm8k_test.jsonl
for variant in fa4_eager_c16 fa4_graph_c16 batchprefill_eager_c16 batchprefill_graph_c16; do
  for round in 0 1 2; do
    python test/runtime/integration/score_llada21_gsm8k.py \
      --tests "$GSM8K_TEST" \
      --responses "$OUTPUT_ROOT/$variant/responses_${round}.jsonl" \
      --expected-count 1319
  done
done
```

The complete concurrency matrix is available through
`run_llada21_graph_matrix.sh`; accepted variants are `fa4_eager`, `fa4_graph`,
`batchprefill_eager`, and `batchprefill_graph`.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = ("fa4_eager", "fa4_graph", "batchprefill_eager", "batchprefill_graph")
    records = {}
    for variant in variants:
        for concurrency in (1, 4, 8, 16):
            directory = args.root / f"{variant}_c{concurrency}"
            manifest = json.loads((directory / "manifest.json").read_text())
            rounds = []
            for repeat in range(3):
                summary = json.loads((directory / f"summary_{repeat}.json").read_text())
                scores = json.loads((directory / f"responses_{repeat}.scores.json").read_text())
                expected = 1319 if concurrency == 16 else 128
                assert summary["requests"] == scores["total"] == expected
                assert summary["timing_valid_for_comparison"]
                assert summary["metrics_after"]["failed_requests"] == 0
                if variant.endswith("_graph"):
                    for name in ("decode_replay_count", "decode_fallback_count"):
                        key = "cuda_graph_" + name
                        delta = summary["metrics_after"].get(key, 0) - summary["metrics_before"].get(key, 0)
                        assert delta > 0 if name == "decode_replay_count" else delta == 0
                rounds.append((summary, scores))
            records[variant, concurrency] = (manifest, rounds)
    lines = ["# FA4 full-model decode CUDA graph: LLaDA2.1-mini / GSM8K", "",
        "## Configuration and measurement", "",
        "Cached `inclusionAI/LLaDA2.1-mini` revision "
        "`20e64e2ad21644d0e5248586ed9c942cdd45de0f`. "
        f"Packages: `{records['fa4_graph', 16][0]['packages']}`.", "",
        "Single GH200 per configuration, TP/DP/EP=1, BF16, native FluxServe paged scheduling. "
        "Configuration follows `test/benchmark/fluxserve/configs/tp1_ep1_llada21_mini.sh`: "
        "max_model_len=65536, max_num_seqs=16, max_scheduled_tokens=2048, "
        "gpu_memory_utilization=0.8, block/page=64, joint_threshold Quality 0.7/0.5, "
        "max_post_steps=16. The unchanged CLI mini_batch_size default is 4. "
        "All requests use the checkpoint chat template, temperature=0 and max_tokens=2048. "
        "64K is the configured capacity, not every request's KV length.", "",
        "FA4 and BatchPrefill each run eager and padded decode graph. Captured batch buckets "
        "are 1/2/4/8/10/12/16; prefill stays eager. FA4 captures all transformer layers, "
        "LM head, joint token selection and block-finished predicates. No vLLM/SGLang APIs are used.", "",
        "Closed-loop HTTP concurrency; stream=False. Latencies cover complete responses, "
        "not TTFT or inter-token latency. Startup, capture and warmup are excluded. "
        "Throughput uses actual server-counted output tokens. Every setting has three rounds. "
        "Concurrency 16 uses all 1319 questions; 1/4/8 use the same first 128 questions. "
        "Do not interpret the subset/full-set boundary as a pure scaling experiment.", "",
        "## Correctness", "",
        "Final GH200 validation: 54 tests passed, including attention versus SDPA, "
        "adapter replay, decoder state progression and real-weight graph/eager comparison. "
        "The model test covers 19 replays across 7 buckets, changing page IDs and KV lengths up to 65536, "
        "and every actual batch size from 1 through 16. Updated tokens and all three "
        "decoder flags, logits and the entire real KV pool must match exactly (atol=rtol=0). "
        "KV-pool replacement also verifies capture invalidation and refusal of stale replay. "
        "This verifies FA4 graph versus FA4 eager; it does not "
        "resolve the earlier FA4-versus-dense-SDPA whole-model parity failure.", "",
        "All graph rounds below require positive actual replay counts and zero decode fallback. "
        "GSM8K scoring uses the same last-signed-decimal extraction against verified official "
        "answers, including length-limited responses. Natural generation can differ across "
        "backends and padded batch shapes; forward counts and accuracy accompany throughput.", "",
        "## Full GSM8K, concurrency 16", "",
        "Values are medians across three rounds; throughput brackets give min–max.", "",
        "| Configuration | Output tokens/s [min–max] | Requests/s | Wall s | Latency mean / P50 / P95 / P99 s | Accuracy range | Forwards range |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in variants:
        _, rounds = records[variant, 16]
        values = lambda key: [s[key] for s, _ in rounds]
        rates = values("server_output_tokens_per_s")
        accuracies = [q["accuracy"] * 100 for _, q in rounds]
        latencies = " / ".join(f"{median(values(k)):.2f}" for k in
                    ("latency_mean_s", "latency_p50_s", "latency_p95_s", "latency_p99_s"))
        forwards = values("model_forwards")
        lines.append(f"| {variant} | {median(rates):.2f} [{min(rates):.2f}–{max(rates):.2f}] | "
                     f"{median(values('requests_per_s')):.3f} | {median(values('elapsed_s')):.2f} | "
                     f"{latencies} | {min(accuracies):.2f}–{max(accuracies):.2f}% | {min(forwards)}–{max(forwards)} |")
    lines += ["", "### FA4 eager / graph completion parity", "",
              "Exact response-text and finish-reason matches, paired by dataset index within each round. "
              "This is end-to-end output parity, not a claim of bitwise whole-model logits.", "",
              "| Concurrency | Exact matches by round | Output-token totals equal | Forward counts equal |",
              "|---:|---|---|---|"]
    for concurrency in (1, 4, 8, 16):
        matches = []
        for repeat in range(3):
            outputs = []
            for variant in ("fa4_eager", "fa4_graph"):
                path = args.root / f"{variant}_c{concurrency}" / f"responses_{repeat}.jsonl"
                rows = map(json.loads, path.read_text().splitlines())
                outputs.append({r["index"]: (r["response"]["choices"][0]["message"]["content"],
                                            r["response"]["choices"][0]["finish_reason"]) for r in rows})
            assert outputs[0].keys() == outputs[1].keys()
            matches.append(f"{sum(outputs[0][i] == outputs[1][i] for i in outputs[0])}/{len(outputs[0])}")
        eager, graph = records["fa4_eager", concurrency][1], records["fa4_graph", concurrency][1]
        tokens_equal = all(a[0]["server_output_tokens"] == b[0]["server_output_tokens"] for a, b in zip(eager, graph))
        forwards_equal = all(a[0]["model_forwards"] == b[0]["model_forwards"] for a, b in zip(eager, graph))
        lines.append(f"| {concurrency} | {', '.join(matches)} | {tokens_equal} | {forwards_equal} |")
    lines += ["", "## Same 128-question concurrency sweep", "",
              "Median output tokens/s [min–max]. Server capacity stays fixed at 16.", "",
              "| HTTP concurrency | FA4 eager | FA4 graph | BatchPrefill eager | BatchPrefill graph |",
              "|---:|---:|---:|---:|---:|"]
    for concurrency in (1, 4, 8):
        cells = []
        for variant in variants:
            rates = [s["server_output_tokens_per_s"] for s, _ in records[variant, concurrency][1]]
            cells.append(f"{median(rates):.2f} [{min(rates):.2f}–{max(rates):.2f}]")
        lines.append(f"| {concurrency} | " + " | ".join(cells) + " |")
    lines += ["", "## GPU memory and graph execution (full-set runs)", "",
              "Peak allocation includes startup and capture. GPU memory is PyTorch allocated memory, "
              "not total device consumption. Capture memory excludes the persistent model/KV pool.", "",
              "| Configuration | Node / job | Peak allocated GiB | Capture s | Decode replays / round | Length stops / round |",
              "|---|---|---:|---:|---|---|"]
    for variant in variants:
        manifest, rounds = records[variant, 16]
        peak = max(s["metrics_after"]["benchmark_gpu_peak_allocated_bytes"] for s, _ in rounds) / 2**30
        capture = rounds[0][0]["metrics_after"].get("cuda_graph_capture_time_s", 0)
        replays = [s["metrics_after"].get("cuda_graph_decode_replay_count", 0)
                   - s["metrics_before"].get("cuda_graph_decode_replay_count", 0) for s, _ in rounds]
        lines.append(f"| {variant} | {manifest['hostname']} / {manifest['slurm_job_id']} | {peak:.2f} | "
                     f"{capture:.2f} | {replays} | {[q['length_limited'] for _, q in rounds]} |")
    lines += ["", "## Same-GPU control", "",
        "All four variants below run sequentially on the same GH200, concurrency 16, "
        "the same first 128 questions, three rounds per variant. "
        "This checks whether the cross-card comparison changes direction. Order is fixed, "
        "not randomized. Other server and decoder settings match the main matrix.", "",
        "| Configuration | Output tokens/s median [min–max] | P95 latency s | Accuracy range | Node / job |",
        "|---|---:|---:|---:|---|"]
    hosts = set()
    for variant in variants:
        directory = args.root / f"same_gpu_{variant}"
        manifest = json.loads((directory / "manifest.json").read_text())
        hosts.add((manifest["hostname"], manifest["slurm_job_id"]))
        rows = [json.loads((directory / f"summary_{i}.json").read_text()) for i in range(3)]
        scores = [json.loads((directory / f"responses_{i}.scores.json").read_text()) for i in range(3)]
        assert all(s["requests"] == 128 and s["timing_valid_for_comparison"] for s in rows)
        assert all(s["total"] == 128 for s in scores)
        rates = [s["server_output_tokens_per_s"] for s in rows]
        accuracy = [s["accuracy"] * 100 for s in scores]
        lines.append(f"| {variant} | {median(rates):.2f} [{min(rates):.2f}–{max(rates):.2f}] | "
                     f"{median(s['latency_p95_s'] for s in rows):.2f} | {min(accuracy):.2f}–{max(accuracy):.2f}% | "
                     f"{manifest['hostname']} / {manifest['slurm_job_id']} |")
    assert len(hosts) == 1
    lines += ["", "## Interpretation and reproduction", "",
        "These are serving measurements with natural EOS, not equal-work attention microbenchmarks. "
        "Different completions and editing iterations affect throughput. Each variant uses a separate "
        "allocated GH200; fixed variant-to-card assignment leaves a possible card/host effect. "
        "FlashInfer eager dispatches backend=auto while its existing decode graph uses backend=fa2, "
        "so its eager/graph ratio includes kernel-dispatch changes. No matched-FA2 eager control, "
        "nsys or ncu run is claimed.", "",
        "The large raw manifests, responses and profiler artifacts are intentionally not committed. "
        "The reviewable aggregate measurements are preserved above, while the commands below regenerate "
        "manifests containing the exact server command, package versions, source and dataset hashes, and "
        "Slurm host/job metadata.", "",
        "During the long benchmark, the FA4 teardown hook was aligned with the executor's "
        "`shutdown_cuda_graphs` API and regression-tested. This changes teardown only, outside "
        "all timed regions. Regenerated manifests also hash untracked runtime source files so "
        "a rerun can be audited against its exact working tree.", "",
        *REPRODUCTION.strip().splitlines(), ""]
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
