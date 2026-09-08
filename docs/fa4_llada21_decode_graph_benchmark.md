# FA4 full-model decode CUDA graph: LLaDA2.1-mini / GSM8K

## Configuration and measurement

Cached `inclusionAI/LLaDA2.1-mini` revision `20e64e2ad21644d0e5248586ed9c942cdd45de0f`. Packages: `{'torch': '2.8.0+cu129', 'transformers': '5.15.1', 'flash-attn-4': '4.0.0b29', 'flashinfer-python': '0.6.18'}`.

Single GH200 per configuration, TP/DP/EP=1, BF16, native FluxServe paged scheduling. Configuration follows `test/benchmark/fluxserve/configs/tp1_ep1_llada21_mini.sh`: max_model_len=65536, max_num_seqs=16, max_scheduled_tokens=2048, gpu_memory_utilization=0.8, block/page=64, joint_threshold Quality 0.7/0.5, max_post_steps=16. The unchanged CLI mini_batch_size default is 4. All requests use the checkpoint chat template, temperature=0 and max_tokens=2048. 64K is the configured capacity, not every request's KV length.

FA4 and BatchPrefill each run eager and padded decode graph. Captured batch buckets are 1/2/4/8/10/12/16; prefill stays eager. FA4 captures all transformer layers, LM head, joint token selection and block-finished predicates. No vLLM/SGLang APIs are used.

Closed-loop HTTP concurrency; stream=False. Latencies cover complete responses, not TTFT or inter-token latency. Startup, capture and warmup are excluded. Throughput uses actual server-counted output tokens. Every setting has three rounds. Concurrency 16 uses all 1319 questions; 1/4/8 use the same first 128 questions. Do not interpret the subset/full-set boundary as a pure scaling experiment.

## Correctness

Final GH200 validation: 54 tests passed, including attention versus SDPA, adapter replay, decoder state progression and real-weight graph/eager comparison. The model test covers 19 replays across 7 buckets, changing page IDs and KV lengths up to 65536, and every actual batch size from 1 through 16. Updated tokens and all three decoder flags, logits and the entire real KV pool must match exactly (atol=rtol=0). KV-pool replacement also verifies capture invalidation and refusal of stale replay. This verifies FA4 graph versus FA4 eager; it does not resolve the earlier FA4-versus-dense-SDPA whole-model parity failure.

All graph rounds below require positive actual replay counts and zero decode fallback. GSM8K scoring uses the same last-signed-decimal extraction against verified official answers, including length-limited responses. Natural generation can differ across backends and padded batch shapes; forward counts and accuracy accompany throughput.

## Full GSM8K, concurrency 16

Values are medians across three rounds; throughput brackets give min–max.

| Configuration | Output tokens/s [min–max] | Requests/s | Wall s | Latency mean / P50 / P95 / P99 s | Accuracy range | Forwards range |
|---|---:|---:|---:|---:|---:|---:|
| fa4_eager | 920.06 [916.90–923.32] | 3.078 | 428.54 | 5.16 / 4.57 / 8.67 / 28.94 | 90.45–90.45% | 17076–17077 |
| fa4_graph | 1904.45 [1902.03–1905.07] | 6.371 | 207.03 | 2.50 / 2.19 / 4.20 / 14.11 | 90.45–90.45% | 17076–17076 |
| batchprefill_eager | 940.20 [939.27–940.91] | 3.211 | 410.73 | 4.94 / 4.38 / 8.04 / 26.73 | 89.54–89.54% | 16595–16595 |
| batchprefill_graph | 1538.42 [1536.68–1560.18] | 5.102 | 258.54 | 3.13 / 2.66 / 5.84 / 17.76 | 89.69–89.76% | 17737–18125 |

### FA4 eager / graph completion parity

Exact response-text and finish-reason matches, paired by dataset index within each round. This is end-to-end output parity, not a claim of bitwise whole-model logits.

| Concurrency | Exact matches by round | Output-token totals equal | Forward counts equal |
|---:|---|---|---|
| 1 | 128/128, 128/128, 128/128 | True | True |
| 4 | 112/128, 110/128, 127/128 | False | False |
| 8 | 128/128, 128/128, 128/128 | True | True |
| 16 | 1319/1319, 1319/1319, 1319/1319 | True | False |

## Same 128-question concurrency sweep

Median output tokens/s [min–max]. Server capacity stays fixed at 16.

| HTTP concurrency | FA4 eager | FA4 graph | BatchPrefill eager | BatchPrefill graph |
|---:|---:|---:|---:|---:|
| 1 | 179.71 [174.93–180.28] | 613.69 [611.14–614.77] | 168.81 [168.31–168.92] | 470.19 [470.08–470.50] |
| 4 | 332.13 [331.61–333.74] | 921.56 [919.26–924.51] | 375.15 [331.30–376.06] | 799.89 [796.66–802.39] |
| 8 | 559.03 [554.18–559.32] | 1301.35 [1300.30–1303.34] | 546.69 [545.73–547.15] | 1072.94 [1065.54–1073.61] |

## GPU memory and graph execution (full-set runs)

Peak allocation includes startup and capture. GPU memory is PyTorch allocated memory, not total device consumption. Capture memory excludes the persistent model/KV pool.

| Configuration | Node / job | Peak allocated GiB | Capture s | Decode replays / round | Length stops / round |
|---|---|---:|---:|---|---|
| fa4_eager | gh111.hsn.cm.delta.internal.ncsa.edu / 3109313 | 72.44 | 0.00 | [0, 0, 0] | [19, 19, 19] |
| fa4_graph | gh007.hsn.cm.delta.internal.ncsa.edu / 3109278 | 74.72 | 4.80 | [16607, 16607, 16607] | [19, 19, 19] |
| batchprefill_eager | gh115.hsn.cm.delta.internal.ncsa.edu / 3109315 | 72.86 | 0.00 | [0, 0, 0] | [15, 15, 15] |
| batchprefill_graph | gh016.hsn.cm.delta.internal.ncsa.edu / 3109318 | 72.99 | 28.13 | [17266, 17660, 17660] | [19, 22, 22] |

## Same-GPU control

All four variants below run sequentially on the same GH200, concurrency 16, the same first 128 questions, three rounds per variant. This checks whether the cross-card comparison changes direction. Order is fixed, not randomized. Other server and decoder settings match the main matrix.

| Configuration | Output tokens/s median [min–max] | P95 latency s | Accuracy range | Node / job |
|---|---:|---:|---:|---|
| fa4_eager | 818.06 [817.60–823.64] | 7.97 | 89.84–89.84% | gh007.hsn.cm.delta.internal.ncsa.edu / 3109278 |
| fa4_graph | 1798.86 [1788.96–1804.90] | 4.02 | 89.84–89.84% | gh007.hsn.cm.delta.internal.ncsa.edu / 3109278 |
| batchprefill_eager | 911.52 [905.96–915.66] | 7.83 | 89.84–89.84% | gh007.hsn.cm.delta.internal.ncsa.edu / 3109278 |
| batchprefill_graph | 1361.85 [1361.39–1368.60] | 5.03 | 89.06–89.06% | gh007.hsn.cm.delta.internal.ncsa.edu / 3109278 |

## Interpretation and reproduction

These are serving measurements with natural EOS, not equal-work attention microbenchmarks. Different completions and editing iterations affect throughput. Each variant uses a separate allocated GH200; fixed variant-to-card assignment leaves a possible card/host effect. FlashInfer eager dispatches backend=auto while its existing decode graph uses backend=fa2, so its eager/graph ratio includes kernel-dispatch changes. No matched-FA2 eager control, nsys or ncu run is claimed.

The large raw manifests, responses and profiler artifacts are intentionally not
committed. The reviewable aggregate measurements are preserved above, while
the commands below regenerate manifests containing the exact server command,
package versions, source and dataset hashes, and Slurm host/job metadata.

During the long benchmark, the FA4 teardown hook was aligned with the executor's
`shutdown_cuda_graphs` API and regression-tested. This changes teardown only,
outside all timed regions. Regenerated manifests also hash untracked runtime
source files so a rerun can be audited against its exact working tree.

### Reproduce the full GSM8K concurrency-16 runs

Run from the repository root inside a single-GH200 Slurm allocation. The
benchmark input is the repository's complete 1,319-request
`data/gsm8k.jsonl`; omitting `--limit` is intentional. Output directories must
not already exist. `run_fa4_graph_benchmark.sh` prepares the pinned GH200
container and standalone FA4 dependencies without importing vLLM/SGLang.

```bash
cd /u/dzhu8/workspace/FluxServe

OUTPUT_ROOT=test/runtime/integration/profiles/llada21_c16_reproduce
mkdir -p "$OUTPUT_ROOT"

COMMON_ARGS=(
  --dataset data/gsm8k.jsonl
  --concurrency 16
  --repeats 3
  --max-num-seqs 16
  --mini-batch-size 4
  --max-model-len 65536
  --max-scheduled-tokens 2048
  --scheduler-num-device-pages 0
  --gpu-memory-utilization 0.8
  --capture-bs 1 2 4 8 10 12 16
)

# FA4 eager
bash test/runtime/integration/run_fa4_graph_benchmark.sh \
  python test/runtime/integration/bench_llada21_quality.py \
  --backend fa4 "${COMMON_ARGS[@]}" \
  --output "$OUTPUT_ROOT/fa4_eager_c16"

# FA4 full-model decode CUDA graph
bash test/runtime/integration/run_fa4_graph_benchmark.sh \
  python test/runtime/integration/bench_llada21_quality.py \
  --backend fa4 --graph "${COMMON_ARGS[@]}" \
  --output "$OUTPUT_ROOT/fa4_graph_c16"

# BatchPrefill eager
bash test/runtime/integration/run_fa4_graph_benchmark.sh \
  python test/runtime/integration/bench_llada21_quality.py \
  --backend flashinfer "${COMMON_ARGS[@]}" \
  --output "$OUTPUT_ROOT/batchprefill_eager_c16"

# BatchPrefill full-model decode CUDA graph
bash test/runtime/integration/run_fa4_graph_benchmark.sh \
  python test/runtime/integration/bench_llada21_quality.py \
  --backend flashinfer --graph "${COMMON_ARGS[@]}" \
  --output "$OUTPUT_ROOT/batchprefill_graph_c16"
```

Each configuration writes `manifest.json`, three `summary_*.json` files and
the corresponding response JSONL files. The reported throughput is
`server_output_tokens_per_s`; latency, model-forward count, graph replay and
GPU-memory fields are recorded in the same summaries. A graph run fails if it
does not record decode replays or performs any eager decode fallback.

To reproduce accuracy, obtain the official GSM8K test split in its original
JSONL form (`question` and `answer` fields), verify that its 1,319 questions
match the request file, and set `GSM8K_TEST` to that path:

```bash
GSM8K_TEST=/path/to/official/gsm8k_test.jsonl

for variant in \
  fa4_eager_c16 fa4_graph_c16 \
  batchprefill_eager_c16 batchprefill_graph_c16
do
  for round in 0 1 2
  do
    python test/runtime/integration/score_llada21_gsm8k.py \
      --tests "$GSM8K_TEST" \
      --responses "$OUTPUT_ROOT/$variant/responses_${round}.jsonl" \
      --expected-count 1319
  done
done
```

To run the full four-concurrency matrix instead, place the verified official
test file at `OUTPUT_ROOT/gsm8k_test.jsonl`, then run one variant per allocated
GPU:

```bash
bash test/runtime/integration/run_fa4_graph_benchmark.sh \
  bash test/runtime/integration/run_llada21_graph_matrix.sh \
  fa4_graph "$OUTPUT_ROOT"
```

The accepted variant names are `fa4_eager`, `fa4_graph`,
`batchprefill_eager`, and `batchprefill_graph`.
