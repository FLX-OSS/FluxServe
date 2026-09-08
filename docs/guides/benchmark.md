## Notice: This is for internal benchamrk only!

## Offline Benchmark
FluxServe supports offline thorughput benchamrk with json-style input files.
```bash
fluxserve bench_offline \
  --model inclusionAI/LLaDA2.0-mini \
  --dataset ./data/humaneval.jsonl \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --batch-size 4 \
  --gen-len 512 \
  --block-length 64 \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --cuda-graph-capture-bs 1 2 4 
```

For LLaDA 2.0 threshold decoding, add `--profile-block-metrics` to write a
separate `*_block_profile.jsonl` sidecar. Each request contains per-block
forward counts, mask-to-token transfers, and transferred-token confidence
summaries. Each forward also reports the unresolved-token confidence
distribution, threshold readiness and deficit, top-2 margin, prediction flip
rate, and fallback streak; blocks summarize their fallback and single-token
forwards. The final zero-transfer KV-commit forward is included. Because the
instrumented run performs extra reductions and retains profiling metadata, use
an uninstrumented run for latency and throughput comparisons.

## Online Benchmark

1. Launch FluxServe engine

```bash
fluxserve serve \
  --model inclusionAI/LLaDA2.0-mini \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --max-num-seqs 4 \
  --max-model-len 4096 \
  --block-length 64 \
  --threshold 0.95 \
  --parallel-decoding threshold \
  --attention-backend flashinfer \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --cuda-graph-capture-bs 1 2 4 
```

2. Check server health
```bash
curl -fsS http://127.0.0.1:8000/health
```

3. Run benchmark
```bash
fluxserve serve \
  --model inclusionAI/LLaDA2.0-mini \
  --dataset ./data/humaneval.jsonl \
  --dataset-output-len 512 \
  --request-rate 1 \
  --max-concurrency 32 \
  --metric-percentiles 50,90,99 \
```


## Third-party API Benchmark
To measure the performance in real-world serving, we also provide scripts to benchmark FluxServe through third-party tools (evalscope-perf). 
To reproduce the results in the figure, refer to [bench.sh](./test/benchmark/fluxserve/bench.sh) and [agentic_bench.sh](./test/benchmark/fluxserve/agentic_bench.sh)