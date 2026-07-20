## Offline Benchmark
FluxServe supports offline thorughput benchamrk with json-style input files.
```bash
python -m fluxserve.cli bench_offline \
  --model inclusionAI/LLaDA2.0-mini \
  --dataset ./data/sample.jsonl \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --batch-size 4 \
  --gen-len 512 \
  --block-length 64 \
  --threshold 0.95 \
  --parallel-decoding threshold \
  --attention-backend flashinfer \
  --kv-cache-layout paged
```

## Online Benchmark

1. Launch FluxServe engine

```bash
python -m fluxserve.cli serve \
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
  --scheduler-policy paged
```

2. Check server health
```bash
curl -fsS http://127.0.0.1:8000/health
```

3. Run benchmark
```bash
python -m fluxserve.cli serve \
  --model inclusionAI/LLaDA2.0-mini \
  --dataset ./data/humaneval.jsonl \
  --dataset-output-len 512 \
  --scheduler-policy paged \
  --request-rate 4 \
  --max-concurrency 32 \
  --metric-percentiles 50,90,99
```
