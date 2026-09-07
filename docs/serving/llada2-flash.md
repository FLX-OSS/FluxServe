# LLaDA2.0-flash on four GPUs

Complete the [Docker installation](../guides/getting_started.md) first. This recipe targets four NVIDIA H100 GPUs with sufficient memory for the checkpoint and cache.

## Launch Command

TP=4, EP=4, Paged Flashinfer (H100, 4 x SM90)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 fluxserve serve \
  --model inclusionAI/LLaDA2.0-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 4 \
  --dp-size 1 \
  --ep-size 4 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 8 \
  --max-model-len 16384 \
  --block-length 64 \
  --parallel-decoding threshold \
  --threshold 0.95 \
  --attention-backend flashinfer \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --cuda-graph-capture-bs 1 2 4 8
```

## Verify the server

After model initialization, use the health check and request in the [quickstart](../guides/quickstart.md), replacing the request model with `inclusionAI/LLaDA2.0-flash`. See [benchmarking](../guides/benchmark.md) to measure the running service.
