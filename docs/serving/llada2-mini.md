# LLaDA2.0-mini on one GPU

This recipe uses one NVIDIA GPU with compute capability 9.0 or newer and enough memory for the model and KV cache. Complete the [Docker installation](../guides/getting_started.md) first.

## Launch configuration

```bash
CUDA_VISIBLE_DEVICES=0 fluxserve serve \
  --model inclusionAI/LLaDA2.0-mini \
  --host 127.0.0.1 --port 8000 \
  --tp-size 1 --dp-size 1 --ep-size 1 \
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

## Configuration notes

- `--max-num-seqs` limits concurrent sequences; `--max-model-len` sets the context-length limit.
- `--block-length` and `--threshold` configure block diffusion decoding.
- The listed CUDA graph capture sizes cover this recipe's batch sizes.
- If model loading or cache allocation exceeds available memory, use a suitable GPU or reduce context length and concurrency before benchmarking.

## Verify and benchmark

Use the readiness check and chat request in the [quickstart](../guides/quickstart.md), then follow the [online benchmark](../guides/benchmark.md#online-benchmark).
