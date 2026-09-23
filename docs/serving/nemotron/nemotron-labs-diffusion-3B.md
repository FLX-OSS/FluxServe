### Nemotron-Labs-Diffusion-3B (TP=EP=1)

```bash
fluxserve launch \
  --model nvidia/Nemotron-Labs-Diffusion-3B \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 8 \
  --max-model-len 8192 \
  --block-length 32 \
  --page-size 32 \
  --parallel-decoding threshold \
  --threshold 0.9 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --cuda-graph-capture-bs 1 2 4 8 \
  --trust-remote-code
```

### Configuration Notes

- Nemotron-Labs-Diffusion uses threshold-based parallel decoding.
- `--block-length` must be a multiple of 32.
- FA4 decode graphs require `--page-size` equal to `--block-length`.
- For self-speculation, use `--parallel-decoding self_speculation` without CUDA graph flags.
