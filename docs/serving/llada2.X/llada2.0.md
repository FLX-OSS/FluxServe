### LLaDA2.0-mini (TP=EP=1)

```bash
fluxserve launch \
  --model inclusionAI/LLaDA2.0-mini \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 16 \
  --max-model-len 65536 \
  --block-length 64 \
  --threshold 0.95 \
  --parallel-decoding threshold \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph
```

### LLaDA2.0-flash (TP=EP=4)

```bash
fluxserve launch \
  --model inclusionAI/LLaDA2.0-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 4 \
  --dp-size 1 \
  --ep-size 4 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 16 \
  --max-model-len 65536 \
  --block-length 64 \
  --parallel-decoding threshold \
  --threshold 0.95 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph
```

### Configuration Notes

- LLaDA2.0 models use threhold-based parallel decoding.
- LLaDA2.0-CAP models use the same commands.
