### LLaDA2.1-mini (TP=EP=1)

```bash
fluxserve launch \
  --model inclusionAI/LLaDA2.1-mini \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --max-num-seqs 16 \
  --max-model-len 65536 \
  --block-length 64 \
  --parallel-decoding joint_threshold \
  --threshold 0.7 \
  --editing-threshold 0.5 \
  --max-post-steps 16 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph
```

### LLaDA2.1-flash (TP=EP=4)

```bash
fluxserve launch \
  --model inclusionAI/LLaDA2.1-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 4 \
  --dp-size 1 \
  --ep-size 4 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 8 \
  --max-model-len 16384 \
  --block-length 32 \
  --parallel-decoding joint_threshold \
  --threshold 0.7 \
  --editing-threshold 0.5 \
  --max-post-steps 16 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph
```

### Configuration Notes

- LLaDA2.1 models use joint threshold parallel decoding method.
- There are two presets for LLaDA2.1: 
  | Preset | `--threshold` | `--editing-threshold` |
  | :---: | :---: | :---: |
  | Quality | 0.7 | 0.5 |
  | Speed | 0.5 | 0.0 |
- `--max-post-steps` bounds mask-free editing per row. A zero editing

