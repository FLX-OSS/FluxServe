### LLaDA2.2-mini (TP=EP=1)
```bash
fluxserve launch \
  --model inclusionAI/LLaDA2.2-mini \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 16 \
  --max-model-len 65536 \
  --block-length 64 \
  --parallel-decoding levenshtein_joint \
  --threshold 0.5 \
  --editing-threshold 0.0 \
  --steps 32 \
  --max-post-steps 16 \
  --max-steps-per-block 1000 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph
```

### LLaDA2.2-flash (TP=EP=4)
```bash
fluxserve launch \
  --model inclusionAI/LLaDA2.2-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 4 \
  --dp-size 1 \
  --ep-size 4 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 16 \
  --max-model-len 65536 \
  --block-length 64 \
  --parallel-decoding levenshtein_joint \
  --threshold 0.5 \
  --editing-threshold 0.0 \
  --steps 32 \
  --max-post-steps 16 \
  --max-steps-per-block 1000 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph
```

### Configuration Notes
- LLaDA2.2 models use levenshtein editing parallel decoding method.
- `--block-length` must be a multiple of 32.
- `--steps` spreads the initial-mask transfers over refinement steps.
- `--max-post-steps` bounds refinement after original masks disappear;
- `--max-steps-per-block` forces final resolution at the hard cap.