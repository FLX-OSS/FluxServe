# LLaDA2.0-mini

```bash
fluxserve launch \
  --model inclusionAI/LLaDA2.0-mini \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --parallel-decoding threshold \
  --threshold 0.95 \
  --attention-backend flashinfer \
  --scheduler-policy paged \
  --use-decode-cuda-graph
```

# LLaDA2.0-flash (TP=EP=4)
```bash
fluxserve launch \
  --model inclusionAI/LLaDA2.0-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 4 \
  --dp-size 1 \
  --ep-size 4 \
  --parallel-decoding threshold \
  --threshold 0.95 \
  --attention-backend flashinfer \
  --scheduler-policy paged \
  --use-decode-cuda-graph
```
