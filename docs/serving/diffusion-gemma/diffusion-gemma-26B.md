### Diffusion-Gemma (TP=EP=1)

```bash
fluxserve launch \
  --model google/diffusiongemma-26B-A4B-it \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 4 \
  --max-model-len 8192 \
  --block-length 256 \
  --canvas-length 256 \
  --page-size 256 \
  --attention-backend flashinfer \
  --kv-cache-layout paged \
  --scheduler-policy default \
  --use-decode-cuda-graph
```

### Configuration Notes

