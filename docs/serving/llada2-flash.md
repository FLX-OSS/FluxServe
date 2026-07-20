# LLaDA-2.0-Flash


## Launch Command

TP=4, EP=4, Paged Flashinfer (H100, 4 x SM90)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m fluxserve.cli serve \
  --model inclusionAI/LLaDA2.0-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 4 \
  --dp-size 1 \
  --ep-size 4 \
  --max-num-seqs 8 \
  --max-model-len 16384 \
  --block-length 64 \
  --parallel-decoding threshold \
  --threshold 0.95 \
  --attention-backend flashinfer \
  --kv-cache-layout paged \
  --scheduler-policy paged
```