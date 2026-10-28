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

### Diffusion-Gemma with FA4 (B200 / SM100)

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
  --page-size 128 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy default \
  --use-decode-cuda-graph
```

### Configuration Notes

- `--attention-backend flashinfer` runs every layer on FlashInfer (SM90, SM100, SM120).
- `--attention-backend fa4` (SM100/SM110 only) runs the sliding-window layers
  (head_dim 256) on FA4's head_dim-256 kernel and, by default, the
  full-attention layers (head_dim 512) as FA4 MLA: their K is the rotary of
  `k_norm`'s scalar times V on 128 of the 512 dims, so they cache those 128
  dims of K next to V (640 instead of 1024 values per token and KV head) and
  run 4-7x faster than FlashInfer at long context. `--full-attention-backend`
  picks `auto` (MLA when the checkpoint's `k_norm` weights are scalars),
  `fa4` (required), or `flashinfer`. It needs the flash-attn-dllm FA4 build
  and 128-token pages; `--page-size` defaults to 128 and other values are
  rejected. Set `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` to keep the compiled
  kernels across restarts.
- `--sliding-window-mode` selects the denoise sliding window. `per_query`
  (default) lets every canvas token see keys within 1023 positions of itself.
  `per_row` gives every canvas token the last `--sliding-window-left` cache
  tokens (default 1023) plus the canvas, which is what the Hugging Face
  reference decoder computes. They differ once the prompt plus committed text
  reaches 769 tokens. `per_row` works with FA4 (eager and CUDA graphs) and with
  eager FlashInfer; FlashInfer decode CUDA graphs implement `per_query` only.

