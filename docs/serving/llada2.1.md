# Serving LLaDA2.1

Complete the [Docker installation](../guides/getting_started.md) first. Run the examples from the FluxServe repository directory with enough GPU memory for the checkpoint and KV cache.

LLaDA2.1-mini and LLaDA2.1-flash use the shared `LLaDA2LLM` model class.
The `joint_threshold` decoder combines Mask-to-Token (M2T) generation with
Token-to-Token (T2T) editing inside the active block.

See the [model support development guide](../../dev-notes/llada2.1-model-support-development-guide.md) for the full design, invariants, and test plan.

## Launch configuration

### Mini — one GPU, TP=1 / EP=1

```bash
CUDA_VISIBLE_DEVICES=0 python -m fluxserve.cli serve \
  --model inclusionAI/LLaDA2.1-mini \
  --host 127.0.0.1 --port 8000 \
  --tp-size 1 --dp-size 1 --ep-size 1 \
  --max-num-seqs 1 \
  --max-model-len 32768 \
  --block-length 32 \
  --parallel-decoding joint_threshold \
  --threshold 0.7 \
  --editing-threshold 0.5 \
  --max-post-steps 16 \
  --attention-backend flashinfer \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --cuda-graph-capture-bs 1
```

### Flash — four GPUs, TP=4 / EP=4

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m fluxserve.cli serve \
  --model inclusionAI/LLaDA2.1-flash \
  --host 127.0.0.1 --port 8000 \
  --tp-size 4 --dp-size 1 --ep-size 4 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 8 \
  --max-model-len 16384 \
  --block-length 32 \
  --parallel-decoding joint_threshold \
  --threshold 0.7 \
  --editing-threshold 0.5 \
  --max-post-steps 16 \
  --attention-backend flashinfer \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --cuda-graph-capture-bs 1 2 4 8
```

## Configuration notes

Both commands use the Quality preset, block length 32, temperature 0, and
`max_post_steps=16`. Pass thresholds explicitly; CLI defaults differ.

| Preset | `--threshold` | `--editing-threshold` |
| --- | --- | --- |
| Quality | 0.7 | 0.5 |
| Speed | 0.5 | 0.0 |

- `--max-post-steps` bounds mask-free editing per row. A zero editing
  threshold enables T2T rewrites; it does not disable editing.
- Decode CUDA graphs use paged FlashInfer KV and padded capture batches.
  Match capture sizes to `--max-num-seqs`.
- For dense-only FlashInfer builds, replace the last five flags in either
  command with `--kv-cache-layout dense --flashinfer-cache-mode dense
  --flashinfer-prefill-mode dense`; this disables decode graphs.
- Use the [quickstart](../guides/quickstart.md) health check and request,
  setting the request model to the selected checkpoint.
