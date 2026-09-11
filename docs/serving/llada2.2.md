# Serving LLaDA2.2

Complete the [Docker installation](../guides/getting_started.md) first. Run the example from the FluxServe repository directory. The four-GPU configuration needs capacity for roughly 206GB of weights plus KV cache and runtime buffers.

LLaDA2.2-flash uses `LLaDA2LLM` with config-driven block routing. Its
`levenshtein_joint` decoder extends M2T/T2T updates with DELETE/SPLIT
operations while keeping the active block length fixed.

See the [model support development guide](../../dev-notes/llada2.2-model-support-development-guide.md) for the full design, invariants, and test plan.

## Launch configuration

Four GPUs, TP=4 / EP=4, paged FlashInfer with decode CUDA graphs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m fluxserve.cli serve \
  --model inclusionAI/LLaDA2.2-flash \
  --host 127.0.0.1 --port 8000 \
  --tp-size 4 --dp-size 1 --ep-size 4 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 8 \
  --max-model-len 16384 \
  --block-length 32 \
  --parallel-decoding levenshtein_joint \
  --threshold 0.5 \
  --editing-threshold 0.0 \
  --steps 32 \
  --max-post-steps 16 \
  --max-steps-per-block 1000 \
  --attention-backend flashinfer \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --cuda-graph-capture-bs 1 2 4 8
```

## Configuration notes

- Pass the shown decoding values explicitly; CLI defaults differ. Temperature
  0 is the only supported setting, and `--block-length` must be a multiple of 32.
- `--steps` spreads the initial-mask transfers over refinement steps.
  `--max-post-steps` bounds refinement after original masks disappear;
  `--max-steps-per-block` forces final resolution at the hard cap.
- Both stop IDs (`156892`, `156900`) load from `generation_config.json`.
- Decode CUDA graphs use padded batches. The fused decoder step is implemented;
  H200 capture/replay validation remains outstanding.
- For dense-only FlashInfer builds, replace the last five flags with
  `--kv-cache-layout dense --flashinfer-cache-mode dense
  --flashinfer-prefill-mode dense`; this disables decode graphs.
- Use the [quickstart](../guides/quickstart.md) health check and request,
  setting the request model to `inclusionAI/LLaDA2.2-flash`.
