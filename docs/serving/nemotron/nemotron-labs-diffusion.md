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

### Nemotron-Labs-Diffusion-8B (TP=EP=1)

```bash
fluxserve launch \
  --model nvidia/Nemotron-Labs-Diffusion-8B \
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

### Nemotron-Labs-Diffusion-14B (TP=EP=1)

```bash
fluxserve launch \
  --model nvidia/Nemotron-Labs-Diffusion-14B \
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

### Nemotron-Labs-Diffusion-14B, self-speculation (TP=EP=1)

```bash
fluxserve launch \
  --model nvidia/Nemotron-Labs-Diffusion-14B \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --max-model-len 8192 \
  --max-scheduled-tokens 2048 \
  --block-length 32 \
  --page-size 32 \
  --parallel-decoding self_speculation \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --cuda-graph-capture-bs 1 2 4 8 16 \
  --trust-remote-code
```

### Configuration Notes

- Nemotron-Labs-Diffusion supports threshold diffusion and self-speculation.
- `--block-length` must be a multiple of 32.
- Decode graphs require `--page-size` equal to `--block-length`, and
  `--cuda-graph-capture-bs` must cover `--max-num-seqs`.
- The same self-speculation flags apply to the 3B and 8B checkpoints.
- For FlashInfer, replace `--attention-backend fa4` with
  `--attention-backend flashinfer --flashinfer-prefill-mode paged
  --flashinfer-cache-mode paged`. Every other flag stays the same.
- Remove the three `--*cuda-graph*` options for an eager baseline. The SDPA
  path is always eager.
- Self-speculation loads the checkpoint's optional `linear_spec_lora` adapter
  when present, and uses it for drafting only. Draft and verify forwards are
  captured as separate graphs; sampling, acceptance and rollback run outside
  them.

