### Nemotron-Labs-Diffusion-14B (TP=1)

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
  --block-length 32 \
  --page-size 32 \
  --parallel-decoding threshold \
  --threshold 0.9 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --trust-remote-code
```

### Nemotron-Labs-Diffusion-14B, self-speculation (TP=1)

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
  --max-thinking-tokens 1024 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --trust-remote-code
```

### Configuration Notes

- Nemotron-Labs-Diffusion supports threshold diffusion and self-speculation.
- `--block-length` must be a multiple of 32.
- Self-speculation only supports FA4.
- Self-speculation loads the checkpoint's optional `linear_spec_lora` adapter
  when present, and uses it for drafting only. Draft and verify forwards are
  captured as separate graphs; sampling, acceptance and rollback run outside
  them.


