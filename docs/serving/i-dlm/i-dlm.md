### I-DLM-8B (TP=EP=1)

```bash
fluxserve launch \
  --model yifanyu/I-DLM-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --max-model-len 40960 \
  --max-scheduled-tokens 2048 \
  --block-length 32 \
  --page-size 32 \
  --parallel-decoding isd \
  --idlm-gen-block-size 4 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --cuda-graph-capture-bs 1 2 4 8 16 \
  --trust-remote-code
```

### I-DLM-32B (TP=EP=1)

```bash
fluxserve launch \
  --model yifanyu/I-DLM-32B \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 8 \
  --max-model-len 8192 \
  --max-scheduled-tokens 2048 \
  --block-length 32 \
  --page-size 32 \
  --parallel-decoding isd \
  --idlm-gen-block-size 4 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --cuda-graph-capture-bs 1 2 4 8 \
  --trust-remote-code
```

### Qwen3-8B + I-DLM-8B-lora-r128, R-ISD (TP=1)

```bash
fluxserve launch \
  --model Qwen/Qwen3-8B \
  --idlm-lora yifanyu/I-DLM-8B-lora-r128 \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --max-model-len 40960 \
  --max-scheduled-tokens 2048 \
  --block-length 32 \
  --page-size 32 \
  --parallel-decoding isd \
  --idlm-gen-block-size 4 \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --use-decode-cuda-graph \
  --cuda-graph-decode-mode padded \
  --cuda-graph-capture-bs 1 2 4 8 16 \
  --trust-remote-code
```

### Configuration Notes

- I-DLM models use Introspective Strided Decoding (ISD). Each step is one
  causal forward of `2N-1` tokens (`N = --idlm-gen-block-size`) and emits 1
  to `N` tokens.
- `--temperature 0` (default) is lossless: the output equals greedy
  autoregressive decoding. The paper's setting is `--temperature 1.0
  --top-k 50 --top-p 0.95`; requests can override these and `seed`.
- `--block-length` is the KV page size; it must be a multiple of 16 and at
  least `2N-1`.
- Thinking traces run long, so keep `--max-model-len` large (40960 at most).
  A request that reaches it ends with `finish_reason="length"`.
- Running out of KV currently hangs the server. Keep the tokens in flight
  within the pool logged at startup (`Profiled paged KV cache: pages=N`,
  `N × 32` tokens; about 438k for 8B, 78k for 32B), as the 32B command does
  with 8 × 8192.
- R-ISD applies the adapter at `[MASK]` positions only, so the output is
  Qwen3-8B's own. It supports TP=1 only.
