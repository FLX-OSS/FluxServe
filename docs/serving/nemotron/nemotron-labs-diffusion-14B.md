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

### Configuration Notes

- Nemotron-Labs-Diffusion uses threshold-based parallel decoding.
- `--block-length` must be a multiple of 32.
- Chat completions use the checkpoint's chat template automatically. Plain
  completions accept an already rendered prompt or token IDs.
- `--temperature` defaults to zero (greedy). Requests can override `temperature`
  and `seed`; random streams belong to individual requests. Only `top_p=1`,
  disabled `top_k`, and zero frequency/presence penalties are supported.
- `ignore_eos=true` continues generation through EOS up to the request budget.
- `--max-thinking-tokens` enables the checkpoint's thinking-budget policy;
  `</think>` is resolved from the tokenizer unless `--end-think-token-id` is set.

### Execution Modes

| Attention backend | Cache / scheduler | Threshold decoding | Self-speculation |
| --- | --- | --- | --- |
| `sdpa` | Dense / default | Eager, requests run sequentially | Eager, use `--max-num-seqs 1` |
| `fa4` | Paged / paged | Eager or decode CUDA graphs | Eager |
| `flashinfer` | Paged / paged | Eager | Eager |

Select self-speculation with `--parallel-decoding self_speculation`; remove
decode graph flags for that mode. The optional `linear_spec_lora` draft adapter
is loaded when present in the local checkpoint snapshot. Base-weight downloads
do not fetch that adapter automatically.

FA4 decode graphs require `--page-size` equal to `--block-length`. Prefill
graphs are unsupported. `--nemotron-prefill-chunk-size` defaults to 1024.
Requests must leave room for complete diffusion blocks, or an extra speculative
block in self-speculation mode, within `--max-model-len`. The architectural
position limit is 262144; this is not a claim of validated long-context quality.

### Validation

Run CPU regressions for all three model families and CI configuration checks:

```bash
python -m pytest test/runtime test/ci_system -q
```

The tests cover weight mapping, attention and cache semantics, request sampling,
EOS handling, and online/offline generation contracts. GPU and official-checkpoint
integration tests require their documented runtime and opt-in settings. Separate
Nemotron AR, diffusion, online, and long-context harnesses live in
`test/runtime/integration/nemotron_*_harness.py`; the unit tests alone do not
establish GPU numerical parity or benchmark quality.
