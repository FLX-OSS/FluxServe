# LLaDA 2.x with the FA4 backend

FluxServe can run LLaDA 2.x attention through the standalone
`flash-attn-4` CuTeDSL package. The adapter consumes FluxServe's own request
metadata and paged KV cache directly; it does not use a compatibility layer
from another serving system.

LLaDA2.1-mini BF16 eager generation has been exercised on GH200 through both
the offline runner and the real HTTP paged scheduler. The end-to-end tests
cover short answers, long prompts, multi-block output, concurrent requests,
EOS/length termination and request-slot reuse. Threshold and hierarchy
decoders are supported by the FA4 runner.

This remains experimental: the strict random-input full-model logits parity
test does not pass, and arbitrary cross-backend token identity is not promised.
The counting example differs from dense SDPA by a comma, although both produce
the correct integer sequence. Full-model decode CUDA Graph is supported for
the TP/DP/EP/PP=1 paged path described below; prefill remains eager. The
full-pipeline validation uses BF16 (the standalone attention tests also cover
FP16). See [end-to-end validation](../fa4_llada_e2e.md) for exact scope.

The adapter skips token gathers and output scatter when all requests have the
full input length (including block decode), writes contiguous paged caches using
flat slot indices, and returns a head-major view backed by token-major storage.
This avoids the output layout round-trip before LLaDA's output projection.
Ragged batches retain zero-filled padding and explicit token gathering.

## Environment

The supported container is built from `docker/Dockerfile.flux-cu129`. It pins:

- Python 3.12
- PyTorch 2.8 with CUDA 12.9
- `flash-attn-4==4.0.0b29`
- `nvidia-cutlass-dsl==4.6.2`

The backend currently requires BF16 or FP16 and a Hopper or supported
Blackwell GPU with compute capability 9.x, 10.x, or 11.x. The page size must be
a multiple of 16. Online paged scheduling additionally requires the page size
to equal the LLaDA block size.

For a non-container installation, install FluxServe with its optional FA4
dependency:

```bash
python -m pip install -e '.[fa4]'
```

Verify the installed package with:

```bash
fluxserve env
```

## Offline generation

```bash
fluxserve bench_offline \
  --model inclusionAI/LLaDA2.0-mini \
  --dataset /path/to/prompts.jsonl \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --page-size 64 \
  --block-length 64
```

## Online serving

Use the paged scheduler so scheduler-owned page IDs are passed directly into
the FluxServe KV pool:

```bash
fluxserve serve \
  --model inclusionAI/LLaDA2.1-mini \
  --apply-template \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --page-size 64 \
  --block-length 64
```

For the tested single-GPU smoke configuration, also set
`--max-num-seqs 2 --max-model-len 512 --scheduler-num-device-pages 32`.
`--apply-template` uses the checkpoint's chat template, matching the offline
end-to-end validator. The default legacy prompt renderer produces different
input tokens and can produce different answers.

FA4 kernels are JIT-compiled for the first unseen shape, so the first request
can take longer. Online TP1/EP1/DP1 decode supports full-model CUDA graphs:

```bash
--use-decode-cuda-graph --cuda-graph-decode-mode padded \
--cuda-graph-capture-bs 1 2 4 8 10 12 16
```

The largest bucket must cover `max_num_seqs`. Page size must equal block size.
Graphs capture the model, LM head, and (for `joint_threshold`) token selection
and block-finished predicates. KV lengths, physical page tables, positions,
prompt protection and editing budgets are updated before each replay. Padding
rows write distinct reserved pages outside the scheduler pool. KV-pool changes
invalidate captures. Prefill remains eager; prefill graph and distributed FA4
graph requests are rejected. The metrics endpoint exposes actual decode capture,
replay and padded-row counts. A runnable 64K/16-request configuration is
`test/benchmark/fluxserve/configs/tp1_ep1_llada21_mini_fa4.sh`.

Real-weight graph correctness can be checked with:

```bash
FLUXSERVE_RUN_FA4_GRAPH_MODEL=1 python -m pytest -q -s \
  test/runtime/integration/test_fa4_decode_graph_model.py
```

This loads the cached LLaDA2.1-mini checkpoint and requires a supported GPU.
The test checks strict graph/eager equality for logits, decoder updates and
the real KV pool, including padded buckets, recycled physical pages and an
actual 65536-token KV prefix-plus-block. Changing the KV allocation must
invalidate the old graphs. This graph/eager check is separate from comparing
FA4 to a different attention backend.

## Attention semantics

For prefill, FluxServe converts every LLaDA query block into a virtual varlen
attention task whose KV length ends at that block. For denoising, it writes the
current block to paged KV first, then attends over the prefix plus the entire
current block with `causal=False`. This reproduces LLaDA's block-causal mask
without a model-specific kernel fork.

Run the metadata and adapter tests with:

```bash
pytest -q test/runtime/test_fa4_attention.py
```

On supported hardware, enable the real kernel parity test with:

```bash
FLUXSERVE_RUN_FA4_SMOKE=1 \
pytest -q test/runtime/integration/test_fa4_llada_attention.py
```
