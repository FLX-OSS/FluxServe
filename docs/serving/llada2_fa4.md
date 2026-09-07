# LLaDA 2.x with the FA4 backend

FluxServe can run LLaDA 2.x attention through the standalone
`flash-attn-4` CuTeDSL package. The adapter consumes FluxServe's own request
metadata and paged KV cache directly; it does not use a compatibility layer
from another serving system.

This is an experimental backend: attention-level BF16 parity has passed on
GH200, but the strict full LLaDA2.1-mini model logits parity test has not passed.
Small attention differences and subsequent model amplification require further
investigation; the exact cause is not established by attention tests alone.
Online CUDA Graph execution is not wired into the FA4 runner yet.

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
  --model inclusionAI/LLaDA2.0-mini \
  --attention-backend fa4 \
  --kv-cache-layout paged \
  --scheduler-policy paged \
  --page-size 64 \
  --block-length 64
```

FA4 kernels are JIT-compiled for the first unseen shape, so the first request
can take longer. CUDA graph capture is disabled for this initial backend;
normal eager execution and shape-level CuTeDSL kernel caching remain enabled.

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
