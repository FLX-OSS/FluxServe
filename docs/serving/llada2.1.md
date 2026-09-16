# Serving LLaDA2.1

Complete the [Docker installation](../guides/getting_started.md) first. Run the examples from the FluxServe repository directory on a GPU with enough memory for the selected checkpoint.

LLaDA2.1 (`inclusionAI/LLaDA2.1-mini`, `inclusionAI/LLaDA2.1-flash`) shares the
LLaDA2.0 architecture and checkpoint layout and loads through the same
`LLaDA2LLM` model class with no overrides. What is new is the decoding
algorithm: joint Mask-to-Token (M2T) + Token-to-Token (T2T) updates, where the
model may *edit* an already-resolved token in the active block when its
confidence clears `editing_threshold`. FluxServe implements this as the
`joint_threshold` parallel decoder.

See the [model support development guide](../../dev-notes/llada2.1-model-support-development-guide.md) for the full
design, invariants, and test plan.

## Presets (from the LLaDA2.1 model card)

Both presets use `block_length=32`, `temperature=0` (the only supported
temperature), and `max_post_steps=16`.

| preset | `--threshold` | `--editing-threshold` |
| --- | --- | --- |
| Quality (Q Mode) | 0.7 | 0.5 |
| Speed (S Mode) | 0.5 | 0.0 |

`editing_threshold=0.0` is the official Speed setting: T2T repairs errors, so
the M2T threshold can drop to 0.5 and unmask more positions per step.

## Baseline (existing threshold decoder, no editing)

LLaDA2.1 checkpoints also run unmodified through the LLaDA2.0 `threshold`
decoder; this is the known-good reference before enabling editing:

```bash
python -m fluxserve.cli bench_offline \
  --model inclusionAI/LLaDA2.1-mini \
  --dataset ./data/sample.jsonl \
  --gen-len 256 --block-length 32 --page-size 32 \
  --parallel-decoding threshold --threshold 0.95
```

Measured on HumanEval (164 problems, gen 256/block 32): pass@1 0.720 with
this baseline vs 0.652 for LLaDA2.0-mini-CAP under the identical config.
Dropping the threshold to 0.5 *without* editing collapses accuracy to 0.329,
which is why the Speed preset requires the joint decoder.

## Online serving with joint_threshold (Quality preset)

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
  --scheduler-policy paged
```

For the Speed preset pass `--threshold 0.5 --editing-threshold 0.0`.

## FA4 with tensor parallelism

The FA4 runner accepts TP4/EP4 with `DP=PP=1`, including padded decode CUDA
graphs. Launch the four-GPU preset from an environment with standalone
`flash-attn-4` installed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash test/benchmark/fluxserve/configs/tp4_ep4_llada21_mini_fa4.sh
```

The launcher requires `TP=EP` (including TP1/EP1 and TP4/EP4). Standard
MoE dispatch (`moe_a2a_backend=none`) is supported; DeepEP graph capture is
unsupported. Q/K/V heads and the paged KV cache are local to each TP rank.
Routed experts are sharded across EP ranks and their outputs are summed;
replicated shared-expert outputs are added once after the sum. Model collectives
run inside the decode graph; fused decoder tokens and completion predicates
are synchronized after replay before advancing blocks or editing budgets.
Graph capture requires `page_size == block_length`, a page size divisible by
16, and batch buckets covering `max_num_seqs`. Prefill graphs remain unsupported.
The same TP path applies to LLaDA2.0 with its `threshold` or `hierarchy` decoder.

LLaDA2.1-mini completed GSM8K evaluation and a 1000-request performance run on
four H200 GPUs with TP4/EP4 and decode graphs. See the
[results and configuration](../../FA4_EVAL_PERF_RESULTS_20260915.md).
The LLaDA2.0-flash performance run was interrupted when its allocation expired.
On a four-GPU node, run the standard MoE EP eager/graph parity test with:

```bash
python -m pytest -q test/runtime/test_moe_tp_ep_cuda_graph.py
```

This test covers routed and shared experts, including their separate CUDA
streams; use the FA4 serving benchmark for full-model validation.

## Offline benchmark with joint_threshold

```bash
python -m fluxserve.cli bench_offline \
  --model inclusionAI/LLaDA2.1-mini \
  --dataset ./data/humaneval.jsonl \
  --gen-len 512 --block-length 32 --page-size 32 \
  --parallel-decoding joint_threshold \
  --threshold 0.7 --editing-threshold 0.5 --max-post-steps 16
```

## Semantics guaranteed by the implementation

- Only the active block is editable; committed blocks are immutable and their
  KV entries are reused.
- The prompt (including an unaligned prompt suffix inside the first
  generation block) is never edited.
- No update path can write `mask_id` back into the block (the mask logit is
  suppressed before the argmax).
- A block completes only on an iteration that applied no update, so the
  forward pass that commits the block's KV saw the block's final tokens.
- Per-row editing budget: after `max_post_steps` mask-free iterations that
  still changed the block, T2T is disabled for that row and the block
  finishes on the next pass. Rows in a batch finish independently.
- Streaming emits only stable blocks; intermediate edits are invisible to
  clients.
- LLaDA2.0 decoding (`--parallel-decoding threshold`) is unchanged;
  `joint_threshold` is opt-in.
