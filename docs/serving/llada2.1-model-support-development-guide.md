# LLaDA2.1 Model Support Development Guide

## 0. Implementation Status (2026-08-29)

- **Phase 1: complete.** GPU smoke validated (job `llada21_real_3428096`):
  2.1-mini loads through `LLaDA2LLM` with zero unmatched weights and reaches
  HumanEval pass@1 0.720 through the existing `threshold` decoder (vs 0.652
  for 2.0-mini-CAP, same config). `load_decoder` now rejects unknown names
  and the broken `hierarchy` `RunnerConfig` default is `threshold`.
- **Phase 2: code + CPU tests complete, GPU validation pending.**
  `JointThresholdDecoder` (`decoders/joint_threshold.py`), `RunnerConfig`
  fields + CLI flags, both runners on the Section 6.1 predicate with the
  Section 8 budget (`DecodeEditBudget` in `runners/utils.py`), prompt
  protection plumbed online + offline. Unit tests:
  `test/test_joint_threshold_decoder.py`,
  `test/test_block_finished_equivalence.py`. GPU validation script ready:
  `tools/llada21_joint.sh` (submit via `tools/flux_joint_run.sbatch`) —
  covers the 2.0 bit-stability regression, both presets, and reference
  agreement. The reference trace harness runs under a transformers-5.2
  sandbox (`tools/tf52_sandbox`) because the checkpoint's modeling code
  needs `create_bidirectional_mask`.
- **Phase 2 GPU validation: complete** (dense/no-graph path, HumanEval-164):
  2.1-mini joint Quality 0.7/0.5 = 0.774, joint Speed 0.5/0.0 = 0.793
  (reference impl same presets: 0.690/0.675); 2.1-flash TP4/EP4 joint
  Quality 0.829, Speed 0.835. Zero unmatched weights on both checkpoints.
- **Phase 4 (CUDA graph) implemented, needs paged-GPU benchmark.** Review
  feedback on B200 (GSM8K, paged + decode graphs) measured 2.1-mini at
  1198 tok/s w/ graphs vs 1133 w/o, below 2.0-mini's 1335: the decode graph
  captured only the transformer forward, leaving the lm_head, the joint
  M2T/T2T selection, the block-finished predicate, and a per-iteration
  GPU->CPU sync (`DecodeEditBudget.update`'s stuck check, plus the branchy
  EOS early-stop) eager between replays. Fixes: the LLaDA2 decode graph now
  captures the full iteration tail for decoders declaring
  `graph_fused_step` (`joint_threshold_graph_step`; only the token-array
  scatter stays eager), the stuck check runs every `max_block_iters` calls
  instead of every call, and the EOS early-stop is branchless. CPU parity
  tests: `TestGraphStepParity` in `test/runtime/test_joint_threshold_decoder.py`.
  The `decode_fallback_count`/`decode_replay_count` counters verify graph
  coverage on hardware; re-benchmark on the B200 paged path.
- Phase 3 (batched/paged/distributed serving hardening): TP4/EP4 on flash
  validated; remaining items unstarted.

## 1. Goal

This guide describes how to add first-class LLaDA2.1 support to FluxServe while
preserving the existing LLaDA2.0 serving behavior.

LLaDA2.1 uses the same model architecture and checkpoint layout as LLaDA2.0,
but introduces token editing during block diffusion decoding. The adaptation is
therefore primarily a decoding and request-lifecycle change, rather than a new
Transformer or MoE implementation.

The guiding constraint is **maximum reuse of the architecture already validated
on LLaDA2.0**. Every section below is written to change as little as possible:
no new model class, no new decoder base-class contract, no scheduler change, no
new per-request state in the engine. Where an earlier draft of this guide
proposed a broader change, Section 17 records why it was withdrawn.

The completed implementation must support:

- `inclusionAI/LLaDA2.1-mini` and `inclusionAI/LLaDA2.1-flash`;
- Mask-to-Token (M2T) and Token-to-Token (T2T) updates;
- the official Speed and Quality presets through explicit decoding parameters;
- dense and paged KV cache execution;
- single-request and batched online serving;
- tensor/expert-parallel execution with bit-identical token state across ranks;
- stable-block streaming without retracting emitted tokens; and
- LLaDA2.0 regression compatibility.

## 2. Non-goals

The first implementation does not need to:

- add a separate `LLaDA2.1LLM` model class;
- change the `ParallelDecoder` base-class contract;
- add per-request editing state to `RequestState` or the C++ scheduler;
- edit tokens in blocks that have already been committed;
- expose unstable token edits through the streaming API;
- support `temperature > 0` for the new decoder (see Section 6.5); or
- optimize CUDA graphs before eager execution is correct.

## 3. Compatibility Assessment (verified)

This section is no longer a plan. The comparison has been run against the
checkpoints on the development machine and the results are recorded here.

Method: parse `config.json`, read the `safetensors` headers directly for every
shard (key set, shape, dtype), and read the token ids out of `tokenizer.json`.
No GPU and no model instantiation required.

### 3.1 Config

`inclusionAI/LLaDA2.0-mini` and `inclusionAI/LLaDA2.1-mini` have **identical
`config.json` files: 48 keys, zero differing values.** This includes
`max_position_embeddings=32768`, `sliding_window=4096`,
`use_sliding_window=false`, `vocab_size=157184`, `num_hidden_layers=20`,
`num_attention_heads=16`, `num_key_value_heads=4`, `hidden_size=2048`,
`num_experts=256`, `num_experts_per_tok=8`, `num_shared_experts=1`,
`first_k_dense_replace=1`, `rope_theta=600000`, `rotary_dim=64`, and
`pad_token_id=156892`.

Against the `LLaDA2.0-mini-CAP` variant the only differences are non-semantic:
the `torch_dtype` -> `dtype` rename in transformers 4.57, and
`transformers_version`.

Note: the `inclusionAI/LLaDA2.0-mini-preview` repository on the Hub carries
`max_position_embeddings=8192`. That is a different, earlier repository and is
not the 2.0 baseline used here. Do not compare against it.

### 3.2 Weights

Against `LLaDA2.0-mini-CAP` (the only 2.0 checkpoint with weights available
locally; the stock 2.0-mini snapshot holds config and modeling code only):

- 14813 tensors on both sides;
- zero keys present in only one side;
- zero shape mismatches on shared keys;
- one dtype difference: `model.layers.{1..19}.mlp.gate.expert_bias` is stored
  as `F32` in 2.1 and `BF16` in 2.0. (Layer 0 is dense because
  `first_k_dense_replace=1`, so it has no gate.)

**The dtype difference requires no code change.** `loader.py:424-428` renames
`model.layers.N.mlp.gate.expert_bias` to `model.layers.N.mlp.correction_bias`,
and `loader.py:412` forces every `.mlp.correction_bias` parameter to `float32`.
2.0 therefore loads bf16 and upcasts; 2.1 loads fp32 directly. Both end at
fp32 in the router.

That rename is load-bearing for a non-obvious reason worth recording, because
it looks like a bug on first reading. `LLaDA2SparseMoeBlock` aliases the same
`nn.Parameter` under two names (`llada2.py:314-316`,
`self.correction_bias = self.gate.expert_bias`). `named_parameters()`
de-duplicates shared parameters and keeps only the first traversal name, which
is `mlp.correction_bias`. So the checkpoint key `mlp.gate.expert_bias` would
never match `dict(self.named_parameters())` in `apply_state_dicts`
(`llada2.py:1099`). The loader rename is what makes it match. Do not "clean up"
either half without the other.

### 3.3 Tokenizer

| token | 2.1-mini | 2.0-mini-CAP | `RunnerConfig` default |
| --- | --- | --- | --- |
| `<\|mask\|>` | 156895 | 156895 | 156895 |
| `<\|endoftext\|>` | 156892 | 156892 | 156892 |

Base vocab (156891) plus added tokens (262) match exactly.

### 3.4 Consequences

LLaDA2.1 loads through the existing `LLaDA2LLM` with no architecture-specific
patch and no id overrides. Two clarifications about the loader, since an
earlier draft overstated both:

- `python/fluxserve/backend/configs/model_config.py` defines
  `class ModelConfig: pass` -- an empty placeholder. The real config comes from
  `AutoConfig.from_pretrained` at `cli.py:201`, i.e. from the checkpoint's own
  `configuration_llada2_moe.py`. There is no architecture registry and nothing
  to dispatch on; `models/__init__.py` exports only `LLaDA2LLM` and
  `loader.py:51` constructs it unconditionally.
- Making `mask_id` / `eos_id` configurable (`--mask-id` / `--eos-id`, derived
  from tokenizer metadata when present, current values as fallback) is still
  worth doing as hygiene, but Section 3.3 shows the hard-coded defaults at
  `forward_batch_info.py:135-136` are correct for 2.1. **This is not a Phase 1
  blocker.** Do it opportunistically.

The only remaining Phase 1 item that needs a GPU is the forward smoke test
(Section 11).

## 4. LLaDA2.1 Decoding Semantics

FluxServe currently resolves masked positions and treats a block as complete
when no masks remain. LLaDA2.1 adds a second update path that can replace an
already resolved token in the active block.

**The reference implementation is available locally.** The 2.1 checkpoint ships
its own decoding loop in `modeling_llada2_moe.py`, `LLaDA2MoeModelLM.generate()`
(the loop body is at lines 1320-1440 of the file in the
`inclusionAI/LLaDA2.1-mini` snapshot). Everything in this section is checked
against that code, not only against the paper. Two properties of the reference
matter for how much it can settle:

- it is **batch-size 1 only** (`x = torch.full((1, total_length), ...)` and `[0]`
  indexing throughout), so FluxServe's batching semantics are an extension with
  no upstream counterpart; and
- it uses **no KV cache** -- it re-runs the full window under a block-causal
  mask on every iteration. Every KV-timing question in Section 8.1 is therefore
  FluxServe's own problem, and the reference cannot arbitrate it.

From the paper, the two update sets at step `t` are:

```text
M2T:  G_t = { i | x_i = [MASK]  and  p(v_i) > tau_mask }
T2T:  D_t = { i | x_i != v_i    and  p(v_i) > tau_edit }
x_{t-1}^i = v_i  if i in G_t union D_t, else unchanged
```

where `v_i` is the model's top candidate at position `i`.

For every active block iteration:

1. Run the model over the current block and its committed prefix.
2. Compute the candidate token `x0` and its probability `x0_p` at every active
   block position.
3. Apply M2T updates to masked positions using `threshold`.
4. Guarantee progress: at least one masked position per row is always
   transferred, so the mask count is strictly decreasing while masks remain.
   This applies per row, and only to rows that still contain masks. Section 6.2
   shows this already falls out of the existing 2.0 code.
5. Apply T2T updates to non-mask, non-prompt positions when the candidate token
   differs from the current token and `x0_p > editing_threshold`.
6. Repeat until the block is stable or the `max_post_steps` budget is spent.

A block is stable when it contains no mask tokens and an iteration performs no
update at all.

**No-remask invariant, and a deliberate deviation.** Neither update path may
write `mask_id` into the block. Once a position is resolved it stays resolved;
only its value may change. This is what makes "no masks remain" a monotone
property and keeps the stability predicate meaningful.

The paper states this property. **The reference code does not enforce it.**
`generate()` never compares the candidate against `mask_id`; if the argmax at an
already-resolved, editable position were `mask_id` with probability above
`editing_threshold`, the reference would write a mask back into the block. In
practice the model presumably never does this, which is why the omission has
gone unnoticed upstream.

FluxServe should enforce the invariant anyway (Section 6.2) -- the existing 2.0
decoder already does the equivalent for M2T via `rm_mask`
(`threshold.py:60-61`), and a runner that commits KV per block cannot tolerate a
non-monotone mask set. **Record this as an intentional divergence from the
reference**, and have the parity test assert that the guard never actually
fires; if it does fire on real inputs, the divergence becomes a real behavioral
difference that has to be resolved with upstream rather than papered over.

**Scope note (resolved).** `D_t` as written in the paper does not explicitly
exclude masked positions, which read literally would make `tau_mask` inert at
`tau_edit = 0`. The reference settles it -- T2T is restricted to positions that
are neither masked nor prompt:

```python
non_mask_positions   = ~active_block_mask
non_prompt_positions = ~prompt_mask_in_block
editable_positions   = non_mask_positions & non_prompt_positions[None, :]
```

Use that reading.

Only the active block is editable. Previously completed blocks form an
immutable prefix and remain eligible for KV-cache reuse.

## 5. Configuration and CLI

`RunnerConfig` already carries `parallel_decoding`, `threshold`, `low_threshold`,
`use_credit`, `mask_id`, and `eos_id`
(`python/fluxserve/backend/execution/forward_batch_info.py:131-136`). Only two
fields are new:

```python
editing_threshold: float = 0.5
max_post_steps: int = 16
num_to_transfer: int = 1
```

`num_to_transfer` **is** an upstream parameter (`generate()` signature, default
`1`). Its semantics: transfer every masked position above `threshold`; if fewer
than `num_to_transfer` qualify, take the top-`min(num_to_transfer, num_masked)`
by confidence instead. At the default of `1` this is exactly what the existing
2.0 decoder already does (Section 6.2), so the field can be added as
configuration without any new selection code, and only needs a real
implementation if a preset ever moves it above 1.

Register a new decoding name, `joint_threshold`. Do not change the semantics of
the existing `threshold` decoder, because existing LLaDA2.0 commands and
benchmarks depend on it.

### 5.1 Official presets

From the LLaDA2.1-mini model card. Both presets share `block_length=32`,
`temperature=0.0`, `top_p=None`, `top_k=None`, and `max_post_steps=16`
(the card gives 5 as the minimum useful value).

| preset | `--threshold` | `--editing-threshold` |
| --- | --- | --- |
| Quality (Q Mode) | 0.7 | 0.5 |
| Speed (S Mode) | 0.5 | 0.0 |

**`editing_threshold = 0.0` is the official Speed Mode setting, not a debug
knob.** An earlier draft of this guide claimed the opposite; see Section 17.
At `0.0` the T2T rule rewrites every non-mask, non-prompt position whose argmax
differs from the current token, on every iteration. That is more forwards per
block, and it is the intended trade: because T2T repairs errors, `tau_mask` can
be dropped to 0.5 so that many more positions unmask per step. The net effect
claimed by the paper is *fewer* total forwards, not more. Measure both halves
(Section 13) rather than reasoning about the editing cost alone.

**Do not take defaults from the reference signature.** Three upstream sources
disagree: `generate()`'s signature says `threshold=0.95, editing_threshold=0.9`;
its own docstring says `editing_threshold` "defaults to 0.5"; the model card
gives the two presets above. The signature defaults are stale. Use the model
card.

Do not embed the presets in the algorithm. Ship them as documented presets that
can be retuned as upstream recommendations evolve.

### 5.2 CLI

Add options to **both** entry points that build a `RunnerConfig`:

```text
--parallel-decoding joint_threshold
--threshold FLOAT
--editing-threshold FLOAT
--max-post-steps INT
```

- `python/fluxserve/cli.py` (`serve`), around the existing
  `--parallel-decoding` / `--threshold` flags at `cli.py:129-131` and the
  `RunnerConfig(...)` construction at `cli.py:314-316`.
- `python/fluxserve/bench_offline.py`, which parses its own
  `--parallel-decoding` (`bench_offline.py:651`) and builds its own
  `RunnerConfig` (`bench_offline.py:271`). Without this, the Phase 4 offline
  benchmarks cannot exercise the new algorithm at all.

Validate at startup: `threshold` and `editing_threshold` in `[0, 1]`,
`max_post_steps` non-negative. `RunnerConfig.threshold` keeps its current
default of `0.9` so 2.0 paths are untouched; `joint_threshold` runs should
always pass `--threshold` explicitly, since neither preset uses `0.9`.

### 5.3 Reject unknown decoder names

`load_decoder` (`decoders/factory.py:28-56`) currently falls through to
`HierarchyDecoder` for any unrecognized `parallel_decoding` value. Convert the
fallback into an explicit dispatch table that raises on unknown names. Do this
in Phase 1.

Two corrections to how this is usually described:

- The failure today is **not silent**. `HierarchyDecoder` and
  `StaticParallelDecoder` do not implement `batch_decode`, and both runners
  call only `batch_decode` (`block_diffusion.py:409`,
  `flashinfer_diffusion.py:1179`). A typo or an unimplemented name therefore
  raises `AttributeError` deep in the decode loop. The dispatch table turns a
  confusing late crash into a clear startup error, which is still worth doing.
- For the same reason, `RunnerConfig.parallel_decoding = "hierarchy"` is a
  **broken default** in the current runners; only `cli.py`'s own default of
  `"threshold"` hides it. Either give `HierarchyDecoder` a `batch_decode`, or
  change the `RunnerConfig` default to `"threshold"`. Do not preserve
  `hierarchy` as the fallback default on the grounds of compatibility -- there
  is nothing working to be compatible with.

Note also that `ParallelDecoder.decode()` (`base.py:38`) is dead code with
respect to both runners. Do not extend it.

### 5.4 Suggested correctness-first command

```bash
CUDA_VISIBLE_DEVICES=0 python -m fluxserve.cli serve \
  --model inclusionAI/LLaDA2.1-mini \
  --host 127.0.0.1 \
  --port 8000 \
  --tp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
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

## 6. Decoder Design

Add:

```text
python/fluxserve/backend/execution/decoders/joint_threshold.py
```

Register it in `decoders/factory.py` and export it from `decoders/__init__.py`.

### 6.1 Interface contract: unchanged

**Do not change `ParallelDecoder`.** `JointThresholdDecoder` implements
`batch_decode(logits, block_start, x, block_length, ...)` with the same
signature and the same "mutate `x` in place, return `None`" contract as
`ThresholdParallelDecoder` (`threshold.py:113-145`). No `batch_decode_ex`, no
result dataclass, no base-class default implementation.

The completion signal stays in the runner, where it already lives. Today both
runners infer it from the **pre-update** gather:

```python
block_finished = (decoding_block == self.decoder.mask_id).sum(dim=1) == 0
```

(`block_diffusion.py:413`, `flashinfer_diffusion.py:1183`, both using the
`decoding_block` gathered *before* `batch_decode` mutated the tokens.)

Replace it, in both runners, with:

```python
after    = gather_blocks(decoding_x.data, decoding_start[seq_ids], self.block_length)
had_mask = (decoding_block == self.decoder.mask_id).any(dim=1)
changed  = (after != decoding_block).any(dim=1)
block_finished = (~had_mask) & (~changed)
```

This one expression serves both model versions:

- **For LLaDA2.0 it is exactly equivalent to the current test.** The 2.0
  decoders apply `transfer_index & mask_index` (`threshold.py:108`, `:137`), so
  a block with no masks cannot change: `~had_mask` implies `~changed`. Zero
  behavioral difference, zero regression risk.
- **For LLaDA2.1 it is exactly the reference's stability predicate.**
  `generate()` snapshots `old_block_tokens` and computes `active_block_mask`
  from the *pre-update* block, applies the transfers, and then breaks on
  `active_block_mask.sum() == 0 and not editing_transfer_index.any()`. When the
  pre-update block has no masks, `mask_transfer_index` is empty, so
  `editing_transfer_index.any()` and `changed` are the same condition. The
  expression above is a per-row vectorization of that break.
- **It preserves KV write timing for free.** See Section 8.1.
- **It is rank-consistent for free.** `after` is read from `x.data` *after* the
  decoder's `broadcast_if_needed`, so every rank compares identical tensors.
  See Section 9.2.

Cost is one extra `gather_blocks` per iteration, which is negligible next to a
model forward.

### 6.2 Selection logic

Reuse `get_transfer_index_threshold` (`threshold.py:34-69`) rather than writing
a new minimum-top-k helper. At `num_to_transfer = 1` its `actual_threshold` line
is equivalent to the reference's two-branch M2T selection, and already provides
everything the M2T path needs:

```python
actual_threshold = (torch.max(confidence, dim=1)[0] - 1e-5).clamp(-1000, threshold).unsqueeze(-1)
transfer_index = confidence >= actual_threshold
```

- it transfers at least the single highest-confidence masked position per row,
  which is the progress guarantee of Section 4 step 4;
- it is automatically a no-op for a row with no masked positions, because that
  row's `confidence` is all `-inf` and `-inf >= -1000` is false. A finished row
  is never forced to update because another row still has masks;
- the `rm_mask` guard at `threshold.py:60-61` already excludes candidates equal
  to `mask_id`.

One bit-exactness detail for the parity test: the reference uses a strict
`confidence > threshold` while `get_transfer_index_threshold` uses
`confidence >= actual_threshold`. The two differ only on an exact tie at the
threshold value. Decide whether to match the reference exactly or to document
the difference, but do not discover it during trace comparison.

The new decoder is then the same function plus one extra term:

```python
# Structural no-remask: mask_id can never be selected as a candidate anywhere.
logits[..., mask_id] = -float("inf")

x0   = torch.argmax(logits, dim=-1)
x0_p = F.softmax(logits.to(torch.float32), dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)

mask_index   = x_block == mask_id
confidence   = torch.where(mask_index, x0_p, -np.inf)
thr          = (confidence.max(dim=1)[0] - 1e-5).clamp(-1000, threshold).unsqueeze(-1)
m2t_transfer = confidence >= thr

editable     = ~mask_index & ~prompt_positions & allow_edit[:, None]
t2t_transfer = editable & (x0 != x_block) & (x0_p > editing_threshold)

x_block = torch.where(m2t_transfer | t2t_transfer, x0, x_block)
```

Suppressing the `mask_id` logit before the argmax makes the no-remask invariant
**structural** instead of a condition that has to be repeated on both update
paths. It also removes an entire failure mode: with the mask logit suppressed,
a row can never stall because every masked position's argmax happened to be
`mask_id`, so the progress guarantee in Section 4 step 4 is unconditional and
the iteration cap in Section 8.1 is a safety net rather than a live code path.

**This line is a deviation from the reference, which has no such guard**
(Section 4). Keep it, and make the parity test assert it never changes the
selected candidate -- see Section 12.

`allow_edit` is the per-row post-edit budget flag from Section 8. Everything
here is a pure tensor function of `(logits, x_block, prompt_positions,
allow_edit)` and should live in a module-level helper so it can be unit-tested
without a model or a GPU.

### 6.3 Broadcast and rank consistency

Every existing decoder ends with `broadcast_if_needed(x.data)`
(`threshold.py:111`, `:145`, `hierarchy.py:129`; implementation at
`decoders/utils.py:62`). The new decoder must do the same, before returning.

With the Section 6.1 formulation this is the *only* thing needed for rank
consistency, because every control-flow predicate is derived from `x.data`
after the broadcast. See Section 9.2.

### 6.4 Prompt positions

The decoder takes prompt protection as an explicit input, not as a heuristic;
see Section 7.

### 6.5 Temperature

The pseudocode above uses a plain `argmax`, unlike the existing decoders, which
route through `add_gumbel_noise` (`decoders/utils.py:28`). Both official presets
specify `temperature=0.0`. The first implementation targets `temperature=0`
only, and the decoder must raise if constructed with a non-zero temperature
rather than silently ignoring it. Sampling support can reuse `add_gumbel_noise`
later, but note that T2T with sampling changes the stability argument: a
resampled token can differ every iteration, so termination would rely entirely
on `max_post_steps`.

## 7. Prompt Protection

The first generation block may contain an unaligned prompt suffix. Non-mask
tokens alone cannot be used to identify editable positions, because prompt
tokens are also non-mask tokens.

The online path makes this concrete: `_execute_paged_decode` computes
`generated_start = max(block_start, len(state.input_ids))`
(`runners/flashinfer_diffusion.py:525`), i.e. the first aligned block starts
inside the prompt whenever `len(input_ids)` is not a multiple of
`block_length`. Those prompt tokens sit in the decode buffer as ordinary
non-mask tokens and would be eligible for T2T editing.

For each request, construct the prompt mask from absolute positions:

```python
positions = block_starts[:, None] + torch.arange(block_length, device=device)
prompt_positions = positions < prompt_lengths[:, None]
```

`prompt_lengths` differs by execution path, and both are already available:

- **Online** (`_execute_paged_decode`): `len(state.input_ids)` per row. Use
  `input_ids`, not `len(state.input_ids) + len(state.output_ids)` -- see the
  note below.
- **Offline** (`BlockDiffusionRunner.generate`): `non_mask_number`, i.e.
  `(prompts != mask_id).sum(dim=-1)` (`runners/block_diffusion.py:250`).
  Offline prompts are right-padded with `mask_id` by `pad_batch`
  (`bench_offline.py:286`), so the padded tail is generation region, not
  prompt, and `non_mask_number` is exactly the prompt boundary. The formula
  above is only valid for right padding; if a left-padding path is ever added,
  prompt positions must be recomputed accordingly.

Pass `prompt_lengths` or the precomputed `prompt_positions` explicitly to the
decoder. Prompt protection must not depend on token values or on a special-token
heuristic.

**Pre-existing quirk to be aware of.** `_execute_paged_decode` filters `mask_id`
and `eos_id` out of the tokens it appends to `output_ids`
(`flashinfer_diffusion.py:531-540`), while the next step rebuilds the decode
buffer as `state.input_ids + state.output_ids`. When anything is filtered,
`len(tokens)` drifts below `block_start`, leaving a `mask_id` gap in the middle
of an already-committed prefix. This is 2.0 behavior and is out of scope here,
but it is why `prompt_lengths` must come from `input_ids` alone, and it is
worth a separate issue.

## 8. Runner and Block Lifecycle

Update both the standard block diffusion runner and the FlashInfer diffusion
runner. Per-request editing state is a **local tensor inside the decode loop**,
indexed by `seq_id`:

```python
post_steps = torch.zeros(max_num_seqs, dtype=torch.long, device=device)
allow_edit = post_steps < max_post_steps          # recomputed each iteration
```

State transitions, using the Section 6.1 predicates:

```text
had_mask                          -> post_steps unchanged, block stays active
~had_mask and changed             -> post_steps += 1, block stays active
~had_mask and ~changed            -> block_finished
post_steps == max_post_steps      -> allow_edit = False for that row
                                     (the NEXT iteration then has ~changed
                                      and finishes the block naturally)
```

**Do not mark a block complete at the moment the budget runs out.** The
iteration that exhausts the budget is by definition an iteration in which an
edit happened, so its forward ran on the *pre-edit* tokens. Committing there
would write a KV entry that does not match the tokens the block ends up with,
and every later block would attend to a stale prefix. Clearing `allow_edit`
instead costs exactly one extra forward and keeps the invariant of Section 8.1
intact.

**Relation to the reference counter.** `generate()` increments `post_steps` at
the *top* of an iteration whose block already has no masks, and breaks on
`post_steps > max_post_steps` **before** running that iteration's forward:

```python
active_block_mask = cur_x[:, -block_length:] == mask_id
if torch.any(active_block_mask) == False:
    post_steps += 1
if post_steps > max_post_steps:
    break
```

So upstream executes at most `max_post_steps` mask-free forwards, and the
stable-detecting pass is counted inside that budget rather than added to it.
The scheme above produces the **same final tokens** -- the reference stops after
the last applied edit, and the `allow_edit=False` pass applies nothing -- while
additionally leaving FluxServe's committed KV consistent with those tokens.
That extra pass is the only reason the bound in Section 8.1 carries a `+1`.

Nothing here belongs in `RequestState`. The whole block loop runs inside a
single `_execute_paged_decode` call, so the state is naturally scoped to that
call -- which is also what structurally guarantees the streaming contract of
Section 10.

**Watch out for batch decomposition.** `_decode_selected_batch` recursively
splits a batch into CUDA-graph-shaped parts and calls itself per part
(`flashinfer_diffusion.py:1100-1124`). Any per-row state must be indexed by
`seq_id`, never by position within the current sub-batch.

For a batched decode, completed rows leave the active batch while other rows
continue. The paged decode loop already does this -- it shrinks
`pending_seq_ids` each iteration (`flashinfer_diffusion.py:494-511`) by
comparing `block_start_tensor` before and after -- and since `block_finished`
is what advances `block_start_tensor`, that filter keeps working unchanged.

### 8.1 The existing one-iteration lag, and where KV is written

Do not model the current loop as "fill the last mask, block is done". The
existing completion test uses `decoding_block`, which was gathered **before**
`batch_decode` mutated the tokens (`flashinfer_diffusion.py:1126` gather,
`:1179` decode, `:1183` test). So after the final mask is filled, the runner
performs one *additional* forward pass, and it is that pass which:

- observes a fully resolved block,
- sets `block_finished`,
- writes the block's KV into the cache via `_update_finished_kv_cache`
  (`flashinfer_diffusion.py:1185`, `block_diffusion.py:416-423`), and
- advances `decoding_start` / `current_decode_block`.

That trailing pass is precisely where the LLaDA2.1 post-edit iterations belong,
and it is why the current bound is `block_length + 1`.

The real invariant behind all of this is:

> **The forward pass that commits a block's KV must be the pass whose input
> tokens are the block's final tokens.**

Both cache paths depend on it. In the dense path `_update_finished_kv_cache`
writes `output.past_key_values`, produced from `decoding_block` -- the
pre-update gather. In the paged FlashInfer path the attention kernel writes K/V
into the page slots during the forward itself, again from the pre-update
tokens; `_update_finished_kv_cache` is skipped entirely when a CUDA graph was
replayed, because `output is None`.

The Section 6.1 predicate satisfies the invariant by construction: it only
declares a block finished on an iteration where nothing changed, so pre-update
tokens == post-update tokens == final tokens.

This is exactly why the completion test must **not** be moved to the
post-update token state. Deriving `block_done` from the updated block -- as an
earlier draft of this guide proposed -- fires one iteration earlier, on a pass
whose input still contained masks, and commits KV for a block that was not yet
resolved. That is a silent correctness regression for LLaDA2.0, not a
refactor. See Section 17.

Iteration accounting:

```python
max_decode_iters = block_length + max_post_steps + 1
```

`block_length` iterations drain the masks (at least one M2T transfer per
iteration, Section 6.2), `max_post_steps` bounds the editing tail, and `+1` is
the `allow_edit=False` pass described in Section 8. The reference's own bound is
`block_length + max_post_steps`; the extra pass is FluxServe's, and exists to
keep KV consistent, not to change tokens. Keep the existing
`raise RuntimeError("paged paged decode block did not finish.")` behavior
(`flashinfer_diffusion.py:513`) and include the request id and post-step count
in the message.

### 8.2 The offline runner needs the same bound

`BlockDiffusionRunner._decode_batches` is a `while torch.any(decoding_flag)`
loop with **no iteration cap** (`block_diffusion.py:367`), unlike the paged
path. Under 2.0 it terminates because masks are strictly consumed. Under 2.1
the editing tail must be bounded by `max_post_steps` there too, or a
pathological input can spin forever. Add the same per-row budget and a loop
guard.

### 8.3 Batch selection assumes mask count is remaining work

`select_batch_sequences_by_mask_number` (`runners/utils.py`) picks the offline
mini-batch by sorting candidates on **descending mask count**. Under 2.1 a row
in its post-edit phase has zero masks but still needs forwards, so it sorts
last and can be starved while other rows still hold masks. It is not a
deadlock -- the row is still in `decoding_flag` and gets picked once the
others drain -- but it is a latency pathology that did not exist in 2.0.
Address it in Phase 3, e.g. by ranking on "rows that are not yet
`block_finished`" rather than on raw mask count.

### 8.4 Commit ordering

Do not increment `current_decode_block`, commit prefix-cache state, or return
block tokens to the engine until `block_finished` is true. In the online path
this is structurally satisfied -- the whole block loop runs inside one
scheduler step -- but the invariant must be preserved if that ever changes.

## 9. KV Cache, Distributed Execution, and Scheduler Requirements

### 9.1 KV cache

The active block is recomputed after every edit, so its KV entries must be
overwritten at the same logical positions. Historical block KV entries remain
unchanged.

For the paged FlashInfer path this already holds and needs confirmation, not
design work: `_make_decode_forward_batch` builds
`positions = decoding_start[seq_ids] + arange(block_length)` and derives
`flashinfer_slot_mapping` from it (`flashinfer_diffusion.py:958-966`), so the
attention kernel writes each active-block position into the same page slot on
every forward. Extra editing iterations overwrite; they do not append.

Verify the following for dense and paged layouts:

- repeated active-block forwards overwrite rather than append KV entries;
- page tables remain stable during post-edit iterations (no new pages are
  allocated per editing iteration);
- the block's KV is committed exactly once, on the stable iteration
  (Section 8.1); and
- scheduler token accounting does not count editing iterations as newly emitted
  output tokens (`reserve_tokens` stays `block_length` per block, not per
  iteration).

Prefix caching is currently disabled unconditionally in the paged scheduler
adapter (`cfg.disable_prefix_cache = True`,
`engine/scheduler_adapter.py:145`), so "an active block must not enter the
prefix cache before it is stable" and "retraction must not expose an unstable
block as a reusable prefix" are not testable today. Record them as preconditions
for enabling prefix caching rather than as Phase 3 exit criteria.

No scheduler API change is required. The block lifecycle is fully expressible
in the Python runner, and the engine-facing contract (`reserve_tokens`,
`decode_block_completed`) is unchanged by editing.

### 9.2 Distributed execution

LLaDA2.0-flash is served at TP=4/EP=4 (`docs/serving/llada2-flash.md`), so this
is a primary path, not an edge case.

Two distinct requirements:

1. **Token state.** `x` must be bit-identical on every rank after each decoding
   iteration. The existing decoders guarantee this with
   `broadcast_if_needed(x.data)`; the new decoder must too (Section 6.3).
2. **Control flow.** `block_finished`, `changed`, and `post_steps` decide
   whether another forward pass runs. Every rank must reach the same decision
   on the same iteration, otherwise ranks disagree on how many forwards to
   execute and the MoE all-to-all / all-reduce collectives hang.

Requirement 2 is satisfied automatically by the Section 6.1 formulation, and
this is a large part of why it is preferred: `decoding_block` is a pre-decode
gather of already-broadcast state, and `after` is re-gathered from `x.data`
after the decoder's broadcast. Both operands are identical on every rank, so
`had_mask`, `changed`, and everything derived from them are identical too. No
predicate is ever computed from rank-local logits or from a `torch.any` over a
sharded tensor, and no extra collective is needed.

Add an assertion (debug-only, behind a flag) that hashes the per-row
`(block_finished, post_steps)` vector across ranks each iteration, so any future
divergence fails loudly instead of hanging.

## 10. Output and Streaming Semantics

FluxServe should emit only stable blocks. Internal token changes are not output
events and must not append to `RequestState.output_ids`.

This is mostly a property to *preserve* rather than build: the online path runs
the entire block loop inside one `_execute_paged_decode` call and returns one
`ForwardStepResult` per completed block
(`flashinfer_diffusion.py:493-548`), and `async_llm.py:293` advances
`current_decode_block` only when `decode_block_completed` is set. The editing
iterations must stay inside that call.

The resulting contract:

```text
active block: mutable and invisible to the client
stable block: appended once and immutable
committed prefix: reusable by KV/prefix cache
```

EOS handling must occur after block stabilization. If EOS appears temporarily
during editing, it must not terminate the request until the block is stable.
After stabilization, truncate at the first EOS according to the existing
`ignore_eos` behavior (`flashinfer_diffusion.py:531-540`). Note that the
`early_stop` paths in both runners scan the whole sequence for EOS
(`block_diffusion.py:428-437`, `flashinfer_diffusion.py:1197-1201`); both
already gate on `block_finished`, so they inherit the corrected predicate
without further change.

## 11. Implementation Phases

### Phase 1: Plumbing

Checkpoint compatibility is already verified (Section 3) and needs no code.
What remains:

- Forward-pass smoke test on 2.1-mini: load through `LLaDA2LLM`, confirm no
  unexpected/missing weights are reported by `apply_state_dicts`, and confirm
  logits have vocabulary dimension 157184. *(GPU required.)*
- Make `load_decoder` raise on unknown decoding names, and fix the broken
  `hierarchy` default (Section 5.3).
- Document a 2.1 baseline command using the existing `threshold` decoder, so
  there is a known-good reference before the new algorithm exists.
- Optionally, expose `--mask-id` / `--eos-id` (Section 3.4; not a blocker).

Exit criterion: LLaDA2.1 generates coherent text through the *existing*
`threshold` decoder, and an unknown `--parallel-decoding` value fails at
startup.

### Phase 2: Reference JointThreshold Decoder

- Add the two `RunnerConfig` fields, the CLI flags, and factory registration.
- Implement the selection helper as a pure, testable PyTorch function
  (Section 6.2), including the `mask_id` logit suppression and the broadcast.
- Protect prompt positions (Section 7).
- Switch both runners to the Section 6.1 `block_finished` expression and add
  the `post_steps` / `allow_edit` budget (Section 8).
- Initially target eager execution, `max_num_seqs=1`, TP=1.

Exit criteria: LLaDA2.0 output is bit-identical before and after the runner
change; and per-iteration token trajectories match the official reference for
deterministic inputs.

### Phase 3: Batched, Paged, and Distributed Serving

- Support different active block offsets in one batch.
- Remove completed rows independently; fix the batch-selection starvation of
  Section 8.3.
- Bound the offline decode loop (Section 8.2).
- Verify paged KV overwrite and block commit timing.
- Run TP=4/EP=4 on flash and verify rank-consistent control flow.
- Add online serving and stable-block streaming tests.

Exit criterion: batched output matches running the same requests independently,
and TP>1 output matches TP=1 output.

### Phase 4: CUDA Graph and Performance

- Editing iterations reuse the same `(batch_size, q_len, kv_len)` shapes as
  ordinary decode iterations, so existing captures should replay unchanged;
  confirm rather than assume.
- Ensure post-edit iterations reuse captured graphs (watch
  `decode_fallback_count`).
- Benchmark mini and flash against the Section 13 targets.
- Ship the Speed and Quality presets with measured numbers attached.

Exit criterion: eager and CUDA-graph outputs match, the measured cost stays
within the Section 13 budget, and no existing LLaDA2.0 benchmark regresses.

## 12. Test Plan

### Unit tests

Against the pure selection helper (no model, no GPU):

- high-confidence M2T updates;
- guaranteed progress when no mask exceeds `threshold` (the `clamp` path);
- T2T replacement above `editing_threshold`;
- unchanged candidates not being counted as edits;
- **no path writes `mask_id` into any position**, verified by feeding logits
  whose unsuppressed argmax *is* `mask_id`;
- prompt positions never being edited, for aligned and unaligned prompts;
- independent completion of batch rows;
- a row with no masks not being forced to update by a row that still has masks;
- `allow_edit=False` disabling T2T while leaving M2T intact;
- deterministic behavior at temperature zero, plus a raise at temperature > 0.

### Runner-level tests

- **Equivalence of the new `block_finished` expression** with the old one for
  `ThresholdParallelDecoder` and `CreditThresholdParallelDecoder`, over
  randomized block states. This is the guard on the Section 6.1 claim and the
  single most important regression test in this change.
- Block stabilization after masks disappear.
- Budget exhaustion clearing `allow_edit` and finishing on the following
  iteration, with the committed KV matching the final tokens.

### Reference parity tests

The reference is `LLaDA2MoeModelLM.generate()` in the checkpoint's own
`modeling_llada2_moe.py` (loop body at lines 1320-1440). It is importable
through `trust_remote_code=True`, so traces can be produced on the same machine
and the same weights -- no upstream repository checkout is needed.

With deterministic decoding, compare FluxServe against it after every iteration,
not only after text decoding. Compare candidate tokens, candidate probabilities
within a documented tolerance, M2T transfer masks, T2T transfer masks,
post-step counters, and final token IDs.

Constraints inherited from the reference:

- **`batch_size=1` only.** The reference indexes `[0]` throughout. Batched
  behavior has no upstream oracle and must be validated instead by the
  batch-versus-independent equivalence test in Phase 3.
- **The reference has no KV cache.** It recomputes the full window each
  iteration, so it can validate the token trajectory but says nothing about
  FluxServe's KV commit timing. Cover that with the runner-level tests above.
- **Assert the two known deviations never bind:** that suppressing the `mask_id`
  logit never changes the selected candidate, and that no `>=` versus `>`
  tie-break at `threshold` ever changes a transfer decision. If either fires,
  it is a real behavioral difference, not a rounding artifact.

Start with both official presets at `batch_size=1`, `temperature=0`,
`block_length=32`, `max_post_steps=16`:

```text
Quality: threshold=0.7  editing_threshold=0.5
Speed:   threshold=0.5  editing_threshold=0.0
```

Pin the checkpoint revision used to produce the reference trace (the local
snapshot is `20e64e2ad21644d0e5248586ed9c942cdd45de0f`), and store the trace as
a fixture so the test does not need to load a 16B model at run time.

### Integration tests

Cover: mini checkpoint loading and generation; flash under its supported TP/EP
configuration; TP=1 versus TP=4 token equality; dense versus paged KV cache;
eager versus CUDA graph; one request versus multiple concurrent requests;
aligned and unaligned prompt lengths; EOS appearing and disappearing during
editing; streaming and non-streaming output equality; and request
cancellation during an active block.

Place them alongside the existing flat suite (`test/test_*.py`), following
`test/test_block_diffusion_offline.py` and
`test/test_cuda_graph_online_lifecycle.py`.

### Regression tests

Run the existing LLaDA2.0 mini and flash tests with `threshold` (and with
`hierarchy` once it has a `batch_decode`). The new algorithm must be opt-in,
and existing commands must retain their prior output and completion semantics.

## 13. Acceptance Criteria

LLaDA2.1 support is complete when:

- both official 2.1 checkpoints load through the shared LLaDA2 model class;
- JointThreshold implements M2T and T2T behavior with prompt protection and the
  no-remask invariant;
- active blocks remain private until stable;
- post-edit iterations terminate independently per request, and budget
  exhaustion never commits KV that disagrees with the emitted tokens;
- reference parity tests pass for both official presets;
- dense and paged serving produce equivalent token IDs;
- TP=1 and TP=4 produce equivalent token IDs, with no rank-divergence
  assertions triggered;
- streaming and non-streaming APIs produce equivalent final text;
- eager and CUDA-graph paths agree;
- LLaDA2.0 output is unchanged, bit-for-bit, by the runner modification; and
- serving commands and both presets are documented.

### Quantitative targets

Token editing trades forwards for quality, and in Speed Mode it is also
supposed to *reduce* total forwards by allowing a lower `tau_mask`. Measure
both directions, using the existing harnesses (`python/fluxserve/bench.py`,
`python/fluxserve/bench_offline.py`, `test/benchmark/`):

| Metric | Target | How measured |
| --- | --- | --- |
| Mean forwards per block | reported per preset; Speed Mode should be *lower* than 2.0 at matched quality | runner `num_forwards` / blocks |
| Post-step distribution | p50/p95 reported; p95 well under `max_post_steps` | `post_steps` histogram |
| Online TPOT vs 2.0 same config | regression stated and justified, not silently accepted | `bench.py` |
| Offline throughput vs 2.0 | same | `bench_offline.py` |
| Task accuracy vs 2.0 | Quality Mode must improve | existing GSM8K eval |
| CUDA-graph decode fallback rate | unchanged vs 2.0 | `decode_fallback_count` |

Publish the accuracy-versus-throughput tradeoff for both presets, on mini and
flash. A configuration that costs throughput without improving accuracy should
not ship as a preset.

## 14. Expected Files to Change

```text
python/fluxserve/cli.py
python/fluxserve/bench_offline.py
python/fluxserve/backend/execution/forward_batch_info.py
python/fluxserve/backend/execution/decoders/__init__.py
python/fluxserve/backend/execution/decoders/factory.py
python/fluxserve/backend/execution/decoders/joint_threshold.py
python/fluxserve/backend/execution/runners/block_diffusion.py
python/fluxserve/backend/execution/runners/flashinfer_diffusion.py
python/fluxserve/backend/execution/runners/utils.py        # Section 8.3
test/test_joint_threshold_decoder.py
test/test_block_finished_equivalence.py
test/test_llada21_online.py
docs/serving/llada2.1.md
```

Explicitly **not** changed, and why:

- `backend/execution/decoders/base.py` -- the decoder contract is unchanged
  (Section 6.1).
- `backend/engine/request.py` -- editing state is loop-local (Section 8).
- `backend/utils/server_args.py` -- it carries no decoding fields, and the only
  `load_decoder` call site (`runners/base.py:160`) passes a `RunnerConfig`.
- `backend/models/llada2.py` and the C++ scheduler -- changes here should be
  driven by a demonstrated incompatibility, and Section 3 found none.

## 15. References

- [LLaDA2.1-mini model card](https://huggingface.co/inclusionAI/LLaDA2.1-mini)
- [LLaDA2.1-flash model card](https://huggingface.co/inclusionAI/LLaDA2.1-flash)
- [LLaDA2.1: Speeding Up Text Diffusion via Token Editing (arXiv:2602.08676)](https://arxiv.org/abs/2602.08676)
- [Official LLaDA2.X repository](https://github.com/inclusionAI/LLaDA2.X)
- SGLang implements the same algorithm under the name `JointThreshold`
  (added in v0.5.9); useful as a second reference implementation.

## 16. Open Questions

Resolved, with the source that settled each:

- ~~Official parameter values~~ -- model card, Section 5.1.
- ~~Upstream remask policy~~ -- the paper claims monotonicity; the reference
  code does **not** enforce it. FluxServe deviates deliberately; see Section 4.
- ~~Checkpoint compatibility~~ -- measured, Section 3.
- ~~T2T scope versus masked positions~~ -- the reference restricts T2T to
  `~mask & ~prompt`, Section 4.
- ~~T2T scope across blocks~~ -- the reference only ever edits the trailing
  `block_length` window of `cur_x`; committed blocks are never revisited.
- ~~Reference implementation and pinned commit~~ -- it ships inside the
  checkpoint, Section 12.
- ~~`max_post_steps` semantics~~ -- counted at the top of mask-free iterations,
  breaking before the forward; Section 8.

Still open:

1. **`num_to_transfer` above 1.** Both presets imply the default of `1`, where
   the existing 2.0 selection is equivalent. If a preset ever raises it, the
   reference's two-branch selection has to be implemented properly and the
   equivalence argument in Section 6.2 no longer applies.
2. **Batched semantics.** The reference is batch-1, so nothing upstream defines
   how rows at different block offsets and different post-step budgets should
   interact. Phase 3's batch-versus-independent equivalence test is the only
   available oracle.
3. **Sampling.** Both presets are `temperature=0`. The reference's
   `_sample_with_temperature_topk_topp` supports `top_k`/`top_p`, but T2T with
   sampling has no termination guarantee beyond `max_post_steps` (Section 6.5).
   Out of scope for the first implementation; revisit if a preset needs it.

## 17. Revision Notes

Changes from the first draft, recorded because each one reverses a specific
earlier recommendation.

**Withdrawn: `batch_decode_ex` / `JointDecodeResult`.** The first draft added a
base-class method returning per-row block status, whose default implementation
derived `block_done` from the *post-update* token state and was described as
reproducing LLaDA2.0 semantics exactly. It does not: the current runners derive
completion from the *pre-update* gather, and the difference is precisely the
iteration on which KV is committed. The default would have committed KV for
blocks that still contained masks at forward time -- a silent correctness
regression for 2.0. It also assumed a base-class `batch_decode` that does not
exist (only `ThresholdParallelDecoder` implements it). Replaced by the
three-line runner-side predicate in Section 6.1, which is provably equivalent
for 2.0 and needs no contract change.

**Withdrawn: `num_to_transfer` and `add_minimum_topk_transfers`.** Not an
upstream parameter, and the existing `actual_threshold` clamp in
`get_transfer_index_threshold` already provides per-row minimum progress,
no-op-on-finished-rows, and mask-candidate exclusion (Section 6.2).

**Corrected: `editing_threshold = 0.0`.** The first draft labelled it a
debugging-only setting that must never ship. It is the official Speed Mode
value. The cost analysis was one-sided: it counted the extra editing forwards
without counting the forwards saved by the lower `tau_mask` that editing makes
safe.

**Corrected: default parameter values.** `editing_threshold` defaults to 0.5
(Quality Mode) rather than 0.9, which matched neither preset.

**Corrected: the unknown-decoder-name failure mode.** It raises
`AttributeError` in the decode loop rather than silently running hierarchy
decoding, because `HierarchyDecoder` has no `batch_decode`. The recommendation
to add a dispatch table stands; the recommendation to keep `hierarchy` as the
default does not, since that default is currently broken.

**Corrected: checkpoint compatibility.** Section 3 replaces a plan with
measurements. In particular, an intermediate draft flagged a
`max_position_embeddings` difference of 8192 vs 32768; that came from the
`LLaDA2.0-mini-preview` repository, not the 2.0 baseline in use, which is
identical to 2.1 in all 48 config keys.

**Second revision, after reading the reference implementation.** The 2.1
checkpoint ships its own decoding loop in `modeling_llada2_moe.py`; it was on
disk the whole time and had not been read. It confirmed the central design
choice of this guide -- the runner-side `(~had_mask) & (~changed)` predicate is
a vectorization of the reference's own break condition -- and corrected three
things. `num_to_transfer` was reinstated: it is a real upstream parameter, not
an invention of the first draft, though the reuse argument survives at its
default of 1. The no-remask invariant was downgraded from "confirmed by the
paper" to "stated by the paper, unenforced by the reference, and deliberately
enforced here". And the `max_post_steps` accounting was reconciled with the
reference's counter, which places the stable-detecting pass inside the budget
rather than after it. It also settled four open questions outright and made
clear that the reference is batch-1 and cacheless, so it cannot arbitrate
FluxServe's batching or KV-timing decisions.

**Added:** the `max_post_steps` truncation hazard and the `allow_edit` fix
(Section 8); the unbounded offline decode loop (Section 8.2); the
mask-count batch-selection starvation (Section 8.3); the CUDA-graph batch
decomposition constraint on per-row state (Section 8); the `expert_bias` /
`correction_bias` aliasing note (Section 3.2); and the `output_ids` filtering
quirk (Section 7).
