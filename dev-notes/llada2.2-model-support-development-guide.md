# LLaDA2.2 Model Support Development Guide

This document explains LLaDA2.2 support in FluxServe: checkpoint dispatch,
block routing, Levenshtein decoding, row state, KV ownership, distributed
execution, CUDA graphs, and the tests that define the integration contract.
Use [Serving LLaDA2.2](../docs/serving/llada2.2.md) for installation and runnable
commands. The [LLaDA2.1 design guide](llada2.1-model-support-development-guide.md)
describes the joint-decoding foundation; this guide specifies the additional
2.2 behavior without assuming that its editing budget works like 2.1's.

## 1. Scope and architecture

`inclusionAI/LLaDA2.2-flash` loads through the shared `LLaDA2LLM` model class.
It adds two independent integration requirements:

- **Block routing:** select an expert pool for each routing block, then select
  experts per token within that pool. This is a model semantic requirement,
  including when the ordinary `threshold` decoder is selected.
- **Levenshtein editing:** consume DELETE/SPLIT operations after joint
  Mask-to-Token (M2T) and Token-to-Token (T2T) updates. These operations can
  remove content or insert mask slots while keeping the block length fixed.

| Component | Responsibility |
| --- | --- |
| `LLaDA2SparseMoeBlock` | Dispatch routing according to checkpoint configuration |
| `llada2_block_routing_topk` | Compute allowed experts, token choices, and unscaled weights |
| `LevenshteinJointDecoder` | Apply the refinement step and adapt eager/graph interfaces |
| `LevenshteinRowState` | Own per-row tracking, schedule counters, finalization, and history |
| Diffusion runners | Own block loops, prompt boundaries, state indexing, and completion |
| CUDA graph runner | Refresh static buffers and replay model plus decoder tail |
| Engine and scheduler | Reserve fixed block capacity and publish stable output |

Temperature-zero decoding is the supported selection mode. The active block
is the only editable region. Cross-block insertion/deletion, editing committed
KV, and emitting provisional edits are outside the contract. Row state belongs
to a runner decode loop; a future API that yields mid-block must persist that
state across yields.

## 2. Checkpoint and configuration contract

### Model metadata

The checkpoint uses the shared attention, QK normalization, rotary embedding,
MoE, and weight-loading structure. Validate key sets, shapes, dtypes, and
configuration for each accepted revision rather than inferring compatibility
from the model name.

| Configuration | Role |
| --- | --- |
| `expert_capacity=48` | Number of allowed experts per routing block |
| `block_size=32` | Number of tokens contributing to one block-routing decision |
| `num_experts=256` | Total routed expert count |
| `num_experts_per_tok=8` | Per-token choices within the allowed pool |
| `norm_topk_prob=true` | Normalize selected bias-free expert weights |
| `routed_scaling_factor=2.5` | Scale applied downstream by the MoE implementation |
| `delete_token_id=156930` | Remove one generated position during the edit scan |
| `split_token_id=156931` | Insert a mask before the position's pre-write token |
| `vocab_size=157184` | Logit vocabulary dimension |

Router correction bias remains FP32. The checkpoint's `gate.expert_bias` name
maps to the model's `correction_bias` alias through the shared loader, as
explained in the 2.1 guide.

`generation_config.json` supplies stop IDs `156892` and `156900`.
`resolve_checkpoint_eos_ids` and decoder normalization propagate the set to
both runners. The primary `eos_id` remains available to callers that require a
single ID; stop membership uses `eos_ids`. Generation limits, model context
limits, and routing block length are distinct settings.

### Decoder configuration

Both serving entry points build a `RunnerConfig` and pass checkpoint edit IDs
to the `levenshtein_joint` factory branch.

| Setting | Recommended value | Meaning |
| --- | --- | --- |
| `parallel_decoding` | `levenshtein_joint` | Select structural joint editing |
| `block_length` | 32 | Fixed active-block length, a multiple of routing `block_size` |
| `threshold` | 0.5 | Strict M2T confidence threshold |
| `editing_threshold` | 0.0 | Strict T2T confidence threshold |
| `steps` | 32 | Number of schedule steps for the initial masks |
| `max_post_steps` | 16 | Refinement budget after original masks disappear |
| `max_steps_per_block` | 1000 | Hard refinement limit, followed by a stability forward |
| `temperature` | 0 | Deterministic candidate selection |

Thresholds must be in `[0, 1]`, `max_post_steps` must be non-negative, and
`max_steps_per_block` must be at least 2. The decoder requires `steps >= 1`;
CLI/config value 0 means that the factory substitutes `block_length`. Pass
preset values explicitly because general CLI defaults differ.

## 3. Block-routing design

For routing logits `g[t, e]` and correction bias `b[e]`, compute:

```text
s[t, e] = sigmoid(float32(g[t, e]))
r[t, e] = s[t, e] + b[e]
block_score[k, e] = max over tokens t in block k of r[t, e]
allowed[k] = top expert_capacity experts by block_score[k]
selected[t] = top num_experts_per_tok experts by r[t], restricted to allowed[k]
weight[t] = s[t, selected[t]]
```

Normalize selected weights when configured. Bias affects expert selection,
not the gathered mixture weights. `llada2_block_routing_topk` returns unscaled
FP32 weights and integer expert IDs; `FusedMoE` applies the routed scaling
factor. Applying that factor in both places changes model output.

Dispatch uses `expert_capacity > 0`. The checkpoint also contains grouped
routing fields, but these must not override block routing. Checkpoints with
absent or zero capacity retain grouped top-k. The custom routing hook requests
`TopKOutputFormat.STANDARD` so a specialized top-k path cannot bypass it.

### Packing and alignment

Flattening `[batch, tokens, experts]` into routing blocks is correct only when
each routing window belongs to one request at the intended absolute positions.

- Startup requires `block_length % config.block_size == 0`.
- Decode rows contain whole active blocks at aligned offsets.
- Prefill segments must each be aligned before ragged/paged packing; validating
  only the total flattened token count is insufficient.
- Padding and graph batch decomposition must preserve complete routing blocks.
- The routing function rejects a flattened token count not divisible by
  `block_size`.

Two misaligned requests whose lengths sum to a multiple of 32 are a required
negative test: the flattened-count guard alone cannot identify their boundary.
Also test multiple routing blocks within a larger decode block, since a valid
multiple is not necessarily equal to `block_size`.

## 4. Levenshtein refinement algorithm

### Step inputs

The tensor program receives logits `[B, L, V]`, pre-update block tokens
`[B, L]`, positional prompt protection `[B, L]`, and per-row state. Selection
suppresses the mask logit and computes candidate confidence with FP32 softmax.
M2T and T2T decisions use the same pre-update block.

Distinguish two types of unresolved masks:

- **Original masks:** mask positions inherited from block entry and still
  carrying original-mask tracking.
- **New masks:** masks inserted by SPLIT or appended as DELETE padding.

Tracking travels with surviving positions during structural edits. An
original-tracking bit is counted only while the corresponding token is still
`mask_id`; filling a token does not require clearing that bit. New masks do
not restart the original-mask schedule or post-mask budget.

### Schedule and M2T selection

Let `N` be the initial mask count, `S=steps`, and `t=step_id`, starting at 0.
For `t < S`, the scheduled number of original-mask transfers is:

```text
floor(N / S) + (1 if t < N % S else 0)
```

At each step, add the number of new masks to that schedule value:

```text
num_need = schedule[t] + new_mask_count
high_conf = masked positions with candidate_probability > threshold
if count(high_conf) >= num_need:
    M2T = high_conf
else:
    M2T = highest-confidence min(num_need, current_mask_count) masked positions
```

After the schedule ends, select all remaining masks. The final refinement
round also selects all remaining masks regardless of confidence. Unlike 2.1,
a step can intentionally transfer no masks when its schedule floor is zero;
structural edits can increase the total mask count.

T2T selects non-mask, non-prompt positions whose candidate differs from the
current token and whose confidence is strictly above `editing_threshold`.
Both write sets are disabled for finalized rows.

### Post-mask accounting and final round

At the start of an active iteration:

```text
if an unresolved original mask remains: post_steps = 0
otherwise:                              post_steps += 1
```

For `max_post_steps > 0`, the final round occurs when
`post_steps >= max_post_steps`. For `max_post_steps == 0`, it is the step
whose M2T selection resolves every remaining original mask. Independently,
`step_id >= max_steps_per_block - 1` forces a final round.

On a final round, resolve every remaining mask and suppress DELETE/SPLIT
candidates at written positions, choosing the best remaining token. Mark the
row finalized after those writes. The following forward applies no edit and
allows the runner to commit KV computed from the final tokens.

### Anti-loop escape

The reference records pre-edit-scan block results and, when a result repeats,
resamples a changed position with its current token suppressed. FluxServe
uses a bounded deterministic escape: select the lowest-confidence changed
position and choose the best alternative candidate, with up to five retries.
Final rounds skip escape so structural tokens cannot be reintroduced after
final-round suppression.

History records the block after candidate writes/escape but before structural
operations. Using the post-scan result instead changes which states count as
repetitions. This mechanism is a quality heuristic; bounded termination must
come from the final-round rules and hard cap, even if escape fails.

### DELETE/SPLIT scan

Scan generated positions left to right using the pre-write token snapshot:

| Position value | Contribution to the result | Tracking |
| --- | --- | --- |
| Protected prompt token | Token unchanged, including literal edit IDs | Not an original generated mask |
| DELETE | No output position | Drop its tracking |
| SPLIT | `[mask_id, old_token_at_position]` | New mask is non-original; restored token keeps prior tracking |
| Ordinary token or unresolved mask | One unchanged position | Preserve tracking |

Truncate the result to `L`, or append non-original masks until its length is
`L`. Insertion cannot shift another request or a committed block.

For example, with `M` denoting a mask:

```text
before writes: [a, b, c, d]
after writes:  [a, DELETE, c, d] -> [a, c, d, M]
after writes:  [a, SPLIT,  c, d] -> [a, M, b, c]
```

SPLIT restores `b` from the forward's input, not the SPLIT token or a second
sample. A protected prompt prefix contributes exactly its original length,
so generated edits cannot move it.

## 5. Vectorized implementation and row state

`apply_edit_operations` is the scalar oracle. Execution uses
`apply_edit_operations_batched`, which assigns contribution counts 0, 1, or 2,
computes exclusive prefix offsets, and scatters primary and secondary
contributions into a staging buffer. Discarded writes use a sink slot outside
the retained result. Truncation/padding returns fixed-shape `[B, L]` tokens and
tracking. No Python per-token scan is needed in the execution path.

`levenshtein_graph_step` combines selection, budget decisions, bounded escape,
structural edits, and stability predicates in one tensor program. Eager
`batch_decode` and the graph adapter call this same function. It mutates the
mask column of logits; tests invoking both paths must use independent copies.

`LevenshteinRowState` is indexed by global sequence ID:

| Field | Shape | Purpose |
| --- | --- | --- |
| `block_start` | `[rows]` | Detect entry into a new block |
| `is_original_mask` | `[rows, L]` | Carry original-mask provenance through edits |
| `initial_mask_count` | `[rows]` | Define the fixed transfer schedule |
| `step_id` | `[rows]` | Count active refinement steps |
| `post_steps` | `[rows]` | Count iterations after original masks disappear |
| `finalized` | `[rows]` | Make the next step inert after a final round |
| `seen_blocks` | `[rows, history, L]` | Store exact pre-scan token sequences |
| `seen_count` | `[rows]` | Restrict history lookup to live entries |

`_begin_block_rows` resets a row when its block offset changes. A block entering
without masks is born finalized. History storage can be reused because lookup
is limited by the reset count. Capture inputs gather row state; returned state
is committed after synchronization. Reordered sub-batches must never exchange
row counters or histories.

The runner's `DecodeEditBudget` provides the generic iteration guard, but its
2.1 `allow_edit` value does not control Levenshtein finalization. For 2.2 the
bound is `max_steps_per_block + 1`, not `block_length + max_post_steps + 1`.
Rows requiring refinement remain eligible even when mask-free; the runner's
Levenshtein selection uses stable sequence order among unfinished rows.

## 6. Invariants and reference-equivalence limits

| Invariant | Required behavior |
| --- | --- |
| Fixed block length | Every refinement result contains exactly `L` positions |
| Prompt preservation | Prompt positions survive writes and the structural scan unchanged |
| No candidate remasking | Candidate selection cannot write `mask_id`; only structural operations create new masks |
| No generated control-token publication | DELETE/SPLIT are consumed before stable output is published |
| Finalization | A finalized generated region has no masks or edit tokens |
| KV consistency | The committing forward used exactly the final block tokens |
| Row ownership | Tracking, counters, and history follow global sequence IDs |
| Bounded execution | A hard-cap final round is followed by an unchanged stability forward |
| Rank agreement | Tokens, state, completion predicates, and next block offsets agree before further collectives |
| Stable output | Active edits are private; a completed block is appended once |

The reference decoder is batch-one and cacheless. FluxServe extends it with
batching, prompt protection, KV commitment, and deterministic escape. Exact
reference parity is conditional on the following differences not changing the
trajectory:

- Suppressing the mask logit changes candidates and confidence normalization
  when the reference assigns probability to masks.
- Deterministic escape replaces random changed-position selection.
- The hard cap resolves remaining masks instead of returning an incomplete
  block; prompt edit tokens remain literal instead of being consumed.
- Scheduled transfers use stable position sorting, which differs from the
  reference's unspecified `topk` ordering when mask confidences tie.

Greedy candidates, escape alternatives, and final-round alternatives use
`argmax`, including its lowest-token-ID tie rule. History lookup compares
complete token sequences in a fixed-shape device buffer. It does not infer
sequence equality from a checksum. Storage is `rows * history * L` int64
values; history length is bounded by the per-block iteration limit.

Parity tests include equal logits and distinct sequences that would collide
under position-weighted sums, such as a block filled with `100` versus one
whose first four tokens are `[101, 99, 99, 101]`. Only exact repeated sequences
may trigger escape.

## 7. KV, EOS, and request lifecycle

Use the same stability predicate as joint decoding:

```python
had_mask = (before == mask_id).any(dim=1)
changed = (after != before).any(dim=1)
block_finished = (~had_mask) & (~changed)
```

`before` is the forward input, not an earlier iteration's snapshot. A final
round may remove all masks and change tokens, so `finalized` alone is not a
commit signal. One more forward on those tokens supplies consistent KV.

Dense execution commits the finishing forward's KV. Paged execution repeatedly
overwrites the active block's fixed page slots, including during graph replay.
Provisional device writes are not permission to advance the committed prefix.
Structural edits change block content, not absolute positions, page tables,
or the number of scheduler tokens reserved per block.

Prompt lengths are positional boundaries. Online paths use original
`input_ids`; offline benchmarking passes explicit prompt lengths. Calls that
omit lengths infer them from right padding and assume no literal prompt mask IDs. Protect unaligned prompt suffixes, including literal
DELETE/SPLIT IDs, before both selection and the edit scan.

The online block loop publishes only after stabilization. Search for EOS only
within generated, committed positions, then truncate according to the stop set
and `ignore_eos`. A role terminator in the prompt or an EOS later edited away
must not end the request. Output-length accounting must respect the request's
generation limit, including a final partial output block. Offline runners stop
at each row's block-aligned end and mask canvas positions beyond its exact
output limit. Token-array export preserves prompt and EOS positions uniformly
across batch sizes; consumers slice the prompt before applying stop policies.

Cancellation, scheduler slot reuse, and any future mid-block suspension must
preserve the correspondence between token state, row state, and KV ownership.
Do not expose provisional active-block KV as a reusable prefix-cache entry.

## 8. Distributed execution and CUDA graphs

### Synchronization contract

In eager decoding, tokens and mutable step outputs are broadcast before
`commit_row_state` scatters them into row storage. History append uses the
synchronized pre-scan token sequence and append decision. Initialization derives from the
synchronized input block; every rank must select the same rows and offsets.

Graph replay similarly returns updated tokens and extra row-state outputs.
The runner broadcasts these, but consumes graph completion predicates locally.
Thus agreement of token/state broadcasts alone is insufficient: divergent
local `changed` or `block_finished` values can still advance ranks differently.
Predicate synchronization or recomputation from synchronized tokens is needed
whenever local decisions can differ. Test this explicitly instead of assuming
that deterministic decoding establishes agreement across devices.

`FLUXSERVE_DEBUG_LLADA22=1` enables eager decoder checks for prompt preservation,
control-token consumption, finalized-mask absence, and cross-rank state hashes.
The fused replay path does not call that eager checker. Graph acceptance tests
must independently assert tokens, state, predicates, and offsets across ranks.

### Capture interface

`graph_fused_step=True` includes the decoder tail with model forward and
lm_head. `graph_extra_state=True` adds fixed-shape state buffers:

1. `graph_inputs` resets new blocks and gathers live state outside capture.
2. `graph_extra_buffers` allocates static state inputs for each capture size.
3. Replay refreshes inputs and resets padding buffers to inert values.
4. `graph_step` produces updated tokens, predicates, and mutable state outputs.
5. The runner synchronizes and commits outputs, appends history, and advances
   only stable rows outside capture.

History lookup occurs inside the tensor program; history append happens
outside capture. Finalized padding rows perform no writes, counter increments,
or history appends. Returned tensors are sliced to the actual batch size.
Dummy attention metadata must also isolate their KV writes from live requests.

Decode CUDA graphs require paged FlashInfer KV. The serving guide enables
padded capture sizes `1 2 4 8`. Dense fallback uses eager execution. Decomposed
capture batches gather state by global row ID, just like eager sub-batches.
Test H200 capture/replay with the actual FlashInfer build and model; fixed
shapes and CPU parity alone do not demonstrate device-kernel compatibility.

## 9. Test plan

### Test environment and reference inputs

Run from the repository root in an installed FluxServe environment with its
runtime dependencies. The reference fixture in `test/runtime/conftest.py`
loads `configuration_llada2_moe.py` and `modeling_llada2_moe.py` from either
`FLUXSERVE_LLADA22_REF_DIR` or the checkpoint's local Hugging Face cache.
Only these code files are required for scripted reference tests, not weights.
The fixture skips reference tests when those files are absent; a skip is not
a parity result. Pin the reference revision in test artifacts.

```bash
OMP_NUM_THREADS=2 python -m pytest \
  test/runtime/test_llada22_block_routing.py \
  test/runtime/test_llada22_levenshtein_decoder.py \
  test/runtime/test_llada22_levenshtein_graph_step.py \
  test/runtime/test_llada22_runner_safety.py \
  test/runtime/test_eos_ids.py -q
```

The `test/ci/ut/runtime.yaml` task runs the runtime and CI pipeline suites in
the existing GH200 runtime container with CUDA hidden. It downloads only the
two reference code files at a pinned revision, avoiding silent reference skips.
Actual GPU acceptance is separate from this CPU regression task.

Run the complete runtime suite for shared-runner and older-decoder regressions:

```bash
OMP_NUM_THREADS=2 python -m pytest test/runtime -q
```

### CPU test matrix

| Area | Cases and assertions | Test location |
| --- | --- | --- |
| Routing | Reference expert IDs/weights, block max, capacity, bias, alignment, legacy dispatch | `test_llada22_block_routing.py` |
| Edit scan | DELETE, SPLIT, mixed operations, truncation, padding, prompt prefix | `test_llada22_levenshtein_decoder.py`, `test_llada22_levenshtein_graph_step.py` |
| Schedule | Divisible/remainder schedules, zero floor, new-mask transfers, exhausted schedule | `test_llada22_levenshtein_decoder.py` |
| Termination | Stable fill, oscillation, post budget, zero post budget, hard cap, stability forward | `test_llada22_levenshtein_decoder.py`, `test_llada22_runner_safety.py` |
| State | Reordered/sparse row IDs, sub-batches, new-block reset, independent rows | Decoder and graph-step suites |
| Reference | Scripted M2T, T2T, DELETE/SPLIT, and budget traces | `TestReferenceParity` in the decoder suite |
| Graph contract | Eager/graph tensor and state parity, refreshed inputs, inert padded rows | `test_llada22_levenshtein_graph_step.py` |
| EOS and publication | Both stop IDs, prompt/future exclusion, transient EOS, `ignore_eos` | `test_llada22_runner_safety.py`, `test_eos_ids.py` |

These files supply regression coverage and extension points; they do not
replace the following adversarial cases or the GPU matrix.

Use the scalar edit scan as an independent oracle for randomized blocks.
Compare every tracking bit as well as tokens. For scripted reference parity,
record per-step write sets, original-mask counts, post counters, pre-scan
history entries, final tokens, and termination reason. Control random escape
or force a single eligible changed position before requiring exact parity.

Include equal-logit candidates, exact threshold comparisons, the explicit
checksum-collision pair above, and both pre-scan and post-scan repeated states.
Mock source-rank broadcasts with different local logits; verify completion
predicates and block offsets in the real runner path, not only decoder state.
Use per-request prefill misalignment even when the flattened total is aligned.

### GPU, distributed, and service matrix

| Dimension | Test | Required evidence |
| --- | --- | --- |
| Weight loading | Full 2.2-flash in a supported TP/EP layout | Expected parameters loaded; real-weight gate parity and valid logits |
| Routing packing | Prefill/decode, multiple lengths, padded rows | Per-request block membership and correct expert choices |
| KV | Dense and paged; budget and hard-cap termination | Device KV equals a forward over final block tokens |
| Batch | Single requests versus concurrent/reordered requests | Independent tokens, state, and completion under controlled numerics |
| Distributed | TP4/EP4 through repeated edits | Rank-equal state, predicates, offsets, and collective participation |
| Graph | H200 eager versus padded/decomposed replay | Real capture/replay, state parity, and replay/fallback counters |
| Padding | Change real batch size between replays | No stale state and no writes into another request's KV |
| API | Stream/non-stream, transient/permanent EOS | Equal final output; no generated DELETE/SPLIT leakage |
| Lifecycle | Cancel and reuse request/page slots | No prior request's state, history, or KV becomes live |
| Regression | 2.0/2.1 threshold and 2.1 joint | Preserved routing dispatch and decoder-specific behavior |

Test graph batch sizes both inside and between capture sizes, with real rows
at different block offsets and at different refinement phases. Inspect actual
KV slots after replay; a scripted Python KV sink cannot establish correctness
of device append kernels. A graph test that silently falls back to eager does
not establish graph correctness.

Exact agreement between ranks in one run is mandatory. Comparisons between
TP layouts or different attention kernels may require numeric tolerances and
quality evaluation, but must not relax the within-run synchronization contract.
A quality score does not prove that both stop IDs work; use controlled prompts
or injected logits that force each stop token separately.

### Quality and performance evaluation

Compare `levenshtein_joint` with `threshold` at 0.95 using the same checkpoint,
input prompts, tokenizer, generation cap, batch size, TP/EP configuration, and
cache path. Include HumanEval and GSM8K, and record source/checkpoint revisions
and full commands with the evaluation artifacts.

Report accuracy/pass@1, empty answers, generated control-token leakage,
completion-length distribution, forwards per block, finalization reasons,
post-step distribution, and escape frequency. Check that request generation
limits are honored before using token counts for throughput.

For performance, separate warmup, prefill, decode, and graph capture. Report
total completion tokens divided by measured generation time as aggregate TPS;
label any mean of batch TPS independently. Compare eager with graphs using
replay/fallback counters and the same workload. Kernel profiles should identify
routing, lm_head, selection, history lookup, synchronization, or cache work
before selecting an optimization target.

The acceptance report must establish routing semantics, structural-edit
semantics, final-token KV, bounded execution, rank consistency, graph replay,
stable service output, and legacy regressions. Evidence should identify its
checkpoint and execution configuration rather than imply that one path proves
all cache, device, and parallel layouts.

## 10. Implementation map

All paths below are relative to the repository root.

| Path | Main responsibility |
| --- | --- |
| `python/fluxserve/backend/models/llada2.py` | Block-routing helper and config-driven MoE dispatch |
| `python/fluxserve/backend/model_loader/loader.py` | Shared checkpoint loading and correction-bias mapping |
| `python/fluxserve/backend/execution/decoders/levenshtein.py` | Scalar oracle, tensor step, row state, eager/graph adapters, diagnostics |
| `python/fluxserve/backend/execution/decoders/factory.py` | Decoder configuration and dispatch |
| `python/fluxserve/backend/execution/decoders/utils.py` | EOS configuration and broadcast helpers |
| `python/fluxserve/backend/execution/forward_batch_info.py` | Runner configuration and validation |
| `python/fluxserve/backend/execution/runners/utils.py` | Iteration guard, row selection, generated-only EOS scan |
| `python/fluxserve/backend/execution/runners/block_diffusion.py` | Row-state ownership, loop bounds, eager completion |
| `python/fluxserve/backend/execution/runners/flashinfer_diffusion.py` | Paged execution, packing checks, stable publication, graph state commit |
| `python/fluxserve/backend/execution/flashinfer_cuda_graph_runner.py` | Static graph buffers, padding, replay input/output transport |
| `python/fluxserve/cli.py` / `python/fluxserve/bench_offline.py` | Alignment checks and online/offline configuration |
| `test/runtime/conftest.py` | Reference-code discovery and import fixture |
| `test/ci/eval/llada2.2-flash-evalscope-gsm8k.yaml` | CI evaluation workload definition |

## 11. References

- [Serving LLaDA2.2](../docs/serving/llada2.2.md)
- [LLaDA2.1 design guide](llada2.1-model-support-development-guide.md)
- [LLaDA2.2-flash checkpoint and reference modeling code](https://huggingface.co/inclusionAI/LLaDA2.2-flash)
- [Official LLaDA2.X repository](https://github.com/inclusionAI/LLaDA2.X)
