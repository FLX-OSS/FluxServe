# LLaDA2.1 Model Support Development Guide

This document explains the LLaDA2.1 integration in FluxServe: model loading,
joint decoding, block ownership, KV consistency, distributed execution, CUDA
graphs, and the tests that define the integration contract. For installation,
presets, and runnable commands, use [Serving LLaDA2.1](../docs/serving/llada2.1.md).

## 1. Scope and architecture

LLaDA2.1-mini and LLaDA2.1-flash use the shared `LLaDA2LLM` implementation.
The adaptation adds joint Mask-to-Token (M2T) and Token-to-Token (T2T)
decoding, rather than a separate Transformer or MoE model. The decoder may
rewrite generated tokens inside the active block; completed blocks remain an
immutable prefix.

The responsibility split is:

| Component | Responsibility |
| --- | --- |
| Model and loader | Load checkpoint parameters and compute logits/KV for the supplied tokens |
| `JointThresholdDecoder` | Select M2T/T2T writes within each active block |
| `DecodeEditBudget` | Track per-row editing limits and iteration counts |
| Diffusion runners | Supply prompt boundaries, execute forwards, determine stability, and advance blocks |
| CUDA graph runner | Replay fixed-shape model and decoder work using refreshed input buffers |
| Engine and scheduler | Own requests, reserve block capacity, and publish completed output |

Editing state belongs to the runner's decode loop. It does not require new
fields in `RequestState` or a new scheduler API because the online runner
resolves a block within one execution call. If execution is changed to yield
mid-block, that state must be persisted before the call returns.

Supported joint decoding is deterministic temperature-zero selection with
`num_to_transfer=1`. Sampling, editing committed blocks, and streaming
provisional edits are outside this contract. The existing `threshold` decoder
remains a separate opt-in choice for a baseline without T2T.

## 2. Checkpoint and configuration contract

### Model loading

Use checkpoint configuration to construct the shared model. Check architecture,
layer dimensions, attention/KV head counts, rotary dimensions, expert layout,
and vocabulary size when accepting a checkpoint revision. Compare every
safetensors key, shape, and dtype; matching architecture names alone does not
establish compatibility.

The router bias has two names that must remain coordinated:

- Checkpoints store `model.layers.N.mlp.gate.expert_bias`.
- The model aliases this parameter as `mlp.correction_bias`; parameter traversal
  can expose the alias instead of the checkpoint name.
- The loader maps the checkpoint name to the runtime name and maintains the
  correction bias in FP32. A checkpoint storing this bias in BF16 must still
  produce an FP32 runtime parameter.

Do not remove the rename independently of the model alias. Loader tests must
check the loaded parameter values, not merely the absence of exceptions.

### Decoder configuration

Both `serve` and `bench_offline` construct a `RunnerConfig`, which is passed to
`load_decoder`. The factory selects `joint_threshold` explicitly and rejects
unknown decoder names and unsupported `num_to_transfer` values.

| Setting | Meaning / constraint |
| --- | --- |
| `parallel_decoding` | `joint_threshold` selects joint editing |
| `threshold` | M2T confidence threshold in `[0, 1]` |
| `editing_threshold` | T2T confidence threshold in `[0, 1]` |
| `max_post_steps` | Non-negative per-row budget for mask-free editing iterations |
| `num_to_transfer` | Must be `1` for this implementation |
| `block_length` | Fixed number of positions processed by one block forward |
| `mask_id` | Mask token, `156895` for these checkpoints |
| `eos_ids` | Normalized stop-token set, resolved from checkpoint generation configuration |

The Quality preset uses thresholds `0.7 / 0.5`; Speed uses `0.5 / 0.0`.
Both use block length 32, temperature 0, and `max_post_steps=16`. Pass preset
values explicitly because CLI defaults also serve other decoders. A zero
editing threshold enables rewrites whenever the greedy candidate differs; it
does not disable T2T.

## 3. Joint decoding algorithm

### Inputs and outputs

For a selected batch of `B` rows, block length `L`, and vocabulary size `V`:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `logits` | `[B, L, V]` | Model output for the active block |
| `x_block` | `[B, L]` | Tokens supplied to that forward |
| `prompt_positions` | `[B, L]` | True for immutable prompt positions |
| `allow_edit` | `[B]` | Whether each row can still perform T2T |

`joint_threshold_update` returns updated tokens and the M2T/T2T write masks.
It suppresses the mask column in `logits` in place, so callers comparing
multiple implementations must clone the logits first.

`batch_decode(logits, block_start, x, block_length, prompt_lengths, allow_edit)`
gathers active blocks, applies the update, scatters into `x.data`, broadcasts
token state, and returns no result. Block completion is a runner decision.

### Candidate selection

Suppress the `mask_id` logit before selecting candidates:

```text
logits[..., mask_id] = -infinity
candidate = argmax(logits)
confidence = softmax(float32(logits))[candidate]
```

For each row, set confidence outside masked positions to negative infinity.
The implementation uses the existing threshold decoder's progress rule:

```text
actual_threshold = clamp(max(masked_confidence) - 1e-5, -1000, threshold)
M2T = masked_confidence >= actual_threshold
T2T = non_mask AND non_prompt AND allow_edit
      AND candidate != current_token AND confidence > editing_threshold
updated = candidate where M2T OR T2T, otherwise current_token
```

This transfers at least one mask while any remain. Several positions can
transfer below the requested threshold when their confidence lies within the
progress rule's tolerance of the maximum. A mask-free row transfers no M2T
positions. T2T never writes into prompt positions and is computed from the
pre-update block, so a newly resolved mask is not also edited in that step.

### Reference equivalence boundaries

The checkpoint's `LLaDA2MoeModelLM.generate()` is the algorithmic reference.
It operates on a single sequence and recomputes the full context without
FluxServe's block KV lifecycle. It is an oracle for selection and token
trajectories, not for batched scheduling or KV commit timing.

Account for these differences explicitly in parity tests:

- Mask-logit suppression prevents remasking; it also renormalizes confidence
  probabilities. The reference does not enforce this suppression.
- FluxServe uses the tolerance-based M2T progress rule and `>=`; the reference
  uses strict threshold comparison and a top-k fallback. Test exact threshold
  equality and near-equal mask confidences, not just random logits.
- FluxServe computes confidence softmax in FP32. Reference/runtime dtype and
  kernel differences can move a threshold decision.
- Budget termination includes a final forward on unchanged tokens so the
  committed KV matches the output. This may add a forward relative to the
  cacheless reference without changing final token IDs.

Use exact token/decision comparisons for scripted inputs away from these
boundaries. Real-model trajectory divergence must be localized and accompanied
by task-quality measurements; it is not automatically evidence of a bug or of
acceptable equivalence.

## 4. Prompt protection and row identity

Prompt protection is positional:

```python
positions = block_start[:, None] + torch.arange(block_length, device=device)
prompt_positions = positions < prompt_lengths[:, None]
```

Online prompt lengths come from `len(state.input_ids)`, excluding generated
output. Offline benchmarking passes explicit prompt lengths and generation
budgets per row. The runner also accepts calls without lengths, deriving them
from right-padded prompt tensors using `mask_id`; this fallback assumes the
actual prompt contains no literal mask token. Left padding requires a different
positional representation.

An unaligned prompt suffix belongs to the first decode block and must remain
unchanged across every editing iteration. A non-mask token is not sufficient
evidence that a position is editable.

Budget tensors are indexed by global `seq_id`, never by a row's local position
in a selected or graph-decomposed sub-batch. Reordering rows must preserve
ownership. Completed rows leave the pending set independently, and their
counters reset before the next block. A mask-free row may still need editing
or a stability forward; zero masks does not mean zero remaining work.

## 5. Block lifecycle and termination

Let `before` be the tokens used by the model forward and `after` the committed
result of decoder selection for the iteration:

```python
had_mask = (before == mask_id).any(dim=1)
changed = (after != before).any(dim=1)
block_finished = (~had_mask) & (~changed)
```

A block is not complete merely because its last mask was filled. The forward
that filled it saw masks, so its KV represents different tokens. Completion
requires a subsequent unchanged, mask-free forward.

`DecodeEditBudget` maintains `post_steps` and `block_iters` per row:

| Condition | Transition |
| --- | --- |
| Forward input contains masks | Keep `post_steps`; apply M2T and permitted T2T |
| Input is mask-free and tokens change | Increment `post_steps` |
| `post_steps >= max_post_steps` | Disable T2T on the next iteration |
| Mask-free input remains unchanged | Complete the block and reset row counters |

With `max_post_steps=0`, T2T is disabled from block entry; M2T still operates.
Budget exhaustion must disable further edits rather than immediately publish
an edited block.

The per-block bound is:

```text
max_block_iters = block_length + max_post_steps + 1
```

At least one mask is resolved per masked iteration, T2T cannot remask, and the
editing budget bounds the remaining changes. The final `+1` provides the
stability forward. The online loop enforces a bounded block execution;
`DecodeEditBudget` also checks for stuck rows periodically to avoid a device
to host synchronization on every update. Its check interval is not permission
to publish an incomplete block.

## 6. KV, scheduler, and output invariants

| Invariant | Consequence |
| --- | --- |
| Committed KV was computed from final block tokens | Never commit immediately after a token-changing iteration |
| The active block owns fixed absolute positions | Editing overwrites existing KV slots; it does not append positions |
| Committed blocks are immutable | Later blocks can reuse their KV without recomputation |
| Only completed blocks are published | Streaming cannot expose or retract provisional T2T edits |
| Prompt positions are immutable | Prompt suffixes are protected even within a generation block |
| Row state is independent | One row's budget or completion cannot terminate another row |
| Iterations do not count as generated tokens | Scheduler reservations and output accounting advance per completed block |

Dense execution copies the finishing forward's KV into the committed cache.
Paged attention writes into active slots during each forward; these writes
remain provisional until the block is stable. Graph replay can therefore have
no ordinary model output object while still updating device KV in place.
Tests must inspect cache content, not rely only on calls to a Python commit
helper.

Online editing iterations remain inside `_execute_paged_decode`. The returned
result publishes a stable block and sets `decode_block_completed`; the engine
then advances request state. Cancellation or row reuse must release/reset the
same ownership and state that ordinary completion would retire.

EOS scanning is restricted to generated positions before the committed block
end. Prompt role terminators and future block contents are excluded. A
transient EOS in an active block cannot terminate a request. Apply the
checkpoint's normalized stop set and `ignore_eos` policy only to stable output,
and test streaming and non-streaming using identical request parameters.

## 7. Distributed execution and CUDA graphs

### Rank agreement

All ranks participating in the same request must agree on tokens, completion,
block offsets, and editing counters before the next model collective.
In eager execution, `batch_decode` broadcasts tokens and the runner computes
predicates from the synchronized result and the previous synchronized input.
This makes budget and block advancement follow the same decisions.

The fused graph returns local `had_mask`, `changed`, and `block_finished`
alongside updated tokens. The runner consumes these predicates directly.
Consequently, graph execution depends on predicate agreement across ranks;
broadcasting tokens alone does not repair a divergent local predicate. A
robust distributed test injects different local selection results and checks
both output state and next-forward decisions. Synchronizing predicates or
recomputing them after token synchronization is necessary if local decisions
can differ.

### Capture and replay interface

`JointThresholdDecoder.graph_fused_step=True` selects a graph tail containing
`joint_threshold_graph_step`: candidate selection, writes, and completion
predicates. The graph includes the model forward and lm_head. Dynamic token
array scatter, synchronization, budget updates, and block advancement remain
outside capture.

Decode graph execution requires the FlashInfer paged cache. The serving
configuration controls it with:

```text
--use-decode-cuda-graph
--cuda-graph-decode-mode padded
--cuda-graph-capture-bs 1 2 4 8
```

Use capture sizes appropriate to the configured batch limit. In padded mode,
unused rows receive protected prompt positions, no editing allowance, and
isolated dummy KV storage; only real rows are returned to the runner. In
decomposed mode, each component gathers budget state using global row IDs.
Every replay refreshes token IDs, positions, page metadata, prompt masks, and
editing allowances. A previous replay's inputs must never survive as live row
state.

## 8. Test plan

Run tests from the repository root in an installed FluxServe environment with
its runtime dependencies. CPU tensor tests do not imply that the full import
chain works in a bare PyTorch-only environment.

```bash
OMP_NUM_THREADS=2 python -m pytest \
  test/runtime/test_joint_threshold_decoder.py \
  test/runtime/test_block_finished_equivalence.py \
  test/runtime/test_eos_ids.py \
  test/runtime/test_block_diffusion_offline.py -q
```

### Selection and state tests

Use small scripted logits and independent expected write masks. Cover M2T above
threshold and its progress fallback, T2T above/below threshold, equal candidate
IDs, equal logits, exact threshold ties, near-maximum confidence ties, mask
suppression, protected prompt suffixes, and unsupported configuration values.
Exercise `allow_edit=False` on both masked and mask-free rows.

Drive complete blocks with a model that stabilizes immediately and one that
alternates two token values indefinitely. Assert final tokens, forwards,
per-row counters, independent completion, and the trailing stability forward.
Use sparse/reordered row IDs and graph decomposition to detect accidental
local-row indexing. Include a literal mask in input as a representation-boundary
case rather than silently treating it as supported prompt text.

### Runner and reference tests

Compare the stability predicate with the original mask-only predicate for
`threshold` and credit-threshold decoding. These decoders cannot change a
resolved block, so the predicates must agree on supported inputs.

Use a scripted KV sink to record the tokens used by each committing forward;
assert equality with the final block. Cover budget exhaustion, unaligned
prompts, prompt EOS, transient generated EOS, multiple stop IDs, `ignore_eos`,
and request reuse/cancellation.

Pin a checkpoint revision for reference traces. Record candidate IDs,
probabilities, M2T/T2T sets, block tokens, and termination reason at each step.
Compare batch-one traces for both presets under controlled numerics, allowing
only an identified no-op stability forward on budget termination. Test batching
against independent FluxServe executions because the reference is batch-one.

### GPU and service matrix

| Dimension | Cases | Required evidence |
| --- | --- | --- |
| Checkpoint | 2.1-mini and 2.1-flash | Expected parameters loaded, valid logits, coherent output |
| Cache | Dense and paged | Same controlled token decisions and final KV content |
| Graph | Eager, padded, decomposed | Replay equivalence and correct real-row slicing |
| Batch | One, several, changing active count | Independent state and completion |
| Parallelism | Supported TP/EP layouts | Rank agreement and consistent collective participation |
| API | Streaming and non-streaming | Same final output, no provisional edits |
| Lifecycle | Cancel, release, reuse slots | No state or KV leakage between requests |
| Regression | LLaDA2.0 threshold paths | Preserved selection and stable-forward behavior |

Cross-TP floating-point differences need a declared comparison policy. Separate
exact rank agreement within one distributed execution from token equality
between executions using different numerical kernels or TP layouts. Quality
scores cannot replace rank or KV assertions.

Inspect graph replay/fallback counters to establish that graph tests actually
replayed captures. Compare real KV slots before and after padded replays and
prove that dummy rows cannot overwrite request data. The CUDA graph tests under
`test/runtime/cuda_graph_tests/` provide cache/replay test patterns; include the
joint decoder explicitly when using those patterns.

### Quality and performance evaluation

Use the launch configurations in the serving guide and `bench_offline` for offline evaluation. Compare both joint
presets with a threshold baseline using the same checkpoint, prompts, tokenizer,
generation limit, hardware, TP/EP layout, and cache configuration. Record the
checkpoint and source revisions and all decoding parameters with results.

Report task accuracy/pass@1, completion lengths, forwards per completed block,
editing-budget distribution, latency, and graph replay/fallback counts. Define
aggregate throughput as total completion tokens divided by total measured
generation time; label any mean of batch TPS separately. Explain warmup and
whether prefill is included. Test generated lengths against each request's
limit so accounting bugs cannot inflate throughput.

A release test report must cover selection, state ownership, final-token KV,
rank agreement, service output, graph execution, and baseline regressions.
Unit-test success alone is not a substitute for that matrix.

The `test/ci/ut/runtime.yaml` task runs runtime regressions and pipeline tests
inside the existing GH200 runtime container with CUDA hidden. This provides
the import dependencies without requiring device execution for CPU cases.

## 9. Implementation map

All paths below are relative to the repository root.

| Path | Main responsibility |
| --- | --- |
| `python/fluxserve/backend/models/llada2.py` | Shared Transformer/MoE model and lm_head |
| `python/fluxserve/backend/model_loader/loader.py` | Checkpoint mapping and router bias loading |
| `python/fluxserve/backend/execution/decoders/joint_threshold.py` | Joint selection, eager adapter, fused graph tail |
| `python/fluxserve/backend/execution/decoders/factory.py` | Explicit decoder dispatch |
| `python/fluxserve/backend/execution/decoders/utils.py` | EOS normalization/resolution and token broadcast |
| `python/fluxserve/backend/execution/forward_batch_info.py` | Runner configuration and validation |
| `python/fluxserve/backend/execution/runners/utils.py` | Editing budget and generated-only EOS scan |
| `python/fluxserve/backend/execution/runners/block_diffusion.py` | Dense/offline block loop and editing inputs |
| `python/fluxserve/backend/execution/runners/flashinfer_diffusion.py` | Paged block execution and graph integration |
| `python/fluxserve/backend/execution/flashinfer_cuda_graph_runner.py` | Static capture buffers and replay metadata |
| `python/fluxserve/cli.py` / `python/fluxserve/bench_offline.py` | Online/offline configuration plumbing |
| `test/runtime/test_joint_threshold_decoder.py` | Selection, budget, factory, and graph-tail tests |
| `test/runtime/test_block_finished_equivalence.py` | Legacy completion-predicate regression |

## 10. References

- [Serving LLaDA2.1](../docs/serving/llada2.1.md)
- [LLaDA2.1-mini checkpoint and reference modeling code](https://huggingface.co/inclusionAI/LLaDA2.1-mini)
- [LLaDA2.1-flash checkpoint](https://huggingface.co/inclusionAI/LLaDA2.1-flash)
- [LLaDA2.1: Speeding Up Text Diffusion via Token Editing](https://arxiv.org/abs/2602.08676)
- [Official LLaDA2.X repository](https://github.com/inclusionAI/LLaDA2.X)
