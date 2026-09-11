# LLaDA2.2 Model Support Development Guide

Companion to `llada2.1-model-support-development-guide.md`, and structured the
same way. Everything the 2.1 adaptation established -- the
`(~had_mask) & (~changed)` completion predicate, the KV-commit invariant,
prompt protection, per-row budgets scoped to the decode loop, and rank
consistency via broadcast-then-predicate -- carries over. This guide covers
only what LLaDA2.2 adds on top, which is two genuinely new things:

1. **a model-side change**: MoE *block routing* replaces group-limited top-k
   routing in the gate; and
2. **a decoder-side change**: *Levenshtein editing* -- `DELETE` and `SPLIT`
   control tokens consumed during block refinement, so the block's content can
   shrink or grow while the block itself stays fixed-length.

Unlike 2.1, which was a pure decoding change, **2.2 cannot run correctly
through the pre-adaptation FluxServe model code**: without the gate change the
router silently falls back to grouped top-k and produces wrong outputs, not
slower ones.

Written against the `inclusionAI/LLaDA2.2-flash` repository files (config,
tokenizer config, generation config, and the full `modeling_llada2_moe.py`),
cached locally at `$DEV_TOOLS/llada22_ref/`. Every
config value and reference-code claim below was re-verified against that cache
on 2026-09-05.

`$DEV_TOOLS` throughout this guide is a scratch directory on the development
cluster -- outside the repository -- holding the job scripts, evaluation
harnesses, sbatch drivers and cached reference files referred to below. Set it
to wherever you keep yours.

## 0. Implementation Status (2026-09-10)

**Phase 1 passed on 4xH100: the model loads, block routing is active, and
levenshtein_joint reaches HumanEval-164 pass@1 0.829 at 619.6 tok/s. Graph,
paged and online acceptance remain unmeasured.** All 2.2 work is still
uncommitted. Do not reapply the old Phase 0 patch to this working tree.

### Guide review

The guide correctly identifies block routing, fixed-length Levenshtein edits,
two EOS ids, and the stability-forward KV invariant. Its main weakness was
mixing design intent, historical patch instructions, and verified results.
The audit found three incorrect assumptions in the previous text:

- end-of-iteration token broadcast alone did **not** synchronize the decoder's
  mask tracking, seen-state history, and finalization decisions (now resolved
  the other way round: the step's *outputs* -- tokens plus every row-state
  field -- are broadcast before they are committed, so the committed state is
  the source rank's whichever way a local near-threshold comparison went);
- blocking candidate writes in the prompt did **not** protect literal
  DELETE/SPLIT tokens already present in that prompt from the edit scan;
- scanning the entire sequence for EOS included prompt role terminators,
  causing offline chat generation to stop after its first stable block.

A fourth gap was mask-count-based decode selection, which deprioritized
mask-free rows still needing T2T or a stability forward. These issues are
fixed and covered by CPU regressions. The previous claim that a rank-hash
diagnostic already existed was also incorrect; it is now implemented as
`FLUXSERVE_DEBUG_LLADA22=1` in the Levenshtein decoder.

### Completion checklist

| Area | Status | Evidence / remaining work |
| --- | --- | --- |
| Block routing and legacy dispatch | Implemented, CPU verified | Reference gate parity; config-dispatch test preserves grouped top-k for absent/zero capacity; STANDARD TopK output |
| Levenshtein M2T/T2T/DELETE/SPLIT | Implemented, CPU verified | Seven scripted reference scenarios plus hard-cap, prompt, row reset, batch decomposition, and anti-loop/state synchronization tests |
| Config, CLI, factory, dual EOS | Implemented, CPU verified | Both entry points resolve checkpoint ids; only generated committed positions participate in runner early stop |
| Prompt protection | Fixed, CPU verified | Literal prompt edit tokens survive; generated edits cannot move the prompt prefix |
| Rank-local editing state | Fixed, CPU verified with broadcast replay | The step's outputs (tokens plus every row-state field) are broadcast before commit, so the committed state is the source rank's; TP4/EP4 ran clean at scale in job 3481121, but no rank-hash assertion was enabled in that run |
| KV stability / hard cap | CPU runner tests passed | Both dense and FlashInfer runner iteration paths commit final forward inputs; actual device cache contents still need GPU validation |
| Online block publication / EOS | CPU tests added | Transient EOS stays private; both stop ids and ignore_eos exercised through `_execute_paged_decode`; HTTP/SSE end-to-end not yet measured |
| Dense/ragged/paged alignment | Code audited | Ragged/paged builders check each prefill length; dense rows and decode blocks are aligned; no claim of device-kernel validation |
| Full weight load and quality | **GPU verified** | Job 3481121 (2026-09-10, TP4/EP4 dense): 0 unmatched keys on all three runs; HumanEval-164 pass@1 0.829 @ 619.6 tok/s (levenshtein_joint) and 0.835 @ 325.1 tok/s (threshold 0.95); 0 empty answers. See *Phase 1 results* below |
| Block routing active at run time | **GPU verified** | Config-driven dispatch confirmed on the real checkpoint (`expert_capacity=48`, `block_size=32`); quality result rules out the silent grouped-topk fallback |
| GSM8K | **Pending GPU** | Phase 1 ran HumanEval only; wired as a CI eval task (below) |
| CI integration | Config written, **blocked on runner** | `test/ci/eval/llada2.2-flash-evalscope-gsm8k.yaml` validates against upstream's scanner and its serve command parses; needs a 4-GPU runner, a workflow fix and a warm HF cache |
| Fused CUDA-graph tail | Implemented, CPU verified | The whole iteration tail is one fixed-shape tensor program shared by eager `batch_decode` and `graph_step`; row state rides through the graph as extra buffers; parity, padding-row inertness and the replay contract are covered on CPU |
| CUDA graphs / paged GPU | **Pending GPU** | No capture, replay or paged GPU result. `use_decode_graph` requires the paged cache, so Stampede3's dense-only runs can never exercise it -- this needs B200/GH200 CI |

Validation commands and current test counts are recorded in Section 13.
Weights and reference code remain locally available under the paths in
Section 16. The previously measured header/gate evidence is in Section 3.2.

### Phase 1 results (job 3481121, 2026-09-10)

One 4xH100 node, TP4/EP4, dense flashinfer path, 7 minutes wall clock.
Results: `tools/llada22_phase1_3481121/summary.md` (outside this repo).

| run | pass@1 | mean gen len | tok/s | forwards |
| --- | --- | --- | --- | --- |
| `levenshtein_joint` 0.5 / 0.0, steps 32, mps 16 | 0.829 | 220 | 619.6 | 990 |
| `threshold` 0.95 | 0.835 | 228 | 325.1 | 2378 |

For reference, 2.1-flash on the same harness scored joint Quality 0.829,
joint Speed 0.835 and threshold 0.95 = 0.799. So 2.2 levenshtein matches the
best 2.1 joint preset, 2.2 threshold beats 2.1 threshold by 3.6 points, and
within 2.2 the Levenshtein decoder is 1.9x the throughput of threshold on
2.4x fewer forwards.

The pass@1 values coinciding with the two 2.1 numbers is arithmetic, not an
artifact: they are 136/164 and 137/164, adjacent integers, and only 9 of 164
generations are byte-identical to the 2.1 runs.

Also confirmed: safetensors headers layout-compatible with 2.1-flash (24161
tensors, 32 shards, zero key/shape/dtype diffs); block-routing gate parity
against the reference gate on real weights (layers 1/16/31, two seeds);
`eos_ids=(156892, 156900)` in the banner of every run; zero empty,
whitespace-only or punctuation-only answers.

**Not established by this run:**

- *Both* EOS ids. Median generation length is 214 with only 3/164 hitting the
  512 cap, so an EOS fires reliably -- but which of the two, and whether both
  work, needs controlled prompts. The mean-generation-length heuristic is not
  evidence here.
- Anything about CUDA graphs. The run was dense with `use_cuda_graph=False`,
  so it validates the vectorized *eager* tail, not the fused graph step.
- GSM8K, HTTP/SSE streaming, paged KV, and 2.0/2.1 bit-identical regression.
- `generated_length` maxima of 542 (levenshtein) and 725 (threshold) exceed
  the 512 generation cap. Unexplained; it does not affect pass@1, which is
  computed by executing the HumanEval tests, but it is not understood.

The banner `MoE block routing active: ...` did **not** appear in that run's
logs. Block routing was nonetheless active -- the dispatch is config-driven
and the real checkpoint carries `expert_capacity=48` / `block_size=32`, and a
fallback to grouped top-k would have produced wrong output rather than 0.829.
The cause was that the `fluxserve` stdlib logger had no handler, so *every*
model- and runner-side log record was dropped. Fixed by
`configure_logging()` in `cli.py`, called from `main()`, `_serve_worker()`
and `bench_offline.run_worker()`; it attaches one handler scoped to the
`fluxserve` namespace so third-party INFO chatter stays out. Regression
tests cover both the banner and the setup.

### Reproducing / next GPU step

```bash
sbatch $DEV_TOOLS/flux22_phase1.sbatch
```

The driver is `$DEV_TOOLS/llada22_phase1.sh`; it
reads the **live** working tree, so pin `FLUX22_FS` to a snapshot when the
tree is in flux (`FluxServe-phase1-3481121` holds the pre-vectorization tree
for A/B). Use `FLUXSERVE_DEBUG_LLADA22=1` for rank/state diagnostics.
Stampede3 requires `--kv-cache-layout dense --flashinfer-cache-mode dense
--flashinfer-prefill-mode dense`; paged/graph validation requires an
installation with the compatible private flashinfer-dllm build, which this
cluster does not have. Phases 2-4 and the rest of the Section 14 acceptance
remain open until measured.

### CI integration

Upstream replaced the hand-written `test/ci/eval/{1N1G,1N4G}/gsm8k.sh` scripts
with `test/ci_system/pipeline.py`, which **auto-discovers** tasks:
`pipeline.py scan --root test/ci --trigger per-commit` builds the GitHub
Actions matrix from the YAML files under `test/ci/`. Adding a config file is
therefore the entire wiring -- no workflow edit is needed to register a task.

`test/ci/eval/llada2.2-flash-evalscope-gsm8k.yaml` runs LLaDA2.2-flash on
GSM8K through `fluxserve serve` + evalscope. It is deliberately the
*production* configuration rather than the one Phase 1 could run: paged KV,
paged scheduler, and `--use-decode-cuda-graph --cuda-graph-decode-mode
padded`. That makes this task the first and only exercise of two things with
zero GPU coverage today -- the fused Levenshtein graph tail and the 2.2 online
serving path. `priority: high` puts it at the head of the dispatch queue,
since a 103B four-GPU eval that queues behind 1gpu unit tests will not finish
inside the 240-minute job timeout.

Verified locally: the file passes `validate_task` and appears in the scan
matrix as `runner=b200-4gpu runtime=host`, and every flag in its serve
command parses through `fluxserve.cli.build_parser` with the intended values.

Three prerequisites are **not** satisfied by the config alone:

1. **A 4-GPU runner must exist.** `b200-4gpu` and `gb200-4gpu` appear only in
   `test/ci_system/test_pipeline.py`; no real task has ever targeted one. Every
   checked-in eval uses `gh200-1gpu`, and LLaDA2.2-flash (~206GB) cannot fit a
   single 96GB GH200 -- and there is no LLaDA2.2-mini. If no 4-GPU pool is
   registered, the job queues forever rather than failing.
2. **`.github/workflows/pr-test.yml` is GH200-locked.** Its install and execute
   steps call `test/ci_system/env/gh200.sh` unconditionally, so a
   `runtime: host` task enters the wrong apptainer image. A ready-to-apply fix
   that selects the wrapper from `matrix.runtime` is kept outside the repo at
   `tools/ci_pr_test_runtime_env.patch`; apply it after syncing upstream/main.
3. **The model must be pre-cached** on the runner. The env wrappers bind a
   shared HF cache (`HF_BIND`), but a cold 206GB download will not complete
   within the ready timeout.

`score_threshold` is provisional -- see the comment in the file. A companion
`perf` task is deliberately **not** written yet: `perf_reference` needs a
measured baseline, and inventing one would gate future runs against a number
nothing has produced.

The CPU suite is a separate gap: `pr-test-cpu.yml` only `ast.parse`s every
tracked Python file, so the 221 `test/runtime` tests run in no workflow at
all. Wiring them is not a one-line change -- `flashinfer_diffusion.py` and
`llada2.py` import FlashInfer and Triton-backed modules at module scope, so a
bare `torch-cpu` runner cannot import them without either the full dependency
set or import guards. The 2.2 reference-parity tests already skip cleanly when
the cached checkpoint files are absent (17 in the decoder suite, 3 in the
routing suite), so only that import chain stands in the way.

## 1. Goal

Add first-class LLaDA2.2 support to FluxServe while preserving LLaDA2.0 and
LLaDA2.1 serving behavior unchanged.

The completed implementation must support:

- `inclusionAI/LLaDA2.2-flash` loading through the shared `LLaDA2LLM` class
  with the block-routing gate selected automatically from the config;
- the Levenshtein joint decoding loop: M2T with the transfer-schedule floor,
  T2T, DELETE/SPLIT consumption, and the reference termination semantics;
- both stop tokens (`156892`, `156900`);
- dense and paged KV cache execution;
- single-request and batched online serving;
- TP4/EP4 execution with bit-identical token state across ranks;
- stable-block streaming without retracting emitted tokens; and
- LLaDA2.0 **and** 2.1 regression compatibility (the new decoder and gate are
  strictly opt-in / config-dispatched). The generated-only EOS scan also
  fixes premature stopping on prompt EOS for older decoders; that bug case
  deliberately changes behavior.

## 2. Non-goals

The first implementation does not need to:

- change the `ParallelDecoder` base-class contract (the `needs_row_state` /
  `make_row_state` extension from the 2.1 Phase 4 work is already in place
  and suffices);
- add per-request editing state to `RequestState` or the C++ scheduler;
- edit tokens in committed blocks;
- vectorize the edit-op scan (Section 9.3) before it is measurably a
  bottleneck;
- support `temperature > 0`;
- capture the Levenshtein iteration tail in CUDA graphs (Section 12,
  Phase 4 -- the 2.1 `graph_fused_step` precedent does not transfer directly
  because of the sequential scan); or
- support checkpoints other than 2.2-flash.

## 3. Compatibility Assessment (verified)

Method as in the 2.1 guide: parse the repository's `config.json`,
`generation_config.json`, and `tokenizer_config.json`, and diff the modeling
file -- all from the local `tools/llada22_ref/` cache. The weight-header comparison, originally a Phase 1 prerequisite, now has
CPU results in Section 3.2; full weight loading still requires GPUs.

### 3.1 Config

`LLaDA2.2-flash` vs `LLaDA2.0-flash-CAP` (local), key by key:

- **new keys:** `block_size=32`, `expert_capacity=48` (block routing,
  Section 4); `delete_token_id=156930`, `split_token_id=156931` (Levenshtein
  ops, Section 5); `use_qk_norm=true` (now explicit -- FluxServe's
  `llada2.py` already hardcodes QK-norm and 2.0-flash weights already contain
  the norm parameters, so this is expected to be a no-op; verify at load);
- **changed:** `max_position_embeddings` 32768 -> **131072**, `rope_theta`
  600000 -> **3000000**, `transformers_version` -> 5.2.0;
- **removed:** `sliding_window` (2.0 carried 4096 with
  `use_sliding_window=false`, i.e. it was inert; non-semantic removal);
- **unchanged:** `vocab_size=157184`, 32 layers, `hidden_size=4096`,
  `num_experts=256`, `num_experts_per_tok=8`, 1 shared expert, GQA 32/4,
  `first_k_dense_replace=1`, `pad_token_id=156892` -- and, **trap:**
  `n_group=8` / `topk_group=4` are *still present* in the config but no
  longer used by the reference model code. Section 4 explains why dispatch
  must therefore key on `expert_capacity`, never on the presence of the group
  fields.

The `rope_theta` / `max_position_embeddings` changes flow through
`AutoConfig.from_pretrained` into the existing RoPE setup with no code
change; there is no architecture registry to touch (same situation as 2.1
Section 3.4).

### 3.2 Weights (reverified 2026-09-08)

Header comparison rerun (`tools/llada22_headers.py`, CPU compute node):
`LLaDA2.2-flash` vs `LLaDA2.1-flash` (2.0-flash-CAP is not in the local
cache; 2.1 was layout-identical to 2.0, so it carries the same
information): **24161 tensors in 32 shards on both sides; zero key, shape,
or even dtype differences.** The loader needs nothing new; expect zero
unmatched weights at load. `mlp.gate.expert_bias` is F32 as in 2.1, so the
`expert_bias` -> `correction_bias` loader-rename note from the 2.1 guide
(Section 3.2 there) applies unchanged, and the block-routing implementation
keeps consuming the fp32 `correction_bias`.

Gate parity on **real weights** also rerun on 2026-09-08
(`tools/llada22_gate_parity.py`, CPU: reads only the gate tensors from the
shards): layers 1/16/31, two seeds each -- FluxServe
`llada2_block_routing_topk` returns bit-identical expert ids and matching
weights (x `routed_scaling_factor` = 2.5) vs the checkpoint's own
`LLaDA2MoeGate` under the `create_bidirectional_mask` shim. Both scripts exited successfully in this audit. The Phase 1 job
also reruns them for the GPU-job record.

### 3.3 Tokenizer and stop tokens

- `<|mask|>` = 156895 and `<|endoftext|>` = 156892, unchanged from 2.0/2.1;
  `tokenizer_config.json` has `model_max_length=262144`.
- `generation_config.json` sets `eos_token_id: [156892, 156900]` -- **a
  list**. 156900 is `<|role_end|>` (verified in `added_tokens_decoder`), the
  chat-template role terminator, so it genuinely terminates chat completions
  and cannot be ignored. Resolution: give the LLaDA decoders an
  `eos_ids: tuple[int, ...]` (with `eos_id = eos_ids[0]` kept for
  compatibility), because `engine/executor.py:113-116` already does
  `getattr(self.runner.decoder, "eos_ids", (decoder.eos_id,))` -- the
  mechanism shipped with Diffusion Gemma
  (`decoders/diffusion_gemma.py:normalize_eos_ids`). The runners' EOS
  early-stop scans must check membership in the set, not equality.
- `156930`/`156931` (DELETE/SPLIT) are **not** in `added_tokens_decoder`;
  they live in base-vocab space. Their authoritative ids come from
  `config.json`, so `RunnerConfig` should default them from the checkpoint
  config at serve/bench time, not hardcode them.

### 3.4 Modeling-file diff

The diff between the 2.2-flash and 2.1-mini `modeling_llada2_moe.py` is
confined to exactly two areas:

- the gate: `group_limited_topk` replaced by `block_routing`
  (`LLaDA2MoeGate.block_routing`, line 244); and
- the decoding loop: `generate()` (line 1649) now delegates to
  `_joint_decode_block` (line 1389) plus helpers, notably
  `_apply_edit_operations_with_tracking` (line 1249).

Attention, MLP, embeddings, and layout are unchanged. Reference defaults from
`generate()`: `threshold=0.5`, `editing_threshold=0.0`, `block_length=32`,
`steps=32`, `max_post_steps=16`, `temperature=0.0`,
`max_steps_per_block=1000` (documented as a "hard safety cap on steps per
block").

### 3.5 Consequences

- Loading works through the existing `LLaDA2LLM`; the **only** model-code
  change is the gate dispatch (Section 4). No new model class.
- The reference decode loop shares the 2.1 reference's two limitations:
  **batch-size 1 only** and **no KV cache** (full-window recompute per
  iteration). As with 2.1, it can arbitrate token trajectories but not
  FluxServe's batching or KV-timing decisions.
- The 2.2 modeling code needs transformers>=5 (`create_bidirectional_mask`).
  Reference tracing runs under the existing transformers-5.2 sandbox
  (`PYTHONNOUSERSITE=1 PYTHONPATH=tools/tf52_sandbox`); the fluxserve env
  stays on 4.57. The CPU parity tests import the reference classes under a
  one-symbol `create_bidirectional_mask` shim instead, so they run in the
  normal env.

## 4. Model Side: Block Routing (required for correctness)

Reference (`LLaDA2MoeGate.block_routing`):

```python
# scores: (num_tokens, num_experts) = sigmoid(logits) + expert_bias
assert num_tokens % block_size == 0
block_scores = scores.view(num_blocks, block_size, E).max(dim=1).values
allowed  = topk(block_scores, k=expert_capacity)     # 48 experts per block
masked   = scores.masked_fill(~allowed_per_token, -inf)
topk_idx = topk(masked, k=top_k)                     # top-8 within the 48
```

`topk_weight` then gathers the **bias-free** sigmoid scores at `topk_idx`,
normalizes, and scales by `routed_scaling_factor` -- same as before.

This is a *semantic* routing change: per-token top-8 is restricted to the 48
experts chosen per 32-token block (block-level max over token scores). It
bounds expert activation per diffusion block, which is what makes
128K-context MoE inference tractable, and the checkpoint was trained with it.
Running grouped top-k on 2.2 weights produces wrong outputs.

### 4.1 Implementation (present in the working tree)

`llada2_block_routing_topk` in `backend/models/llada2.py`, wired into
`LLaDA2SparseMoeBlock` via the `TopK` layer's `custom_routing_function` hook
with the output format forced to `STANDARD` -- the triton-kernel format
routes on raw logits and would bypass the custom function. Two decisions
worth recording:

- **Dispatch keys on `config.expert_capacity > 0`.** The current
  `LLaDA2SparseMoeBlock` (`llada2.py:295-333`) reads `n_group`/`topk_group`
  and sets `use_grouped_topk=True` when they validate -- and the 2.2 config
  *still carries* `n_group=8`/`topk_group=4`, so without the new dispatch the
  gate silently does group routing. The reference itself ignores the group
  fields when block routing is configured; FluxServe matches that.
- Weights come back **unscaled** from the routing function;
  `routed_scaling_factor` stays applied downstream in `FusedMoE`, like every
  other routing path.

`test/runtime/test_llada22_block_routing.py` verifies exact id and weight
parity against the checkpoint's own `LLaDA2MoeGate` on random weights and
hidden states, plus the capacity / block-max / bias-selection properties.
Weights make this an end-to-end logits test in Phase 1, not a new design
question.

### 4.2 Constraints

- **Block boundaries are per-request diffusion blocks, not arbitrary 32-token
  windows of the flattened batch.** The reference only ever sees a batch-1,
  block-aligned window where `view(-1, 32, E)` is trivially correct. In
  FluxServe:
  - decode forwards are `[batch, block_length]` per row; with
    `block_length` a multiple of `block_size` and decode windows starting at
    `block_length`-aligned positions, the flattened view tiles routing
    windows exactly;
  - prefill packs multiple blocks per request; requests are padded/aligned to
    `block_length` in all current paths, so per-request segments are
    block-aligned -- but this must be **asserted, not assumed**, especially
    for varlen/ragged prefill and DP padding. The routing function raises on
    any misaligned flattened token count; keep that raise.
- Enforce at startup that `block_length % config.block_size == 0` when block
  routing is active (equal is the recommended case). Already enforced:
  `cli._check_block_routing_alignment` (from `b4e6506`).
- TP/EP: the gate is replicated and already runs in full precision on every
  rank; block routing is a deterministic function of its input, so ranks
  agree without extra collectives. EP dispatch consumes `topk_idx` as
  before.
- CUDA graphs: block routing is reshape + max + two top-ks with static shapes
  for a fixed batch -- graph-safe. The `num_tokens % block_size == 0` assert
  must hold for every captured shape.

## 5. LLaDA2.2 Decoding Semantics (Levenshtein editing)

Reference loop: `_joint_decode_block` (batch-1, cacheless). Each iteration
over the active block:

1. Forward; take block logits.
2. **M2T with a transfer-schedule floor.** `transfer_schedule =
   spread(initial_mask_count, steps)`; at step `s` the M2T set is "every mask
   above `threshold`", but if that has fewer than
   `schedule[s] + new_mask_count` entries, take that many top-confidence
   masks instead. `new_mask_count` counts masks that exist now but were not
   original masks -- i.e. masks created by earlier edit ops -- so the floor
   grows to re-resolve them promptly. After `steps` steps, M2T covers all
   masks.
3. **T2T** exactly as in 2.1 (`~mask & ~prompt`, strict
   `> editing_threshold`, candidate differs from current) -- except the
   candidate may now *be* `delete_token_id` or `split_token_id`. Writing an
   edit op into the block is how deletion and insertion are expressed.
4. **Anti-loop resampling.** A per-block set of previously seen pre-edit
   block states; on a repeat, resample one randomly chosen changed position
   with its current token's logit set to `-inf`, up to 5 tries, until the
   state is novel.
5. **Edit-op consumption** (`_apply_edit_operations_with_tracking`): scan the
   block left to right; `DELETE` removes its position; `SPLIT` at position
   `i` expands to `[mask, old_token_i]` (the pre-write token is restored
   after the freshly inserted mask). The result is truncated / right-padded
   with masks back to exactly `block_length`. A parallel `is_original_mask`
   bool list goes through the same scan; positions created by edits are
   marked non-original.
6. **Termination.** `post_steps` counts iterations whose block holds **zero
   original masks**, and *resets to 0* while any original mask remains. The
   block ends when:
   - *stable*: the post-edit block equals the pre-step state and holds no
     masks; or
   - *budget*: zero masks (original and new) and
     `post_steps > max_post_steps`; or
   - the `max_steps_per_block` hard cap fires.
7. **Final round** (budget about to fire): M2T is forced to cover all
   remaining masks, and any selected candidate equal to DELETE/SPLIT is
   resampled with those two logits suppressed, so the block terminates fully
   resolved with no surviving edit tokens.

### 5.1 What carries over from 2.1 unchanged

- **The block is fixed-length throughout.** The runner-visible sequence
  length never changes: edit ops rewrite block *content*, and the scan
  re-pads to `block_length`. KV pages, `decoding_start`, scheduler token
  accounting, and streaming are untouched by editing. This is the single
  most important simplification, and it is by upstream design.
- **The completion predicate `(~had_mask) & (~changed)` is again exactly the
  reference's stability test**, and it again guarantees the KV-commit
  invariant: the forward that commits a block's KV is the one whose input
  tokens are the block's final tokens (2.1 guide, Sections 6.1 and 8.1). No
  runner predicate change is needed.
- **Prompt protection.** The reference derives its prompt mask from "non-mask
  at block entry"; FluxServe keeps using explicit `prompt_lengths` (identical
  result, already plumbed through `_decoder_editing_kwargs`, and it protects
  correctly even if a prompt token were to equal an edit token).
- Only the active block is ever edited; committed blocks are immutable.

### 5.2 What does NOT carry over -- the real work items

- **Mask count is no longer monotone.** SPLIT and DELETE re-padding create
  new masks mid-block. The 2.1 structural no-remask guarantee (mask-logit
  suppression) still applies to *candidate selection*, but the invariant "a
  block without masks stays without masks" is gone. `DecodeEditBudget` as
  built for 2.1 (`post_steps` never resets; bound
  `block_length + max_post_steps + 1`) has the **wrong semantics** for 2.2:
  `post_steps` must reset while original masks remain, and the loop bound
  must come from `max_steps_per_block`, not from the mask count.
- **New per-row state**: `is_original_mask` (`[B, block_length]` bool), a
  per-row `step_id` for the transfer schedule, and the per-row seen-state
  sets -- loop-local, indexed by `seq_id`, same ownership pattern as the 2.1
  budget but genuinely new state (Section 9.1).
- **Edit-op application is a sequential scan** -- per row an O(32) Python
  list operation (Section 9.3).
- **Anti-loop resampling uses RNG in the reference** (`torch.randint` picks
  *which* changed position to resample). FluxServe replaces this with a
  deterministic choice (Section 7.1).
- **Two EOS ids** (Section 3.3).
- **M2T uses the schedule floor, not the 2.1 clamp trick.** The
  `get_transfer_index_threshold` reuse argument from 2.1 Section 6.2 does not
  apply; the 2.2 selection (threshold set, else top-`num_need`) is
  implemented as written. It degenerates to "at least one" only when
  `steps >= block_length` and no new masks exist.

## 6. Configuration and CLI

All in place as of Phase 0.5 (the first four fields and their CLI flags on
both entry points shipped with `b4e6506`; `eos_ids` was added 2026-09-05):

```python
steps: int = 0                     # M2T schedule length; 0 means block_length
max_steps_per_block: int = 1000    # reference hard cap
delete_token_id: int = 156930      # default from checkpoint config, not hardcoded
split_token_id: int = 156931       # ditto
eos_ids: tuple[int, ...] = (156892,)   # 2.2 serve/bench passes (156892, 156900)
```

`threshold`, `editing_threshold`, and `max_post_steps` are reused as-is from
the 2.1 work. `delete_token_id` / `split_token_id` / `eos_ids` should be
populated from the checkpoint's `config.json` / `generation_config.json` at
serve and bench time; the literals above are fallbacks only.

Register the decoding name `levenshtein_joint` (implemented;
`load_decoder` already raises on unknown names since 2.1 Phase 1). Do not
change `threshold` or `joint_threshold` semantics.

Startup validation for 2.2 checkpoints: `block_length % config.block_size
== 0` (recommended: equal, i.e. 32); `max_model_len <= 131072`; block
routing active (i.e. `expert_capacity > 0` was honored). The CLI now sets
`generation_block_size = block_length` for LLaDA as well as Diffusion Gemma,
so the processor reserves enough context for a whole generation block.

### 6.1 Recommended settings

From the model card and the `generate()` defaults -- there is a single
recommended mode, shaped like 2.1's Speed preset (T2T plus Levenshtein ops
carry the quality):

```text
--threshold 0.5  --editing-threshold 0.0  --block-length 32
--steps 32  --max-post-steps 16  --max-steps-per-block 1000
temperature 0
```

### 6.2 Suggested correctness-first command (Stampede3: dense flags mandatory)

```bash
python -m fluxserve.cli serve \
  --model inclusionAI/LLaDA2.2-flash \
  --host 127.0.0.1 --port 8000 \
  --tp-size 4 --dp-size 1 --ep-size 4 \
  --max-num-seqs 1 \
  --max-model-len 32768 \
  --block-length 32 \
  --parallel-decoding levenshtein_joint \
  --threshold 0.5 --editing-threshold 0.0 \
  --steps 32 --max-post-steps 16 --max-steps-per-block 1000 \
  --attention-backend flashinfer \
  --kv-cache-layout dense \
  --flashinfer-cache-mode dense \
  --flashinfer-prefill-mode dense
```

(On CI containers with flashinfer-dllm 0.6.18, switch to the paged flags as
in the 2.1 guide Section 5.4.)

## 7. Decoder Design

`LevenshteinJointDecoder` in `decoders/levenshtein.py` (implemented). Same
in-place `batch_decode` contract as every other decoder, extended with the
keyword arguments the runner hooks already pass
(`prompt_lengths=..., row_state=..., seq_ids=...`); it declares
`needs_row_state = True` and provides `make_row_state(num_rows, block_length,
device)`, which is how `_make_decode_loop_state` knows to build a
`LevenshteinRowState` instead of the 2.1 `DecodeEditBudget`.

The decoder implements the full Section 5 loop body: schedule-floor M2T, T2T
with edit-op candidates, anti-loop escape, the sequential edit-op scan with
`is_original_mask` tracking, reference `post_steps` semantics, and the
final-round force-resolve with DELETE/SPLIT logit suppression. It ends with
`broadcast_if_needed(x.data)` like every decoder, and raises if constructed
with non-zero temperature.

### 7.1 Deliberate deviations from the reference

The first three deviations are documented in the decoder docstring. CPU
regressions cover them and the prompt-preservation extension:

1. **Mask-logit suppression** (as in 2.1): `mask_id` can never be selected as
   a candidate. The reference does not enforce no-remask; FluxServe does,
   structurally. Note this is narrower than in 2.1 -- masks still *appear*
   via SPLIT/padding, but never via candidate selection.
2. **Deterministic anti-loop choice.** Where the reference resamples a
   `torch.randint`-chosen changed position, FluxServe resamples the
   **lowest-confidence** changed position. This makes decoding reproducible
   and rank-consistent with no seeded-generator plumbing: the choice is a
   pure function of post-broadcast state, so the 2.1 Section 9.2 argument
   applies unchanged. Cost: exact trace parity with the reference is
   conditional on the anti-loop never firing (Section 13).
3. **Force-resolve at `max_steps_per_block`.** The reference warns and can
   return residual masks when the hard cap fires; FluxServe force-resolves
   the remaining masks (final-round semantics) instead, because a committed
   block containing `mask_id` would poison the KV prefix and the output
   filter.

4. **Literal prompt edit tokens are preserved.** The upstream reference warns
   and still consumes these tokens. FluxServe treats the prompt prefix as
   opaque input even when it contains DELETE/SPLIT ids. This extends the
   reference semantics to enforce the serving prompt-protection contract.

## 8. Prompt Protection

Carries over from 2.1 Section 7 verbatim: `prompt_positions` from absolute
positions vs per-row `prompt_lengths` (online: `len(state.input_ids)`;
offline: `non_mask_number`), plumbed through `_decoder_editing_kwargs`. Two
2.2-specific notes:

- T2T must never write DELETE/SPLIT into a prompt position -- this falls out
  of the same `~prompt` term, no extra logic; and
- the edit-op scan operates only on the block; when the first generation
  block contains an unaligned prompt suffix, DELETE/SPLIT consumption must
  not shift prompt tokens. The decoder treats prompt positions as fixed scan
  prefix; the parity tests cover an unaligned-prompt scenario.

## 9. Runner and Block Lifecycle

### 9.1 Per-row state

`LevenshteinRowState`, owned by the runner loop (built by
`_make_decode_loop_state`, passed via `_decoder_editing_kwargs`),
self-initializing on block entry and indexed by `seq_id` -- never by
position within a CUDA-graph-shaped sub-batch (same decomposition hazard as
2.1 Section 8). Contents per row: `is_original_mask`, `step_id`,
`post_steps`, and the seen-states set for the anti-loop.

The 2.1 `DecodeEditBudget` state transitions do not apply. 2.2 semantics:

```text
any original mask remains        -> post_steps = 0
no original masks this iteration -> post_steps += 1
stable (no masks, no change)     -> block_finished    (predicate unchanged)
no masks and post_steps > max_post_steps
                                 -> final round: force-resolve, suppress
                                    DELETE/SPLIT; next iteration is stable
step_id == max_steps_per_block   -> hard cap: force-resolve (Section 7.1)
```

As in 2.1, budget termination is arranged so the **next** iteration finds the
block unchanged and finishes it naturally -- FluxServe performs exactly one
extra stability forward relative to the reference, which is what keeps the
committed KV consistent with the final tokens. The CPU parity tests encode
this as "token-identical, with one extra forward allowed on
budget-terminated blocks".

### 9.2 Iteration cap

The per-block iteration cap becomes `max_steps_per_block + 1` (the `+1` is
the stability forward). The 2.1 bound `block_length + max_post_steps + 1` is
wrong for 2.2 because mask count is not monotone. Both runners' loop guards
must use the decoder-appropriate bound -- the runner reads it from the
decoder (`max_block_iters`), which the hooks already support. The offline
runner's formerly-unbounded loop got its guard in the 2.1 work; 2.2 only
changes the number.

### 9.3 The edit-op scan is eager and sequential

Per row it is the reference's O(32) Python list scan, applied after a
`.tolist()` sync. The decode loop already synchronizes scalars, but these
additional reads can still introduce device-to-host stalls. It is intrinsically eager: it cannot be
captured in a CUDA graph, and at large batch it is O(B*32) Python work per
iteration. The pragmatic order is: correctness first (Phases 1-3, eager),
then measure; a prefix-sum compaction vectorization is the known follow-up
if profiles demand it (Section 12, Phase 4).

### 9.4 Batch selection

Implemented: both runners select Levenshtein decode rows in stable sequence
order from the unfinished set. A mask-free editing row is therefore not
pushed behind rows with more masks. Prefill and 2.0/2.1 decode selection keep
their existing mask-count ordering. Online execution already tracks a
shrinking pending-row set. This is a completion policy, not a measured
throughput optimization.

## 10. KV Cache and Distributed Execution

- **KV story unchanged from 2.1.** The block is fixed-length; extra editing
  iterations overwrite the same page slots
  (`_make_decode_forward_batch` positions are `decoding_start + arange`);
  KV commits exactly once, on the stability forward, whose input tokens are
  final by construction of the predicate. Nothing about DELETE/SPLIT touches
  page tables or token accounting.
- **Rank consistency.** Candidate ids and confidences are broadcast before
  threshold/schedule decisions. Final-round alternate candidates and anti-loop
  replacement ids are also broadcast before writes. Thus tracking, seen-state
  sets, and finalization evolve from the same decisions; the final token
  broadcast remains in place. Deterministic anti-loop position selection alone
  would not have guaranteed this with rank-local floating-point differences.
  `FLUXSERVE_DEBUG_LLADA22=1` checks prompt preservation, edit-token consumption,
  finalized-mask absence, and cross-rank hashes of token and row state.
  CPU broadcast-replay tests deliberately supply different per-rank logits;
  TP4/EP4 NCCL execution is still an outstanding GPU acceptance item.

## 11. Output and Streaming Semantics

Unchanged contract: active block mutable and invisible; stable block appended
once and immutable. Editing iterations stay inside one
`_execute_paged_decode` call. Two 2.2-specific points:

- **EOS handling** is membership in `eos_ids` (both 156892 and 156900),
  applied only after block stabilization, with the existing `ignore_eos`
  behavior. An EOS that appears and is later edited away must not terminate
  the request -- gating on `block_finished` already ensures this.
- **DELETE/SPLIT tokens never reach the output.** The final-round suppression
  plus force-resolve guarantee a stable block contains no edit tokens; the decoder
  now checks this under `FLUXSERVE_DEBUG_LLADA22=1`, excluding the protected
  prompt prefix.

## 12. Implementation Phases

### Phase 0: No-weights implementation -- DONE (working tree)

Block-routing gate + parity test; `LevenshteinJointDecoder` + parity tests;
registration. See Section 0.

### Phase 0.5: Plumbing (no GPU) -- DONE (2026-09-05)

- Code is present in the working tree; no patch reapplication is needed.
- `RunnerConfig` fields + CLI flags: already present from `b4e6506`.
- `eos_ids` on the LLaDA decoders; runners' EOS scans use the set. Done.
- `generation_block_size = block_length` for LLaDA in `cli.py`. Done.
- `block_length % block_size` startup check: already present from `b4e6506`.
- Historical Phase 0.5 result: 161 passed / 15 skipped on 2026-09-05.
  The current expanded-suite result is in Section 13.

### Phase 1: Load + smoke (4xH100, dense flags on Stampede3) -- GPU blocked

Job: `sbatch tools/flux22_phase1.sbatch` (outside the repo, in
`$DEV_TOOLS`; driver `llada22_phase1.sh`).

- ~~Safetensors header comparison~~ -- **done 2026-09-06 on the login node**
  (`llada22_headers.py`): zero diffs vs 2.1-flash, Section 3.2.
- ~~Numerical gate check on real weights~~ -- **done 2026-09-06**
  (`llada22_gate_parity.py`): ids bit-identical, weights match, Section 3.2.
- TP4/EP4 load through `LLaDA2LLM`: zero unmatched weights, "MoE block
  routing active" logged at startup, `eos_ids=(156892, 156900)` in the
  bench banner. *(GPU job, `lev_single` run.)*
- Single-request generation sanity via `levenshtein_joint` with the
  Section 6.1 settings; coherent generation and explicit controlled traces
  for both stop ids (mean length below the cap alone is insufficient).
  *(GPU job; explicit EOS traces still to add.)*
- Bonus in the same job: HumanEval-164 at bs 8 for `levenshtein_joint`
  (2.1-flash joint reference 0.829/0.835) and a `threshold` 0.95 baseline
  (2.1-flash: 0.799).

Exit criterion: coherent single-request generation, zero unmatched weights,
gate parity on real weights.

### Phase 2: Reference parity at scale

- Batch-1 traces via the checkpoint's own `generate()` under the
  transformers-5.2 sandbox (`tools/llada21_ref_trace.py` pattern), same
  machine, same weights. Compare per-iteration: candidates, M2T/T2T sets,
  edit-op streams, `is_original_mask`, `post_steps`/`step_id`, final tokens.
- **Parity is conditional on the anti-loop not firing** (deterministic
  deviation, Section 7.1) -- assert trace-side whether it fired; on
  divergent traces, fall back to the 2.1 lesson: cross-implementation
  trajectories diverge on numerics anyway, and **pass@1 / task accuracy is
  the accepted judge** (HumanEval/GSM8K harnesses from the 2.1 campaign;
  baselines to beat or match: 2.1-flash joint Quality 0.829 / Speed 0.835
  HumanEval, 0.9453 GSM8K-256 jq).

### Phase 3: Batched, paged, distributed — CPU coverage added, GPU pending

Same structure as 2.1: batch-vs-independent equivalence (the only oracle for
batched semantics), different block offsets per batch, row-independent
completion, batch-selection fix (Section 9.4), TP4/EP4 rank-consistency
assertions, paged KV verification (CI containers -- Stampede3 stays dense),
online streaming tests.

### Phase 4: Performance — not measured

- Measure the eager edit-op scan cost at batch; vectorize only if it shows.
- CUDA graphs: the transformer forward captures as today. The 2.1
  `graph_fused_step` tail does **not** transfer -- the scan and the
  seen-state sets are host-side. Options, in order of preference: capture
  forward-only and keep the tail eager (measure first); a fused tail that
  stops before the scan; vectorized scan then fuse. Decide on profiles, not
  upfront.
- Benchmark against 2.1-flash same hardware/flags; publish
  accuracy-vs-throughput. The paper's claim structure (editing reduces total
  forwards) should be validated with mean-forwards-per-block and the
  `post_steps`/`step_id` histograms, as in 2.1 Section 13.

## 13. Test Plan and Results

CPU verification (2026-09-08, conda environment `fluxserve`, cached reference
modeling code loaded under the documented transformers mask shim):

```bash
source $DEV_TOOLS/flux_gpu_env.sh
OMP_NUM_THREADS=2 python -m pytest test/runtime -q --disable-warnings
```

Latest complete result: **221 passed, 15 skipped**. CUDA-dependent tests are skipped
on this CPU node. This result verifies runtime logic, not GPU numerics.

Coverage present:

- `test_llada22_block_routing.py`: reference gate parity, capacity/bias/block
  properties, and actual MoE constructor dispatch for absent/zero/positive
  expert capacity (expert execution stubbed for CPU).
- `test_llada22_levenshtein_decoder.py`: seven reference-loop parity scenarios,
  schedule, edit scan, prompt write protection, literal prompt edit ids,
  prompt-prefix preservation during generated edits, hard-cap cleanup,
  reordered sparse seq_ids versus independent runs, next-block reset,
  replayed source-rank decisions with divergent logits, and debug checks.
- `test_llada22_runner_safety.py`: dual EOS with prompt/future exclusion,
  editing-row selection, global row-state plumbing and iteration bound,
  both real runner loops committing final forward inputs after hard-cap
  resolution, and online stable-block results for transient/permanent EOS
  with both ids and `ignore_eos`. Model forwards/KV sinks are scripted;
  this is not a GPU cache or HTTP streaming test.
- `test_eos_ids.py`: normalization, all LLaDA decoder factories, and checkpoint
  generation-config resolution. Existing 2.0/2.1 runtime regressions run as
  part of the full suite.
- `test_llada22_block_routing.py` also guards the startup banner (logged on
  the first MoE layer only, absent when routing is off) and the `fluxserve`
  logging setup that makes it visible.
- `test_llada22_levenshtein_graph_step.py`: the fused decode-graph tail --
  the batched edit scan against the scalar oracle over randomized blocks, the
  block-state hash, `graph_step` reproducing `batch_decode` token for token
  and state field for state field across a multi-iteration block, the
  new-block reset happening outside the capture, and a simulated
  `replay_decode` proving a padded batch returns the unpadded result and that
  padding rows write nothing and advance no counter.

Outstanding: full-model GPU load, real-weight logits/quality comparisons,
actual TP4/EP4 collectives, paged cache and CUDA-graph execution, HTTP streaming
versus non-streaming equivalence, and performance measurements. In particular,
CPU legacy-dispatch tests do not establish bit-identical 2.0/2.1 GPU outputs.

## 14. Acceptance Criteria

LLaDA2.2 support is complete when:

- `LLaDA2.2-flash` loads through the shared model class with block routing
  selected from config and zero unmatched weights;
- the block-routing gate matches the reference gate exactly on real weights;
- `levenshtein_joint` implements the Section 5 loop with prompt protection,
  no-remask candidate selection, and no edit token ever emitted;
- active blocks remain private until stable; budget/hard-cap termination
  never commits KV that disagrees with emitted tokens;
- both stop tokens terminate requests correctly;
- CPU parity suites pass, and GPU accuracy meets or explains any gap vs the
  reference implementation on HumanEval/GSM8K;
- batched output matches independent runs; TP=1 control flow assertions
  never fire at TP4/EP4 (token-equality vs TP=1 is impractical at 103B --
  use the rank-hash assertion plus accuracy instead);
- streaming and non-streaming agree;
- 2.0 and 2.1 behavior is bit-unchanged; and
- the serve command and recommended settings are documented with measured
  numbers.

## 15. Implementation Files

```text
(core 2.2 support)
dev-notes/llada2.2-model-support-development-guide.md
python/fluxserve/backend/execution/decoders/__init__.py
python/fluxserve/backend/execution/decoders/factory.py
python/fluxserve/backend/execution/decoders/levenshtein.py
python/fluxserve/backend/models/llada2.py
test/runtime/test_llada22_block_routing.py
test/runtime/test_llada22_levenshtein_decoder.py

(Phase 0.5, done 2026-09-05; 2.2 flags/fields themselves were in b4e6506)
python/fluxserve/cli.py                                    # eos_ids resolution, generation_block_size
python/fluxserve/bench_offline.py                          # eos_ids resolution
python/fluxserve/backend/execution/forward_batch_info.py   # RunnerConfig.eos_ids
python/fluxserve/backend/execution/decoders/utils.py       # normalize/resolve eos helpers
python/fluxserve/backend/execution/decoders/factory.py     # eos_ids to every decoder
python/fluxserve/backend/execution/decoders/threshold.py   # eos_ids attribute
python/fluxserve/backend/execution/decoders/joint_threshold.py
python/fluxserve/backend/execution/decoders/levenshtein.py
python/fluxserve/backend/execution/decoders/hierarchy.py
python/fluxserve/backend/execution/runners/block_diffusion.py   # multi-EOS early-stop scan
python/fluxserve/backend/execution/runners/flashinfer_diffusion.py  # scan + output truncation
test/runtime/test_eos_ids.py

(2026-09-08 audit fixes and additional coverage)
python/fluxserve/backend/execution/runners/utils.py        # generated-only EOS scan
python/fluxserve/backend/execution/decoders/levenshtein.py # prompt scan, synchronized state, debug diagnostics
python/fluxserve/backend/execution/runners/block_diffusion.py # editing-row selection + EOS scan
python/fluxserve/backend/execution/runners/flashinfer_diffusion.py # same
test/runtime/test_llada22_runner_safety.py
```

Explicitly **not** changed: `decoders/base.py` (the `needs_row_state`
extension already exists), `engine/request.py` and the C++ scheduler
(editing state stays loop-local), `engine/executor.py` (already
`eos_ids`-aware).

## 16. References

- [LLaDA2.2-flash model card](https://huggingface.co/inclusionAI/LLaDA2.2-flash)
- [Official LLaDA2.X repository](https://github.com/inclusionAI/LLaDA2.X)
  (includes `LLaDA2_2_tech_report.pdf`)
- Local reference cache: `$DEV_TOOLS/llada22_ref/`
  (config, tokenizer config, generation config, modeling file)
- `dev-notes/llada2.1-model-support-development-guide.md` -- the base this
  guide builds on
- `docs/serving/llada2-flash.md` -- TP4/EP4 flash serving setup

## 17. Open Questions

1. Resolved by code audit: ragged and paged prefill builders validate each
   selected request length modulo `block_length`; dense prefill uses aligned
   rectangular rows. A flattened-count check alone would **not** detect two
   misaligned rows whose lengths happen to sum to a multiple of block size.
   Keep both per-request builder checks and the gate's total-count check.
2. ~~What is token 156900?~~ Resolved: `<|role_end|>`, the chat-template
   role terminator, listed in `generation_config.json` as a stop token --
   treat as a hard stop via `eos_ids`.
3. Whether a 2.2-mini appears upstream (would make GPU phases cheap); check
   the `inclusionAI` org before each GPU phase.
4. Whether `LLaDA2_2_tech_report.pdf` documents deviations between the
   published checkpoint and the modeling file's decode loop; the modeling
   file is the operative reference either way.
5. CUDA-graph strategy for the eager edit-op scan (Section 12 Phase 4) --
   deferred to profiles.
6. How far the deterministic anti-loop deviation moves outputs on real
   inputs -- measure how often the reference's anti-loop fires in Phase 2
   traces; if it is rare, the deviation is near-free.

## 18. Revision Notes

**2026-09-08 implementation audit:** replaced stale patch-application status
with a checked completion matrix; fixed prompt EOS early stopping, literal
prompt edit-token consumption, pre-broadcast row-state divergence, and
Levenshtein decode-row selection; added runtime/online CPU regressions and
opt-in rank/state diagnostics. GPU submission was attempted but blocked by
node restrictions and login authentication; no GPU acceptance is claimed.


**2026-09-05 rewrite** (supersedes the copy embedded in the Phase 0 patch):

- **Status updated**: weights are now local and verified (the patch copy
  predates the download); the patch's clean application onto `3a68e8a` was
  verified; the runner hooks (`_make_decode_loop_state`,
  `_decoder_editing_kwargs`) are confirmed present on the branch from the
  2.1 Phase 4 work.
- **Corrected an overstatement in the old Section 0**: the patch does *not*
  contain the `RunnerConfig`/CLI fields or the `cli`
  block-length-alignment check -- the factory reads the knobs via `getattr`
  defaults. Those items moved to an explicit Phase 0.5 -- where it then
  turned out (2026-09-05) they were already on the branch via the user's
  `b4e6506` commit; Phase 0.5 only had to add `eos_ids` and the LLaDA
  `generation_block_size` wiring.
- **Resolved**: token 156900 is `<|role_end|>` (Section 17.2). The two-EOS
  design landed on the `decoder.eos_ids` mechanism because the executor
  already prefers it (`executor.py:113-116`) -- no engine change needed,
  which the old guide had left open.
- **Added**: the Stampede3 dense-only constraint inherited from the 2.1
  campaign; Phase 2's pass@1-as-judge fallback (2.1 lesson: cross-impl
  trace parity dies on numerics regardless of algorithm parity); the
  Phase 4 note that 2.1's `graph_fused_step` tail does not transfer to the
  sequential edit-op scan; measured 2.1-flash baselines for Phase 2/4
  comparison.
- Restructured to mirror the 2.1 guide's section layout so the two documents
  can be read side by side.
