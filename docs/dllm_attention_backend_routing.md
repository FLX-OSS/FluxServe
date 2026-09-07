# dLLM attention backend routing

FluxServe should select an attention backend once for each scheduled model
forward, not once in every transformer layer. The selected plan is stored in
`ForwardBatch.attention_execution_plan`; every layer then consumes the same
packing, cache, and backend decision.

The routing pipeline is:

1. The scheduler constructs one or more legal `AttentionScheduleCandidate`s
   under request, token, and KV-page budgets. For example, a conservative
   candidate has separate prefill and denoise forwards, while another candidate
   has one mixed forward.
2. The runner converts each candidate into `AttentionBatchCharacteristics`.
   A work item records the semantic phase (`prefill` or `denoise`), Q/KV length,
   and query offset. Semantic phase is required because dLLM decode commonly
   has `Q=block_length`, not autoregressive `Q=1`.
3. Installed adapters advertise `AttentionBackendCapability`. Correctness
   constraints are applied before any performance comparison.
4. A profile-backed `AttentionCostModel` predicts latency for compatible
   choices. `select_schedule()` compares the sum of split-forward estimates
   with a fused-forward estimate. The selector freezes each resulting
   `AttentionExecutionPlan` on its `ForwardBatch`.
5. The attention operator executes that plan. It may validate tensor shape and
   dtype, but must not change the backend independently in different layers.
   The current patch establishes this selection contract; FA4 and POD execution
   adapters still need to be connected before those capabilities are advertised
   by the production runner.

## Why POD belongs partly in the scheduler

`BatchPODWithPagedKVCacheWrapper` is not merely another implementation of the
same single-request call. Its benefit comes from placing compatible prefill and
decode work in one launch. For dLLM, block denoise should be represented as
P-side block-query work unless a future kernel explicitly supports it on the D
side. Therefore POD is compatible only when all of the following are true:

- the scheduler emitted a genuinely mixed prefill/denoise model forward;
- prefill was decomposed into block-causal tasks with a reversible output map;
- Q/KV page metadata and offsets are shared by both phases;
- the installed POD kernel supports the model geometry and dtype;
- the profile model predicts that the fused plan beats split alternatives.

Merely seeing both request types in a queue is not enough. The current paged
execution path calls prefill and decode separately, so it must advertise
`scheduler_fused=False` and POD will be rejected by the selector.

## Cost-model boundary

The initial selector accepts a callback through `CallableCostModel`. Production
deployment should replace it with a profile table keyed at least by GPU,
backend/build version, dtype, head geometry, page/block size, phase counts,
total Q tokens, and a bucketed KV-length distribution. Unknown shapes should
fall back to a configured safe backend and optionally be profiled outside the
latency-critical request.

The schedule comparison initially sums predicted attention costs. A fuller
model may also include packing, planning, model-launch, and output-scatter
overheads. Cold-start/JIT cost and steady-state kernel latency should be separate values.
A POD shape that wins after warmup may still be a poor first-request choice.

## Integration sequence

1. Attach BatchPrefill capability and plans to the existing paged runner while
   preserving its current execution behavior.
2. Add an FA4 adapter that consumes paged KV and virtual block-task metadata;
   obtain build-specific support from vLLM instead of hard-coding head dims.
3. Add a POD all-P-side adapter and correctness tests against FP32 SDPA.
4. Extend the paged scheduler/runner with a mixed-forward candidate and output
   scatter map. Only then advertise `scheduler_fused=True`.
5. Feed benchmark results into the cost model and compare full model-forward
   latency, not attention launch count alone.
