# Dynamic Scheduler Correctness and Integration Plan

## Objective

Make `scheduler_policy=dynamic` affect the order in which runnable diffusion
requests receive decode work, while preserving native paged-scheduler ownership
of KV-cache pages and request lifecycle state.

The current implementation in
`python/fluxserve/backend/engine/dynamic_scheduler.py` collects trajectory
metrics, but only ranks requests in `_pending`. Once a request is admitted it is
never ranked again. Consequently, trajectory scores do not currently influence
execution; the implementation behaves like a cohort-based, request-ID-sorted
admission gate rather than a dynamic scheduler.

## Problems to fix

1. **Policy and execution are disconnected.** Active requests are not
   reprioritized after `observe_results()` updates their trajectory.
2. **Cold-start ordering is not FIFO.** Equal neutral scores are broken by
   `request_id`, which can produce arbitrary or lexicographic ordering.
3. **Cohort gating causes head-of-line blocking.** New work waits until every
   request in the current cohort completes a block, even when other requests
   could run safely.
4. **The native scheduler's population differs from the policy's cohort.**
   Requests remain active in the native scheduler while the adapter starts a
   new cohort, so the policy cannot describe or control the full runnable set.
5. **Most advertised features are absent from runtime metrics.** The runner
   currently emits block length, transferred tokens, remaining masks, and
   progress, but not confidence deficit, top-2 margin, fallback, readiness, or
   flip rate.
6. **Block-level observations lose intra-block convergence information.** A
   final transferred-token count does not reveal how many iterative forwards or
   fallbacks were needed.
7. **No starvation protection exists.** Prioritizing convergence can postpone
   difficult requests indefinitely under sustained load.
8. **`rank()` mutates scores.** Repeated ranking without new evidence changes
   difficulty, making scheduling frequency affect priority.
9. **Percentile scores are candidate-set-relative.** Scores change when the
   waiting set changes and are not comparable across cohorts.
10. **`finish()` may finish an unknown native request.** A pending request has
    not been submitted to the native scheduler, but `finish()` forwards the
    event unconditionally.
11. **Metric values are not finite/range checked.** NaN, infinity, and invalid
    fractions can poison ranking.

## Target architecture

Use the native paged scheduler as the owner of request and KV-cache lifecycle.
The dynamic policy should select or prioritize the next *runnable forward*;
requests must not be removed and resubmitted merely to change their order.

Preferred integration order:

1. Extend the native scheduler API with a priority/update operation for active
   requests, or add an execution-plan selection hook that accepts policy order.
2. At every forward/block boundary, publish trajectory updates for completed
   requests.
3. Ask the policy to rank all runnable active requests plus newly pending
   requests.
4. Apply the resulting order to the next native execution plan.
5. Add an age/fairness term and a bounded maximum wait so difficult requests
   cannot starve.

If the native scheduler cannot be reprioritized, implement an explicit
block-boundary runnable queue in the Python execution plane, but retain native
request/page state and document the additional synchronization constraints.

## Staged implementation plan

### Phase 0: establish a reproducible baseline

- Capture default, paged, and dynamic results with identical model, dataset,
  request rate, concurrency, and decode settings.
- Record p50/p90/p99 latency, completion rate, request throughput, and block
  iteration counts.
- Add a scheduler trace containing admission, plan, block completion, finish,
  and abort events keyed by request ID.

**Exit criterion:** baseline traces prove which requests actually execute in
each plan and provide a before/after comparison.

### Phase 1: make policy state safe and deterministic

- Keep cold-start ordering FIFO using an explicit arrival sequence number.
- Make `rank()` side-effect free; update EWMA state only in observation methods.
- Add finite-value and range validation for all metrics.
- Replace candidate-relative percentiles with bounded absolute normalization,
  or clearly separate absolute features from within-batch relative features.
- Add tests for duplicate IDs, empty input, one-request input, ties, NaN, and
  out-of-range values.

**Exit criterion:** identical inputs and observations produce identical ranks,
and ranking alone never changes policy state.

### Phase 2: repair lifecycle and admission semantics

- Track pending, active, runnable, finished, and aborted IDs separately.
- Guard native `finish()` so it is called only for submitted requests.
- Replace the all-members cohort barrier with block-boundary readiness for each
  request, unless the native scheduler explicitly requires a cohort barrier.
- Ensure a finished/aborted request cannot remain in any queue or cohort.
- Add tests for cancellation during pending, active, and boundary states.

**Exit criterion:** no request is lost, duplicated, finished twice, or blocked
behind an unrelated slow request.

### Phase 3: connect ranking to execution

- Define the scheduler/native API needed to prioritize active requests.
- Update active-request priorities after each `observe_results()` call.
- Ensure the next execution plan reflects the new priority order.
- Include newly admitted requests with a neutral prior and arrival age.
- Add fairness aging and a maximum consecutive bypass limit.

**Exit criterion:** a controlled test with deliberately different trajectories
shows the expected request order in subsequent plans; FIFO remains available as
a policy/configuration option.

### Phase 4: provide meaningful trajectory metrics

- Instrument the diffusion runner at every iterative forward, not only when a
  block completes.
- Emit confidence deficit, top-1/top-2 margin, readiness, flip rate, fallback
  status, masks before/after, transferred tokens, and iteration count.
- Define whether commit forwards are included and keep that convention stable.
- Aggregate metrics at block boundaries without losing worst-case or streak
  information.

**Exit criterion:** each policy feature is backed by a non-default runtime
metric, and synthetic trajectories classify as converging, near-ready,
stable-ambiguity, unstable-ambiguity, and stalled as intended.

### Phase 5: performance and rollout validation

- Benchmark default, paged, and dynamic across request rates and concurrency.
- Compare p50/p90/p99 latency, throughput, fairness, and scheduler overhead.
- Test short and long prompts, mixed output lengths, cancellations, and failed
  forwards.
- Verify no KV-cache leaks, stale native IDs, or unbounded policy-state growth.
- Keep dynamic opt-in until it meets a regression budget, for example no more
  than 5% p50/p90 regression and a documented p99 or throughput benefit.

**Exit criterion:** dynamic scheduling is enabled only for configurations with
  validated metrics and a documented fallback to paged/FIFO behavior.

## Required tests

- Unit tests for score normalization, smoothing, evidence, classification, and
  metric validation.
- Adapter tests for FIFO cold start, active reranking, partial block
  completion, cohort replacement, finish, abort, duplicate submission, and
  starvation aging.
- Native integration tests proving priority changes are reflected in execution
  plans without invalid cache operations.
- End-to-end tests covering concurrent requests and cancellation.
- Regression tests comparing dynamic-disabled behavior with the existing paged
  scheduler.

## Open design decisions

- Should dynamic prioritize likely-to-converge requests, difficult requests,
  or use a configurable objective?
- Can the C++ scheduler accept active-request priorities without moving cache
  pages?
- What is the fairness bound: maximum wait time, maximum bypasses, or weighted
  aging?
- Which metrics are cheap enough to collect on every forward in production?
- Should policy scores be absolute, percentile-based, or a hybrid?

## Success definition

Dynamic scheduling is complete when trajectory observations change the next
execution plan, request order is deterministic and starvation-bounded, native
request/cache lifecycle remains correct, runtime metrics support the advertised
features, and benchmarks demonstrate a measurable benefit under at least one
realistic workload without unacceptable latency regressions.
