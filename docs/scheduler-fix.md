# Minimal Native FIFO Fix

## Summary

Make native scheduling FIFO within each existing scheduler state class. Preserve the current priority order between prefilling, submitted, decoding, and retracted requests, while replacing lexicographic request-ID tie-breaking with
submission order.

Implement this entirely inside flux-scheduler; no Python or binding API changes are required.

## Implementation Changes

- Add a std::uint64_t arrival_sequence member to native Request, supplied by Scheduler at construction and exposed through a read-only accessor.
- Add a monotonic next_arrival_sequence_ counter to Scheduler.
- In SubmitRequests(), assign sequence numbers in request_specs vector order. Increment the counter only for IDs that are not already present.
- Change the candidate comparator in newForwardOperation() to order by:
    1. existing state priority;
    2. arrival_sequence;
    3. request ID as a defensive final tie-breaker.

- Keep all existing batching, token-budget, cache allocation, prefill-first, and retraction behavior unchanged.
- Do not bound Python admission or change FifoPagedSchedulerAdapter._admit(): native scheduling already enforces max_batch_size and max_scheduled_tokens.
- Do not add arrival_sequence to RequestSpec or Python bindings. The sequence is scheduler-local implementation state.

## Tests

Add native-binding tests under test/runtime and run them through:

./gh200/run_flux_container.sh python -m pytest -q \
test/runtime/test_paged_scheduler_block_prefill.py \
test/runtime/test_fifo_paged_scheduler.py

Cover these cases:

- Submit IDs in deliberately non-lexicographic order such as ["request-2", "request-10", "request-1"]; verify the first plan preserves submission order.
- Set max_batch_size=2; verify successive plans choose the oldest eligible requests first.
- Submit requests in separate calls; verify FIFO order spans submission batches.
- Verify state priority remains unchanged: a partially prefilling request still precedes later submitted requests, and the configured prefill/decode priority still applies.
- Submit a duplicate active ID; verify it neither replaces the original request nor consumes an arrival-sequence position.
- Repeat the same submissions with fresh scheduler instances and verify identical plans, preserving deterministic scheduling.

## Acceptance Criteria
- Native plans remain deterministic across repeated runs.
- No public Python API, scheduler configuration, cache lifecycle, or batching behavior changes.
- Existing paged scheduler tests continue to pass.
- “FIFO” means FIFO among requests in the same native state-priority class, not strict global run-to-completion ordering.
- All scheduler replicas, if present, receive identical ordered SubmitRequests() calls, so locally assigned sequence numbers remain deterministic.
- Duplicate-ID rejection semantics are otherwise unchanged; broader request-ID reuse handling is outside this minimal fix.
- The new ordering applies to every user of the native scheduler, including default, paged, and dynamic policies. For dynamic scheduling, this preserves the order in which the Python policy submits equally eligible requests.