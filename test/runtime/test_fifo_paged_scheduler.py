import pytest

from flux_scheduler import ExecutionEvent, ForwardEvent, RequestSpec, Scheduler, SchedulerConfig


def _scheduler(*, max_batch_size=8, max_scheduled_tokens=64, mixed=False):
    cfg = SchedulerConfig()
    cfg.page_size = 4
    cfg.decode_input_tokens = 4
    cfg.max_batch_size = max_batch_size
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.num_device_pages = 128
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True
    cfg.enable_mixed_prefill_decode = mixed
    return Scheduler(cfg)


def _request(request_id, length=8):
    spec = RequestSpec()
    spec.request_id = request_id
    spec.tokens = list(range(length))
    spec.prefill_length = length
    return spec


def _ids(plan):
    return [request_id for op in plan.forward for request_id in op.request_ids]


def _abort(scheduler, request_ids):
    events = ExecutionEvent()
    for request_id in request_ids:
        event = ForwardEvent.Abort()
        event.request_id = request_id
        events.add_event(event)
    scheduler.advance(events)


def test_first_plan_preserves_submission_order():
    scheduler = _scheduler()
    ids = ["request-2", "request-10", "request-1"]
    scheduler.submit_requests([_request(request_id) for request_id in ids])
    assert _ids(scheduler.next_execution_plan()) == ids


def test_batch_limit_selects_oldest_eligible_requests_across_plans():
    scheduler = _scheduler(max_batch_size=2)
    ids = ["request-2", "request-10", "request-1", "request-0", "request-3"]
    scheduler.submit_requests([_request(request_id) for request_id in ids])
    for start in range(0, len(ids), 2):
        expected = ids[start : start + 2]
        assert _ids(scheduler.next_execution_plan()) == expected
        _abort(scheduler, expected)
    assert _ids(scheduler.next_execution_plan()) == []


def test_fifo_spans_submission_calls():
    scheduler = _scheduler()
    scheduler.submit_requests([_request("request-2"), _request("request-10")])
    scheduler.submit_requests([_request("request-1"), _request("request-0")])
    assert _ids(scheduler.next_execution_plan()) == [
        "request-2", "request-10", "request-1", "request-0"
    ]


def test_partial_prefill_precedes_later_submissions():
    scheduler = _scheduler(max_scheduled_tokens=4)
    scheduler.submit_requests([_request("request-2", length=8)])
    assert _ids(scheduler.next_execution_plan()) == ["request-2"]
    scheduler.submit_requests([_request("request-1", length=4)])
    assert _ids(scheduler.next_execution_plan()) == ["request-2"]
    assert _ids(scheduler.next_execution_plan()) == ["request-1"]


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("decoding", [False, True])
def test_configured_prefill_decode_priority_is_preserved(mixed, decoding):
    scheduler = _scheduler(max_scheduled_tokens=4, mixed=mixed)
    scheduler.submit_requests([_request("request-2", length=4)])
    assert _ids(scheduler.next_execution_plan()) == ["request-2"]
    if decoding:
        assert _ids(scheduler.next_execution_plan()) == ["request-2"]
        assert scheduler.decoding_size() == 1
    scheduler.submit_requests([_request("request-1", length=4)])
    expected = "request-2" if mixed else "request-1"
    assert _ids(scheduler.next_execution_plan()) == [expected]


def test_duplicate_active_id_preserves_original_request_and_fifo_position():
    scheduler = _scheduler()
    scheduler.submit_requests([_request("request-2"), _request("request-10")])
    scheduler.submit_requests([
        _request("request-2", length=16),
        _request("request-1"),
        _request("request-1", length=12),
        _request("request-0"),
    ])
    assert scheduler.waiting_size() == 4
    plan = scheduler.next_execution_plan()
    assert _ids(plan) == ["request-2", "request-10", "request-1", "request-0"]
    assert [length for op in plan.forward for length in op.input_lengths] == [8] * 4


def test_duplicate_active_id_still_validates_request_spec():
    scheduler = _scheduler()
    scheduler.submit_requests([_request("request-2")])
    invalid = _request("request-2")
    invalid.prefill_length = len(invalid.tokens) + 1
    with pytest.raises(ValueError, match="prefill_length"):
        scheduler.submit_requests([invalid])
    assert _ids(scheduler.next_execution_plan()) == ["request-2"]


def test_fresh_schedulers_produce_identical_plans():
    snapshots = []
    for _ in range(3):
        scheduler = _scheduler(max_batch_size=2)
        scheduler.submit_requests([_request("request-2"), _request("request-10")])
        scheduler.submit_requests([_request("request-2"), _request("request-1")])
        plans = []
        for expected in (["request-2", "request-10"], ["request-1"]):
            plan = scheduler.next_execution_plan()
            assert _ids(plan) == expected
            plans.append([
                (
                    list(op.request_ids),
                    list(op.request_pool_indices),
                    list(op.input_lengths),
                    list(op.input_ids),
                    [list(pages) for pages in op.occupied_pages],
                )
                for op in plan.forward
            ])
            _abort(scheduler, expected)
        snapshots.append(plans)
    assert snapshots[0] == snapshots[1] == snapshots[2]
