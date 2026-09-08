from types import SimpleNamespace

from fluxserve.backend.engine.scheduler_adapter import (
    DynamicSchedulerAdapter,
    DynamicSchedulingPolicy,
)


def test_policy_uses_neutral_percentile_for_single_request():
    policy = DynamicSchedulingPolicy()
    ranked = policy.rank(["one"])
    assert ranked[0].request_id == "one"
    assert ranked[0].evidence == 0.0
    assert ranked[0].difficulty_score == 0.5


def test_policy_tracks_block_evidence_and_stalled_state():
    policy = DynamicSchedulingPolicy()
    metrics = {
        "remaining_masks": 10,
        "block_length": 16,
        "transferred_tokens": 1,
        "confidence_deficit": 0.5,
        "top2_margin": 0.02,
        "is_fallback_forward": True,
        "readiness_fraction": 0.0,
        "flip_rate": 0.0,
    }
    for _ in range(3):
        policy.observe_block("stalled", metrics)
    result = policy.rank(["stalled"])[0]
    assert result.observed_forwards == 3
    assert result.evidence == 1.0
    assert result.state == "stalled"


class _Native:
    def __init__(self):
        self.submitted = []
        self.finished = []
        self.aborted = []

    def submit(self, requests):
        self.submitted.append([request.rid for request in requests])

    def next_plan(self):
        return None

    def advance_forward(self, **kwargs):
        pass

    def finish(self, rid):
        self.finished.append(rid)

    def abort(self, rid):
        self.aborted.append(rid)


def test_adapter_keeps_cohort_stable_until_block_boundary():
    native = _Native()
    adapter = DynamicSchedulerAdapter(native, max_batch_size=2)
    requests = [SimpleNamespace(rid="a"), SimpleNamespace(rid="b"), SimpleNamespace(rid="c")]
    adapter.submit(requests)
    assert native.submitted == [["a", "b"]]

    adapter.submit([requests[2]])
    assert native.submitted == [["a", "b"]]

    adapter.observe_results([
        SimpleNamespace(rid="a", decode_block_completed=True, finished=False, trajectory_metrics={}),
    ])
    assert native.submitted == [["a", "b"]]

    adapter.observe_results([
        SimpleNamespace(rid="b", decode_block_completed=True, finished=False, trajectory_metrics={}),
    ])
    assert native.submitted == [["a", "b"], ["c"]]


def test_adapter_removes_pending_abort_and_admits_next_request():
    native = _Native()
    adapter = DynamicSchedulerAdapter(native, max_batch_size=1)
    first, second = SimpleNamespace(rid="first"), SimpleNamespace(rid="second")
    adapter.submit([first, second])
    adapter.abort("second")
    adapter.observe_results([
        SimpleNamespace(rid="first", decode_block_completed=True, finished=True, trajectory_metrics={}),
    ])
    assert native.submitted == [["first"]]
    assert native.aborted == []
