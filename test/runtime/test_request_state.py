from fluxserve.backend.engine.processor import OutputProcessor
from fluxserve.backend.engine.request import RequestState


def test_output_processor_delegates_normal_completion_to_request_state():
    state = RequestState(rid="req-1", input_ids=[1, 2], max_new_tokens=4)
    processor = OutputProcessor()

    output = processor.make_output(state, [3], "hello", "stop")

    assert state.output_ids == [3]
    assert state.decoded_text == "hello"
    assert state.finished_reason == "stop"
    assert output.finish_reason == "stop"
    assert output.meta["prompt_tokens"] == 2
    assert output.meta["completion_tokens"] == 1


def test_request_state_owns_error_and_abort_transitions():
    processor = OutputProcessor()

    error_state = RequestState(rid="error", input_ids=[], max_new_tokens=1)
    error = processor.make_error_output(error_state, "failed")
    assert error_state.finished_reason == "error"
    assert error.finish_reason == "error"
    assert error.error == "failed"

    abort_state = RequestState(rid="abort", input_ids=[], max_new_tokens=1)
    aborted = processor.make_abort_output(abort_state, "cancelled")
    assert abort_state.finished_reason == "abort"
    assert aborted.finish_reason == "abort"
    assert aborted.error == "cancelled"


def test_request_state_latency_uses_monotonic_clock(monkeypatch):
    values = iter([11.0, 12.0, 13.0, 14.0])
    monkeypatch.setattr("fluxserve.backend.engine.request.time.monotonic", lambda: next(values))

    state = RequestState(rid="timed", input_ids=[], max_new_tokens=1, created_time=10.0)
    state.mark_queued()
    state.mark_scheduled()
    state.mark_execution_done()
    state.append_output([1], "x", "stop")

    assert state.queue_latency_s == 1.0
    assert state.execution_latency_s == 1.0
    assert state.e2e_latency_s == 4.0
