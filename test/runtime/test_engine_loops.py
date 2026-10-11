"""Both AsyncLLM scheduling loops, driven end to end by fake executors.

The FIFO loop hands whole requests to ``execute_batch``; the execution-plan
loop asks a plan scheduler for forward ops and streams per-op results. These
tests pin the observable contract of each loop -- what the client receives,
what the scheduler is told, what the executor is asked to release, and how
the metrics move -- so the loops can be restructured safely.
"""

import asyncio
from types import SimpleNamespace

from fluxserve.backend.engine.async_llm import AsyncLLM
from fluxserve.backend.engine.executor import ExecutionResult, ForwardStepResult
from fluxserve.backend.engine.io_struct import GenerateReqInput
from fluxserve.backend.utils.server_args import ServerArgs


def _engine(executor, scheduler=None, **server_args):
    args = ServerArgs(**{"max_num_seqs": 4, "max_model_len": 256, **server_args})
    return AsyncLLM(args, executor, tokenizer=SimpleNamespace(), scheduler=scheduler)


async def _collect(engine, rid, input_ids, max_tokens=8):
    request = GenerateReqInput(
        input_ids=input_ids, rid=rid, sampling_params={"max_tokens": max_tokens}
    )
    return [output async for output in engine.generate_request(request)]


async def _run(engine, *requests):
    try:
        return await asyncio.wait_for(
            asyncio.gather(*(_collect(engine, *request) for request in requests)),
            timeout=5,
        )
    finally:
        await engine.shutdown()


# --------------------------------------------------------------------------
# FIFO loop
# --------------------------------------------------------------------------


class _BatchExecutor:
    def __init__(self, results=None, error=None):
        self.results = results or {}
        self.error = error
        self.batches = []

    async def execute_batch(self, states):
        self.batches.append([(state.rid, state.max_new_tokens) for state in states])
        if self.error is not None:
            raise self.error
        return [self.results[state.rid] for state in states if state.rid in self.results]


def test_fifo_loop_streams_one_final_output_per_request():
    executor = _BatchExecutor(
        results={
            "a": ExecutionResult(rid="a", token_ids=[5, 6], text="ab", finish_reason="stop"),
            "b": ExecutionResult(rid="b", token_ids=[7, 8, 9], text="cde"),
        }
    )
    engine = _engine(executor)

    out_a, out_b = asyncio.run(_run(engine, ("a", [1, 2, 3], 4), ("b", [1], 3)))

    assert len(out_a) == 1 and len(out_b) == 1
    assert (out_a[0].token_ids, out_a[0].text, out_a[0].finish_reason) == ([5, 6], "ab", "stop")
    assert (out_b[0].token_ids, out_b[0].text, out_b[0].finish_reason) == ([7, 8, 9], "cde", "length")
    assert out_a[0].meta["prompt_tokens"] == 3
    assert out_a[0].meta["completion_tokens"] == 2
    assert out_b[0].meta["completion_tokens"] == 3
    assert out_a[0].error is None

    # Every request reached the executor exactly once, with its clamped budget.
    scheduled = sorted(entry for batch in executor.batches for entry in batch)
    assert scheduled == [("a", 4), ("b", 3)]

    assert engine._states == {}
    assert engine.scheduler._active == set()
    metrics = engine.get_metrics_snapshot()
    assert metrics["total_requests"] == 2
    assert metrics["successful_requests"] == 2
    assert metrics["failed_requests"] == 0
    assert metrics["running_requests"] == 0
    assert metrics["waiting_requests"] == 0
    assert metrics["prompt_tokens"] == 4
    assert metrics["generated_tokens"] == 5


def test_fifo_loop_batches_respect_max_num_seqs():
    executor = _BatchExecutor(
        results={rid: ExecutionResult(rid=rid, token_ids=[1], text="x") for rid in "abc"}
    )
    engine = _engine(executor, max_num_seqs=2)

    asyncio.run(_run(engine, ("a", [1]), ("b", [1]), ("c", [1])))

    assert all(len(batch) <= 2 for batch in executor.batches)
    assert sorted(rid for batch in executor.batches for rid, _ in batch) == ["a", "b", "c"]


def test_fifo_loop_reports_executor_failure_to_every_request_in_the_batch():
    executor = _BatchExecutor(error=RuntimeError("boom"))
    engine = _engine(executor)

    (out_a,) = asyncio.run(_run(engine, ("a", [1, 2])))

    assert len(out_a) == 1
    assert out_a[0].finish_reason == "error"
    assert out_a[0].error == "boom"
    assert engine._states == {}
    assert engine.scheduler._active == set()
    metrics = engine.get_metrics_snapshot()
    assert metrics["failed_requests"] == 1
    assert metrics["successful_requests"] == 0
    assert metrics["running_requests"] == 0


def test_rejected_request_is_reported_without_reaching_the_executor():
    executor = _BatchExecutor()
    engine = _engine(executor, max_model_len=4)

    (out,) = asyncio.run(_run(engine, ("too-long", [1, 2, 3, 4, 5])))

    assert len(out) == 1
    assert out[0].finish_reason == "error"
    assert "exceeds max_model_len=4" in out[0].error
    assert executor.batches == []
    metrics = engine.get_metrics_snapshot()
    assert metrics["total_requests"] == 1
    assert metrics["failed_requests"] == 1


# --------------------------------------------------------------------------
# Execution-plan loop
# --------------------------------------------------------------------------


class _PlanScheduler:
    """Emits one forward op covering every live request, like a tiny paged plan."""

    uses_execution_plans = True

    def __init__(self, cache_ops=()):
        self.pending = []
        self.cache_ops = list(cache_ops)
        self.advances = []
        self.aborted = []

    def submit(self, states):
        self.pending.extend(state.rid for state in states)

    def next_plan(self):
        forward = [SimpleNamespace(request_ids=list(self.pending))] if self.pending else []
        return SimpleNamespace(cache=list(self.cache_ops), forward=forward)

    def advance_forward(self, *, token_results, reserve_tokens, finished_ids):
        self.advances.append(
            {
                "token_results": dict(token_results),
                "reserve_tokens": dict(reserve_tokens),
                "finished_ids": list(finished_ids),
            }
        )
        for rid in finished_ids:
            self.pending.remove(rid)

    def abort(self, rid):
        self.aborted.append(rid)
        if rid in self.pending:
            self.pending.remove(rid)


class _PlanExecutor:
    """Replays a scripted list of step results per request id."""

    def __init__(self, steps=None, error=None):
        self.steps = {rid: list(results) for rid, results in (steps or {}).items()}
        self.error = error
        self.calls = []
        self.released = []

    async def execute_forward_plan(self, op, states_by_id):
        self.calls.append(sorted(states_by_id))
        if self.error is not None:
            raise self.error
        return [self.steps[rid].pop(0) for rid in op.request_ids if rid in states_by_id]

    async def release_requests(self, request_ids):
        self.released.append(sorted(request_ids))


def test_plan_loop_streams_each_decode_block_and_drives_the_scheduler():
    executor = _PlanExecutor(
        steps={
            "a": [
                ForwardStepResult(
                    rid="a", token_ids=[5, 6], text="ab",
                    reserve_tokens=4, decode_block_completed=True,
                ),
                ForwardStepResult(
                    rid="a", token_ids=[7], text="c",
                    finished=True, finish_reason="stop", decode_block_completed=True,
                ),
            ]
        }
    )
    scheduler = _PlanScheduler()
    engine = _engine(executor, scheduler)
    seen_states = []
    original = executor.execute_forward_plan

    async def spy(op, states_by_id):
        seen_states.extend(states_by_id.values())
        return await original(op, states_by_id)

    executor.execute_forward_plan = spy

    (out,) = asyncio.run(_run(engine, ("a", [1, 2, 3])))

    assert [(o.token_ids, o.text, o.finish_reason) for o in out] == [
        ([5, 6], "ab", None),
        ([7], "c", "stop"),
    ]
    assert out[0].meta == {}
    assert out[1].meta["completion_tokens"] == 3
    assert scheduler.advances == [
        {"token_results": {"a": [5, 6]}, "reserve_tokens": {"a": 4}, "finished_ids": []},
        {"token_results": {"a": [7]}, "reserve_tokens": {}, "finished_ids": ["a"]},
    ]
    assert executor.released == [["a"]]
    state = seen_states[0]
    assert state.output_ids == [5, 6, 7]
    assert state.decoded_text == "abc"
    assert state.current_decode_block == 2
    assert engine._states == {}
    metrics = engine.get_metrics_snapshot()
    assert metrics["successful_requests"] == 1
    assert metrics["generated_tokens"] == 3
    assert metrics["running_requests"] == 0


def test_plan_loop_finishes_requests_independently_within_one_op():
    executor = _PlanExecutor(
        steps={
            "a": [ForwardStepResult(rid="a", token_ids=[5], text="a", finished=True, finish_reason="stop")],
            "b": [
                ForwardStepResult(rid="b", token_ids=[6], text="b", reserve_tokens=4),
                ForwardStepResult(rid="b", token_ids=[7], text="c", finished=True, finish_reason="length"),
            ],
        }
    )
    scheduler = _PlanScheduler()
    engine = _engine(executor, scheduler)

    out_a, out_b = asyncio.run(_run(engine, ("a", [1]), ("b", [1])))

    assert [(o.token_ids, o.finish_reason) for o in out_a] == [([5], "stop")]
    assert [(o.token_ids, o.finish_reason) for o in out_b] == [([6], None), ([7], "length")]
    # After "a" finishes it is no longer offered to the executor.
    assert executor.calls[-1] == ["b"]
    assert sorted(rid for released in executor.released for rid in released) == ["a", "b"]
    finished = [rid for advance in scheduler.advances for rid in advance["finished_ids"]]
    assert sorted(finished) == ["a", "b"]


def test_plan_loop_fails_and_releases_requests_when_the_executor_raises():
    executor = _PlanExecutor(error=RuntimeError("kv exhausted"))
    scheduler = _PlanScheduler()
    engine = _engine(executor, scheduler)

    (out,) = asyncio.run(_run(engine, ("a", [1, 2])))

    assert len(out) == 1
    assert out[0].finish_reason == "error"
    assert out[0].error == "kv exhausted"
    assert scheduler.aborted == ["a"]
    assert scheduler.advances == []
    assert executor.released == [["a"]]
    assert engine._states == {}
    assert engine.get_metrics_snapshot()["failed_requests"] == 1


def test_plan_loop_rejects_cache_operations():
    executor = _PlanExecutor()
    scheduler = _PlanScheduler(cache_ops=[object()])
    engine = _engine(executor, scheduler)

    (out,) = asyncio.run(_run(engine, ("a", [1])))

    assert len(out) == 1
    assert out[0].finish_reason == "error"
    assert "cache operations" in out[0].error
    assert executor.calls == []
    assert scheduler.aborted == ["a"]
    assert executor.released == [["a"]]
