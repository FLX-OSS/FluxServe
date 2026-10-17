"""Rank 0 -> worker command round trip of DistributedGenerationExecutor.

``torch.distributed.broadcast_object_list`` is replaced by an in-memory
channel: rank 0 appends each broadcast command, and the worker loop pops them
in order. That checks the wire format end to end -- every command rank 0 sends
must be decoded by a worker into the same call on its base executor.
"""

import asyncio
from types import SimpleNamespace

import pytest

from fluxserve.backend.distributed.launch import DistributedContext
from fluxserve.backend.engine import distributed_executor as module
from fluxserve.backend.engine.distributed_executor import DistributedGenerationExecutor
from fluxserve.backend.engine.request import RequestState


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    async def startup(self):
        self.calls.append(("startup",))
        return {"graphs": 1}

    async def shutdown(self):
        self.calls.append(("shutdown",))

    async def execute_batch(self, requests):
        self.calls.append(("execute_batch", [_state_fields(req) for req in requests]))
        return []

    async def execute_forward_plan(self, op, states_by_id):
        self.calls.append(
            (
                "execute_forward_plan",
                _op_fields(op),
                {rid: _state_fields(state) for rid, state in states_by_id.items()},
            )
        )
        return []

    async def release_requests(self, request_ids):
        self.calls.append(("release_requests", list(request_ids)))


def _state_fields(state):
    return {
        "rid": state.rid,
        "input_ids": state.input_ids,
        "max_new_tokens": state.max_new_tokens,
        "ignore_eos": state.ignore_eos,
        "prompt_text": state.prompt_text,
        "sampling_params": state.sampling_params,
        "output_ids": state.output_ids,
        "decoded_text": state.decoded_text,
        "plan_prefill_done": state.plan_prefill_done,
        "current_decode_block": state.current_decode_block,
    }


def _op_fields(op):
    return {
        "request_ids": list(op.request_ids),
        "input_ids": list(op.input_ids),
        "input_lengths": list(op.input_lengths),
        "extend_prefix_lens": list(op.extend_prefix_lens),
        "occupied_pages": [list(pages) for pages in op.occupied_pages],
        "num_extends": op.num_extends(),
    }


def _state(rid, **overrides):
    state = RequestState(
        rid=rid,
        input_ids=[11, 12, 13],
        max_new_tokens=32,
        ignore_eos=True,
        prompt_text="hello",
        sampling_params={"temperature": 0.5, "seed": 7},
    )
    state.output_ids = [21, 22]
    state.decoded_text = "ab"
    state.plan_prefill_done = True
    state.current_decode_block = 2
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _plan_op():
    # Shaped like flux_scheduler's forward op: num_extends is a method.
    return SimpleNamespace(
        request_ids=["a", "b"],
        input_ids=[1, 2, 3, 4],
        input_lengths=[3, 1],
        extend_prefix_lens=[0, 32],
        occupied_pages=[[0], [1, 2]],
        num_extends=lambda: 1,
    )


@pytest.fixture
def channel(monkeypatch):
    sent = []

    def broadcast_object_list(payload, src):
        assert src == 0
        if payload[0] is None:
            payload[0] = sent.pop(0)
        else:
            sent.append(payload[0])

    monkeypatch.setattr(module.dist, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    return sent


def _pair():
    leader_base, worker_base = _RecordingExecutor(), _RecordingExecutor()
    leader = DistributedGenerationExecutor(
        leader_base, DistributedContext(rank=0, world_size=2)
    )
    worker = DistributedGenerationExecutor(
        worker_base, DistributedContext(rank=1, local_rank=1, world_size=2)
    )
    return leader, leader_base, worker, worker_base


def test_every_rank0_command_replays_identically_on_the_worker(channel):
    leader, leader_base, worker, worker_base = _pair()
    states = {"a": _state("a"), "b": _state("b", output_ids=[], current_decode_block=0)}

    async def scenario():
        assert await leader.startup() == {"graphs": 1}
        await leader.execute_batch([states["a"]])
        await leader.execute_forward_plan(_plan_op(), states)
        await leader.release_requests(["a"])
        await leader.release_requests([])  # no-op: nothing is broadcast
        await leader.shutdown_workers()
        assert [command["kind"] for command in channel] == [
            "startup", "generate", "forward_plan", "release_requests", "shutdown",
        ]
        await worker.run_worker_loop()

    asyncio.run(scenario())

    assert channel == [], "the worker consumed every broadcast command"
    assert worker_base.calls == leader_base.calls
    kinds = [call[0] for call in worker_base.calls]
    assert kinds == [
        "startup", "execute_batch", "execute_forward_plan", "release_requests", "shutdown",
    ]
    _, op, replayed_states = worker_base.calls[2]
    assert op == _op_fields(_plan_op())
    assert replayed_states == {rid: _state_fields(state) for rid, state in states.items()}


def test_single_rank_executor_never_broadcasts(channel):
    base = _RecordingExecutor()
    executor = DistributedGenerationExecutor(base, DistributedContext())

    async def scenario():
        await executor.startup()
        await executor.execute_batch([_state("a")])
        await executor.execute_forward_plan(_plan_op(), {"a": _state("a"), "b": _state("b")})
        await executor.release_requests(["a"])
        await executor.run_worker_loop()  # returns at once when not distributed
        await executor.shutdown_workers()

    asyncio.run(scenario())

    assert channel == []
    assert [call[0] for call in base.calls] == [
        "startup", "execute_batch", "execute_forward_plan", "release_requests", "shutdown",
    ]


@pytest.mark.parametrize(
    "call",
    [
        lambda executor: executor.execute_batch([]),
        lambda executor: executor.execute_forward_plan(_plan_op(), {}),
        lambda executor: executor.release_requests(["a"]),
    ],
)
def test_workers_refuse_leader_only_calls(channel, call):
    _, _, worker, _ = _pair()
    with pytest.raises(RuntimeError, match="only be called on rank 0"):
        asyncio.run(call(worker))
    assert channel == []


def test_leader_refuses_to_run_the_worker_loop(channel):
    leader, _, _, _ = _pair()
    with pytest.raises(RuntimeError, match="nonzero ranks"):
        asyncio.run(leader.run_worker_loop())


def test_worker_rejects_unknown_commands(channel):
    _, _, worker, _ = _pair()
    channel.append({"kind": "bogus"})
    with pytest.raises(RuntimeError, match="Unknown distributed executor command"):
        asyncio.run(worker.run_worker_loop())


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"occupied_pages": [[0]]}, "must match request_ids length"),
        ({"num_extends": 3}, "num_extends is out of range"),
    ],
)
def test_worker_validates_forward_plan_payloads(channel, overrides, match):
    _, _, worker, _ = _pair()
    plan = {
        "request_ids": ["a", "b"],
        "input_ids": [1],
        "input_lengths": [1, 1],
        "extend_prefix_lens": [0, 0],
        "occupied_pages": [[0], [1]],
        "num_extends": 1,
        **overrides,
    }
    channel.append({"kind": "forward_plan", "plan": plan, "requests": []})
    with pytest.raises(ValueError, match=match):
        asyncio.run(worker.run_worker_loop())
