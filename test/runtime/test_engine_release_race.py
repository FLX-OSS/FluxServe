"""Request release must wait for an already-dispatched background forward."""

import asyncio
from types import SimpleNamespace
import threading

import pytest

from fluxserve.backend.engine.async_llm import AsyncLLM
from fluxserve.backend.engine.request import RequestState
from fluxserve.backend.utils.server_args import ServerArgs


@pytest.mark.parametrize("cancel_forward", [False, True])
def test_release_cannot_race_or_be_undone_by_an_inflight_forward(cancel_forward):
    async def scenario():
        started = threading.Event()
        finish = threading.Event()
        slots = {}
        events = []

        class Executor:
            offload_execution = True

            async def forward(self):
                started.set()
                assert finish.wait(5), "test failed to unblock the worker"
                slots["cancelled"] = 0
                events.append("forward_done")

            async def release_requests(self, ids):
                events.append("release")
                for rid in ids:
                    slots.pop(rid, None)

        executor = Executor()
        engine = AsyncLLM(ServerArgs(), executor, tokenizer=SimpleNamespace())
        forward = asyncio.create_task(engine._execute(executor.forward))
        assert await asyncio.to_thread(started.wait, 5)
        if cancel_forward:
            forward.cancel()
        release = asyncio.create_task(engine._release_executor_requests(["cancelled"]))
        # Let the release task try to run while the worker is still blocked.
        await asyncio.sleep(0.02)
        was_released_early = bool(events)
        finish.set()
        if cancel_forward:
            with pytest.raises(asyncio.CancelledError):
                await forward
        else:
            await forward
        await release
        assert not was_released_early
        assert events == ["forward_done", "release"]
        assert slots == {}

    asyncio.run(scenario())


def test_abort_releases_the_slot_when_cancelled_at_the_executor_lock():
    """The disconnect path aborts from inside its own cancelled task, and the
    release waits on the executor lock behind an in-flight forward. Losing that
    release strands the request's runner slot for the life of the server."""
    async def scenario():
        started = threading.Event()
        finish = threading.Event()
        slots = {"r0": 0}
        events = []

        class Executor:
            offload_execution = True

            async def forward(self):
                started.set()
                assert finish.wait(5), "test failed to unblock the worker"

            async def release_requests(self, ids):
                events.append("release")
                for rid in ids:
                    slots.pop(rid, None)

        executor = Executor()
        engine = AsyncLLM(ServerArgs(), executor, tokenizer=SimpleNamespace())
        engine.scheduler = SimpleNamespace(abort=lambda rid: events.append("scheduler"))
        state = RequestState(rid="r0", input_ids=[1], max_new_tokens=1)
        engine._states["r0"] = state

        forward = asyncio.create_task(engine._execute(executor.forward))
        assert await asyncio.to_thread(started.wait, 5)

        async def disconnected():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await engine.abort("r0", "client disconnected")
                raise

        request = asyncio.create_task(disconnected())
        await asyncio.sleep(0)
        request.cancel()
        # anyio re-delivers the cancel at every checkpoint until the scope
        # exits, and acquiring the executor lock is one.
        for _ in range(4):
            await asyncio.sleep(0)
            request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert events == ["scheduler"], "release cannot precede the live forward"
        assert state.finished_reason == "abort", "the client is told before the wait"

        finish.set()
        await forward
        for _ in range(100):
            if "release" in events:
                break
            await asyncio.sleep(0.01)
        assert slots == {}, "the disconnected request kept its runner slot"
        assert events == ["scheduler", "release"]

    asyncio.run(scenario())
