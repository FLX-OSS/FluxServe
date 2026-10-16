"""A request the paged scheduler drops under KV exhaustion fails cleanly.

With the host cache disabled, flux_scheduler aborts the longest request on its
own when the device pool runs out and reports it nowhere. The engine must fail
that request instead of leaving its client waiting with its runner slot held,
and must not spin on the request-less forward op an idle scheduler returns:
that spin never yields, so the whole server stops answering.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from fluxserve.backend.engine.async_llm import AsyncLLM
from fluxserve.backend.engine.executor import ForwardStepResult
from fluxserve.backend.engine.io_struct import GenerateReqInput
from fluxserve.backend.utils.server_args import ServerArgs


class StepExecutor:
    """Each decode row commits ``advance`` tokens and reserves as many, like ISD.

    A request holds one of ``slots`` runner slots until it finishes or is
    released, and an op with a new request and no free slot fails before any
    compute, as the paged runners' ``_paged_slot`` does.
    """

    offload_execution = False

    def __init__(self, slots, advance=3):
        self.slots = slots
        self.advance = advance
        self.held = set()

    async def execute_forward_plan(self, op, states_by_id):
        request_ids = list(op.request_ids)
        for rid in request_ids:
            if rid not in self.held and len(self.held) >= self.slots:
                raise RuntimeError(
                    f"paged scheduled more concurrent requests than max_num_seqs={self.slots}"
                )
            self.held.add(rid)
        results = []
        prefill = int(op.num_extends())
        for row, rid in enumerate(request_ids):
            state = states_by_id.get(rid)
            if state is None:
                continue
            if row < prefill:
                results.append(ForwardStepResult(rid=rid, token_ids=[], text=""))
                continue
            remaining = state.max_new_tokens - len(state.output_ids)
            tokens = [7] * min(self.advance, remaining)
            finished = len(tokens) >= remaining
            results.append(ForwardStepResult(
                rid=rid, token_ids=tokens, text="", finished=finished,
                finish_reason="length" if finished else None,
                reserve_tokens=0 if finished else self.advance,
            ))
            if finished:
                self.held.discard(rid)
        return results

    async def release_requests(self, request_ids):
        self.held.difference_update(request_ids)


def run_overcommitted(max_num_seqs, num_device_pages, num_requests, max_tokens):
    """Run more KV demand than the pool holds; return the finals and the executor."""
    pytest.importorskip("flux_scheduler")
    from fluxserve.backend.engine.scheduler_adapter import PagedSchedulerAdapter

    outcome = {}

    async def scenario():
        adapter = PagedSchedulerAdapter(
            max_batch_size=max_num_seqs, max_scheduled_tokens=2048, page_size=32,
            num_device_pages=num_device_pages, max_model_len=4096, full_prompt_prefill=True,
        )
        executor = StepExecutor(slots=max_num_seqs)
        engine = AsyncLLM(
            ServerArgs(max_model_len=4096, max_num_seqs=max_num_seqs), executor,
            tokenizer=SimpleNamespace(), scheduler=adapter,
        )

        async def request(prompt_len, max_tokens):
            last = None
            async for output in engine.generate_request(GenerateReqInput(
                input_ids=list(range(1, prompt_len + 1)),
                sampling_params={"max_tokens": max_tokens, "ignore_eos": True},
            )):
                last = output
            return last

        finals = await asyncio.wait_for(
            asyncio.gather(*[request(150 + i, max_tokens) for i in range(num_requests)]),
            timeout=60,
        )
        outcome["executor"] = executor
        outcome["states_left"] = len(engine._states)
        # Still serving: the loop yields to new work instead of spinning.
        outcome["again"] = await asyncio.wait_for(
            asyncio.gather(*[request(100 + i, 64) for i in range(max_num_seqs)]), timeout=30
        )
        await engine.shutdown()
        outcome["finals"] = finals

    # A regression spins without yielding, so asyncio's own timeouts never
    # fire; watch the loop from another thread instead.
    def run():
        try:
            asyncio.run(scenario())
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(120)
    assert not worker.is_alive(), "the engine loop stopped yielding; a server would hang"
    if "error" in outcome:
        raise outcome["error"]

    finals = outcome["finals"]
    dropped = [output for output in finals if output.error]
    assert dropped, "the pool is too small for all requests"
    assert [output.error for output in dropped if "KV cache exhausted" not in output.error] == []
    assert len(dropped) + sum(output.finish_reason == "length" for output in finals) == num_requests
    assert outcome["executor"].held == set(), "a dropped request kept its runner slot"
    assert outcome["states_left"] == 0
    assert all(output.finish_reason == "length" and not output.error for output in outcome["again"])


def test_requests_dropped_under_kv_exhaustion_fail_and_the_engine_keeps_serving():
    # 160 pages of 32 hold 5120 tokens; six 2048-token requests need 13k.
    run_overcommitted(max_num_seqs=8, num_device_pages=160, num_requests=6, max_tokens=2048)


def test_a_dropped_request_frees_its_runner_slot_before_the_scheduler_reuses_it():
    # Eight requests wait behind four. A drop frees the victim's scheduler slot
    # at once, and the next plan admits a waiting request into it; the runner
    # must have released the victim by then, or that request fails for want of
    # a slot.
    run_overcommitted(max_num_seqs=4, num_device_pages=64, num_requests=12, max_tokens=1024)
