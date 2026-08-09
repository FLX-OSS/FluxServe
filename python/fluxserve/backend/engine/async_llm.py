# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import AsyncIterator

from fluxserve.backend.engine.executor import GenerationExecutor
from fluxserve.backend.engine.io_struct import GenerateReqInput, GenerateReqOutput
from fluxserve.backend.engine.processor import InputProcessor, OutputProcessor
from fluxserve.backend.engine.request import RequestState
from fluxserve.backend.engine.scheduler_adapter import DefaultSchedulerAdapter
from fluxserve.backend.metrics.engine import EngineMetrics
from fluxserve.backend.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


class AsyncLLM:
    def __init__(
        self,
        server_args: ServerArgs,
        executor: GenerationExecutor,
        tokenizer=None,
        scheduler=None,
    ):
        self.server_args = server_args
        self.executor = executor
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                server_args.model_name,
                trust_remote_code=server_args.trust_remote_code,
            )
        self.tokenizer = tokenizer
        self.scheduler = (
            DefaultSchedulerAdapter(server_args.max_num_seqs)
            if scheduler is None
            else scheduler
        )
        self.input_processor = InputProcessor(server_args, self.tokenizer)
        self.output_processor = OutputProcessor()
        self.metrics = EngineMetrics()
        self._states: dict[str, RequestState] = {}
        self._new_request_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._closed = False

    async def _execute(self, method, *args):
        """Run synchronous CUDA/NCCL coroutine bodies away from the HTTP loop."""
        if not getattr(self.executor, "offload_execution", False):
            result = method(*args)
            return await result if inspect.isawaitable(result) else result

        def run():
            result = method(*args)
            if inspect.isawaitable(result):
                return asyncio.run(result)
            return result

        return await asyncio.to_thread(run)

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    async def startup_executor(self) -> None:
        startup = getattr(self.executor, "startup", None)
        if startup is not None:
            await self._execute(startup)

    async def shutdown(self) -> None:
        self._closed = True
        self._new_request_event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def generate_request(self, obj: GenerateReqInput) -> AsyncIterator[GenerateReqOutput]:
        await self.start()
        try:
            states = self.input_processor.make_states(obj)
        except Exception as exc:  # noqa: BLE001
            self.metrics.record_failed_before_submit()
            yield GenerateReqOutput(rid="", error=str(exc), finish_reason="error")
            return

        if len(states) != 1:
            outputs = await asyncio.gather(*[self._collect_one(state) for state in states])
            for output in outputs:
                yield output
            return

        state = states[0]
        state.mark_queued()
        self._states[state.rid] = state
        self.metrics.record_submitted(state.prompt_token_count)
        self.scheduler.submit([state])
        self._new_request_event.set()
        try:
            while True:
                output = await state.queue.get()
                yield output
                if output.finished:
                    break
        except asyncio.CancelledError:
            await self.abort(state.rid, "client disconnected")
            raise

    async def abort(self, rid: str, reason: str = "aborted") -> None:
        state = self._states.get(rid)
        self.scheduler.abort(rid)
        await self._release_executor_requests([rid])
        if state is not None and not state.finished:
            output = self.output_processor.make_abort_output(state, reason)
            self.metrics.record_aborted(state)
            self._states.pop(state.rid, None)
            await state.queue.put(output)

    def get_metrics_snapshot(self) -> dict[str, int | float]:
        snapshot = self.metrics.snapshot()
        stats = getattr(self.executor, "cuda_graph_stats", None)
        if stats is not None:
            snapshot.update({f"cuda_graph_{k}": v for k, v in stats().items()})
        return snapshot

    async def _collect_one(self, state: RequestState) -> GenerateReqOutput:
        output = None
        async for chunk in self.generate_request(
            GenerateReqInput(
                input_ids=state.input_ids,
                sampling_params=state.sampling_params,
                rid=state.rid,
                stream=False,
            )
        ):
            if output is None:
                output = chunk
            else:
                output.text += chunk.text
                output.token_ids.extend(chunk.token_ids)
                output.finish_reason = chunk.finish_reason
                output.error = chunk.error
        assert output is not None
        return output

    async def _run_loop(self) -> None:
        if getattr(self.scheduler, "uses_execution_plans", False):
            await self._run_execution_plan_loop()
            return
        await self._run_fifo_loop()

    async def _run_fifo_loop(self) -> None:
        while not self._closed:
            batch = self.scheduler.next_batch()
            if batch is None:
                self._new_request_event.clear()
                await self._new_request_event.wait()
                continue

            states = [self._states[rid] for rid in batch.request_ids if rid in self._states]
            if not states:
                continue
            for state in states:
                state.mark_scheduled()
            self.metrics.record_scheduled(len(states))
            try:
                results = await self._execute(self.executor.execute_batch, states)
            except Exception as exc:  # noqa: BLE001
                logger.exception("online generation batch failed")
                for state in states:
                    state.mark_execution_done()
                    await state.queue.put(
                        self.output_processor.make_error_output(state, str(exc))
                    )
                    self.metrics.record_finished(state, error=True)
                    self.scheduler.finish(state.rid)
                    self._states.pop(state.rid, None)
                continue

            for state in states:
                state.mark_execution_done()
            states_by_id = {state.rid: state for state in states}
            for result in results:
                state = states_by_id.get(result.rid)
                if state is None:
                    continue
                if self._states.get(state.rid) is not state or state.finished:
                    continue
                output = self.output_processor.make_output(
                    state,
                    result.token_ids,
                    result.text,
                    result.finish_reason,
                )
                await state.queue.put(output)
                self.metrics.record_finished(state)
                self.scheduler.finish(state.rid)
                self._states.pop(state.rid, None)

    async def _run_execution_plan_loop(self) -> None:
        while not self._closed:
            if not self._states:
                self._new_request_event.clear()
                await self._new_request_event.wait()
                continue

            plan = self.scheduler.next_plan()
            cache_ops = list(plan.cache)
            if cache_ops:
                await self._fail_all_active(
                    "paged scheduler emitted cache operations, which are not "
                    "supported by the Python execution plane yet"
                )
                continue

            forward_ops = list(plan.forward)
            if not forward_ops:
                self._new_request_event.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._new_request_event.wait(), timeout=0.001)
                continue

            for op in forward_ops:
                await self._execute_forward_op(op)

    async def _execute_forward_op(self, op) -> None:
        states_by_id = {
            rid: self._states[rid]
            for rid in list(op.request_ids)
            if rid in self._states
        }
        if not states_by_id:
            return

        first_scheduled = [
            state for state in states_by_id.values() if state.scheduled_time is None
        ]
        for state in states_by_id.values():
            state.mark_scheduled()
        if first_scheduled:
            self.metrics.record_scheduled(len(first_scheduled))

        try:
            results = await self._execute(
                self.executor.execute_forward_plan, op, states_by_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("online paged forward op failed")
            await self._fail_states(states_by_id.values(), str(exc))
            return

        token_results: dict[str, list[int]] = {}
        reserve_tokens: dict[str, int] = {}
        finished_ids: list[str] = []

        for result in results:
            state = self._states.get(result.rid)
            if state is None or state.finished:
                continue

            if result.finished:
                state.mark_execution_done()
            output = self.output_processor.make_output(
                state,
                result.token_ids,
                result.text,
                result.finish_reason if result.finished else None,
            )
            if result.decode_block_completed:
                state.mark_decode_block_done()
            if result.token_ids:
                token_results[result.rid] = result.token_ids

            if result.finished:
                finished_ids.append(result.rid)
                await state.queue.put(output)
                self.metrics.record_finished(state)
                self._states.pop(state.rid, None)
            elif result.reserve_tokens > 0:
                reserve_tokens[result.rid] = result.reserve_tokens
                if output.text or output.token_ids:
                    await state.queue.put(output)

        self.scheduler.advance_forward(
            token_results=token_results,
            reserve_tokens=reserve_tokens,
            finished_ids=finished_ids,
        )
        await self._release_executor_requests(finished_ids)

    async def _fail_states(self, states, error: str) -> None:
        failed_ids = []
        for state in list(states):
            if state.finished:
                continue
            state.mark_execution_done()
            await state.queue.put(self.output_processor.make_error_output(state, error))
            self.metrics.record_finished(state, error=True)
            self.scheduler.abort(state.rid)
            self._states.pop(state.rid, None)
            failed_ids.append(state.rid)
        await self._release_executor_requests(failed_ids)

    async def _release_executor_requests(self, request_ids) -> None:
        release = getattr(self.executor, "release_requests", None)
        if release is not None and request_ids:
            await release(request_ids)

    async def _fail_all_active(self, error: str) -> None:
        await self._fail_states(list(self._states.values()), error)
