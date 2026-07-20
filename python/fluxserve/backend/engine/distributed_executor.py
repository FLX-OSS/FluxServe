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

import logging
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

from fluxserve.backend.distributed.launch import DistributedContext
from fluxserve.backend.engine.executor import ExecutionResult, GenerationExecutor
from fluxserve.backend.engine.request import RequestState

logger = logging.getLogger(__name__)

_CMD_GENERATE = "generate"
_CMD_FORWARD_PLAN = "forward_plan"
_CMD_RELEASE_REQUESTS = "release_requests"
_CMD_SHUTDOWN = "shutdown"


@dataclass
class ForwardPlanPayload:
    request_ids: list[str]
    input_ids: list[int]
    input_lengths: list[int]
    extend_prefix_lens: list[int]
    occupied_pages: list[list[int]]
    num_extends_value: int

    def num_extends(self) -> int:
        return self.num_extends_value


class DistributedGenerationExecutor:
    offload_execution = True

    def __init__(self, base_executor: GenerationExecutor, context: DistributedContext):
        self.base_executor = base_executor
        self.context = context

    async def execute_batch(self, requests: list[RequestState]) -> list[ExecutionResult]:
        if not self.context.is_distributed:
            return await self.base_executor.execute_batch(requests)
        if not self.context.is_rank0:
            raise RuntimeError("execute_batch must only be called on rank 0.")
        _broadcast_command(_make_generate_command(requests))
        return await self.base_executor.execute_batch(requests)

    async def execute_forward_plan(self, op, states_by_id):
        if self.context.is_distributed:
            if not self.context.is_rank0:
                raise RuntimeError("execute_forward_plan must only be called on rank 0.")
            _broadcast_command(_make_forward_plan_command(op, states_by_id))
        return await self.base_executor.execute_forward_plan(op, states_by_id)

    async def release_requests(self, request_ids) -> None:
        request_ids = list(request_ids)
        if not request_ids:
            return
        if self.context.is_distributed:
            if not self.context.is_rank0:
                raise RuntimeError("release_requests must only be called on rank 0.")
            _broadcast_command({"kind": _CMD_RELEASE_REQUESTS, "request_ids": request_ids})
        release = getattr(self.base_executor, "release_requests", None)
        if release is not None:
            await release(request_ids)

    async def run_worker_loop(self) -> None:
        if not self.context.is_distributed:
            return
        if self.context.is_rank0:
            raise RuntimeError("run_worker_loop must only be called on nonzero ranks.")

        while True:
            command = _receive_command()
            kind = command.get("kind")
            if kind == _CMD_SHUTDOWN:
                logger.info("distributed worker rank=%s received shutdown", self.context.rank)
                return
            if kind == _CMD_FORWARD_PLAN:
                op = _forward_plan_from_payload(command["plan"])
                states = [_state_from_payload(item) for item in command["requests"]]
                await self.base_executor.execute_forward_plan(
                    op, {state.rid: state for state in states}
                )
                continue
            if kind == _CMD_RELEASE_REQUESTS:
                release = getattr(self.base_executor, "release_requests", None)
                if release is not None:
                    await release(command["request_ids"])
                continue
            if kind != _CMD_GENERATE:
                raise RuntimeError(f"Unknown distributed executor command: {kind!r}")
            requests = [_state_from_payload(item) for item in command["requests"]]
            await self.base_executor.execute_batch(requests)

    def shutdown_workers(self) -> None:
        if self.context.is_distributed and self.context.is_rank0 and dist.is_initialized():
            _broadcast_command({"kind": _CMD_SHUTDOWN})


def _broadcast_command(command: dict[str, Any]) -> None:
    payload = [command]
    dist.broadcast_object_list(payload, src=0)


def _receive_command() -> dict[str, Any]:
    payload: list[Any] = [None]
    dist.broadcast_object_list(payload, src=0)
    command = payload[0]
    if not isinstance(command, dict):
        raise RuntimeError(f"Invalid distributed executor command payload: {command!r}")
    return command


def _make_generate_command(requests: list[RequestState]) -> dict[str, Any]:
    return {
        "kind": _CMD_GENERATE,
        "requests": [_state_to_payload(req) for req in requests],
    }


def _make_forward_plan_command(op, states_by_id) -> dict[str, Any]:
    request_ids = [str(rid) for rid in op.request_ids]
    return {
        "kind": _CMD_FORWARD_PLAN,
        "plan": {
            "request_ids": request_ids,
            "input_ids": [int(x) for x in op.input_ids],
            "input_lengths": [int(x) for x in op.input_lengths],
            "extend_prefix_lens": [int(x) for x in op.extend_prefix_lens],
            "occupied_pages": [[int(x) for x in pages] for pages in op.occupied_pages],
            "num_extends": int(op.num_extends()),
        },
        "requests": [_state_to_payload(states_by_id[rid]) for rid in request_ids],
    }


def _forward_plan_from_payload(payload: dict[str, Any]) -> ForwardPlanPayload:
    request_ids = [str(x) for x in payload["request_ids"]]
    occupied_pages = [[int(x) for x in pages] for pages in payload["occupied_pages"]]
    input_lengths = [int(x) for x in payload["input_lengths"]]
    num_extends = int(payload["num_extends"])
    if len(occupied_pages) != len(request_ids) or len(input_lengths) != len(request_ids):
        raise ValueError("Forward plan row metadata must match request_ids length.")
    if num_extends < 0 or num_extends > len(request_ids):
        raise ValueError("Forward plan num_extends is out of range.")
    return ForwardPlanPayload(
        request_ids=request_ids,
        input_ids=[int(x) for x in payload["input_ids"]],
        input_lengths=input_lengths,
        extend_prefix_lens=[int(x) for x in payload["extend_prefix_lens"]],
        occupied_pages=occupied_pages,
        num_extends_value=num_extends,
    )


def _state_to_payload(req: RequestState) -> dict[str, Any]:
    return {
        "rid": req.rid,
        "input_ids": list(req.input_ids),
        "max_new_tokens": req.max_new_tokens,
        "ignore_eos": req.ignore_eos,
        "prompt_text": req.prompt_text,
        "sampling_params": dict(req.sampling_params),
        "output_ids": list(req.output_ids),
        "decoded_text": req.decoded_text,
        "plan_prefill_done": req.plan_prefill_done,
        "current_decode_block": req.current_decode_block,
    }


def _state_from_payload(payload: dict[str, Any]) -> RequestState:
    state = RequestState(
        rid=str(payload["rid"]),
        input_ids=[int(x) for x in payload["input_ids"]],
        max_new_tokens=int(payload["max_new_tokens"]),
        ignore_eos=bool(payload["ignore_eos"]),
        prompt_text=str(payload.get("prompt_text", "")),
        sampling_params=dict(payload.get("sampling_params", {})),
    )
    state.output_ids = [int(x) for x in payload.get("output_ids", [])]
    state.decoded_text = str(payload.get("decoded_text", ""))
    state.plan_prefill_done = bool(payload.get("plan_prefill_done", False))
    state.current_decode_block = int(payload.get("current_decode_block", 0))
    return state
