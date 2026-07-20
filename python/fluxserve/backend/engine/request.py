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
import time
from dataclasses import dataclass, field
from typing import Any

from .io_struct import GenerateReqOutput


@dataclass
class RequestState:
    rid: str
    input_ids: list[int]
    max_new_tokens: int
    ignore_eos: bool = False
    prompt_text: str = ""
    sampling_params: dict[str, Any] = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)
    queued_time: float | None = None
    scheduled_time: float | None = None
    execution_start_time: float | None = None
    execution_end_time: float | None = None
    completed_time: float | None = None
    output_ids: list[int] = field(default_factory=list)
    decoded_text: str = ""
    finished_reason: str | None = None
    plan_prefill_done: bool = False
    current_decode_block: int = 0
    queue: asyncio.Queue[GenerateReqOutput] = field(default_factory=asyncio.Queue)

    @property
    def finished(self) -> bool:
        return self.finished_reason is not None

    def append_output(self, token_ids: list[int], text: str, finish_reason: str | None) -> GenerateReqOutput:
        self.output_ids.extend(token_ids)
        self.decoded_text += text
        if finish_reason is not None:
            self.finished_reason = finish_reason
            self.completed_time = time.time()
        return GenerateReqOutput(
            rid=self.rid,
            text=text,
            token_ids=token_ids,
            finish_reason=self.finished_reason,
            meta=self.output_metadata() if self.finished else {},
        )

    @property
    def prompt_token_count(self) -> int:
        return len(self.input_ids)

    @property
    def completion_token_count(self) -> int:
        return len(self.output_ids)

    @property
    def queue_latency_s(self) -> float | None:
        if self.queued_time is None or self.scheduled_time is None:
            return None
        return self.scheduled_time - self.queued_time

    @property
    def execution_latency_s(self) -> float | None:
        if self.execution_start_time is None or self.execution_end_time is None:
            return None
        return self.execution_end_time - self.execution_start_time

    @property
    def e2e_latency_s(self) -> float | None:
        if self.completed_time is None:
            return None
        return self.completed_time - self.created_time

    def mark_queued(self) -> None:
        self.queued_time = time.time()

    def mark_scheduled(self) -> None:
        now = time.time()
        if self.scheduled_time is None:
            self.scheduled_time = now
        if self.execution_start_time is None:
            self.execution_start_time = now

    def mark_decode_block_done(self) -> None:
        self.current_decode_block += 1

    def mark_execution_done(self) -> None:
        self.execution_end_time = time.time()

    def output_metadata(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_token_count,
            "completion_tokens": self.completion_token_count,
            "queue_latency_s": self.queue_latency_s,
            "execution_latency_s": self.execution_latency_s,
            "e2e_latency_s": self.e2e_latency_s,
        }
