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

from dataclasses import dataclass, field


@dataclass
class EngineMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    aborted_requests: int = 0
    waiting_requests: int = 0
    running_requests: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    queue_latencies: list[float] = field(default_factory=list)
    execution_latencies: list[float] = field(default_factory=list)
    e2e_latencies: list[float] = field(default_factory=list)

    def record_submitted(self, prompt_tokens: int) -> None:
        self.total_requests += 1
        self.waiting_requests += 1
        self.prompt_tokens += prompt_tokens

    def record_scheduled(self, count: int) -> None:
        self.waiting_requests = max(0, self.waiting_requests - count)
        self.running_requests += count

    def record_finished(self, state, *, error: bool = False, aborted: bool = False) -> None:
        self.running_requests = max(0, self.running_requests - 1)
        self.generated_tokens += state.completion_token_count
        if aborted:
            self.aborted_requests += 1
        elif error:
            self.failed_requests += 1
        else:
            self.successful_requests += 1
        if state.queue_latency_s is not None:
            self.queue_latencies.append(state.queue_latency_s)
        if state.execution_latency_s is not None:
            self.execution_latencies.append(state.execution_latency_s)
        if state.e2e_latency_s is not None:
            self.e2e_latencies.append(state.e2e_latency_s)

    def record_aborted(self, state) -> None:
        if state.scheduled_time is None:
            self.waiting_requests = max(0, self.waiting_requests - 1)
        else:
            self.running_requests = max(0, self.running_requests - 1)
        self.aborted_requests += 1
        self.generated_tokens += state.completion_token_count
        if state.queue_latency_s is not None:
            self.queue_latencies.append(state.queue_latency_s)
        if state.execution_latency_s is not None:
            self.execution_latencies.append(state.execution_latency_s)
        if state.e2e_latency_s is not None:
            self.e2e_latencies.append(state.e2e_latency_s)

    def record_failed_before_submit(self) -> None:
        self.total_requests += 1
        self.failed_requests += 1

    def snapshot(self) -> dict[str, int | float]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "aborted_requests": self.aborted_requests,
            "waiting_requests": self.waiting_requests,
            "running_requests": self.running_requests,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "queue_latency_avg_s": _avg(self.queue_latencies),
            "execution_latency_avg_s": _avg(self.execution_latencies),
            "e2e_latency_avg_s": _avg(self.e2e_latencies),
        }


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
