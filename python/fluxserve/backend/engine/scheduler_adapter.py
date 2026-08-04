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

from dataclasses import dataclass
from typing import Iterable

from .request import RequestState


@dataclass
class ScheduledBatch:
    request_ids: list[str]


class DefaultSchedulerAdapter:
    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self._queue: list[str] = []
        self._active: set[str] = set()

    def submit(self, requests: Iterable[RequestState]) -> None:
        for req in requests:
            if req.rid not in self._active:
                self._active.add(req.rid)
                self._queue.append(req.rid)

    def next_batch(self) -> ScheduledBatch | None:
        if not self._queue:
            return None
        request_ids = self._queue[: self.max_batch_size]
        del self._queue[: self.max_batch_size]
        return ScheduledBatch(request_ids=request_ids)

    def finish(self, request_id: str) -> None:
        self._active.discard(request_id)

    def abort(self, request_id: str) -> None:
        self.finish(request_id)
        self._queue = [rid for rid in self._queue if rid != request_id]


class FluxSchedulerAdapter(DefaultSchedulerAdapter):
    """Thin runtime wrapper around flux_scheduler.

    The current online milestone executes whole requests through the existing
    BlockDiffusionRunner. We still submit to the native scheduler so deployment
    catches packaging/build issues early; batching falls back to FIFO request
    groups until the runner exposes true per-plan execution.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_scheduled_tokens: int,
        page_size: int,
        num_device_pages: int,
    ):
        super().__init__(max_batch_size=max_batch_size)
        try:
            from flux_scheduler import RequestSpec, Scheduler, SchedulerConfig
        except ImportError as exc:
            raise RuntimeError(
                "flux_scheduler is not installed. Install tests/flux-scheduler "
                "or use DefaultSchedulerAdapter for tests."
            ) from exc

        cfg = SchedulerConfig()
        cfg.max_batch_size = max_batch_size
        cfg.max_scheduled_tokens = max_scheduled_tokens
        cfg.page_size = page_size
        cfg.num_device_pages = num_device_pages
        self._request_spec_cls = RequestSpec
        self._scheduler = Scheduler(cfg)

    def submit(self, requests: Iterable[RequestState]) -> None:
        reqs = list(requests)
        specs = []
        for req in reqs:
            spec = self._request_spec_cls()
            spec.request_id = req.rid
            spec.tokens = list(req.input_ids)
            specs.append(spec)
        if specs:
            self._scheduler.submit_requests(specs)
        super().submit(reqs)


class PagedSchedulerAdapter:
    """Plan-oriented wrapper around the native C++ scheduler."""

    uses_execution_plans = True

    def __init__(
        self,
        max_batch_size: int,
        max_scheduled_tokens: int,
        page_size: int,
        num_device_pages: int,
        max_model_len: int,
    ):
        try:
            from flux_scheduler import (
                ExecutionEvent,
                ForwardEvent,
                RequestSpec,
                Scheduler,
                SchedulerConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "flux_scheduler is not installed. Install flux-scheduler before "
                "using scheduler_policy='paged'."
            ) from exc

        cfg = SchedulerConfig()
        cfg.max_batch_size = max_batch_size
        cfg.max_scheduled_tokens = max_scheduled_tokens
        cfg.page_size = page_size
        cfg.decode_input_tokens = page_size
        cfg.num_device_pages = num_device_pages
        cfg.disable_l2_cache = True
        cfg.disable_prefix_cache = True
        self._request_spec_cls = RequestSpec
        self._execution_event_cls = ExecutionEvent
        self._forward_event = ForwardEvent
        self._scheduler = Scheduler(cfg)
        self.page_size = int(page_size)
        self.max_model_len = int(max_model_len)
        self._active: set[str] = set()

    def submit(self, requests: Iterable[RequestState]) -> None:
        specs = []
        for req in requests:
            if req.rid in self._active:
                continue
            self._active.add(req.rid)
            spec = self._request_spec_cls()
            spec.request_id = req.rid
            spec.tokens = list(req.input_ids)
            spec.prefill_length = req.aligned_prefill_length(self.page_size)
            if spec.prefill_length > self.max_model_len:
                raise ValueError(
                    f"aligned prefill length {spec.prefill_length} exceeds "
                    f"max_model_len={self.max_model_len}"
                )
            specs.append(spec)
        if specs:
            self._scheduler.submit_requests(specs)

    def next_plan(self):
        return self._scheduler.next_execution_plan()

    def advance_forward(
        self,
        *,
        token_results: dict[str, list[int]] | None = None,
        finished_ids: Iterable[str] = (),
        reserve_tokens: dict[str, int] | None = None,
    ) -> None:
        event = self._execution_event_cls()
        has_events = False
        for rid, tokens in (token_results or {}).items():
            if tokens:
                ev = self._forward_event.ExtendResult()
                ev.request_id = rid
                ev.tokens = list(tokens)
                event.add_event(ev)
                has_events = True
        for rid, reserve in (reserve_tokens or {}).items():
            ev = self._forward_event.UpdateReserveNumTokens()
            ev.request_id = rid
            ev.reserve_num_tokens_in_next_schedule_event = int(reserve)
            event.add_event(ev)
            has_events = True
        for rid in finished_ids:
            ev = self._forward_event.Finish()
            ev.request_id = rid
            event.add_event(ev)
            self._active.discard(rid)
            has_events = True
        if has_events:
            self._scheduler.advance(event)

    def finish(self, request_id: str) -> None:
        self.advance_forward(finished_ids=[request_id])

    def abort(self, request_id: str) -> None:
        if request_id not in self._active:
            return
        event = self._execution_event_cls()
        ev = self._forward_event.Abort()
        ev.request_id = request_id
        event.add_event(ev)
        self._scheduler.advance(event)
        self._active.discard(request_id)
