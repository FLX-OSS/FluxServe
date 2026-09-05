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
from typing import Any, Iterable

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
        # A request may be finished before its queued batch is consumed.
        # Remove the stale entry so a later resubmission starts a fresh FIFO
        # position instead of scheduling the completed request again.
        self._queue = [rid for rid in self._queue if rid != request_id]

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


@dataclass
class SchedulerRank:
    request_id: str
    difficulty_score: float
    convergence_score: float
    evidence: float
    observed_forwards: int
    state: str


@dataclass
class _Trajectory:
    observed_forwards: int = 0
    previous_difficulty: float = 0.5
    difficulty: float = 0.5
    progress_ewma: float = 0.5
    fallback_streak: float = 0.0
    single_token_rate: float = 0.0
    previous_median: float | None = None
    previous_top1: Any = None
    block: dict[str, float] = field(default_factory=dict)


def _percentiles(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.5] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        percentile = ((i + j - 1) / 2) / (len(values) - 1)
        for index, _ in ordered[i:j]:
            ranks[index] = percentile
        i = j
    return ranks


class DynamicSchedulingPolicy:
    """Interpretable confidence-trajectory ranker."""

    def __init__(self, *, convergence_first: bool = True):
        self.convergence_first = convergence_first
        self._states: dict[str, _Trajectory] = {}

    def remove(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def observe_block(self, request_id: str, metrics: dict[str, Any] | None) -> None:
        state = self._states.setdefault(request_id, _Trajectory())
        state.observed_forwards += 1
        metrics = metrics or {}

        def number(*names: str, default: float = 0.0) -> float:
            for name in names:
                value = metrics.get(name)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
            return default

        masks = number("remaining_masks", "masks_remaining", default=0.0)
        block_length = max(number("block_length", default=1.0), 1.0)
        transferred = number("transferred_tokens", "tokens_transferred", default=0.0)
        remaining = number("remaining_mask_fraction", default=masks / block_length)
        progress = number("progress", "transfer_fraction", default=transferred / max(masks + transferred, 1.0))
        deficit = number("confidence_deficit", default=0.0)
        margin = number("top2_margin", "margin", default=0.0)
        fallback = bool(metrics.get("is_fallback_forward", metrics.get("fallback", False)))
        state.fallback_streak = state.fallback_streak + 1 if fallback else 0.0
        state.progress_ewma = 0.6 * progress + 0.4 * state.progress_ewma
        state.single_token_rate = 0.8 * state.single_token_rate + 0.2 * float(transferred == 1)
        state.block = {
            "remaining": remaining,
            "deficit": deficit,
            "margin": margin,
            "readiness": number("readiness_fraction", "readiness", default=0.0),
            "flip_rate": number("flip_rate", default=0.0),
        }

    def rank(self, request_ids: Iterable[str]) -> list[SchedulerRank]:
        request_ids = list(request_ids)
        states = [self._states.setdefault(rid, _Trajectory()) for rid in request_ids]
        fields = {
            "remaining": [s.block.get("remaining", 0.0) for s in states],
            "deficit": [s.block.get("deficit", 0.0) for s in states],
            "margin": [s.block.get("margin", 0.0) for s in states],
            "progress": [s.progress_ewma for s in states],
            "fallback": [s.fallback_streak for s in states],
            "single": [s.single_token_rate for s in states],
            "readiness": [s.block.get("readiness", 0.0) for s in states],
        }
        normalized = {name: _percentiles(values) for name, values in fields.items()}
        result = []
        for i, (rid, state) in enumerate(zip(request_ids, states, strict=True)):
            if state.observed_forwards == 0 and not state.block:
                result.append(SchedulerRank(rid, 0.5, 0.5, 0.0, 0, "converging"))
                continue
            flip = state.block.get("flip_rate", 0.0)
            stable = (1.0 - flip) * normalized["deficit"][i] * (1.0 - normalized["margin"][i])
            raw = (
                0.22 * normalized["remaining"][i]
                + 0.18 * normalized["deficit"][i]
                + 0.18 * (1.0 - normalized["margin"][i])
                + 0.16 * (1.0 - normalized["progress"][i])
                + 0.10 * normalized["fallback"][i]
                + 0.07 * normalized["single"][i]
                + 0.05 * (1.0 - normalized["readiness"][i])
                + 0.04 * stable
            )
            state.difficulty = 0.7 * state.previous_difficulty + 0.3 * raw
            state.previous_difficulty = state.difficulty
            evidence = min(1.0, state.observed_forwards / 3.0)
            result.append(SchedulerRank(rid, state.difficulty, 1.0 - state.difficulty, evidence, state.observed_forwards, self._classify(state)))
        result.sort(key=lambda r: (-(r.convergence_score if self.convergence_first else r.difficulty_score), -r.evidence, r.request_id))
        return result

    @staticmethod
    def _classify(state: _Trajectory) -> str:
        block = state.block
        if state.fallback_streak >= 3 and state.single_token_rate >= 0.4:
            return "stalled"
        if block.get("flip_rate", 0.0) < 0.25 and block.get("margin", 0.0) < 0.1 and block.get("deficit", 0.0) > 0.1:
            return "stable ambiguity"
        if block.get("flip_rate", 0.0) >= 0.25 and block.get("margin", 0.0) < 0.1:
            return "unstable ambiguity"
        if block.get("readiness", 0.0) > 0.5 and state.progress_ewma > 0.5:
            return "converging"
        if block.get("readiness", 0.0) > 0.25:
            return "near-ready"
        return "converging"


class DynamicSchedulerAdapter(PagedSchedulerAdapter):
    """Python admission policy layered over the native paged scheduler."""

    uses_execution_plans = True

    def __init__(self, native_scheduler, max_batch_size: int):
        self.native = native_scheduler
        self.max_batch_size = int(max_batch_size)
        self.policy = DynamicSchedulingPolicy()
        self._pending: dict[str, Any] = {}
        self._active: set[str] = set()
        self._cohort: set[str] = set()
        self._cohort_completed: set[str] = set()

    def submit(self, requests: Iterable[Any]) -> None:
        for request in requests:
            if request.rid not in self._active:
                self._pending[request.rid] = request
        self._admit_if_ready()

    def _admit_if_ready(self) -> None:
        if self._cohort or not self._pending:
            return
        ranked = self.policy.rank(self._pending)
        selected = [self._pending[item.request_id] for item in ranked[: self.max_batch_size]]
        self.native.submit(selected)
        for request in selected:
            self._pending.pop(request.rid, None)
            self._active.add(request.rid)
        self._cohort = {request.rid for request in selected}

    def next_plan(self):
        return self.native.next_plan()

    def observe_results(self, results: Iterable[Any]) -> None:
        completed = set()
        finished = set()
        for result in results:
            if getattr(result, "decode_block_completed", False):
                self.policy.observe_block(result.rid, getattr(result, "trajectory_metrics", None))
                completed.add(result.rid)
            if getattr(result, "finished", False):
                finished.add(result.rid)
                self._active.discard(result.rid)
                self.policy.remove(result.rid)
                self._cohort_completed.add(result.rid)
        self._cohort_completed.update(completed)
        if self._cohort and self._cohort_completed >= self._cohort:
            self._cohort.clear()
            self._cohort_completed.clear()
            self._active.difference_update(finished)
            self._admit_if_ready()

    def advance_forward(self, **kwargs) -> None:
        self.native.advance_forward(**kwargs)

    def finish(self, request_id: str) -> None:
        self._pending.pop(request_id, None)
        self._active.discard(request_id)
        self._cohort.discard(request_id)
        self._cohort_completed.discard(request_id)
        self.policy.remove(request_id)
        self.native.finish(request_id)
        self._admit_if_ready()

    def abort(self, request_id: str) -> None:
        was_active = request_id in self._active
        self._pending.pop(request_id, None)
        self._active.discard(request_id)
        self._cohort.discard(request_id)
        self._cohort_completed.discard(request_id)
        self.policy.remove(request_id)
        if was_active:
            self.native.abort(request_id)
        self._admit_if_ready()
