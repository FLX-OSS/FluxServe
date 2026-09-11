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


"""Batch-level attention backend selection for diffusion language models.

The selector is intentionally independent of CUDA and of any one attention
implementation.  A scheduler or runner describes the work it intends to put
in one model forward, advertises the backends that are actually available, and
freezes the resulting :class:`AttentionExecutionPlan` on ``ForwardBatch``.

Keeping selection outside an individual transformer layer is important: every
layer must use the same packing and scheduling decision, and POD is useful only
when the scheduler has deliberately constructed a compatible mixed batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Callable, Iterable, Mapping, Protocol, Sequence


class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    FLASHINFER_BATCH_PREFILL = "flashinfer_batch_prefill"
    FLASH_ATTENTION_4 = "flashattention4"
    FLASHINFER_POD = "flashinfer_pod"


class AttentionPhase(str, Enum):
    PREFILL = "prefill"
    DENOISE = "denoise"


class AttentionMaskKind(str, Enum):
    NONE = "none"
    BLOCK_CAUSAL = "block_causal"
    CUSTOM = "custom"


@dataclass(frozen=True)
class AttentionWorkItem:
    """Attention-relevant features of one scheduled request/task."""

    request_id: str
    phase: AttentionPhase
    q_len: int
    kv_len: int
    q_offset: int = 0

    def __post_init__(self) -> None:
        if self.q_len <= 0:
            raise ValueError(f"q_len must be positive, got {self.q_len}")
        if self.kv_len <= 0:
            raise ValueError(f"kv_len must be positive, got {self.kv_len}")
        if self.q_offset < 0:
            raise ValueError(f"q_offset must be non-negative, got {self.q_offset}")
        if self.q_offset + self.q_len > self.kv_len:
            raise ValueError(
                "query range must fit in KV range: "
                f"q_offset={self.q_offset}, q_len={self.q_len}, kv_len={self.kv_len}"
            )


@dataclass(frozen=True)
class AttentionBatchCharacteristics:
    """Stable, CPU-side description of one possible attention launch."""

    work_items: tuple[AttentionWorkItem, ...]
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    block_length: int
    page_size: int | None = None
    mask_kind: AttentionMaskKind = AttentionMaskKind.BLOCK_CAUSAL
    uses_paged_kv: bool = True
    cuda_graph: bool = False
    # This is stronger than merely containing both phases.  It says the runner
    # will execute them in the same model forward and has one shared output map.
    scheduler_fused: bool = False

    def __post_init__(self) -> None:
        if not self.work_items:
            raise ValueError("an attention batch must contain at least one work item")
        if self.num_q_heads <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("attention head counts and head_dim must be positive")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")
        if self.page_size is not None and self.page_size <= 0:
            raise ValueError("page_size must be positive when present")
        if self.scheduler_fused and not self.is_mixed:
            raise ValueError("scheduler_fused is meaningful only for a mixed batch")

    @property
    def phases(self) -> frozenset[AttentionPhase]:
        return frozenset(item.phase for item in self.work_items)

    @property
    def is_mixed(self) -> bool:
        return len(self.phases) > 1

    @property
    def total_q_tokens(self) -> int:
        return sum(item.q_len for item in self.work_items)

    @property
    def total_kv_tokens(self) -> int:
        return sum(item.kv_len for item in self.work_items)

    @property
    def max_q_len(self) -> int:
        return max(item.q_len for item in self.work_items)

    @property
    def max_kv_len(self) -> int:
        return max(item.kv_len for item in self.work_items)

    @property
    def all_queries_block_aligned(self) -> bool:
        return all(
            item.q_len % self.block_length == 0
            and item.q_offset % self.block_length == 0
            for item in self.work_items
        )


CompatibilityCheck = Callable[[AttentionBatchCharacteristics], str | None]


@dataclass(frozen=True)
class AttentionBackendCapability:
    """Correctness and runtime constraints for one installed backend.

    ``extra_check`` returns ``None`` when compatible, or a user-facing reason
    when incompatible.  It is useful for build-specific constraints such as
    FA4 head dimensions and GPU architecture support.
    """

    backend: AttentionBackend
    available: bool = True
    supported_dtypes: frozenset[str] = field(
        default_factory=lambda: frozenset({"float16", "bfloat16"})
    )
    supported_head_dims: frozenset[int] | None = None
    supports_paged_kv: bool = True
    supports_dense_kv: bool = False
    supports_block_causal: bool = True
    supports_custom_mask: bool = False
    supports_mixed_phases: bool = False
    supports_cuda_graph: bool = False
    requires_mixed_phases: bool = False
    requires_scheduler_fusion: bool = False
    requires_block_aligned_queries: bool = False
    extra_check: CompatibilityCheck | None = field(
        default=None, compare=False, repr=False
    )

    def incompatibility(self, batch: AttentionBatchCharacteristics) -> str | None:
        if not self.available:
            return "backend is not installed or initialized"
        if batch.dtype not in self.supported_dtypes:
            return f"dtype {batch.dtype!r} is unsupported"
        if (
            self.supported_head_dims is not None
            and batch.head_dim not in self.supported_head_dims
        ):
            return f"head_dim={batch.head_dim} is unsupported"
        if batch.uses_paged_kv and not self.supports_paged_kv:
            return "paged KV cache is unsupported"
        if not batch.uses_paged_kv and not self.supports_dense_kv:
            return "dense KV cache is unsupported"
        if batch.mask_kind == AttentionMaskKind.BLOCK_CAUSAL:
            if not self.supports_block_causal:
                return "block-causal attention is unsupported"
        elif batch.mask_kind == AttentionMaskKind.CUSTOM:
            if not self.supports_custom_mask:
                return "custom attention masks are unsupported"
        if batch.is_mixed and not self.supports_mixed_phases:
            return "mixed prefill/denoise batches are unsupported"
        if self.requires_mixed_phases and not batch.is_mixed:
            return "backend requires a mixed prefill/denoise batch"
        if self.requires_scheduler_fusion and not batch.scheduler_fused:
            return "backend requires a scheduler-fused model forward"
        if batch.cuda_graph and not self.supports_cuda_graph:
            return "CUDA graph execution is unsupported"
        if self.requires_block_aligned_queries and not batch.all_queries_block_aligned:
            return "queries and query offsets must be block aligned"
        if self.extra_check is not None:
            return self.extra_check(batch)
        return None


class AttentionCostModel(Protocol):
    """Returns predicted attention latency in milliseconds, or ``None``."""

    def estimate_ms(
        self,
        backend: AttentionBackend,
        batch: AttentionBatchCharacteristics,
    ) -> float | None: ...


@dataclass(frozen=True)
class AttentionExecutionPlan:
    backend: AttentionBackend
    reason: str
    characteristics: AttentionBatchCharacteristics
    predicted_latency_ms: float | None = None
    rejected_backends: Mapping[AttentionBackend, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AttentionScheduleCandidate:
    """One legal scheduler choice, consisting of one or more model forwards."""

    name: str
    batches: tuple[AttentionBatchCharacteristics, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("schedule candidate name must not be empty")
        if not self.batches:
            raise ValueError("schedule candidate must contain at least one batch")


@dataclass(frozen=True)
class AttentionSchedulePlan:
    candidate_name: str
    forwards: tuple[AttentionExecutionPlan, ...]
    reason: str
    predicted_latency_ms: float | None = None


class AttentionBackendSelector:
    """Select a compatible backend once per scheduled model forward."""

    def __init__(
        self,
        capabilities: Iterable[AttentionBackendCapability],
        *,
        fallback_order: Sequence[AttentionBackend] = (
            AttentionBackend.FLASHINFER_BATCH_PREFILL,
            AttentionBackend.FLASH_ATTENTION_4,
            AttentionBackend.SDPA,
            AttentionBackend.FLASHINFER_POD,
        ),
        cost_model: AttentionCostModel | None = None,
    ) -> None:
        self.capabilities = {item.backend: item for item in capabilities}
        self.fallback_order = tuple(fallback_order)
        self.cost_model = cost_model

    def select(
        self,
        batch: AttentionBatchCharacteristics,
        *,
        requested_backend: AttentionBackend | str | None = None,
    ) -> AttentionExecutionPlan:
        requested = (
            AttentionBackend(requested_backend)
            if requested_backend is not None
            else None
        )
        rejected: dict[AttentionBackend, str] = {}
        compatible: list[AttentionBackend] = []
        for backend, capability in self.capabilities.items():
            reason = capability.incompatibility(batch)
            if reason is None:
                compatible.append(backend)
            else:
                rejected[backend] = reason

        if requested is not None:
            capability = self.capabilities.get(requested)
            if capability is None:
                raise ValueError(f"requested backend {requested.value!r} is not registered")
            if requested in rejected:
                raise ValueError(
                    f"requested backend {requested.value!r} is incompatible: "
                    f"{rejected[requested]}"
                )
            return AttentionExecutionPlan(
                backend=requested,
                reason="explicitly requested",
                characteristics=batch,
                rejected_backends=rejected,
            )

        if not compatible:
            details = ", ".join(
                f"{backend.value}: {reason}"
                for backend, reason in rejected.items()
            )
            raise RuntimeError(f"no compatible attention backend ({details})")

        estimates: dict[AttentionBackend, float] = {}
        if self.cost_model is not None:
            for backend in compatible:
                estimate = self.cost_model.estimate_ms(backend, batch)
                if estimate is not None:
                    estimate = float(estimate)
                    if estimate < 0 or not isfinite(estimate):
                        raise ValueError(
                            f"invalid cost estimate for {backend.value}: {estimate}"
                        )
                    estimates[backend] = estimate

        # A partial profile table is useful: profiled candidates compete by
        # measured cost, while an unknown backend does not win accidentally.
        if estimates:
            selected = min(estimates, key=estimates.__getitem__)
            return AttentionExecutionPlan(
                backend=selected,
                reason="lowest profile-guided predicted latency",
                characteristics=batch,
                predicted_latency_ms=estimates[selected],
                rejected_backends=rejected,
            )

        for backend in self.fallback_order:
            if backend in compatible:
                return AttentionExecutionPlan(
                    backend=backend,
                    reason="first compatible backend in fallback order",
                    characteristics=batch,
                    rejected_backends=rejected,
                )

        selected = compatible[0]
        return AttentionExecutionPlan(
            backend=selected,
            reason="compatible backend",
            characteristics=batch,
            rejected_backends=rejected,
        )

    def select_schedule(
        self,
        candidates: Iterable[AttentionScheduleCandidate],
    ) -> AttentionSchedulePlan:
        """Compare scheduler-produced fused and split execution candidates.

        A candidate is profile-comparable only when every one of its forwards
        has a latency estimate.  If no complete estimate exists, the first
        legal candidate is returned; callers should put their conservative
        current schedule first.
        """

        legal: list[
            tuple[AttentionScheduleCandidate, tuple[AttentionExecutionPlan, ...]]
        ] = []
        failures: list[str] = []
        for candidate in candidates:
            try:
                forwards = tuple(self.select(batch) for batch in candidate.batches)
            except (RuntimeError, ValueError) as exc:
                failures.append(f"{candidate.name}: {exc}")
                continue
            legal.append((candidate, forwards))

        if not legal:
            details = "; ".join(failures)
            raise RuntimeError(f"no legal attention schedule candidate ({details})")

        predicted: list[
            tuple[float, AttentionScheduleCandidate, tuple[AttentionExecutionPlan, ...]]
        ] = []
        for candidate, forwards in legal:
            if all(plan.predicted_latency_ms is not None for plan in forwards):
                total = sum(float(plan.predicted_latency_ms) for plan in forwards)
                predicted.append((total, candidate, forwards))

        if predicted:
            total, candidate, forwards = min(predicted, key=lambda item: item[0])
            return AttentionSchedulePlan(
                candidate_name=candidate.name,
                forwards=forwards,
                reason="lowest profile-guided total attention latency",
                predicted_latency_ms=total,
            )

        candidate, forwards = legal[0]
        return AttentionSchedulePlan(
            candidate_name=candidate.name,
            forwards=forwards,
            reason="first legal scheduler candidate; no complete profile estimate",
        )


class CallableCostModel:
    """Small adapter for profilers/autotuners that expose a Python callback."""

    def __init__(
        self,
        callback: Callable[
            [AttentionBackend, AttentionBatchCharacteristics], float | None
        ],
    ) -> None:
        self.callback = callback

    def estimate_ms(
        self,
        backend: AttentionBackend,
        batch: AttentionBatchCharacteristics,
    ) -> float | None:
        return self.callback(backend, batch)
