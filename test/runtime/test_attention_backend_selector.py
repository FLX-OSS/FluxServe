import pytest

from fluxserve.backend.layers.attention.selector import (
    AttentionBackend,
    AttentionBackendCapability,
    AttentionBackendSelector,
    AttentionBatchCharacteristics,
    AttentionMaskKind,
    AttentionPhase,
    AttentionScheduleCandidate,
    AttentionWorkItem,
    CallableCostModel,
)


def _batch(*phases, head_dim=128, scheduler_fused=False, cuda_graph=False):
    items = tuple(
        AttentionWorkItem(
            request_id=f"request-{index}",
            phase=phase,
            q_len=64,
            kv_len=512 + index * 64,
            q_offset=448 + index * 64,
        )
        for index, phase in enumerate(phases)
    )
    return AttentionBatchCharacteristics(
        work_items=items,
        num_q_heads=16,
        num_kv_heads=4,
        head_dim=head_dim,
        dtype="bfloat16",
        block_length=64,
        page_size=64,
        mask_kind=AttentionMaskKind.BLOCK_CAUSAL,
        scheduler_fused=scheduler_fused,
        cuda_graph=cuda_graph,
    )


def _batch_prefill_capability():
    return AttentionBackendCapability(
        backend=AttentionBackend.FLASHINFER_BATCH_PREFILL,
        supports_mixed_phases=True,
        supports_cuda_graph=True,
        requires_block_aligned_queries=True,
    )


def _pod_capability():
    return AttentionBackendCapability(
        backend=AttentionBackend.FLASHINFER_POD,
        supports_mixed_phases=True,
        requires_mixed_phases=True,
        requires_scheduler_fusion=True,
        requires_block_aligned_queries=True,
    )


def test_pod_is_rejected_when_scheduler_did_not_fuse_the_phases():
    selector = AttentionBackendSelector(
        [_batch_prefill_capability(), _pod_capability()]
    )
    batch = _batch(AttentionPhase.PREFILL, AttentionPhase.DENOISE)

    plan = selector.select(batch)

    assert plan.backend == AttentionBackend.FLASHINFER_BATCH_PREFILL
    assert "scheduler-fused" in plan.rejected_backends[AttentionBackend.FLASHINFER_POD]


def test_profile_model_can_choose_pod_for_a_scheduler_fused_mixed_batch():
    costs = {
        AttentionBackend.FLASHINFER_BATCH_PREFILL: 0.153,
        AttentionBackend.FLASHINFER_POD: 0.118,
    }
    selector = AttentionBackendSelector(
        [_batch_prefill_capability(), _pod_capability()],
        cost_model=CallableCostModel(lambda backend, _batch: costs.get(backend)),
    )
    batch = _batch(
        AttentionPhase.PREFILL,
        AttentionPhase.DENOISE,
        scheduler_fused=True,
    )

    plan = selector.select(batch)

    assert plan.backend == AttentionBackend.FLASHINFER_POD
    assert plan.predicted_latency_ms == pytest.approx(0.118)


def test_fa4_build_specific_head_dim_constraint_is_checked_before_cost():
    fa4 = AttentionBackendCapability(
        backend=AttentionBackend.FLASH_ATTENTION_4,
        supported_head_dims=frozenset({64, 128, 256}),
        supports_mixed_phases=True,
    )
    selector = AttentionBackendSelector(
        [_batch_prefill_capability(), fa4],
        cost_model=CallableCostModel(lambda backend, _batch: 0.01),
    )

    plan = selector.select(_batch(AttentionPhase.DENOISE, head_dim=512))

    assert plan.backend == AttentionBackend.FLASHINFER_BATCH_PREFILL
    assert plan.rejected_backends[AttentionBackend.FLASH_ATTENTION_4] == (
        "head_dim=512 is unsupported"
    )


def test_explicit_incompatible_backend_fails_instead_of_silently_falling_back():
    selector = AttentionBackendSelector(
        [_batch_prefill_capability(), _pod_capability()]
    )

    with pytest.raises(ValueError, match="requires a mixed"):
        selector.select(
            _batch(AttentionPhase.DENOISE),
            requested_backend=AttentionBackend.FLASHINFER_POD,
        )


def test_cuda_graph_filters_out_non_graph_backend():
    fa4 = AttentionBackendCapability(
        backend=AttentionBackend.FLASH_ATTENTION_4,
        supports_mixed_phases=True,
        supports_cuda_graph=False,
    )
    selector = AttentionBackendSelector([_batch_prefill_capability(), fa4])

    plan = selector.select(
        _batch(AttentionPhase.DENOISE, cuda_graph=True)
    )

    assert plan.backend == AttentionBackend.FLASHINFER_BATCH_PREFILL
    assert plan.rejected_backends[AttentionBackend.FLASH_ATTENTION_4] == (
        "CUDA graph execution is unsupported"
    )


def test_scheduler_can_compare_split_batchprefill_with_fused_pod():
    def estimate(backend, batch):
        if batch.is_mixed:
            return {
                AttentionBackend.FLASHINFER_BATCH_PREFILL: 0.153,
                AttentionBackend.FLASHINFER_POD: 0.118,
            }.get(backend)
        if backend == AttentionBackend.FLASHINFER_BATCH_PREFILL:
            return 0.075
        return None

    selector = AttentionBackendSelector(
        [_batch_prefill_capability(), _pod_capability()],
        cost_model=CallableCostModel(estimate),
    )
    split = AttentionScheduleCandidate(
        name="split",
        batches=(
            _batch(AttentionPhase.PREFILL),
            _batch(AttentionPhase.DENOISE),
        ),
    )
    fused = AttentionScheduleCandidate(
        name="fused",
        batches=(
            _batch(
                AttentionPhase.PREFILL,
                AttentionPhase.DENOISE,
                scheduler_fused=True,
            ),
        ),
    )

    plan = selector.select_schedule((split, fused))

    assert plan.candidate_name == "fused"
    assert plan.predicted_latency_ms == pytest.approx(0.118)
    assert plan.forwards[0].backend == AttentionBackend.FLASHINFER_POD
