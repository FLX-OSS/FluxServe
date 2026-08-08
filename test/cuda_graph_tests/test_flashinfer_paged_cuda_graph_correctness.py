"""Single-GPU correctness checks for FlashInfer paged attention CUDA graphs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch


NUM_Q_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 64
WORKSPACE_BYTES = 128 * 1024 * 1024
RTOL = 1e-2
ATOL = 1e-2


@dataclass(frozen=True)
class Case:
    name: str
    q_len: int
    kv_len: int
    q_offset: int


CASES = (
    Case("prefill", q_len=128, kv_len=128, q_offset=0),
)


def _require_cuda_and_flashinfer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    try:
        import flashinfer
    except ImportError:
        pytest.skip("flashinfer is not installed")
    if not hasattr(flashinfer, "BatchPrefillWithPagedKVCacheWrapper"):
        pytest.skip("FlashInfer paged prefill wrapper is unavailable")
    return flashinfer


def _block_causal_mask(case: Case, device: torch.device) -> torch.Tensor:
    q_pos = torch.arange(case.q_len, device=device) + case.q_offset
    k_pos = torch.arange(case.kv_len, device=device)
    return (q_pos[:, None] // PAGE_SIZE) >= (k_pos[None, :] // PAGE_SIZE)


def _metadata(case: Case, page_ids: torch.Tensor, device: torch.device):
    num_pages = (case.kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    return (
        torch.tensor([0, case.q_len], dtype=torch.int32, device=device),
        torch.tensor([0, num_pages], dtype=torch.int32, device=device),
        page_ids[:num_pages].to(dtype=torch.int32),
        torch.tensor(
            [(case.kv_len - 1) % PAGE_SIZE + 1],
            dtype=torch.int32,
            device=device,
        ),
    )


def _make_wrapper(
    flashinfer,
    device: torch.device,
):
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    # FlashInfer 0.6.13's use_cuda_graph=True custom-mask path fails on GH200
    # before capture. A normally planned wrapper still retains fixed metadata
    # buffers and lets us validate whether its run kernels are capturable.
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )


def _plan(
    wrapper,
    case: Case,
    metadata,
    mask: torch.Tensor,
    *,
    num_q_heads: int = NUM_Q_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
) -> None:
    qo_indptr, kv_indptr, kv_indices, last_page_len = metadata
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=HEAD_DIM,
        page_size=PAGE_SIZE,
        custom_mask=mask.flatten(),
        causal=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        disable_split_kv=True,
    )


def _eager_reference(flashinfer, case, q, kv_cache, metadata, mask):
    wrapper = _make_wrapper(
        flashinfer,
        q.device,
    )
    _plan(wrapper, case, metadata, mask)
    return wrapper.run(q, kv_cache, enable_pdl=False)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_llada2_mini_paged_attention_cuda_graph_correctness(case: Case) -> None:
    flashinfer = _require_cuda_and_flashinfer()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)

    num_case_pages = (case.kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_cache_pages = num_case_pages + 1
    page_ids = torch.arange(num_case_pages, dtype=torch.int32, device=device)
    metadata = _metadata(case, page_ids, device)
    mask = _block_causal_mask(case, device)
    q = torch.randn(
        case.q_len,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.randn(
        num_cache_pages,
        2,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )

    eager = _eager_reference(flashinfer, case, q, kv_cache, metadata, mask)
    graph_wrapper = _make_wrapper(
        flashinfer,
        device,
    )
    _plan(graph_wrapper, case, metadata, mask)
    direct = graph_wrapper.run(q, kv_cache, enable_pdl=False)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(direct, eager, rtol=RTOL, atol=ATOL)

    # Warm up on a side stream before capture to avoid lazy initialization in the graph.
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            graph_wrapper.run(q, kv_cache, enable_pdl=False)
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = graph_wrapper.run(q, kv_cache, enable_pdl=False)
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(graph_output, eager, rtol=RTOL, atol=ATOL)
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(graph_output, eager, rtol=RTOL, atol=ATOL)

    q.add_(torch.randn_like(q) * 0.125)
    mutated_q_eager = _eager_reference(
        flashinfer, case, q, kv_cache, metadata, mask
    )
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(graph_output, mutated_q_eager, rtol=RTOL, atol=ATOL)
    assert not torch.allclose(graph_output, eager, rtol=RTOL, atol=ATOL)

    # Swap the first and spare physical pages without changing metadata shapes.
    mutated_page_ids = page_ids.clone()
    mutated_page_ids[0] = num_cache_pages - 1
    mutated_metadata = _metadata(case, mutated_page_ids, device)
    metadata[2].copy_(mutated_metadata[2])
    mutated_metadata_eager = _eager_reference(
        flashinfer, case, q, kv_cache, mutated_metadata, mask
    )
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        graph_output, mutated_metadata_eager, rtol=RTOL, atol=ATOL
    )
    assert not torch.allclose(
        graph_output, mutated_q_eager, rtol=RTOL, atol=ATOL
    )


def test_fluxserve_runner_selects_prefill_buckets() -> None:
    _require_cuda_and_flashinfer()
    from fluxserve.backend.execution.flashinfer_cuda_graph_runner import (
        FlashInferCudaGraphRunner,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    runner = FlashInferCudaGraphRunner(device, capture_sizes=(64, 128, 256))
    assert runner.bucket(96) == 128
    assert runner.bucket(192) == 256
    assert runner.can_run(
        q_len=96,
        q_offset=0,
        page_size=PAGE_SIZE,
    )


def test_fluxserve_runner_decode_eligibility() -> None:
    _require_cuda_and_flashinfer()
    from fluxserve.backend.execution.flashinfer_cuda_graph_runner import (
        FlashInferCudaGraphRunner,
    )

    runner = FlashInferCudaGraphRunner("cuda:0")
    assert not runner.can_run(
        q_len=64,
        q_offset=128,
        page_size=PAGE_SIZE,
    )
    assert runner.can_run_decode(batch_size=1, q_len=64, kv_len=64)
    assert runner.can_run_decode(batch_size=2, q_len=64, kv_len=256)
    assert runner.can_run_decode(batch_size=4, q_len=64, kv_len=256)
    assert runner.can_run_decode(batch_size=8, q_len=64, kv_len=256)
    assert runner.can_run_decode(batch_size=1, q_len=64, kv_len=256)
    assert not runner.can_run_decode(batch_size=3, q_len=64, kv_len=256)
    assert not runner.can_run_decode(batch_size=1, q_len=32, kv_len=256)
    assert not runner.can_run_decode(batch_size=1, q_len=64, kv_len=224)
    assert not runner.can_run_decode(batch_size=1, q_len=64, kv_len=32)

    larger_runner = FlashInferCudaGraphRunner(
        "cuda:0", decode_capture_batch_sizes=(1, 2, 4, 8, 16)
    )
    assert larger_runner.can_run_decode(batch_size=16, q_len=64, kv_len=256)
    assert not larger_runner.can_run_decode(batch_size=32, q_len=64, kv_len=256)


def test_fluxserve_runner_decomposes_decode_batches_without_padding() -> None:
    _require_cuda_and_flashinfer()
    from fluxserve.backend.execution.flashinfer_cuda_graph_runner import (
        FlashInferCudaGraphRunner,
    )

    expected = {
        1: (1,),
        2: (2,),
        3: (2, 1),
        4: (4,),
        5: (4, 1),
        6: (4, 2),
        7: (4, 2, 1),
        8: (8,),
        11: (8, 2, 1),
    }
    for batch_size, parts in expected.items():
        assert FlashInferCudaGraphRunner.decompose_batch_size(batch_size) == parts
        assert sum(parts) == batch_size

    assert FlashInferCudaGraphRunner.decompose_batch_size(
        12, (1, 2, 4, 8)
    ) == (8, 4)
    assert FlashInferCudaGraphRunner.decompose_batch_size(
        16, (1, 2, 4, 8, 16)
    ) == (16,)
    assert FlashInferCudaGraphRunner.decompose_batch_size(
        31, (1, 2, 4, 8, 16)
    ) == (16, 8, 4, 2, 1)
    assert FlashInferCudaGraphRunner.decompose_batch_size(
        48, (1, 2, 4, 8, 16, 32)
    ) == (32, 16)


def test_fluxserve_runner_selects_reachable_online_decode_graphs() -> None:
    _require_cuda_and_flashinfer()
    from fluxserve.backend.execution.flashinfer_cuda_graph_runner import (
        FlashInferCudaGraphRunner,
    )

    assert FlashInferCudaGraphRunner.capture_batch_sizes(0) == ()
    assert FlashInferCudaGraphRunner.capture_batch_sizes(1) == (1,)
    assert FlashInferCudaGraphRunner.capture_batch_sizes(3) == (1, 2)
    assert FlashInferCudaGraphRunner.capture_batch_sizes(4) == (1, 2, 4)
    assert FlashInferCudaGraphRunner.capture_batch_sizes(6) == (1, 2, 4)
    assert FlashInferCudaGraphRunner.capture_batch_sizes(8) == (1, 2, 4, 8)
    assert FlashInferCudaGraphRunner.capture_batch_sizes(12) == (1, 2, 4, 8)
    assert FlashInferCudaGraphRunner.capture_batch_sizes(16) == (1, 2, 4, 8, 16)
    assert FlashInferCudaGraphRunner.capture_batch_sizes(32) == (
        1, 2, 4, 8, 16, 32
    )
