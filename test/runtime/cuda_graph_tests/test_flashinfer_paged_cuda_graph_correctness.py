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


def test_diffusion_gemma_graph_replays_different_prompt_lengths() -> None:
    # FlashInfer 0.6.18 cannot mix the Gemma paged and LLaDA2 block-extend
    # generated wrapper variants in one CUDA process. Serving instantiates one
    # model family, so exercise Gemma first in this shared test process.
    _assert_diffusion_gemma_graph_replays_different_prompt_lengths()


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


def test_native_block_extend_graph_captures_paged_append() -> None:
    flashinfer = _require_cuda_and_flashinfer()
    device = torch.device("cuda:0")
    batch_size, q_len, kv_len = 2, 64, 128
    pages_per_row = kv_len // PAGE_SIZE
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    qo_indptr = torch.arange(
        batch_size + 1, dtype=torch.int32, device=device
    ) * q_len
    kv_indptr = torch.arange(
        batch_size + 1, dtype=torch.int32, device=device
    ) * pages_per_row
    kv_indices = torch.arange(
        batch_size * pages_per_row, dtype=torch.int32, device=device
    )
    last_page_len = torch.full(
        (batch_size,), PAGE_SIZE, dtype=torch.int32, device=device
    )
    q_offsets = torch.full(
        (batch_size,), kv_len - q_len, dtype=torch.int32, device=device
    )
    kv_offsets = torch.zeros(batch_size, dtype=torch.int32, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr,
        paged_kv_indptr_buf=kv_indptr,
        paged_kv_indices_buf=kv_indices,
        paged_kv_last_page_len_buf=last_page_len,
        q_offsets_buf=q_offsets,
        kv_offsets_buf=kv_offsets,
        backend="fa2",
        block_extend=True,
        block_size=q_len,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        page_size=PAGE_SIZE,
        causal=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        q_offsets=q_offsets,
        kv_offsets=kv_offsets,
    )
    nnz = batch_size * q_len
    append_batch_indices = torch.arange(
        batch_size, dtype=torch.int32, device=device
    ).repeat_interleave(q_len)
    append_positions = (
        q_offsets.unsqueeze(1)
        + torch.arange(q_len, dtype=torch.int32, device=device).unsqueeze(0)
    ).reshape(-1)
    q = torch.randn(nnz, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(nnz, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    cache = torch.zeros(
        batch_size * pages_per_row,
        2,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )

    def run():
        flashinfer.append_paged_kv_cache(
            k, v, append_batch_indices, append_positions,
            cache, kv_indices, kv_indptr, last_page_len,
        )
        return wrapper.run(q, cache, enable_pdl=False)

    for _ in range(2):
        eager = run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(graph_output, eager, rtol=RTOL, atol=ATOL)
    k.add_(0.125)
    eager_mutated = run().clone()
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(graph_output, eager_mutated, rtol=RTOL, atol=ATOL)


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


def test_fluxserve_runner_isolates_model_family_operations() -> None:
    _require_cuda_and_flashinfer()
    from fluxserve.backend.execution.flashinfer_cuda_graph_runner import (
        FlashInferCudaGraphRunner,
    )

    llada2 = FlashInferCudaGraphRunner("cuda:0", model_family="llada2")
    gemma = FlashInferCudaGraphRunner(
        "cuda:0", model_runner=object(), model_family="diffusion_gemma"
    )

    with pytest.raises(RuntimeError, match="diffusion_gemma.*llada2"):
        llada2.run_gemma_decode(
            key=(), input_ids=None, inputs_embeds=None, position_ids=None,
            cache=None, metadata=None,
        )
    with pytest.raises(RuntimeError, match="llada2.*diffusion_gemma"):
        gemma.capture_decode_batch_sizes(None)

    llada2._gemma_decode_graphs[()] = object()
    llada2.invalidate_llada2(log=False)
    assert () in llada2._gemma_decode_graphs

    gemma._graphs[()] = object()
    gemma.invalidate_gemma(log=False)
    assert () in gemma._graphs


def test_fluxserve_runner_rejects_ambiguous_model_family_configuration() -> None:
    from fluxserve.backend.execution.flashinfer_cuda_graph_runner import (
        FlashInferCudaGraphRunner,
    )

    with pytest.raises(ValueError, match="require model_runner"):
        FlashInferCudaGraphRunner(
            "cuda:0", model_family="diffusion_gemma"
        )
    with pytest.raises(ValueError, match="do not accept model_runner"):
        FlashInferCudaGraphRunner(
            "cuda:0", model_runner=object(), model_family="llada2"
        )


def _assert_diffusion_gemma_graph_replays_different_prompt_lengths() -> None:
    flashinfer = _require_cuda_and_flashinfer()
    from fluxserve.backend.execution.flashinfer_cuda_graph_runner import (
        FlashInferCudaGraphRunner,
        _GemmaAttentionGraphState,
    )
    from fluxserve.backend.layers.attention.diffusion_gemma_flashinfer import (
        DiffusionGemmaLayerGeometry,
        DiffusionGemmaPagedKVCache,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cache = DiffusionGemmaPagedKVCache(
        layer_geometries=[DiffusionGemmaLayerGeometry(NUM_KV_HEADS, HEAD_DIM)],
        max_length=256,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device=device,
        batch_size=1,
    )
    k_cache, v_cache = cache.layer_paged_kv(0)
    torch.manual_seed(17)
    k_cache.normal_()
    v_cache.normal_()
    canvas = 64
    q = torch.randn(
        canvas, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    state = _GemmaAttentionGraphState(
        workspace=workspace,
        cache=cache,
        batch_size=1,
        q_len=canvas,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        sliding_window=None,
        dtype=q.dtype,
        sm_scale=HEAD_DIM**-0.5,
    )

    initial_metadata = cache.build_metadata(
        phase="denoise", seq_ids=(0,), q_offsets=(64,), q_lens=(canvas,),
        kv_lens=(128,), max_q_len=canvas,
    )
    captured_metadata = FlashInferCudaGraphRunner._make_gemma_graph_metadata(
        initial_metadata, cache
    )
    for _ in range(2):
        state.run(q, (k_cache, v_cache), captured_metadata)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = state.run(q, (k_cache, v_cache), captured_metadata)

    for prompt_length in (64, 160):
        metadata = cache.build_metadata(
            phase="denoise",
            seq_ids=(0,),
            q_offsets=(prompt_length,),
            q_lens=(canvas,),
            kv_lens=(prompt_length + canvas,),
            max_q_len=canvas,
        )
        FlashInferCudaGraphRunner._copy_gemma_metadata(
            captured_metadata, metadata
        )
        graph.replay()
        torch.cuda.synchronize(device)

        eager_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device),
            "NHD", backend="fa2",
        )
        eager_wrapper.plan(
            metadata.qo_indptr, metadata.kv_indptr, metadata.kv_indices,
            metadata.last_page_len, num_qo_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS, head_dim_qk=HEAD_DIM,
            page_size=PAGE_SIZE, causal=False, q_data_type=q.dtype,
            kv_data_type=k_cache.dtype, sm_scale=HEAD_DIM**-0.5,
            disable_split_kv=True,
        )
        eager = eager_wrapper.run(q, (k_cache, v_cache))
        torch.testing.assert_close(graph_output, eager, rtol=RTOL, atol=ATOL)


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
    assert FlashInferCudaGraphRunner.capture_batch_sizes(6) == (1, 2, 4, 6)
    assert FlashInferCudaGraphRunner.capture_batch_sizes(8) == (1, 2, 4, 6, 8)
    assert FlashInferCudaGraphRunner.capture_batch_sizes(12) == (1, 2, 4, 6, 8, 10, 12)
    assert FlashInferCudaGraphRunner.capture_batch_sizes(16) == tuple([1, *range(2, 17, 2)])
    assert FlashInferCudaGraphRunner.capture_batch_sizes(32) == tuple([1, *range(2, 33, 2)])
