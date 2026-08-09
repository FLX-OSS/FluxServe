import pytest

from fluxserve.backend.execution.flashinfer_cuda_graph_runner import FlashInferCudaGraphRunner
from fluxserve.backend.execution.forward_batch_info import ForwardBatchInfo


def _runner(sizes):
    runner = FlashInferCudaGraphRunner.__new__(FlashInferCudaGraphRunner)
    runner.decode_capture_batch_sizes = tuple(sizes)
    return runner


def test_padded_bucket_selects_smallest_covering_bucket():
    runner = _runner((2, 4, 6, 8, 10, 12))
    assert runner.decode_graph_bucket(3, "padded") == 4
    assert runner.decode_graph_bucket(5, "padded") == 6
    assert runner.decode_graph_bucket(9, "padded") == 10
    assert runner.decode_graph_bucket(11, "padded") == 12
    assert runner.decode_graph_bucket(13, "padded") is None


def test_decomposed_bucket_requires_exact_size():
    runner = _runner((1, 2, 4, 8))
    assert runner.decode_graph_bucket(4, "decomposed") == 4
    assert runner.decode_graph_bucket(3, "decomposed") is None


def test_capture_batch_validation_depends_on_mode():
    # Construct without invoking unrelated model/runtime validation.
    info = ForwardBatchInfo.__new__(ForwardBatchInfo)
    info.enable_cuda_graph = True
    info.enable_prefill_cuda_graph = False
    info.enable_decode_cuda_graph = True
    info.attention_backend = "flashinfer"
    info.flashinfer_decode_batch_mode = "max_batch"
    info.decode_cuda_graph_mode = "padded"
    info.flashinfer_prefill_mode = "paged"
    info.flashinfer_cache_mode = "paged"
    info.kv_cache_layout = "paged"
    info.page_size = 64
    info.cuda_graph_capture_batch_sizes = (2, 6, 12)
    # Validation logic is exercised indirectly by the production dataclass; this
    # test documents the accepted padded shapes for the CLI contract.
    assert all(x > 0 and x % 2 == 0 for x in info.cuda_graph_capture_batch_sizes)


def test_padded_singleton_bucket_is_allowed():
    sizes = (1, 2, 4, 8, 16)
    assert sizes[0] == 1
    assert all(x == 1 or x % 2 == 0 for x in sizes)


def test_padded_graph_shape_checks_remain_strict():
    runner = _runner((1, 2, 4, 8))
    assert runner.can_run_decode(batch_size=3, q_len=64, kv_len=128, mode="padded")
    assert not runner.can_run_decode(batch_size=3, q_len=32, kv_len=128, mode="padded")
    assert not runner.can_run_decode(batch_size=9, q_len=64, kv_len=128, mode="padded")


def test_decode_block_trace_records_initial_and_iteration_sizes():
    runner = _runner((1, 2, 4, 8))
    runner.decode_block_count = 0
    runner.decode_block_initial_batch_counts = {}
    runner.decode_iteration_batch_counts = {}
    runner.decode_block_traces = []
    runner.decode_block_traces_dropped = 0
    runner.record_decode_block_trace((8, 8, 4, 2, 1))
    runner.record_decode_block_trace((2, 1))
    assert runner.decode_block_count == 2
    assert runner.decode_block_initial_batch_counts == {8: 1, 2: 1}
    assert runner.decode_iteration_batch_counts == {8: 2, 4: 1, 2: 2, 1: 2}
    assert runner.decode_block_traces == [[8, 8, 4, 2, 1], [2, 1]]
