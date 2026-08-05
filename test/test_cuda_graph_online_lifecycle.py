import asyncio
from unittest.mock import AsyncMock, Mock

from fluxserve.backend.engine.async_llm import AsyncLLM
from fluxserve.backend.engine.distributed_executor import DistributedGenerationExecutor


class _Context:
    is_distributed = False
    is_rank0 = True


def test_local_executor_delegates_cuda_graph_lifecycle():
    base = Mock()
    base.startup = AsyncMock(return_value={"decode_replay_count": 3})
    base.shutdown = AsyncMock()
    base.cuda_graph_stats.return_value = {"decode_replay_count": 3}
    executor = DistributedGenerationExecutor(base, _Context())

    assert asyncio.run(executor.startup()) == {"decode_replay_count": 3}
    assert executor.cuda_graph_stats() == {"decode_replay_count": 3}
    asyncio.run(executor.shutdown_workers())

    base.startup.assert_awaited_once()
    base.shutdown.assert_awaited_once()


def test_engine_metrics_include_cuda_graph_stats():
    executor = Mock()
    executor.cuda_graph_stats.return_value = {
        "prefill_replay_count": 2,
        "decode_replay_count": 7,
    }
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.executor = executor
    engine.metrics = Mock()
    engine.metrics.snapshot.return_value = {"total_requests": 1}

    assert engine.get_metrics_snapshot() == {
        "total_requests": 1,
        "cuda_graph_prefill_replay_count": 2,
        "cuda_graph_decode_replay_count": 7,
    }


def test_engine_metrics_include_online_decode_decomposition_stats():
    executor = Mock()
    executor.cuda_graph_stats.return_value = {
        "decode_decomposed_plan_count": 2,
        "decode_component_replay_count": 5,
        "decode_replay_bs_1": 2,
        "decode_replay_bs_2": 3,
    }
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.executor = executor
    engine.metrics = Mock()
    engine.metrics.snapshot.return_value = {}

    assert engine.get_metrics_snapshot() == {
        "cuda_graph_decode_decomposed_plan_count": 2,
        "cuda_graph_decode_component_replay_count": 5,
        "cuda_graph_decode_replay_bs_1": 2,
        "cuda_graph_decode_replay_bs_2": 3,
    }
