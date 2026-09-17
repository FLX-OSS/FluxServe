"""Check token/KV progression, including the final unmasked refresh forward."""
from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.execution.decoders.hierarchy import HierarchyDecoder
from fluxserve.backend.execution.decoders.threshold import ThresholdParallelDecoder
from fluxserve.backend.execution.decoders.joint_threshold import JointThresholdDecoder
from fluxserve.backend.execution.runners.fa4_diffusion import FA4DiffusionRunner
from fluxserve.backend.managers.kvcache.dense import TokenArray


def test_executor_shutdown_releases_fa4_graphs():
    import asyncio
    from fluxserve.backend.engine.executor import BlockDiffusionExecutor

    calls = []
    runner = object.__new__(FA4DiffusionRunner)
    runner.fa4_graph_runner = SimpleNamespace(invalidate=lambda: calls.append("released"))
    asyncio.run(BlockDiffusionExecutor(runner, tokenizer=None).shutdown())
    assert calls == ["released"]


@pytest.mark.parametrize("kind", ["threshold", "hierarchy"])
def test_decode_refreshes_final_tokens_before_advancing(kind, monkeypatch):
    monkeypatch.setattr("fluxserve.backend.execution.decoders.hierarchy.broadcast_if_needed", lambda _: None)
    monkeypatch.setattr("fluxserve.backend.execution.decoders.threshold.broadcast_if_needed", lambda _: None)
    runner = object.__new__(FA4DiffusionRunner)
    runner.device, runner.block_length = "cpu", 2
    runner.early_stop, runner.num_forwards = False, 0
    cls = HierarchyDecoder if kind == "hierarchy" else ThresholdParallelDecoder
    runner.decoder = cls(temperature=0, threshold=.9, mask_id=9, eos_id=8)
    runner.past_key_values = SimpleNamespace(layer_paged_kv=lambda _: None)
    runner._make_decode_forward_batch = lambda *_: None
    seen = []
    def model(tokens, **kwargs):
        seen.append(tokens.clone())
        logits = torch.zeros(2, 2, 10)
        logits[..., 3] = 20
        return SimpleNamespace(logits=logits)
    runner.model = model
    x = TokenArray(torch.tensor([[1, 9], [2, 9]]), 0, 9, 8, "cpu")
    starts = torch.zeros(2, dtype=torch.long)
    seq_ids = torch.arange(2)
    runner._decode_selected_batch(x, seq_ids, starts, 2, None, 1)
    assert x.data.tolist() == [[1, 3], [2, 3]]
    assert starts.tolist() == [0, 0]  # Cached inputs still included masks.
    runner._decode_selected_batch(x, seq_ids, starts, 2, None, 1)
    assert seen[-1].tolist() == [[1, 3], [2, 3]]
    assert starts.tolist() == [2, 2]
    assert runner.num_forwards == 2


@pytest.mark.parametrize("max_post_steps", [0, 1, 3])
def test_joint_decode_loop_protects_prompt_and_commits_after_edit_budget(max_post_steps, monkeypatch):
    monkeypatch.setattr(
        "fluxserve.backend.execution.decoders.joint_threshold.broadcast_if_needed",
        lambda _: None,
    )
    runner = object.__new__(FA4DiffusionRunner)
    runner.device, runner.block_length = "cpu", 2
    runner.early_stop, runner.num_forwards = False, 0
    runner.runner_config = SimpleNamespace(max_post_steps=max_post_steps)
    runner.decoder = JointThresholdDecoder(.7, .5, mask_id=9, eos_id=8)
    runner.past_key_values = SimpleNamespace(layer_paged_kv=lambda _: None)
    runner._make_decode_forward_batch = lambda *_: None
    seen = []

    def model(tokens, **kwargs):
        seen.append(tokens.clone())
        logits = torch.zeros(tokens.shape[0], 2, 10)
        # Deliberately oscillate to exercise the per-row editing budget.
        logits[..., 3 if len(seen) % 2 else 4] = 20
        return SimpleNamespace(logits=logits)

    runner.model = model
    x = TokenArray(torch.tensor([[1, 9], [2, 9]]), 0, 9, 8, "cpu")
    starts = torch.zeros(2, dtype=torch.long)
    runner._decode_batches(
        x, starts, 2, None, 1, 2,
        prompt_lengths=torch.ones(2, dtype=torch.long),
    )
    assert x.data[:, 0].tolist() == [1, 2]
    assert starts.tolist() == [2, 2]
    assert torch.equal(seen[-1], x.data)
    assert len(seen) == max_post_steps + 2


@pytest.mark.parametrize("tp_size,ep_size,dp_size,pp_size,backend,error", [
    (1, 1, 1, 1, "none", None),
    (4, 4, 1, 1, "none", None),
    (4, 4, 1, 1, "deepep", "moe_a2a_backend"),
    (4, 1, 1, 1, "none", "TP=EP"),
    (1, 4, 1, 1, "none", "TP=EP"),
    (4, 4, 4, 1, "none", "DP/PP=1"),
    (4, 4, 1, 2, "none", "DP/PP=1"),
])
def test_decode_graph_parallel_topology(
    tp_size, ep_size, dp_size, pp_size, backend, error, monkeypatch
):
    from contextlib import nullcontext
    import fluxserve.backend.execution.runners.fa4_diffusion as module
    from fluxserve.backend.distributed.launch import validate_local_launch_config
    from fluxserve.backend.layers.moe.utils import MoeA2ABackend

    monkeypatch.setattr(module, "get_moe_a2a_backend", lambda: MoeA2ABackend(backend))

    monkeypatch.setattr(module, "validate_fa4_runtime", lambda _: None)
    monkeypatch.setattr(module.BlockDiffusionRunner, "__init__", lambda *a, **k: None)
    monkeypatch.setattr(
        "fluxserve.backend.execution.fa4_cuda_graph_runner.FA4CudaGraphRunner",
        lambda sizes: tuple(sizes),
    )
    with pytest.raises(ValueError, match=error) if error else nullcontext():
        runner = FA4DiffusionRunner(
            model_config=SimpleNamespace(model_type="llada2"),
            server_args=SimpleNamespace(tp_size=tp_size, ep_size=ep_size, dp_size=dp_size, pp_size=pp_size,
                                        max_num_seqs=4),
            runner_config=SimpleNamespace(
                attention_backend="fa4", kv_cache_layout="paged",
                enable_prefill_cuda_graph=False, enable_decode_cuda_graph=True,
                decode_cuda_graph_mode="padded", page_size=16, block_length=16,
                cuda_graph_capture_batch_sizes=[1, 2, 4], supported_batch_sizes=[4],
            ),
            device="cpu",
        )
    if error is None:
        validate_local_launch_config(
            tp_size=tp_size, ep_size=ep_size, dp_size=dp_size, pp_size=pp_size,
            enable_dp_attention=False, device="cuda", visible_device_count=tp_size,
        )
        assert runner.fa4_graph_runner == (1, 2, 4)


def test_fused_decode_synchronizes_tokens_and_progress_before_advancing(monkeypatch):
    import fluxserve.backend.execution.runners.fa4_diffusion as module

    group = object()
    monkeypatch.setattr(module, "get_tp_group", lambda: SimpleNamespace(
        world_size=4, device_group=group))
    monkeypatch.setattr(module.dist, "get_global_rank", lambda g, r: 0)
    calls = []

    def broadcast(tensor, src, group):
        calls.append(tensor.shape)
        if tensor.dtype == torch.long:
            tensor.fill_(3)
        else:
            # Root finished, while this rank's unsynchronized predicates did not.
            tensor.copy_(torch.tensor([[False], [False], [True]]))

    monkeypatch.setattr(module.dist, "broadcast", broadcast)
    runner = object.__new__(FA4DiffusionRunner)
    runner.device, runner.block_length = "cpu", 2
    runner.early_stop, runner.num_forwards = False, 0
    runner.decoder = SimpleNamespace(mask_id=9, eos_id=8)
    runner.fa4_graph_runner = SimpleNamespace(replay=lambda *args: SimpleNamespace(
        logits=None, step=(torch.tensor([[4, 4]]), torch.tensor([False]),
                           torch.tensor([True]), torch.tensor([False]))))
    x = TokenArray(torch.tensor([[3, 3]]), 0, 9, 8, "cpu")
    starts = torch.zeros(1, dtype=torch.long)
    runner._decode_selected_batch(x, torch.tensor([0]), starts, 2, None, 1)
    assert calls == [torch.Size([1, 2]), torch.Size([3, 1])]
    assert x.data.tolist() == [[3, 3]]
    assert starts.tolist() == [2]


def test_ep_shared_expert_output_is_added_once(monkeypatch):
    from fluxserve.backend.models import llada2
    from fluxserve.backend.layers.moe.utils import MoeA2ABackend

    monkeypatch.setattr(llada2, "get_moe_expert_parallel_world_size", lambda: 4)
    monkeypatch.setattr(llada2, "get_moe_a2a_backend", lambda: MoeA2ABackend.NONE)
    block = object.__new__(llada2.LLaDA2SparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.tp_size, block.num_shared_experts, block.alt_stream = 4, 1, None
    block._forward_router_experts = lambda x: x * 2
    block._forward_shared_experts = lambda x: x * 3
    reduced = []

    def all_reduce(x):
        reduced.append(x.clone())
        return x * 4

    monkeypatch.setattr(llada2, "tensor_model_parallel_all_reduce", all_reduce)
    x = torch.ones(2, 3, 8)
    output = block.forward_normal(x)
    torch.testing.assert_close(output, x * 11)  # 4 * routed(2) + shared(3)
    assert len(reduced) == 1
    torch.testing.assert_close(reduced[0], x.view(-1, 8) * 2)
