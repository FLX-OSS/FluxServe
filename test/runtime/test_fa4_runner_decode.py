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
