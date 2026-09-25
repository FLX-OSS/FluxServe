"""Self-speculation graphs: weight ownership, rollback positions and replay."""

from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.execution.nemotron_cuda_graph_runner import NemotronCudaGraphRunner
from fluxserve.backend.execution.runners.nemotron import get_nemotron_runner
from fluxserve.backend.model_loader.nemotron import NemotronDraftAdapter
from test_nemotron_flashinfer_graph import BLOCK, replay_fixture


@pytest.mark.parametrize("backend", ["fa4", "flashinfer"])
@pytest.mark.parametrize("fail", [False, True])
def test_capture_owns_phase_weights_and_restores_base(monkeypatch, backend, fail):
    model = object.__new__(get_nemotron_runner(backend, "self_speculation"))
    layer = torch.nn.Linear(2, 2, bias=False)
    base = layer.weight.data
    draft = base.clone() + 1
    model.lora = NemotronDraftAdapter([(layer, base, draft)])
    model.past_key_values = SimpleNamespace(device="cpu")
    graph = NemotronCudaGraphRunner((1, 2), backend=backend)
    graph.cache = model.past_key_values
    graph.pool = object()
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    seen = []

    def capture(runner, batch_size, causal):
        seen.append((batch_size, causal, layer.weight.data_ptr()))
        assert layer.weight.data_ptr() == (base if causal else draft).data_ptr()
        if fail:
            raise RuntimeError("capture failed")
        return object()

    monkeypatch.setattr(graph, "_capture", capture)
    if fail:
        with pytest.raises(RuntimeError, match="capture failed"):
            graph.capture(model)
    else:
        graph.capture(model)
        assert len(seen) == 4
        graph.capture(model)
        assert len(seen) == 4, "unchanged captures must be reused"
    assert not model.lora.enabled
    assert layer.weight.data_ptr() == base.data_ptr()


@pytest.mark.parametrize("backend", ["fa4", "flashinfer"])
@pytest.mark.parametrize("causal", [False, True])
def test_replay_updates_unaligned_prefix_after_rollback_and_recycles_rows(backend, causal):
    graph, model, stub, state = replay_fixture()
    graph.backend = backend
    entry = graph.entries.pop((2, False))
    entry.dummy_pages.fill_(31)
    graph.entries[(2, causal)] = entry
    if backend == "fa4":
        entry.state = None
    # Prefixes diverge after different acceptance lengths, then a previous
    # row finishes and its slot is reused. The padded row must lose its old KV.
    for prefixes, slots in [([17, 35], [1, 3]), ([20, 36], [1, 3]), ([5], [2])]:
        positions = torch.tensor(prefixes)[:, None] + torch.arange(BLOCK)
        graph.replay(model, input_ids=torch.zeros_like(positions),
                     position_ids=positions, seq_ids=torch.tensor(slots), causal=causal)
        count = len(slots)
        table = model.past_key_values.page_table[slots]
        expected = table.gather(1, positions // BLOCK).long() * BLOCK + positions % BLOCK
        assert torch.equal(entry.metadata.slot_mapping.view(2, BLOCK)[:count], expected)
        assert entry.metadata.kv_lens[:count].tolist() == [p + BLOCK for p in prefixes]
    assert entry.metadata.page_table[1].tolist() == [31] * 4
    assert entry.metadata.kv_lens[1].item() == BLOCK
    assert stub.replays == 3
    if backend == "flashinfer":
        assert len(state.wrapper.plans) == 1
        assert state.kv_indptr_buf.tolist() == [0, 2, 3]


@pytest.mark.parametrize("backend", ["fa4", "flashinfer"])
def test_spec_launch_uses_graph_logits_and_eager_fallback(backend):
    runner = object.__new__(get_nemotron_runner(backend, "self_speculation"))
    runner.device = "cpu"
    runner.block_length = BLOCK
    rows = [SimpleNamespace(seq_id=2, prefix_length=19, block=torch.arange(BLOCK))]
    logits = torch.randn(1, BLOCK, 5)
    seen = []

    def replay(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(logits=logits)

    runner._graph_replay = replay
    runner._paged_forward = lambda **kwargs: pytest.fail("unexpected eager fallback")
    for causal in (False, True):
        assert runner._spec_launch(rows, causal=causal) is logits
        assert seen[-1]["causal"] is causal
        assert seen[-1]["positions"].tolist() == [list(range(19, 19 + BLOCK))]
    runner._graph_replay = lambda **kwargs: None
    runner._paged_forward = lambda **kwargs: logits
    assert runner._spec_launch(rows, causal=True) is logits


def test_reloading_adapter_invalidates_graphs_before_replacing_weights(monkeypatch):
    from fluxserve.backend.model_loader import nemotron

    runner = object.__new__(get_nemotron_runner("fa4", "self_speculation"))
    seen = []
    runner.nemotron_graph_runner = SimpleNamespace(invalidate=lambda: seen.append("invalidate"))
    runner.lora = SimpleNamespace(apply=lambda enabled: seen.append(enabled))
    runner.model = runner.model_config = None
    monkeypatch.setattr(nemotron, "load_nemotron_lora", lambda *args: seen.append("load"))
    runner.load_draft_adapter()
    assert seen == ["invalidate", False, "load"]


@pytest.fixture
def isolated_token_paged_state():
    """Keep this test's eager FlashInfer state out of the process-wide cache.

    ``_STATES`` is keyed by device and lives for the process, and the state is
    created lazily by the first dispatch. Creating it under ``inference_mode``
    makes its plan buffers inference tensors permanently, so any later test that
    plans outside inference mode dies on `Inplace update to inference tensor`.
    That made the suite order-dependent: this file before
    ``test_nemotron_flashinfer.py`` broke ``test_live_flashinfer_matches_dense``.
    """
    from fluxserve.backend.layers.attention import flashinfer_token

    saved = dict(flashinfer_token._STATES)
    flashinfer_token._STATES.clear()
    try:
        yield
    finally:
        flashinfer_token._STATES.clear()
        flashinfer_token._STATES.update(saved)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and paged attention kernels")
@pytest.mark.parametrize("backend", ["fa4", "flashinfer"])
@pytest.mark.parametrize("adapter", [False, True])
@torch.inference_mode()
def test_gpu_graph_matches_eager_logits_acceptance_and_rollback(
    backend, adapter, isolated_token_paged_state
):
    """Real model/kernels with small weights; exercise the actual speculate loop."""
    from fluxserve.backend.execution.decoders.nemotron import NemotronThresholdDecoder, load_thinking_budget
    from fluxserve.backend.execution.runners.nemotron_selfspec_paged import SpecRow
    from fluxserve.backend.managers.kvcache.paged import PagedKVCache
    from fluxserve.backend.models.nemotron_diffusion import NemotronLabsDiffusionLLM
    from test_nemotron_model import tiny_config

    torch.manual_seed(17)
    config = tiny_config(hidden_size=256, intermediate_size=512, head_dim=128,
                         num_attention_heads=2, num_key_value_heads=1,
                         vocab_size=128, mask_token_id=100, eos_token_id=11, block_size=32)
    runner = object.__new__(get_nemotron_runner(backend, "self_speculation"))
    runner.model = NemotronLabsDiffusionLLM(config).eval().to(device="cuda")
    for parameter in runner.model.parameters():
        # Keep RoPE buffers float32, as the production loader does.
        parameter.data = parameter.data.to(torch.bfloat16)
        if parameter.dim() == 1:
            parameter.fill_(1)
        else:
            parameter.normal_(std=0.04)
    runner.device = "cuda"
    runner.block_length = BLOCK
    runner.num_forwards = 0
    runner.runner_config = SimpleNamespace(threshold=0.9, mask_id=100, eos_id=11, eos_ids=(11,))
    runner.init_decoder()
    runner.draft_decoder = NemotronThresholdDecoder(threshold=0, mask_id=100, eos_ids=(11,), draft=True)
    runner.max_denoise_steps = BLOCK
    runner.thinking_budget = load_thinking_budget(runner.runner_config)
    runner.tp_group = SimpleNamespace(barrier=lambda: None)
    runner._make_forward_batch = lambda *args, **kwargs: None
    runner.past_key_values = PagedKVCache(
        num_layers=2, batch_size=4, local_kv_heads=1, max_length=256,
        head_dim=128, page_size=BLOCK, reserve_dummy_page=4,
        dtype=torch.bfloat16, device="cuda",
    )
    runner.lora = None
    if adapter:
        layers = []
        for layer in runner.model.model.layers:
            module = layer.self_attn.o_proj
            base = module.weight.data
            draft = base + torch.randn_like(base) * 0.03
            layers.append((module, base, draft))
        runner.lora = NemotronDraftAdapter(layers)
    graph = NemotronCudaGraphRunner((1, 4), backend=backend)
    graph.capture(runner)

    def rows():
        result = []
        for index, (prefix, budget) in enumerate(((17, 20), (35, 37), (70, 54))):
            block = torch.full((BLOCK,), 100, dtype=torch.long, device="cuda")
            block[0] = index + 1
            result.append(SpecRow(index=index, seq_id=index, prompt_length=prefix,
                                  generation_length=budget, prefix_length=prefix,
                                  block=block, seed=index + 1, stop_on_eos=False))
        return result

    # Check logits as well as decoded tokens: verify can hide a broken drafter.
    for causal in (False, True):
        runner.past_key_values.data.zero_()
        runner.nemotron_graph_runner = None
        with runner._adapters(enabled=not causal):
            eager = runner._spec_launch(rows(), causal=causal).clone()
        runner.past_key_values.data.zero_()
        runner.nemotron_graph_runner = graph
        with runner._adapters(enabled=not causal):
            graphed = runner._spec_launch(rows(), causal=causal).clone()
        torch.testing.assert_close(graphed, eager, atol=0.025, rtol=0.025)
    results = []
    for enabled in (False, True):
        runner.past_key_values.data.zero_()
        runner.nemotron_graph_runner = graph if enabled else None
        batch = rows()
        runner._run_speculation(batch)
        results.append([(r.emitted, r.prefix_length, r.stats.accepted_per_iteration) for r in batch])
    assert results[0] == results[1]
    assert graph.replay_count > 2
    assert graph.padded_rows > 0
    assert any(length < BLOCK for row in results[1] for length in row[2])
    graph.invalidate()
