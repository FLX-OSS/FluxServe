"""Native FlashInfer planning, suffix causality and Nemotron dispatch."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from fluxserve.backend.execution.forward_batch_info import ForwardBatch, RunnerConfig
from fluxserve.backend.execution.runners.nemotron import get_nemotron_runner
from fluxserve.backend.execution.runners.nemotron_fa4 import build_nemotron_paged_metadata
from fluxserve.backend.execution.runners.nemotron_flashinfer import (
    NemotronFlashInferDiffusionRunner,
    NemotronFlashInferSelfSpecRunner,
)
from fluxserve.backend.layers.attention.base import AttentionForwardConfig
from fluxserve.backend.layers.attention.flashinfer_token import (
    FlashInferTokenPagedAttention,
    FlashInferTokenPagedState,
)
from fluxserve.backend.layers.attention.forward import AttentionForward
from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args
from test_nemotron_model import checkpoint_config, serve_args


class ReferenceWrapper:
    """Emulate the public wrapper from its CSR page list, not slot_mapping."""

    def __init__(self, workspace, *, kv_layout, backend):
        assert kv_layout == "NHD" and backend == "fa2"
        self.plans = []

    def plan(self, qo_indptr, kv_indptr, indices, last_page_len, **kwargs):
        self.plans.append((qo_indptr, kv_indptr, indices, last_page_len, kwargs))

    def run(self, q, cache):
        qo, kv, indices, last, options = self.plans[-1]
        outputs = []
        for row in range(len(last)):
            pages = indices[kv[row]:kv[row + 1]].long()
            length = (len(pages) - 1) * options["page_size"] + int(last[row])
            k, v = [t[pages].flatten(0, 1)[:length].transpose(0, 1) for t in cache]
            query = q[qo[row]:qo[row + 1]].transpose(0, 1)
            q_len = query.shape[1]
            mask = None
            if options["causal"]:
                mask = torch.arange(length)[None, :] <= (
                    torch.arange(q_len)[:, None] + length - q_len
                )
            outputs.append(F.scaled_dot_product_attention(
                query.float(), k.float(), v.float(), attn_mask=mask,
                scale=options["sm_scale"], enable_gqa=True,
            ).to(q.dtype).transpose(0, 1))
        return torch.cat(outputs)


def attention_case(device="cpu", head_dim=8, offsets=(17, 29)):
    torch.manual_seed(42)
    metadata = replace(build_nemotron_paged_metadata(
        phase="decode", q_offsets=torch.tensor(offsets, device=device),
        q_lens=torch.tensor([3, 5], device=device),
        page_table=torch.tensor([[4, 1, -1], [0, 3, 2]], device=device),
        max_input_len=5, block_length=32, page_size=16, causal=False,
    ), backend="flashinfer")
    config = AttentionForwardConfig(
        layer_id=0, num_heads=4, num_kv_heads=2, head_dim=head_dim,
        num_key_value_groups=2, scale=head_dim ** -0.5,
    )
    q = torch.randn(2, 4, 5, head_dim, device=device, dtype=torch.bfloat16)
    k, v = [torch.randn(2, 2, 5, head_dim, device=device, dtype=q.dtype) for _ in range(2)]
    cache = tuple(torch.randn(5, 16, 2, head_dim, device=device, dtype=q.dtype) for _ in range(2))
    return config, metadata, q, k, v, cache


def expected_attention(q, k, v, original_cache, metadata, config):
    output = torch.zeros_like(q)
    for row, (offset, length) in enumerate(zip(
        metadata.q_offsets_cpu, metadata.q_lens_cpu, strict=True
    )):
        positions = torch.arange(offset, device=q.device)
        physical = metadata.page_table[row, positions // 16].long()
        prefix = [t[physical, positions % 16].transpose(0, 1) for t in original_cache]
        keys = torch.cat((prefix[0], k[row, :, :length]), dim=1)
        values = torch.cat((prefix[1], v[row, :, :length]), dim=1)
        mask = None
        if metadata.causal:
            mask = torch.arange(offset + length, device=q.device)[None, :] <= (
                offset + torch.arange(length, device=q.device)[:, None]
            )
        output[row, :, :length] = F.scaled_dot_product_attention(
            q[row, :, :length].float(), keys.float(), values.float(),
            attn_mask=mask, enable_gqa=True, scale=config.scale,
        ).to(q.dtype)
    return output


@pytest.mark.parametrize("causal", [False, True])
def test_packed_attention_matches_dense_and_writes_only_owned_slots(monkeypatch, causal):
    monkeypatch.setenv("FLASHINFER_WORKSPACE_SIZE", "1024")
    config, metadata, q, k, v, cache = attention_case()
    metadata = replace(metadata, causal=causal)
    original = tuple(t.clone() for t in cache)
    state = FlashInferTokenPagedState("cpu", wrapper_factory=ReferenceWrapper)
    router = AttentionForward(config)
    router.flashinfer_token_paged = FlashInferTokenPagedAttention(config, state=state)
    output, _ = router.forward(
        q, k, v, past_key_values=cache, use_cache=True,
        forward_batch=ForwardBatch(paged_attention_metadata=metadata),
    )
    torch.testing.assert_close(output, expected_attention(q, k, v, original, metadata, config))
    untouched = torch.ones(80, dtype=torch.bool)
    untouched[metadata.slot_mapping] = False
    for before, after in zip(original, cache, strict=True):
        torch.testing.assert_close(before.flatten(0, 1)[untouched], after.flatten(0, 1)[untouched])
    qo, kv, indices, last, options = state.wrapper.plans[0]
    assert qo.tolist() == [0, 3, 8]
    assert kv.tolist() == [0, 2, 5]
    assert indices.tolist() == [4, 1, 0, 3, 2]
    assert last.tolist() == [4, 2]
    assert options["causal"] is causal
    assert torch.count_nonzero(output[0, :, 3:]) == 0


def test_plan_reuse_is_per_forward_and_includes_causality_pages_and_scale(monkeypatch):
    monkeypatch.setenv("FLASHINFER_WORKSPACE_SIZE", "1024")
    config, metadata, q, k, v, cache = attention_case()
    state = FlashInferTokenPagedState("cpu", wrapper_factory=ReferenceWrapper)
    adapter = FlashInferTokenPagedAttention(config, state=state)
    adapter.forward(q, k, v, cache, metadata)
    adapter.forward(q, k, v, cache, metadata)  # next transformer layer
    assert len(state.wrapper.plans) == 1
    metadata = replace(metadata, causal=True)
    adapter.forward(q, k, v, cache, metadata)
    assert len(state.wrapper.plans) == 2
    metadata = replace(metadata, page_table=torch.tensor([[1, 4, -1], [3, 0, 2]]))
    # Test planning in isolation: the runner normally rebuilds slot_mapping too.
    state.run(q.transpose(1, 2).reshape(-1, 4, 8)[:8], cache, metadata, config)
    assert len(state.wrapper.plans) == 3
    state.run(q.transpose(1, 2).reshape(-1, 4, 8)[:8], cache, metadata, replace(config, scale=0.5))
    assert len(state.wrapper.plans) == 4


def test_requested_flashinfer_never_falls_back_to_fa4_or_dense():
    config, metadata, q, k, v, cache = attention_case()
    with pytest.raises(RuntimeError, match="FlashInfer token paged attention"):
        AttentionForward(config).forward(
            q, k, v, past_key_values=cache,
            forward_batch=ForwardBatch(paged_attention_metadata=metadata),
        )


@pytest.mark.parametrize("decoding,cls", [
    ("threshold", NemotronFlashInferDiffusionRunner),
    ("self_speculation", NemotronFlashInferSelfSpecRunner),
])
def test_normalization_and_shared_dispatch(decoding, cls):
    args = serve_args(
        attention_backend="flashinfer", kv_cache_layout="paged",
        parallel_decoding=decoding, scheduler_policy="paged",
    )
    assert normalize_nemotron_args(args, checkpoint_config())
    assert get_nemotron_runner(args.attention_backend, args.parallel_decoding) is cls


@pytest.mark.parametrize("overrides,match", [
    ({"kv_cache_layout": "dense"}, "kv-cache-layout paged"),
    ({"flashinfer_cache_mode": "dense"}, "flashinfer-cache-mode paged"),
    ({"flashinfer_prefill_mode": "ragged"}, "flashinfer-prefill-mode paged"),
    ({"use_prefill_cuda_graph": True}, "not captured"),
])
def test_unsupported_flashinfer_combinations_fail_early(overrides, match):
    values = dict(attention_backend="flashinfer", kv_cache_layout="paged")
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        normalize_nemotron_args(serve_args(**values), checkpoint_config())


@pytest.mark.parametrize("flag", ["use_decode_cuda_graph", "use_cuda_graph"])
def test_decode_graphs_are_accepted_on_the_flashinfer_path(flag):
    # Decode graphs used to be FA4's alone. The runner now captures them here
    # too, re-planning FlashInfer outside each replay.
    args = serve_args(**{
        "attention_backend": "flashinfer", "kv_cache_layout": "paged", flag: True,
    })
    assert normalize_nemotron_args(args, checkpoint_config())


@pytest.mark.parametrize("cls", [NemotronFlashInferDiffusionRunner, NemotronFlashInferSelfSpecRunner])
def test_runner_initialization_does_not_require_fa4_or_dllm(monkeypatch, cls):
    from fluxserve.backend.execution.runners import nemotron_flashinfer as module

    def init(self, *args, runner_config, _allow_flashinfer, **kwargs):
        assert _allow_flashinfer
        self.runner_config = runner_config
        self.block_length = runner_config.block_length
        self.init_decoder()

    monkeypatch.setattr(module, "require_flashinfer_token_paged", lambda: ReferenceWrapper)
    monkeypatch.setattr(module.BlockDiffusionRunner, "__init__", init)
    runner = cls(model_config=checkpoint_config(), runner_config=RunnerConfig(
        attention_backend="flashinfer", kv_cache_layout="paged", page_size=32,
        block_length=32, flashinfer_cache_mode="paged", flashinfer_prefill_mode="paged",
    ))
    assert runner.paged_attention_backend == "flashinfer"
    # No graph runner without the decode-graph flag in the config above.
    assert runner.nemotron_graph_runner is None
    assert runner._request_seeds == {}
    if cls is NemotronFlashInferSelfSpecRunner:
        assert runner._request_prefix == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA allocation")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("offsets", [(0, 0), (17, 29)])
def test_live_flashinfer_matches_dense(causal, offsets):
    pytest.importorskip("flashinfer")
    config, metadata, q, k, v, cache = attention_case("cuda", head_dim=128, offsets=offsets)
    metadata = replace(metadata, causal=causal)
    expected = expected_attention(q, k, v, cache, metadata, config)
    adapter = FlashInferTokenPagedAttention(config)
    actual = adapter.forward(q, k, v, cache, metadata)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


def test_real_runner_forward_marks_native_flashinfer_metadata():
    from fluxserve.backend.managers.kvcache import PagedKVCache

    captured = {}

    class Model:
        model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1))

        def __call__(self, tokens, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(logits=tokens)

    runner = object.__new__(NemotronFlashInferDiffusionRunner)
    runner.device = "cpu"
    runner.block_length = 32
    runner.num_forwards = 0
    runner.model = Model()
    runner._make_forward_batch = lambda *args, **kwargs: ForwardBatch()
    runner.past_key_values = PagedKVCache(
        num_layers=1, batch_size=1, local_kv_heads=2, max_length=64,
        head_dim=8, page_size=16, dtype=torch.bfloat16, device="cpu",
    )
    runner._paged_forward(
        seq_ids=torch.tensor([0]), tokens=torch.ones(1, 32, dtype=torch.long),
        q_offsets=torch.tensor([17]), causal=True, is_prefill=False, max_input_len=32,
    )
    metadata = captured["forward_batch"].paged_attention_metadata
    assert metadata.backend == "flashinfer" and metadata.causal
    assert metadata.kv_lens.tolist() == [49]
    assert metadata.q_offsets_cpu == (17,)
    assert captured["position_ids"].tolist() == [list(range(17, 49))]


def diffusion_harness():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent / "integration" / "nemotron_diffusion_harness.py"
    spec = importlib.util.spec_from_file_location("diffusion_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mutation", ["none", "tokens", "causal_commit", "missing_fixture"])
def test_flashinfer_checkpoint_gate_detects_divergence(tmp_path, mutation):
    import copy

    dense = {"results": {"case": {
        "generated": [1, 2],
        "stats": {"prefill_calls": 1, "denoise_calls": 1, "commit_calls": 1},
    }}}
    candidate = copy.deepcopy(dense)
    candidate["provenance"] = {"test": True}
    candidate["results"]["case"]["launches"] = [
        {"prefill": True, "causal": True},
        {"prefill": False, "causal": False},
        {"prefill": False, "causal": True},
    ]
    if mutation == "tokens":
        candidate["results"]["case"]["generated"] = [3, 4]
    elif mutation == "causal_commit":
        candidate["results"]["case"]["launches"][-1]["causal"] = False
    elif mutation == "missing_fixture":
        candidate["results"].clear()
    torch.save(candidate, tmp_path / "flashinfer.pt")
    record = diffusion_harness().compare_flashinfer(tmp_path, dense)
    assert all(record["checks"].values()) == (mutation == "none")


def test_flashinfer_checkpoint_gate_requires_an_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        diffusion_harness().compare_flashinfer(tmp_path, {"results": {}})
