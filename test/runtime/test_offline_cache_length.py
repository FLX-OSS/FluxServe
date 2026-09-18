from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fluxserve.cli.bench_offline import build_runner_config, resolve_paged_cache_length
from fluxserve.cli import build_parser
from fluxserve.backend.execution.runners import diffusion_gemma


def cache_args(*options):
    return build_parser().parse_args([
        "bench_offline", "--model", "model", "--dataset", "data.jsonl",
        "--block-length", "256", "--gen-len", "512", "--batch-size", "4",
        *options,
    ])


def batch_info():
    return SimpleNamespace(
        input_lengths=[150, 441],
        padded_gen_lens=[522, 519],
        sorted_indices=[0, 1],
        max_length=953,
        prefill_lengths=[256],
        supported_batch_sizes=(1, 2, 4),
    )


def test_gemma_automatic_capacity_includes_complete_canvases():
    args = cache_args()
    assert args.max_model_length is None
    assert resolve_paged_cache_length(args, batch_info(), is_diffusion_gemma=True) == 1209


@pytest.mark.parametrize("capacity", [1209, 1536, 2048])
def test_explicit_capacity_reaches_runner_configuration(capacity):
    args = cache_args("--max-model-length", str(capacity))
    batch = batch_info()
    batch.max_length = resolve_paged_cache_length(args, batch, is_diffusion_gemma=True)
    config = build_runner_config(args, batch)
    assert config.max_length == capacity
    assert config.gen_length == 512
    assert max(config.cache_lengths) == capacity


@pytest.mark.parametrize("length", ["0", "-1"])
def test_rejects_nonpositive_capacity(length):
    with pytest.raises(ValueError, match="must be positive"):
        resolve_paged_cache_length(cache_args("--max-model-length", length), batch_info())


def test_rejects_capacity_that_ignores_canvas_rounding():
    with pytest.raises(ValueError, match="requires at least 1209"):
        resolve_paged_cache_length(
            cache_args("--max-model-length", "1024"), batch_info(),
            is_diffusion_gemma=True,
        )
    assert resolve_paged_cache_length(
        cache_args("--max-model-length", "1209"), batch_info(),
        is_diffusion_gemma=True,
    ) == 1209


def test_capacity_covers_warmup_for_small_dataset():
    batch = batch_info()
    batch.input_lengths = [10]
    batch.padded_gen_lens = [22]
    assert resolve_paged_cache_length(cache_args(), batch, is_diffusion_gemma=True) == 512


def test_llada_capacity_accounts_for_batch_padding_and_block_rounding():
    args = cache_args("--block-length", "64")
    batch = batch_info()
    # 441 + 522 = 963, rounded to a full 64-token block.
    assert resolve_paged_cache_length(args, batch) == 1024


def test_dense_cache_keeps_existing_capacity_and_rejects_override():
    args = cache_args("--kv-cache-layout", "dense")
    assert resolve_paged_cache_length(args, batch_info()) == 953
    args.max_model_length = 2048
    with pytest.raises(ValueError, match="requires paged KV cache"):
        resolve_paged_cache_length(args, batch_info())


def test_gemma_preallocated_cache_survives_larger_measured_batch(monkeypatch):
    allocations = []

    def allocate(**kwargs):
        allocations.append(kwargs)
        return SimpleNamespace(
            **{k: v for k, v in kwargs.items() if k != "layer_geometries"},
            layer_geometries=tuple(kwargs["layer_geometries"]),
        )

    monkeypatch.setattr(diffusion_gemma, "DiffusionGemmaPagedKVCache", allocate)
    graph_runner = SimpleNamespace(invalidate_gemma=Mock())
    runner = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(layers=[
            SimpleNamespace(self_attn=SimpleNamespace(num_kv_heads=1, head_dim=128))
        ])),
        runner_config=SimpleNamespace(page_size=256),
        block_length=256,
        device="cpu",
        flashinfer_graph_runner=graph_runner,
    )
    capacity = resolve_paged_cache_length(
        cache_args("--max-model-length", "2048"), batch_info(),
        is_diffusion_gemma=True,
    )
    cache = diffusion_gemma.DiffusionGemmaRunner._paged_cache(runner, capacity, batch_size=4)
    # Initial allocation precedes capture. Neither warmup nor measured batches
    # may replace that allocation and invalidate the subsequently captured graph.
    graph_runner.invalidate_gemma.reset_mock()
    for required in (512, 918, 1209):
        assert diffusion_gemma.DiffusionGemmaRunner._paged_cache(
            runner, required, batch_size=4
        ) is cache
    assert len(allocations) == 1
    assert allocations[0]["max_length"] == 2048
    graph_runner.invalidate_gemma.assert_not_called()
