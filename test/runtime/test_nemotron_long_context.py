"""Long-context limits, chunked prefill and absolute-position boundaries."""

from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.model_loader.nemotron import (
    MAX_SUPPORTED_POSITIONS, check_nemotron_context_limit, normalize_nemotron_args,
)
from fluxserve.backend.execution.runners.nemotron_fa4 import build_nemotron_paged_metadata
from fluxserve.backend.models.nemotron_diffusion import nemotron_query_scale
from test_nemotron_model import checkpoint_config, serve_args


@pytest.mark.parametrize("length", [16384, 16385, 32768, 65536, 131072, 262144])
def test_normalizer_accepts_the_checkpoint_window(length):
    args = serve_args(max_model_len=length, attention_backend="fa4", kv_cache_layout="paged")
    assert normalize_nemotron_args(args, checkpoint_config())
    assert args.max_model_len == length


def test_smaller_checkpoint_and_serving_limits_are_still_enforced():
    with pytest.raises(ValueError, match="32768"):
        check_nemotron_context_limit(32769, checkpoint_config(max_position_embeddings=32768))
    with pytest.raises(ValueError, match="8192"):
        check_nemotron_context_limit(8193, checkpoint_config(), serving_limit=8192)
    with pytest.raises(ValueError, match="262144"):
        check_nemotron_context_limit(MAX_SUPPORTED_POSITIONS + 1)


@pytest.mark.parametrize("offset", [16383, 16384, 32767, 65535, 262112])
def test_paged_slots_retain_absolute_positions_across_long_context_boundaries(offset):
    page_count = (offset + 32 + 31) // 32
    table = torch.arange(page_count - 1, -1, -1).unsqueeze(0)
    metadata = build_nemotron_paged_metadata(
        phase="decode", q_offsets=torch.tensor([offset]), q_lens=torch.tensor([32]),
        page_table=table, max_input_len=32, block_length=32, page_size=32, causal=True,
    )
    positions = torch.arange(offset, offset + 32)
    assert metadata.kv_lens.tolist() == [offset + 32]
    assert torch.equal(metadata.slot_mapping, table[0, positions // 32] * 32 + positions % 32)
    expected = 1 + 0.1 * torch.log(1 + torch.floor(positions / 16384))
    torch.testing.assert_close(nemotron_query_scale(positions, 0.1, 16384).squeeze(-1), expected)


def test_chunked_dense_prefill_matches_full_causal_logits_and_kv():
    from test_nemotron_model import initialized_tiny_model
    from test_nemotron_diffusion import make_runner

    model = initialized_tiny_model(seed=8)
    runner = make_runner(model)
    runner.runner_config.nemotron_prefill_chunk_size = 32
    prompt = torch.randint(0, 64, (1, 273), generator=torch.Generator().manual_seed(2))
    expected = model(prompt, use_cache=True, attention_mask=runner._causal_mask(273, 0, "cpu"))
    config = model.model.config
    cache = torch.zeros(config.num_hidden_layers, 2, 1, config.num_key_value_heads, 273, config.head_dim)
    logits = runner._prefill(prompt, cache)
    torch.testing.assert_close(logits, expected.logits[:, -17:], atol=1e-4, rtol=1e-4)
    for layer in range(config.num_hidden_layers):
        for kind in range(2):
            torch.testing.assert_close(cache[layer, kind], expected.past_key_values[2 * layer + kind],
                                       atol=1e-4, rtol=1e-4)
    assert runner._last_prefill_calls == 9


@pytest.mark.parametrize("speculative", [False, True])
def test_paged_prefill_chunks_long_prompts_without_sampling_intermediate_chunks(speculative):
    from test_nemotron_fa4 import make_paged_runner
    from test_nemotron_selfspec_paged import make_runner

    def script(index, rows, tokens, length):
        logits = torch.full((len(rows), length, 16), -torch.inf)
        logits[..., 1] = 0
        return logits

    runner = make_runner(script) if speculative else make_paged_runner(script)
    runner.runner_config.nemotron_prefill_chunk_size = 1024
    runner.generate(torch.full((1, 16387), 5), [16387], [4],
                    [{"temperature": 0.7, "seed": 123}])
    calls = [call for call in runner.launches if call["prefill"]]
    assert len(calls) == 17
    assert [call["offsets"] for call in calls] == [[1024 * i] for i in range(17)]
    assert max(len(call["tokens"][0]) for call in calls) == 1024
    assert len(calls[-1]["tokens"][0]) == 3
    assert runner.last_stats[0]["prefill_calls"] == 17


def test_input_budget_reserves_space_for_a_speculative_tail():
    from fluxserve.backend.engine.processor import InputProcessor

    args = SimpleNamespace(max_model_len=32768, generation_block_size=32,
                           sampling_defaults={"temperature": 0.7, "seed": 3},
                           speculative_context_margin=32)
    processor = InputProcessor(args, None)
    state = processor.make_state(dict(
        rid="r", text=None, input_ids=[5] * 32700, sampling_params={"max_tokens": 1024},
    ))
    assert len(state.input_ids) + state.max_new_tokens + 32 <= args.max_model_len
    with pytest.raises(ValueError, match="generation block"):
        processor.make_state(dict(rid="s", text=None, input_ids=[5] * 32740, sampling_params={}))
