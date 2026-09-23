"""Thinking budget: force an end-of-thinking marker when the allowance is spent.

The reference enforces this differently per mode -- block diffusion injects the
marker into the block that carries the budget past its limit, self-speculation
forces the next seed -- so the policy is tested on its own and then at each
runner's boundary. The arithmetic is the part most likely to be wrong by one.
"""

from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.execution.decoders.nemotron import (
    ThinkingBudget,
    load_thinking_budget,
)
from test_nemotron_diffusion import (
    BLOCK,
    EOS_ID,
    MASK_ID,
    RecordingModel,
    logits_for,
    make_runner,
    stub_config,
)

END_THINK = 13


def budget(limit=None, marker=END_THINK):
    return ThinkingBudget(max_thinking_tokens=limit, end_think_token_id=marker)


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def test_disabled_unless_both_settings_are_given():
    assert not ThinkingBudget().enabled
    with pytest.raises(ValueError, match="together or not at all"):
        ThinkingBudget(max_thinking_tokens=8)
    with pytest.raises(ValueError, match="together or not at all"):
        ThinkingBudget(end_think_token_id=END_THINK)
    with pytest.raises(ValueError, match="non-negative"):
        ThinkingBudget(max_thinking_tokens=-1, end_think_token_id=END_THINK)


def test_a_disabled_budget_forces_nothing():
    disabled = ThinkingBudget()
    assert disabled.satisfied([]) is True
    assert disabled.block_injection_offset(0, BLOCK) is None
    assert disabled.force_next_seed(10**6) is False


def test_the_marker_lands_in_the_block_that_crosses_the_limit():
    # Limit 40, blocks of 32: the first block ends at 32, still inside.
    assert budget(40).block_injection_offset(0, 32) is None
    # The second covers 32..63 and crosses at 40, which is offset 8.
    assert budget(40).block_injection_offset(32, 32) == 8


def test_a_limit_on_a_block_boundary_injects_at_the_next_block_start():
    # Limit 32 with blocks of 32: the first block ends exactly at the limit,
    # so it is not the one that crosses.
    assert budget(32).block_injection_offset(0, 32) is None
    assert budget(32).block_injection_offset(32, 32) == 0


def test_a_spent_budget_overwrites_the_seed():
    """Offset zero is the seed position; the reference overwrites it too."""
    assert budget(8).block_injection_offset(32, 32) == 0
    assert budget(0).block_injection_offset(0, 32) == 0


def test_the_offset_always_lands_inside_the_block():
    for limit in range(0, 200):
        for tokens_before in range(0, 200, 32):
            offset = budget(limit).block_injection_offset(tokens_before, 32)
            if offset is not None:
                assert 0 <= offset < 32, (limit, tokens_before, offset)


def test_a_budget_larger_than_the_output_never_fires():
    assert budget(10**6).block_injection_offset(0, 32) is None
    assert budget(10**6).force_next_seed(100) is False


def test_an_existing_marker_satisfies_the_budget():
    assert budget(4).satisfied([1, 2, END_THINK, 3])
    assert not budget(4).satisfied([1, 2, 3])
    assert not budget(4).satisfied([])
    assert budget(4).satisfied(torch.tensor([[1, END_THINK]]))
    assert not budget(4).satisfied(torch.tensor([[1, 2]]))
    assert not budget(4).satisfied(torch.zeros(1, 0, dtype=torch.long))


def test_the_next_seed_is_forced_strictly_past_the_limit():
    """Self-speculation's rule: `total_generated > limit`, not `>=`."""
    assert budget(8).force_next_seed(8) is False
    assert budget(8).force_next_seed(9) is True


def test_the_factory_reads_runner_config_fields():
    assert not load_thinking_budget(SimpleNamespace()).enabled
    built = load_thinking_budget(
        SimpleNamespace(max_thinking_tokens=16, end_think_token_id=END_THINK)
    )
    assert built.enabled
    assert (built.max_thinking_tokens, built.end_think_token_id) == (16, END_THINK)


# --------------------------------------------------------------------------
# Diffusion runner
# --------------------------------------------------------------------------


def diffusion_runner(script, *, limit, marker=END_THINK):
    runner = make_runner(RecordingModel(script, stub_config()), gen_length=4 * BLOCK)
    runner.runner_config.max_thinking_tokens = limit
    runner.runner_config.end_think_token_id = marker
    runner.thinking_budget = load_thinking_budget(runner.runner_config)
    return runner


def resolving_script(index, input_ids):
    if index == 0:
        return logits_for([(9, 0.99), (1, 0.99)])
    return logits_for([(10, 0.9), (11, 0.9), (12, 0.9), (14, 0.9)])


def test_diffusion_injects_the_marker_before_denoising():
    # Limit 6 with blocks of 4: block 0 covers 0..3, block 1 covers 4..7 and
    # crosses at 6, which is offset 2.
    runner = diffusion_runner(resolving_script, limit=6)
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[2 * BLOCK])

    blocks = [
        call["tokens"] for call in runner.model.calls
        if call["kind"] == "bidirectional"
    ]
    # The first denoise of block 1 already carries the marker, so the decoder
    # sees a resolved position rather than writing over it.
    assert blocks[1][2] == END_THINK
    assert blocks[0][2] != END_THINK, "block 0 is inside the allowance"


def test_diffusion_leaves_the_block_alone_when_the_model_already_stopped():
    def script(index, input_ids):
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)])
        # The model emits the marker on its own in the first block. It has to
        # land on a *masked* position: offset 0 is the seed, which the decoder
        # never writes, so a prediction there would be discarded.
        return logits_for([(9, 0.9), (END_THINK, 0.9), (12, 0.9), (14, 0.9)])

    runner = diffusion_runner(script, limit=6)
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[2 * BLOCK])
    blocks = [
        call["tokens"] for call in runner.model.calls
        if call["kind"] == "bidirectional"
    ]
    # Nothing is forced into block 1: offset 2 holds a mask on entry.
    assert blocks[1][2] == MASK_ID


def test_diffusion_is_unchanged_when_the_budget_is_off():
    with_budget = diffusion_runner(resolving_script, limit=None, marker=None)
    assert not with_budget.thinking_budget.enabled
    with_budget.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                         generation_lengths=[2 * BLOCK])
    blocks = [
        call["tokens"] for call in with_budget.model.calls
        if call["kind"] == "bidirectional"
    ]
    assert all(END_THINK not in tokens for tokens in blocks)


def test_a_spent_budget_replaces_the_seed_of_the_next_block():
    runner = diffusion_runner(resolving_script, limit=0)
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[BLOCK])
    first_denoise = next(
        call for call in runner.model.calls if call["kind"] == "bidirectional"
    )
    assert first_denoise["tokens"][0] == END_THINK, "offset 0 is the seed"


# --------------------------------------------------------------------------
# Self-speculation runner
# --------------------------------------------------------------------------


def selfspec_runner(script, *, limit, marker=END_THINK):
    from test_nemotron_selfspec import make_selfspec_runner

    runner = make_selfspec_runner(script, gen_length=4 * BLOCK)
    runner.runner_config.max_thinking_tokens = limit
    runner.runner_config.end_think_token_id = marker
    runner.thinking_budget = load_thinking_budget(runner.runner_config)
    return runner


def spec_script(index, input_ids):
    if index == 0:
        return logits_for([(9, 0.99), (1, 0.99)])
    if index % 2 == 1:
        return logits_for([(9, 0.9), (10, 0.9), (11, 0.9), (12, 0.9)])
    return logits_for([(10, 0.9), (11, 0.9), (12, 0.9), (14, 0.9)])


def test_self_speculation_forces_the_next_seed():
    # Each iteration accepts the whole block, so after one iteration five
    # tokens exist (the prefill seed plus four) and 5 > 4.
    runner = selfspec_runner(spec_script, limit=4)
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[3 * BLOCK])

    drafts = [
        call["tokens"] for call in runner.model.calls
        if call["kind"] == "bidirectional"
    ]
    assert drafts[0][0] != END_THINK, "the first block is inside the allowance"
    assert drafts[1][0] == END_THINK, "the second block's seed is forced"


def test_self_speculation_leaves_the_seed_alone_within_the_allowance():
    runner = selfspec_runner(spec_script, limit=10**6)
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[2 * BLOCK])
    drafts = [
        call["tokens"] for call in runner.model.calls
        if call["kind"] == "bidirectional"
    ]
    assert all(tokens[0] != END_THINK for tokens in drafts)


# --------------------------------------------------------------------------
# Configuration surface
# --------------------------------------------------------------------------


def test_the_end_think_id_resolves_from_the_checkpoint():
    from fluxserve.backend.model_loader.nemotron import resolve_end_think_token_id
    from test_nemotron_model import checkpoint_config

    resolved = resolve_end_think_token_id(checkpoint_config())
    if resolved is None:
        pytest.skip("Nemotron tokenizer_config.json is not in the local cache")
    assert resolved == END_THINK


def test_normalization_resolves_and_validates_the_budget():
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args
    from test_nemotron_model import checkpoint_config, serve_args

    args = serve_args(max_thinking_tokens=64)
    if normalize_nemotron_args(args, checkpoint_config()) and args.end_think_token_id:
        assert args.end_think_token_id == END_THINK

    explicit = serve_args(max_thinking_tokens=64, end_think_token_id=12)
    normalize_nemotron_args(explicit, checkpoint_config())
    assert explicit.end_think_token_id == 12, "an explicit id wins"

    with pytest.raises(ValueError, match="non-negative"):
        normalize_nemotron_args(
            serve_args(max_thinking_tokens=-1), checkpoint_config()
        )
    with pytest.raises(ValueError, match="outside vocab_size"):
        normalize_nemotron_args(
            serve_args(max_thinking_tokens=8, end_think_token_id=999999),
            checkpoint_config(),
        )
    with pytest.raises(ValueError, match="no effect without"):
        normalize_nemotron_args(
            serve_args(end_think_token_id=END_THINK), checkpoint_config()
        )


def test_the_settings_reach_the_runner_config():
    from fluxserve.backend.execution.forward_batch_info import RunnerConfig
    from fluxserve.backend.model_loader.nemotron import (
        apply_nemotron_runner_config,
    )
    from test_nemotron_model import checkpoint_config

    config = RunnerConfig()
    apply_nemotron_runner_config(
        config,
        checkpoint_config(),
        SimpleNamespace(max_thinking_tokens=64, end_think_token_id=END_THINK),
    )
    assert load_thinking_budget(config).enabled

    plain = RunnerConfig()
    apply_nemotron_runner_config(plain, checkpoint_config())
    assert not load_thinking_budget(plain).enabled, (
        "a config built without the flags must behave exactly as before"
    )


def test_a_prediction_at_the_seed_position_is_discarded():
    """Worth stating: offset 0 holds a resolved token, so it is never written.

    A budget test that puts the marker at position 0 of the denoising logits
    measures nothing, because the decoder only writes masked positions.
    """
    runner = diffusion_runner(
        lambda index, input_ids: (
            logits_for([(9, 0.99), (1, 0.99)])
            if index == 0
            else logits_for([(END_THINK, 0.9), (11, 0.9), (12, 0.9), (14, 0.9)])
        ),
        limit=10**6,
    )
    output = runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                             generation_lengths=[BLOCK])
    generated = output[0, 2:].tolist()[:BLOCK]
    assert generated[0] == 1, "the seed survives the model's own prediction"
    assert END_THINK not in generated
