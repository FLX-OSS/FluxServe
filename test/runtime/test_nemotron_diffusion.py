"""Nemotron block-diffusion decoder and runner contract tests.

These run on CPU against a recording stub model. What they pin down is the
block execution contract from the development guide Section 6.1 -- the exact
forward sequence, which forward owns the committed KV, where the seed comes
from, and the boundary policy for EOS and budgets -- none of which needs real
weights to be wrong.
"""

from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.execution.decoders.nemotron import (
    load_thinking_budget,
    NemotronThresholdDecoder,
    load_nemotron_decoder,
)
from fluxserve.backend.execution.runners.nemotron_diffusion import (
    NemotronBlockBudgetExceeded,
    NemotronDiffusionRunner,
)

MASK_ID = 7
EOS_ID = 3
VOCAB = 16
BLOCK = 4


def decoder(**overrides):
    values = dict(threshold=0.9, mask_id=MASK_ID, eos_ids=(EOS_ID,))
    values.update(overrides)
    return NemotronThresholdDecoder(**values)


def logits_for(rows, vocab=VOCAB):
    """Build logits whose softmax gives the requested per-position profile.

    Each row entry is ``(token_id, probability)``; the remaining mass is spread
    over the other tokens so the confidence is exactly the value asked for.
    """
    batch = torch.zeros(1, len(rows), vocab)
    for position, (token_id, probability) in enumerate(rows):
        others = (1.0 - probability) / (vocab - 1)
        batch[0, position] = torch.log(torch.full((vocab,), others))
        batch[0, position, token_id] = torch.log(torch.tensor(probability))
    return batch


# --------------------------------------------------------------------------
# Decoder selection policy
# --------------------------------------------------------------------------


def test_commits_every_position_at_or_above_the_threshold():
    block = torch.tensor([[MASK_ID, MASK_ID, MASK_ID, MASK_ID]])
    logits = logits_for([(1, 0.95), (2, 0.5), (4, 0.9), (5, 0.2)])
    predicted, transfer = decoder(threshold=0.9).select(logits, block)
    # 0.9 is inclusive; 0.5 and 0.2 are below and only survive via the
    # highest-confidence fallback, which position 0 already occupies.
    assert transfer.tolist() == [[True, False, True, False]]
    assert predicted[0, 0] == 1 and predicted[0, 2] == 4


def test_lowest_confidence_block_still_commits_exactly_one_position():
    block = torch.tensor([[MASK_ID, MASK_ID, MASK_ID]])
    logits = logits_for([(1, 0.2), (2, 0.5), (4, 0.3)])
    _, transfer = decoder(threshold=0.9).select(logits, block)
    assert int(transfer.sum()) == 1
    assert transfer[0, 1], "the highest-confidence position must be the one"


def test_exact_ties_commit_one_position_not_several():
    """LLaDA's decoder commits everything within 1e-5 of the max; not here."""
    block = torch.tensor([[MASK_ID, MASK_ID, MASK_ID]])
    logits = logits_for([(1, 0.4), (2, 0.4), (4, 0.4)])
    _, transfer = decoder(threshold=0.9).select(logits, block)
    assert int(transfer.sum()) == 1


def test_already_resolved_positions_are_never_rewritten():
    block = torch.tensor([[5, MASK_ID, 6]])
    logits = logits_for([(1, 0.99), (2, 0.99), (4, 0.99)])
    _, transfer = decoder(threshold=0.5).select(logits, block)
    assert transfer.tolist() == [[False, True, False]]

    updated = decoder(threshold=0.5).step(logits, block.clone())
    assert updated.tolist() == [[5, 2, 6]]


def test_mask_token_predictions_are_allowed_by_default():
    """The reference does not refuse them; LLaDA's `rm_mask=True` does."""
    block = torch.tensor([[MASK_ID, MASK_ID]])
    logits = logits_for([(MASK_ID, 0.99), (1, 0.95)])
    _, permissive = decoder(threshold=0.9).select(logits, block)
    assert permissive[0, 0], "reference policy commits the mask prediction"

    _, strict = decoder(threshold=0.9, allow_mask_prediction=False).select(
        logits, block
    )
    assert not strict[0, 0]
    assert strict[0, 1]


def test_confidence_is_computed_in_the_logits_dtype_by_default():
    block = torch.tensor([[MASK_ID]])
    logits = logits_for([(1, 0.95)]).to(torch.bfloat16)
    # Reference parity: no float32 upcast is inserted.
    assert decoder().confidence_dtype is None
    predicted, transfer = decoder(threshold=0.5).select(logits, block)
    assert transfer.tolist() == [[True]]
    assert int(predicted[0, 0]) == 1


def test_bfloat16_threshold_comparison_keeps_reference_scalar_rounding():
    block = torch.tensor([[MASK_ID, MASK_ID]])
    logits = logits_for([(1, 0.9), (2, 0.99)]).to(torch.bfloat16)
    confidence = logits.softmax(-1).amax(-1)
    assert confidence[0, 0].float() < 0.9
    assert bool((confidence >= 0.9).all())
    _, transfer = decoder(threshold=0.9).select(logits, block)
    assert transfer.tolist() == [[True, True]]


@pytest.mark.parametrize("temperature", [-0.1, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="finite non-negative"):
        decoder(temperature=temperature)


# --------------------------------------------------------------------------
# EOS policy
# --------------------------------------------------------------------------


def test_eos_behind_an_unresolved_mask_is_not_terminal():
    behind_mask = torch.tensor([[1, MASK_ID, EOS_ID, MASK_ID]])
    assert not bool(decoder().eos_is_terminal(behind_mask).all())

    resolved = torch.tensor([[1, 2, EOS_ID, MASK_ID]])
    assert bool(decoder().eos_is_terminal(resolved).all())


def test_block_without_eos_is_never_terminal():
    assert not bool(decoder().eos_is_terminal(torch.tensor([[1, 2, 4, 5]])).all())


def test_first_eos_index_reports_length_when_absent():
    assert int(decoder().first_eos_index(torch.tensor([[1, EOS_ID, 2]]))[0]) == 1
    assert int(decoder().first_eos_index(torch.tensor([[1, 2, 4]]))[0]) == 3


def test_decoder_factory_reads_runner_config_fields():
    built = load_nemotron_decoder(
        SimpleNamespace(threshold=0.5, mask_id=100, eos_id=11, eos_ids=(11,))
    )
    assert (built.threshold, built.mask_id, built.eos_ids) == (0.5, 100, (11,))


# --------------------------------------------------------------------------
# Runner block contract
# --------------------------------------------------------------------------


def classify_mask(attention_mask, query_len: int) -> str:
    """Name the attention pattern by rebuilding it, not by sampling it.

    A one-token causal mask is all-True, so `mask.all()` cannot tell the two
    patterns apart. Comparing against the exact expected mask also catches a
    mask that is neither.
    """
    if attention_mask is None:
        return "none"
    keys = attention_mask.shape[-1]
    prefix = keys - query_len
    expected_causal = (
        torch.arange(keys).unsqueeze(0)
        <= prefix + torch.arange(query_len).unsqueeze(1)
    )
    if torch.equal(attention_mask[0], expected_causal):
        return "causal"
    if bool(attention_mask.all()):
        return "bidirectional"
    return "other"


class RecordingModel:
    """A stub that records the attention pattern and cache use of each call.

    Present key/values are tagged with the call index so a test can prove which
    forward's KV ended up in the committed prefix.
    """

    def __init__(self, script, config):
        self.script = script
        self.model = SimpleNamespace(config=config)
        self.calls = []

    def __call__(self, *, input_ids, position_ids, past_key_values, use_cache,
                 attention_mask=None, **kwargs):
        index = len(self.calls)
        kind = classify_mask(attention_mask, input_ids.shape[1])
        self.calls.append({
            "kind": kind,
            "use_cache": bool(use_cache),
            "tokens": input_ids[0].tolist(),
            "positions": position_ids[0].tolist(),
            "cache_length": past_key_values.shape[4],
            "mask_shape": tuple(attention_mask.shape) if attention_mask is not None
            else None,
        })
        logits = self.script(index, input_ids)
        present = None
        if use_cache:
            length = past_key_values.shape[4]
            config = self.model.config
            present = [
                torch.full(
                    (1, config.num_key_value_heads, length, config.head_dim),
                    float(index + 1),
                )
                for _ in range(2 * config.num_hidden_layers)
            ]
        return SimpleNamespace(logits=logits, past_key_values=present)


def stub_config():
    return SimpleNamespace(
        num_hidden_layers=2,
        num_key_value_heads=2,
        head_dim=8,
        hidden_size=32,
        num_attention_heads=4,
        vocab_size=VOCAB,
    )


def make_runner(model, *, threshold=0.9, steps=0, early_stop=True, gen_length=BLOCK):
    runner = object.__new__(NemotronDiffusionRunner)
    runner.model = model
    runner.device = "cpu"
    runner.runner_config = SimpleNamespace(
        threshold=threshold,
        mask_id=MASK_ID,
        eos_id=EOS_ID,
        eos_ids=(EOS_ID,),
        block_length=BLOCK,
        steps=steps,
        gen_length=gen_length,
    )
    runner.init_decoder()
    runner.block_length = BLOCK
    runner.max_denoise_steps = steps if steps > 0 else BLOCK
    runner.num_forwards = 0
    runner.early_stop = early_stop
    runner.thinking_budget = load_thinking_budget(runner.runner_config)
    runner.last_stats = []
    return runner


def confident(tokens):
    """Logits that resolve every listed position with confidence ~1."""
    return logits_for([(token, 0.999) for token in tokens])


def test_full_block_costs_denoise_calls_plus_one_commit():
    """`D + 1`, and no extra diffusion forward just to notice completion."""

    def script(index, input_ids):
        if index == 0:  # prefill: seed the block with token 1
            return confident([9, 9, 1])
        # Every denoise call resolves the whole remaining block at once.
        return confident([2, 4, 5, 6])

    model = RecordingModel(script, stub_config())
    runner = make_runner(model)
    prompts = torch.tensor([[9, 9, 9]])
    runner.generate(prompts, prompt_lengths=[3], generation_lengths=[BLOCK])

    kinds = [(call["kind"], call["use_cache"]) for call in model.calls]
    assert kinds == [
        ("causal", True),          # prefill
        ("bidirectional", False),  # denoise
        ("causal", True),          # commit
    ]
    stats = runner.last_stats[0]
    assert (stats["denoise_calls"], stats["commit_calls"]) == (1, 1)
    assert stats["denoise_per_block"] == [1]
    assert stats["total_calls"] == stats["denoise_calls"] + 2


def test_multi_step_block_still_commits_exactly_once():
    def script(index, input_ids):
        if index == 0:
            return confident([9, 9, 1])
        # One position per denoising call: everything below threshold, so only
        # the highest-confidence position is committed each time.
        return logits_for([(2, 0.4), (4, 0.3), (5, 0.2), (6, 0.1)])

    model = RecordingModel(script, stub_config())
    runner = make_runner(model, threshold=0.9)
    runner.generate(torch.tensor([[9, 9, 9]]), prompt_lengths=[3],
                    generation_lengths=[BLOCK])

    stats = runner.last_stats[0]
    # Position 0 is seeded, so three masked positions remain, one per call.
    assert stats["denoise_calls"] == 3
    assert stats["commit_calls"] == 1
    assert [call["kind"] for call in model.calls] == [
        "causal", "bidirectional", "bidirectional", "bidirectional", "causal",
    ]


def test_block_is_seeded_from_the_previous_causal_forward():
    def script(index, input_ids):
        if index == 0:
            return confident([9, 1])  # prefill's last logit predicts token 1
        return confident([2, 4, 5, 6])

    model = RecordingModel(script, stub_config())
    runner = make_runner(model)
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[BLOCK])

    denoise = model.calls[1]
    assert denoise["tokens"][0] == 1, "position 0 carries the seed"
    assert denoise["tokens"][1:] == [MASK_ID] * (BLOCK - 1)
    assert denoise["positions"] == [2, 3, 4, 5], "absolute positions"


def test_committed_prefix_holds_the_causal_forwards_keys():
    """G1 at the runner level: the denoise forward cannot reach the cache."""
    captured = {}

    def script(index, input_ids):
        if index == 0:
            return confident([9, 1])
        return confident([2, 4, 5, 6])

    model = RecordingModel(script, stub_config())
    runner = make_runner(model)

    original = runner._commit_kv

    def spy(cache, present, start, end):
        original(cache, present, start, end)
        captured.setdefault("writes", []).append((start, end, float(present[0][0, 0, 0, 0])))

    runner._commit_kv = spy
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[BLOCK])

    # Call 0 is the prefill (tag 1.0), call 2 is the commit (tag 3.0). The
    # denoise forward at call 1 never produces a cache write at all.
    assert captured["writes"] == [(0, 2, 1.0), (2, 6, 3.0)]
    assert all(
        not call["use_cache"]
        for call in model.calls
        if call["kind"] == "bidirectional"
    )


def test_denoise_sees_the_committed_prefix_plus_its_own_block():
    def script(index, input_ids):
        if index == 0:
            return confident([9, 9, 9, 1])
        return confident([2, 4, 5, 6])

    model = RecordingModel(script, stub_config())
    runner = make_runner(model)
    runner.generate(torch.tensor([[9, 9, 9, 9]]), prompt_lengths=[4],
                    generation_lengths=[BLOCK])

    prefill, denoise, commit = model.calls
    assert prefill["cache_length"] == 4
    assert denoise["cache_length"] == 4 + BLOCK
    assert denoise["mask_shape"] == (1, BLOCK, 4 + BLOCK)
    assert commit["cache_length"] == 4 + BLOCK
    assert commit["mask_shape"] == (1, BLOCK, 4 + BLOCK)


def test_exhausted_denoise_budget_fails_loudly():
    def script(index, input_ids):
        if index == 0:
            return confident([9, 1])
        # Always predicts the mask token, so nothing ever resolves.
        return logits_for([(MASK_ID, 0.99)] * BLOCK)

    model = RecordingModel(script, stub_config())
    runner = make_runner(model, steps=2)
    with pytest.raises(NemotronBlockBudgetExceeded, match="masked position"):
        runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                        generation_lengths=[BLOCK])


def test_eos_stops_generation_after_committing_its_block():
    def script(index, input_ids):
        if index == 0:
            return confident([9, 1])
        return confident([2, EOS_ID, 5, 6])

    model = RecordingModel(script, stub_config())
    runner = make_runner(model)
    output = runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                             generation_lengths=[3 * BLOCK])

    stats = runner.last_stats[0]
    assert stats["blocks"] == 1, "later blocks are not started"
    assert stats["commit_calls"] == 1, "the terminal block is still committed"
    generated = output[0, 2:].tolist()
    assert generated[:2] == [1, EOS_ID]
    assert set(generated[2:]) == {MASK_ID}, "nothing past EOS is published"


def test_budget_not_divisible_by_block_length_runs_whole_blocks():
    def script(index, input_ids):
        if index == 0:
            return confident([9, 1])
        return confident([2, 4, 5, 6])

    model = RecordingModel(script, stub_config())
    runner = make_runner(model)
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[BLOCK + 1])

    stats = runner.last_stats[0]
    assert stats["blocks"] == 2, "a partial budget still runs whole blocks"
    assert stats["internal_tokens"] == 2 * BLOCK
    assert stats["returned_tokens"] == BLOCK + 1, "extra work is not returned"


def test_zero_budget_returns_without_any_forward():
    model = RecordingModel(lambda index, ids: confident([1]), stub_config())
    runner = make_runner(model)
    output = runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                             generation_lengths=[0])
    assert model.calls == []
    assert runner.last_stats[0]["total_calls"] == 0
    assert output.shape[1] == 2


def test_mask_classifier_distinguishes_a_single_token_causal_prefill():
    """Guard the guard: `mask.all()` would call this bidirectional."""
    single = torch.ones(1, 1, 1, dtype=torch.bool)
    assert classify_mask(single, 1) == "causal"
    assert classify_mask(torch.ones(1, BLOCK, BLOCK, dtype=torch.bool), BLOCK) == (
        "bidirectional"
    )
    causal = NemotronDiffusionRunner._causal_mask(BLOCK, 5, "cpu")
    assert classify_mask(causal, BLOCK) == "causal"
    bidirectional = NemotronDiffusionRunner._bidirectional_mask(BLOCK, 5, "cpu")
    assert classify_mask(bidirectional, BLOCK) == "bidirectional"
