"""Linear self-speculation: acceptance, rollback and the draft adapter.

The algorithm is exercised on CPU against a recording stub model, plus a real
tiny model for the fused LoRA weights. What matters here is that emitted tokens
come from the verifier, that the committed prefix advances by the accepted
length rather than a whole block, and that the draft adapter is off for every
verifying forward.
"""

from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.execution.decoders.nemotron import (
    load_thinking_budget,
)

from fluxserve.backend.execution.runners.nemotron_selfspec import (
    NemotronSelfSpecRunner,
    SelfSpecStats,
)
from test_nemotron_diffusion import (
    BLOCK,
    EOS_ID,
    MASK_ID,
    VOCAB,
    RecordingModel,
    classify_mask,
    logits_for,
    stub_config,
)

# --------------------------------------------------------------------------
# Acceptance rule
# --------------------------------------------------------------------------


def accepted(verified, drafted):
    return NemotronSelfSpecRunner.accepted_length(
        torch.tensor(verified), torch.tensor(drafted)
    )


def test_a_total_mismatch_still_accepts_the_verifier_bonus_token():
    # verified[0] is a correct autoregressive step whatever the draft guessed.
    assert accepted([9, 9, 9, 9], [1, 2, 3, 4]) == 1


def test_matching_prefix_lengthens_acceptance():
    # verified[i] is compared against drafted[i + 1].
    assert accepted([2, 9, 9, 9], [1, 2, 3, 4]) == 2
    assert accepted([2, 3, 9, 9], [1, 2, 3, 4]) == 3


def test_a_fully_correct_draft_accepts_the_whole_block():
    assert accepted([2, 3, 4, 9], [1, 2, 3, 4]) == BLOCK


def test_acceptance_never_exceeds_the_block():
    assert accepted([2, 3, 4, 5], [1, 2, 3, 4]) == BLOCK


# --------------------------------------------------------------------------
# Draft selection
# --------------------------------------------------------------------------


def make_selfspec_runner(script, *, draft_threshold=0.0, gen_length=BLOCK):
    runner = object.__new__(NemotronSelfSpecRunner)
    runner.model = RecordingModel(script, stub_config())
    runner.device = "cpu"
    runner.runner_config = SimpleNamespace(
        threshold=0.9, mask_id=MASK_ID, eos_id=EOS_ID, eos_ids=(EOS_ID,),
        block_length=BLOCK, steps=0, gen_length=gen_length,
    )
    runner.init_decoder()
    runner.block_length = BLOCK
    runner.max_denoise_steps = BLOCK
    runner.num_forwards = 0
    runner.early_stop = True
    runner.last_stats = []
    runner.draft_threshold = draft_threshold
    runner.draft_decoder = draft_decoder(draft_threshold)
    runner.lora = None
    runner.thinking_budget = load_thinking_budget(runner.runner_config)
    runner.server_args = SimpleNamespace()
    return runner


def draft_decoder(threshold):
    from fluxserve.backend.execution.decoders.nemotron import (
        NemotronThresholdDecoder,
    )

    return NemotronThresholdDecoder(
        threshold=threshold, mask_id=MASK_ID, eos_ids=(EOS_ID,)
    )


def reference_draft_rule(logits, block, threshold):
    """The inline selection inside `linear_spec_generate`, transcribed."""
    is_mask = block == MASK_ID
    if threshold <= 0:
        return is_mask.clone()
    tokens = logits.argmax(-1)
    probabilities = torch.softmax(logits, -1)
    confidence = probabilities.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    confidence = torch.where(
        is_mask, confidence, torch.full_like(confidence, -float("inf"))
    )
    unmask = confidence >= threshold
    if not bool(unmask.any()):
        best = confidence.view(-1).argmax()
        unmask = torch.zeros_like(is_mask)
        unmask.view(-1)[best] = True
    return unmask


def test_zero_threshold_fills_every_masked_position_in_one_forward():
    block = torch.tensor([[1, MASK_ID, MASK_ID, MASK_ID]])
    logits = logits_for([(9, 0.2), (10, 0.2), (11, 0.2), (12, 0.2)])
    updated = draft_decoder(0.0).step(logits, block.clone())
    assert updated.tolist() == [[1, 10, 11, 12]]


def test_the_draft_rule_and_the_diffusion_decoder_are_the_same_function():
    """Checked exhaustively rather than argued.

    The reference adds the argmax only when nothing clears the threshold; the
    decoder adds it unconditionally. Those agree because the argmax is the
    maximum confidence, so it clears whenever anything does. This pins that
    down over a grid, so a future change to either rule shows up here.
    """
    import itertools

    grid = [0.05, 0.2, 0.5, 0.7, 0.95]
    compared = 0
    for threshold in (0.0, 0.3, 0.5, 0.9):
        decoder = draft_decoder(threshold)
        for combination in itertools.product(grid, repeat=3):
            for seeded in (True, False):
                first = 1 if seeded else MASK_ID
                block = torch.tensor([[first, MASK_ID, MASK_ID, MASK_ID]])
                logits = logits_for(
                    [(9, 0.9)]
                    + [(10 + i, p) for i, p in enumerate(combination)]
                )
                _, produced = decoder.select(logits, block)
                expected = reference_draft_rule(logits, block, threshold)
                assert torch.equal(produced, expected), (
                    threshold, combination, seeded
                )
                compared += 1
    assert compared == 1000


def test_a_lone_confident_position_does_not_drag_in_a_weak_one():
    block = torch.tensor([[1, MASK_ID, MASK_ID, MASK_ID]])
    logits = logits_for([(9, 0.9), (10, 0.2), (11, 0.1), (12, 0.7)])
    _, transfer = draft_decoder(0.5).select(logits, block)
    assert transfer.tolist() == [[False, False, False, True]]


def test_when_nothing_clears_exactly_one_position_is_drafted():
    block = torch.tensor([[1, MASK_ID, MASK_ID, MASK_ID]])
    logits = logits_for([(9, 0.9), (10, 0.2), (11, 0.1), (12, 0.3)])
    _, transfer = draft_decoder(0.5).select(logits, block)
    assert int(transfer.sum()) == 1
    assert transfer[0, 3], "the highest-confidence masked position"


# --------------------------------------------------------------------------
# The iteration
# --------------------------------------------------------------------------


def test_one_iteration_is_prefill_then_draft_then_verify():
    def script(index, input_ids):
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)])
        if index == 1:  # draft fills the block
            return logits_for([(9, 0.9), (10, 0.9), (11, 0.9), (12, 0.9)])
        # verify agrees with the draft everywhere, plus a bonus token
        return logits_for([(10, 0.9), (11, 0.9), (12, 0.9), (13, 0.9)])

    runner = make_selfspec_runner(script, gen_length=BLOCK)
    output = runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                             generation_lengths=[BLOCK])

    kinds = [(call["kind"], call["use_cache"]) for call in runner.model.calls]
    assert kinds == [
        ("causal", True),          # prefill
        ("bidirectional", False),  # draft
        ("causal", True),          # verify
    ]
    stats = runner.last_stats[0]
    assert stats["draft_calls"] == 1 and stats["verify_calls"] == 1
    assert stats["accepted_per_iteration"] == [BLOCK]
    # The seed plus the whole accepted block, truncated to the budget.
    assert output[0, 2:].tolist()[:BLOCK] == [1, 10, 11, 12]


def test_emitted_tokens_come_from_the_verifier_not_the_draft():
    def script(index, input_ids):
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)])
        if index == 1:
            return logits_for([(9, 0.9), (10, 0.9), (14, 0.9), (14, 0.9)])
        # The verifier disagrees from the second position onwards.
        return logits_for([(10, 0.9), (13, 0.9), (13, 0.9), (13, 0.9)])

    runner = make_selfspec_runner(script, gen_length=2 * BLOCK)
    output = runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                             generation_lengths=[2 * BLOCK])

    stats = runner.last_stats[0]
    # draft[1] == 2 matches verified[0]; draft[2] == 7 != verified[1] == 6.
    assert stats["accepted_per_iteration"][0] == 2
    generated = output[0, 2:].tolist()
    assert generated[:3] == [1, 10, 13], "verifier tokens, never the draft's 14"


def test_the_committed_prefix_advances_by_the_accepted_length():
    """Rollback: a rejected tail must not stay in the prefix."""
    def script(index, input_ids):
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)])
        if index % 2 == 1:
            return logits_for([(9, 0.9), (10, 0.9), (14, 0.9), (14, 0.9)])
        return logits_for([(10, 0.9), (13, 0.9), (13, 0.9), (13, 0.9)])

    runner = make_selfspec_runner(script, gen_length=2 * BLOCK)
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[2 * BLOCK])

    calls = runner.model.calls
    # Prompt is 2 tokens; the first draft sits at positions 2..5.
    assert calls[1]["positions"][0] == 2
    # Two tokens were accepted, so the next iteration starts at 4, not 2+4.
    assert calls[3]["positions"][0] == 4
    assert runner.last_stats[0]["accepted_per_iteration"][:2] == [2, 2]


def test_eos_inside_the_accepted_tokens_stops_generation():
    def script(index, input_ids):
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)])
        if index == 1:
            return logits_for([(9, 0.9), (10, 0.9), (11, 0.9), (12, 0.9)])
        return logits_for([(10, 0.9), (EOS_ID, 0.9), (12, 0.9), (13, 0.9)])

    runner = make_selfspec_runner(script, gen_length=4 * BLOCK)
    output = runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                             generation_lengths=[4 * BLOCK])

    assert runner.last_stats[0]["iterations"] == 1
    generated = [t for t in output[0, 2:].tolist() if t != MASK_ID]
    assert generated == [1, 10, EOS_ID]


def test_batching_is_refused_rather_than_silently_serialized():
    runner = make_selfspec_runner(lambda *a: None)
    with pytest.raises(ValueError, match="one request at a time"):
        runner.generate(torch.tensor([[9, 9], [9, 9]]), prompt_lengths=[2, 2],
                        generation_lengths=[BLOCK, BLOCK])


def test_stats_report_acceptance_and_tokens_per_forward():
    stats = SelfSpecStats(
        prefill_calls=1, draft_calls=2, verify_calls=2, iterations=2,
        accepted_per_iteration=[4, 2], returned_tokens=7,
    )
    assert stats.total_calls == 5
    assert stats.mean_acceptance == pytest.approx(3.0)
    assert stats.as_dict()["tokens_per_forward"] == pytest.approx(1.4)
    assert SelfSpecStats().mean_acceptance == 0.0


# --------------------------------------------------------------------------
# Draft adapter
# --------------------------------------------------------------------------


def test_fused_adapter_swaps_o_proj_and_restores_it(tmp_path):
    from safetensors.torch import save_file

    from fluxserve.backend.model_loader.nemotron import load_nemotron_lora
    from test_nemotron_model import initialized_tiny_model, tiny_config

    config = tiny_config()
    model = initialized_tiny_model()
    rank, alpha = 4, 8.0
    tensors, expected = {}, []
    torch.manual_seed(1)
    for index, block in enumerate(model.model.layers):
        in_features = int(block.self_attn.o_proj.weight.shape[1])
        out_features = int(block.self_attn.o_proj.weight.shape[0])
        lora_a = torch.randn(rank, in_features)
        lora_b = torch.randn(out_features, rank)
        prefix = f"base_model.model.encoder.layers.{index}.self_attn.o_proj"
        tensors[f"{prefix}.lora_A.weight"] = lora_a
        tensors[f"{prefix}.lora_B.weight"] = lora_b
        expected.append((alpha / rank) * (lora_b @ lora_a))
    path = tmp_path / "adapter_model.safetensors"
    save_file(tensors, str(path))
    (tmp_path / "adapter_config.json").write_text(
        '{"r": 4, "lora_alpha": 8.0, "target_modules": ["o_proj"]}'
    )

    base_weights = [
        block.self_attn.o_proj.weight.data.clone() for block in model.model.layers
    ]
    adapter = load_nemotron_lora(model, config, path)
    assert adapter is not None and len(adapter.layers) == config.num_hidden_layers

    adapter.apply(True)
    for index, block in enumerate(model.model.layers):
        delta = block.self_attn.o_proj.weight.data - base_weights[index]
        assert torch.allclose(delta, expected[index], atol=1e-4)

    adapter.apply(False)
    for index, block in enumerate(model.model.layers):
        assert torch.equal(block.self_attn.o_proj.weight.data, base_weights[index])


def test_a_missing_adapter_is_a_configuration_not_an_error(tmp_path):
    from fluxserve.backend.model_loader.nemotron import load_nemotron_lora
    from test_nemotron_model import initialized_tiny_model, tiny_config

    assert load_nemotron_lora(
        initialized_tiny_model(), tiny_config(), tmp_path / "absent.safetensors"
    ) is None


def test_an_adapter_targeting_other_modules_is_refused(tmp_path):
    from safetensors.torch import save_file

    from fluxserve.backend.model_loader.nemotron import load_nemotron_lora
    from test_nemotron_model import initialized_tiny_model, tiny_config

    path = tmp_path / "adapter_model.safetensors"
    save_file({"x": torch.zeros(1)}, str(path))
    (tmp_path / "adapter_config.json").write_text(
        '{"r": 4, "lora_alpha": 8.0, "target_modules": ["q_proj", "o_proj"]}'
    )
    with pytest.raises(ValueError, match="o_proj only"):
        load_nemotron_lora(initialized_tiny_model(), tiny_config(), path)


def test_the_adapter_is_off_for_every_verifying_forward():
    """The adapter specialises drafting; AR semantics must stay unmodified."""
    seen = []

    def script(index, input_ids):
        seen.append(runner.lora.enabled)
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)])
        if index == 1:
            return logits_for([(9, 0.9), (10, 0.9), (11, 0.9), (12, 0.9)])
        return logits_for([(10, 0.9), (11, 0.9), (12, 0.9), (13, 0.9)])

    runner = make_selfspec_runner(script, gen_length=BLOCK)

    class FakeAdapter:
        enabled = False

        def apply(self, enabled):
            self.enabled = bool(enabled)

    runner.lora = FakeAdapter()
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[BLOCK])

    assert seen == [False, True, False], "prefill off, draft on, verify off"
    assert runner.lora.enabled is False, "left disabled afterwards"
