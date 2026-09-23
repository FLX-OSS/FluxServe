"""EOS policy must be request-local across every Nemotron decoding path."""

import pytest
import torch

from test_nemotron_diffusion import (
    BLOCK, EOS_ID, MASK_ID, RecordingModel, confident, make_runner, stub_config,
)
from test_nemotron_selfspec import make_selfspec_runner
import test_nemotron_fa4 as paged
import test_nemotron_selfspec_paged as spec
from fluxserve.backend.execution.runners.nemotron import get_nemotron_runner


@pytest.mark.parametrize("backend", ["sdpa", "fa4", "flashinfer"])
@pytest.mark.parametrize("mode", ["threshold", "self_speculation"])
@pytest.mark.parametrize("seed", [1, EOS_ID])
def test_mixed_batch_preserves_tokens_after_eos_only_when_requested(
    backend, mode, seed, monkeypatch,
):
    def script(index, tokens):
        if tokens.shape[1] == 2:
            return confident([9, seed])
        return confident([EOS_ID] * BLOCK)

    if backend == "sdpa":
        runner = (make_runner(RecordingModel(script, stub_config()))
                  if mode == "threshold" else make_selfspec_runner(script))
    else:
        module = paged if mode == "threshold" else spec
        monkeypatch.setattr(module, "_paged_runner_cls", get_nemotron_runner(backend, mode))

        def paged_script(index, seq_ids, tokens, length):
            return script(index, tokens).expand(len(seq_ids), -1, -1).clone()

        runner = (module.make_paged_runner(paged_script) if mode == "threshold"
                  else module.make_runner(paged_script))
    if backend == "sdpa" and mode == "self_speculation":
        # The dense self-speculation API accepts one request per call.
        output = torch.cat([
            runner.generate(
                torch.tensor([[9, 9]]), prompt_lengths=[2],
                generation_lengths=[BLOCK], sampling_params={"ignore_eos": ignore},
            ) for ignore in (False, True)
        ])[:, 2:]
    else:
        output = runner.generate(
            torch.tensor([[9, 9], [9, 9]]), prompt_lengths=[2, 2],
            generation_lengths=[BLOCK, BLOCK],
            sampling_params=[{"ignore_eos": False}, {"ignore_eos": True}],
        )[:, 2:]
    stopped = [seed] if seed == EOS_ID else [seed, EOS_ID]
    assert output[0].tolist() == stopped + [MASK_ID] * (BLOCK - len(stopped))
    assert output[1].tolist() == [seed] + [EOS_ID] * (BLOCK - 1)
    assert runner.early_stop is True
    assert runner.last_stats[-1]["returned_tokens"] == BLOCK


@pytest.mark.parametrize("backend", ["fa4", "flashinfer"])
@pytest.mark.parametrize("mode", ["threshold", "self_speculation"])
def test_online_eos_seed_obeys_each_requests_policy(backend, mode, monkeypatch):
    module = paged if mode == "threshold" else spec
    monkeypatch.setattr(module, "_paged_runner_cls", get_nemotron_runner(backend, mode))

    def script(index, seq_ids, tokens, length):
        return paged.logits_for([(2, 0.999)] * length, batch=len(seq_ids))

    runner = module.make_plan_runner(script)
    states = {
        rid: paged.FakeState(rid, [5] * 34, max_new_tokens=BLOCK, ignore_eos=ignore)
        for rid, ignore in [("stop", False), ("continue", True)]
    }
    runner._paged_request_slots.update({"stop": 0, "continue": 1})
    runner._request_seeds.update({rid: EOS_ID for rid in states})
    if mode == "self_speculation":
        runner._request_prefix.update({rid: 34 for rid in states})
    op = paged.FakePlanOp(list(states), [], [], [], [[0, 1, 2], [3, 4, 5]], 0)
    results = {item.rid: item for item in module.run_plan(runner, op, states)}
    assert results["stop"].token_ids == []
    assert results["stop"].finish_reason == "stop"
    assert results["continue"].token_ids == [EOS_ID, 2, 2, 2]
    assert results["continue"].finish_reason == "length"


@pytest.mark.parametrize("backend", ["fa4", "flashinfer"])
def test_online_selfspec_one_token_budget_returns_prefill_seed(backend, monkeypatch):
    monkeypatch.setattr(spec, "_paged_runner_cls", get_nemotron_runner(backend, "self_speculation"))

    def unexpected_forward(*args):
        pytest.fail("a one-token budget needs only the prefill seed")

    runner = spec.make_plan_runner(unexpected_forward)
    states = {"r0": paged.FakeState("r0", [5] * 34, max_new_tokens=1)}
    runner._paged_request_slots["r0"] = 0
    runner._request_seeds["r0"] = 6
    runner._request_prefix["r0"] = 34
    op = paged.FakePlanOp(["r0"], [], [], [], [[0, 1, 2]], 0)
    result, = spec.run_plan(runner, op, states)
    assert result.token_ids == [6]
    assert result.finished and result.finish_reason == "length"
