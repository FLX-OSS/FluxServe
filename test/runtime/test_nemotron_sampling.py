"""Sampling parity, request ownership, and serving argument propagation."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluxserve.backend.execution.decoders.nemotron import NemotronThresholdDecoder
from fluxserve.backend.execution.nemotron_sampling import (
    NemotronSampling, make_sampling, validate_sampling_params,
)


def reference_helpers():
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            "nvidia/Nemotron-Labs-Diffusion-14B", "modeling_nemotron_labs_diffusion.py",
            revision="f8c3e2c078e193599b8882d965b1001c456ba738", local_files_only=True,
        )
    except Exception:
        pytest.skip("pinned checkpoint reference source is not cached")
    nodes = [node for node in ast.parse(Path(path).read_text()).body
             if isinstance(node, ast.FunctionDef)
             and node.name in ("_add_gumbel_noise", "_get_transfer_index")]
    scope = {"torch": torch, "F": torch.nn.functional, "np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), scope)
    return scope


@pytest.mark.parametrize("temperature", [0.2, 0.7, 1.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_diffusion_matches_pinned_reference_with_identical_noise(temperature, dtype):
    reference = reference_helpers()
    logits = torch.randn(1, 7, 13, generator=torch.Generator().manual_seed(8)).to(dtype)
    block = torch.tensor([[3, 12, 12, 12, 12, 12, 12]])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        predicted, transfer = reference["_get_transfer_index"](
            logits, temperature, block == 12, block, torch.tensor([[6]]), threshold=0.2,
        )
    decoder = NemotronThresholdDecoder(threshold=0.2, mask_id=12, eos_ids=(11,))
    actual, selected = decoder.select(logits, block, sampling=NemotronSampling(temperature, 91))
    assert torch.equal(actual[block == 12], predicted[block == 12])
    assert torch.equal(selected, transfer)


@pytest.mark.parametrize("temperature", [0.3, 0.8, 1.2])
def test_draft_and_verify_use_reference_multinomial_and_scaled_confidence(temperature):
    logits = torch.randn(1, 8, 17, generator=torch.Generator().manual_seed(4))
    probabilities = torch.softmax(logits / temperature, -1)
    expected = torch.multinomial(
        probabilities.view(-1, 17), 1, generator=torch.Generator().manual_seed(22)
    ).view(1, 8)
    confidence = probabilities.gather(-1, expected.unsqueeze(-1)).squeeze(-1)
    block = torch.full((1, 8), 16)
    decoder = NemotronThresholdDecoder(threshold=0.7, mask_id=16, eos_ids=(15,), draft=True)
    actual, transfer = decoder.select(logits, block, sampling=NemotronSampling(temperature, 22))
    assert torch.equal(actual, expected)
    expected_transfer = confidence >= 0.7
    if not expected_transfer.any():
        expected_transfer.view(-1)[confidence.argmax()] = True
    assert torch.equal(transfer, expected_transfer)
    assert torch.equal(NemotronSampling(temperature, 22).categorical(logits), expected)


@pytest.mark.parametrize("method", ["diffusion", "categorical"])
def test_sampling_distribution_and_greedy_rng_state(method):
    logits = torch.tensor([0., 0.5, 1.]).expand(20000, -1)
    sampling = NemotronSampling(0.7, 88)
    tokens = getattr(sampling, method)(logits)
    frequencies = torch.bincount(tokens, minlength=3) / len(tokens)
    torch.testing.assert_close(frequencies, torch.softmax(logits[0] / 0.7, -1), atol=0.015, rtol=0)
    greedy = NemotronSampling(0, 88)
    assert (getattr(greedy, method)(logits) == 2).all()
    assert not greedy._generators, "greedy selection must not consume random draws"


@pytest.mark.parametrize("params", [
    {"temperature": -1}, {"temperature": float("nan")}, {"temperature": float("inf")},
    {"temperature": "0.7"}, {"seed": -1}, {"seed": 1.5}, {"seed": True},
    {"seed": 2**63}, {"top_p": 0.9}, {"top_k": 10}, {"presence_penalty": 1},
])
def test_invalid_or_unsupported_parameters_do_not_silently_fall_back(params):
    with pytest.raises(ValueError):
        validate_sampling_params(params)


def stochastic_logits(batch, length):
    logits = torch.linspace(-1, 1, 16).expand(batch, length, 16).clone()
    logits[..., 3] = -torch.inf  # EOS
    logits[..., 7] = -torch.inf  # mask
    return logits


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("backend", ["fa4", "flashinfer"])
def test_seeded_request_is_independent_of_batch_neighbors(monkeypatch, speculative, backend):
    import test_nemotron_fa4 as diffusion_tests
    import test_nemotron_selfspec_paged as spec_tests
    from fluxserve.backend.execution.runners.nemotron import get_nemotron_runner

    module = spec_tests if speculative else diffusion_tests
    monkeypatch.setattr(module, "_paged_runner_cls", get_nemotron_runner(
        backend, "self_speculation" if speculative else "threshold"
    ))
    script = lambda index, rows, tokens, length: stochastic_logits(len(rows), length)
    make = module.make_runner if speculative else module.make_paged_runner
    params = [{"temperature": 0.7, "seed": 17}, {"temperature": 1.2, "seed": 9}]
    single = make(script).generate(torch.tensor([[5, 5, 5]]), [3], [12], [params[0]])
    batched = make(script).generate(torch.tensor([[5, 5, 5], [6, 6, 7]]), [3, 2], [12, 8], params)
    reversed_batch = make(script).generate(torch.tensor([[6, 6, 7], [5, 5, 5]]), [2, 3], [8, 12], params[::-1])
    assert torch.equal(single[0], batched[0])
    assert torch.equal(single[0], reversed_batch[1])


def test_online_sampling_state_survives_plans_and_is_released():
    from test_nemotron_fa4 import make_plan_runner, FakeState

    runner = make_plan_runner(lambda index, rows, tokens, length: stochastic_logits(len(rows), length))
    request = FakeState("request", [5, 5])
    request.sampling_params = {"temperature": 0.7, "seed": 12}
    logits = stochastic_logits(1, 4)
    reference = make_sampling(params=request.sampling_params)
    for _ in range(3):
        actual = runner._sampling_for_request(request).categorical(logits)
        assert torch.equal(actual, reference.categorical(logits))
    runner._release_paged_slot("request")
    assert "request" not in runner._request_sampling
    assert torch.equal(runner._sampling_for_request(request).categorical(logits),
                       make_sampling(params=request.sampling_params).categorical(logits))


def test_input_processor_resolves_seed_before_distributed_serialization():
    from fluxserve.backend.engine.processor import InputProcessor
    from fluxserve.backend.engine.distributed_executor import _state_to_payload, _state_from_payload

    args = SimpleNamespace(max_model_len=32768, generation_block_size=32,
                           sampling_defaults={"temperature": 0.8, "seed": None})
    processor = InputProcessor(args, None)
    state = processor.make_state(dict(rid="r", input_ids=[1, 2], text=None, sampling_params={}))
    assert state.sampling_params["temperature"] == 0.8
    assert isinstance(state.sampling_params["seed"], int)
    copied = _state_from_payload(_state_to_payload(state))
    logits = stochastic_logits(1, 4)
    assert torch.equal(make_sampling(params=state.sampling_params).categorical(logits),
                       make_sampling(params=copied.sampling_params).categorical(logits))


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_http_forwards_temperature_and_seed_and_rejects_invalid_sampling(chat, invalid):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fluxserve.backend.entrypoints.http_server import create_app
    from fluxserve.backend.engine.io_struct import GenerateReqOutput

    requests = []

    async def generate(req):
        requests.append(req)
        yield GenerateReqOutput(rid="r", text="answer", finish_reason="stop")

    engine = SimpleNamespace(
        server_args=SimpleNamespace(model_name="model", apply_template=False,
                                    sampling_defaults={"temperature": 0.0}),
        tokenizer=None, generate_request=generate,
    )
    body = {"temperature": 0.7, "seed": 123, "top_p": 1.0}
    if invalid:
        body["top_p"] = 0.9
    body.update({"messages": [{"role": "user", "content": "hello"}]} if chat
                else {"prompt": "hello"})
    response = TestClient(create_app(engine)).post(
        "/v1/chat/completions" if chat else "/v1/completions", json=body,
    )
    if invalid:
        assert response.status_code == 400
        assert "top_p" in response.json()["error"]
        assert not requests
        return
    assert response.status_code == 200
    assert requests[0].sampling_params["temperature"] == 0.7
    assert requests[0].sampling_params["seed"] == 123


def test_executor_passes_per_request_sampling_and_budgets():
    from fluxserve.backend.engine.executor import BlockDiffusionExecutor
    from fluxserve.backend.engine.request import RequestState
    from test_engine_executor import _Runner, _Tokenizer

    class Runner(_Runner):
        supports_request_sampling = True

        def generate(self, prompt, prompt_lengths, generation_lengths, sampling_params):
            assert generation_lengths == [2, 4]
            assert sampling_params == [
                {"temperature": 0.7, "seed": 1, "ignore_eos": False},
                {"temperature": 0, "ignore_eos": True},
            ]
            return super().generate(prompt, prompt_lengths)

    requests = [RequestState("a", [2], 2, sampling_params={"temperature": 0.7, "seed": 1}),
                RequestState("b", [2, 3], 4, ignore_eos=True, sampling_params={"temperature": 0})]
    asyncio.run(BlockDiffusionExecutor(Runner(), _Tokenizer()).execute_batch(requests))
