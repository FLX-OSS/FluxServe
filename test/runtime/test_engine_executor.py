import asyncio

import torch

from fluxserve.backend.engine.executor import BlockDiffusionExecutor
from fluxserve.backend.engine.request import RequestState


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token) for token in token_ids)


class _Decoder:
    mask_id = 0
    eos_id = 1
    eos_ids = (1, 106, 50)


class _RunnerConfig:
    gen_length = 4


class _Runner:
    device = "cpu"
    decoder = _Decoder()
    runner_config = _RunnerConfig()
    early_stop = True
    requires_prompt_lengths = True

    def generate(self, prompt, prompt_lengths):
        rows = []
        suffixes = ([7, 106, 8, 9], [6, 50, 5, 4])
        for row, suffix in zip(prompt, suffixes, strict=True):
            rows.append(torch.cat((row, torch.tensor(suffix))))
        return torch.stack(rows)


def test_executor_stops_at_any_configured_eos_id():
    executor = BlockDiffusionExecutor(_Runner(), _Tokenizer())
    requests = [
        RequestState(rid="eos-106", input_ids=[2], max_new_tokens=4),
        RequestState(rid="eos-50", input_ids=[2, 3], max_new_tokens=4),
    ]

    results = asyncio.run(executor.execute_batch(requests))

    assert [result.token_ids for result in results] == [[7], [6]]
    assert [result.text for result in results] == ["7", "6"]
    assert [result.finish_reason for result in results] == ["stop", "stop"]
