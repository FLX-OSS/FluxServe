# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from fluxserve.diagnostics.sgl_kernel_trace import trace_phase

from .request import RequestState


@dataclass
class ExecutionResult:
    rid: str
    token_ids: list[int]
    text: str
    finish_reason: str = "length"


@dataclass
class ForwardStepResult:
    rid: str
    token_ids: list[int]
    text: str
    finished: bool = False
    finish_reason: str | None = None
    reserve_tokens: int = 0


class GenerationExecutor(Protocol):
    async def execute_batch(self, requests: list[RequestState]) -> list[ExecutionResult]:
        ...


class PlanExecutor(Protocol):
    async def execute_forward_plan(
        self, op, states_by_id: dict[str, RequestState]
    ) -> list[ForwardStepResult]:
        ...


class BlockDiffusionExecutor:
    def __init__(self, runner, tokenizer):
        self.runner = runner
        self.tokenizer = tokenizer

    @trace_phase("request_eager")
    async def execute_batch(self, requests: list[RequestState]) -> list[ExecutionResult]:
        if not requests:
            return []
        device = getattr(self.runner, "device", "cuda")
        mask_id = self.runner.decoder.mask_id
        prompt_ids = [req.input_ids + req.output_ids for req in requests]
        max_prompt = max(len(ids) for ids in prompt_ids)
        prompt = torch.full(
            (len(requests), max_prompt),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        for i, ids in enumerate(prompt_ids):
            prompt[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

        original_gen_length = self.runner.runner_config.gen_length
        original_early_stop = self.runner.early_stop
        self.runner.runner_config.gen_length = max(req.max_new_tokens for req in requests)
        if any(req.ignore_eos for req in requests):
            self.runner.early_stop = False
        try:
            output = self.runner.generate(prompt)
        finally:
            self.runner.runner_config.gen_length = original_gen_length
            self.runner.early_stop = original_early_stop

        results = []
        eos_id = self.runner.decoder.eos_id
        mask_id = self.runner.decoder.mask_id
        for i, req in enumerate(requests):
            row = output[i].detach().cpu().tolist()
            generated = row[max_prompt : max_prompt + req.max_new_tokens]
            if not req.ignore_eos and eos_id in generated:
                generated = generated[: generated.index(eos_id)]
                finish_reason = "stop"
            else:
                finish_reason = "length"
            if req.ignore_eos:
                generated = [tok for tok in generated if tok != mask_id]
            else:
                generated = [tok for tok in generated if tok != mask_id and tok != eos_id]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(
                ExecutionResult(
                    rid=req.rid,
                    token_ids=generated,
                    text=text,
                    finish_reason=finish_reason,
                )
            )
        return results

    @trace_phase("request_eager")
    async def execute_forward_plan(
        self, op, states_by_id: dict[str, RequestState]
    ) -> list[ForwardStepResult]:
        if hasattr(self.runner, "execute_paged_forward_plan"):
            return await self.runner.execute_paged_forward_plan(
                op,
                states_by_id,
                self.tokenizer,
            )
        raise RuntimeError(
            "paged scheduler_policy requires FlashInfer paged KV execution."
        )

    async def release_requests(self, request_ids) -> None:
        release = getattr(self.runner, "release_paged_requests", None)
        if release is not None:
            release(request_ids)

    async def _execute_decode_block_batch(
        self, requests: list[RequestState], block_size: int
    ) -> list[ForwardStepResult]:
        original_max_new = [req.max_new_tokens for req in requests]
        for req in requests:
            remaining = max(0, req.max_new_tokens - req.completion_token_count)
            req.max_new_tokens = min(block_size, remaining)
        try:
            active = [req for req in requests if req.max_new_tokens > 0]
            batch_results = await self.execute_batch(active)
        finally:
            for req, max_new_tokens in zip(requests, original_max_new, strict=True):
                req.max_new_tokens = max_new_tokens

        by_id = {result.rid: result for result in batch_results}
        step_results: list[ForwardStepResult] = []
        for req in requests:
            result = by_id.get(req.rid)
            if result is None:
                step_results.append(
                    ForwardStepResult(
                        rid=req.rid,
                        token_ids=[],
                        text="",
                        finished=True,
                        finish_reason="length",
                    )
                )
                continue

            projected_completion = req.completion_token_count + len(result.token_ids)
            finished = (
                result.finish_reason == "stop"
                or projected_completion >= req.max_new_tokens
            )
            finish_reason = result.finish_reason if finished else None
            reserve_tokens = 0 if finished else block_size
            step_results.append(
                ForwardStepResult(
                    rid=req.rid,
                    token_ids=result.token_ids,
                    text=result.text,
                    finished=finished,
                    finish_reason=finish_reason,
                    reserve_tokens=reserve_tokens,
                )
            )
        return step_results
