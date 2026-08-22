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

import time
from typing import Any

from fluxserve.backend.engine.io_struct import GenerateReqInput, GenerateReqOutput
from fluxserve.backend.engine.request import RequestState


class InputProcessor:
    def __init__(self, server_args, tokenizer):
        self.server_args = server_args
        self.tokenizer = tokenizer

    def make_states(self, obj: GenerateReqInput) -> list[RequestState]:
        return [self.make_state(item) for item in obj.iter_items()]

    def make_state(self, item: dict[str, Any]) -> RequestState:
        input_ids = item["input_ids"]
        prompt_text = item["text"] or ""
        if input_ids is None:
            input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = [int(x) for x in input_ids]

        params = item["sampling_params"] or {}
        remaining_context = self.server_args.max_model_len - len(input_ids)
        if remaining_context <= 0:
            raise ValueError(
                f"request input length {len(input_ids)} exceeds max_model_len={self.server_args.max_model_len}"
            )

        generation_block_size = max(
            1, int(getattr(self.server_args, "generation_block_size", 1))
        )
        usable_context = (
            remaining_context // generation_block_size * generation_block_size
        )
        if usable_context <= 0:
            raise ValueError(
                f"request input length {len(input_ids)} does not leave room for "
                f"a generation block of {generation_block_size} tokens within "
                f"max_model_len={self.server_args.max_model_len}"
            )
        max_new_tokens = int(params.get("max_tokens", params.get("max_new_tokens", 128)))
        max_new_tokens = max(1, min(max_new_tokens, usable_context))
        ignore_eos = bool(params.get("ignore_eos", False))

        return RequestState(
            rid=str(item["rid"]),
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            ignore_eos=ignore_eos,
            prompt_text=prompt_text,
            sampling_params=dict(params),
        )


class OutputProcessor:
    def make_output(
        self,
        state: RequestState,
        token_ids: list[int],
        text: str,
        finish_reason: str | None,
    ) -> GenerateReqOutput:
        state.output_ids.extend(token_ids)
        state.decoded_text += text
        if finish_reason is not None:
            state.finished_reason = finish_reason
            state.completed_time = time.time()
        return GenerateReqOutput(
            rid=state.rid,
            text=text,
            token_ids=token_ids,
            finish_reason=state.finished_reason,
            meta=state.output_metadata() if state.finished else {},
        )

    def make_error_output(self, state: RequestState, error: str) -> GenerateReqOutput:
        state.finished_reason = "error"
        state.completed_time = time.time()
        return GenerateReqOutput(
            rid=state.rid,
            error=error,
            finish_reason="error",
            meta=state.output_metadata(),
        )

    def make_abort_output(self, state: RequestState, reason: str) -> GenerateReqOutput:
        state.finished_reason = "abort"
        state.completed_time = time.time()
        return GenerateReqOutput(
            rid=state.rid,
            error=reason,
            finish_reason="abort",
            meta=state.output_metadata(),
        )
