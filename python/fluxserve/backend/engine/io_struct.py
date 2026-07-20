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

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerateReqInput:
    text: str | list[str] | None = None
    input_ids: list[int] | list[list[int]] | None = None
    sampling_params: dict[str, Any] | list[dict[str, Any]] | None = None
    rid: str | list[str] | None = None
    stream: bool = False

    def normalize(self) -> None:
        if (self.text is None) == (self.input_ids is None):
            raise ValueError("Exactly one of text or input_ids must be provided.")

        if self.text is not None:
            self.is_single = isinstance(self.text, str)
            self.batch_size = 1 if self.is_single else len(self.text)
        else:
            if not self.input_ids:
                raise ValueError("input_ids must not be empty.")
            self.is_single = isinstance(self.input_ids[0], int)
            self.batch_size = 1 if self.is_single else len(self.input_ids)

        if self.sampling_params is None:
            self.sampling_params = {} if self.is_single else [{} for _ in range(self.batch_size)]
        elif self.is_single and isinstance(self.sampling_params, list):
            if len(self.sampling_params) != 1:
                raise ValueError("single request sampling_params list must have length 1.")
            self.sampling_params = self.sampling_params[0]
        elif not self.is_single and isinstance(self.sampling_params, dict):
            self.sampling_params = [self.sampling_params.copy() for _ in range(self.batch_size)]

        if self.rid is None:
            self.rid = uuid.uuid4().hex if self.is_single else [uuid.uuid4().hex for _ in range(self.batch_size)]

    def iter_items(self):
        self.normalize()
        texts = [self.text] if self.is_single and self.text is not None else self.text
        input_ids = [self.input_ids] if self.is_single and self.input_ids is not None else self.input_ids
        params = [self.sampling_params] if self.is_single else self.sampling_params
        rids = [self.rid] if self.is_single else self.rid
        for idx in range(self.batch_size):
            yield {
                "rid": rids[idx],
                "text": None if texts is None else texts[idx],
                "input_ids": None if input_ids is None else input_ids[idx],
                "sampling_params": params[idx],
            }


@dataclass
class GenerateReqOutput:
    rid: str
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None or self.error is not None


@dataclass
class AbortReq:
    rid: str
    reason: str = "aborted"


@dataclass
class HealthCheckOutput:
    status: str = "ok"
