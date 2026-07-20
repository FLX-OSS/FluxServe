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

from typing import TYPE_CHECKING, NamedTuple

import torch

from fluxserve.backend.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)

if TYPE_CHECKING:
    from fluxserve.backend.layers.moe.topk import TopKOutput


class StandardDispatchOutput(NamedTuple):
    """Standard dispatch output."""

    hidden_states: torch.Tensor
    topk_output: TopKOutput

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


assert isinstance(StandardDispatchOutput, DispatchOutput)


class StandardCombineInput(NamedTuple):
    """Standard combine input."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.STANDARD


assert isinstance(StandardCombineInput, CombineInput)


class StandardDispatcher(BaseDispatcher):

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        return StandardDispatchOutput(
            hidden_states=hidden_states, topk_output=topk_output
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if isinstance(combine_input, StandardCombineInput):
            return combine_input.hidden_states
        else:
            # TODO: this branch should be removed in the future
            assert isinstance(combine_input, torch.Tensor)
            return combine_input
