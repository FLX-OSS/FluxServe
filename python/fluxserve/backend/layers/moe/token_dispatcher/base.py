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

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TypeGuard, Union, runtime_checkable

import torch

if TYPE_CHECKING:
    from fluxserve.backend.layers.moe.token_dispatcher import (
        DeepEPLLCombineInput,
        DeepEPLLOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalOutput,
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from fluxserve.backend.layers.moe.topk import TopKOutput

# ------------------------------ Dispatch Output -------------------------------------


class DispatchOutputChecker:

    @staticmethod
    def format_is_standard(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[StandardDispatchOutput]:
        return dispatch_output.format.is_standard()

    @staticmethod
    def format_is_deepep_normal(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPNormalOutput]:
        return dispatch_output.format.is_deepep_normal()

    @staticmethod
    def format_is_deepep_ll(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPLLOutput]:
        return dispatch_output.format.is_deepep_ll()

    @staticmethod
    def format_is_deepep(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[Union[DeepEPNormalOutput, DeepEPLLOutput]]:
        return dispatch_output.format.is_deepep()


class DispatchOutputFormat(Enum):

    STANDARD = "standard"
    DEEPEP_NORMAL = "deepep_normal"
    DEEPEP_LL = "deepep_ll"

    def is_standard(self) -> bool:
        return self == DispatchOutputFormat.STANDARD

    def is_deepep_normal(self) -> bool:
        return self == DispatchOutputFormat.DEEPEP_NORMAL

    def is_deepep_ll(self) -> bool:
        return self == DispatchOutputFormat.DEEPEP_LL

    def is_deepep(self) -> bool:
        return self in [
            DispatchOutputFormat.DEEPEP_NORMAL,
            DispatchOutputFormat.DEEPEP_LL,
        ]


@runtime_checkable
class DispatchOutput(Protocol):
    """Protocol for dispatch outputs in different formats."""

    # TODO: add hidden_states to the protocol

    @property
    def format(self) -> DispatchOutputFormat: ...


# ------------------------------ Combine Input -------------------------------------


class CombineInputChecker:
    @staticmethod
    def format_is_standard(
        combine_input: CombineInput,
    ) -> TypeGuard[StandardCombineInput]:
        return combine_input.format == CombineInputFormat.STANDARD

    @staticmethod
    def format_is_deepep_normal(
        combine_input: CombineInput,
    ) -> TypeGuard[DeepEPNormalCombineInput]:
        return combine_input.format == CombineInputFormat.DEEPEP_NORMAL

    @staticmethod
    def format_is_deepep_ll(
        combine_input: CombineInput,
    ) -> TypeGuard[DeepEPLLCombineInput]:
        return combine_input.format == CombineInputFormat.DEEPEP_LL

    @staticmethod
    def format_is_deepep(
        combine_input: CombineInput,
    ) -> TypeGuard[Union[DeepEPNormalCombineInput, DeepEPLLCombineInput]]:
        return combine_input.format in [
            CombineInputFormat.DEEPEP_NORMAL,
            CombineInputFormat.DEEPEP_LL,
        ]


class CombineInputFormat(Enum):
    STANDARD = "standard"
    DEEPEP_NORMAL = "deepep_normal"
    DEEPEP_LL = "deepep_ll"


@runtime_checkable
class CombineInput(Protocol):
    """Protocol for combine inputs in different formats."""

    # TODO: add hidden_states to the protocol

    @property
    def format(self) -> CombineInputFormat: ...


# ------------------------------ Base Dispatcher -------------------------------------


class BaseDispatcherConfig(ABC):
    """Base class for dispatcher configs."""

    pass


class BaseDispatcher(ABC):
    """Base class for dispatchers."""

    @abstractmethod
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs
    ) -> DispatchOutput:
        pass

    @abstractmethod
    def combine(self, combine_input: CombineInput, **kwargs) -> torch.Tensor:
        pass
