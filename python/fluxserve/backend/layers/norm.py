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

"""
    Normalization layers.
"""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.normalized_shape = tuple((hidden_size,))

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        flux_kernel_output = self._forward_flux_kernel(x, residual)
        if flux_kernel_output is not None:
            return flux_kernel_output

        return self._forward_torch(x, residual)

    def _forward_flux_kernel(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        if not x.is_cuda or x.numel() == 0:
            return None

        try:
            from flux_kernel.ops import rmsnorm
        except Exception:
            return None

        try:
            return rmsnorm(
                x,
                self.weight,
                self.variance_epsilon,
                residual=residual,
            )
        except Exception:
            logger.debug(
                "flux_kernel RMSNorm failed; falling back to torch",
                exc_info=True,
            )
            return None

    def _forward_torch(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x
            x = F.rms_norm(
                x.to(torch.float32),
                self.normalized_shape,
                self.weight,
                self.variance_epsilon,
            )
            return x.to(input_dtype), residual

        out = F.rms_norm(
            x.to(torch.float32),
            self.normalized_shape,
            self.weight,
            self.variance_epsilon,
        )
        return out.to(input_dtype)


__all__ = ["RMSNorm"]
