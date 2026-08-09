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
    Activation kernels.
"""

from __future__ import annotations

import torch

from flux_kernel.cuda.activation import load_activation


def silu_and_mul(input: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute ``silu(input[..., :d]) * input[..., d:]`` into ``out``."""
    if input.shape[-1] % 2:
        raise ValueError("input's last dimension must be even")
    expected_shape = input.shape[:-1] + (input.shape[-1] // 2,)
    if out.shape != expected_shape:
        raise ValueError(f"out shape must be {expected_shape}, got {tuple(out.shape)}")
    if out.device != input.device or out.dtype != input.dtype:
        raise ValueError("out must have the same device and dtype as input")

    load_activation()
    return torch.ops.flux_kernel.silu_and_mul(out, input)


__all__ = ["silu_and_mul"]
