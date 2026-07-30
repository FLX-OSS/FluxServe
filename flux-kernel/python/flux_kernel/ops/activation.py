# Copyright (c) 2026 FLUX-OSS

"""Activation kernels with SGL-compatible calling conventions."""

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
