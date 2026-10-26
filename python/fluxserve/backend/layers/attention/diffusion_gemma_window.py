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
    Diffusion-Gemma sliding-window semantics.

Prefill and commit are causal: a query at position ``q`` sees keys
``q - (W - 1) <= k <= q``. The denoise canvas is bidirectional and has two
readings of the window:

``per_query`` (FluxServe default)
    every canvas query sees keys with ``|q - k| < W``.
``per_row``
    every canvas query sees the same cache slice, the last ``window_left``
    keys before the canvas (``k >= q_start - window_left``), plus the whole
    canvas. This is what the HF reference decoder does: it reads the encoder's
    sliding-window cache (``W - 1`` tokens) and appends the canvas.

The two only differ once the prefix reaches ``W - canvas_length + 1`` tokens.
"""

from __future__ import annotations

import torch

SLIDING_WINDOW_MODES = ("per_query", "per_row")


def validate_sliding_window_mode(mode: str, window_left: int | None = None) -> None:
    if mode not in SLIDING_WINDOW_MODES:
        raise ValueError(
            f"sliding_window_mode must be one of {SLIDING_WINDOW_MODES}, got {mode!r}"
        )
    if window_left is not None and int(window_left) < 0:
        raise ValueError(f"sliding_window_left must be non-negative, got {window_left!r}")


def sliding_window_mask(
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    *,
    phase: str,
    sliding_window: int,
    mode: str = "per_query",
    window_left: int | None = None,
    q_start: int | torch.Tensor = 0,
) -> torch.Tensor:
    """Window part of the mask, shape ``q_pos.shape + (k_pos.shape[-1],)``; True = visible.

    ``q_start`` is the first canvas position (the ``per_row`` anchor): an int, or
    a tensor of shape ``q_pos.shape[:-1]``. Causality is the caller's job.
    """
    q = q_pos.long().unsqueeze(-1)
    k = k_pos.long().unsqueeze(-2) if k_pos.dim() > 1 else k_pos.long()
    width = int(sliding_window)
    if phase != "denoise" or mode == "per_query":
        return (q - k).abs() < width
    left = width - 1 if window_left is None else int(window_left)
    if isinstance(q_start, torch.Tensor):  # one anchor per row of q_pos[..., :]
        q_start = q_start.long().reshape(*q_start.shape, 1, 1)
    return (k >= q_start - left) & (k - q < width)


def fa4_window_kwargs(
    *,
    phase: str,
    sliding_window: int,
    mode: str = "per_query",
    window_left: int | None = None,
) -> dict:
    """FA4 ``causal`` / ``window_size`` / ``window_left_anchor`` for one phase.

    FA4 aligns windows bottom-right per sequence (the queries are the last
    ``q_len`` of ``kv_len`` keys), so these reproduce ``sliding_window_mask``.
    """
    width = int(sliding_window)
    if phase in ("prefill", "commit"):
        return dict(causal=True, window_size=(width - 1, None), window_left_anchor="query")
    if phase != "denoise":
        raise ValueError(f"unknown Diffusion-Gemma phase {phase!r}")
    if mode == "per_query":
        return dict(causal=False, window_size=(width - 1, width - 1), window_left_anchor="query")
    left = width - 1 if window_left is None else int(window_left)
    return dict(causal=False, window_size=(left, width - 1), window_left_anchor="first_query")


__all__ = [
    "SLIDING_WINDOW_MODES",
    "fa4_window_kwargs",
    "sliding_window_mask",
    "validate_sliding_window_mode",
]
