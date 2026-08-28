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

"""LLaDA2.1 joint Mask-to-Token / Token-to-Token decoding.

Reference: ``LLaDA2MoeModelLM.generate()`` shipped inside the
``inclusionAI/LLaDA2.1-mini`` checkpoint. Two deliberate deviations from that
reference (see docs/serving/llada2.1-model-support-development-guide.md):

- the ``mask_id`` logit is suppressed before the argmax so that no update path
  can ever write a mask back into the block (the paper states this invariant;
  the reference does not enforce it); and
- M2T uses ``confidence >= actual_threshold`` (matching the existing 2.0
  decoder) where the reference uses a strict ``>``. The two differ only on an
  exact floating-point tie at the threshold value.
"""

import numpy as np

import torch
import torch.nn.functional as F

from fluxserve.backend.execution.decoders.base import ParallelDecoder
from fluxserve.backend.execution.decoders.utils import broadcast_if_needed


def joint_threshold_update(
    logits,
    x_block,
    mask_id,
    threshold,
    editing_threshold,
    prompt_positions,
    allow_edit,
):
    """Joint M2T + T2T selection over one active block per row.

    Pure tensor function so it can be unit-tested without a model or a GPU.
    ``logits`` is mutated: the ``mask_id`` column is set to ``-inf``.

    Args:
        logits: ``[B, L, V]`` model logits for the active block.
        x_block: ``[B, L]`` current tokens of the active block.
        mask_id: mask token id.
        threshold: M2T confidence threshold (tau_mask).
        editing_threshold: T2T confidence threshold (tau_edit).
        prompt_positions: ``[B, L]`` bool, True where the position belongs to
            the immutable prompt.
        allow_edit: ``[B]`` bool, False once a row has spent its post-edit
            budget; disables T2T while leaving M2T intact.

    Returns:
        ``(x_updated, m2t_transfer, t2t_transfer)``.
    """
    # Structural no-remask: mask_id can never be selected as a candidate, so
    # neither update path can write a mask and no row can stall on a masked
    # position whose argmax was mask_id.
    logits[..., mask_id] = -float("inf")

    x0 = torch.argmax(logits, dim=-1)
    x0_p = torch.squeeze(
        torch.gather(
            F.softmax(logits.to(torch.float32), dim=-1),
            dim=-1,
            index=torch.unsqueeze(x0, -1),
        ),
        -1,
    )

    mask_index = x_block == mask_id
    confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))
    # Transfers every masked position above threshold, and always at least the
    # single highest-confidence masked position per row (progress guarantee).
    # A row with no masks has all -inf confidence and transfers nothing.
    actual_threshold = (
        (torch.max(confidence, dim=1)[0] - 1e-5)
        .clamp(-1000, threshold)
        .unsqueeze(-1)
    )
    m2t_transfer = confidence >= actual_threshold

    editable = (~mask_index) & (~prompt_positions) & allow_edit.unsqueeze(-1)
    t2t_transfer = editable & (x0 != x_block) & (x0_p > editing_threshold)

    x_updated = torch.where(m2t_transfer | t2t_transfer, x0, x_block)
    return x_updated, m2t_transfer, t2t_transfer


class JointThresholdDecoder(ParallelDecoder):
    """LLaDA2.1 joint threshold decoding (M2T + T2T token editing)."""

    # Signals the runners to pass prompt_lengths / allow_edit to batch_decode.
    needs_editing_inputs = True

    def __init__(
        self,
        threshold,
        editing_threshold,
        temperature=0,
        mask_id=156895,
        eos_id=156892,
    ):
        if temperature not in (0, 0.0):
            raise ValueError(
                "JointThresholdDecoder only supports temperature=0; "
                f"got {temperature!r}. T2T editing with sampling has no "
                "termination guarantee beyond max_post_steps."
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
        if not 0.0 <= editing_threshold <= 1.0:
            raise ValueError(
                f"editing_threshold must be in [0, 1], got {editing_threshold!r}"
            )
        super().__init__(temperature, mask_id=mask_id)
        self.threshold = threshold
        self.editing_threshold = editing_threshold
        self.eos_id = eos_id

    def batch_decode(
        self,
        logits,
        block_start,
        x,
        block_length,
        prompt_lengths=None,
        allow_edit=None,
    ):
        """Decode active blocks of multiple rows, indicated by 1-d block_start.

        Same in-place contract as ``ThresholdParallelDecoder.batch_decode``.
        ``prompt_lengths`` (``[B]``, absolute prompt length per row) is
        required: prompt protection must come from explicit positions, never
        from token values. ``allow_edit`` (``[B]`` bool) defaults to all-True.
        """
        if prompt_lengths is None:
            raise ValueError(
                "JointThresholdDecoder.batch_decode requires prompt_lengths; "
                "the runner must pass the per-row prompt boundary explicitly."
            )
        B, T = x.data.shape
        device = x.data.device

        offset = torch.arange(block_length, device=device).unsqueeze(
            0
        ) + block_start.unsqueeze(1)  # [B, block_length] absolute positions

        x_block = torch.gather(x.data, 1, offset.clamp(max=T - 1))
        prompt_positions = offset < prompt_lengths.unsqueeze(1)
        if allow_edit is None:
            allow_edit = torch.ones(B, dtype=torch.bool, device=device)

        x_updated, _, _ = joint_threshold_update(
            logits,
            x_block,
            self.mask_id,
            self.threshold,
            self.editing_threshold,
            prompt_positions,
            allow_edit,
        )

        x_flat = x.data.view(-1)
        flat_idx = offset.clamp(max=T - 1) + torch.arange(
            B, device=device
        ).unsqueeze(1) * T
        x_flat[flat_idx] = x_updated

        broadcast_if_needed(x.data)
