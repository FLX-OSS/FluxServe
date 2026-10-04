"""Test-only tensor step for LLaDA threshold decoding.

This models the work between logits projection and the runner's token scatter.
It deliberately calls the production selection helper so the probe keeps its
rounding and mask-token rules.
"""

import torch

from fluxserve.backend.execution.decoders.threshold import (
    get_transfer_index_threshold,
)


def threshold_tail(logits, block, mask_id: int, threshold: float):
    had_mask = (block == mask_id).any(dim=1)
    mask_index = block == mask_id
    predicted, transfer = get_transfer_index_threshold(
        logits,
        temperature=0,
        mask_index=mask_index,
        x=block,
        mask_id=mask_id,
        threshold=threshold,
    )
    updated = torch.where(transfer & mask_index, predicted, block)
    changed = (updated != block).any(dim=1)
    finished = ~had_mask & ~changed
    return updated, had_mask, changed, finished
