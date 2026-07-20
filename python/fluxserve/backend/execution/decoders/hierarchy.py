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

import math

import torch
import torch.nn.functional as F

from fluxserve.backend.execution.decoders.base import ParallelDecoder
from fluxserve.backend.execution.decoders.utils import add_gumbel_noise, broadcast_if_needed


class HierarchyDecoder(ParallelDecoder):
    """
        Decode tokens hierarchically to force separate decisions.
        Only supports batch size 1.
    """

    def __init__(
        self,
        temperature,
        remasking="low_confidence",
        mask_id=126336,
        eos_id=126081,
        threshold=None,
        low_threshold=0.4,
    ):
        super().__init__(temperature, remasking, mask_id)
        self.iter = 0
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.threshold = threshold
        self.low_threshold = low_threshold

    def get_transfer_index(self, logits, mask_index, iter_threshold, **kwargs):
        del kwargs
        B, L = mask_index.shape
        assert B == 1
        device = logits.device

        if not math.isclose(self.temperature, 0.0):
            logits_with_noise = add_gumbel_noise(logits, temperature=self.temperature)
        else:
            logits_with_noise = logits

        x0 = torch.argmax(logits_with_noise, dim=-1)
        x0_logp = F.log_softmax(logits, dim=-1).gather(
            -1, x0.unsqueeze(-1)
        ).squeeze(-1)
        x0_p = x0_logp.exp()

        neg_inf_val = torch.finfo(x0_p.dtype).min
        confidence = torch.where(
            mask_index,
            x0_p,
            torch.tensor(neg_inf_val, device=device, dtype=x0_p.dtype),
        )

        prev = torch.cat(
            [mask_index.new_zeros((B, 1), dtype=torch.bool), mask_index[:, :-1]],
            dim=1,
        )
        starts = torch.logical_and(mask_index, torch.logical_not(prev))

        seg_id = torch.cumsum(starts.to(torch.int64), dim=-1) - 1
        seg_id = torch.where(mask_index, seg_id, 0)

        seg_max = torch.full((B, L), neg_inf_val, device=device, dtype=confidence.dtype)
        seg_max = torch.scatter_reduce(
            seg_max,
            dim=1,
            index=seg_id,
            src=confidence,
            reduce="amax",
            include_self=True,
        )

        seg_max_at_pos = seg_max.gather(dim=1, index=seg_id)
        transfer_index = confidence == seg_max_at_pos

        if self.low_threshold is not None:
            transfer_index = torch.logical_and(
                transfer_index, torch.gt(confidence, self.low_threshold)
            )
        if iter_threshold is not None:
            transfer_index = torch.logical_or(
                transfer_index, torch.gt(confidence, iter_threshold)
            )

        top1_idx = torch.argmax(confidence, dim=-1)
        top1 = torch.nn.functional.one_hot(top1_idx, num_classes=L).to(torch.bool)
        transfer_index = torch.logical_or(transfer_index, top1)

        return x0, transfer_index

    def block_init(self, block_x, block_id):
        del block_x, block_id
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = x[:, block_start:block_end] == self.mask_id
        assert mask_index.shape[1] == logits.shape[1]

        x0, transfer_index = self.get_transfer_index(
            logits, mask_index, iter_threshold
        )
        self.iter += 1
        transfer_index = torch.logical_and(transfer_index, mask_index)
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        broadcast_if_needed(x.data)
