
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

import numpy as np

import torch
import torch.nn.functional as F

from fluxserve.backend.execution.decoders.base import ParallelDecoder
from fluxserve.backend.execution.decoders.utils import (
    add_gumbel_noise,
    broadcast_if_needed,
)


def get_transfer_index_threshold(
    logits,
    temperature,
    mask_index,
    x,
    mask_id,
    threshold,
    rm_mask=True,
    use_float64=False,
    **kwargs,
):
    if use_float64:
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )
    else:
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        p = F.softmax(logits.to(torch.float32), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )

    if rm_mask:
        mask_index = mask_index & (x0 != mask_id)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    actual_threshold = (
        torch.max(confidence, dim=1)[0] - 1e-5
    ).clamp(-1000, threshold).unsqueeze(-1)
    transfer_index = confidence >= actual_threshold
    return x0, transfer_index


class ThresholdParallelDecoder(ParallelDecoder):
    """
        Parallel decoding driven by a confidence threshold.
    """
    def __init__(
            self,
            temperature,
            threshold,
            remasking='low_confidence',
            mask_id=126336,
            eos_id=126081,
            use_float64=False,
    ):
        super().__init__(temperature, remasking, mask_id)
        self.threshold = threshold
        self.eos_id = eos_id
        self.use_float64 = use_float64

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in the same block of multiple samples.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index_threshold(
            logits,
            self.temperature,
            mask_index,
            curr_x,
            self.mask_id,
            threshold=iter_threshold,
            use_float64=self.use_float64,
        )
        transfer_index = torch.logical_and(transfer_index, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)
        broadcast_if_needed(x.data)

    def batch_decode(self, logits, block_start, x, block_length, iter_threshold=None):
        """ Decode the logits in the different blocks of multiple samples, indicated by 1-d block_start tensor.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        B, T = x.data.shape
        device = x.data.device

        offset = torch.arange(block_length, device=device).unsqueeze(0) + block_start.unsqueeze(1)  # [B, block_length]

        x_block = torch.gather(x.data, 1, offset.clamp(max=T - 1)) 

        mask_index = (x_block == self.mask_id)

        x0, transfer_index = get_transfer_index_threshold(
            logits,
            self.temperature,
            mask_index,
            x_block,
            self.mask_id,
            threshold=iter_threshold,
            use_float64=self.use_float64,
        )

        transfer_index = transfer_index & mask_index

        x_updated = torch.where(transfer_index, x0, x_block)

        x_flat = x.data.view(-1)
        flat_idx = offset + torch.arange(B, device=device).unsqueeze(1) * T
        x_flat[flat_idx] = x_updated

        broadcast_if_needed(x.data)


class CreditThresholdParallelDecoder(ThresholdParallelDecoder):
    """ 
        Deocde tokens in parallel based on a threshold + credit.
    """
    def __init__(self,
                 credit_alpha=0.7,
                 boost_gamma=0.2,
                 decay_beta=0.8,
                 **kwargs):
        super().__init__(**kwargs)

        self.credit_alpha = credit_alpha
        self.boost_gamma = boost_gamma
        self.decay_beta = decay_beta

        self._credit_mats = {}
        self._credit_iters = {}

    def _apply_credit_fusion(self, logits, mask_index, key):
        """
        EMA-based credit fusion (no CM, no pre-credit):
        - Maintains a per-block CreditMatrix (EMA with decay).
        - Accumulates enhanced top-1 probability only on masked positions.
        - Returns fused_logits.
        """
        B, L, V = logits.shape
        device = logits.device

        mat = self._credit_mats.get(key, None)
        if mat is None or mat.shape != (B, L, V) or mat.device != device:
            mat = torch.zeros((B, L, V), dtype=torch.float32, device=device)
            self._credit_mats[key] = mat
            self._credit_iters[key] = 0

        iter_idx = self._credit_iters[key]

        if iter_idx > 0:
            mat.mul_(self.decay_beta)

        probs = F.softmax(logits.to(torch.float32), dim=-1)
        top1_probs, top1_idx = torch.max(probs, dim=-1)
        enhanced = top1_probs.pow(self.boost_gamma).to(mat.dtype)
        update_vals = enhanced * mask_index.to(enhanced.dtype)
        mat.scatter_add_(2, top1_idx.unsqueeze(-1), update_vals.unsqueeze(-1))

        fused_logits = logits + self.credit_alpha * torch.log(mat + 1)
        self._credit_iters[key] = iter_idx + 1
        return fused_logits

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in a block."""
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        key = (block_start, block_end)
        used_logits = self._apply_credit_fusion(logits, mask_index, key)

        x0, transfer_index = get_transfer_index_threshold(
            used_logits,
            self.temperature,
            mask_index,
            curr_x,
            self.mask_id,
            threshold=iter_threshold,
            use_float64=self.use_float64,
        )

        transfer_index = torch.logical_and(transfer_index, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)

        if hasattr(x, 'data'):
            has_mask = (x.data == self.mask_id).any()
        else:
            if x.dim() > 0:
                has_mask = (x == self.mask_id).any()
            else:
                has_mask = (x == self.mask_id)

        if not has_mask:
            self._credit_mats.clear()
            self._credit_iters.clear()
        broadcast_if_needed(x.data)
