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

from typing import Any, Optional

import torch

from fluxserve.backend.execution.forward_batch_info import ForwardBatch
from fluxserve.backend.layers.attention.base import (
    AttentionForwardConfig,
    DenseAttention,
)
from fluxserve.backend.layers.attention.flashinfer import (
    FlashInferPagedAttention,
    FlashInferPagedPrefillAttention,
    FlashInferRaggedAttention,
    FlashInferRaggedPrefillAttention,
)


class AttentionForward:
    """Routes one attention call to ragged FlashInfer or dense attention."""

    def __init__(self, config: AttentionForwardConfig):
        self.config = config
        self.dense = DenseAttention(config)
        self.flashinfer_ragged_prefill = FlashInferRaggedPrefillAttention(config)
        self.flashinfer_paged_prefill = FlashInferPagedPrefillAttention(config)
        self.flashinfer_paged = FlashInferPagedAttention(config)
        self.flashinfer_ragged = FlashInferRaggedAttention(config)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        past_key_values: Any = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        if self.flashinfer_paged_prefill.can_run(
            q,
            k,
            past_key_values,
            attention_mask,
            forward_batch,
        ):
            present_key_values = (k, v) if use_cache else None
            return (
                self.flashinfer_paged_prefill.forward(
                    q,
                    k,
                    v,
                    past_key_values,
                    forward_batch,
                ),
                present_key_values,
            )
        if (
            forward_batch is not None
            and getattr(forward_batch, "use_flashinfer_paged_prefill", False)
        ):
            raise RuntimeError(
                "FlashInfer paged prefill was requested but cannot run."
            )

        if self.flashinfer_ragged_prefill.can_run(q, k, attention_mask, forward_batch):
            present_key_values = (k, v) if use_cache else None
            return (
                self.flashinfer_ragged_prefill.forward(q, k, v, forward_batch),
                present_key_values,
            )
        if (
            forward_batch is not None
            and getattr(forward_batch, "use_flashinfer_prefill", False)
        ):
            raise RuntimeError(
                "FlashInfer ragged prefill was requested but cannot run."
            )

        if self.flashinfer_paged.can_run(q, past_key_values, attention_mask, forward_batch):
            present_key_values = (k, v) if use_cache else None
            return (
                self.flashinfer_paged.forward(q, k, v, past_key_values, forward_batch),
                present_key_values,
            )
        if (
            forward_batch is not None
            and getattr(forward_batch, "use_flashinfer_paged_decode", False)
        ):
            raise RuntimeError("FlashInfer paged decode was requested but cannot run.")

        k, v = self.dense.splice_cache(k, v, past_key_values)
        present_key_values = (k, v) if use_cache else None

        if self.flashinfer_ragged.can_run(
            q,
            k,
            past_key_values,
            attention_mask,
            forward_batch,
        ):
            return (
                self.flashinfer_ragged.forward(q, k, v, forward_batch),
                present_key_values,
            )

        return self.dense.forward(q, k, v, attention_mask), present_key_values
