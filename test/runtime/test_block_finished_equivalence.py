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

"""LLaDA2.0 regression guard for the new block-completion predicate.

The runners used to declare a block finished when the pre-update gather held
no masks. They now use ``(~had_mask) & (~changed)`` so that LLaDA2.1 editing
iterations keep a block active. For the 2.0 decoders the two must be
identical on every iteration, because those decoders only ever write masked
positions: a mask-free block cannot change.
"""

import pytest
import torch

from fluxserve.backend.execution.decoders import (
    CreditThresholdParallelDecoder,
    ThresholdParallelDecoder,
)
from fluxserve.backend.execution.runners.utils import gather_blocks

VOCAB = 32
MASK_ID = 28
EOS_ID = 29
BLOCK_LENGTH = 8


class FakeTokenArray:
    def __init__(self, data):
        self.data = data


def make_decoder(kind):
    kwargs = dict(
        temperature=0,
        threshold=0.9,
        mask_id=MASK_ID,
        eos_id=EOS_ID,
    )
    if kind == "credit":
        return CreditThresholdParallelDecoder(**kwargs)
    return ThresholdParallelDecoder(**kwargs)


@pytest.mark.parametrize("kind", ["threshold", "credit"])
def test_new_predicate_equivalent_for_20_decoders(kind):
    torch.manual_seed(1234)
    B, T = 6, 40
    for trial in range(50):
        decoder = make_decoder(kind)
        data = torch.randint(0, VOCAB - 4, (B, T))
        # Random mask patterns, including fully-resolved rows.
        block_start = torch.randint(0, (T - BLOCK_LENGTH) // 8 + 1, (B,)) * 8
        for b in range(B):
            if trial % 5 == 0 and b == 0:
                continue  # leave row 0 fully resolved sometimes
            n_masks = int(torch.randint(0, BLOCK_LENGTH + 1, (1,)))
            pos = torch.randperm(BLOCK_LENGTH)[:n_masks]
            data[b, block_start[b] + pos] = MASK_ID
        x = FakeTokenArray(data.clone())
        logits = torch.randn(B, BLOCK_LENGTH, VOCAB)

        decoding_block = gather_blocks(x.data, block_start, BLOCK_LENGTH)
        old_predicate = (decoding_block == MASK_ID).sum(dim=1) == 0

        decoder.batch_decode(logits, block_start, x, BLOCK_LENGTH)

        after = gather_blocks(x.data, block_start, BLOCK_LENGTH)
        had_mask = (decoding_block == MASK_ID).any(dim=1)
        changed = (after != decoding_block).any(dim=1)
        new_predicate = (~had_mask) & (~changed)

        # 2.0 decoders only write masked positions, so a mask-free block
        # cannot change...
        assert not (changed & ~had_mask).any()
        # ...which makes the new predicate identical to the old one.
        assert torch.equal(new_predicate, old_predicate)


@pytest.mark.parametrize("kind", ["threshold", "credit"])
def test_20_decoder_never_touches_resolved_positions(kind):
    torch.manual_seed(7)
    B, T = 4, 24
    decoder = make_decoder(kind)
    data = torch.randint(0, VOCAB - 4, (B, T))
    block_start = torch.zeros(B, dtype=torch.long)
    data[:, 1] = MASK_ID
    x = FakeTokenArray(data.clone())
    logits = torch.randn(B, BLOCK_LENGTH, VOCAB)
    decoder.batch_decode(logits, block_start, x, BLOCK_LENGTH)
    resolved = data[:, :BLOCK_LENGTH] != MASK_ID
    assert torch.equal(
        x.data[:, :BLOCK_LENGTH][resolved], data[:, :BLOCK_LENGTH][resolved]
    )
