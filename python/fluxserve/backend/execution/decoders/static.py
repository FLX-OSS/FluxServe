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

from fluxserve.backend.execution.decoders.base import ParallelDecoder
from fluxserve.backend.execution.decoders.utils import broadcast_if_needed, get_num_transfer_tokens, get_transfer_index

class StaticParallelDecoder(ParallelDecoder):
    """ 
        Decode tokens in a fixed number of steps (static).
    """
    def __init__(
            self,
            temperature,
            steps,
            remasking='low_confidence',
            mask_id=126336,
    ):
        super().__init__(temperature, remasking, mask_id)
        self.steps = steps
        self.iter = 0
        self.mask_id = mask_id

    def block_init(self, block_x, block_id):
        block_mask_index = block_x == self.mask_id
        self.num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, self.steps
        )
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in a block."""
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index(
            logits,
            self.temperature,
            self.remasking,
            mask_index,
            curr_x,
            self.num_transfer_tokens[:, self.iter],
            None,
        )
        self.iter += 1
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        broadcast_if_needed(x.data)
