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

import torch
import torch.distributed as dist

def calculate_op_num(x, hidden_size=4096, mlp_hidden_size = 12288, vocab_size = 126464, num_hidden_layers=32, cache_length=0):
    cfg_factor = 1
    qkv_ops = 4*x.shape[0]*hidden_size*hidden_size*x.shape[1]*2
    attn_ops = x.shape[0]*(cache_length)*x.shape[1]*hidden_size*2
    ffn_ops = 3*x.shape[0]*hidden_size*mlp_hidden_size*x.shape[1]*2
    layer_ops = qkv_ops + attn_ops + ffn_ops
    op_num = cfg_factor * (num_hidden_layers*layer_ops + x.shape[0]*hidden_size*vocab_size*x.shape[1]*2)
    return op_num/1e12 

def gather_sequence_block(partial_data, partial_start, partial_end, block_start, block_end, rank, world_size):
    """ Gather the wanted block data from the partitioned data.

    Each process contains a partition specified by `partial_start` and `partial_end`.
    The wanted block is located between `block_start` and `block_end`.

    We want to gather the data within the block range from the partitioned data.
    """
    if partial_start >= block_end or partial_end <= block_start:
        # there is no overlap, nothing is needed from partial_data
        arr = partial_data[:, 0:0]
    elif block_start >= partial_start and block_end <= partial_end:
        # the needed block is within partial_data.
        arr = partial_data[:, (block_start - partial_start):(block_end - partial_start)]
    elif block_start <= partial_start and block_end >= partial_end:
        # the needed partition is within the block.
        arr = partial_data
    elif partial_start >= block_start and partial_end >= block_end:
        # the needed block is overlapped in the front of partial_data
        arr = partial_data[:, 0:(block_end - partial_start)]
    else:
        # the needed block is overlapped at the end of partial_data
        arr = partial_data[:, (block_start - partial_start):(partial_end - partial_start)]
    arr = arr.contiguous()

    shape_list = [
            torch.zeros(len(arr.shape), dtype=torch.int64, device=partial_data.device) for _ in range(world_size)
    ]
    dist.all_gather(shape_list, torch.tensor(arr.shape, dtype=torch.int64, device=partial_data.device))
    part_list = [
            torch.zeros(*tuple(shape.tolist()), dtype=partial_data.dtype, device=partial_data.device) for shape in shape_list
    ]
    dist.all_gather(part_list, arr)
    return torch.cat(part_list, dim=1)
