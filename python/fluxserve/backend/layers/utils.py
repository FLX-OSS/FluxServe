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

import logging
import re
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def narrow_padded_param_and_loaded_weight(param, loaded_weight, *args, **kwargs):
    del args, kwargs
    return param, loaded_weight


def get_layer_id(weight_name):
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None


def pad_or_narrow_weight(
    loaded_weight: torch.Tensor, input_dim: int, start_idx: int, shard_size: int
) -> torch.Tensor:
    # Padding with zeros for special case such as qwen2_5_VL's mlp which is not 8-aligned
    valid_size = max(loaded_weight.shape[input_dim] - start_idx, 0)

    if valid_size > 0:
        loaded_slice = loaded_weight.narrow(input_dim, start_idx, valid_size)
        pad_shape = list(loaded_weight.shape)
        pad_shape[input_dim] = shard_size - valid_size
        pad = torch.zeros(
            pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
        )
        return torch.cat([loaded_slice, pad], dim=input_dim)

    # All padding
    pad_shape = list(loaded_weight.shape)
    pad_shape[input_dim] = shard_size
    return torch.zeros(
        pad_shape, dtype=loaded_weight.dtype, device=loaded_weight.device
    )


class PPMissingLayer(torch.nn.Identity):
    # Adapted from
    # https://github.com/vllm-project/vllm/blob/18ed3132d2bfe1df9a74729457b69243955221e8/vllm/model_executor/models/utils.py#L468C1-L486C1
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.return_tuple = kwargs.get("return_tuple", False)

    def forward(self, *args, **kwargs):
        """
        Return the first arg from args or the first value from kwargs.

        Wraps the input in a tuple if `self.return_tuple` is True.
        """
        input = args[0] if args else next(iter(kwargs.values()))
        return (input,) if self.return_tuple else input
