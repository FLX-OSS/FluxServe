# Adapted from https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/python/dinfer/decoding/utils.py
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

import copy

import torch


class TokenArray:
    """ A token array to support read, update and expansion.

    We need to access the tokens that have been generated and write new tokens to the array.
    Some algorithms require to expand the token array.

    Parameters
    ----------
    prompt : Torch.Tensor
        The array that contains the input prompt.
    gen_length : int
        The number of tokens to be generated.
    mask_id : int
        the mask id of the masked tokens
    device : Torch.Device
        The device where the token array is placed on.
    """
    def __init__(self, prompt, gen_length, mask_id, eos_id, device):
        self.prompt = prompt.to(device)
        self.data = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
        self.data[:, :prompt.shape[1]] = prompt.clone()
        self.gen_length = gen_length
        self.eos_id = eos_id
        self.mask_id = mask_id

    @property
    def total_length(self):
        return self.prompt.shape[1] + self.gen_length

    @property
    def batch_size(self):
        return self.prompt.shape[0]

    @property
    def device(self):
        return self.data.device

    def expand(self, new_len):
        pass

    def get_generated_tokens(self):
        if self.batch_size == 1:
            self.data[self.data == self.mask_id] = self.eos_id
            return self.data[self.data != self.eos_id].unsqueeze(0)
        else:
            self.data[self.data == self.mask_id] = self.eos_id
            return self.data

    def select_seqs(self, idx):
        arr = copy.copy(self)
        arr.prompt = self.prompt[idx]
        arr.data = self.data[idx]
        return arr

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, vals):
        self.data[idx] = vals

class KVCache:
    """ The KV-cache

    Parameters
    ----------
    past_key_values : List[torch.Tensor]
        The keys and values of each transformer layer.
    """
    def __init__(self, past_key_values, backend='vllm', length=2048, cache_align_size=128):
        if backend == 'vllm':
            assert len(past_key_values) % 2 == 0
            self._data = past_key_values
        else:
            self.cache_align_size = cache_align_size
            assert len(past_key_values) % 2 == 0
            self._raw_data = past_key_values
            self._consolidate_raw()

            n = -(-(self._raw_data.shape[4] + 64) // self.cache_align_size)
            next_pow2 = 1 << (n - 1).bit_length() if n > 1 else 1
            self.length = next_pow2 * self.cache_align_size

            device = self._raw_data.device
            num_layer, _, batch_size, num_heads, seq_len, hidden_dim = self._raw_data.shape
            self._data = torch.zeros(num_layer, 2, batch_size, num_heads, self.length, hidden_dim, device=device, dtype=torch.bfloat16)
            self._data[:, :, :, :, :seq_len] = self._raw_data

    def consolidate(self):
        if isinstance(self._data, torch.Tensor):
            return

        num_layers = len(self._data) // 2
        inner_shape = self._data[0].shape
        # The shape is [num_layers, 2, batch_size, num_heads, seq_len, hidden_dim]
        self._data = torch.stack(self._data, dim=0).reshape(num_layers, 2, *inner_shape)

    def _consolidate_raw(self):
        if isinstance(self._raw_data, torch.Tensor):
            return

        num_layers = len(self._raw_data) // 2
        inner_shape = self._raw_data[0].shape
        # The shape is [num_layers, 2, batch_size, num_heads, seq_len, hidden_dim]
        self._raw_data = torch.stack(self._raw_data, dim=0).reshape(num_layers, 2, *inner_shape)

    @property
    def num_layers(self):
        assert isinstance(self._data, torch.Tensor)
        return self._data.shape[0]

    @property
    def seq_len(self):
        assert isinstance(self._data, torch.Tensor)
        return self._data.shape[4]

    def get_keys(self, layer_idx):
        """ Get the keys of a transformer layer.
        """
        assert isinstance(self._data, torch.Tensor)
        return self._data[layer_idx][0]

    def get_values(self, layer_idx):
        """ Get the values of a transformer layer.
        """
        assert isinstance(self._data, torch.Tensor)
        return self._data[layer_idx][1]

    def update(self, key_states, val_states, layer_idx, replace_position=None, backend='vllm'):
        """ Update the keys and values of a transformer layer.

        Parameters
        ----------
        key_states : torch.Tensor
            The keys in a block of tokens. The shape is [batch_size, num_heads, seq_len, hidden_dim]
        val_states : torch.Tensor
            The values in a block of tokens. The shape is [batch_size, num_heads, seq_len, hidden_dim]
        layer_idx : int
            The index of the transformer layer
        replace_position : Tuple[int]
            The start and the end position where keys and values should be updated.

        Returns
        -------
        torch.Tensor: the new keys for the entire sequence of the transformer layer.
        torch.Tensor: the new values for the entire sequence of the transformer layer.
        """
        if backend ==  'vllm':
            # This is dual cache.
            if replace_position is not None:
                keys = self.get_keys(layer_idx).slice_scatter(key_states, dim=2, start=replace_position[0], end=replace_position[1])
                values = self.get_values(layer_idx).slice_scatter(val_states, dim=2, start=replace_position[0], end=replace_position[1])
            else:
                # This is prefix cache.
                keys = torch.cat([self.get_keys(layer_idx), key_states], dim=2)
                values = torch.cat([self.get_values(layer_idx), val_states], dim=2)
        else:
            cache_length = self.get_keys(layer_idx).shape[2]
            block_length = key_states.shape[2]
            keys = self.get_keys(layer_idx).slice_scatter(key_states, dim=2, start=cache_length - block_length, end=cache_length)
            values = self.get_values(layer_idx).slice_scatter(val_states, dim=2, start=cache_length - block_length, end=cache_length)
        return keys, values


__all__ = [
    "KVCache",
    "TokenArray",
]
