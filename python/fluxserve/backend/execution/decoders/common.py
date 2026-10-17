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

"""
    Helpers shared by decoder families.
"""

import torch.distributed as dist


def broadcast_if_needed(x, src=0, group=None):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.broadcast(x, src=src)


def normalize_eos_ids(value):
    """Deduplicated tuple of EOS token ids from an int or an iterable of ints."""
    values = (value,) if isinstance(value, int) else tuple(value)
    eos_ids = tuple(dict.fromkeys(int(item) for item in values))
    if not eos_ids:
        raise ValueError("at least one EOS token id is required")
    return eos_ids


def resolve_checkpoint_eos_ids(model_name, trust_remote_code=True):
    """EOS ids declared by the checkpoint's generation_config.json, as a
    deduplicated tuple; empty tuple when the checkpoint declares none (the
    decoder's built-in default then applies). LLaDA2.2 declares two stop
    tokens ([156892, 156900]); 2.0/2.1 ship no generation_config.json.
    """
    from transformers import GenerationConfig

    try:
        gen_config = GenerationConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    except OSError:
        return ()
    value = getattr(gen_config, "eos_token_id", None)
    if value is None:
        return ()
    return normalize_eos_ids(value)

