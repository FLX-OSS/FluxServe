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

from contextlib import contextmanager
from typing import Any, Dict, Optional

from fluxserve.backend.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)


_config: Optional[Dict[str, Any]] = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> Optional[Dict[str, Any]]:
    return _config


def fused_experts(*args, **kwargs):
    from fluxserve.backend.layers.moe.fused_moe_triton.fused_moe import fused_experts as impl

    return impl(*args, **kwargs)


def get_config_file_name(*args, **kwargs):
    from fluxserve.backend.layers.moe.fused_moe_triton.fused_moe_triton_config import (
        get_config_file_name as impl,
    )

    return impl(*args, **kwargs)


def moe_align_block_size(*args, **kwargs):
    from fluxserve.backend.layers.moe.fused_moe_triton.moe_align_block_size import (
        moe_align_block_size as impl,
    )

    return impl(*args, **kwargs)


def try_get_optimal_moe_config(*args, **kwargs):
    from fluxserve.backend.layers.moe.fused_moe_triton.fused_moe_triton_config import (
        try_get_optimal_moe_config as impl,
    )

    return impl(*args, **kwargs)


__all__ = [
    "FusedMoE",
    "FusedMoeWeightScaleSupported",
    "override_config",
    "get_config",
    "fused_experts",
    "get_config_file_name",
    "moe_align_block_size",
    "try_get_optimal_moe_config",
]

