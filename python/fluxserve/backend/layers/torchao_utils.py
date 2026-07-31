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
Common utilities for torchao.
"""

import logging
import os
import pwd
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


def get_gemlite_cache_path() -> str:
    return f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"


def save_gemlite_cache(print_error: bool = False) -> bool:
    try:
        from gemlite.core import GemLiteLinearTriton

        GemLiteLinearTriton.cache_config(get_gemlite_cache_path())
    except Exception:
        if print_error:
            logger.error("Failed to save the GemLite cache.")
        return False
    return True


def proj_filter(
    module: torch.nn.Module,
    fqn: str,
):
    """Filter function for quantizing projection layers."""
    return "proj" in fqn


def apply_torchao_config_to_model(
    model: torch.nn.Module,
    torchao_config: str,
    filter_fn: Optional[Callable] = proj_filter,
):
    """Quantize a modelwith torchao quantization specified by torchao_config

    Args:
       `model`: a model to be quantized based on torchao_config
       `torchao_config` (str): type of quantization and their arguments we want to use to
        quantize the model, e.g. int4wo-128 means int4 weight only quantization with group_size
        128
    """
    # Lazy import to suppress some warnings
    from torchao.quantization import (
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )

    if torchao_config == "" or torchao_config is None:
        return model
    elif "int8wo" in torchao_config:
        quantize_(model, int8_weight_only(), filter_fn=filter_fn)
    elif "int8dq" in torchao_config:
        quantize_(model, int8_dynamic_activation_int8_weight(), filter_fn=filter_fn)
    elif "int4wo" in torchao_config:
        group_size = int(torchao_config.split("-")[-1])
        assert group_size in [
            32,
            64,
            128,
            256,
        ], f"int4wo groupsize needs to be one of [32, 64, 128, 256] but got {group_size}"
        quantize_(model, int4_weight_only(group_size=group_size), filter_fn=filter_fn)
    elif "gemlite" in torchao_config:
        # gemlite-<packing_bitwidth>-<bit_width>-<group_size> or
        # gemlite-<bit_width>-<group_size> (packing_bitwidth defaults to 32)
        from gemlite.core import GemLiteLinearTriton
        from torchao.quantization import gemlite_uintx_weight_only

        _quant_args = torchao_config.split("-")
        bit_width = int(_quant_args[-2])
        group_size = None if _quant_args[-1] == "None" else int(_quant_args[-1])

        try:
            packing_bitwidth = int(_quant_args[-3])
        except (ValueError, IndexError):
            # if only 2 inputs found or conversion fails, use default value
            packing_bitwidth = 32

        quantize_(
            model, gemlite_uintx_weight_only(group_size, bit_width, packing_bitwidth)
        )

        # try to load gemlite kernel config
        GemLiteLinearTriton.load_config(get_gemlite_cache_path())

    else:
        raise ValueError(f"Unexpected config: {torchao_config}")

    return model
