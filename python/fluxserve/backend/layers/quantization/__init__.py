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

from fluxserve.backend.layers.quantization.base_config import QuantizationConfig

QUANTIZATION_METHODS = ("fp8", "modelopt_fp8", "modelopt_fp4")

def get_quantization_config(quantization: str):
    if quantization == "fp8":
        from fluxserve.backend.layers.quantization.fp8 import Fp8Config

        return Fp8Config
    if quantization in ("modelopt_fp8", "modelopt_fp4"):
        from fluxserve.backend.layers.quantization.modelopt_quant import (
            ModelOptFp4Config,
            ModelOptFp8Config,
        )

        return {
            "modelopt_fp8": ModelOptFp8Config,
            "modelopt_fp4": ModelOptFp4Config,
        }[quantization]
    else:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS)}"
        )


def monkey_patch_isinstance_for_vllm_base_layer(reverse: bool = False):
    del reverse
