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
    Fused operators for activation layers.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig

from fluxserve.backend.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from fluxserve.backend.layers.quantization.base_config import QuantizationConfig
from fluxserve.backend.utils.runtime_utils import (
    is_cuda,
    resolve_obj_by_qualname,
    set_weight_attrs,
    CustomOp
)

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

logger = logging.getLogger(__name__)


class SiluAndMul(CustomOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        silu_and_mul(x, out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _is_cuda and "silu_and_mul" in globals():
            return self.forward_cuda(x)
        return self.forward_native(x)


class GeluAndMul(CustomOp):
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.approximate = approximate

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        if self.approximate == "tanh":
            gelu_tanh_and_mul(x, out)
        elif self.approximate == "none":
            gelu_and_mul(x, out)
        else:
            raise RuntimeError("GeluAndMul only support tanh or none")
        return out

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _is_cuda and "gelu_and_mul" in globals():
            return self.forward_cuda(x)
        return self.forward_native(x)


class NewGELU(CustomOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cuda(x) if _is_cuda else self.forward_native(x)


class ReLU2(nn.Module):
    """
    Applies the squared Rectified Linear Unit function.
    y = max(0, x)^2
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        return x * x


class QuickGELU(CustomOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cuda(x) if _is_cuda else self.forward_native(x)


class XIELU(CustomOp):
    """
    Applies the xIELU activation function introduced in https://arxiv.org/abs/2411.13010
    If the user has installed the nickjbrowning/XIELU, we import xIELU CUDA
    Otherwise, we emit a single warning and use xIELU Python
    """

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
        dtype: torch.dtype = torch.bfloat16,
        with_vector_loads: bool = False,
    ):
        super().__init__()
        self.alpha_p = nn.Parameter(
            torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=dtype)) - 1).unsqueeze(
                0
            )
        )
        self.alpha_n = nn.Parameter(
            torch.log(
                torch.exp(torch.tensor(alpha_n_init - beta, dtype=dtype)) - 1
            ).unsqueeze(0)
        )
        self.register_buffer("beta", torch.tensor(beta, dtype=dtype))
        self.register_buffer("eps", torch.tensor(eps, dtype=dtype))
        self.with_vector_loads = with_vector_loads
        # Temporary until xIELU CUDA fully implemented
        self._beta_scalar = float(self.beta.detach().cpu().float().item())
        self._eps_scalar = float(self.eps.detach().cpu().float().item())

        self._xielu_cuda_obj = None
        try:
            import xielu.ops  # noqa: F401

            self._xielu_cuda_obj = torch.classes.xielu.XIELU()
            msg = "Using experimental xIELU CUDA."
            try:
                from torch._dynamo import allow_in_graph

                self._xielu_cuda_fn = allow_in_graph(self._xielu_cuda)
                msg += " Enabled torch._dynamo for xIELU CUDA."
            except Exception as err:
                msg += (
                    f" Could not enable torch._dynamo for xIELU ({err}) - "
                    "this may result in slower performance."
                )
                self._xielu_cuda_fn = self._xielu_cuda
            logger.warning_once(msg)
        except Exception as err:
            pass
            # logger.warning_once(
            #     "CUDA-fused xIELU not available (%s) –"
            #     " falling back to a Python version.\n"
            #     "For CUDA xIELU (experimental), `pip install git+https://github.com/nickjbrowning/XIELU`",
            #     str(err),
            # )

    def _xielu_python(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = nn.functional.softplus(self.alpha_p)
        alpha_n = self.beta + nn.functional.softplus(self.alpha_n)
        return torch.where(
            x > 0,
            alpha_p * x * x + self.beta * x,
            (torch.expm1(torch.min(x, self.eps)) - x) * alpha_n + self.beta * x,
        )

    def _xielu_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Firewall function to prevent torch.compile from seeing .item()"""
        assert self._xielu_cuda_obj is not None, "XIELU CUDA object must not be None"
        original_shape = x.shape
        # CUDA kernel expects 3D tensors, reshape if needed
        while x.dim() < 3:
            x = x.unsqueeze(0)
        if x.dim() > 3:
            x = x.view(-1, 1, x.size(-1))
        if original_shape != x.shape:
            logger.warning_once(
                "Warning: xIELU input tensor expects 3 dimensions"
                " but got (shape: %s). Reshaping to (shape: %s).\n"
                "Note: For SGLang this may be expected if sending"
                "[B*S,D] instead of [B,S,D].",
                original_shape,
                x.shape,
            )
        result = self._xielu_cuda_obj.forward(
            x,
            self.alpha_p,
            self.alpha_n,
            # Temporary until xIELU CUDA fully implemented -> self.{beta,eps}.item()
            self._beta_scalar,
            self._eps_scalar,
            self.with_vector_loads,
        )
        return result.view(original_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._xielu_cuda_obj is not None and input.is_cuda:
            if not torch._dynamo.is_compiling():
                return self._xielu_cuda_fn(input)
            else:
                logger.warning_once(
                    "torch._dynamo is compiling, using Python version of xIELU."
                )
        return self._xielu_python(input)


class ScaledActivation(nn.Module):
    """An activation function with post-scale parameters.

    This is used for some quantization methods like AWQ.
    """

    def __init__(
        self,
        act_module: nn.Module,
        intermediate_size: int,
        input_is_parallel: bool = True,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.act = act_module
        self.input_is_parallel = input_is_parallel
        if input_is_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            intermediate_size_per_partition = divide(intermediate_size, tp_size)
        else:
            intermediate_size_per_partition = intermediate_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.scales = nn.Parameter(
            torch.empty(intermediate_size_per_partition, dtype=params_dtype)
        )
        set_weight_attrs(self.scales, {"weight_loader": self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) / self.scales

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if self.input_is_parallel:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = param_data.shape[0]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "gelu_new": NewGELU(),
    "relu2": ReLU2(),
    "xielu": XIELU(),
}


def get_act_fn(
    act_fn_name: str,
    quant_config: Optional[QuantizationConfig] = None,
    intermediate_size: Optional[int] = None,
    input_is_parallel: bool = True,
    params_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(f"Activation function {act_fn_name!r} is not supported.")

    act_fn = _ACTIVATION_REGISTRY[act_fn_name]
    if quant_config is not None and act_fn_name in quant_config.get_scaled_act_names():
        if intermediate_size is None:
            raise ValueError(
                "intermediate_size must be specified for scaled "
                "activation functions."
            )
        return ScaledActivation(
            act_fn, intermediate_size, input_is_parallel, params_dtype
        )
    return act_fn


def get_cross_encoder_activation_function(config: PretrainedConfig):
    if (
        hasattr(config, "sbert_ce_default_activation_function")
        and config.sbert_ce_default_activation_function is not None
    ):

        function_name = config.sbert_ce_default_activation_function
        assert function_name.startswith("torch.nn.modules."), (
            "Loading of activation functions is restricted to "
            "torch.nn.modules for security reasons"
        )
        return resolve_obj_by_qualname(function_name)()
    else:
        # adapt bge-reranker
        return nn.Identity()


if not _is_cuda:
    logger.info("Using Native Activation.")
