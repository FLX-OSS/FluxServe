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

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fluxserve.backend.utils.runtime_utils import CustomOp

from fluxserve.backend.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from fluxserve.backend.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from fluxserve.backend.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from fluxserve.backend.utils.runtime_utils import (
    get_bool_env_var,
    is_cuda,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from fluxserve.backend.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


has_triton_kernels = (
    Path(__file__).parents[1]
    / "moe"
    / "fused_moe_triton"
    / "triton_kernels_moe.py"
).exists()


_is_cuda = is_cuda()
_debug_moe = get_bool_env_var("FLUXSERVE_DEBUG_MOE")
_debug_moe_create_printed = False
_debug_moe_forward_printed = False


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        return F.linear(x, layer.weight, bias)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, use_triton_kernels: bool = False):
        super().__init__()
        self.use_triton_kernels = use_triton_kernels and has_triton_kernels
        self.with_bias = False

        self.triton_kernel_moe_forward = None
        self.triton_kernel_moe_with_bias_forward = None
        if torch.cuda.is_available() and has_triton_kernels:
            from fluxserve.backend.layers.moe.fused_moe_triton.triton_kernels_moe import (
                triton_kernel_moe_forward as _tk_forward,
            )
            from fluxserve.backend.layers.moe.fused_moe_triton.triton_kernels_moe import (
                triton_kernel_moe_with_bias_forward as _tk_with_bias_forward,
            )

            self.triton_kernel_moe_forward = _tk_forward
            self.triton_kernel_moe_with_bias_forward = _tk_with_bias_forward

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_weight_n, w13_weight_k = 2 * intermediate_size_per_partition, hidden_size
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        del layer
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)
        global _debug_moe_create_printed
        if _debug_moe and not _debug_moe_create_printed:
            _debug_moe_create_printed = True
            print(
                "[moe_debug] "
                f"pid={os.getpid()} event=create_moe_runner "
                f"backend={MoeRunnerBackend.TRITON.value} "
                f"use_triton_kernels={self.use_triton_kernels} "
                f"has_triton_kernels={has_triton_kernels} "
                f"runner={self.runner.__class__.__name__} "
                f"fused_func={getattr(self.runner, 'fused_func', None) is not None} "
                f"inplace={moe_runner_config.inplace} "
                f"top_k={moe_runner_config.top_k}"
            )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from fluxserve.backend.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        if self.use_triton_kernels:
            if self.with_bias:
                assert self.triton_kernel_moe_with_bias_forward is not None
                output = self.triton_kernel_moe_with_bias_forward(
                    hidden_states=x,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    b1=layer.w13_weight_bias,
                    b2=layer.w2_weight_bias,
                    topk_output=topk_output,
                    moe_runner_config=moe_runner_config,
                    w1_pcg=None,
                    w2_pcg=None,
                )
            else:
                assert self.triton_kernel_moe_forward is not None
                output = self.triton_kernel_moe_forward(
                    hidden_states=x,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_output=topk_output,
                    moe_runner_config=moe_runner_config,
                )
            return StandardCombineInput(hidden_states=output)
        else:
            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                b13=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
    def forward_native(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from fluxserve.backend.layers.moe.fused_moe_native import moe_forward_native
        from fluxserve.backend.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config
        assert moe_runner_config.activation == "silu", (
            f"activation = {moe_runner_config.activation} is not supported."
        )

        output = moe_forward_native(
            layer,
            x,
            topk_output,
            moe_runner_config,
        )
        return StandardCombineInput(hidden_states=output)

    def forward(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        global _debug_moe_forward_printed
        path = None
        if _is_cuda:
            path = "cuda_triton_runner"
            if _debug_moe and not _debug_moe_forward_printed:
                _debug_moe_forward_printed = True
                print(
                    "[moe_debug] "
                    f"pid={os.getpid()} event=forward path={path} "
                    f"use_triton_kernels={self.use_triton_kernels} "
                    f"has_triton_kernels={has_triton_kernels} "
                    f"runner={getattr(self, 'runner', None).__class__.__name__} "
                    f"fused_func={getattr(getattr(self, 'runner', None), 'fused_func', None) is not None}"
                )
            return self.forward_cuda(layer, dispatch_output)
        if self.use_triton_kernels:
            path = "triton_kernel"
            if _debug_moe and not _debug_moe_forward_printed:
                _debug_moe_forward_printed = True
                print(
                    "[moe_debug] "
                    f"pid={os.getpid()} event=forward path={path} "
                    f"use_triton_kernels={self.use_triton_kernels} "
                    f"has_triton_kernels={has_triton_kernels} "
                    f"runner={getattr(self, 'runner', None).__class__.__name__} "
                    f"fused_func={getattr(getattr(self, 'runner', None), 'fused_func', None) is not None}"
                )
            return self.forward_cuda(layer, dispatch_output)
        path = "native"
        if _debug_moe and not _debug_moe_forward_printed:
            _debug_moe_forward_printed = True
            print(
                "[moe_debug] "
                f"pid={os.getpid()} event=forward path={path} "
                f"use_triton_kernels={self.use_triton_kernels} "
                f"has_triton_kernels={has_triton_kernels} "
                f"runner={getattr(self, 'runner', None).__class__.__name__} "
                f"fused_func={getattr(getattr(self, 'runner', None), 'fused_func', None) is not None}"
            )
        return self.forward_native(layer, dispatch_output)
