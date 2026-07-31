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

import functools
import importlib
import os
import warnings

from packaging import version
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.library import Library


fluxserve_lib = Library("fluxserve", "FRAGMENT")


class CustomOp(torch.nn.Module):
    def enter_torch_compile(self, num_tokens: int = 0):
        del num_tokens

    def leave_torch_compile(self):
        pass


class SamplingBatchInfo:
    has_custom_logit_processor = False
    custom_logit_processor = None


class _Progress:
    @staticmethod
    def tqdm(iterable=None, *args, **kwargs):
        del args, kwargs
        return iterable if iterable is not None else []

    @staticmethod
    def trange(*args, **kwargs):
        del kwargs
        return range(*args)


try:
    import tqdm as tqdm_progress
except ImportError:
    tqdm_progress = _Progress()


def add_prefix(name: str, prefix: str) -> str:
    return name if not prefix else f"{prefix}.{name}"


def align(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def round_up(x: int, alignment: int) -> int:
    return align(x, alignment)


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def dump_to_file(*args, **kwargs):
    del args, kwargs


def crash_on_warnings() -> bool:
    return get_bool_env_var("SGLANG_CRASH_ON_WARNINGS")


def require_nvidia_cuda(device: Any = "cuda") -> None:
    device_str = str(device)
    if not (device_str == "cuda" or device_str.startswith("cuda:") or device_str.isdigit()):
        raise RuntimeError(
            f"FluxServe backend supports NVIDIA CUDA only, got device {device_str!r}."
        )
    if getattr(torch.version, "hip", None):
        raise RuntimeError("FluxServe backend supports NVIDIA CUDA only, not HIP/ROCm.")
    if not torch.cuda.is_available():
        raise RuntimeError("FluxServe backend requires an available NVIDIA CUDA GPU.")


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: List[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
):
    import torch.library

    my_lib = target_lib or fluxserve_lib
    lib_name = my_lib.m.name if hasattr(my_lib.m, "name") else "fluxserve"
    try:
        if hasattr(torch.ops, lib_name) and hasattr(getattr(torch.ops, lib_name), op_name):
            return
    except (AttributeError, RuntimeError):
        pass

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)

    try:
        my_lib.define(op_name + schema_str)
        my_lib.impl(op_name, op_func, "CUDA")
        if fake_impl is not None:
            my_lib._register_fake(op_name, fake_impl)
    except RuntimeError as error:
        if "Tried to register an operator" in str(error) and "multiple times" in str(error):
            return
        raise


def get_cuda_version() -> Tuple[int, int]:
    version = torch.version.cuda
    if not version:
        return (0, 0)
    parts = version.split(".")
    return (int(parts[0]), int(parts[1]))


def get_device_capability(device: Any = None) -> Tuple[int, int]:
    require_nvidia_cuda("cuda")
    return torch.cuda.get_device_capability(device)


def get_device_core_count(device: Any = None) -> int:
    del device
    return 0


def get_device_name(device: Any = None) -> str:
    require_nvidia_cuda("cuda")
    return torch.cuda.get_device_name(device)


def get_compiler_backend() -> str:
    return os.environ.get("SGLANG_TORCH_COMPILE_BACKEND", "inductor")


def get_bool_env_var(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def get_available_gpu_memory(device: str = "cuda", gpu_id: int = 0, empty_cache: bool = True) -> float:
    require_nvidia_cuda(device)
    if empty_cache:
        torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(gpu_id if device == "cuda" else device)
    return free / (1024**3)


def is_cpu() -> bool:
    return not torch.cuda.is_available()


def is_cuda() -> bool:
    return torch.cuda.is_available()


def is_cpu() -> bool:
    return not torch.cuda.is_available()


def is_hip() -> bool:
    return False


def is_npu() -> bool:
    return False


def is_xpu() -> bool:
    return False


def is_flashinfer_available() -> bool:
    try:
        return importlib.util.find_spec("flashinfer") is not None
    except (ImportError, ValueError):
        return False


def is_flashinfer_dllm_available() -> bool:
    try:
        return importlib.util.find_spec("flashinfer.dllm") is not None
    except (ImportError, ValueError):
        return False


def is_sm100_supported() -> bool:
    return get_device_capability() >= (10, 0)


def is_sm90_supported() -> bool:
    return get_device_capability() >= (9, 0)


def log_info_on_rank0(message: str):
    print(message)


def log_warning_once(message: str):
    warnings.warn(message, stacklevel=2)


def print_warning_once(message: str):
    warnings.warn(message, stacklevel=2)


def prepare_weight_cache(weight: torch.Tensor, *args, **kwargs):
    del args, kwargs
    return weight


def supports_custom_op() -> bool:
    return hasattr(torch, "ops")


def maybe_torch_compile(fn=None, **compile_kwargs):
    if fn is None:
        return lambda wrapped: maybe_torch_compile(wrapped, **compile_kwargs)

    if not hasattr(torch, "compile"):
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        compiled = torch.compile(fn, **compile_kwargs)
        return compiled(*args, **kwargs)

    return wrapped


def monkey_patch_torch_compile():
    if version.parse(torch.__version__) < version.parse("2.8.0"):
        # These things are cacheable by torch.compile. torch.compile just doesn't know it.
        # This was fixed in PyTorch 2.8, but until then, we monkey patch.
        import torch._higher_order_ops.auto_functionalize as af

        af.auto_functionalized_v2._cacheable = True
        af.auto_functionalized._cacheable = True



def resolve_obj_by_qualname(qualname: str):
    module_name, obj_name = qualname.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), obj_name)


def get_offloader():
    return None


def update_param(param, new_param):
    param.data = new_param


def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))


def inplace_all_reduce(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError("FluxServe inplace_all_reduce custom op has not been extracted.")


def inplace_fused_experts(*args, **kwargs):
    from fluxserve.backend.layers.moe.fused_moe_triton.fused_moe import (
        inplace_fused_experts as impl,
    )

    return impl(*args, **kwargs)


def outplace_fused_experts(*args, **kwargs):
    from fluxserve.backend.layers.moe.fused_moe_triton.fused_moe import (
        outplace_fused_experts as impl,
    )

    return impl(*args, **kwargs)


def is_non_idle_and_non_empty(forward_mode: Any, hidden_states: torch.Tensor) -> bool:
    del forward_mode
    return hidden_states is not None and hidden_states.numel() > 0


def make_layers(
    num_hidden_layers: int,
    layer_fn,
    pp_rank: int = 0,
    pp_size: int = 1,
    prefix: str = "",
):
    layers_per_rank = (num_hidden_layers + pp_size - 1) // pp_size
    start_layer = min(pp_rank * layers_per_rank, num_hidden_layers)
    end_layer = min(start_layer + layers_per_rank, num_hidden_layers)
    layers = torch.nn.ModuleList(
        [
            layer_fn(layer_id, add_prefix(str(layer_id), prefix))
            for layer_id in range(start_layer, end_layer)
        ]
    )
    return layers, start_layer, end_layer


def next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def set_weight_attrs(weight: torch.Tensor, attrs: dict):
    for key, value in attrs.items():
        setattr(weight, key, value)
