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

from typing import Optional

import torch
import torch.distributed as dist


class _Group:
    def __init__(
        self,
        group_ranks=None,
        local_rank: Optional[int] = None,
        backend: Optional[str] = None,
        **kwargs,
    ):
        del backend, kwargs
        self.group_ranks = group_ranks or [[0]]
        self.device_group: Optional[dist.ProcessGroup] = None
        self.cpu_group: Optional[dist.ProcessGroup] = None
        self.ca_comm: object = None
        self.unique_name = f"group_{id(self)}"

        if dist.is_available() and dist.is_initialized():
            global_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            global_rank = 0
            world_size = 1

        self.rank = global_rank
        self.local_rank = global_rank if local_rank is None else local_rank
        self.rank_in_group = self.local_rank
        self.world_size = world_size

        for ranks in self.group_ranks:
            if global_rank in ranks:
                self.world_size = len(ranks)
                if local_rank is None:
                    self.rank_in_group = ranks.index(global_rank)
                    self.local_rank = self.rank_in_group
                else:
                    self.rank_in_group = local_rank
                break
        self.is_first_rank = self.rank_in_group == 0
        self.is_last_rank = self.rank_in_group == self.world_size - 1

    def barrier(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.device_group)

    def all_gather_into_tensor(self, output_tensor, input_tensor):
        if self.world_size == 1:
            output_tensor.copy_(input_tensor)
            return output_tensor
        dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=self.device_group
        )
        return output_tensor

    def reduce_scatter_tensor(self, output_tensor, input_tensor):
        if self.world_size == 1:
            output_tensor.copy_(input_tensor)
            return output_tensor
        dist.reduce_scatter_tensor(
            output_tensor, input_tensor, group=self.device_group
        )
        return output_tensor

    def all_gather(self, input_tensor, output_tensor_list=None):
        if output_tensor_list is None:
            output_tensor_list = [
                torch.empty_like(input_tensor) for _ in range(self.world_size)
            ]
        if self.world_size == 1:
            output_tensor_list[0].copy_(input_tensor)
            return output_tensor_list
        dist.all_gather(output_tensor_list, input_tensor, group=self.device_group)
        return output_tensor_list


_TP_SIZE = 1
_TP_RANK = 0
_EP_SIZE = 1
_EP_RANK = 0
_TP_GROUP = _Group()
_PP_GROUP = _Group()

GroupCoordinator = _Group


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0
    return numerator // denominator


def get_tensor_model_parallel_world_size():
    return _TP_SIZE


def get_tensor_model_parallel_rank():
    return _TP_RANK


def get_tensor_model_parallel_group():
    return _TP_GROUP.device_group


def get_tensor_model_parallel_cpu_group():
    return None


def get_tp_group():
    return _TP_GROUP


def get_pp_group():
    return _PP_GROUP


def get_moe_expert_parallel_world_size():
    return _EP_SIZE


def get_moe_expert_parallel_rank():
    return _EP_RANK


def get_moe_tensor_parallel_world_size():
    if _EP_SIZE <= 1:
        return _TP_SIZE
    return max(1, _TP_SIZE // _EP_SIZE)


def get_moe_tensor_parallel_rank():
    moe_tp_size = get_moe_tensor_parallel_world_size()
    if moe_tp_size <= 1:
        return 0
    return _TP_RANK // max(1, _EP_SIZE)


def get_attention_tp_size():
    return _TP_SIZE


def get_attention_tp_rank():
    return _TP_RANK


def get_attention_tp_group():
    return _TP_GROUP


def set_custom_all_reduce(enabled: bool):
    del enabled


def split_tensor_along_last_dim(tensor, num_partitions):
    return torch.chunk(tensor, num_partitions, dim=-1)


def tensor_model_parallel_all_gather(tensor):
    if _TP_SIZE == 1:
        return tensor
    output = [torch.empty_like(tensor) for _ in range(_TP_SIZE)]
    dist.all_gather(output, tensor, group=_TP_GROUP.device_group)
    return torch.cat(output, dim=-1)


def tensor_model_parallel_all_reduce(tensor):
    if _TP_SIZE > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, group=_TP_GROUP.device_group)
    return tensor


def tensor_model_parallel_reduce_scatter(tensor):
    if _TP_SIZE == 1:
        return tensor
    chunks = torch.chunk(tensor, _TP_SIZE, dim=0)
    out = torch.empty_like(chunks[0])
    dist.reduce_scatter(out, list(chunks), group=_TP_GROUP.device_group)
    return out


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: str = "nccl",
    **kwargs,
):
    del backend, kwargs
    global _TP_SIZE, _TP_RANK, _EP_SIZE, _EP_RANK
    global _TP_GROUP, _PP_GROUP
    _TP_SIZE = tensor_model_parallel_size
    _EP_SIZE = expert_model_parallel_size
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _TP_RANK = rank % max(1, _TP_SIZE)
        _EP_RANK = rank % max(1, _EP_SIZE)
        tp_groups = [
            list(range(head, min(head + _TP_SIZE, world_size)))
            for head in range(0, world_size, _TP_SIZE)
        ]
        _TP_GROUP = _Group(tp_groups, local_rank=_TP_RANK)
        for idx, ranks in enumerate(tp_groups):
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                _TP_GROUP.device_group = group
                _TP_GROUP.unique_name = f"tp_{idx}"
        if pipeline_model_parallel_size <= 1:
            _PP_GROUP = _Group([[rank]], local_rank=0)
        else:
            _PP_GROUP = _Group([list(range(world_size))], local_rank=rank // max(1, _TP_SIZE))
        _PP_GROUP.device_group = dist.group.WORLD
    else:
        _TP_RANK = 0
        _EP_RANK = 0
        _TP_GROUP = _Group([[0]], local_rank=0)
        _PP_GROUP = _Group([[0]], local_rank=0)


def init_distributed_environment(*args, **kwargs):
    if not dist.is_available() or dist.is_initialized():
        return

    world_size = kwargs.pop("world_size", args[0] if len(args) > 0 else 1)
    rank = kwargs.pop("rank", args[1] if len(args) > 1 else 0)
    init_method = kwargs.pop("init_method", args[2] if len(args) > 2 else "env://")
    backend = kwargs.pop("backend", args[4] if len(args) > 4 else "nccl")

    if backend != "nccl":
        raise RuntimeError(
            f"FluxServe backend supports NVIDIA CUDA/NCCL only, got distributed backend {backend!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("FluxServe backend requires an available NVIDIA CUDA GPU.")

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        **kwargs,
    )


# Compatibility alias used by extracted modules.
import fluxserve.backend.distributed.parallel_state as parallel_state  # noqa: E402
