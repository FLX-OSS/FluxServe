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

import logging
import os
import signal
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fluxserve.backend import distributed
from fluxserve.backend.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)

_SHUTDOWN_GRACE_PERIOD_S = 3.0
_TERMINATE_GRACE_PERIOD_S = 5.0
_SUPERVISOR_POLL_INTERVAL_S = 0.1


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"
    init_method: str = "env://"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def reject_external_distributed_launch(
    environ: dict[str, str] | None = None,
) -> None:
    env = os.environ if environ is None else environ
    world_size = _read_int_env(env, "WORLD_SIZE", default=1)
    if world_size > 1:
        raise RuntimeError(
            "External distributed launch is not supported. Run `fluxserve serve` "
            "directly and use --tp-size/--ep-size to launch local GPU workers."
        )


def should_launch_local_workers(world_size: int) -> bool:
    return int(world_size) > 1


def validate_local_launch_config(
    *,
    tp_size: int,
    ep_size: int,
    dp_size: int,
    pp_size: int,
    enable_dp_attention: bool,
    device: str,
    visible_device_count: int,
    world_size: int | None = None,
) -> None:
    effective_world_size = tp_size if world_size is None else world_size
    if effective_world_size < 1:
        raise ValueError(f"Internal launch requires tp_size >= 1, got {tp_size}.")
    is_dp_attention_topology = enable_dp_attention and dp_size > 1
    if is_dp_attention_topology:
        if dp_size != ep_size:
            raise ValueError("DP-attention launch requires dp_size == ep_size.")
        if effective_world_size != tp_size * dp_size:
            raise ValueError("DP-attention launch requires world_size == tp_size * dp_size.")
    elif tp_size != ep_size:
        raise ValueError(
            "Internal launch requires tp_size == ep_size; "
            f"got tp_size={tp_size}, ep_size={ep_size}."
        )
    if dp_size != 1 and not is_dp_attention_topology:
        raise ValueError("Internal launch requires dp_size == 1.")
    if pp_size != 1:
        raise ValueError("Internal launch requires pp_size == 1.")
    if enable_dp_attention and not is_dp_attention_topology:
        raise ValueError("Internal launch does not support enable_dp_attention.")
    device_str = str(device)
    if not (
        device_str == "cuda"
        or device_str.startswith("cuda:")
        or device_str.isdigit()
    ):
        raise ValueError(
            f"Internal launch requires a CUDA device, got {device_str!r}."
        )
    if visible_device_count < effective_world_size:
        raise ValueError(
            "Internal launch does not have enough visible CUDA devices; "
            f"requested {effective_world_size}, found {visible_device_count}. Set "
            "CUDA_VISIBLE_DEVICES to select the worker GPUs."
        )


def launch_local_workers(
    worker: Callable[..., None],
    args,
    *,
    world_size: int | None = None,
) -> None:
    """Spawn and supervise one local process per tensor-parallel rank."""
    requested_tp_size = int(args.tp_size)
    world_size = int(requested_tp_size if world_size is None else world_size)
    validate_local_launch_config(
        tp_size=requested_tp_size,
        ep_size=int(args.ep_size),
        dp_size=int(args.dp_size),
        pp_size=int(args.pp_size),
        enable_dp_attention=bool(args.enable_dp_attention),
        device=str(args.device),
        visible_device_count=torch.cuda.device_count(),
        world_size=world_size,
    )
    with tempfile.TemporaryDirectory(prefix="fluxserve-rdzv-") as rendezvous_dir:
        init_method = f"file://{os.path.join(rendezvous_dir, 'store')}"
        logger.info("launching %s local FluxServe GPU workers", world_size)
        process_context = mp.spawn(
            _local_worker_entry,
            args=(world_size, init_method, worker, args),
            nprocs=world_size,
            join=False,
        )
        _supervise_processes(process_context)


def _supervise_processes(process_context) -> None:
    previous_handlers = {}
    shutdown_requested_at = None
    force_shutdown = False

    def request_shutdown(signum, _frame):
        nonlocal shutdown_requested_at, force_shutdown
        if shutdown_requested_at is None:
            shutdown_requested_at = time.monotonic()
            logger.info(
                "received %s; waiting for FluxServe workers to shut down",
                signal.Signals(signum).name,
            )
            if signum == signal.SIGTERM:
                rank0_process = process_context.processes[0]
                if rank0_process.is_alive():
                    os.kill(rank0_process.pid, signal.SIGTERM)
        else:
            force_shutdown = True
            logger.warning("received another shutdown signal; stopping workers")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_shutdown)
        while not process_context.join(timeout=_SUPERVISOR_POLL_INTERVAL_S):
            if shutdown_requested_at is None:
                continue
            if force_shutdown:
                break
            grace_period_expired = (
                time.monotonic() - shutdown_requested_at
                >= _SHUTDOWN_GRACE_PERIOD_S
            )
            if grace_period_expired:
                break
    finally:
        _stop_remaining_processes(process_context.processes)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _stop_remaining_processes(processes) -> None:
    remaining = [process for process in processes if process.is_alive()]
    for process in remaining:
        process.terminate()
    for process in remaining:
        process.join(timeout=_TERMINATE_GRACE_PERIOD_S)
    for process in remaining:
        if process.is_alive():
            process.kill()
            process.join()


def _local_worker_entry(
    local_rank: int,
    world_size: int,
    init_method: str,
    worker: Callable[..., None],
    args,
) -> None:
    # The terminal sends SIGINT to the whole foreground process group. Only
    # rank 0 owns Uvicorn; other ranks must stay alive for its shutdown broadcast.
    if local_rank != 0:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.update(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    worker(args, init_method=init_method)


def get_distributed_context(
    *,
    backend: str = "nccl",
    init_method: str = "env://",
    environ: dict[str, str] | None = None,
) -> DistributedContext:
    env = os.environ if environ is None else environ
    world_size = _read_int_env(env, "WORLD_SIZE", default=1)
    if world_size <= 1:
        return DistributedContext(backend=backend, init_method=init_method)

    rank = _read_required_int_env(env, "RANK")
    local_rank = _read_required_int_env(env, "LOCAL_RANK")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE), got rank={rank}, world_size={world_size}")
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got {local_rank}")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
    )


def validate_distributed_config(server_args: ServerArgs, context: DistributedContext) -> None:
    if server_args.pp_size != 1:
        raise ValueError("Distributed launch v1 requires pp_size == 1.")
    is_dp_attention_topology = server_args.enable_dp_attention and server_args.dp_size > 1
    if server_args.dp_size != 1 and not is_dp_attention_topology:
        raise ValueError("Distributed launch v1 requires dp_size == 1.")
    if server_args.enable_dp_attention and not is_dp_attention_topology:
        raise ValueError("Distributed launch v1 does not support enable_dp_attention.")
    if is_dp_attention_topology and server_args.dp_size != server_args.ep_size:
        raise ValueError("DP-attention launch requires dp_size == ep_size.")
    if server_args.ep_size < 1:
        raise ValueError("Distributed launch v1 requires ep_size >= 1.")
    num_experts = getattr(server_args.model_config, "num_experts", None)
    if num_experts is not None and int(num_experts) % server_args.ep_size != 0:
        raise ValueError(
            "Distributed launch v1 requires model_config.num_experts to be divisible by ep_size; "
            f"got num_experts={num_experts}, ep_size={server_args.ep_size}."
        )
    if context.is_distributed:
        if server_args.tp_size != context.world_size:
            raise ValueError(
                "Distributed launch v1 requires tp_size == WORLD_SIZE; "
                f"got tp_size={server_args.tp_size}, WORLD_SIZE={context.world_size}."
            )
        if not is_dp_attention_topology and server_args.ep_size != context.world_size:
            raise ValueError(
                "Distributed launch v1 requires ep_size == WORLD_SIZE; "
                f"got ep_size={server_args.ep_size}, WORLD_SIZE={context.world_size}."
            )
    else:
        if server_args.tp_size != 1:
            raise ValueError("tp_size > 1 requires a distributed worker context.")
        if server_args.ep_size != 1:
            raise ValueError("ep_size > 1 requires a distributed worker context.")


def initialize_distributed(
    server_args: ServerArgs,
    *,
    backend: str = "nccl",
    init_method: str = "env://",
) -> DistributedContext:
    context = get_distributed_context(backend=backend, init_method=init_method)
    validate_distributed_config(server_args, context)
    if context.is_distributed:
        torch.cuda.set_device(context.local_rank)
        distributed.init_distributed_environment(
            world_size=context.world_size,
            rank=context.rank,
            init_method=context.init_method,
            backend=context.backend,
        )
    distributed.initialize_model_parallel(
        server_args.tp_size,
        server_args.ep_size,
        1,
        backend=context.backend,
    )
    logger.info(
        "distributed context rank=%s local_rank=%s world_size=%s backend=%s tp_size=%s ep_size=%s moe_tp_size=%s",
        context.rank,
        context.local_rank,
        context.world_size,
        context.backend,
        server_args.tp_size,
        server_args.ep_size,
        server_args.tp_size // server_args.ep_size,
    )
    return context


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _read_required_int_env(env, name: str) -> int:
    if name not in env:
        raise ValueError(f"{name} must be set when WORLD_SIZE > 1.")
    return _parse_int(name, env[name])


def _read_int_env(env, name: str, *, default: int) -> int:
    if name not in env:
        return default
    return _parse_int(name, env[name])


def _parse_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
