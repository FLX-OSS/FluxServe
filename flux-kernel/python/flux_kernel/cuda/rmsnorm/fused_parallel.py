"""Optional CUDA RMSNorm wrapper.

The copied CUDA source lives next to this file under `csrc/`. Loading requires a
prebuilt TVM-FFI shared library at `objs/rmsnorm_fused_parallel.so`.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from flux_kernel.cuda.rmsnorm.build import (
    SO_PATH,
    build_rmsnorm_fused_parallel,
)


def _shared_library_path() -> Path:
    return SO_PATH


def has_rmsnorm_fused_parallel() -> bool:
    return _shared_library_path().exists()


def ensure_rmsnorm_fused_parallel_built(
    *,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    so_path = build_rmsnorm_fused_parallel(force=force, verbose=verbose)
    _load_rmsnorm_module.cache_clear()
    return so_path


@functools.cache
def _load_rmsnorm_module():
    import tvm_ffi

    so_path = _shared_library_path()
    if not so_path.exists():
        raise RuntimeError(
            f"rmsnorm library not found at {so_path}. "
            "Build the CUDA sources under csrc/ before using this optional path."
        )
    return tvm_ffi.load_module(str(so_path))


def rmsnorm_fused_parallel(
    input1: torch.Tensor,
    weight1: torch.Tensor,
    output1: torch.Tensor,
    input2: torch.Tensor,
    weight2: torch.Tensor,
    output2: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
) -> None:
    _load_rmsnorm_module().rmsnorm_fused_parallel(
        input1,
        weight1,
        output1,
        input2,
        weight2,
        output2,
        float(eps),
        bool(enable_pdl),
    )


__all__ = [
    "ensure_rmsnorm_fused_parallel_built",
    "has_rmsnorm_fused_parallel",
    "rmsnorm_fused_parallel",
]
