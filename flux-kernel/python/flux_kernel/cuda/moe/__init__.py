import functools
from pathlib import Path

import torch

from flux_kernel.cuda.build import build_cuda_library

ROOT = Path(__file__).resolve().parent


def build_moe(*, force: bool = False, verbose: bool = False) -> Path:
    return build_cuda_library(ROOT, "moe", force=force, verbose=verbose)


@functools.cache
def load_moe() -> None:
    torch.ops.load_library(str(build_moe()))


__all__ = ["build_moe", "load_moe"]
