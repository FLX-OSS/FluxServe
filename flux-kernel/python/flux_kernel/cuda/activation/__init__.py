import functools
from pathlib import Path

import torch

from flux_kernel.cuda.build import build_cuda_library

ROOT = Path(__file__).resolve().parent


def build_activation(*, force: bool = False, verbose: bool = False) -> Path:
    return build_cuda_library(ROOT, "activation", force=force, verbose=verbose)


@functools.cache
def load_activation() -> None:
    torch.ops.load_library(str(build_activation()))


__all__ = ["build_activation", "load_activation"]
