import functools
from pathlib import Path

import torch

from flux_kernel.cuda.build import build_cuda_library

ROOT = Path(__file__).resolve().parent


def build_rope(*, force: bool = False, verbose: bool = False) -> Path:
    return build_cuda_library(
        ROOT,
        "rope",
        force=force,
        verbose=verbose,
        default_arches=("90", "100a", "120a"),
    )


@functools.cache
def load_rope() -> None:
    torch.ops.load_library(str(build_rope()))


__all__ = ["build_rope", "load_rope"]
