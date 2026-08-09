# Copyright (c) 2026 FLUX-OSS

"""Shared builder for focused CUDA libraries."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, include_paths, library_paths


def build_cuda_library(
    root: Path,
    name: str,
    *,
    force: bool,
    verbose: bool,
    default_arches: tuple[str, ...] = ("90",),
) -> Path:
    source = root / "csrc" / f"{name}.cu"
    objects = root / "objs"
    obj = objects / f"{name}.o"
    library = objects / f"{name}.so"
    stamp = objects / f"{name}.build"
    arch_override = os.environ.get("FLUX_KERNEL_CUDA_ARCH")
    arches = tuple(arch_override.split(",")) if arch_override else default_arches
    if not arches or any(not arch.removesuffix("a").isdigit() for arch in arches):
        raise ValueError(
            "FLUX_KERNEL_CUDA_ARCH must be a comma-separated list such as 90,120a"
        )
    arch_signature = ",".join(arches)
    signature = f"torch={torch.__version__}\ncuda={torch.version.cuda}\narch={arch_signature}\nabi={int(torch._C._GLIBCXX_USE_CXX11_ABI)}\n"
    dependencies = [source, *root.glob("csrc/**/*.h"), *root.glob("csrc/**/*.cuh")]
    newest_dependency = max(path.stat().st_mtime for path in dependencies)
    if not force and library.exists() and stamp.exists() and stamp.read_text() == signature and library.stat().st_mtime > newest_dependency:
        return library
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME could not be resolved")
    objects.mkdir(parents=True, exist_ok=True)
    private_include = root / "csrc" / "include"
    bundled_include = Path(__file__).resolve().parent / "rmsnorm" / "csrc" / "include"
    includes = [f"-I{root / 'csrc'}"]
    if private_include.exists():
        includes.append(f"-I{private_include}")
    includes.append(f"-I{bundled_include}")
    try:
        import flashinfer

        flashinfer_data = Path(flashinfer.__file__).resolve().parent / "data"
        includes.append(f"-I{flashinfer_data / 'cutlass' / 'include'}")
    except ImportError:
        pass
    includes.extend(f"-I{path}" for path in include_paths(device_type="cuda"))
    libs = [f"-L{path}" for path in library_paths(device_type="cuda")]
    gencodes = [
        flag
        for arch in arches
        for flag in ("-gencode", f"arch=compute_{arch},code=sm_{arch}")
    ]
    compile_cmd = [str(Path(CUDA_HOME) / "bin" / "nvcc"), "-std=c++17", "-O3", "--expt-relaxed-constexpr", "--compiler-options=-fPIC", *gencodes, f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}", *includes, "-c", str(source), "-o", str(obj)]
    link_cmd = [os.environ.get("CXX", "g++"), str(obj), "-shared", *libs, "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda", "-lcudart", "-o", str(library)]
    if verbose:
        print(" ".join(compile_cmd)); print(" ".join(link_cmd))
    subprocess.check_call(compile_cmd); subprocess.check_call(link_cmd)
    stamp.write_text(signature, encoding="utf-8")
    return library
