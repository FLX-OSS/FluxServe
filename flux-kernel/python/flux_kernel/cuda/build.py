# Copyright (c) 2026 FLUX-OSS

"""Shared builder for focused CUDA libraries."""

from __future__ import annotations

import os
import re
import subprocess
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, include_paths, library_paths


_ARCH_SPLIT_RE = re.compile(r"[,;\s]+")


def normalize_cuda_arch(arch: str) -> str:
    """Normalize Torch/NVCC architecture spellings to an NVCC suffix."""
    value = arch.strip().lower()
    if value.startswith(("sm_", "compute_")):
        value = value.split("_", 1)[1]
    value = value.removesuffix("+ptx")
    suffix = "a" if value.endswith("a") else ""
    value = value.removesuffix("a")
    if "." in value:
        parts = value.split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"invalid CUDA architecture: {arch!r}")
        value = f"{int(parts[0])}{int(parts[1])}"
    if not value.isdigit() or len(value) < 2:
        raise ValueError(f"invalid CUDA architecture: {arch!r}")
    return f"{value}{suffix}"


def parse_cuda_arch_list(value: str) -> tuple[str, ...]:
    """Parse a comma, semicolon, or whitespace separated architecture list."""
    result: list[str] = []
    for token in _ARCH_SPLIT_RE.split(value.strip()):
        if not token:
            continue
        arch = normalize_cuda_arch(token)
        if arch not in result:
            result.append(arch)
    if not result:
        raise ValueError("CUDA architecture list must not be empty")
    return tuple(result)


def nvcc_supported_arches(nvcc: str) -> frozenset[str]:
    """Return SASS architectures advertised by NVCC, or an empty set."""
    try:
        output = subprocess.check_output(
            [nvcc, "--list-gpu-code"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        return frozenset()
    arches = {
        normalize_cuda_arch(match)
        for match in re.findall(r"(?:sm_|compute_)([0-9]+a?)", output)
    }
    # NVCC lists architecture families (for example sm_100) but also accepts
    # their feature-specific targets (sm_100a).
    arches.update(
        f"{arch.removesuffix('a')}a"
        for arch in tuple(arches)
        if int(arch.removesuffix("a")) >= 90
    )
    return frozenset(arches)


def resolve_cuda_arches(
    default_arches: Sequence[str],
    *,
    nvcc: str,
    env: Mapping[str, str] | None = None,
    detected_capability: tuple[int, int] | None = None,
    supported_arches: frozenset[str] | None = None,
    legacy_env_names: Sequence[str] = (),
) -> tuple[str, ...]:
    """Resolve architectures consistently for every Flux Kernel library."""
    env = os.environ if env is None else env
    requested = env.get("FLUX_KERNEL_CUDA_ARCH", "").strip()
    explicit = bool(requested)
    if not requested:
        requested = env.get("TORCH_CUDA_ARCH_LIST", "").strip()
        explicit = bool(requested)
    if not requested:
        for name in legacy_env_names:
            requested = env.get(name, "").strip()
            if requested:
                warnings.warn(
                    f"{name} is deprecated; use FLUX_KERNEL_CUDA_ARCH instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                explicit = True
                break

    if requested:
        arches = parse_cuda_arch_list(requested)
    else:
        if detected_capability is None:
            try:
                if torch.cuda.is_available():
                    detected_capability = torch.cuda.get_device_capability()
            except Exception:
                detected_capability = None
        if detected_capability is not None:
            major, minor = detected_capability
            suffix = "a" if major >= 10 else ""
            arches = (normalize_cuda_arch(f"{major}.{minor}{suffix}"),)
        else:
            arches = tuple(normalize_cuda_arch(arch) for arch in default_arches)

    supported = nvcc_supported_arches(nvcc) if supported_arches is None else supported_arches
    if supported:
        unsupported = tuple(arch for arch in arches if arch not in supported)
        if unsupported and explicit:
            requested_names = ", ".join(f"sm_{arch}" for arch in unsupported)
            raise RuntimeError(
                f"NVCC at {nvcc} does not support requested architecture(s): "
                f"{requested_names}"
            )
        arches = tuple(arch for arch in arches if arch in supported)
    if not arches:
        raise RuntimeError(f"NVCC at {nvcc} supports none of the requested CUDA architectures")
    return arches


def build_cuda_library(
    root: Path,
    name: str,
    *,
    force: bool,
    verbose: bool,
    default_arches: tuple[str, ...] = ("90", "100a"),
) -> Path:
    source = root / "csrc" / f"{name}.cu"
    objects = root / "objs"
    obj = objects / f"{name}.o"
    library = objects / f"{name}.so"
    stamp = objects / f"{name}.build"
    nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc") if CUDA_HOME is not None else "nvcc"
    arches = resolve_cuda_arches(default_arches, nvcc=nvcc)
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
    compile_cmd = [nvcc, "-std=c++17", "-O3", "--expt-relaxed-constexpr", "--compiler-options=-fPIC", *gencodes, f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}", *includes, "-c", str(source), "-o", str(obj)]
    link_cmd = [os.environ.get("CXX", "g++"), str(obj), "-shared", *libs, "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda", "-lcudart", "-o", str(library)]
    if verbose:
        print(f"Building flux-kernel {name} for: {', '.join(f'sm_{arch}' for arch in arches)}")
        print(" ".join(compile_cmd)); print(" ".join(link_cmd))
    subprocess.check_call(compile_cmd); subprocess.check_call(link_cmd)
    stamp.write_text(signature, encoding="utf-8")
    return library
