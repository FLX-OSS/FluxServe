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


"""Local builder for CUDA RMSNorm kernel."""

from __future__ import annotations

import importlib
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CSRC_DIR = ROOT / "csrc"
OBJS_DIR = ROOT / "objs"
SO_PATH = OBJS_DIR / "rmsnorm_fused_parallel.so"
STAMP_PATH = OBJS_DIR / "rmsnorm_fused_parallel.build"
SOURCES = [
    CSRC_DIR / "rmsnorm_fused_parallel.cu",
    CSRC_DIR / "flashinfer_rmsnorm_fused_parallel_binding.cu",
]
HEADER_ROOT = CSRC_DIR / "include"


def _cuda_home() -> Path:
    return Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))


def _nvcc() -> str:
    return os.environ.get("FLASHINFER_NVCC", str(_cuda_home() / "bin" / "nvcc"))


def _cxx() -> str:
    cxx = os.environ.get("CXX", "g++")
    if shutil.which(cxx) is not None or Path(cxx).exists():
        return cxx
    return "g++"


def _site_paths() -> list[Path]:
    paths: list[str] = []
    try:
        paths.extend(site.getsitepackages())
    except Exception:
        pass
    paths.extend(sys.path)

    unique = []
    seen = set()
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        path_str = str(path)
        if path.exists() and path_str not in seen:
            unique.append(path)
            seen.add(path_str)
    return unique


def _cuda_toolkit_roots() -> list[Path]:
    roots = [_cuda_home()]
    unique = []
    seen = set()
    for root in roots:
        root_str = str(root)
        if root.exists() and root_str not in seen:
            unique.append(root)
            seen.add(root_str)
    return unique


def _add_existing_dir(dirs: list[str], seen: set[str], path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in seen:
        dirs.append(path_str)
        seen.add(path_str)


def _resolve_include_dirs() -> list[str]:
    dirs = [str(HEADER_ROOT), str(CSRC_DIR)]
    seen = set(dirs)

    for cuda_root in _cuda_toolkit_roots():
        cuda_include = cuda_root / "include"
        if (cuda_include / "cuda_runtime.h").exists():
            _add_existing_dir(dirs, seen, cuda_include)
        if (cuda_include / "cccl").exists():
            _add_existing_dir(dirs, seen, cuda_include / "cccl")

    for base_path in _site_paths():
        for candidate in sorted(base_path.glob("nvidia/cu*/include"), reverse=True):
            if (candidate / "cuda_runtime.h").exists():
                _add_existing_dir(dirs, seen, candidate)
            if (candidate / "cccl").exists():
                _add_existing_dir(dirs, seen, candidate / "cccl")

    try:
        tvm_ffi = importlib.import_module("tvm_ffi")
        _add_existing_dir(dirs, seen, Path(tvm_ffi.__file__).parent / "include")
    except ImportError:
        pass

    try:
        flashinfer = importlib.import_module("flashinfer")
        fi_root = Path(flashinfer.__file__).parent / "data"
        for subdir in (
            fi_root / "csrc" / "nv_internal",
            fi_root / "csrc" / "nv_internal" / "include",
            fi_root / "include",
            fi_root / "cutlass" / "include",
        ):
            _add_existing_dir(dirs, seen, subdir)
        spdlog = fi_root / "spdlog" / "include"
        if (spdlog / "spdlog" / "spdlog.h").exists():
            _add_existing_dir(dirs, seen, spdlog)
            return dirs
    except ImportError:
        pass

    if (Path("/usr/include") / "spdlog" / "spdlog.h").exists():
        _add_existing_dir(dirs, seen, Path("/usr/include"))
    return dirs


def _resolve_cuda_lib_flags() -> list[str]:
    lib_candidates = []
    for cuda_root in _cuda_toolkit_roots():
        lib_candidates.extend([cuda_root / "lib64", cuda_root / "lib"])
    for base_path in _site_paths():
        lib_candidates.extend(sorted(base_path.glob("nvidia/cu*/lib"), reverse=True))

    unique = []
    seen = set()
    for candidate in lib_candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in seen:
            unique.append(candidate)
            seen.add(candidate_str)

    flags = [f"-L{path}" for path in unique]
    cuda_lib_dir = unique[0] if unique else _cuda_home() / "lib64"
    for candidate in unique:
        if (candidate / "libcudart.so").exists() or list(candidate.glob("libcudart.so.*")):
            cuda_lib_dir = candidate
            break

    stubs_dir = cuda_lib_dir / "stubs"
    if stubs_dir.exists():
        flags.append(f"-L{stubs_dir}")

    cudart_so = cuda_lib_dir / "libcudart.so"
    cudart_versioned = sorted(cuda_lib_dir.glob("libcudart.so.*"))
    if cudart_so.exists():
        flags.append("-lcudart")
    elif cudart_versioned:
        flags.append(f"-l:{cudart_versioned[-1].name}")
    else:
        flags.append("-lcudart")
    flags.append("-lcuda")
    return flags


def _normalize_cuda_arch(arch: str) -> str:
    has_suffix = arch.endswith("a")
    arch_clean = arch.rstrip("a")
    if "." in arch_clean:
        major_s, minor_s = arch_clean.split(".", 1)
        major = int(major_s)
        minor = int(minor_s)
    else:
        major = int(arch_clean[:-1])
        minor = int(arch_clean[-1])
    suffix = "a" if has_suffix or major >= 9 else ""
    return f"{major}{minor}{suffix}"


def _detect_cuda_archs() -> list[str]:
    arch_list = os.environ.get("FLASHINFER_CUDA_ARCH_LIST", "").strip()
    if arch_list:
        return sorted({_normalize_cuda_arch(arch) for arch in arch_list.split()})

    direct = (
        os.environ.get("FLUXSERVE_RMSNORM_CUDA_ARCH", "").strip()
        or os.environ.get("TOKENSPEED_CUDA_ARCH", "").strip()
    )
    if direct:
        return [_normalize_cuda_arch(direct)]

    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return [_normalize_cuda_arch(f"{major}{minor}")]
    except Exception:
        pass

    return ["100a"]


def _build_signature() -> str:
    archs = ",".join(_detect_cuda_archs())
    return f"nvcc={_nvcc()}\ncxx={_cxx()}\narchs={archs}\n"


def _prepare_cuda_toolchain_env() -> None:
    path_entries = [
        entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry
    ]
    candidates = [Path(_nvcc()).resolve().parent]

    for cuda_root in _cuda_toolkit_roots():
        candidates.append(cuda_root / "bin")
        candidates.append(cuda_root / "nvvm" / "bin")

    for base_path in _site_paths():
        for cuda_root in sorted(base_path.glob("nvidia/cu*"), reverse=True):
            candidates.append(cuda_root / "bin")
            candidates.append(cuda_root / "nvvm" / "bin")

    for candidate in reversed(candidates):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in path_entries:
            path_entries.insert(0, candidate_str)
    if path_entries:
        os.environ["PATH"] = os.pathsep.join(path_entries)


def _is_up_to_date() -> bool:
    if not SO_PATH.exists():
        return False
    if (
        not STAMP_PATH.exists()
        or STAMP_PATH.read_text(encoding="utf-8") != _build_signature()
    ):
        return False
    so_mtime = SO_PATH.stat().st_mtime
    headers = [
        CSRC_DIR / "tvm_ffi_utils.h",
        *HEADER_ROOT.rglob("*.h"),
        *HEADER_ROOT.rglob("*.cuh"),
    ]
    return all(so_mtime > path.stat().st_mtime for path in SOURCES + headers)


def build_rmsnorm_fused_parallel(*, force: bool = False, verbose: bool = False) -> Path:
    """Build the local TokenSpeed CUDA RMSNorm shared library if needed."""

    nvcc = _nvcc()
    if shutil.which(nvcc) is None and not Path(nvcc).exists():
        raise RuntimeError(f"nvcc was not found: {nvcc}")

    if not force and _is_up_to_date():
        return SO_PATH

    _prepare_cuda_toolchain_env()
    OBJS_DIR.mkdir(parents=True, exist_ok=True)

    arch_flags = [
        f"-gencode=arch=compute_{arch},code=sm_{arch}"
        for arch in _detect_cuda_archs()
    ]
    nvcc_flags = [
        "-std=c++17",
        "-O2",
        "--expt-relaxed-constexpr",
        "--compiler-options=-fPIC",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_F16",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        *arch_flags,
    ]
    include_flags = [f"-I{path}" for path in _resolve_include_dirs()]

    objects = []
    for source in SOURCES:
        obj = OBJS_DIR / f"{source.stem}.o"
        cmd = [nvcc, *nvcc_flags, *include_flags, "-c", str(source), "-o", str(obj)]
        if verbose:
            print(" ".join(cmd))
        subprocess.check_call(cmd)
        objects.append(obj)

    link_cmd = [
        _cxx(),
        *[str(obj) for obj in objects],
        "-shared",
        *_resolve_cuda_lib_flags(),
        "-o",
        str(SO_PATH),
    ]
    if verbose:
        print(" ".join(link_cmd))
    subprocess.check_call(link_cmd)
    STAMP_PATH.write_text(_build_signature(), encoding="utf-8")
    return SO_PATH


__all__ = ["SO_PATH", "build_rmsnorm_fused_parallel"]
