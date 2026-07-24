# Building FluxServe with uv

This guide installs FluxServe directly on an Ubuntu host with uv. It targets
NVIDIA Hopper and Blackwell GPUs with compute capability 9.0 or newer. The
validated software stack mirrors `docker/Dockerfile.flux-cu129`: Python 3.12,
CUDA 12.9, and PyTorch 2.8.0 with CUDA 12.9 wheels.

For an isolated environment with system dependencies already installed, use
the [Docker workflow](getting_started.md) instead.

## Prerequisites

Install the following before creating the Python environment:

- Ubuntu 22.04 on x86_64
- An NVIDIA driver compatible with CUDA 12.9
- CUDA Toolkit 12.9, including `nvcc`
- Git and Git LFS
- A C/C++ build toolchain, CMake 3.22 or newer, Ninja, pkg-config, and OpenSSL
  development headers
- An NVIDIA GPU with compute capability 9.0 or newer

On Ubuntu, the non-CUDA build tools can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git git-lfs libssl-dev ninja-build pkg-config
```

Install the NVIDIA driver and CUDA Toolkit from NVIDIA's CUDA 12.9
documentation. Confirm that both are visible before continuing:

```bash
nvidia-smi
nvcc --version
```

## Select the CUDA architecture

Set the architecture for the GPU that will run FluxServe. Building only for
the local architecture reduces native compilation time.

| GPU family | Compute capability | Setting |
| --- | ---: | --- |
| Hopper (for example, H100 or H200) | 9.0 | `TORCH_CUDA_ARCH_LIST=9.0` |
| Blackwell datacenter (for example, B100 or B200) | 10.0 | `TORCH_CUDA_ARCH_LIST=10.0` |
| Blackwell variants with compute capability 12.0 | 12.0 | `TORCH_CUDA_ARCH_LIST=12.0` |

The commands below use Hopper. Replace `9.0` with the compute capability of
the target GPU. Multiple targets can be separated by semicolons, such as
`TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"`, at the cost of longer builds.

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=64
export FLASHINFER_WORKSPACE_BASE=/tmp
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Reduce `MAX_JOBS` if the host runs out of memory during native compilation.

## Install with uv

Install uv using its official installer, then clone FluxServe:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/FLX-OSS/FluxServe.git
cd FluxServe
```

Create the Python 3.12 environment and install the locked dependencies:

```bash
uv python install 3.12
uv sync --locked
```

If this checkout was previously built through a container, remove its native
build artifacts before syncing on the host. Container-created files can belong
to a different user and prevent CMake from updating its cache:

```bash
sudo rm -rf flux-scheduler/build flux-kernel/python/build \
  flux-kernel/python/flux_kernel/cuda/rmsnorm/objs
uv sync --locked
```

The sync installs FluxServe, `flux-kernel`, and `flux-scheduler` in editable
mode. It also supplies torch, TVM FFI, and the project-specific FlashInfer fork
to the isolated Flux kernel build environment. FlashInfer is installed at its
pinned commit.

Use `uv run` to execute commands without activating the environment:

```bash
uv run fluxserve --help
```

Alternatively, activate it with `source .venv/bin/activate`.

## Inspect the installation

The following commands display the installed Python and CUDA configuration:

```bash
uv run python --version
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
uv run python -c "import flashinfer, flux_kernel, flux_scheduler, fluxserve; print('imports successful')"
```

## Launch FluxServe

```bash
uv run fluxserve serve \
  --model inclusionAI/LLaDA2.0-mini \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 4 \
  --ep-size 4 \
  --dp-size 1
```

## Troubleshooting

- **`nvcc` is missing:** install the CUDA 12.9 Toolkit and ensure
  `$CUDA_HOME/bin` is on `PATH`.
- **The NVIDIA driver is incompatible:** install a driver version that supports
  CUDA 12.9, then confirm it with `nvidia-smi`.
- **A CUDA kernel has no image for the GPU:** set `TORCH_CUDA_ARCH_LIST` to the
  target GPU's compute capability and run `uv sync --reinstall-package
  flux-kernel`.
- **A native build cannot import torch or FlashInfer:** use `uv sync --locked`
  from the repository root. The uv metadata adds these packages to the
  isolated `flux-kernel` build environment.
- **CMake cannot find OpenSSL:** install the development package with
  `sudo apt-get install libssl-dev`. If OpenSSL is installed in a nonstandard
  prefix, export `OPENSSL_ROOT_DIR` with that prefix before syncing.
- **CMake cannot write `cmake.check_cache`:** the build directory was likely
  created by a container or another user. Remove `flux-scheduler/build` using
  the cleanup command above, then run `uv sync --locked` again.
- **Compilation exhausts memory:** lower `MAX_JOBS`, remove generated `build`
  and `objs` directories for the affected native package, and sync again.
