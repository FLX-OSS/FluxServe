## Getting Started

### Prerequisites

- NVIDIA GPUs with compute capability 9.0+
- CUDA 12.9+


### Steps
Run Docker Environment 
```bash
docker pull flxoss/fluxserve:v0.1-cu130-fa4-nemotron

docker run -itd \
  --shm-size 32g \
  --gpus all \
  --ipc=host \
  --network=host \
  --pid=host \
  --privileged \
  --name flux_workspace \
  flxoss/fluxserve:v0.1-cu130-fa4-nemotron \
  /bin/bash
```

Inside the container:

```bash
git clone https://github.com/FLX-OSS/FluxServe
cd FluxServe
```

Install the fluxserve related package:

```bash
git clone https://github.com/FLX-OSS/FluxServe
cd FluxServe
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -e flux-kernel/python/ --no-build-isolation
pip install -e flux-scheduler
pip install -e .
```

Verify the fluxserve installation and environment pkgs:

```bash
fluxserve env 
```

### Install without Docker (venv)

Each install profile pins the packages of one FluxServe image, so a venv gets the same stack:

| GPU | Profile | `TORCH_CUDA_ARCH_LIST` | Pinned to |
|---|---|---|---|
| SM120 (RTX PRO 6000 Blackwell) | `pyproject_sm120.toml` | `12.0` | the image built from `docker/Dockerfile.flux-sm120` |
| SM100 (B200) | `pyproject_sm100.toml` | `10.0a` | the `flux-cu130-gemma` image (sglang v0.5.12.post1 cu130 base with FlashInfer-dllm) |
| SM90 (H100/H200) | `pyproject_sm90.toml` | `9.0` | the image built from `docker/Dockerfile.flux-cu130-fa4` |

Prerequisites:

- Python 3.12
- CUDA 13.0 toolkit with `nvcc` (`CUDA_HOME` set or `nvcc` on `PATH`), a C++20 compiler, and OpenSSL headers.
  On an HPC cluster this is usually `module load python/3.12 cuda/13.0.2`.
- A checkout that no container also uses. flux-kernel (`flux-kernel/python/flux_kernel/cuda/*/objs`) and
  flux-scheduler (`flux-scheduler/build`) build into the source tree, and objects built in a container may not
  load on the host.
- Keep the venv (about 6 GB), `PIP_CACHE_DIR` and `TMPDIR` off small or NFS home directories. The git
  installs also clone about 1 GB of submodules.

Create the venv:

```bash
python3.12 -m venv /path/to/envs/fluxserve
source /path/to/envs/fluxserve/bin/activate
```

Install from the repo root, with the profile and architecture for your GPU:

```bash
PROFILE=sm120 ARCH=12.0                         # B200: PROFILE=sm100 ARCH=10.0a, H100/H200: PROFILE=sm90 ARCH=9.0
python -m pip install --upgrade pip
export TORCH_CUDA_ARCH_LIST=$ARCH               # same as the image; flux-kernel builds this arch only
export BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1   # stop FlashInfer's build hooks from installing extra wheels

ln -sf pyproject_$PROFILE.toml pyproject.toml   # pip only reads pyproject.toml
pip install -e ".[$PROFILE,dev]" --extra-index-url https://download.pytorch.org/whl/cu130
pip install --no-deps --group sm100-nodeps      # B200 only, see below
pip install --no-build-isolation -e flux-kernel/python/   # needs torch and FlashInfer installed first
pip install -e flux-scheduler
fluxserve env
```

The profile's extra (`sm120`, `sm100` or `sm90`) holds the runtime stack and `dev` the test and build tools.
`+cu130` torch builds only exist on the PyTorch index, so the install fails if `--extra-index-url` is missing.

The B200 image installs FlashInfer without its dependencies, because FlashInfer declares newer cutlass-dsl,
cuda-tile and cudnn-frontend versions than the image runs. The image also runs NCCL 2.31.2 and nvJitLink 13.3,
which conflict with torch's pins. pip's resolver can't express either, so the `sm100-nodeps` group installs them
with `--no-deps`. Afterwards pip reports these conflicts, and `pip check` lists the same FlashInfer and torch
conflicts it lists inside the image.

CI lints the profiles on every pull request that changes them. The `ci:run-pypi` label builds the SM120 and SM100
venvs on the GPU runners with `test/ci_system/pypi/verify_venv.sh`, which you can also run yourself in a GPU
allocation, for example `bash test/ci_system/pypi/verify_venv.sh sm120 --smoke`. SM90 has no GPU runner yet, so CI
only lints its profile.

At runtime, mirror the image environment:

```bash
export TORCH_CUDA_ARCH_LIST=12.0                # 10.0a on B200, 9.0 on H100/H200; must match the build, or flux-kernel rebuilds on first use
export PATH="$PATH:$CUDA_HOME/nvvm/bin"         # nvcc must stay available for FlashInfer JIT
# Use a cache root that containers don't also use: JIT objects built in a container may not load on the host.
CACHE_ROOT=/tmp/$USER-fluxserve-sm120-cache
export FLASHINFER_WORKSPACE_BASE=$CACHE_ROOT XDG_CACHE_HOME=$CACHE_ROOT/xdg \
       TRITON_CACHE_DIR=$CACHE_ROOT/triton TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/torchinductor
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_TRUST_REMOTE_CODE=1 HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false
```

## Next steps

Follow the [quickstart](quickstart.md) for your first request.
