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

### Install without Docker (SM120, venv)

`pyproject_sm120.toml` pins the packages of the SM120 (RTX PRO 6000 Blackwell) image built from
`docker/Dockerfile.flux-sm120`, so a venv gets the same stack.

Prerequisites:

- Python 3.12
- CUDA 13.0 toolkit with `nvcc` (`CUDA_HOME` set or `nvcc` on `PATH`), a C++20 compiler, and OpenSSL headers.
  On an HPC cluster this is usually `module load python/3.12 cuda/13.0.2`.
- A checkout that no container also uses. flux-kernel (`flux-kernel/python/flux_kernel/cuda/*/objs`) and
  flux-scheduler (`flux-scheduler/build`) build into the source tree, and objects built in a container may not
  load on the host.
- Keep the venv (about 6 GB), `PIP_CACHE_DIR` and `TMPDIR` off small or NFS home directories. The FA4 git
  install also clones about 1 GB of submodules.

Create the venv:

```bash
python3.12 -m venv /path/to/envs/fluxserve-sm120
source /path/to/envs/fluxserve-sm120/bin/activate
```

Install from the repo root:

```bash
python -m pip install --upgrade pip
export TORCH_CUDA_ARCH_LIST=12.0                # same as the image; flux-kernel builds sm_120 only
export BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1   # stop FlashInfer's build hooks from installing extra wheels

ln -sf pyproject_sm120.toml pyproject.toml      # pip only reads pyproject.toml
pip install -e ".[sm120,dev]" --extra-index-url https://download.pytorch.org/whl/cu130
pip install --no-build-isolation -e flux-kernel/python/   # needs torch and FlashInfer installed first
pip install -e flux-scheduler
fluxserve env
```

The `sm120` extra holds the runtime stack and `dev` the test and build tools. `+cu130` torch builds only exist on
the PyTorch index, so the install fails if `--extra-index-url` is missing.

At runtime, mirror the image environment:

```bash
export TORCH_CUDA_ARCH_LIST=12.0                # must match the build, or flux-kernel rebuilds on first use
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
