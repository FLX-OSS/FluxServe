## Getting Started

### Prerequisites

- NVIDIA GPUs with compute capability 9.0+
- Docker with GPU support

### Option 1: Run Docker Environment (Recommended)

```bash
docker build -f docker/Dockerfile.flux-cu129 -t flux:cu129 .

docker run -itd \
  --shm-size 32g \
  --gpus all \
  --ipc=host \
  --network=host \
  --pid=host \
  --privileged \
  --name flux_workspace \
  flux:cu129 \
  /bin/bash
```

Inside the container:

```bash
git clone https://github.com/FLX-OSS/FluxServe
cd FluxServe
```

Install the flux-kernel package:

```bash
pip install -e flux-kernel/python/ --no-build-isolation
```

Install the flux-scheduler package:

```bash
pip install -e flux-scheduler
```

Install the Python runtime:

```bash
pip install -e .
```

Launch FluxServe engine:

```bash
python -m fluxserve.cli serve \
  --model inclusionAI/LLaDA2.0-mini \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 1 \
  --ep-size 1 \
  --dp-size 1
```

### Option 2: Run the Foundry-Compatible Docker Environment

Foundry requires PyTorch 2.9 or newer and its current recipe targets PyTorch
2.11 with CUDA 13.0. The dedicated image keeps that stack isolated from
FluxServe's default PyTorch 2.8/CUDA 12.9 environment.

The local Foundry checkout must exist at `foundry/` inside the FluxServe
checkout. Build and launch the toolchain image from the repository root:

```bash
docker build \
  -f docker/Dockerfile.flux-foundry-cu130 \
  -t flux:foundry-cu130 \
  .

bash docker/commands/start_foundry_docker.sh
docker exec -it flux_foundry_workspace bash --login
```

Inside the container, install the mounted projects without allowing
FluxServe's default dependency pins to replace PyTorch 2.11:

```bash
cd /workspace/FluxServe
python -m pip install -e . --no-deps
python -m pip install -e flux-kernel/python --no-build-isolation --no-deps
python -m pip install -e flux-scheduler --no-build-isolation --no-deps
python -m pip install -e foundry --no-build-isolation
```

If the container was started from an older image, install the scheduler's
OpenSSL headers before retrying its editable build:

```bash
apt-get update && apt-get install -y --no-install-recommends libssl-dev
```

Verify the CUDA/PyTorch stack and native modules:

```bash
python - <<'PY'
import torch
import flux_kernel
import flux_scheduler
import fluxserve
import foundry
import foundry.ops

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("FluxServe and Foundry imports successful")
PY

ldd foundry/python/foundry/libcuda_hook.so
python -m pytest foundry/tests/test_imports.py
python -m fluxserve.cli --help
```

The launcher accepts `IMAGE_NAME`, `CONTAINER_NAME`, `HOST_FLUX_SERVE`,
`HOST_HF_CACHE`, and `CACHE_ROOT` overrides. The default native build target is
Hopper (SM90); override `TORCH_CUDA_ARCH_LIST` when building packages if needed.


### Option 3: Run UV Environment

For host prerequisites, CUDA architecture selection, and troubleshooting, see
[Building FluxServe with uv](uv_build.md).

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

Inspect the installation to display the installed Python and CUDA configuration and check fluxserve cli:

```bash
uv run python --version
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
uv run python -c "import flashinfer, flux_kernel, flux_scheduler, fluxserve; print('imports successful')"
uv run fluxserve --help
```

Launch FluxServe

```bash
uv run fluxserve serve \
  --model inclusionAI/LLaDA2.0-mini \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 1 \
  --ep-size 1 \
  --dp-size 1
```
