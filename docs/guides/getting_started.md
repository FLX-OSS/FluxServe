## Getting Started

### Prerequisites

- NVIDIA GPUs with compute capability 9.0+

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

### Option 2: Run UV Environment (Deprecated)

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
