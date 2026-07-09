## Getting Started

### Prerequisites

- NVIDIA GPU host
- Docker with GPU support

### Run Local Docker Environment

```bash
docker build -f docker/Dockerfile.flux-cu129 -t flux:cu129

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

## Install Packages

Install the Python runtime:

```bash
pip install -e .
```

Install the flux-kernel package.

```bash
pip install -e flux-kernel/python/ --no-build-isolation
```

Install the flux-scheduler package:

```bash
pip install -e flux-scheduler/python/
```

## Launch

```bash
torchrun --nproc-per-node 1 -m fluxserve.cli serve \
  --model inclusionAI/LLaDA2.0-mini \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 1 \
  --ep-size 4 \
  --dp-size 1
```