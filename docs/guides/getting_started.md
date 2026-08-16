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
export PIP_BREAK_SYSTEM_PACKAGES=1
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
fluxserve serve \
  --model inclusionAI/LLaDA2.0-mini \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 1 \
  --ep-size 1 \
  --dp-size 1
```

### Option 2: Run UV Environment (Deprecated)