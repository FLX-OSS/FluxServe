## Getting Started

### Prerequisites

- NVIDIA GPUs with compute capability 9.0+
- CUDA 12.9+


### Steps
Run Docker Environment 
```bash
docker pull flxoss/fluxserve:v0.1-cu130-fa4

docker run -itd \
  --shm-size 32g \
  --gpus all \
  --ipc=host \
  --network=host \
  --pid=host \
  --privileged \
  --name flux_workspace \
  fluxserve:v0.1-cu130-fa4 \
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

## Next steps

Follow the [quickstart](quickstart.md) for your first request.
