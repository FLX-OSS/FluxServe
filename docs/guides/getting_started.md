# Docker installation

Build the CUDA environment, install the three FluxServe packages, then follow the [quickstart](quickstart.md).

## Prerequisites

- A Linux host with NVIDIA GPUs of compute capability 9.0 or newer.
- An NVIDIA driver compatible with CUDA 12.9, Docker, and NVIDIA Container Toolkit configured for GPU access.
- Git, network access to package registries and model checkpoints, and enough disk space for the image and model weights.
- Enough GPU memory for your checkpoint and serving configuration. The four-GPU Flash recipe targets H100 GPUs; a GPU count alone does not establish memory capacity.

## Clone the source and build

Run these commands on the host. The Dockerfile is part of the repository, so clone it before building:

```bash
git clone https://github.com/FLX-OSS/FluxServe
cd FluxServe
docker build -f docker/Dockerfile.flux-cu129 -t flux:cu129 .
```

## Start the workspace

The existing development-container configuration shares host resources and runs with elevated privileges. Use it on a trusted development machine; it is not a hardened public-service deployment recipe.

```bash
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

docker exec -it flux_workspace /bin/bash
```

## Install FluxServe inside the container

The image provides the CUDA and Python dependencies. Clone the source inside the container and install the kernel, scheduler, and runtime in that order:

```bash
git clone https://github.com/FLX-OSS/FluxServe
cd FluxServe
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -e flux-kernel/python/ --no-build-isolation
pip install -e flux-scheduler
pip install -e .
```

Keep this shell in the repository directory for examples that use bundled datasets. To run a second command while the server is active, open another container shell with `docker exec -it flux_workspace /bin/bash` and enter the cloned repository.

## Next steps

Follow the [quickstart](quickstart.md) for your first request, or use the [multi-GPU Flash guide](../serving/llada2-flash.md).
