#!/bin/bash

set -euo pipefail

HOST_HF_CACHE="${HOST_HF_CACHE:-/home/ypzhao/.cache/huggingface}"
HOST_FLUX_SERVE="${HOST_FLUX_SERVE:-/home/ypzhao/flx-oss/FluxServe}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/flx-foundry-cache}"

CONTAINER_NAME="${CONTAINER_NAME:-flux_foundry_workspace}"
IMAGE_NAME="${IMAGE_NAME:-flux:foundry-cu130}"

if [[ ! -d "$HOST_FLUX_SERVE/foundry" ]]; then
  echo "Foundry checkout not found at $HOST_FLUX_SERVE/foundry" >&2
  exit 1
fi

mkdir -p \
  "$HOST_HF_CACHE" \
  "$HOST_HF_CACHE/modules" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/xdg"

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  --network=host \
  --pid=host \
  --privileged \
  -e CC="${CC:-/usr/bin/cc}" \
  -e CXX="${CXX:-/usr/bin/g++}" \
  -e FLASHINFER_WORKSPACE_BASE=/tmp \
  -e HF_HOME=/mnt/huggingface \
  -e HF_HUB_CACHE=/mnt/huggingface/hub \
  -e HF_MODULES_CACHE=/mnt/huggingface/modules \
  -e PIP_NO_CACHE_DIR=1 \
  -e TORCH_HOME=/tmp/fluxserve-cache/torch \
  -e TRITON_CACHE_DIR=/tmp/fluxserve-cache/triton \
  -e XDG_CACHE_HOME=/tmp/fluxserve-cache/xdg \
  -v "$HOST_HF_CACHE":/mnt/huggingface \
  -v "$CACHE_ROOT":/tmp/fluxserve-cache \
  -v "$HOST_FLUX_SERVE":/workspace/FluxServe \
  -w /workspace/FluxServe \
  "$IMAGE_NAME" \
  tail -f /dev/null

echo "Container '$CONTAINER_NAME' is running."
echo "Image -> $IMAGE_NAME"
echo "FluxServe -> /workspace/FluxServe"
echo "Foundry -> /workspace/FluxServe/foundry"
