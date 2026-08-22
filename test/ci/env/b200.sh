#!/bin/bash

set -euo pipefail

IMAGE="${APPTAINER_IMAGE:-/blue/juwang.ucf/yo493393.ucf/flux-cu130-gemma.sif}"
HF_BIND="${HF_BIND:-/blue/juwang.ucf/yo493393.ucf/huggingface:/mnt/huggingface}"
FLUXSERVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
USER_NAME="${USER_NAME:-$(id -un)}"
CACHE_ROOT="/tmp/${USER_NAME}-cache"
mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/xdg"

CONTAINER_COMMAND=(/bin/bash --noprofile --norc -i)
if [[ "$#" -gt 0 ]]; then
    CONTAINER_COMMAND=("$@")
fi

exec apptainer exec --nv --bind "${HF_BIND}" \
                        --bind "${FLUXSERVE_ROOT}:${FLUXSERVE_ROOT}" \
                        --pwd "${FLUXSERVE_ROOT}" \
                        "${IMAGE}" \
                        env \
                        "GITHUB_WORKSPACE=${GITHUB_WORKSPACE:-${FLUXSERVE_ROOT}}" \
                        "XDG_CACHE_HOME=${CACHE_ROOT}/xdg" \
                        "TRITON_CACHE_DIR=${CACHE_ROOT}/triton" \
                        "TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor" \
                        "HF_HOME=/mnt/huggingface" \
                        "HF_HUB_CACHE=/mnt/huggingface/hub" \
                        "TRANSFORMERS_TRUST_REMOTE_CODE=1" \
                        "HF_DATASETS_TRUST_REMOTE_CODE=1" \
                        "TOKENIZERS_PARALLELISM=false" \
                        "${CONTAINER_COMMAND[@]}"
