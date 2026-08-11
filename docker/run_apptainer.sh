#!/bin/bash

set -euo pipefail

IMAGE="${APPTAINER_IMAGE:-/u/yzhao25/flux-cu129.sif}"
HF_BIND="${HF_BIND:-/work/nvme/bekz/yzhao25/huggingface:/mnt/huggingface}"
FLUXSERVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLUXSERVE_PYTHONPATH="${FLUXSERVE_ROOT}/python:${PYTHONPATH:-}"
GH200_CC="${CC:-/usr/bin/cc}"
GH200_CXX="${CXX:-/usr/bin/g++}"
if [[ -z "${CXX:-}" || "${CXX:-}" == "CC" ]]; then
    GH200_CXX="/usr/bin/g++"
fi
USER_NAME="${USER_NAME:-$(id -un)}"
CACHE_ROOT="/tmp/${USER_NAME}-cache"
mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/xdg"

CONTAINER_COMMAND=(/bin/bash --noprofile --norc -i)
if [[ "$#" -gt 0 ]]; then
    CONTAINER_COMMAND=("$@")
fi

exec apptainer exec --nv --bind "${HF_BIND}" \
                        --pwd "${FLUXSERVE_ROOT}" \
                        "${IMAGE}" \
                        env \
                        "CC=${GH200_CC}" \
                        "CXX=${GH200_CXX}" \
                        "XDG_CACHE_HOME=${CACHE_ROOT}/xdg" \
                        "TRITON_CACHE_DIR=${CACHE_ROOT}/triton" \
                        "TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor" \
                        "PYTHONPATH=${FLUXSERVE_PYTHONPATH}" \
                        "HF_HOME=/mnt/huggingface" \
                        "HF_HUB_CACHE=/mnt/huggingface/hub" \
                        "TRANSFORMERS_TRUST_REMOTE_CODE=1" \
                        "HF_DATASETS_TRUST_REMOTE_CODE=1" \
                        "TOKENIZERS_PARALLELISM=false" \
                        "${CONTAINER_COMMAND[@]}"
