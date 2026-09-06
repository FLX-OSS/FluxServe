#!/bin/bash

set -euo pipefail

IMAGE="${APPTAINER_IMAGE:-/projects/bekz/yzhao25/flux-cu129-gemma.sif}"
HF_BIND="${HF_BIND:-/work/nvme/bekz/yzhao25/huggingface:/mnt/huggingface}"
FLUXSERVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FLUXSERVE_PYTHONPATH="${FLUXSERVE_ROOT}/python:${PYTHONPATH:-}"
GH200_CC="${CC:-/usr/bin/cc}"
GH200_CXX="${CXX:-/usr/bin/g++}"
if [[ -z "${CXX:-}" || "${CXX:-}" == "CC" ]]; then
    GH200_CXX="/usr/bin/g++"
fi
USER_NAME="${USER_NAME:-$(id -un)}"
CACHE_ROOT="/tmp/${USER_NAME}-cache"
mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/xdg"

CONTAINER_CMD=(
    apptainer exec --nv --bind "${HF_BIND}" \
                        --bind "${FLUXSERVE_ROOT}:${FLUXSERVE_ROOT}" \
                        --pwd "${FLUXSERVE_ROOT}" \
                        "${IMAGE}" \
                        env \
                        "CC=${GH200_CC}" \
                        "CXX=${GH200_CXX}" \
                        "GITHUB_WORKSPACE=${GITHUB_WORKSPACE:-${FLUXSERVE_ROOT}}" \
                        "XDG_CACHE_HOME=${CACHE_ROOT}/xdg" \
                        "TRITON_CACHE_DIR=${CACHE_ROOT}/triton" \
                        "TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor" \
                        "PYTHONPATH=${FLUXSERVE_PYTHONPATH}" \
                        "HF_HOME=/mnt/huggingface" \
                        "HF_HUB_CACHE=/mnt/huggingface/hub" \
                        "TRANSFORMERS_TRUST_REMOTE_CODE=1" \
                        "HF_DATASETS_TRUST_REMOTE_CODE=1" \
                        "TOKENIZERS_PARALLELISM=false"
)

if (( $# > 0 )); then
    # Allow callers to run a non-interactive command in the same environment.
    exec "${CONTAINER_CMD[@]}" "$@"
fi

exec "${CONTAINER_CMD[@]}" /bin/bash --noprofile --norc -i
