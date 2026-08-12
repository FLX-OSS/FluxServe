#!/bin/bash

set -euo pipefail

IMAGE="${APPTAINER_IMAGE:-/projects/bekz/yzhao25/sglang-h200.sif}"
HF_BIND="${HF_BIND:-/work/nvme/bekz/yzhao25/huggingface:/mnt/huggingface}"
H200_CC="${CC:-/usr/bin/cc}"
H200_CXX="${CXX:-/usr/bin/g++}"
if [[ -z "${CXX:-}" || "${CXX:-}" == "CC" ]]; then
    H200_CXX="/usr/bin/g++"
fi
USER_NAME="${USER_NAME:-$(id -un)}"
CACHE_ROOT="/tmp/${USER_NAME}-cache"
mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/xdg"

exec apptainer exec --nv --bind "${HF_BIND}" \
                        "${IMAGE}" \
                        env \
                        "CC=${H200_CC}" \
                        "CXX=${H200_CXX}" \
                        "XDG_CACHE_HOME=${CACHE_ROOT}/xdg" \
                        "TRITON_CACHE_DIR=${CACHE_ROOT}/triton" \
                        "TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor" \
                        "HF_HOME=/mnt/huggingface" \
                        "HF_HUB_CACHE=/mnt/huggingface/hub" \
                        "TRANSFORMERS_TRUST_REMOTE_CODE=1" \
                        "HF_DATASETS_TRUST_REMOTE_CODE=1" \
                        "TOKENIZERS_PARALLELISM=false" \
                        bash -c "set -e; \
                                 bash scripts/install.sh; \
                                 bash test/benchmark/fluxserve/configs/humaneval/tp1_ep1_mini.sh
                                 bash test/benchmark/fluxserve/configs/humaneval/tp4_ep4_flash.sh
                                 "
