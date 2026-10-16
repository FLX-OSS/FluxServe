#!/bin/bash

# CI wrapper for the SM120 (RTX PRO 6000) runner. The image's /opt/venv is
# read-only, so commands run under the system Python with a sitecustomize that
# adds the venv's site-packages, and `pip install --user` gives writable
# installs. CI is isolated from the developer's $HOME on this shared node:
#   - HOME is a CI-only directory (no ~/.bashrc, no ~/.local). It persists
#     between runs, so it also caches the evalscope venv and ModelScope data.
#   - PYTHONUSERBASE lives in the checkout, so user installs land in a per-run
#     directory that the workflow deletes along with the work dir.

set -euo pipefail

IMAGE="${APPTAINER_IMAGE:-/blue/juwang.ucf/yo493393.ucf/flux-sm120.sif}"
HF_BIND="${HF_BIND:-/blue/juwang.ucf/yo493393.ucf/huggingface:/mnt/huggingface}"
FLUXSERVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
USER_NAME="${USER_NAME:-$(id -un)}"
CACHE_ROOT="/tmp/${USER_NAME}-cache"
CI_HOME="${FLUXSERVE_CI_HOME:-/tmp/${USER_NAME}-fluxserve-ci-home}"
CI_SHIMS="${CI_HOME}/.ci-shims"
CI_USERBASE="${FLUXSERVE_ROOT}/.ci-pyuser"
mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/xdg" \
         "${CI_SHIMS}/bin" "${CI_SHIMS}/python" "${CI_USERBASE}"
ln -sfn /usr/bin/python3 "${CI_SHIMS}/bin/python"
# Let the writable system Python load the image venv's .pth files as well as its packages.
cat > "${CI_SHIMS}/python/sitecustomize.py" <<'PY'
import os
import site
import sys

venv = os.environ.get("VIRTUAL_ENV")
if sys.prefix == sys.base_prefix and venv:
    site.addsitedir(os.path.join(venv, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"))
PY

CONTAINER_COMMAND=(/bin/bash --noprofile --norc -i)
if [[ "$#" -gt 0 ]]; then
    CONTAINER_COMMAND=("$@")
fi

# PATH is the image's PATH minus /opt/venv/bin, so python/python3 resolve to
# the system interpreter, with per-run user scripts (e.g. fluxserve) first.
exec apptainer exec --nv --bind "${HF_BIND}" \
                        --bind "${FLUXSERVE_ROOT}:${FLUXSERVE_ROOT}" \
                        --bind "${CI_HOME}:${CI_HOME}" \
                        --pwd "${FLUXSERVE_ROOT}" \
                        "${IMAGE}" \
                        env \
                        "HOME=${CI_HOME}" \
                        "PYTHONUSERBASE=${CI_USERBASE}" \
                        "PATH=${CI_USERBASE}/bin:${CI_SHIMS}/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/nvvm/bin" \
                        "PYTHONPATH=${CI_SHIMS}/python" \
                        "VIRTUAL_ENV=/opt/venv" \
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
