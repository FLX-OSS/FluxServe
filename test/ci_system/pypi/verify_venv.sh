#!/usr/bin/env bash
# Build a venv from a pip install profile (pyproject_<profile>.toml) on its GPU host and check it against the
# profile's reference image. .github/workflows/pr-test-pypi.yml runs it; locally, run it in a GPU allocation:
#   srun -p hpg-rtx6000 --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=02:00:00 --pty bash -l
#   bash test/ci_system/pypi/verify_venv.sh sm120 --smoke
#   srun -p hpg-b200 --gres=gpu:b200:1 --cpus-per-task=16 --mem=64G --time=02:00:00 --pty bash -l
#   bash test/ci_system/pypi/verify_venv.sh sm100 --smoke
# --in-place builds in this checkout (CI's fresh checkout) instead of a clone with the working-tree changes,
# --smoke runs the serving smoke test, --eval the GSM8K CI spec. Everything lives under $WORK.

set -euo pipefail

usage() {
    echo "Usage: bash $0 sm120|sm100 [--in-place] [--smoke] [--eval]" >&2
    exit 2
}

PROFILE=${1:-}
[[ $# -gt 0 ]] && shift
IN_PLACE=0
RUN_SMOKE=0
RUN_EVAL=0
for arg in "$@"; do
    case "$arg" in
        --in-place) IN_PLACE=1 ;;
        --smoke) RUN_SMOKE=1 ;;
        --eval) RUN_EVAL=1 ;;
        *) usage ;;
    esac
done

NODEPS_GROUPS=()
case "$PROFILE" in
    sm120)
        DEFAULT_IMAGE=/blue/juwang.ucf/yo493393.ucf/flux-sm120.sif
        IMAGE_PYTHON=(/opt/venv/bin/python -I)
        ARCH=12.0
        STAMP_ARCH=120
        CAPABILITY=12.0
        GPU_NAME="RTX PRO 6000"
        ;;
    sm100)
        DEFAULT_IMAGE=/blue/juwang.ucf/yo493393.ucf/flux-cu130-gemma.sif
        IMAGE_PYTHON=(python3 -I)
        ARCH=10.0a
        STAMP_ARCH=100a
        CAPABILITY=10.0
        GPU_NAME="B200"
        NODEPS_GROUPS=(sm100-nodeps)
        ;;
    *) usage ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/../../.." && pwd)"
USER_NAME="$(id -un)"
IMAGE="${APPTAINER_IMAGE:-$DEFAULT_IMAGE}"
WORK="${WORK:-/tmp/${USER_NAME}-fs-venv-${PROFILE}}"
HF_ROOT="${FLUXSERVE_HF_HOME:-/blue/juwang.ucf/yo493393.ucf/huggingface}"
PIP_CACHE="${FLUXSERVE_PYPI_PIP_CACHE:-/tmp/${USER_NAME}-fluxserve-pypi/pip-cache}"
EVAL_SPEC="${EVAL_SPEC:-test/ci/1N1G-SM120/eval/nemotron-diffusion-14b-evalscope-gsm8k.yaml}"
PT_INDEX=https://download.pytorch.org/whl/cu130

[[ ! -e "$WORK" ]] || { echo "$WORK already exists; remove it or set WORK." >&2; exit 1; }
ARTIFACTS="$WORK/artifacts"
mkdir -p "$WORK/home" "$WORK/tmp" "$WORK/cache" "$ARTIFACTS" "$PIP_CACHE"
exec > >(tee "$ARTIFACTS/verify.log") 2>&1
trap 'echo "Stopped at line $LINENO; log: $ARTIFACTS/verify.log" >&2' ERR

RESULTS=()
check() {  # check NAME COMMAND...: run one check, record the result, keep going
    local name=$1
    shift
    echo
    echo "== $name"
    if "$@"; then
        echo "PASS: $name"
        RESULTS+=("PASS|$name")
    else
        echo "FAIL: $name"
        RESULTS+=("FAIL|$name")
    fi
}

# The runner's PATH starts with ~/.local/bin (uv and user-site scripts), and its HOME is on NFS.
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/opt/slurm/bin
export HOME="$WORK/home" TMPDIR="$WORK/tmp" PIP_CACHE_DIR="$PIP_CACHE"
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export __Init_Default_Modules=1  # keep Lmod from restoring the default modules in nested login shells
unset PYTHONPATH VIRTUAL_ENV

echo "== Preflight ($PROFILE on $(hostname))"
gpu=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
[[ "$gpu" == *"$GPU_NAME"* ]] || { echo "Expected a $GPU_NAME GPU, found: ${gpu:-none}" >&2; exit 1; }
IFS=, read -r _ gpu_total gpu_free <<<"$gpu"
gpu_total=${gpu_total// /}
gpu_free=${gpu_free// /}
echo "GPU: $gpu"
if ((RUN_SMOKE || RUN_EVAL)); then
    # The smoke and eval servers take 85% of GPU memory; another user may hold some on a shared node.
    ((gpu_free * 10 >= gpu_total * 9)) || { echo "Only $gpu_free of $gpu_total MiB GPU memory is free." >&2; exit 1; }
    if (exec 3<>/dev/tcp/127.0.0.1/18000) 2>/dev/null; then
        echo "Port 18000 is already in use." >&2
        exit 1
    fi
fi
free_kb=$(df -Pk "$WORK" | awk 'NR == 2 {print $4}')
((free_kb >= 60 * 1024 * 1024)) || { echo "Need 60 GB free under $WORK, have $((free_kb / 1048576)) GB." >&2; exit 1; }

if ((IN_PLACE)); then
    CHECKOUT="$SRC"
else
    echo "== Clone $SRC with its working-tree changes"
    CHECKOUT="$WORK/FluxServe"
    git clone -q "$SRC" "$CHECKOUT"
    if ! git -C "$SRC" diff --quiet HEAD; then
        git -C "$SRC" diff --binary HEAD | git -C "$CHECKOUT" apply
    fi
    cp "$SRC"/pyproject_*.toml "$CHECKOUT/"
fi
cd "$CHECKOUT"

echo "== Check the profile against $IMAGE"
# HOME now points into $WORK, so apptainer no longer binds the real home: bind the scripts' tree explicitly.
apptainer exec --cleanenv --bind "$SRC" --bind "$WORK" "$IMAGE" "${IMAGE_PYTHON[@]}" \
    "$SCRIPT_DIR/profile_checks.py" image "$CHECKOUT/pyproject_$PROFILE.toml" \
    --snapshot "$ARTIFACTS/image-snapshot.json"

echo "== Toolchain and venv"
set +u  # Lmod and venv activation reference unset variables
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge
module load python/3.12 cuda/13.0.2
python3.12 -m venv "$WORK/venv"
# shellcheck source=/dev/null
source "$WORK/venv/bin/activate"  # also unsets PYTHONHOME, which the python module sets
set -u
unset PYTHONHOME
[[ "$("$CUDA_HOME/bin/nvcc" --version)" == *"release 13.0"* ]] || { echo "CUDA_HOME=$CUDA_HOME is not CUDA 13.0" >&2; exit 1; }
export PIP_REQUIRE_VIRTUALENV=1
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-16}}"
export TORCH_CUDA_ARCH_LIST=$ARCH BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1
export PATH="$PATH:$CUDA_HOME/nvvm/bin"
export FLASHINFER_WORKSPACE_BASE="$WORK/cache" XDG_CACHE_HOME="$WORK/cache/xdg" CUDA_CACHE_PATH="$WORK/cache/nv" \
       TRITON_CACHE_DIR="$WORK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torchinductor"
export HF_HOME="$HF_ROOT" HF_HUB_CACHE="$HF_ROOT/hub"
export TRANSFORMERS_TRUST_REMOTE_CODE=1 HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false

echo "== Install (docs/guides/getting_started.md)"
python -m pip install --upgrade pip
python -c 'import pip, sys; sys.exit(tuple(map(int, pip.__version__.split(".")[:2])) < (25, 1))' \
    || { echo "pip >= 25.1 is needed for --group" >&2; exit 1; }
ln -sf "pyproject_$PROFILE.toml" pyproject.toml
pip install -e ".[$PROFILE,dev]" --extra-index-url "$PT_INDEX"
for group in "${NODEPS_GROUPS[@]}"; do
    pip install --no-deps --group "$group"
done
pip install --no-build-isolation -e flux-kernel/python/
pip install -e flux-scheduler
pip list --format=freeze >"$ARTIFACTS/venv-freeze.txt"

check "venv matches the profile and the image" python "$SCRIPT_DIR/profile_checks.py" venv \
    "pyproject_$PROFILE.toml" --snapshot "$ARTIFACTS/image-snapshot.json" --checkout "$CHECKOUT" \
    --stamp-arch "$STAMP_ARCH" --capability "$CAPABILITY"
check "fluxserve env" fluxserve env

if ((RUN_SMOKE)); then
    check "serving smoke test" python "$SCRIPT_DIR/serve_smoke.py" --repo-root "$CHECKOUT" --out-dir "$ARTIFACTS"
fi

if ((RUN_EVAL)); then
    runner=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["runner"]["labels"][0])' "$EVAL_SPEC")
    check "GSM8K eval ($EVAL_SPEC)" python test/ci_system/pipeline.py execute --config "$EVAL_SPEC" \
        --runner "$runner" --work-dir "$PWD" --print-plan --result-json "$ARTIFACTS/eval-result.json"
fi

echo
echo "== Summary ($PROFILE)"
failed=0
for result in "${RESULTS[@]}"; do
    echo "${result%%|*}  ${result#*|}"
    [[ "${result%%|*}" == PASS ]] || failed=1
done
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "### pip install profile \`$PROFILE\` on $(hostname)"
        echo
        echo "| Check | Result |"
        echo "|---|---|"
        for result in "${RESULTS[@]}"; do
            echo "| ${result#*|} | ${result%%|*} |"
        done
    } >>"$GITHUB_STEP_SUMMARY"
fi
echo "Log: $ARTIFACTS/verify.log"
exit $failed
