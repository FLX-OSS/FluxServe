#!/usr/bin/env bash
set -euo pipefail

# Capture Nsight Compute counters for the FlashInfer POD attention kernel.
# Extra arguments are passed directly to batch_pod.py. This launcher defaults
# to a single measured iteration because NCU substantially slows kernels.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${PROFILE_DIR:-${SCRIPT_DIR}/profiling}"
REPORT_NAME="${REPORT_NAME:-batch_pod_ncu}"
NCU_BIN="${NCU_BIN:-}"
PYTHON_BIN="${PYTHON_BIN:-}"


if [[ -z "${NCU_BIN}" ]]; then
  if command -v ncu >/dev/null 2>&1; then
    NCU_BIN="$(command -v ncu)"
  elif [[ -x /usr/local/cuda/bin/ncu ]]; then
    NCU_BIN=/usr/local/cuda/bin/ncu
  else
    echo "error: ncu was not found; set NCU_BIN or add /usr/local/cuda/bin to PATH" >&2
    exit 127
  fi
fi

if ! "${NCU_BIN}" --version >/dev/null 2>&1; then
  echo "error: unable to execute ${NCU_BIN}" >&2
  exit 127
fi

mkdir -p "${PROFILE_DIR}"

NCU_ARGS=(
  --force-overwrite
  --target-processes all
  --kernel-name BatchPODWithKVCacheTensorKernel
  --section SpeedOfLight
  --section MemoryWorkloadAnalysis
  --section LaunchStats
  --section Occupancy
  --section SchedulerStats
  --section WarpStateStats
  --export "${PROFILE_DIR}/${REPORT_NAME}"
  python "${SCRIPT_DIR}/batch_pod.py"
  --warmup 3
  --iters 10
  --workspace-mb "${WORKSPACE_MB:-512}"
  --prefill-seq-lens 128,256,512,1024,2048,4096,4096,8192,8192,8192,16384,16384
  --decode-kv-lens 256,512,1024,2048
  "$@"
)

exec "${NCU_BIN}" "${NCU_ARGS[@]}"
