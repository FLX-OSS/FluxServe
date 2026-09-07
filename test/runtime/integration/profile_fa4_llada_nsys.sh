#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${PROFILE_DIR:-${SCRIPT_DIR}/profiles/fa4_llada}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PHASE="${PHASE:-decode}"
BACKEND="${BACKEND:-fa4}"
REPORT="${REPORT:-${PROFILE_DIR}/${PHASE}_${BACKEND}_nsys}"

mkdir -p "${PROFILE_DIR}"

exec nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --export=sqlite \
  --output="${REPORT}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_fa4_llada.py" \
  --phase "${PHASE}" \
  --profile-only "${BACKEND}" \
  --profile-warmup "${PROFILE_WARMUP:-50}" \
  --profile-iterations "${PROFILE_ITERATIONS:-20}" \
  "$@"
