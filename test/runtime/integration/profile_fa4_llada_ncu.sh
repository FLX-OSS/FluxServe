#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${PROFILE_DIR:-${SCRIPT_DIR}/profiles/fa4_llada}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PHASE="${PHASE:-decode}"
BACKEND="${BACKEND:-fa4}"
REPORT="${REPORT:-${PROFILE_DIR}/${PHASE}_${BACKEND}_ncu}"

mkdir -p "${PROFILE_DIR}"

exec ncu \
  --force-overwrite \
  --target-processes all \
  --profile-from-start off \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section LaunchStats \
  --section Occupancy \
  --export "${REPORT}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_fa4_llada.py" \
  --phase "${PHASE}" \
  --profile-only "${BACKEND}" \
  --profile-warmup "${PROFILE_WARMUP:-50}" \
  --profile-iterations "${PROFILE_ITERATIONS:-1}" \
  "$@"
