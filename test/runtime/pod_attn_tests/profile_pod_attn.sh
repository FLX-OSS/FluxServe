#!/usr/bin/env bash
set -euo pipefail

# Capture a node-level CUDA trace for the BatchPOD attention benchmark.
# Extra arguments are passed directly to batch_pod.py, for example:
#   ./profile_pod_attn.sh --prefill-seq-lens 512,1024 --decode-kv-lens 512,1024

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${PROFILE_DIR:-${SCRIPT_DIR}/profiling}"
TRACE_NAME="${TRACE_NAME:-batch_pod}"
NSYS_BIN="${NSYS_BIN:-}"

if [[ -z "${NSYS_BIN}" ]]; then
  if command -v nsys >/dev/null 2>&1; then
    NSYS_BIN="$(command -v nsys)"
  elif [[ -x /usr/local/cuda/bin/nsys ]]; then
    NSYS_BIN=/usr/local/cuda/bin/nsys
  else
    echo "error: nsys was not found; set NSYS_BIN or add /usr/local/cuda/bin to PATH" >&2
    exit 127
  fi
fi

if ! "${NSYS_BIN}" --version >/dev/null 2>&1; then
  echo "error: unable to execute ${NSYS_BIN}" >&2
  exit 127
fi

mkdir -p "${PROFILE_DIR}"

# Keep the measured run short and deterministic by default.  The benchmark's
# command-line arguments can override these values (argparse uses the last one).
exec "${NSYS_BIN}" profile \
  --force-overwrite=true \
  --cuda-graph-trace=node \
  --export=sqlite \
  -o "${PROFILE_DIR}/${TRACE_NAME}" \
  python3 "${SCRIPT_DIR}/batch_pod.py" \
  --warmup 3 \
  --iters 10 \
  --workspace-mb "${WORKSPACE_MB:-512}" \
  --prefill-seq-lens 256,512,1024,2048 \
  --decode-kv-lens 256,512,1024,2048 \
  "$@"
