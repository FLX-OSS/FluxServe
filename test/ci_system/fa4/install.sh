#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# Fail before starting a server if another service owns the configured port.
python - "$1" <<'PY'
from pathlib import Path
import socket
import sys
if Path(sys.argv[1]).exists():
    raise SystemExit(f"Archive previous output before rerunning: {sys.argv[1]}")
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 8000))
PY
# Keep the GH200 image's torch and CUDA stack intact.
python -m venv --system-site-packages .ci-artifacts/fa4-runtime
.ci-artifacts/fa4-runtime/bin/python -m pip install --no-deps --no-build-isolation -e .
.ci-artifacts/fa4-runtime/bin/python -m pip install --no-deps \
  flash-attn-4==4.0.0b29 nvidia-cutlass-dsl==4.6.2 \
  nvidia-cutlass-dsl-libs-base==4.6.2 nvidia-cutlass-dsl-libs-cu12==4.6.2 \
  nvidia-cutlass-dsl-libs-core==4.6.2 quack-kernels==0.5.3 \
  apache-tvm-ffi==0.1.12 torch-c-dlpack-ext==0.1.5
python -m venv .ci-artifacts/fa4-evalscope
.ci-artifacts/fa4-evalscope/bin/python -m pip install \
  'evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@acd09b44384d53174768bb1063f675420f76fae9'
