#!/usr/bin/env bash
# Run within a single-GPU Slurm allocation. Keeps the image's PyTorch unchanged.
set -euo pipefail
cd /u/dzhu8/workspace/FluxServe
graph_deps=/tmp/fa4graph-pinned
if [[ ! -d "$graph_deps/flash_attn" ]]; then
    bash test/ci/env/gh200.sh python -m pip install --no-deps --target "$graph_deps" \
        flash-attn-4==4.0.0b29 nvidia-cutlass-dsl==4.6.2 \
        nvidia-cutlass-dsl-libs-base==4.6.2 nvidia-cutlass-dsl-libs-cu12==4.6.2 \
        nvidia-cutlass-dsl-libs-core==4.6.2 quack-kernels==0.5.3 \
        apache-tvm-ffi==0.1.12 torch-c-dlpack-ext==0.1.5
fi
exec bash test/ci/env/gh200.sh env \
    PYTHONPATH="$graph_deps:/u/dzhu8/workspace/FluxServe/python" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$@"
