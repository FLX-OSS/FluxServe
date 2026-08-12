#!/usr/bin/env bash

set -euo pipefail

python -m fluxserve.cli bench_offline \
    --model inclusionAI/LLaDA2.0-flash \
    --dataset ./data/humaneval.jsonl \
    --tp-size 4 \
    --dp-size 1 \
    --ep-size 4 \
    --batch-size 1 \
    --mini-batch-size 1 \
    --gen-len 2048 \
    --block-length 64 \
    --threshold 0.95 \
    --page-size 64 \
    --use-decode-cuda-graph \
    --output-dir "logs/humaneval_bench_flx_flash_humaneval_tp4_ep4"
