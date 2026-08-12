#!/usr/bin/env bash

set -euo pipefail

python -m fluxserve.cli bench_offline \
    --model inclusionAI/LLaDA2.0-mini \
    --dataset ./data/humaneval.jsonl \
    --tp-size 1 \
    --dp-size 1 \
    --ep-size 1 \
    --batch-size 16 \
    --mini-batch-size 16 \
    --gen-len 2048 \
    --block-length 64 \
    --threshold 0.95 \
    --page-size 64 \
    --use-decode-cuda-graph \
    --output-dir "logs/humaneval_bench_flx_mini_humaneval_tp1_ep1"
