#!/usr/bin/env bash

set -euo pipefail

exec fluxserve launch \
    --model nvidia/Nemotron-Labs-Diffusion-8B \
    --host 127.0.0.1 \
    --port 8000 \
    --tp-size 1 \
    --dp-size 1 \
    --ep-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 8 \
    --max-model-len 8192 \
    --max-scheduled-tokens 2048 \
    --block-length 32 \
    --page-size 32 \
    --parallel-decoding self_speculation \
    --threshold 0.9 \
    --attention-backend fa4 \
    --kv-cache-layout paged \
    --scheduler-policy paged \
    --use-decode-cuda-graph \
    --cuda-graph-decode-mode padded \
    --cuda-graph-capture-bs 1 2 4 8 \
    --trust-remote-code