#!/usr/bin/env bash

set -euo pipefail

exec fluxserve serve \
    --model inclusionAI/LLaDA2.0-mini \
    --host 127.0.0.1 \
    --port 8000 \
    --tp-size 4 \
    --dp-size 1 \
    --ep-size 4 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 32 \
    --max-model-len 65536 \
    --max-scheduled-tokens 4096 \
    --block-length 64 \
    --parallel-decoding threshold \
    --threshold 0.95 \
    --attention-backend flashinfer \
    --kv-cache-layout paged \
    --scheduler-policy paged \
    --use-decode-cuda-graph \
    --cuda-graph-decode-mode padded \
    --cuda-graph-capture-bs 1 2 4 8 10 12 16 18 20 22 24 26 28 30 32\
