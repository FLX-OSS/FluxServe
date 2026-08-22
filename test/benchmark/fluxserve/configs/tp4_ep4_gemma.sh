#!/usr/bin/env bash

set -euo pipefail

exec fluxserve serve \
    --model google/diffusiongemma-26B-A4B-it \
    --host 127.0.0.1 \
    --port 8000 \
    --tp-size 4 \
    --dp-size 1 \
    --ep-size 4 \
    --gpu-memory-utilization 0.6 \
    --max-num-seqs 4 \
    --max-model-len 32876 \
    --max-scheduled-tokens 2048 \
    --block-length 256 \
    --canvas-length 256 \
    --page-size 256 \
    --attention-backend flashinfer \
    --flashinfer-prefill-mode paged \
    --flashinfer-cache-mode paged \
    --kv-cache-layout paged \
    --scheduler-policy default \
    --use-decode-cuda-graph