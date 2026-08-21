#!/usr/bin/env bash

set -euo pipefail

exec fluxserve serve \
    --model google/diffusiongemma-26B-A4B-it \
    --host 127.0.0.1 \
    --port 8000 \
    --tp-size 4 \
    --dp-size 1 \
    --ep-size 4 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 16 \
    --max-model-len 65536 \
    --max-scheduled-tokens 2048 \
    --block-length 256 \
    --attention-backend flashinfer \
    --flashinfer-prefill-mode paged \
    --flashinfer-cache-mode paged \
    --kv-cache-layout paged \
    --page-size 256
