#!/usr/bin/env bash

set -euo pipefail

# LLaDA2.1-mini, joint M2T/T2T decoding, Quality preset (0.7/0.5).
exec fluxserve serve \
    --model inclusionAI/LLaDA2.1-mini \
    --host 127.0.0.1 \
    --port 8000 \
    --tp-size 1 \
    --dp-size 1 \
    --ep-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 16 \
    --max-model-len 65536 \
    --max-scheduled-tokens 2048 \
    --block-length 64 \
    --parallel-decoding joint_threshold \
    --threshold 0.7 \
    --editing-threshold 0.5 \
    --max-post-steps 16 \
    --attention-backend flashinfer \
    --kv-cache-layout paged \
    --scheduler-policy paged \
    --use-decode-cuda-graph \
    --cuda-graph-decode-mode padded \
    --cuda-graph-capture-bs 1 2 4 8 10 12 16 \
