#!/usr/bin/env bash

set -euo pipefail

exec fluxserve serve \
   --model inclusionAI/LLaDA2.2-flash \
   --host 127.0.0.1 \
   --port 8000 \
   --tp-size 4 \
   --dp-size 1 \
   --ep-size 4 \
   --gpu-memory-utilization 0.85 \
   --max-num-seqs 16 \
   --max-model-len 65536 \
   --max-scheduled-tokens 2048 \
   --block-length 64 \
   --parallel-decoding levenshtein_joint \
   --threshold 0.5 \
   --editing-threshold 0.0 \
   --steps 32 \
   --max-post-steps 16 \
   --max-steps-per-block 1000 \
   --attention-backend fa4 \
   --kv-cache-layout paged \
   --scheduler-policy paged \
   --use-decode-cuda-graph \
   --cuda-graph-decode-mode padded