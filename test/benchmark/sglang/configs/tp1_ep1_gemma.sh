#!/usr/bin/env bash

set -euo pipefail

exec sglang serve \
    --model-type llm \
    --model-path google/diffusiongemma-26B-A4B-it \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --expert-parallel-size 1 \
    --data-parallel-size 1 \
    --max-running-requests 16 \
    --mem-fraction-static 0.8 \
    --dllm-algorithm Gemma4Renoise \
    --attention-backend flashinfer \
    --dllm-algorithm-config ./test/benchmark/sglang/configs/gemma_config.yaml \
    --disable-radix-cache \
    --cuda-graph-backend-decode full
