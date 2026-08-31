#!/usr/bin/env bash

set -euo pipefail


exec sglang serve  \
    --model-path inclusionAI/LLaDA2.0-mini \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --expert-parallel-size 4 \
    --data-parallel-size 1 \
    --max-running-requests 32 \
    --mem-fraction-static 0.8 \
    --dllm-algorithm LowConfidence \
    --dllm-algorithm-config ./test/benchmark/sglang/configs/dllm_config.yaml \
    --attention-backend flashinfer \
    --disable-radix-cache \
    --disable-piecewise-cuda-graph \
    --chat-template ./test/benchmark/sglang/configs/llada2_chat_template.jinja
