#!/usr/bin/env bash

set -euo pipefail


exec sglang serve  \
    --model-path inclusionAI/LLaDA2.0-flash \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --expert-parallel-size 4 \
    --data-parallel-size 1 \
    --max-running-requests 16 \
    --mem-fraction-static 0.8 \
    --dllm-algorithm LowConfidence \
    --attention-backend flashinfer \
    --disable-radix-cache \
    --chat-template ./test/benchmark/sglang/llada2_chat_template.jinja
