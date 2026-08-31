#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec vllm serve \
    --model google/diffusiongemma-26B-A4B-it \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.80 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --chat-template "${SCRIPT_DIR}/tool_chat_template_gemma4.jinja"
