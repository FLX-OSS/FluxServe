#!/usr/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# Install EvalScope
EVALSCOPE_COMMIT=acd09b44384d53174768bb1063f675420f76fae9
python3 -m pip install "evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

CONFIGS=(
    tp1_ep1_mini
    tp4_ep4_flash
)

MODELS=(
    inclusionAI/LLaDA2.0-mini
    inclusionAI/LLaDA2.0-flash
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${SCRIPT_DIR}/../../../data/openai/bigcodebench.jsonl"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs/$(date +%Y%m%d_%H%M%S)"
SERVER_PID=
SERVER_LOG=

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping FluxServe (pgid $SERVER_PID)..."
        kill -TERM -"$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}

wait_for_ready() {
    local start=$SECONDS
    until curl -sf -o /dev/null http://127.0.0.1:8000/health; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server died early. Last log lines:" >&2
            tail -100 "$SERVER_LOG" >&2
            return 1
        fi
        if (( SECONDS - start > 1200 )); then
            echo "Timeout waiting for server" >&2
            return 1
        fi
        sleep 5
    done
}

wait_for_port_free() {
    for _ in {1..90}; do
        if python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 8000)); s.close()" 2>/dev/null; then
            return
        fi
        sleep 1
    done
    echo "Port 8000 is still in use" >&2
    return 1
}

trap stop_server EXIT

mkdir -p "$OUTPUTS_DIR"

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    MODEL="${MODELS[$i]}"
    SERVER_LOG="${OUTPUTS_DIR}/${CONFIG}_server.log"

    echo "=== Running $CONFIG ==="
    setsid bash "${SCRIPT_DIR}/configs/${CONFIG}.sh" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    wait_for_ready

    python3 -m evalscope.cli.cli perf \
        --model "$MODEL" \
        --url http://127.0.0.1:8000/v1/chat/completions \
        --api openai \
        --tokenizer-path "$MODEL" \
        --dataset line_by_line \
        --dataset-path "$DATASET_PATH" \
        --max-tokens 2048 \
        --no-stream \
        --parallel 16 \
        --number 1000 \
        --rate 16 \
        --name "$CONFIG" \
        --outputs-dir "$OUTPUTS_DIR" \
        --no-timestamp

    stop_server
    wait_for_port_free
done
