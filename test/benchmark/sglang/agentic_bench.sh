#!/usr/bin/bash

set -euo pipefail

export PYTHONNOUSERSITE=1

EVALSCOPE_COMMIT=acd09b44384d53174768bb1063f675420f76fae9
EVALSCOPE_VENV="${EVALSCOPE_VENV:-/tmp/evalscope-venv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASET_PATH="${REPO_ROOT}/data/agentic/swe_smith_flash.json"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs/$(date +%Y%m%d_%H%M%S)"
SERVER_PID=
SERVER_LOG=

usage() {
    echo "Usage: $0"
}

while (( $# > 0 )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "Dataset not found: $DATASET_PATH" >&2
    exit 1
fi

python -m venv "${EVALSCOPE_VENV}"
"${EVALSCOPE_VENV}/bin/python" -m pip install \
    "evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping SGLang (pgid $SERVER_PID)..."
        kill -TERM -"$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}

wait_for_ready() {
    local start=$SECONDS
    until curl -sf -o /dev/null http://127.0.0.1:8000/health 2>/dev/null; do
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

run_agentic_perf() {
    local config=$1
    local model=$2
    local output_dir="${OUTPUTS_DIR}/${config}"

    SERVER_LOG="${output_dir}_server.log"
    mkdir -p "$output_dir"

    echo "=== Running ${config} ==="
    setsid bash "${SCRIPT_DIR}/configs/${config}.sh" >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    wait_for_ready

    # Agentic-eval specific options: SWE-Smith dataset, multi-turn requests,
    # and the intentionally small two-request run.
    "${EVALSCOPE_VENV}/bin/python" -m evalscope.cli.cli perf \
        --model "$model" \
        --url http://127.0.0.1:8000/v1/chat/completions \
        --api openai \
        --tokenizer-path "$model" \
        --dataset swe_smith \
        --dataset-path "$DATASET_PATH" \
        --max-tokens 512 \
        --multi-turn \
        --no-stream \
        --number 32 \
        --parallel 16 \
        --name "$config" \
        --outputs-dir "$output_dir" \
        --no-timestamp

    stop_server
    wait_for_port_free
}

trap stop_server EXIT

run_agentic_perf tp4_ep4_flash inclusionAI/LLaDA2.0-flash

exit 0
