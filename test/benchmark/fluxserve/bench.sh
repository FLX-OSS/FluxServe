#!/usr/bin/bash

set -euo pipefail

export PYTHONNOUSERSITE=1

EVALSCOPE_COMMIT=acd09b44384d53174768bb1063f675420f76fae9
EVALSCOPE_VENV="${EVALSCOPE_VENV:-/tmp/evalscope-venv}"
python -m venv "${EVALSCOPE_VENV}"
"${EVALSCOPE_VENV}/bin/python" -m pip install \
    "evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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

run_perf() {
    local benchmark=$1
    local config=$2
    local model=$3
    local dataset=$4
    local number_flag=${5:-}
    local number_arg=${6:-}
    local output_dir="${OUTPUTS_DIR}/${benchmark}/${config}"
    local dataset_path="${REPO_ROOT}/data/${dataset}"

    if [[ ! -f "$dataset_path" ]]; then
        echo "Dataset not found: $dataset_path" >&2
        return 1
    fi

    SERVER_LOG="${output_dir}_server.log"
    mkdir -p "$output_dir"

    echo "=== Running ${benchmark}/${config} ==="
    setsid bash "${SCRIPT_DIR}/configs/${config}.sh" >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    wait_for_ready

    perf_args=(
        -m evalscope.cli.cli perf
        --model "$model"
        --url http://127.0.0.1:8000/v1/chat/completions
        --api openai
        --tokenizer-path "$model"
        --dataset line_by_line
        --dataset-path "$dataset_path"
        --max-tokens 2048
        --no-stream
        --num 1000
        --parallel 16
        --rate 16
        --name "${benchmark}_${config}"
        --outputs-dir "$output_dir"
        --no-timestamp
    )
    if [[ -n "$number_arg" ]]; then
        perf_args+=("$number_flag" "$number_arg")
    fi

    "${EVALSCOPE_VENV}/bin/python" "${perf_args[@]}"
    stop_server
    wait_for_port_free
}

trap stop_server EXIT

run_perf gsm8k tp1_ep1_mini inclusionAI/LLaDA2.0-mini openai/gsm8k_openai.jsonl 
run_perf bigcodebench tp1_ep1_mini inclusionAI/LLaDA2.0-mini openai/bigcodebench.jsonl 

exit 0
