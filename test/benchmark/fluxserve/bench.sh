#!/usr/bin/bash

set -euo pipefail

# Keep lazy native builds deterministic for the RTX PRO 6000 benchmark.
# export FLUX_KERNEL_CUDA_ARCH="${FLUX_KERNEL_CUDA_ARCH:-120}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs/$(date +%Y%m%d_%H%M%S)"
SERVER_PID=
SERVER_LOG=
RATES=(1 2 4 8 16)
SELECTED_CONFIG=
SELECTED_MODEL=
SELECTED_DATASET=gsm8k.jsonl
SELECTED_BENCHMARK=gsm8k
NUM=1000
PARALLEL=16
while (( $# )); do
    case "$1" in
        --help|-h)
            echo "Usage: bash $0 [--config NAME --model MODEL] [--output-dir DIR]"
            echo 'Options: --dataset PATH_RELATIVE_TO_DATA --benchmark NAME --rates 1,2,4,8,16 --num 1000 --parallel 16'
            echo 'Default: original experiment matrix. --config selects a single experiment (GSM8K by default).'
            exit 0 ;;
        --config|--model|--output-dir|--dataset|--benchmark|--rates|--num|--parallel)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --config) SELECTED_CONFIG=$2 ;;
                --model) SELECTED_MODEL=$2 ;;
                --output-dir) OUTPUTS_DIR=$2 ;;
                --dataset) SELECTED_DATASET=$2 ;;
                --benchmark) SELECTED_BENCHMARK=$2 ;;
                --rates)
                    [[ "$2" =~ ^[0-9]+([.][0-9]+)?(,[0-9]+([.][0-9]+)?)*$ ]] || { echo 'Use comma-separated positive rates' >&2; exit 2; }
                    IFS=, read -r -a RATES <<< "$2" ;;
                --num) NUM=$2 ;;
                --parallel) PARALLEL=$2 ;;
            esac
            shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$NUM" =~ ^[1-9][0-9]*$ && "$PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo '--num and --parallel must be positive integers' >&2; exit 2; }
for rate in "${RATES[@]}"; do
    [[ "$rate" =~ [1-9] ]] || { echo '--rates must be positive' >&2; exit 2; }
done
if [[ -n "$SELECTED_CONFIG" || -n "$SELECTED_MODEL" ]]; then
    [[ "$SELECTED_CONFIG" =~ ^[a-zA-Z0-9_-]+$ && -n "$SELECTED_MODEL" ]] || { echo '--config and --model must be supplied together' >&2; exit 2; }
    [[ "$SELECTED_BENCHMARK" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid benchmark name' >&2; exit 2; }
    [[ -f "$SCRIPT_DIR/configs/$SELECTED_CONFIG.sh" ]] || { echo 'Config not found' >&2; exit 2; }
    [[ -f "$REPO_ROOT/data/$SELECTED_DATASET" ]] || { echo 'Dataset not found' >&2; exit 2; }
elif [[ "$SELECTED_DATASET" != gsm8k.jsonl || "$SELECTED_BENCHMARK" != gsm8k ]]; then
    echo '--dataset and --benchmark require --config and --model' >&2; exit 2
fi

EVALSCOPE_COMMIT=acd09b44384d53174768bb1063f675420f76fae9
EVALSCOPE_VENV="${EVALSCOPE_VENV:-/tmp/evalscope-venv}"
python -m venv "${EVALSCOPE_VENV}"
"${EVALSCOPE_VENV}/bin/python" -m pip install \
    "evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

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

    for rate in "${RATES[@]}"; do
        local rate_output_dir="${output_dir}/rate_${rate}"
        mkdir -p "$rate_output_dir"

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
            --num "$NUM"
            --parallel "$PARALLEL"
            --rate "$rate"
            --name "${benchmark}_${config}_rate_${rate}"
            --outputs-dir "$rate_output_dir"
            --no-timestamp
        )
        if [[ -n "$number_arg" ]]; then
            perf_args+=("$number_flag" "$number_arg")
        fi

        echo "=== Running ${benchmark}/${config} at rate ${rate} ==="
        "${EVALSCOPE_VENV}/bin/python" "${perf_args[@]}" 2>&1 | tee "${rate_output_dir}/perf.log"
    done
    stop_server
    wait_for_port_free
}

trap stop_server EXIT

if [[ -n "$SELECTED_CONFIG" ]]; then
    run_perf "$SELECTED_BENCHMARK" "$SELECTED_CONFIG" "$SELECTED_MODEL" "$SELECTED_DATASET"
    exit 0
fi

run_perf gsm8k tp1_ep1_mini inclusionAI/LLaDA2.0-mini gsm8k.jsonl 
run_perf gsm8k tp4_ep4_flash inclusionAI/LLaDA2.0-mini gsm8k.jsonl 
run_perf bigcodebench tp1_ep1_mini inclusionAI/LLaDA2.0-mini openai/bigcodebench.jsonl 
run_perf bigcodebench tp4_ep4_flash inclusionAI/LLaDA2.0-mini openai/bigcodebench.jsonl 
run_perf gsm8k tp1_ep1_gemma google/diffusiongemma-26B-A4B-it gsm8k.jsonl 
run_perf gsm8k tp4_ep4_gemma google/diffusiongemma-26B-A4B-it gsm8k.jsonl 
run_perf bigcodebench tp1_ep1_gemma google/diffusiongemma-26B-A4B-it openai/bigcodebench.jsonl 
run_perf bigcodebench tp4_ep4_gemma google/diffusiongemma-26B-A4B-it openai/bigcodebench.jsonl 


exit 0
