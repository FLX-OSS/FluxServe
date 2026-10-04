#!/usr/bin/bash

set -euo pipefail

# Keep lazy native builds deterministic for the RTX PRO 6000 benchmark.
# export FLUX_KERNEL_CUDA_ARCH="${FLUX_KERNEL_CUDA_ARCH:-120}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs/$(date +%Y%m%d_%H%M%S)"
SERVER_PID=
SERVER_LOG=
NUMS=(64 128 256 512 1024)
PARALLELS=(1 2 4 8 16)
SELECTED_CONFIG=
SELECTED_MODEL=
TOKENIZER_PATH=
SELECTED_DATASET=gsm8k.jsonl
SELECTED_BENCHMARK=gsm8k
while (( $# )); do
    case "$1" in
        --help|-h)
            echo "Usage: bash $0 [--config NAME --model MODEL] [--output-dir DIR]"
            echo 'Options: --dataset PATH_RELATIVE_TO_DATA --benchmark NAME --num 64,128,256,512,1024 --parallel 1,2,4,8,16'
            echo 'Tokenizer: --tokenizer-path LOCAL_DIR (default: resolve model from Hugging Face cache)'
            echo 'Default: original experiment matrix. --config selects a single experiment (GSM8K by default).'
            exit 0 ;;
        --config|--model|--tokenizer-path|--output-dir|--dataset|--benchmark|--num|--parallel)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --config) SELECTED_CONFIG=$2 ;;
                --model) SELECTED_MODEL=$2 ;;
                --tokenizer-path) TOKENIZER_PATH=$2 ;;
                --output-dir) OUTPUTS_DIR=$2 ;;
                --dataset) SELECTED_DATASET=$2 ;;
                --benchmark) SELECTED_BENCHMARK=$2 ;;
                --num)
                    [[ "$2" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || { echo '--num requires comma-separated positive integers' >&2; exit 2; }
                    IFS=, read -r -a NUMS <<< "$2" ;;
                --parallel)
                    [[ "$2" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || { echo '--parallel requires comma-separated positive integers' >&2; exit 2; }
                    IFS=, read -r -a PARALLELS <<< "$2" ;;
            esac
            shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -n "$SELECTED_CONFIG" || -n "$SELECTED_MODEL" ]]; then
    [[ "$SELECTED_CONFIG" =~ ^[a-zA-Z0-9_-]+$ && -n "$SELECTED_MODEL" ]] || { echo '--config and --model must be supplied together' >&2; exit 2; }
    [[ "$SELECTED_BENCHMARK" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid benchmark name' >&2; exit 2; }
    [[ -f "$SCRIPT_DIR/configs/$SELECTED_CONFIG.sh" ]] || { echo 'Config not found' >&2; exit 2; }
    [[ -f "$REPO_ROOT/data/$SELECTED_DATASET" ]] || { echo 'Dataset not found' >&2; exit 2; }
elif [[ "$SELECTED_DATASET" != gsm8k.jsonl || "$SELECTED_BENCHMARK" != gsm8k ]]; then
    echo '--dataset and --benchmark require --config and --model' >&2; exit 2
fi

EVALSCOPE_COMMIT=6f1400f1bf0eea7ea31351597dc0bae0c1d8afe0
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
    local tokenizer_path

    if [[ ! -f "$dataset_path" ]]; then
        echo "Dataset not found: $dataset_path" >&2
        return 1
    fi

    # This EvalScope revision uses modelscope.AutoTokenizer. Give it a local
    # Hugging Face snapshot so it does not look up the model on ModelScope.
    tokenizer_path=$("${EVALSCOPE_VENV}/bin/python" - "$model" "$TOKENIZER_PATH" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

model, override = sys.argv[1:]
if override or Path(model).is_dir():
    path = Path(override or model).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"Tokenizer directory not found: {path}")
else:
    tokenizer_files = [
        "tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab.*",
        "merges.txt", "*.model", "*.tiktoken", "chat_template*", "config.json",
    ]
    try:
        path = snapshot_download(model, allow_patterns=tokenizer_files, local_files_only=True)
    except LocalEntryNotFoundError:
        # Fetch tokenizer/config files only; model weights are loaded by FluxServe.
        path = snapshot_download(
            model,
            allow_patterns=tokenizer_files,
        )
print(path)
PY
    )
    echo "Using tokenizer: $tokenizer_path"

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
        --tokenizer-path "$tokenizer_path"
        --dataset line_by_line
        --dataset-path "$dataset_path"
        --max-tokens 2048
        --no-stream
        --num "${NUMS[@]}"
        --parallel "${PARALLELS[@]}"
        --name "${benchmark}_${config}"
        --outputs-dir "$output_dir"
        --no-timestamp
    )
    if [[ -n "$number_arg" ]]; then
        perf_args+=("$number_flag" "$number_arg")
    fi

    echo "=== Running ${benchmark}/${config} at concurrency ${PARALLELS[*]} ==="
    "${EVALSCOPE_VENV}/bin/python" "${perf_args[@]}" 2>&1 | tee "${output_dir}/perf.log"
    stop_server
    wait_for_port_free
}

trap stop_server EXIT

if [[ -n "$SELECTED_CONFIG" ]]; then
    run_perf "$SELECTED_BENCHMARK" "$SELECTED_CONFIG" "$SELECTED_MODEL" "$SELECTED_DATASET"
    exit 0
fi


run_perf bigcodebench tp1_nemotron_3b nvidia/Nemotron-Labs-Diffusion-3B bigcodebench.jsonl


exit 0
