#!/usr/bin/env bash

set -euo pipefail

# Install EvalScope.
EVALSCOPE_COMMIT=acd09b44384d53174768bb1063f675420f76fae9
EVALSCOPE_VENV="${EVALSCOPE_VENV:-/tmp/evalscope-venv}"
python -m venv "${EVALSCOPE_VENV}"
"${EVALSCOPE_VENV}/bin/python" -m pip install \
    "evalscope @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

CONFIGS=(
    tp1_ep1_mini
    tp4_ep4_flash
    tp4_ep4_gemma
)

MODELS=(
    inclusionAI/LLaDA2.0-mini
    inclusionAI/LLaDA2.0-flash
    google/diffusiongemma-26B-A4B-it
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs/$(date +%Y%m%d_%H%M%S)"
MIN_ACCURACY="${GSM8K_MIN_ACCURACY:-0.5}"
SERVER_PID=
SERVER_LOG=

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping FluxServe (pgid $SERVER_PID)..."
        kill -TERM -"$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

wait_for_ready() {
    local start=$SECONDS
    until curl -fs -o /dev/null http://127.0.0.1:8000/health 2>/dev/null; do
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

trap stop_server EXIT

mkdir -p "$OUTPUTS_DIR"

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    MODEL="${MODELS[$i]}"
    SERVER_LOG="${OUTPUTS_DIR}/${CONFIG}_server.log"
    REPORT_DIR="${OUTPUTS_DIR}/${CONFIG}"

    echo "=== Running $CONFIG ==="
    setsid bash "${REPO_ROOT}/test/benchmark/fluxserve/configs/${CONFIG}.sh" \
        >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    wait_for_ready

    "${EVALSCOPE_VENV}/bin/python" -m evalscope.cli.cli eval \
        --model "$MODEL" \
        --model-id "$CONFIG" \
        --eval-type openai_api \
        --api-url http://127.0.0.1:8000/v1/chat/completions \
        --api-key EMPTY \
        --datasets gsm8k \
        --eval-batch-size 16 \
        --judge-strategy rule \
        --generation-config '{"max_tokens": 2048, "temperature": 0.0}' \
        --work-dir "$REPORT_DIR" \
        --no-timestamp \
        --no-collect-perf

    REPORT_FILE="$(find "$REPORT_DIR/reports" -type f -name gsm8k.json -print -quit)"
    if [[ -z "$REPORT_FILE" ]]; then
        echo "EvalScope did not produce a GSM8K report" >&2
        exit 1
    fi

    "${EVALSCOPE_VENV}/bin/python" - "$REPORT_FILE" "$MIN_ACCURACY" <<'PY'
import json
import sys

report_path, minimum = sys.argv[1], float(sys.argv[2])
with open(report_path, encoding="utf-8") as stream:
    report = json.load(stream)
accuracy = float(report["score"])
print(f"GSM8K accuracy: {accuracy:.4f}; required: {minimum:.4f}")
if accuracy < minimum:
    raise SystemExit(f"accuracy {accuracy:.4f} is below {minimum:.4f}")
PY

    stop_server
    SERVER_PID=
done

exit 0
