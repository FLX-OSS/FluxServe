#!/usr/bin/env bash

set -euo pipefail

SGLANG_ROOT="${SGLANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SGLANG_ROOT}"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="${MODEL:-inclusionAI/LLaDA2.0-mini}"
DATASET="${DATASET:-/u/yzhao25/FluxServe/data/humaneval.jsonl}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
REQUEST_RATES="${REQUEST_RATES:-1 2 4 8 16}"
NUM_PROMPTS="${NUM_PROMPTS:-164}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-8}"
OUTPUT_LEN="${OUTPUT_LEN:-2048}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
DLLM_ALGORITHM="${DLLM_ALGORITHM:-LowConfidence}"
DLLM_BLOCK_SIZE="${DLLM_BLOCK_SIZE:-64}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
DISABLE_FLASHINFER_DEEPSEEK_TOPK="${DISABLE_FLASHINFER_DEEPSEEK_TOPK:-always}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-600}"

RESULT_ROOT="${RESULT_ROOT:-${SGLANG_ROOT}/online_bench_runs_gsm8k_flash_tp1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/humaneval_online_tp${TP_SIZE}_ep${EP_SIZE}_${RUN_ID}}"
SERVER_LOG="${SERVER_LOG:-${RUN_DIR}/server.log}"
MANIFEST="${RUN_DIR}/manifest.txt"
DLLM_CONFIG="${RUN_DIR}/dllm_algorithm_config.json"
SERVER_PID=""

die() {
  echo "error: $*" >&2
  exit 1
}

is_positive_number() {
  awk -v value="$1" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }'
}

validate_result() {
  python - "$1" "$2" "$3" "$4" <<'PY'
import json
import math
import sys

path, expected_prompts, expected_rate, expected_concurrency = (
    sys.argv[1],
    int(sys.argv[2]),
    float(sys.argv[3]),
    int(sys.argv[4]),
)
with open(path, encoding="utf-8") as result_file:
    lines = [line for line in result_file if line.strip()]

assert len(lines) == 1, f"expected one JSONL result, found {len(lines)}"
result = json.loads(lines[0])
assert result["completed"] == expected_prompts, result.get("completed")
assert float(result["request_rate"]) == expected_rate, result.get("request_rate")
assert result["max_concurrency"] == expected_concurrency, result.get("max_concurrency")

errors = result.get("errors", [])
assert len(errors) == expected_prompts, f"expected {expected_prompts} error entries"
assert not [error for error in errors if error], "one or more requests failed"

for metric in (
    "duration",
    "request_throughput",
    "output_throughput",
    "total_throughput",
    "mean_e2e_latency_ms",
):
    value = result.get(metric)
    assert isinstance(value, (int, float)), f"missing numeric metric: {metric}"
    assert math.isfinite(value) and value > 0, f"invalid {metric}: {value}"

assert result.get("total_output_tokens", 0) > 0
assert len(result.get("input_lens", [])) == expected_prompts
assert len(result.get("output_lens", [])) == expected_prompts
PY
}

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill -- "-${SERVER_PID}" >/dev/null 2>&1 || kill "${SERVER_PID}" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" >/dev/null 2>&1 || break
      sleep 1
    done
    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      kill -KILL -- "-${SERVER_PID}" >/dev/null 2>&1 || true
    fi
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -f "${DATASET}" ]] || die "dataset not found: ${DATASET}"
for name in TP_SIZE EP_SIZE DP_SIZE NUM_PROMPTS WARMUP_PROMPTS OUTPUT_LEN \
  MAX_CONCURRENCY MAX_RUNNING_REQUESTS DLLM_BLOCK_SIZE STARTUP_TIMEOUT_SEC; do
  value="${!name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer"
done
[[ "${DP_SIZE}" == "1" ]] || die "this benchmark requires DP_SIZE=1"
(( EP_SIZE <= TP_SIZE )) || die "EP_SIZE must not exceed TP_SIZE"
(( TP_SIZE % EP_SIZE == 0 )) || die "TP_SIZE must be divisible by EP_SIZE"
[[ "${DISABLE_FLASHINFER_DEEPSEEK_TOPK}" == "always" || \
  "${DISABLE_FLASHINFER_DEEPSEEK_TOPK}" == "never" ]] || die \
  "DISABLE_FLASHINFER_DEEPSEEK_TOPK must be always or never"
is_positive_number "${MEM_FRACTION_STATIC}" || die "MEM_FRACTION_STATIC must be positive"
for rate in ${REQUEST_RATES}; do
  is_positive_number "${rate}" || die "invalid request rate: ${rate}"
done

mkdir -p "${RUN_DIR}"
[[ -w "${RUN_DIR}" ]] || die "result directory is not writable: ${RUN_DIR}"
printf '{"block_size": %s}\n' "${DLLM_BLOCK_SIZE}" >"${DLLM_CONFIG}"

if [[ "${DISABLE_FLASHINFER_DEEPSEEK_TOPK}" == "always" ]]; then
  TOPK_SITECUSTOMIZE_DIR="${SGLANG_ROOT}/llada_scripts/sglang_topk_sitecustomize"
  [[ -f "${TOPK_SITECUSTOMIZE_DIR}/sitecustomize.py" ]] || die \
    "DeepSeek top-k hook not found: ${TOPK_SITECUSTOMIZE_DIR}"
  export SGLANG_DISABLE_FLASHINFER_DEEPSEEK_TOPK=1
  export PYTHONPATH="${TOPK_SITECUSTOMIZE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
else
  unset SGLANG_DISABLE_FLASHINFER_DEEPSEEK_TOPK
fi

VISIBLE_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
[[ "${VISIBLE_GPU_COUNT}" == "${TP_SIZE}" ]] || die \
  "expected ${TP_SIZE} visible GPUs, found ${VISIBLE_GPU_COUNT}; set CUDA_VISIBLE_DEVICES"

SERVER_CMD=(
  sglang serve
  --model-path "${MODEL}"
  --trust-remote-code
  --host "${SERVER_HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --expert-parallel-size "${EP_SIZE}"
  --data-parallel-size "${DP_SIZE}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --dllm-algorithm "${DLLM_ALGORITHM}"
  --dllm-algorithm-config "${DLLM_CONFIG}"
  --attention-backend "${ATTENTION_BACKEND}"
  --disable-radix-cache
)

{
  printf 'run_id=%s\nmodel=%s\ndataset=%s\n' "${RUN_ID}" "${MODEL}" "${DATASET}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'server_host=%s\nclient_host=%s\nport=%s\n' "${SERVER_HOST}" "${CLIENT_HOST}" "${PORT}"
  printf 'tp_size=%s\nep_size=%s\ndp_size=%s\n' "${TP_SIZE}" "${EP_SIZE}" "${DP_SIZE}"
  printf 'request_rates=%s\nnum_prompts=%s\nwarmup_prompts=%s\noutput_len=%s\n' \
    "${REQUEST_RATES}" "${NUM_PROMPTS}" "${WARMUP_PROMPTS}" "${OUTPUT_LEN}"
  printf 'max_concurrency=%s\nmax_running_requests=%s\n' \
    "${MAX_CONCURRENCY}" "${MAX_RUNNING_REQUESTS}"
  printf 'mem_fraction_static=%s\ndllm_algorithm=%s\ndllm_block_size=%s\n' \
    "${MEM_FRACTION_STATIC}" "${DLLM_ALGORITHM}" "${DLLM_BLOCK_SIZE}"
  printf 'attention_backend=%s\nprimary_metrics=request_throughput,output_throughput,total_throughput,e2e_latency\n' \
    "${ATTENTION_BACKEND}"
  printf 'disable_flashinfer_deepseek_topk=%s\ndisable_ignore_eos=true\n' \
    "${DISABLE_FLASHINFER_DEEPSEEK_TOPK}"
  printf 'server_command='
  printf '%q ' "${SERVER_CMD[@]}"
  printf '\n'
} >"${MANIFEST}"

setsid "${SERVER_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"
echo "Started SGLang server pid=${SERVER_PID}; log=${SERVER_LOG}"

for _ in $(seq 1 "${STARTUP_TIMEOUT_SEC}"); do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    wait "${SERVER_PID}" || true
    tail -n 100 "${SERVER_LOG}" >&2 || true
    die "server exited before becoming healthy"
  fi
  if curl -fsS "http://${CLIENT_HOST}:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://${CLIENT_HOST}:${PORT}/health" >/dev/null 2>&1; then
  tail -n 100 "${SERVER_LOG}" >&2 || true
  die "server did not become healthy within ${STARTUP_TIMEOUT_SEC} seconds"
fi

COMMON_BENCH_ARGS=(
  -m sglang.bench_serving
  --backend sglang-oai-chat
  --base-url "http://${CLIENT_HOST}:${PORT}"
  --dataset-name openai
  --dataset-path "${DATASET}"
  --model "${MODEL}"
  --num-prompts "${NUM_PROMPTS}"
  --sharegpt-output-len "${OUTPUT_LEN}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --warmup-requests "${WARMUP_PROMPTS}"
  --seed 0
  --disable-ignore-eos
  --output-details 
)

for rate in ${REQUEST_RATES}; do
  RATE_LABEL="${rate//./p}"
  OUTPUT_FILE="${RUN_DIR}/rate_${RATE_LABEL}.jsonl"
  CLIENT_LOG="${RUN_DIR}/rate_${RATE_LABEL}.log"
  BENCH_CMD=(
    python "${COMMON_BENCH_ARGS[@]}"
    --request-rate "${rate}"
    --output-file "${OUTPUT_FILE}"
    --tag "humaneval_tp${TP_SIZE}_ep${EP_SIZE}_rate${rate}"
  )
  {
    printf 'benchmark_command_rate_%s=' "${rate}"
    printf '%q ' "${BENCH_CMD[@]}"
    printf '\n'
  } >>"${MANIFEST}"

  echo "Running HumanEval benchmark at request rate ${rate}"
  "${BENCH_CMD[@]}" >"${CLIENT_LOG}" 2>&1
  validate_result "${OUTPUT_FILE}" "${NUM_PROMPTS}" "${rate}" "${MAX_CONCURRENCY}"
done

echo "Completed HumanEval rate sweep; results=${RUN_DIR}"
