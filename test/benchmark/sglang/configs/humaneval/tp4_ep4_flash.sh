set -euo pipefail

export PYTHONNOUSERSITE=1

python test/benchmark/sglang/bench_offline.py \
  --model-path inclusionAI/LLaDA2.0-flash \
  --dataset-path data/humaneval.jsonl \
  --batch-size 16 \
  --tp-size 1 \
  --ep-size 1 \
  --dp-size 1 \
  --moe-dp-size 1 \
  --gen-length 2048 \
  --dllm-block-size 64 \
  --dllm-algorithm LowConfidence \
  --attention-backend flashinfer \
  --max-running-requests 16 \
  --result-filename humaneval_bench_sglang_flash_humaneval_tp4_ep4.jsonl
