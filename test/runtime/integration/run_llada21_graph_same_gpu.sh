#!/usr/bin/env bash
# Same-card control after the main graph matrix, on the same 128 GSM8K questions.
set -euo pipefail
output_root=$1
for variant in fa4_eager fa4_graph batchprefill_eager batchprefill_graph; do
    backend=flashinfer
    graph_args=()
    if [[ "$variant" == fa4_* ]]; then backend=fa4; fi
    if [[ "$variant" == *_graph ]]; then graph_args=(--graph); fi
    output="$output_root/same_gpu_${variant}"
    python test/runtime/integration/bench_llada21_quality.py \
        --backend "$backend" "${graph_args[@]}" --dataset data/gsm8k.jsonl \
        --limit 128 --repeats 3 --concurrency 16 --max-num-seqs 16 --mini-batch-size 4 \
        --max-model-len 65536 --max-scheduled-tokens 2048 \
        --scheduler-num-device-pages 0 --gpu-memory-utilization 0.8 \
        --capture-bs 1 2 4 8 10 12 16 --output "$output"
    for round in 0 1 2; do
        python test/runtime/integration/score_llada21_gsm8k.py \
            --tests "$output_root/gsm8k_test.jsonl" --responses "$output/responses_${round}.jsonl" \
            --expected-count 128
    done
done
