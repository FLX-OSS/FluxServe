#!/usr/bin/env bash
# Invoke through run_fa4_graph_benchmark.sh on one allocated GPU per variant.
set -euo pipefail
variant=$1
output_root=$2
backend=flashinfer
graph_args=()
case "$variant" in
    fa4_eager) backend=fa4 ;;
    fa4_graph) backend=fa4; graph_args=(--graph) ;;
    batchprefill_eager) ;;
    batchprefill_graph) graph_args=(--graph) ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
for concurrency in 16 1 4 8; do
    limit=128
    if [[ "$concurrency" == 16 ]]; then limit=0; fi
    output="$output_root/${variant}_c${concurrency}"
    python test/runtime/integration/bench_llada21_quality.py \
        --backend "$backend" "${graph_args[@]}" \
        --dataset data/gsm8k.jsonl --limit "$limit" --repeats 3 \
        --concurrency "$concurrency" --max-num-seqs 16 --mini-batch-size 4 \
        --max-model-len 65536 --max-scheduled-tokens 2048 \
        --scheduler-num-device-pages 0 --gpu-memory-utilization 0.8 \
        --capture-bs 1 2 4 8 10 12 16 --output "$output"
    count=$limit
    if [[ "$count" == 0 ]]; then count=1319; fi
    for round in 0 1 2; do
        python test/runtime/integration/score_llada21_gsm8k.py \
            --tests "$output_root/gsm8k_test.jsonl" \
            --responses "$output/responses_${round}.jsonl" --expected-count "$count"
    done
done
