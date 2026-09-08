#!/usr/bin/bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Prepare dataset
EVALSCOPE_COMMIT=acd09b44384d53174768bb1063f675420f76fae9
python3 -m pip install "evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

[ -f build_swe_smith_dataset.py ] || wget https://raw.githubusercontent.com/modelscope/evalscope/${EVALSCOPE_COMMIT}/examples/perf/build_swe_smith_dataset.py \
    -O build_swe_smith_dataset.py

# Note: Only 71 conversations can be built
[ -f agentic_dataset_flash.json ] || python3 build_swe_smith_dataset.py \
    --model-path inclusionAI/LLaDA2.0-flash \
    --first-turn-length 2000 \
    --subsequent-turn-length 256 \
    --min-turns 3 \
    --max-turns 6 \
    --number 64 \
    --output-path agentic_dataset_flash.json \
    --num-workers 16
