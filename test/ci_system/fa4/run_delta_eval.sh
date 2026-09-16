#!/usr/bin/env bash
# Run after allocating four Delta H200 GPUs. All installs stay outside the SIF.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" != "--inside" ]]; then
    export FA4_EVAL_RUN_DIR="${FA4_EVAL_RUN_DIR:-$ROOT/.ci-artifacts/delta-fa4-eval-$(date +%Y%m%d-%H%M%S)}"
    exec apptainer exec --nv \
        --bind "$ROOT:$ROOT" \
        --bind "${FA4_HF_CACHE:-/work/nvme/bekz/yzhao25/huggingface}:/mnt/huggingface:ro" \
        --pwd "$ROOT" \
        "${APPTAINER_IMAGE:-/projects/bekz/yzhao25/sglang-h200.sif}" \
        bash "$ROOT/test/ci_system/fa4/run_delta_eval.sh" --inside
fi

python - <<'PY'
import torch
assert torch.cuda.device_count() == 4, 'Run inside a four-GPU Slurm allocation'
for i in range(4):
    assert torch.cuda.get_device_capability(i)[0] == 9, 'This preset targets H200/Hopper'
    print(i, torch.cuda.get_device_name(i))
PY

mkdir -p "$FA4_EVAL_RUN_DIR"
export PYTHONPATH="$ROOT/python:$ROOT/flux-kernel/python"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ TOKENIZERS_PARALLELISM=false
export HF_HOME="$FA4_EVAL_RUN_DIR/hf"
export HF_HUB_CACHE=/mnt/huggingface/hub
export HF_DATASETS_CACHE="$FA4_EVAL_RUN_DIR/datasets-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR="$ROOT/.ci-artifacts/delta-fa4-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.ci-artifacts/delta-fa4-cache/inductor"
export TORCH_EXTENSIONS_DIR="$ROOT/.ci-artifacts/delta-fa4-cache/extensions"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
RUNTIME="$ROOT/.ci-artifacts/delta-fa4-runtime"
EVALENV="$ROOT/.ci-artifacts/delta-fa4-evalscope"
export FA4_RUNTIME="$RUNTIME" FA4_EVALENV="$EVALENV"

python -m venv --system-site-packages "$RUNTIME"
python -c 'import torch; print("torch==" + torch.__version__)' > "$RUNTIME/constraints.txt"
"$RUNTIME/bin/python" -m pip install -c "$RUNTIME/constraints.txt" \
    -e '.[fa4]' ./flux-scheduler PyYAML
python -m venv "$EVALENV"
"$EVALENV/bin/python" -m pip install \
    'evalscope @ git+https://github.com/modelscope/evalscope.git@acd09b44384d53174768bb1063f675420f76fae9' \
    pyarrow PyYAML

"$RUNTIME/bin/python" - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from flash_attn.cute import flash_attn_varlen_func
import flux_kernel, flux_scheduler
root = Path(snapshot_download('inclusionAI/LLaDA2.1-mini', local_files_only=True))
index = json.loads((root / 'model.safetensors.index.json').read_text())
missing = [name for name in set(index['weight_map'].values()) if not (root / name).is_file()]
assert not missing, f'Missing model shards: {missing}'
print('Local model:', root)
PY

"$EVALENV/bin/python" - <<'PY'
import json
import os
import shlex
from pathlib import Path
import pyarrow.parquet as pq
import yaml

run = Path(os.environ['FA4_EVAL_RUN_DIR'])
cache = Path('/mnt/huggingface/hub/datasets--openai--gsm8k')
revision = (cache / 'refs/main').read_text().strip()
snapshot = cache / 'snapshots' / revision / 'main'
local = run / 'gsm8k'
local.mkdir(exist_ok=True)
for split, expected in [('train', 7473), ('test', 1319)]:
    files = sorted(snapshot.glob(f'{split}-*.parquet'))
    assert files, f'Missing {split} parquet under {snapshot}'
    rows = [row for path in files for row in pq.read_table(path).to_pylist()]
    assert len(rows) == expected, f'{split}: expected {expected} rows, got {len(rows)}'
    with (local / f'main_{split}.jsonl').open('w') as output:
        for row in rows:
            assert 'question' in row and '####' in row['answer']
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Local GSM8K {split}: {len(rows)} examples')

# The pinned GSM8K adapter calls datasets.load_dataset(path, name='main').
# Declare the named config explicitly, including separate few-shot/train data.
metadata = {'configs': [{'config_name': 'main', 'data_files': [
    {'split': split, 'path': f'main_{split}.jsonl'} for split in ('train', 'test')
]}]}
(local / 'README.md').write_text('---\n' + yaml.safe_dump(metadata) + '---\n')

source = Path('test/ci/eval/llada2.1-mini-fa4-graph-gsm8k.yaml')
task = yaml.safe_load(source.read_text())
task['install'] = []  # Dependencies were installed into external virtualenvs above.
task['env'].update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
server = shlex.split(task['server']['command'].split('&&', 1)[-1])
server[server.index('--ep-size') + 1] = '4'
assert server[server.index('--tp-size') + 1] == '4'
server[:1] = [str(Path(os.environ['FA4_RUNTIME']) / 'bin/python'), '-m', 'fluxserve.cli']
task['server']['command'] = shlex.join(server)
evaluation = shlex.split(task['eval']['command'])
evaluation[0] = str(Path(os.environ['FA4_EVALENV']) / 'bin/python')
evaluation[evaluation.index('--work-dir') + 1] = str(run / 'eval')
evaluation += ['--dataset-args', json.dumps({'gsm8k': {'dataset_id': str(local)}})]
task['eval']['command'] = shlex.join(evaluation)
(run / 'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
print('Generated config:', run / 'task.yaml')
PY

# Preserve the active virtualenv for pipeline subprocesses as well.
source "$RUNTIME/bin/activate"
set +e
python test/ci_system/pipeline.py execute \
    --config "$FA4_EVAL_RUN_DIR/task.yaml" --runner gh200-1gpu \
    --work-dir "$ROOT" --print-plan \
    --result-json "$FA4_EVAL_RUN_DIR/result.json" \
    2>&1 | tee "$FA4_EVAL_RUN_DIR/pipeline.log"
status=${PIPESTATUS[0]}
if [[ -f .ci-artifacts/server.log ]]; then
    cp .ci-artifacts/server.log "$FA4_EVAL_RUN_DIR/server.log"
fi
printf 'Exit status: %s\nResults: %s\n' "$status" "$FA4_EVAL_RUN_DIR"
exit "$status"
