# FA4 LLaDA2.1-mini CI

The four `test/ci/{eval,perf}/llada2.1-mini-fa4-{eager,graph}-gsm8k.yaml`
tasks are discoverable by the existing CI scanner for manual and per-commit
triggers. No changes to the FlashInfer configs or benchmark shell script are
required. Configuration discovery and execution-plan validation have passed;
GPU evaluation has not yet been run for these tasks.

Inside a GH200 allocation, enter the project container:

```bash
# Run from your checkout. Set paths for your own cluster.
export APPTAINER_IMAGE=/path/to/compatible-gh200.sif
export HF_BIND=/path/to/huggingface-cache:/mnt/huggingface
# Explicitly bypass any optional, machine-local environment.
export FLUXSERVE_CONTAINER_VENV=/nonexistent
bash test/ci_system/env/gh200.sh
```

The container must provide the normal FluxServe runtime dependencies, including
torch, flux-kernel and flux-scheduler (the GitHub workflow installs these).
The task install stage creates `.ci-artifacts/fa4-runtime` with system site
packages, installs FluxServe from the current checkout and pinned standalone
FA4 dependencies, and installs pinned EvalScope in `.ci-artifacts/fa4-evalscope`.
The YAML activates the runtime and calls `fluxserve serve` directly. It does not
use `serve.py`, a personal `.venvs` directory, or a user-specific checkout path.
Use a compatible base environment (validated package target: Python 3.12,
PyTorch 2.8.0+cu129 on GH200). A bare GitHub-hosted CPU runner cannot run this GPU
evaluation; the repository workflow selects a self-hosted GH200 runner.
Do not run multiple tasks concurrently in one container: they share port 8000,
runtime directories and the pipeline server-log path.

```bash
python -m pip install PyYAML
MODE=graph  # or eager
for KIND in eval perf; do
  python test/ci_system/pipeline.py execute \
    --config "test/ci/$KIND/llada2.1-mini-fa4-$MODE-gsm8k.yaml" \
    --runner gh200-1gpu --work-dir "$PWD" --print-plan \
    --result-json ".ci-artifacts/$KIND-fa4-$MODE-result.json" || break
  cp .ci-artifacts/server.log ".ci-artifacts/$KIND-fa4-$MODE-server.log"
done
```

Accuracy uses full GSM8K with eval batch size 4, rule judging, temperature 0,
max_tokens 2048, and the inherited minimum score 0.90. Perf uses 1000 requests,
rate 16, concurrency limit 16 and non-streaming responses. A postcheck requires
exactly 1000 successful requests and zero failures. No performance regression
threshold is claimed until a same-environment FA4 baseline is established.
These tasks test serving quality/performance, not attention/SDPA numeric parity.

Outputs are under `.ci-artifacts/{eval,perf}-llada21-fa4-{eager,graph}/`.
Archive previous output directories before repeating a task; the
install stage rejects existing task output directories. Inspect the JSON result for failure
details and preserve `.ci-artifacts/server.log` even when a task fails.

```bash
tar --exclude='fa4-runtime' --exclude='fa4-evalscope' \
  -czf fa4-ci-artifacts.tar.gz -C .ci-artifacts .
```
