# CI/CD Design for FluxServe

This document proposes a CI workflow for FluxServe based on the design used by
TokenSpeed. It is an implementation blueprint: FluxServe currently has tests,
but the YAML task registry, CI pipeline helper, and GitHub Actions workflows
described below still need to be added.

## Goals

FluxServe needs to test code across machines with different GPU models and GPU
counts without putting hardware-specific logic into every test file or GitHub
workflow. The proposed design separates four concerns:

1. Test files contain assertions and test logic.
2. Test registrations assign GPU tests to suites such as `runtime-1gpu`.
3. YAML task specifications define triggers, runners, setup, and commands.
4. GitHub Actions discovers the YAML tasks and dispatches them to matching
   self-hosted runners.

The resulting flow is:

```text
Pull request, push, nightly schedule, or manual request
                         |
                         v
              GitHub Actions scan job
                         |
                         v
          Read and validate tests/ci/**/*.yaml
                         |
                         v
       Expand each task into task x runner jobs
                         |
                         v
        Matching self-hosted GPU runner starts
                         |
                         v
       Install -> test/eval/perf -> report -> clean
```

Humans define the tests, suites, runner labels, thresholds, and trigger policy.
GitHub Actions and Python scripts execute those rules automatically. An AI
agent is not required to select or approve tests.

## Proposed repository layout

```text
.github/workflows/
  lint.yml                    # CPU-only formatting and static checks
  pr-test.yml                 # per-commit GPU matrix
  nightly.yml                 # expensive eval/performance tasks

tests/
  ci/
    ut/                       # unit and integration task YAML
    eval/                     # model correctness task YAML
    perf/                     # performance task YAML
    README.md                 # task schema and runner-label conventions
  ci_system/
    pipeline.py               # scan and execute task specifications
    ci_register.py            # static test-suite registration
    ci_utils.py               # timeout, retry, reporting, cleanup
    install_deps.sh           # reproducible runner installation
  runtime/
    run_ci_suite.py           # collect and execute registered files

  flux-engine/                # existing engine tests
  flux-kernel/                # existing kernel correctness tests
  attention/                  # existing attention tests and benchmarks
  flux-metrics/               # existing metrics tests
  misc/                       # existing focused regression tests
```

The existing test directories do not need to be moved immediately. The suite
runner can recursively scan the current paths.

## Test layers

CI should be divided by cost and hardware requirements.

### CPU checks

Run these on GitHub-hosted CPU runners for every pull request:

- Formatting and linting.
- Import and packaging checks.
- CI pipeline unit tests.
- Tests that do not import or initialize CUDA.

These checks should be fast and should fail before scarce GPU runners are used.

### One-GPU correctness

Suggested initial targets include:

- `tests/flux-kernel/test_rmsnorm.py`
- `tests/flux-kernel/test_qk_rmsnorm.py`
- Single-GPU attention correctness tests.
- Model loading and model-hub regression tests.
- Paged KV-cache tests that only require one device.
- Single-GPU engine and CLI smoke tests.

### Multi-GPU correctness

Use separate suites for tests that require distributed execution:

- `tests/flux-engine/test_distributed_executor.py`
- `tests/flux-engine/test_distributed_launch.py`
- Tensor-parallel and expert-parallel model tests.
- Tests that explicitly require two, four, or eight visible GPUs.

A runner label is scheduling metadata, not validation. A runner labeled
`h100-2gpu` must be configured so that exactly the intended two GPUs are visible
through `CUDA_VISIBLE_DEVICES` or the cluster scheduler.

### Model evaluation

Evaluation jobs should start a real FluxServe server, wait for readiness, run a
dataset such as GSM8K, HumanEval, or GPQA, and enforce an accuracy threshold.
These jobs are usually nightly or manual because they download checkpoints and
occupy multiple GPUs for a long time.

### Performance regression

Performance tasks should record structured metrics such as:

- Time to first token.
- Inter-token latency.
- End-to-end latency.
- Input and output throughput.
- Requests per second.
- Peak GPU memory.

Performance should be compared only on a stable runner family with controlled
software and clocks. A functional PR check should not fail because an unrelated
cluster is temporarily noisy.

## YAML task specification

Each YAML file represents one logical task. Each declared runner label expands
into an independent CI matrix job.

Example one-GPU FluxServe task:

```yaml
api_version: ci.fluxserve.io/v1
name: ut-runtime-1gpu
type: ut

triggers:
  - per-commit
  - manual

runner:
  labels:
    - h100-1gpu
    - b200-1gpu
  env:
    b200-1gpu:
      FLUXSERVE_ATTENTION_BACKEND: flashinfer

env:
  CI: "true"

install:
  - bash tests/ci_system/install_deps.sh

ut:
  commands:
    - python3 tests/runtime/run_ci_suite.py --device cuda --suite runtime-1gpu

report:
  github_step_summary: true
```

Recommended task types are:

- `ut`: run one or more test commands.
- `server_smoke`: start FluxServe and send a small request.
- `eval`: start FluxServe and run an accuracy evaluation.
- `perf`: start FluxServe and run a benchmark with reference checks.

Recommended triggers are:

- `per-commit`: every applicable non-draft pull request and push to `main`.
- `manual`: selected through GitHub's **Run workflow** interface.
- `nightly`: scheduled expensive coverage.
- `debug`: temporary manual diagnostics.

Task priority can control contention for shared machines:

```yaml
priority: high
```

or per runner:

```yaml
priority:
  b200-1gpu: low
```

This is useful when a one-GPU task and an eight-GPU evaluation share the same
physical host.

## Registering test files into suites

GPU test files can opt into CI with a module-level marker:

```python
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")
```

A distributed test can register for a different suite:

```python
register_cuda_ci(est_time=180, suite="runtime-2gpu")
```

A file may participate in multiple suites:

```python
register_cuda_ci(est_time=120, suite="runtime-1gpu")
register_cuda_ci(est_time=180, suite="runtime-2gpu")
```

Runner-specific exclusions should be explicit:

```python
register_cuda_ci(
    est_time=120,
    suite="runtime-1gpu",
    disabled_on_runners=["linux-mi35*"],
    disabled_on_runners_reason="CUDA-only kernel",
)
```

The collector should parse these calls with Python's AST instead of importing
the test modules. Static discovery avoids CUDA initialization, checkpoint
loading, and optional-dependency failures during the lightweight scan job. The
registration arguments therefore need to be literal constants.

`est_time` is used for scheduling rather than pass/fail behavior. If a suite is
sharded, tests can be sorted by estimated duration and greedily assigned to the
currently lightest shard.

## How a pull request runs

The proposed `.github/workflows/pr-test.yml` should perform the following work:

1. Trigger only when FluxServe source, kernels, tests, packaging, or the CI
   configuration changes.
2. Skip the expensive GPU matrix while a pull request is a draft.
3. Check out the pull request in a lightweight scan job.
4. Run:

   ```bash
   python3 tests/ci_system/pipeline.py scan \
     --root tests/ci \
     --trigger per-commit
   ```

5. Convert the emitted JSON into a GitHub Actions matrix.
6. Run every entry on `${{ matrix.runner }}` with `fail-fast: false`.
7. Install and execute the task.
8. Upload a structured JSON result even if the task fails.
9. Clean the checkout, server processes, ports, virtual environment, and GPU
   state.

`fail-fast: false` is important because a failure on one GPU family should not
hide results from the other GPU families.

## Runner labels across clusters

Use hardware-oriented labels in task YAML:

```text
h100-1gpu
h100-2gpu
h100-8gpu
b200-1gpu
b200-4gpu
b200-8gpu
```

The physical cluster name should normally remain an infrastructure detail. For
example:

| TokenSpeed-style label | Physical location | Visible allocation |
|---|---|---|
| `h100-1gpu` | Cluster A | one H100 |
| `h100-2gpu` | Cluster A | two H100s on one node |
| `b200-4gpu` | Cluster B | four B200s on one node |
| `b200-8gpu` | Cluster B | eight B200s on one node |

Register each isolated machine or allocation as a GitHub self-hosted runner
with the corresponding label. GitHub then routes a matrix job to any online
runner with that label.

Before accepting work, a runner should verify:

```bash
nvidia-smi
python3 -c 'import torch; print(torch.cuda.device_count())'
```

Do not run this CI directly on a shared login node. FluxServe tests may start
servers, occupy ports, download large checkpoints, create files under `/tmp`,
and terminate stale child processes. Use a dedicated node, container, VM, or
scheduler allocation.

## Slurm integration

There are two reasonable Slurm deployments.

### Runner inside an allocation

Allocate the GPUs first and start a temporary GitHub runner inside the job. The
runner exits when the allocation ends. This preserves the existing GitHub
matrix without teaching the workflow about Slurm.

### Gateway runner submitting jobs

A persistent GitHub runner on a cluster gateway can translate labels into
`sbatch` options:

| CI label | Example Slurm request |
|---|---|
| `h100-1gpu` | `--partition=h100 --gres=gpu:h100:1` |
| `h100-2gpu` | `--partition=h100 --gres=gpu:h100:2` |
| `b200-8gpu` | `--partition=b200 --gres=gpu:b200:8` |

The wrapper must submit the job, wait for it, copy back the result JSON, and
return the Slurm exit status to GitHub. The simpler first implementation is a
runner inside a dedicated allocation.

## Executing a task locally or in a scheduler job

Matrix generation can be tested without GPUs:

```bash
python3 tests/ci_system/pipeline.py scan \
  --root tests/ci \
  --trigger per-commit
```

Preview one task without executing commands:

```bash
python3 tests/ci_system/pipeline.py execute \
  --config tests/ci/ut/ut-runtime-1gpu.yaml \
  --runner h100-1gpu \
  --work-dir "$PWD" \
  --print-plan \
  --dry-run
```

Execute the full task on an appropriate worker:

```bash
python3 tests/ci_system/pipeline.py execute \
  --config tests/ci/ut/ut-runtime-1gpu.yaml \
  --runner h100-1gpu \
  --work-dir "$PWD" \
  --print-plan \
  --result-json .ci-artifacts/result.json
```

For closer parity with GitHub Actions, split installation from execution:

```bash
python3 tests/ci_system/pipeline.py execute \
  --config tests/ci/ut/ut-runtime-1gpu.yaml \
  --runner h100-1gpu \
  --work-dir "$PWD" \
  --only-stage install \
  --keep-runner-state

python3 tests/ci_system/pipeline.py execute \
  --config tests/ci/ut/ut-runtime-1gpu.yaml \
  --runner h100-1gpu \
  --work-dir "$PWD" \
  --skip-stage install \
  --reuse-runner-state \
  --result-json .ci-artifacts/result.json
```

## Task execution lifecycle

The pipeline executor should:

1. Validate the YAML schema and selected runner.
2. Build the task environment, including `CI_RUNNER_LABEL`.
3. Create or reuse a runner-specific virtual environment.
4. Terminate stale FluxServe servers from an earlier failed job.
5. Run installation commands.
6. Run the selected task stages.
7. Enforce command and per-file timeouts.
8. Stop server and child processes in a `finally` block.
9. Record command output and parsed metrics.
10. Write `.ci-artifacts/result.json` and a GitHub step summary.
11. Clean temporary state unless explicitly asked to preserve it.

For `eval` and `perf`, the lifecycle includes a managed server:

```text
install FluxServe
       |
start fluxserve server and capture server.log
       |
poll a readiness URL until HTTP 200 or timeout
       |
run evaluation or performance client
       |
parse metrics and enforce thresholds
       |
stop the entire server process group
```

## Running registered test files

The suite runner should execute each registered file in a separate process
rather than one large `pytest tests` command. For example:

```bash
python3 /absolute/path/to/test_file.py
```

Per-file processes provide better isolation for CUDA state, distributed process
groups, crashes, and hangs. They also make per-file timeouts and diagnostics
straightforward.

The runner should report:

- Enabled and skipped files with reasons.
- Estimated and actual duration.
- Passed, failed, and timed-out files.
- Retry attempts, if enabled.
- CUDA coredump information when available.

Retries should be conservative. Retry known nondeterministic accuracy or
performance assertions, but do not hide syntax errors, import failures, or
deterministic correctness failures.

## Reporting and merge policy

Every task should produce a machine-readable result such as:

```json
{
  "ok": true,
  "task": "ut-runtime-1gpu",
  "type": "ut",
  "runner": "h100-1gpu",
  "executed_stages": ["install", "ut"],
  "command_results": []
}
```

Upload this file as a GitHub Actions artifact even on failure. The step summary
should show the runner, executed stages, failed command or file, accuracy
threshold result, and performance comparison when applicable.

Recommended required PR checks are:

- CPU lint and static checks.
- CPU unit tests.
- One representative one-GPU correctness suite.
- Multi-GPU correctness when distributed code changes.

Cross-vendor tests and large evaluations can initially be advisory or nightly
until runner reliability and test stability are established.

## Suggested implementation sequence

1. Add CPU lint and a small CPU pytest workflow.
2. Implement and unit-test YAML validation and matrix generation.
3. Add one `h100-1gpu` self-hosted runner and a small kernel task.
4. Add static test registration and per-file execution with timeouts.
5. Add B200 and multi-GPU runners.
6. Add a FluxServe server smoke task with readiness polling and cleanup.
7. Add nightly correctness evaluations with explicit score thresholds.
8. Add performance tracking after the functional CI is stable.

Start with a small reliable matrix. A fast, trusted one-GPU check provides more
value to pull requests than a broad matrix that frequently fails because of
runner or infrastructure problems.
