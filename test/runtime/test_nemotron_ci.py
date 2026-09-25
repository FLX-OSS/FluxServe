"""The Nemotron CI task specs, checked against the code they drive.

A CI YAML is a second copy of the serving configuration, so it can drift from
the CLI silently and only fail on a runner an hour into a job. These tests run
every Nemotron task's server command through the real argument parser and the
real normalizer, and validate the task through the pipeline's own validator.
"""

import functools
import pathlib
import shlex
import sys

import pytest

from nemotron_test_utils import MODEL_SIZES, checkpoint_config

CI_ROOT = pathlib.Path(__file__).resolve().parents[1] / "ci"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def nemotron_tasks():
    yaml = pytest.importorskip("yaml")
    tasks = []
    for path in sorted(CI_ROOT.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if "nemotron" in str(data.get("name", "")):
            tasks.append((path, data))
    return tasks


@functools.lru_cache(maxsize=1)
def pipeline_module():
    sys.path.insert(0, str(REPO_ROOT / "test" / "ci_system"))
    import pipeline

    return pipeline


def server_args_for(data):
    """Parse a task's server command exactly as the runner will."""
    from fluxserve.cli.launch import build_parser

    tokens = shlex.split(data["server"]["command"])
    return build_parser().parse_args(tokens[tokens.index("launch"):])


def task_ids():
    return [path.name for path, _ in nemotron_tasks()]


def test_there_are_nemotron_ci_tasks_at_all():
    assert nemotron_tasks(), "no Nemotron task specs were found under test/ci"


def test_tasks_pass_the_pipelines_own_validator():
    pipeline = pipeline_module()
    for path, data in nemotron_tasks():
        pipeline.validate_task(data, path)


@pytest.mark.parametrize("index", range(len(nemotron_tasks())), ids=task_ids())
def test_every_task_matches_the_upstream_resource_layout(index):
    path, data = nemotron_tasks()[index]
    labels = data["runner"]["labels"]
    args = server_args_for(data)
    assert labels == [f"gh200-{args.tp_size}gpu"]
    assert path.parent.parent.name == f"1N{args.tp_size}G"
    assert args.dp_size == 1, "data parallelism is not part of this integration"
    assert args.tp_size == args.ep_size, (
        "attention and expert parallelism share ranks in this stack"
    )
    assert args.tp_size == 1, "only single-GPU serving has been verified"


def test_tasks_use_current_runtime_and_isolated_evaluation_environment():
    for path, data in nemotron_tasks():
        text = path.read_text()
        assert "test/ci_system/fa4/" not in text
        assert ".ci-artifacts/fa4-runtime" not in text
        assert "fluxserve launch" in data["server"]["command"]
        assert data["eval"]["command"].startswith("/tmp/evalscope-venv/bin/python ")
        assert "acd09b44384d53174768bb1063f675420f76fae9" in data["eval"]["install"][0]


@pytest.mark.parametrize("index", range(len(nemotron_tasks())), ids=task_ids())
def test_every_server_command_parses_and_normalizes(index):
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args

    _, data = nemotron_tasks()[index]
    args = server_args_for(data)
    assert normalize_nemotron_args(args, checkpoint_config(args.model_name)) is True
    assert args.block_length == 32, "the checkpoint's own block_size"


@pytest.mark.parametrize("index", range(len(nemotron_tasks())), ids=task_ids())
def test_context_stays_inside_the_supported_window(index):
    from fluxserve.backend.model_loader.nemotron import MAX_SUPPORTED_POSITIONS

    _, data = nemotron_tasks()[index]
    args = server_args_for(data)
    assert args.max_model_len <= MAX_SUPPORTED_POSITIONS, (
        "query temperature scaling is unvalidated beyond this, so a task that "
        "asks for more would be rejected at startup"
    )


@pytest.mark.parametrize("index", range(len(nemotron_tasks())), ids=task_ids())
def test_graph_tasks_satisfy_the_capture_preconditions(index):
    _, data = nemotron_tasks()[index]
    args = server_args_for(data)
    if not args.use_decode_cuda_graph:
        return
    assert args.attention_backend in ("fa4", "flashinfer"), (
        "graphs exist only on the paged path"
    )
    assert args.cuda_graph_decode_mode == "padded"
    page_size = args.page_size or args.block_length
    assert page_size == args.block_length, (
        "the graph replay maps absolute position to page by integer division, "
        "which assumes page_size == block_length"
    )
    buckets = args.cuda_graph_capture_bs
    assert buckets, "padded capture needs explicit buckets"
    assert max(buckets) >= args.max_num_seqs, (
        f"buckets {buckets} do not cover max_num_seqs {args.max_num_seqs}, so a "
        "full batch would find no graph"
    )


def test_every_model_size_has_a_gpu_gate_in_the_pr_matrix():
    expected = {
        f"eval-nemotron-diffusion-{size.lower()}-gsm8k"
        for size in MODEL_SIZES
    }
    per_commit = {
        data["name"] for _, data in nemotron_tasks()
        if "per-commit" in data["triggers"]
    }
    assert per_commit == expected
    matrix = pipeline_module().build_matrix(
        CI_ROOT / "1N1G", REPO_ROOT, trigger="per-commit"
    )
    discovered = {entry["name"] for entry in matrix["include"]}
    assert expected <= discovered


@pytest.mark.parametrize("index", range(len(nemotron_tasks())), ids=task_ids())
def test_gates_prefer_a_graph_captured_paged_backend(index):
    _, data = nemotron_tasks()[index]
    args = server_args_for(data)
    assert args.attention_backend in ("fa4", "flashinfer")
    assert args.use_decode_cuda_graph
    tokens = shlex.split(data["eval"]["command"])
    assert tokens[tokens.index("--model") + 1] == args.model_name


def test_evaluation_outputs_are_unique_across_model_sizes():
    outputs = []
    for _, data in nemotron_tasks():
        tokens = shlex.split(data["eval"]["command"])
        outputs.append(tokens[tokens.index("--work-dir") + 1])
    assert len(outputs) == len(set(outputs))
