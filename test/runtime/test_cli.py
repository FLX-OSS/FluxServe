import os
import subprocess
import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest

from fluxserve.cli import build_parser, main


@pytest.mark.parametrize(
    ("command", "module_name", "handler_name", "options"),
    [
        ("launch", "fluxserve.cli.launch", "launch", ["--model", "model"]),
        ("bench", "fluxserve.cli.bench", "run_serving_benchmark", ["--model", "model", "--dataset", "data.jsonl"]),
        ("bench_offline", "fluxserve.cli.bench_offline", "bench_offline", ["--model", "model", "--dataset", "data.jsonl"]),
        ("env", "fluxserve.cli.env", "main", []),
    ],
)
def test_dispatch(monkeypatch, command, module_name, handler_name, options):
    handler = AsyncMock() if command == "bench" else Mock()
    module = ModuleType(module_name)
    setattr(module, handler_name, handler)
    monkeypatch.setitem(sys.modules, module_name, module)

    main([command, *options])

    if command == "env":
        handler.assert_called_once_with()
    else:
        handler.assert_called_once()
        assert handler.call_args.args[0].command == command
    if command == "bench":
        handler.assert_awaited_once()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["serve", "--model", "model"],
        ["bench", "serve", "--model", "model", "--dataset", "data.jsonl"],
        ["launch"],
        ["bench", "--model", "model"],
        ["bench_offline", "--model", "model"],
    ],
)
def test_rejects_removed_commands_and_missing_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_benchmark_metrics_and_json_options():
    args = build_parser().parse_args([
        "bench", "--model", "model", "--dataset", "data.jsonl",
        "--metrics", "queue,e2e", "--extra-body", '{"temperature": 0}',
    ])
    assert args.metrics == ("E2E", "QUEUE")
    assert args.extra_body == {"temperature": 0}
    assert args.timeout == 3600
    assert args.request_rate == float("inf")
    assert not hasattr(args, "bench_type")
    assert not hasattr(args, "dispatch_function")


@pytest.mark.parametrize("metrics", ["", "QUEUE", "E2E,UNKNOWN"])
def test_rejects_invalid_metrics(metrics):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([
            "bench", "--model", "model", "--dataset", "data.jsonl",
            "--metrics", metrics,
        ])
    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["launch", "bench_offline"])
def test_backend_defaults_and_explicit_tracking(command):
    options = [command, "--model-name", "model"]
    if command == "bench_offline":
        options += ["--dataset", "data.jsonl", "--batch_size", "4"]
    args = build_parser().parse_args(options)
    assert args.model_name == "model"
    assert args.attention_backend == ("fa4" if command == "launch" else "flashinfer")
    assert args.attention_backend_explicit is False
    args = build_parser().parse_args(options + ["--attention-backend", "sdpa"])
    assert args.attention_backend == "sdpa"
    assert args.attention_backend_explicit is True
    if command == "bench_offline":
        assert args.batch_size == 4


def test_help_does_not_import_command_runtimes():
    # A fresh process detects eager imports even when other tests loaded a runtime.
    script = '''
import importlib.abc
import runpy
import shutil
import sys

blocked = (
    "fluxserve.cli.launch", "fluxserve.cli.bench", "fluxserve.cli.bench_offline", "fluxserve.cli.env",
    "fluxserve.backend.execution.runners", "fluxserve.backend.engine",
    "fluxserve.backend.entrypoints.http_server", "flux_kernel",
)
class BlockRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("Help imported runtime: " + fullname)

sys.meta_path.insert(0, BlockRuntimeImports())
console = shutil.which("fluxserve")
assert console is not None, "Install FluxServe to run its console entrypoint"
for command in ([], ["launch"], ["bench"], ["bench_offline"], ["env"]):
    sys.argv = ["fluxserve", *command, "--help"]
    try:
        runpy.run_path(console, run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
assert not any(name in sys.modules for name in blocked)
'''
    result = subprocess.run(
        [sys.executable, "-c", script], env=os.environ.copy(),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "{launch,bench,bench_offline,env}" in result.stdout
