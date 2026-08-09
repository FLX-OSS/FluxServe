import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


MODEL_BASE_ADDR = 0x7F0000000000
GRAPH_BASE_ADDR = 0x080000000000
REGION_SIZE = "1GB"
GRAPH_DIR_NAME = "graph"
HOOK_ARCHIVE_NAME = "hook_archive"
SAVE_RESULTS_NAME = "save_results.json"
LOAD_RESULTS_NAME = "load_results.json"


def _foundry_hook_path() -> Path:
    try:
        spec = importlib.util.find_spec("foundry.ops")
    except ModuleNotFoundError:
        spec = None
    if spec is None or spec.origin is None:
        pytest.skip("foundry.ops is not installed")

    hook_path = Path(spec.origin).resolve().parent / "libcuda_hook.so"
    if not hook_path.is_file():
        pytest.skip(f"Foundry CUDA hook is missing: {hook_path}")
    return hook_path


def _make_attention_workload(device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class SimpleAttention(nn.Module):
        def __init__(self, hidden_size: int = 64, num_heads: int = 4):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
            self.output = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, hidden_states):
            batch_size, sequence_length, hidden_size = hidden_states.shape
            query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)

            def split_heads(tensor):
                return tensor.view(
                    batch_size, sequence_length, self.num_heads, self.head_dim
                ).transpose(1, 2)

            attention = functional.scaled_dot_product_attention(
                split_heads(query),
                split_heads(key),
                split_heads(value),
                dropout_p=0.0,
                is_causal=False,
            )
            attention = attention.transpose(1, 2).contiguous().view(
                batch_size, sequence_length, hidden_size
            )
            return self.output(attention)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    attention = SimpleAttention().eval()
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.uniform_(-0.05, 0.05)
    hidden_states = torch.randn(1, 8, 64, device=device)
    return attention, hidden_states


def _initialize_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Foundry test subprocess")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    return device


def _run_save(workspace: Path) -> None:
    import foundry as fdry
    import torch

    device = _initialize_cuda()
    graph_dir = workspace / GRAPH_DIR_NAME
    hook_archive = workspace / HOOK_ARCHIVE_NAME
    graph_dir.mkdir(parents=True, exist_ok=True)

    fdry.set_allocation_region(MODEL_BASE_ADDR, fdry.parse_size(REGION_SIZE))
    attention, hidden_states = _make_attention_workload(device)

    warmup_start = time.perf_counter()
    eager_output = attention(hidden_states)
    torch.cuda.synchronize()
    warmup_ms = (time.perf_counter() - warmup_start) * 1000

    fdry.set_allocation_region(GRAPH_BASE_ADDR, fdry.parse_size(REGION_SIZE))
    capture_start = time.perf_counter()
    graph = fdry.CUDAGraph()
    graph_pool = torch.cuda.graph_pool_handle()
    with fdry.graph(graph, pool=graph_pool):
        output = attention(hidden_states)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - capture_start) * 1000

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, eager_output, rtol=1e-3, atol=1e-3)

    graph_path = graph_dir / "graph.json"
    serialization_start = time.perf_counter()
    graph.save(str(graph_path), output_tensors=output)
    serialization_ms = (time.perf_counter() - serialization_start) * 1000

    packing_start = time.perf_counter()
    fdry.pack_fatbins_to_folder(str(hook_archive))
    fdry.set_pack_fatbins_on_exit(False)
    packing_ms = (time.perf_counter() - packing_start) * 1000
    fdry.stop_allocation_region()

    results = {
        "warmup_ms": warmup_ms,
        "capture_ms": capture_ms,
        "serialization_ms": serialization_ms,
        "packing_ms": packing_ms,
        "baseline_startup_ms": warmup_ms + capture_ms,
    }
    (workspace / SAVE_RESULTS_NAME).write_text(json.dumps(results, indent=2))


def _run_load(workspace: Path) -> None:
    import foundry as fdry
    import torch

    device = _initialize_cuda()
    graph_path = workspace / GRAPH_DIR_NAME / "graph.json"
    hook_archive = workspace / HOOK_ARCHIVE_NAME

    module_load_start = time.perf_counter()
    fdry.load_cuda_modules_and_libraries(str(hook_archive))
    module_load_ms = (time.perf_counter() - module_load_start) * 1000

    fdry.set_allocation_region(MODEL_BASE_ADDR, fdry.parse_size(REGION_SIZE))
    attention, hidden_states = _make_attention_workload(device)

    fdry.set_allocation_region(GRAPH_BASE_ADDR, fdry.parse_size(REGION_SIZE))
    graph_load_start = time.perf_counter()
    graph, output = fdry.CUDAGraph.load(str(graph_path))
    graph_load_ms = (time.perf_counter() - graph_load_start) * 1000

    replay_start = time.perf_counter()
    graph.replay()
    torch.cuda.synchronize()
    first_replay_ms = (time.perf_counter() - replay_start) * 1000
    first_output = output.clone()
    eager_output = attention(hidden_states)
    torch.cuda.synchronize()
    torch.testing.assert_close(first_output, eager_output, rtol=1e-3, atol=1e-3)

    hidden_states.add_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    changed_output = output.clone()
    changed_eager = attention(hidden_states)
    torch.cuda.synchronize()
    torch.testing.assert_close(changed_output, changed_eager, rtol=1e-3, atol=1e-3)
    assert not torch.allclose(changed_output, first_output)
    fdry.stop_allocation_region()

    results = {
        "module_load_ms": module_load_ms,
        "graph_load_ms": graph_load_ms,
        "first_replay_ms": first_replay_ms,
        "restore_ready_ms": module_load_ms + graph_load_ms,
    }
    (workspace / LOAD_RESULTS_NAME).write_text(json.dumps(results, indent=2))


def _run_subprocess(mode: str, workspace: Path, hook_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    current_preload = env.get("LD_PRELOAD")
    env["LD_PRELOAD"] = (
        f"{hook_path}:{current_preload}" if current_preload else str(hook_path)
    )
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), f"--{mode}", str(workspace)],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _assert_subprocess_succeeded(mode: str, result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"Foundry {mode} subprocess failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Foundry's CUDA hook requires Linux")
def test_foundry_attention_save_load_and_replay(tmp_path: Path) -> None:
    hook_path = _foundry_hook_path()

    save_result = _run_subprocess("save", tmp_path, hook_path)
    _assert_subprocess_succeeded("SAVE", save_result)

    expected_artifacts = [
        tmp_path / GRAPH_DIR_NAME / "graph.json",
        tmp_path / GRAPH_DIR_NAME / "graph.cugraph",
        tmp_path / HOOK_ARCHIVE_NAME / "fatbin_image_packed.img",
        tmp_path / HOOK_ARCHIVE_NAME / "fatbin_entrypoint_packed.txt",
        tmp_path / SAVE_RESULTS_NAME,
    ]
    for artifact in expected_artifacts:
        assert artifact.is_file() and artifact.stat().st_size > 0, (
            f"Missing or empty Foundry artifact: {artifact}"
        )

    load_result = _run_subprocess("load", tmp_path, hook_path)
    _assert_subprocess_succeeded("LOAD", load_result)
    assert (tmp_path / LOAD_RESULTS_NAME).stat().st_size > 0

    save_timings = json.loads((tmp_path / SAVE_RESULTS_NAME).read_text())
    load_timings = json.loads((tmp_path / LOAD_RESULTS_NAME).read_text())
    all_timings = {**save_timings, **load_timings}
    assert all(math.isfinite(value) and value > 0 for value in all_timings.values())

    speedup = save_timings["baseline_startup_ms"] / load_timings["restore_ready_ms"]
    assert math.isfinite(speedup) and speedup > 0
    print(
        "\nFoundry attention startup benchmark\n"
        f"  warmup:          {save_timings['warmup_ms']:.3f} ms\n"
        f"  capture:         {save_timings['capture_ms']:.3f} ms\n"
        f"  serialization:   {save_timings['serialization_ms']:.3f} ms\n"
        f"  fatbin packing:  {save_timings['packing_ms']:.3f} ms\n"
        f"  module load:     {load_timings['module_load_ms']:.3f} ms\n"
        f"  graph load:      {load_timings['graph_load_ms']:.3f} ms\n"
        f"  first replay:    {load_timings['first_replay_ms']:.3f} ms\n"
        f"  startup speedup: {speedup:.3f}x"
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    workspace = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    if mode == "--save" and workspace is not None:
        _run_save(workspace)
    elif mode == "--load" and workspace is not None:
        _run_load(workspace)
