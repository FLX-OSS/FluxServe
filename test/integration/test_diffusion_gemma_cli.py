import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
RUN_CLI_SMOKE = "FLUXSERVE_RUN_DIFFUSION_GEMMA_CLI"
DATASET = Path(__file__).parents[1] / "data" / "diffusion_gemma_smoke.jsonl"


@pytest.mark.skipif(
    os.environ.get(RUN_CLI_SMOKE) != "1",
    reason=f"set {RUN_CLI_SMOKE}=1 to run the official checkpoint through the CLI",
)
def test_official_diffusion_gemma_offline_cli(tmp_path):
    model_name = os.environ.get("DIFFUSION_GEMMA_MODEL", MODEL_ID)
    command = [
        sys.executable,
        "-m",
        "fluxserve.cli",
        "bench_offline",
        "--model",
        model_name,
        "--dataset",
        str(DATASET),
        "--dataset-format",
        "openai",
        "--batch-size",
        "1",
        "--mini-batch-size",
        "1",
        "--tp-size",
        "1",
        "--dp-size",
        "1",
        "--ep-size",
        "1",
        "--gen-len",
        "8",
        "--block-length",
        "8",
        "--canvas-length",
        "8",
        "--max-denoising-steps",
        "2",
        "--attention-backend",
        "sdpa",
        "--kv-cache-layout",
        "dense",
        "--output-dir",
        str(tmp_path),
        "--log-file",
        "cli.log",
        "--exp-name",
        "diffusion-gemma-smoke",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    log = (tmp_path / "cli.log").read_text()
    assert "runner=DiffusionGemmaRunner" in log
    assert "canvas_length=8" in log
    assert "max_denoising_steps=2" in log
    assert "eos_ids=(1, 106, 50)" in log
    assert "[Iter=" in log

    result_files = list(tmp_path.glob("diffusion-gemma-smoke_*.jsonl"))
    assert len(result_files) == 1
    rows = [json.loads(line) for line in result_files[0].read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "diffusion-gemma-smoke"
    assert isinstance(row["answer"], str)
    assert row["generated_length"] >= 0
    assert all(math.isfinite(row[key]) for key in ("tpf", "tps", "fps"))
