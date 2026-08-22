import json
import math
import os
import subprocess
import sys

import pytest

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
RUN_CLI_SMOKE = "FLUXSERVE_RUN_DIFFUSION_GEMMA_CLI"


@pytest.mark.skipif(
    os.environ.get(RUN_CLI_SMOKE) != "1",
    reason=f"set {RUN_CLI_SMOKE}=1 to run the official checkpoint through the CLI",
)
def test_official_diffusion_gemma_offline_cli(tmp_path):
    model_name = os.environ.get("DIFFUSION_GEMMA_MODEL", MODEL_ID)
    dataset = tmp_path / "batch-smoke.jsonl"
    dataset_rows = [
        {
            "messages": [{"role": "user", "content": "Name one prime number."}],
            "metadata": {"task_id": "short"},
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Explain in one concise sentence why the daytime sky "
                        "appears blue instead of violet."
                    ),
                }
            ],
            "metadata": {"task_id": "long"},
        },
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in dataset_rows))
    command = [
        sys.executable,
        "-m",
        "fluxserve.cli",
        "bench_offline",
        "--model",
        model_name,
        "--dataset",
        str(dataset),
        "--dataset-format",
        "openai",
        "--batch-size",
        "2",
        "--mini-batch-size",
        "2",
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
    assert len(rows) == 2
    rows_by_id = {row["id"]: row for row in rows}
    assert set(rows_by_id) == {"short", "long"}
    assert rows_by_id["short"]["generated_length"] <= 18
    assert rows_by_id["long"]["generated_length"] <= 8
    for row in rows:
        assert isinstance(row["answer"], str)
        assert row["generated_length"] >= 0
        assert all(math.isfinite(row[key]) for key in ("tpf", "tps", "fps"))
