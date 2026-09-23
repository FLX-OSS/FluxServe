"""Online serving validation for Nemotron-Labs-Diffusion.

Covers the two things the offline harness structurally cannot: continuously
scheduled paged execution, and decode CUDA graphs. Neither is reachable from
``generate``, which allocates its own KV cache -- that invalidates a captured
graph -- and graphs are only prepared through ``prepare_online_cuda_graphs`` on
the served path.

The gate is a chain of equalities, each isolating one layer:

    offline dense  ==  online eager paged  ==  online with decode graphs

The first equality tests the online plan path: per-request seeds surviving
across plan calls, block-by-block scheduling, page allocation. The second tests
graph capture and replay alone, because everything else is held fixed. The
diffusion harness separately ties offline dense to the checkpoint's own
``generate``, so the chain reaches the reference.

Prompts go over the wire as ``input_ids`` rather than text, so online and
offline see byte-identical inputs with no tokenizer in the comparison.

Modes, run as separate processes because each loads the model:

    --mode offline   dense runner, one request at a time
    --mode serve     launch a server and run the scenarios (--graphs to capture)
    --mode compare   read the artifacts, gate, write the summary

See ``docs/serving/nemotron/nemotron-labs-diffusion-14B.md`` for configuration.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES = REPO_ROOT / "test" / "runtime" / "data" / "nemotron_ar_fixtures.json"
MODEL = "nvidia/Nemotron-Labs-Diffusion-14B"
REVISION = "f8c3e2c078e193599b8882d965b1001c456ba738"
BLOCK_LENGTH = 32
MAX_NUM_SEQS = 4
THRESHOLD = 0.9

# Optional served lanes, each with the offline baseline that can predict it. A
# different attention backend still has to reproduce the dense text, so it reuses
# the standard baseline; a different decoding mode or a thinking budget produces
# a different — still deterministic — answer, which only its own offline lane
# knows.
EXTRA_LANES = (
    ("flashinfer", "online_offline"),
    ("selfspec", "online_offline_selfspec"),
    ("thinking", "online_offline_thinking"),
    ("extension", "online_offline_extension"),
)


def load_fixtures(path: str) -> list[dict]:
    with open(path) as handle:
        manifest = json.load(handle)
    return manifest["diffusion"]


def extension_fixtures(fixtures: list[dict]) -> list[dict]:
    """Mechanical boundary fixtures; repeated prompts are not a quality test."""
    source = fixtures[0]["input_ids"]
    result = []
    for length in (16383, 16385, 32768):
        tokens = (source * ((length + len(source) - 1) // len(source)))[:length]
        result.append(dict(name=f"context_{length}", input_ids=tokens,
                           length=length, max_new_tokens=32,
                           temperature=0.0, seed=17))
    for seed in (17, 29):
        result.append(dict(fixtures[0], name=f"sample_07_seed_{seed}",
                           temperature=0.7, seed=seed))
    return result


def provenance() -> dict:
    def git(*command):
        try:
            return subprocess.check_output(
                ["git", *command], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "python": sys.executable,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }


# ---------------------------------------------------------------------------
# Offline lane
# ---------------------------------------------------------------------------


def budget_args(thinking, config):
    """The two thinking-budget fields the runner reads off a config object.

    ``apply_nemotron_runner_config`` takes them from an argparse-shaped object,
    which is what the server passes; offline lanes have no such object, so this
    stands in for one and resolves the marker the same way the server does.
    """
    from types import SimpleNamespace

    from fluxserve.backend.model_loader.nemotron import resolve_end_think_token_id

    if thinking is None:
        return None
    marker = resolve_end_think_token_id(config)
    if marker is None:
        raise ValueError(
            "this checkpoint declares no end-of-thinking token, so a thinking "
            "budget cannot be enforced"
        )
    return SimpleNamespace(
        max_thinking_tokens=int(thinking), end_think_token_id=int(marker)
    )


def run_offline(fixtures, device: str, *, decoding="threshold",
                thinking=None) -> dict:
    """Dense diffusion, one request at a time, decoded exactly as serving does."""
    from transformers import AutoConfig, AutoTokenizer

    from fluxserve.backend.execution.forward_batch_info import RunnerConfig
    from fluxserve.backend.execution.runners.nemotron import get_nemotron_runner
    from fluxserve.backend.model_loader.nemotron import (
        apply_nemotron_runner_config,
    )
    from fluxserve.backend.utils.server_args import ServerArgs

    config = AutoConfig.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    max_length = max(
        fixture["length"] + fixture["max_new_tokens"] for fixture in fixtures
    ) + BLOCK_LENGTH
    runner_config = RunnerConfig(
        gen_length=max(f["max_new_tokens"] for f in fixtures),
        block_length=BLOCK_LENGTH,
        max_length=max_length,
        threshold=THRESHOLD,
        attention_backend="sdpa",
    )
    budget = budget_args(thinking, config)
    apply_nemotron_runner_config(runner_config, config, budget)
    runner = get_nemotron_runner("sdpa", decoding)(
        model_config=config,
        server_args=ServerArgs(
            model_name=MODEL, model_config=config, device=device,
            max_num_seqs=1, max_model_len=max_length,
        ),
        runner_config=runner_config,
        device=device,
    )
    if decoding == "self_speculation":
        # The served path loads the draft adapter, so this lane has to as well
        # or the two are not running the same model.
        runner.load_draft_adapter()
    mask_id = runner.decoder.mask_id
    eos_ids = set(runner.decoder.eos_ids)

    results = {}
    for fixture in fixtures:
        prompt = torch.tensor(
            [fixture["input_ids"]], dtype=torch.long, device=device
        )
        output = runner.generate(
            prompt,
            prompt_lengths=[fixture["length"]],
            generation_lengths=[int(fixture["max_new_tokens"])],
            sampling_params=[{key: fixture[key] for key in ("temperature", "seed")
                              if key in fixture}],
        )
        ids = output[0, fixture["length"]:].detach().cpu().tolist()
        # Match the executor's publication rule so the comparison is about the
        # model, not about two different ways of trimming its output.
        stop = next((i for i, t in enumerate(ids) if t in eos_ids), None)
        if stop is not None:
            ids = ids[:stop]
        ids = [t for t in ids if t != mask_id and t not in eos_ids]
        results[fixture["name"]] = {
            "token_ids": ids,
            "text": tokenizer.decode(ids, skip_special_tokens=True),
            "stats": runner.last_stats[0],
        }
        if budget is not None:
            marker = int(budget.end_think_token_id)
            # Token ids are only visible on this side of the wire, so the
            # marker's position is gated here and the server lane is held to
            # this lane's text.
            results[fixture["name"]]["thinking"] = {
                "budget": int(budget.max_thinking_tokens),
                "marker": marker,
                "marker_index": ids.index(marker) if marker in ids else None,
                "bound": int(budget.max_thinking_tokens) + BLOCK_LENGTH,
            }
        print(f"[offline] {fixture['name']} {len(ids)} tokens", flush=True)
    return results


# ---------------------------------------------------------------------------
# Server lane
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, payload=None, timeout: float = 600):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url, data=data,
        headers={"content-type": "application/json"} if data else {},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def wait_ready(base_url: str, process, timeout: float = 2400) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"server exited during startup with code {process.returncode}"
            )
        try:
            status, body = request_json(f"{base_url}/health", timeout=5)
            if status == 200 and body == {"status": "ok"}:
                return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise AssertionError("timed out waiting for server readiness")


def launch_server(port: int, *, graphs: bool, max_model_len: int, log_path: Path,
                  tp_size: int = 1, backend: str = "fa4",
                  decoding: str = "threshold", max_num_seqs: int = MAX_NUM_SEQS,
                  thinking: int | None = None):
    command = [
        sys.executable, "-m", "fluxserve.cli.launch", "launch",
        "--model", MODEL,
        "--host", "127.0.0.1", "--port", str(port),
        "--tp-size", str(tp_size), "--dp-size", "1",
        "--ep-size", str(tp_size),
        "--max-num-seqs", str(max_num_seqs),
        "--max-model-len", str(max_model_len),
        "--max-new-tokens", "64",
        "--block-length", str(BLOCK_LENGTH),
        "--page-size", str(BLOCK_LENGTH),
        "--attention-backend", backend,
        "--kv-cache-layout", "paged",
        "--scheduler-policy", "paged",
        "--parallel-decoding", decoding,
        "--threshold", str(THRESHOLD),
        "--trust-remote-code",
    ]
    if backend == "flashinfer":
        # The Nemotron FlashInfer path is one task per request on the public
        # paged prefill wrapper; both modes have to say paged or it refuses.
        command += ["--flashinfer-cache-mode", "paged",
                    "--flashinfer-prefill-mode", "paged"]
    if thinking is not None:
        command += ["--max-thinking-tokens", str(thinking)]
    if graphs:
        command += ["--use-decode-cuda-graph", "--cuda-graph-decode-mode", "padded"]
    handle = open(log_path, "w")
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, handle, command


def complete(base_url: str, fixture: dict) -> str:
    status, body = request_json(
        f"{base_url}/v1/completions",
        {
            "model": MODEL,
            "input_ids": fixture["input_ids"],
            "max_tokens": int(fixture["max_new_tokens"]),
            **{key: fixture[key] for key in ("temperature", "seed") if key in fixture},
        },
    )
    if status != 200:
        raise AssertionError(f"completion failed for {fixture['name']}: {body}")
    return body["choices"][0]["text"]


def disconnect_midstream(port: int, fixture: dict) -> None:
    """Start a streaming request and drop the connection partway through."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    connection.request(
        "POST", "/v1/completions",
        body=json.dumps({
            "model": MODEL,
            "input_ids": fixture["input_ids"],
            "max_tokens": 64,
            "ignore_eos": True,
            "stream": True,
        }),
        headers={"content-type": "application/json"},
    )
    response = connection.getresponse()
    response.read(1)
    connection.close()


def run_serve(fixtures, *, graphs: bool, output_dir: Path, tp_size: int = 1,
              label: str | None = None, backend: str = "fa4",
              decoding: str = "threshold", max_num_seqs: int = MAX_NUM_SEQS,
              thinking: int | None = None, flood: bool = True) -> dict:
    max_model_len = max(
        fixture["length"] + fixture["max_new_tokens"] for fixture in fixtures
    ) + 4 * BLOCK_LENGTH
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    label = label or ("graphs" if graphs else "eager")
    log_path = output_dir / f"server_{label}.log"
    process, handle, command = launch_server(
        port, graphs=graphs, max_model_len=max_model_len, log_path=log_path,
        tp_size=tp_size, backend=backend, decoding=decoding,
        max_num_seqs=max_num_seqs, thinking=thinking,
    )
    record = {"graphs": graphs, "tp_size": tp_size, "label": label,
              "backend": backend, "decoding": decoding,
              "max_num_seqs": max_num_seqs, "thinking": thinking,
              "command": command, "scenarios": {}}
    try:
        wait_ready(base_url, process)
        print(f"[{label}] server ready on {port}", flush=True)
        record["metrics_after_startup"] = request_json(f"{base_url}/metrics")[1]

        # 1. One at a time: the baseline the offline lane is compared against.
        record["scenarios"]["sequential"] = {
            fixture["name"]: complete(base_url, fixture) for fixture in fixtures
        }
        print(f"[{label}] sequential done", flush=True)

        # 2. A single request while the graph buckets cover four rows, so the
        #    replay is padded.
        record["scenarios"]["single"] = {
            fixtures[0]["name"]: complete(base_url, fixtures[0])
        }
        record["metrics_after_single"] = request_json(f"{base_url}/metrics")[1]

        # 3. Mixed prompt lengths in flight together.
        with ThreadPoolExecutor(max_workers=len(fixtures)) as pool:
            texts = list(pool.map(lambda f: complete(base_url, f), fixtures))
        record["scenarios"]["concurrent"] = {
            fixture["name"]: text for fixture, text in zip(fixtures, texts)
        }
        print(f"[{label}] concurrent done", flush=True)

        # 4. The same prompt twice: no state may leak between requests.
        record["scenarios"]["repeat"] = {
            fixtures[0]["name"]: complete(base_url, fixtures[0])
        }

        # 5. Abandon a stream, then re-run everything: pages and request slots
        #    released by a cancellation must be reusable without corruption.
        disconnect_midstream(port, fixtures[0])
        time.sleep(5)
        record["scenarios"]["after_cancel"] = {
            fixture["name"]: complete(base_url, fixture) for fixture in fixtures
        }
        print(f"[{label}] after-cancel done", flush=True)

        # 6. More requests in flight than the scheduler has slots, so pages and
        #    request slots are recycled under pressure rather than at idle.
        if flood:
            queued = [fixtures[i % len(fixtures)] for i in range(3 * max_num_seqs)]
            with ThreadPoolExecutor(max_workers=len(queued)) as pool:
                texts = list(pool.map(lambda f: complete(base_url, f), queued))
            replies: dict[str, list[str]] = {}
            for fixture, text in zip(queued, texts):
                replies.setdefault(fixture["name"], []).append(text)
            record["flood_requests"] = len(queued)
            record["flood_replies_consistent"] = all(
                len(set(group)) == 1 for group in replies.values()
            )
            record["scenarios"]["flood"] = {
                name: group[0] for name, group in replies.items()
            }
            print(f"[{label}] flood of {len(queued)} done", flush=True)

        record["metrics_final"] = request_json(f"{base_url}/metrics")[1]
    finally:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=60)
        handle.close()
    record["server_exit_code"] = process.returncode
    record["provenance"] = provenance()
    return record


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(output_dir: Path) -> dict:
    def load(name):
        path = output_dir / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    offline = load("online_offline")
    eager = load("online_eager")
    graphed = load("online_graphs")
    if offline is None or eager is None:
        raise AssertionError("need the offline and eager artifacts to compare")

    record = {"graphs_present": graphed is not None, "fixtures": {}}
    names = list(offline["results"].keys())

    for name in names:
        expected = offline["results"][name]["text"]
        entry = {
            "offline_text": expected,
            "offline_tokens": len(offline["results"][name]["token_ids"]),
        }
        for label, lane in (("eager", eager), ("graphs", graphed)):
            if lane is None:
                continue
            scenarios = lane["scenarios"]
            entry[label] = {
                scenario: scenarios[scenario].get(name)
                for scenario in scenarios
                if name in scenarios[scenario]
            }
            entry[f"{label}_matches_offline"] = (
                scenarios["sequential"].get(name) == expected
            )
            entry[f"{label}_concurrent_matches_sequential"] = (
                scenarios["concurrent"].get(name)
                == scenarios["sequential"].get(name)
            )
            entry[f"{label}_after_cancel_matches_sequential"] = (
                scenarios["after_cancel"].get(name)
                == scenarios["sequential"].get(name)
            )
        if graphed is not None:
            entry["graphs_match_eager"] = (
                graphed["scenarios"]["sequential"].get(name)
                == eager["scenarios"]["sequential"].get(name)
            )
        record["fixtures"][name] = entry

    first = names[0]
    repeat_ok = {
        label: lane["scenarios"]["repeat"].get(first)
        == lane["scenarios"]["sequential"].get(first)
        for label, lane in (("eager", eager), ("graphs", graphed))
        if lane is not None
    }
    record["repeat_is_deterministic"] = repeat_ok

    # Tensor parallelism changes reduction order, so cross-TP text equality is
    # measured and reported rather than required: a temperature-zero trajectory
    # can diverge from a single near-tie. What *is* gated is that the TP=4
    # server is internally consistent, because that catches real sharding bugs.
    tp4 = load("online_tp4")
    record["tp4_present"] = tp4 is not None
    if tp4 is not None:
        agree = {
            name: tp4["scenarios"]["sequential"].get(name)
            == eager["scenarios"]["sequential"].get(name)
            for name in names
        }
        record["tp4"] = {
            "tp_size": tp4.get("tp_size"),
            "agrees_with_tp1": agree,
            "agreement_rate": sum(agree.values()) / max(len(agree), 1),
            "texts": tp4["scenarios"]["sequential"],
            "divergent_fixtures": sorted(n for n, ok in agree.items() if not ok),
        }

    checks = {
        "eager_online_matches_offline_dense": all(
            item["eager_matches_offline"] for item in record["fixtures"].values()
        ),
        "eager_concurrent_matches_sequential": all(
            item["eager_concurrent_matches_sequential"]
            for item in record["fixtures"].values()
        ),
        "eager_survives_cancellation": all(
            item["eager_after_cancel_matches_sequential"]
            for item in record["fixtures"].values()
        ),
        "repeats_are_deterministic": all(repeat_ok.values()),
        "eager_server_exited_cleanly": eager["server_exit_code"] in (0, -2, 130),
    }

    if graphed is not None:
        metrics = graphed.get("metrics_final") or {}
        startup = graphed.get("metrics_after_startup") or {}
        single = graphed.get("metrics_after_single") or {}
        record["graph_metrics"] = {
            "denoise_captures": startup.get("cuda_graph_denoise_capture_count"),
            "commit_captures": startup.get("cuda_graph_commit_capture_count"),
            "replays": metrics.get("cuda_graph_decode_replay_count"),
            "padded_rows_after_single": single.get("cuda_graph_decode_padded_rows"),
            "capture_time_s": startup.get("cuda_graph_capture_time_s"),
            "capture_memory_bytes": startup.get("cuda_graph_capture_memory_bytes"),
        }
        buckets = record["graph_metrics"]
        checks.update({
            "graphs_match_eager_online": all(
                item["graphs_match_eager"] for item in record["fixtures"].values()
            ),
            "graphs_match_offline_dense": all(
                item["graphs_matches_offline"] for item in record["fixtures"].values()
            ),
            "graphs_concurrent_matches_sequential": all(
                item["graphs_concurrent_matches_sequential"]
                for item in record["fixtures"].values()
            ),
            "graphs_survive_cancellation": all(
                item["graphs_after_cancel_matches_sequential"]
                for item in record["fixtures"].values()
            ),
            # Both phases must be captured, and in equal numbers: one denoise
            # and one commit variant per bucket.
            "both_phases_captured": bool(
                buckets["denoise_captures"]
                and buckets["commit_captures"]
                and buckets["denoise_captures"] == buckets["commit_captures"]
            ),
            "graphs_were_actually_replayed": bool(buckets["replays"]),
            # A single request under four-row buckets must pad, otherwise the
            # run silently fell back to eager and proved nothing about graphs.
            "padding_path_exercised": bool(buckets["padded_rows_after_single"]),
            "graphs_server_exited_cleanly": graphed["server_exit_code"]
            in (0, -2, 130),
        })
    if tp4 is not None:
        checks.update({
            "tp4_all_requests_completed": all(
                bool(text) for text in tp4["scenarios"]["sequential"].values()
            ),
            "tp4_concurrent_matches_sequential": all(
                tp4["scenarios"]["concurrent"].get(name)
                == tp4["scenarios"]["sequential"].get(name)
                for name in names
            ),
            "tp4_survives_cancellation": all(
                tp4["scenarios"]["after_cancel"].get(name)
                == tp4["scenarios"]["sequential"].get(name)
                for name in names
            ),
            "tp4_repeat_is_deterministic": (
                tp4["scenarios"]["repeat"].get(names[0])
                == tp4["scenarios"]["sequential"].get(names[0])
            ),
            "tp4_server_exited_cleanly": tp4["server_exit_code"] in (0, -2, 130),
        })

    for label, lane in (("eager", eager), ("graphs", graphed), ("tp4", tp4)):
        if lane is None or "flood" not in lane.get("scenarios", {}):
            continue
        checks[f"{label}_flood_matches_sequential"] = all(
            lane["scenarios"]["flood"].get(name)
            == lane["scenarios"]["sequential"].get(name)
            for name in lane["scenarios"]["flood"]
        )
        # Every copy of a prompt in the flood must come back the same, not just
        # the one that happened to be recorded.
        checks[f"{label}_flood_replies_agree"] = bool(
            lane.get("flood_replies_consistent")
        )

    record["extra_lanes"] = {}
    for label, baseline_name in EXTRA_LANES:
        lane = load(f"online_{label}")
        if lane is None:
            continue
        baseline = load(baseline_name)
        if baseline is None:
            raise AssertionError(
                f"lane '{label}' needs its offline baseline {baseline_name}.json"
            )
        entry = {
            "backend": lane.get("backend"),
            "decoding": lane.get("decoding"),
            "thinking": lane.get("thinking"),
            "baseline": baseline_name,
            "server_exit_code": lane["server_exit_code"],
            "fixtures": {},
        }
        for name, expected in baseline["results"].items():
            served = lane["scenarios"]["sequential"].get(name)
            item = {
                "matches_offline": served == expected["text"],
                "offline_tokens": len(expected["token_ids"]),
                "non_empty": bool(served),
            }
            for scenario in ("concurrent", "after_cancel", "flood"):
                served_scenario = lane["scenarios"].get(scenario) or {}
                if name in served_scenario:
                    item[f"{scenario}_matches_sequential"] = (
                        served_scenario[name] == served
                    )
            thinking = expected.get("thinking")
            if thinking is not None:
                item["thinking"] = thinking
                # The marker is invisible in the served text, so the bound is
                # checked on the offline lane's ids and the server is held to
                # that lane's text.
                item["marker_within_budget"] = (
                    thinking["marker_index"] is not None
                    and thinking["marker_index"] <= thinking["bound"]
                )
            entry["fixtures"][name] = item
        record["extra_lanes"][label] = entry

        items = list(entry["fixtures"].values())
        checks[f"{label}_matches_offline"] = bool(items) and all(
            item["matches_offline"] for item in items
        )
        checks[f"{label}_all_requests_completed"] = bool(items) and all(
            item["non_empty"] for item in items
        )
        for scenario in ("concurrent", "after_cancel", "flood"):
            key = f"{scenario}_matches_sequential"
            values = [item[key] for item in items if key in item]
            if values:
                checks[f"{label}_{key}"] = all(values)
        checks[f"{label}_server_exited_cleanly"] = (
            lane["server_exit_code"] in (0, -2, 130)
        )
        markers = [
            item["marker_within_budget"]
            for item in items
            if "marker_within_budget" in item
        ]
        if markers:
            checks[f"{label}_marker_within_budget"] = all(markers)
    record["checks"] = checks
    record["passed"] = all(checks.values())
    return record


def render(record: dict) -> str:
    lines = [
        "# Nemotron-Labs-Diffusion online serving validation",
        "",
        f"**Result: {'PASS' if record['passed'] else 'FAIL'}**",
        "",
        "The gate is a chain: offline dense == online eager paged == online with "
        "decode graphs. The first equality tests the continuously scheduled "
        "paged path, the second tests graph capture and replay with everything "
        "else held fixed.",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for key, value in record["checks"].items():
        lines.append(f"| {key} | {'pass' if value else '**FAIL**'} |")

    if record.get("graph_metrics"):
        lines += ["", "## Graph metrics", "", "| Key | Value |", "| --- | --- |"]
        for key, value in record["graph_metrics"].items():
            lines.append(f"| {key} | `{value}` |")

    lines += [
        "", "## Per fixture", "",
        "| Fixture | offline tokens | eager == offline | graphs == eager | "
        "concurrent == sequential | after cancel == sequential |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, item in record["fixtures"].items():
        lines.append(
            f"| {name} | {item['offline_tokens']} | "
            f"{item.get('eager_matches_offline')} | "
            f"{item.get('graphs_match_eager', 'n/a')} | "
            f"{item.get('eager_concurrent_matches_sequential')} | "
            f"{item.get('eager_after_cancel_matches_sequential')} |"
        )

    failures = [
        (name, item)
        for name, item in record["fixtures"].items()
        if not item.get("eager_matches_offline", True)
        or not item.get("graphs_match_eager", True)
    ]
    for name, item in failures[:3]:
        lines += [
            "", f"### Divergence in {name}", "",
            f"- offline: `{item['offline_text'][:200]}`",
            f"- eager:   `{str(item.get('eager', {}).get('sequential'))[:200]}`",
            f"- graphs:  `{str(item.get('graphs', {}).get('sequential'))[:200]}`",
        ]

    if record.get("tp4_present"):
        tp4 = record["tp4"]
        lines += [
            "", f"## Tensor parallelism (TP={tp4['tp_size']})", "",
            "Reduction order differs under sharding, so agreement with TP=1 is "
            "reported rather than gated; a temperature-zero trajectory can "
            "diverge from a single near-tie. The gated part is that the TP=4 "
            "server is internally consistent.",
            "",
            f"- agreement with TP=1: `{tp4['agreement_rate']:.2f}` "
            f"({sum(tp4['agrees_with_tp1'].values())}/"
            f"{len(tp4['agrees_with_tp1'])} fixtures)",
        ]
        if tp4["divergent_fixtures"]:
            lines.append(f"- divergent: `{tp4['divergent_fixtures']}`")

    for label, entry in (record.get("extra_lanes") or {}).items():
        lines += [
            "", f"## Lane `{label}`", "",
            f"- backend `{entry['backend']}`, decoding `{entry['decoding']}`, "
            f"thinking budget `{entry['thinking']}`, baseline "
            f"`{entry['baseline']}`, server exit `{entry['server_exit_code']}`",
            "",
            "| Fixture | offline tokens | == offline | concurrent | after cancel "
            "| flood | marker |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, item in entry["fixtures"].items():
            thinking = item.get("thinking") or {}
            marker = "n/a" if not thinking else (
                f"{thinking['marker_index']} <= {thinking['bound']}"
                if item.get("marker_within_budget")
                else f"**{thinking['marker_index']}** > {thinking['bound']}"
            )
            lines.append(
                f"| {name} | {item['offline_tokens']} | {item['matches_offline']} | "
                f"{item.get('concurrent_matches_sequential', 'n/a')} | "
                f"{item.get('after_cancel_matches_sequential', 'n/a')} | "
                f"{item.get('flood_matches_sequential', 'n/a')} | {marker} |"
            )

    lines += ["", "## Provenance", ""]
    lines.append(f"- repeat determinism: `{record.get('repeat_is_deterministic')}`")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("offline", "serve", "compare"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    parser.add_argument("--extension-fixtures", action="store_true",
                        help="exercise 16K/32K boundaries and seeded temperature 0.7")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--backend", default="fa4", choices=("fa4", "flashinfer"),
                        help="served attention backend; graphs require fa4")
    parser.add_argument("--decoding", default="threshold",
                        choices=("threshold", "self_speculation"))
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="scheduler slots; self-speculation forces 1")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="serve and decode offline with this token budget")
    parser.add_argument("--no-flood", action="store_true",
                        help="skip the oversubscription scenario")
    parser.add_argument(
        "--label",
        default=None,
        help="artifact name suffix; defaults to eager/graphs",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        record = compare(output_dir)
        (output_dir / "online_metrics.json").write_text(json.dumps(record, indent=2))
        (output_dir / "online_summary.md").write_text(render(record))
        print(json.dumps(record["checks"], indent=2), flush=True)
        print(f"ONLINE {'PASS' if record['passed'] else 'FAIL'}", flush=True)
        return 0 if record["passed"] else 1

    fixtures = load_fixtures(args.fixtures)
    if args.extension_fixtures:
        fixtures = extension_fixtures(fixtures)
    # Self-speculation is a batch-size-one mode in the reference and the runner
    # refuses to batch rather than silently serialising, so the slot count is a
    # requirement of the mode, not a tuning knob.
    max_num_seqs = args.max_num_seqs or (
        1 if args.decoding == "self_speculation" else MAX_NUM_SEQS
    )
    if args.mode == "offline":
        payload = {
            "results": run_offline(fixtures, args.device, decoding=args.decoding,
                                   thinking=args.thinking_budget),
            "provenance": provenance(),
            "decoding": args.decoding,
            "thinking": args.thinking_budget,
        }
        name = "online_offline" + (f"_{args.label}" if args.label else "")
    else:
        payload = run_serve(
            fixtures, graphs=args.graphs, output_dir=output_dir,
            tp_size=args.tp_size, label=args.label, backend=args.backend,
            decoding=args.decoding, max_num_seqs=max_num_seqs,
            thinking=args.thinking_budget, flood=not args.no_flood,
        )
        payload["results"] = {}
        name = f"online_{payload['label']}"
    (output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"[{args.mode}] wrote {name}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
