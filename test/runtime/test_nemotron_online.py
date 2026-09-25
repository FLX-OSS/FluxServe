"""The online validation gate, exercised without a GPU.

The lanes themselves need an H100 and a running server, but the part that
decides pass or fail is ordinary logic and worth pinning down: it must attribute
a divergence to the layer that caused it, and it must refuse to pass a graphs
lane that never actually replayed a padded bucket.
"""

import functools
import importlib.util
import json
import pathlib
import sys

import pytest

NAMES = ["chat0_gen32", "chat1_gen64"]
TEXT = {name: f"answer for {name}" for name in NAMES}


@functools.lru_cache(maxsize=1)
def harness():
    path = (
        pathlib.Path(__file__).resolve().parent
        / "integration"
        / "nemotron_online_harness.py"
    )
    spec = importlib.util.spec_from_file_location("nemotron_online_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_offline(directory, texts=None):
    texts = texts or TEXT
    (directory / "online_offline.json").write_text(json.dumps({
        "results": {
            name: {"text": texts[name], "token_ids": [1, 2, 3], "stats": {}}
            for name in NAMES
        },
        "provenance": {"lane": "offline"},
    }))


def write_lane(directory, label, texts=None, *, captures=(3, 3), replays=12,
               padded=8, exit_code=0, concurrent=None, after_cancel=None,
               repeat=None):
    texts = texts or TEXT
    scenarios = {
        "sequential": dict(texts),
        "concurrent": dict(concurrent or texts),
        "after_cancel": dict(after_cancel or texts),
        "single": {NAMES[0]: texts[NAMES[0]]},
        "repeat": {NAMES[0]: (repeat or texts)[NAMES[0]]},
    }
    (directory / f"online_{label}.json").write_text(json.dumps({
        "graphs": label == "graphs",
        "command": [],
        "scenarios": scenarios,
        "metrics_after_startup": {
            "cuda_graph_denoise_capture_count": captures[0],
            "cuda_graph_commit_capture_count": captures[1],
        },
        "metrics_after_single": {"cuda_graph_decode_padded_rows": padded},
        "metrics_final": {"cuda_graph_decode_replay_count": replays},
        "server_exit_code": exit_code,
        "results": {},
        "provenance": {"lane": label},
    }))


def gate(directory):
    return harness().compare(directory)


def failures(record):
    return sorted(key for key, value in record["checks"].items() if not value)


def test_extension_fixtures_and_http_preserve_sampling(monkeypatch):
    module = harness()
    source = [dict(name="short", input_ids=[1, 2, 3], length=3, max_new_tokens=64)]
    fixtures = module.extension_fixtures(source)
    assert [f["length"] for f in fixtures[:3]] == [16383, 16385, 32768]
    assert all(len(f["input_ids"]) == f["length"] for f in fixtures)
    assert [f["seed"] for f in fixtures[3:]] == [17, 29]
    requests = []

    def request(url, payload):
        requests.append(payload)
        return 200, {"choices": [{"text": "answer"}]}

    monkeypatch.setattr(module, "request_json", request)
    assert module.complete("http://localhost", fixtures[-1]) == "answer"
    assert requests[0]["temperature"] == 0.7
    assert requests[0]["seed"] == 29


def test_every_layer_agreeing_passes(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    write_lane(tmp_path, "graphs")
    record = gate(tmp_path)
    assert record["passed"], failures(record)
    assert record["graphs_present"] is True
    assert record["graph_metrics"]["denoise_captures"] == 3
    assert harness().render(record).startswith(
        "# Nemotron-Labs-Diffusion online serving validation"
    )


def test_a_broken_online_path_is_not_blamed_on_graphs(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager", {NAMES[0]: "wrong", NAMES[1]: TEXT[NAMES[1]]})
    write_lane(tmp_path, "graphs")
    record = gate(tmp_path)
    assert not record["passed"]
    assert "eager_online_matches_offline_dense" in failures(record)
    # The graphs lane still matches the offline reference, so the break is the
    # plan path rather than capture.
    assert record["checks"]["graphs_match_offline_dense"] is True


def test_graphs_diverging_from_eager_is_attributed_to_capture(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    write_lane(tmp_path, "graphs", {NAMES[0]: "different", NAMES[1]: TEXT[NAMES[1]]})
    record = gate(tmp_path)
    assert not record["passed"]
    assert record["checks"]["eager_online_matches_offline_dense"] is True
    assert record["checks"]["graphs_match_eager_online"] is False


def test_a_graphs_lane_that_never_padded_cannot_pass(tmp_path):
    """Otherwise a silent eager fallback would look like a graph success."""
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    write_lane(tmp_path, "graphs", padded=0)
    record = gate(tmp_path)
    assert failures(record) == ["padding_path_exercised"]


def test_a_missing_phase_capture_cannot_pass(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    write_lane(tmp_path, "graphs", captures=(3, 0))
    assert failures(gate(tmp_path)) == ["both_phases_captured"]

    write_lane(tmp_path, "graphs", captures=(3, 2))
    assert "both_phases_captured" in failures(gate(tmp_path)), (
        "one variant per bucket per phase, so the counts must be equal"
    )


def test_graphs_that_never_replayed_cannot_pass(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    write_lane(tmp_path, "graphs", replays=0)
    assert failures(gate(tmp_path)) == ["graphs_were_actually_replayed"]


def test_concurrency_and_cancellation_are_gated_separately(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager", concurrent={NAMES[0]: "raced", NAMES[1]: "x"})
    write_lane(tmp_path, "graphs")
    assert "eager_concurrent_matches_sequential" in failures(gate(tmp_path))

    write_lane(tmp_path, "eager", after_cancel={NAMES[0]: "stale", NAMES[1]: "y"})
    assert "eager_survives_cancellation" in failures(gate(tmp_path))


def test_nondeterministic_repeats_cannot_pass(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager", repeat={NAMES[0]: "drifted"})
    write_lane(tmp_path, "graphs")
    assert "repeats_are_deterministic" in failures(gate(tmp_path))


def test_an_unclean_server_exit_cannot_pass(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager", exit_code=1)
    write_lane(tmp_path, "graphs")
    assert "eager_server_exited_cleanly" in failures(gate(tmp_path))


def test_the_graphs_lane_is_optional_but_the_others_are_not(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    record = gate(tmp_path)
    assert record["passed"] and record["graphs_present"] is False
    assert not any(key.startswith("graphs_") for key in record["checks"])
    assert "_No FA4" not in harness().render(record)

    (tmp_path / "online_eager.json").unlink()
    with pytest.raises(AssertionError, match="need the offline and eager"):
        gate(tmp_path)


@pytest.mark.parametrize("model_path", [None, "/tmp/nemotron-3b-pinned"])
def test_server_command_carries_the_paged_and_graph_flags(model_path):
    module = harness()
    _, handle, command = None, None, None
    try:
        import subprocess
        from unittest import mock

        with mock.patch.object(subprocess, "Popen") as popen, \
             mock.patch("builtins.open", mock.mock_open()):
            popen.return_value = object()
            _, handle, command = module.launch_server(
                12345, graphs=True, max_model_len=512,
                log_path=pathlib.Path("/dev/null"),
                model_path=model_path,
            )
    finally:
        pass
    assert command[:4] == [sys.executable, "-m", "fluxserve.cli.launch", "launch"]
    assert command[command.index("--model") + 1] == (model_path or module.MODEL)
    for flag in (
        "--attention-backend", "fa4",
        "--kv-cache-layout", "paged",
        "--scheduler-policy", "paged",
        "--use-decode-cuda-graph",
        "--cuda-graph-decode-mode", "padded",
    ):
        assert flag in command
    assert str(module.BLOCK_LENGTH) in command


# --------------------------------------------------------------------------
# Tensor parallelism lane
# --------------------------------------------------------------------------


def write_tp4(directory, texts=None, *, concurrent=None, after_cancel=None,
              repeat=None, exit_code=0):
    texts = texts or TEXT
    (directory / "online_tp4.json").write_text(json.dumps({
        "graphs": False, "tp_size": 4, "label": "tp4", "command": [],
        "scenarios": {
            "sequential": dict(texts),
            "concurrent": dict(concurrent or texts),
            "after_cancel": dict(after_cancel or texts),
            "single": {NAMES[0]: texts[NAMES[0]]},
            "repeat": {NAMES[0]: (repeat or texts)[NAMES[0]]},
        },
        "server_exit_code": exit_code, "results": {},
        "provenance": {"lane": "tp4"},
    }))


def test_cross_tp_disagreement_is_reported_not_gated(tmp_path):
    """Sharding changes reduction order; a near-tie can flip a whole trajectory.

    Requiring byte equality across TP would make the gate flaky for a reason
    that is not a bug, so the rate is measured and the internal-consistency
    checks do the gating.
    """
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    write_tp4(tmp_path, {NAMES[0]: "sharded differently", NAMES[1]: TEXT[NAMES[1]]})
    record = gate(tmp_path)
    assert record["passed"], failures(record)
    assert record["tp4"]["agreement_rate"] == pytest.approx(0.5)
    assert record["tp4"]["divergent_fixtures"] == [NAMES[0]]
    assert "Tensor parallelism (TP=4)" in harness().render(record)


def test_an_internally_inconsistent_tp4_server_cannot_pass(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")

    write_tp4(tmp_path, concurrent={NAMES[0]: "raced", NAMES[1]: "x"})
    assert "tp4_concurrent_matches_sequential" in failures(gate(tmp_path))

    write_tp4(tmp_path, after_cancel={NAMES[0]: "stale", NAMES[1]: "y"})
    assert "tp4_survives_cancellation" in failures(gate(tmp_path))

    write_tp4(tmp_path, repeat={NAMES[0]: "drifted"})
    assert "tp4_repeat_is_deterministic" in failures(gate(tmp_path))

    write_tp4(tmp_path, exit_code=1)
    assert "tp4_server_exited_cleanly" in failures(gate(tmp_path))


def test_an_empty_tp4_response_cannot_pass(tmp_path):
    """A sharded server that silently returns nothing must not look healthy."""
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    write_tp4(tmp_path, {NAMES[0]: "", NAMES[1]: TEXT[NAMES[1]]})
    assert "tp4_all_requests_completed" in failures(gate(tmp_path))


def test_the_tp4_lane_is_optional(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    record = gate(tmp_path)
    assert record["passed"] and record["tp4_present"] is False
    assert not any(key.startswith("tp4_") for key in record["checks"])


def test_the_server_command_shards_attention_and_experts_together():
    module = harness()
    import subprocess
    from unittest import mock

    with mock.patch.object(subprocess, "Popen"), \
         mock.patch("builtins.open", mock.mock_open()):
        _, _, command = module.launch_server(
            12345, graphs=False, max_model_len=512,
            log_path=pathlib.Path("/dev/null"), tp_size=4,
        )
    assert command[command.index("--tp-size") + 1] == "4"
    assert command[command.index("--ep-size") + 1] == "4"
    assert command[command.index("--dp-size") + 1] == "1"
