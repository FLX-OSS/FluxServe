# Copyright (c) 2026 FLUX-OSS
# SPDX-License-Identifier: MIT

"""The gates of the GPU validation lanes, exercised without a GPU.

Each lane needs an H100 to produce its artifacts, but the part that decides pass
or fail is ordinary logic: it has to attribute a divergence to the layer that
caused it, and it must not pass a run that never exercised what the lane exists
for -- an adapter that changed no weight, a thinking budget whose marker never
appeared, a long-context probe whose query scale stayed at one.
"""

import functools
import importlib.util
import json
import pathlib
import sys

import pytest
import torch

INTEGRATION = pathlib.Path(__file__).resolve().parent / "integration"


def load_harness(name: str):
    path = INTEGRATION / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def diffusion():
    return load_harness("nemotron_diffusion_harness")


@functools.lru_cache(maxsize=1)
def online():
    return load_harness("nemotron_online_harness")


@functools.lru_cache(maxsize=1)
def long_context():
    return load_harness("nemotron_long_context_harness")


# ---------------------------------------------------------------------------
# Diffusion harness: thinking budget, draft adapter, paged self-speculation
# ---------------------------------------------------------------------------


def thinking_pair(marker_index=4, budget=8, block_length=32):
    generated = [7, 7, 7, 7, 7, 7, 7, 7]
    generated[marker_index] = 99
    expected = {"generated": generated, "nfe": 5, "budget": budget, "marker": 99}
    produced = {
        "generated": list(generated),
        "stats": {"denoise_calls": 3, "commit_calls": 2},
    }
    return expected, produced, block_length


def test_thinking_pass_passes_when_the_marker_lands_inside_the_bound():
    expected, produced, block = thinking_pair()
    item = diffusion().compare_extra_pass("thinking", expected, produced, {}, block)
    assert item["tokens_match"] and item["nfe_match"]
    assert item["marker_within_budget"]
    assert item["marker_index_candidate"] == 4


def test_thinking_pass_fails_when_the_marker_never_appears():
    expected, produced, block = thinking_pair()
    produced["generated"] = [7] * 8
    item = diffusion().compare_extra_pass("thinking", expected, produced, {}, block)
    assert item["marker_index_candidate"] is None
    assert not item["marker_within_budget"]
    assert not item["tokens_match"]


def test_thinking_pass_fails_when_the_marker_is_past_the_bound():
    """The budget bounds the marker to the block that crosses it, not beyond."""
    expected = {"generated": [7] * 50 + [99], "nfe": 5, "budget": 8, "marker": 99}
    produced = {
        "generated": [7] * 50 + [99],
        "stats": {"denoise_calls": 3, "commit_calls": 2},
    }
    item = diffusion().compare_extra_pass("thinking", expected, produced, {}, 32)
    assert item["tokens_match"]
    assert not item["marker_within_budget"]


def test_thinking_pass_counts_denoise_and_commit_forwards():
    expected, produced, block = thinking_pair()
    produced["stats"]["commit_calls"] = 7
    item = diffusion().compare_extra_pass("thinking", expected, produced, {}, block)
    assert item["candidate_calls"] == 10 and not item["nfe_match"]


def test_lora_pass_reports_acceptance_against_the_adapter_free_run():
    expected = {"generated": [1, 2, 3], "nfe": 9, "adapted_modules": 40}
    produced = {
        "generated": [1, 2, 3],
        "stats": {"total_calls": 9, "mean_acceptance": 3.5,
                  "tokens_per_forward": 0.9},
        "adapter": {"layers": 40, "layer0_delta_nrms": 1e-2},
    }
    fixture = {"selfspec": {"stats": {"mean_acceptance": 2.0}}}
    item = diffusion().compare_extra_pass(
        "selfspec_lora", expected, produced, fixture, 32
    )
    assert item["tokens_match"] and item["nfe_match"]
    assert item["acceptance_without_adapter"] == 2.0
    assert item["adapter_improves_acceptance"]
    # Self-speculation counts prefill + draft + verify, not denoise + commit.
    assert item["candidate_calls"] == 9


def test_divergence_is_localized_rather_than_just_reported():
    expected = {"generated": [1, 2, 3, 4], "nfe": 4}
    produced = {"generated": [1, 2, 9, 4], "stats": {"total_calls": 4,
                                                     "mean_acceptance": 1.0,
                                                     "tokens_per_forward": 1.0}}
    item = diffusion().compare_extra_pass(
        "selfspec_lora", expected, produced, {}, 32
    )
    assert not item["tokens_match"]
    assert item["first_divergence_index"] == 2


def paged_selfspec_artifact(tmp_path, mode, *, mutation="none"):
    stats = {"total_calls": 11, "accepted_per_iteration": [3, 2, 4],
             "mean_acceptance": 3.0}
    dense = {"results": {"case": {
        "selfspec": {"generated": [1, 2, 3], "stats": dict(stats)},
    }}}
    candidate = {"results": {"case": {
        "selfspec": {"generated": [1, 2, 3], "stats": dict(stats)},
    }}, "provenance": {"lane": mode}}
    if mutation == "tokens":
        candidate["results"]["case"]["selfspec"]["generated"] = [1, 2, 4]
    elif mutation == "acceptance":
        candidate["results"]["case"]["selfspec"]["stats"][
            "accepted_per_iteration"
        ] = [2, 3, 4]
    elif mutation == "calls":
        candidate["results"]["case"]["selfspec"]["stats"]["total_calls"] = 12
    elif mutation == "missing":
        candidate["results"].clear()
    torch.save(candidate, tmp_path / f"{mode}.pt")
    return dense


def minimal_diffusion_pair(**dense_extra):
    forwards = [
        {"tokens": list(range(34)), "causal": True, "use_cache": True, "length": 34},
        {"tokens": [5] * 32, "causal": False, "use_cache": False, "length": 32},
        {"tokens": [5] * 32, "causal": True, "use_cache": True, "length": 32},
    ]
    reference = {"results": {"case": {
        "generated": [1, 2, 3], "nfe": 2, "forwards": forwards,
        "seeds": [7, 8], "committed_kv": [],
    }}, "provenance": {"lane": "reference"}}
    dense = {"results": {"case": {
        "generated": [1, 2, 3],
        "stats": {"denoise_calls": 1, "commit_calls": 1, "blocks": 1,
                  "prefill_calls": 1},
        "forwards": forwards, "seeds": [7, 8], "committed_kv": [],
        **dense_extra,
    }}, "provenance": {"lane": "dense"}}
    return reference, dense


def test_a_failed_optional_pass_fails_the_gate_instead_of_vanishing(tmp_path):
    """An adapter pass that raised must not be indistinguishable from one that
    was never requested, or --lora could silently stop testing anything."""
    reference, dense = minimal_diffusion_pair(
        selfspec_lora_error="RuntimeError: peft could not attach"
    )
    torch.save(reference, tmp_path / "reference.pt")
    torch.save(dense, tmp_path / "dense.pt")
    record = diffusion().compare(tmp_path, 0.05)
    assert record["optional_pass_errors"] == {
        "dense:selfspec_lora": "RuntimeError: peft could not attach"
    }
    assert record["checks"]["optional_passes_ran"] is False
    assert not record["passed"]
    assert "peft could not attach" in diffusion().render(record)


def test_a_clean_run_without_optional_passes_still_passes(tmp_path):
    reference, dense = minimal_diffusion_pair()
    torch.save(reference, tmp_path / "reference.pt")
    torch.save(dense, tmp_path / "dense.pt")
    record = diffusion().compare(tmp_path, 0.05)
    assert "optional_passes_ran" not in record["checks"]
    assert record["passed"], failures(record)


@pytest.mark.parametrize(
    "mutation", ["none", "tokens", "acceptance", "calls", "missing"]
)
def test_paged_selfspec_gate_detects_divergence(tmp_path, mutation):
    mode = "selfspec_fa4"
    dense = paged_selfspec_artifact(tmp_path, mode, mutation=mutation)
    record = diffusion().compare_paged_selfspec(tmp_path, dense, mode)
    assert all(record["checks"].values()) == (mutation == "none")


# ---------------------------------------------------------------------------
# Online harness: optional lanes
# ---------------------------------------------------------------------------


NAMES = ["chat0_gen32", "chat1_gen64"]
TEXT = {name: f"answer for {name}" for name in NAMES}


def write_offline(directory, suffix="", texts=None, thinking=None):
    texts = texts or TEXT
    results = {}
    for name in texts:
        entry = {"text": texts[name], "token_ids": [1, 2, 3], "stats": {}}
        if thinking is not None:
            entry["thinking"] = thinking
        results[name] = entry
    name = "online_offline" + (f"_{suffix}" if suffix else "")
    (directory / f"{name}.json").write_text(
        json.dumps({"results": results, "provenance": {"lane": name}})
    )


def write_lane(directory, label, texts=None, *, exit_code=0, flood=True,
               flood_consistent=True, backend="fa4", decoding="threshold",
               thinking=None, concurrent=None):
    texts = texts or TEXT
    scenarios = {
        "sequential": dict(texts),
        "concurrent": dict(concurrent or texts),
        "after_cancel": dict(texts),
        "single": {list(texts)[0]: texts[list(texts)[0]]},
        "repeat": {list(texts)[0]: texts[list(texts)[0]]},
    }
    if flood:
        scenarios["flood"] = dict(texts)
    payload = {
        "graphs": label == "graphs",
        "backend": backend,
        "decoding": decoding,
        "thinking": thinking,
        "command": [],
        "scenarios": scenarios,
        "metrics_after_startup": {"cuda_graph_denoise_capture_count": 3,
                                  "cuda_graph_commit_capture_count": 3},
        "metrics_after_single": {"cuda_graph_decode_padded_rows": 8},
        "metrics_final": {"cuda_graph_decode_replay_count": 12},
        "server_exit_code": exit_code,
        "flood_replies_consistent": flood_consistent,
        "results": {},
        "provenance": {"lane": label},
    }
    (directory / f"online_{label}.json").write_text(json.dumps(payload))


def failures(record):
    return sorted(key for key, value in record["checks"].items() if not value)


def test_flashinfer_lane_is_held_to_the_dense_offline_text(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    write_lane(tmp_path, "flashinfer", backend="flashinfer")
    record = online().compare(tmp_path)
    assert record["checks"]["flashinfer_matches_offline"]
    assert record["extra_lanes"]["flashinfer"]["backend"] == "flashinfer"
    assert failures(record) == []


def test_flashinfer_lane_divergence_is_attributed_to_that_lane(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    write_lane(tmp_path, "flashinfer", backend="flashinfer",
               texts={name: "something else" for name in NAMES})
    record = online().compare(tmp_path)
    assert failures(record) == ["flashinfer_matches_offline"]


def test_a_lane_without_its_offline_baseline_is_an_error(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    write_lane(tmp_path, "selfspec", decoding="self_speculation")
    with pytest.raises(AssertionError):
        online().compare(tmp_path)


def test_selfspec_lane_uses_its_own_baseline(tmp_path):
    """Self-speculation emits the verifier's tokens, so the threshold lane's
    text cannot predict it; only its own offline lane can."""
    spec_text = {name: f"spec {name}" for name in NAMES}
    write_offline(tmp_path)
    write_offline(tmp_path, suffix="selfspec", texts=spec_text)
    write_lane(tmp_path, "eager")
    write_lane(tmp_path, "selfspec", texts=spec_text,
               decoding="self_speculation")
    record = online().compare(tmp_path)
    assert record["checks"]["selfspec_matches_offline"]
    assert failures(record) == []


def test_thinking_lane_gates_the_marker_position(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager")
    for index, expected_pass in ((4, True), (None, False), (99, False)):
        write_offline(
            tmp_path, suffix="thinking",
            thinking={"budget": 8, "marker": 99, "marker_index": index,
                      "bound": 40},
        )
        write_lane(tmp_path, "thinking", thinking=8)
        record = online().compare(tmp_path)
        assert record["checks"]["thinking_marker_within_budget"] == expected_pass


def test_flood_requires_every_copy_of_a_prompt_to_agree(tmp_path):
    write_offline(tmp_path)
    write_lane(tmp_path, "eager", flood_consistent=False)
    record = online().compare(tmp_path)
    assert failures(record) == ["eager_flood_replies_agree"]


def test_flood_scenario_is_optional(tmp_path):
    """Long-prompt lanes skip oversubscription, and that must not fail the gate
    for the lanes that did run it."""
    write_offline(tmp_path)
    write_lane(tmp_path, "eager", flood=False)
    record = online().compare(tmp_path)
    assert "eager_flood_matches_sequential" not in record["checks"]
    assert failures(record) == []


# ---------------------------------------------------------------------------
# Long-context harness
# ---------------------------------------------------------------------------


def long_context_record(**overrides):
    offsets = {
        "0": {"nrms": 1e-3, "max_abs": 0.05, "top1_agreement": 1.0,
              "rotary_max_abs": 1e-6,
              "query_scale": {"min": 1.0, "max": 1.0}},
        "32768": {"nrms": 2e-3, "max_abs": 0.08, "top1_agreement": 1.0,
                  "rotary_max_abs": 1e-6,
                  "query_scale": {"min": 1.1099, "max": 1.1099}},
    }
    offsets["32768"].update(overrides)
    return {"probe_length": 129, "offsets": offsets,
            "provenance": {"lane": "long_context"}}


def test_long_context_tolerance_comes_from_the_offset_zero_row():
    record = long_context_record()
    long_context().gate(record)
    assert record["tolerance"] == pytest.approx(5e-3)
    assert record["passed"]
    # A large floor raises the bar rather than lowering it.
    loose = long_context_record()
    loose["offsets"]["0"]["nrms"] = 0.01
    long_context().gate(loose)
    assert loose["tolerance"] == pytest.approx(0.04)


def test_long_context_fails_a_vacuous_run():
    """If the query scale stayed at one the lane tested the same identity every
    other lane already covers, so it cannot be allowed to pass."""
    record = long_context_record(query_scale={"min": 1.0, "max": 1.0})
    long_context().gate(record)
    assert not record["passed"]
    assert "offset_32768_scale_is_not_identity" in failures(record)


@pytest.mark.parametrize(
    "overrides, failing",
    [
        ({"top1_agreement": 0.98}, "offset_32768_top1_agrees"),
        ({"nrms": 0.5}, "offset_32768_within_tolerance"),
        ({"rotary_max_abs": 1.0}, "rotary_matches_reference"),
    ],
)
def test_long_context_gate_detects_divergence(overrides, failing):
    record = long_context_record(**overrides)
    long_context().gate(record)
    assert not record["passed"] and failing in failures(record)


def test_long_context_requires_a_high_offset():
    record = long_context_record()
    del record["offsets"]["32768"]
    long_context().gate(record)
    assert not record["passed"]
    assert "high_offsets_were_probed" in failures(record)


def test_long_context_summary_names_the_scale_it_exercised():
    record = long_context_record()
    long_context().gate(record)
    text = long_context().render(record)
    assert "1.1099" in text and "PASS" in text
