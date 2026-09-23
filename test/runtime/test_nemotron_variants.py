"""All model sizes against official metadata, without downloading weights."""

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import sys

import pytest
import torch

from nemotron_test_utils import (
    CHECKPOINT_ROOT, MODEL_SIZES, add_checkpoint_args, checkpoint_config,
    checkpoint_metadata, check_artifact_models, resolve_revision, validate_fixture_model,
)
from fluxserve.backend.model_loader import nemotron as loader
from fluxserve.backend.models import nemotron_diffusion as modeling

MODELS = [f"nvidia/Nemotron-Labs-Diffusion-{size}" for size in MODEL_SIZES]
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("model", MODELS)
def test_every_official_tensor_maps_to_the_actual_model_geometry(model, monkeypatch):
    config = checkpoint_config(model)
    header = json.loads((CHECKPOINT_ROOT / model.rsplit("-", 1)[-1]
                         / "weights_header.json").read_text())
    plan = modeling.nemotron_weight_plan(config)
    assert set(plan) == set(header)
    for name, (_, _, shape) in plan.items():
        assert tuple(header[name]["shape"]) == shape, name
        assert header[name]["dtype"] == "BF16", name

    # Instantiate the full geometry without tensor storage. RoPE is tested
    # separately; avoid putting meta buffers into its shared module cache.
    monkeypatch.setattr(modeling, "get_rope", lambda *a, **kw: torch.nn.Identity())
    with torch.device("meta"):
        model_instance = modeling.NemotronLabsDiffusionLLM(config)
    params = dict(model_instance.named_parameters())
    expected_destinations = {}
    for _, (target, shard, shape) in plan.items():
        if shard is None:
            expected_destinations[target] = shape
        else:
            prior = expected_destinations.get(target, (0, shape[1]))
            expected_destinations[target] = (prior[0] + shape[0], shape[1])
    assert set(params) == set(expected_destinations)
    for name, shape in expected_destinations.items():
        assert tuple(params[name].shape) == shape, name


@pytest.mark.parametrize("model", MODELS)
def test_decoding_ids_and_rope_follow_each_checkpoint(model, monkeypatch):
    config = checkpoint_config(model)
    directory = CHECKPOINT_ROOT / model.rsplit("-", 1)[-1]
    monkeypatch.setattr(loader, "resolve_nemotron_snapshot", lambda *a, **kw: directory)
    assert modeling.is_nemotron_diffusion_config(config)
    assert modeling.nemotron_head_dim(config) == 128
    assert loader.nemotron_decoding_ids(config) == {
        "mask_id": 100, "eos_id": 11, "eos_ids": (11,),
    }
    theta, scaling = modeling.nemotron_rope_parameters(config)
    assert theta == (1e9 if model.endswith("14B") else 1e6)
    assert scaling["factor"] == 16
    assert loader.nemotron_block_length(config) == 32
    loader.check_nemotron_context_limit(config.max_position_embeddings, config)
    with pytest.raises(ValueError, match="total positions"):
        loader.check_nemotron_context_limit(config.max_position_embeddings + 1, config)


@pytest.mark.parametrize("model", MODELS)
def test_official_draft_adapter_matches_every_attention_output_projection(model):
    config = checkpoint_config(model)
    directory = CHECKPOINT_ROOT / model.rsplit("-", 1)[-1]
    adapter = json.loads((directory / "adapter_config.json").read_text())
    header = json.loads((directory / "adapter_header.json").read_text())
    assert adapter["target_modules"] == ["o_proj"]
    assert adapter["base_model_name_or_path"] == model
    expected = {}
    for layer in range(config.num_hidden_layers):
        prefix = f"base_model.model.encoder.layers.{layer}.self_attn.o_proj."
        expected[prefix + "lora_A.weight"] = [adapter["r"], config.num_attention_heads * config.head_dim]
        expected[prefix + "lora_B.weight"] = [config.hidden_size, adapter["r"]]
    assert set(header) == set(expected)
    for name, shape in expected.items():
        assert header[name]["shape"] == shape, name
        assert header[name]["dtype"] == "F32", name


@pytest.mark.parametrize("model", MODELS)
def test_loader_resolves_weights_at_the_configs_revision(model, monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    (tmp_path / "model.safetensors").touch()

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert loader.resolve_nemotron_snapshot(checkpoint_config(model)) == tmp_path
    assert calls[0]["repo_id"] == model
    assert calls[0]["revision"] == resolve_revision(model)
    assert calls[0]["allow_patterns"] == ["model.safetensors", "*.json", "*.jinja"]


def test_shared_token_fixtures_are_backed_by_identical_official_tokenizers():
    manifest = json.loads((ROOT / "test/runtime/data/nemotron_ar_fixtures.json").read_text())
    for model in MODELS:
        validate_fixture_model(manifest, model)
        assert checkpoint_metadata(model)["provenance"]["tokenizer_sha256"] == manifest["tokenizer_file_sha256"]
    with pytest.raises(ValueError, match="fixture manifest"):
        validate_fixture_model(manifest, "nvidia/unrelated-model")


@pytest.mark.parametrize("model", MODELS)
def test_harness_cli_selects_the_revision_for_its_size(model):
    parser = argparse.ArgumentParser()
    add_checkpoint_args(parser)
    args = parser.parse_args(["--model", model])
    assert resolve_revision(args.model, args.revision) == checkpoint_metadata(model)["provenance"]["revision"]
    assert resolve_revision(model, "custom-revision") == "custom-revision"


def test_comparison_refuses_cross_model_or_cross_revision_artifacts():
    def artifact(model, revision):
        return {"provenance": {"model": model, "revision": revision}}

    left = artifact(MODELS[0], "a")
    check_artifact_models(left, left, None)
    with pytest.raises(ValueError, match="different checkpoints"):
        check_artifact_models(left, left, expected_checkpoint=(MODELS[1], "a"))
    for right in [artifact(MODELS[1], "a"), artifact(MODELS[0], "b"), {}]:
        with pytest.raises(ValueError, match="different checkpoints"):
            check_artifact_models(left, right)


@pytest.mark.parametrize("kind", ["diffusion", "online"])
def test_harness_comparison_rejects_mixed_sizes_before_comparing_tokens(kind, tmp_path):
    from test_nemotron_validation_gates import diffusion, online

    artifacts = [
        {"provenance": {"model": model, "revision": resolve_revision(model)}}
        for model in MODELS[:2]
    ]
    if kind == "diffusion":
        torch.save(artifacts[0], tmp_path / "reference.pt")
        torch.save(artifacts[1], tmp_path / "dense.pt")
        with pytest.raises(ValueError, match="different checkpoints"):
            diffusion().compare(tmp_path, 0.05)
    else:
        (tmp_path / "online_offline.json").write_text(json.dumps(artifacts[0]))
        (tmp_path / "online_eager.json").write_text(json.dumps(artifacts[1]))
        with pytest.raises(ValueError, match="different checkpoints"):
            online().compare(tmp_path)


@pytest.mark.parametrize("size", MODEL_SIZES)
def test_documented_launch_commands_are_valid_for_their_checkpoint(size):
    from fluxserve.cli.launch import build_parser

    doc = ROOT / "docs/serving/nemotron/nemotron-labs-diffusion.md"
    section = doc.read_text().split(f"### Nemotron-Labs-Diffusion-{size} ", 1)[1]
    command = section.split("```bash\n", 1)[1].split("```", 1)[0]
    tokens = shlex.split(command.replace("\\\n", " "))
    args = build_parser().parse_args(tokens[1:])
    assert args.model_name == f"nvidia/Nemotron-Labs-Diffusion-{size}"
    assert loader.normalize_nemotron_args(args, checkpoint_config(args.model_name))
    assert args.page_size == args.block_length == 32


@pytest.mark.parametrize("name", ["ar", "diffusion", "online", "long_context"])
def test_every_harness_main_accepts_model_selection(name, monkeypatch, tmp_path):
    path = ROOT / f"test/runtime/integration/nemotron_{name}_harness.py"
    spec = importlib.util.spec_from_file_location(f"variant_{name}_harness", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    class Selected(Exception):
        pass

    def capture(model, revision):
        assert model == MODELS[0]
        assert revision is None
        raise Selected

    monkeypatch.setattr(module, "resolve_revision", capture)
    argv = [str(path), "--model", MODELS[0], "--output", str(tmp_path)]
    if name in ("diffusion", "online"):
        argv += ["--mode", "compare"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(Selected):
        module.main()
