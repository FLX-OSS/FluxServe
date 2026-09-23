"""Pinned official checkpoint metadata shared by CPU tests and GPU harnesses."""

import json
from pathlib import Path
from types import SimpleNamespace

CHECKPOINT_ROOT = Path(__file__).parent / "data" / "nemotron_checkpoints"
MODEL_SIZES = ("3B", "8B", "14B")
DEFAULT_MODEL = "nvidia/Nemotron-Labs-Diffusion-14B"


def checkpoint_metadata(model):
    size = model.rsplit("-", 1)[-1]
    if model != f"nvidia/Nemotron-Labs-Diffusion-{size}" or size not in MODEL_SIZES:
        raise ValueError(f"No pinned Nemotron checkpoint metadata for {model!r}")
    directory = CHECKPOINT_ROOT / size
    return {
        name: json.loads((directory / f"{name}.json").read_text())
        for name in ("config", "generation_config", "provenance")
    }


def checkpoint_config(model):
    metadata = checkpoint_metadata(model)
    return SimpleNamespace(
        **metadata["config"], _name_or_path=model,
        _commit_hash=metadata["provenance"]["revision"],
    )


def add_checkpoint_args(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Nemotron checkpoint repository or local snapshot")
    parser.add_argument("--revision", default=None,
                        help="defaults to the recorded revision for each official size")


def resolve_revision(model, revision=None):
    if revision is not None:
        return revision
    return checkpoint_metadata(model)["provenance"]["revision"]


def validate_fixture_model(manifest, model):
    compatible = [manifest["repo_id"], *manifest.get("compatible_repo_ids", [])]
    if model not in compatible:
        raise ValueError(f"fixture manifest targets {compatible}, not {model}")


def check_artifact_models(*artifacts, expected_checkpoint=None):
    identities = []
    for artifact in artifacts:
        if artifact is None:
            continue
        source = artifact.get("provenance", {})
        identities.append((source.get("model"), source.get("revision")))
    if expected_checkpoint is not None:
        identities.append(expected_checkpoint)
    # Old 14B artifacts omit these fields. They remain readable together, but
    # must never silently stand in for a newly selected 3B or 8B checkpoint.
    if len(set(identities)) > 1:
        raise ValueError(f"Cannot compare artifacts from different checkpoints: {identities}")
