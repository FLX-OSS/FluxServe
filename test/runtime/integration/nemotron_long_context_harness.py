"""Long-context validation for Nemotron-Labs-Diffusion: positions past 16384.

``_get_llama_4_attn_scale`` is an exact identity below
``original_max_position_embeddings``, which is 16384 for this checkpoint, so
every other lane in this suite runs with the query temperature equal to one and
says nothing about it. The same is true of the YaRN cos/sin cache, whose
interesting entries are the high ones.

Rather than pay for a 16K prompt, this harness puts a short window at high
absolute positions. That works because both implementations take the scale from
a position tensor the caller supplies -- the reference from ``cache_position``,
FluxServe from ``position_ids`` -- and because ``create_causal_mask`` returns an
already-prepared 4D mask as-is, so handing the reference its own causal mask
stops a large offset from silently turning the mask non-causal against an empty
cache.

Gated: top-1 agreement at every probed position, a logit gap no worse than four
times the offset-zero gap between the same two implementations, and cos/sin
agreement. The query scale actually in force at each offset is recorded, so a
run where it stayed 1.0 -- which would make the whole lane vacuous -- cannot
look like a pass.

    python nemotron_long_context_harness.py --output DIR [--offsets 0,16320,...]
"""

from __future__ import annotations

import argparse
import functools
import gc
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES = REPO_ROOT / "test" / "runtime" / "data" / "nemotron_ar_fixtures.json"
MODEL = "nvidia/Nemotron-Labs-Diffusion-14B"
REVISION = "f8c3e2c078e193599b8882d965b1001c456ba738"
PROBE_LENGTH = 129
# 16320 straddles the 16384 boundary inside one window, so the same forward
# carries scaled and unscaled positions; the rest sit in later YaRN bands.
DEFAULT_OFFSETS = (0, 16320, 32768, 131072)


@functools.lru_cache(maxsize=1)
def ar_harness():
    """The Phase 1 harness, for its reference and candidate wrappers."""
    path = Path(__file__).with_name("nemotron_ar_harness.py")
    spec = importlib.util.spec_from_file_location("nemotron_ar_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }


def probe_tokens(fixtures_path: str, length: int) -> list[int]:
    """Real token ids, cycled to the probe length; random ids hide nothing here
    but make the logits meaningless."""
    with open(fixtures_path) as handle:
        manifest = json.load(handle)
    source = manifest["diffusion"][0]["input_ids"]
    return (source * ((length + len(source) - 1) // len(source)))[:length]


def causal_masks(length: int, device: str):
    """The reference wants a 4D mask, FluxServe's dense path a 3D one."""
    lower = torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=device)
    )
    return lower.unsqueeze(0).unsqueeze(0), lower.unsqueeze(0)


def query_scale(config, positions: torch.Tensor):
    from fluxserve.backend.models.nemotron_diffusion import nemotron_query_scale

    parameters = getattr(config, "rope_parameters", None) or {}
    scale = nemotron_query_scale(
        positions,
        parameters.get("llama_4_scaling_beta"),
        parameters.get("original_max_position_embeddings"),
    )
    if scale is None:
        return None
    return {"min": float(scale.min()), "max": float(scale.max())}


def logit_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    difference = candidate - reference
    reference_rms = float(reference.pow(2).mean().sqrt())
    agreement = (
        candidate.argmax(dim=-1) == reference.argmax(dim=-1)
    ).float().mean()
    return {
        "nrms": float(difference.pow(2).mean().sqrt()) / max(reference_rms, 1e-12),
        "max_abs": float(difference.abs().max()),
        "top1_agreement": float(agreement),
    }


def run(output_dir: Path, offsets, device: str, fixtures_path: str) -> dict:
    from transformers import AutoConfig

    module = ar_harness()
    config = AutoConfig.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True
    )
    tokens = probe_tokens(fixtures_path, PROBE_LENGTH)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    mask_4d, mask_3d = causal_masks(PROBE_LENGTH, device)

    position_sets = {}
    for offset in offsets:
        flat = torch.arange(
            offset, offset + PROBE_LENGTH, device=device, dtype=torch.long
        )
        position_sets[offset] = flat

    reference_logits = {}
    reference_rotary = {}
    reference = module.ReferenceAR(MODEL, device, revision=REVISION)
    try:
        with torch.no_grad():
            for offset, flat in position_sets.items():
                output = reference.model.encoder(
                    input_ids=input_ids,
                    position_ids=flat.unsqueeze(0),
                    past_key_values=None,
                    use_cache=False,
                    cache_position=flat,
                    attention_mask=mask_4d,
                    use_causal_mask=True,
                )
                logits = reference.model.diffusion_head(output.last_hidden_state)
                reference_logits[offset] = logits.float().cpu()
                _, cos, sin = reference.rotary(flat)
                reference_rotary[offset] = (cos, sin)
                print(f"[reference] offset {offset} done", flush=True)
    finally:
        reference.release()
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    record = {"probe_length": PROBE_LENGTH, "offsets": {},
              "provenance": provenance()}
    candidate = module.FluxServeAR.load(config, device, max_length=PROBE_LENGTH)
    with torch.no_grad():
        for offset, flat in position_sets.items():
            output = candidate.model(
                input_ids=input_ids,
                position_ids=flat.unsqueeze(0),
                past_key_values=None,
                use_cache=False,
                attention_mask=mask_3d,
            )
            metrics = logit_metrics(
                output.logits.float().cpu(), reference_logits[offset]
            )
            _, cos, sin = candidate.rotary(flat)
            expected_cos, expected_sin = reference_rotary[offset]
            half = cos.shape[-1]
            metrics["rotary_max_abs"] = max(
                float((cos - expected_cos.reshape(-1, expected_cos.shape[-1])
                       [:, :half]).abs().max()),
                float((sin - expected_sin.reshape(-1, expected_sin.shape[-1])
                       [:, :half]).abs().max()),
            )
            metrics["query_scale"] = query_scale(config, flat.unsqueeze(0))
            record["offsets"][str(offset)] = metrics
            print(f"[flux] offset {offset} {metrics}", flush=True)

    gate(record)
    return record


def gate(record: dict) -> dict:
    """Derive the tolerance from the offset-zero row, then score the rest.

    Kept separate from the measuring so it can be tested without a GPU, the way
    the other harnesses' gates are.
    """
    control = record["offsets"].get("0")
    if control is None:
        raise ValueError("offset 0 is the numeric floor and must be probed")
    floor = control["nrms"]
    # Four times the offset-zero gap, with an absolute minimum so a tiny floor
    # cannot turn into an impossible target.
    tolerance = max(4 * floor, 5e-3)
    record["floor_nrms"] = floor
    record["tolerance"] = tolerance
    checks = {
        "floor_is_small": floor <= 0.05,
        "control_top1_agrees": control["top1_agreement"] == 1.0,
        "rotary_matches_reference": all(
            item["rotary_max_abs"] <= 1e-3 for item in record["offsets"].values()
        ),
        "high_offsets_were_probed": any(
            key != "0" for key in record["offsets"]
        ),
    }
    for key, item in record["offsets"].items():
        if key == "0":
            continue
        checks[f"offset_{key}_top1_agrees"] = item["top1_agreement"] == 1.0
        checks[f"offset_{key}_within_tolerance"] = item["nrms"] <= tolerance
        # Without this the lane could pass while testing the same identity the
        # short-context lanes already test.
        checks[f"offset_{key}_scale_is_not_identity"] = bool(
            item["query_scale"] and item["query_scale"]["max"] > 1.0
        )
    record["checks"] = checks
    record["passed"] = all(checks.values())
    return record


def render(record: dict) -> str:
    lines = [
        "# Nemotron-Labs-Diffusion long-context validation",
        "",
        f"**Result: {'PASS' if record['passed'] else 'FAIL'}**",
        "",
        f"Probe window {record['probe_length']} tokens at increasing absolute "
        "positions. The query temperature is an identity below 16384, so the "
        "offset-zero row is the numeric floor and the scale column shows what "
        "each other row actually exercised.",
        "",
        f"- floor nrms: `{record['floor_nrms']:.3e}`, tolerance "
        f"`{record['tolerance']:.3e}`",
        "",
        "| Offset | top-1 agreement | logit nrms | max abs | rotary max abs | "
        "query scale |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, item in record["offsets"].items():
        scale = item["query_scale"]
        scale_cell = "disabled" if not scale else (
            f"{scale['min']:.4f}..{scale['max']:.4f}"
        )
        lines.append(
            f"| {key} | {item['top1_agreement']:.4f} | {item['nrms']:.3e} | "
            f"{item['max_abs']:.3e} | {item['rotary_max_abs']:.3e} | "
            f"{scale_cell} |"
        )
    lines += ["", "## Checks", "", "| Check | Result |", "| --- | --- |"]
    for key, value in record["checks"].items():
        lines.append(f"| {key} | {'pass' if value else '**FAIL**'} |")
    lines += ["", "## Provenance", "", f"- `{record['provenance']}`"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    parser.add_argument(
        "--offsets",
        default=",".join(str(value) for value in DEFAULT_OFFSETS),
        help="comma-separated absolute position offsets; 0 must be included",
    )
    args = parser.parse_args()

    offsets = [int(value) for value in args.offsets.split(",") if value != ""]
    if 0 not in offsets:
        raise SystemExit("offset 0 is the numeric floor and must be included")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = run(output_dir, offsets, args.device, args.fixtures)
    (output_dir / "long_context_metrics.json").write_text(
        json.dumps(record, indent=2)
    )
    (output_dir / "long_context_summary.md").write_text(render(record))
    print(json.dumps(record["checks"], indent=2), flush=True)
    print(f"LONG_CONTEXT {'PASS' if record['passed'] else 'FAIL'}", flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
