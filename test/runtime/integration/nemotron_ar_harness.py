"""Phase 1 AR validation harness for Nemotron-Labs-Diffusion.

Autoregressive decoding is the simplest of the checkpoint's three modes and the
one that isolates every non-decoding question: weight binding, YaRN rotary
embeddings, grouped query attention, the output projection and tensor
parallelism. This harness compares FluxServe's dense causal path against the
checkpoint's own modeling code on a checked-in fixture manifest and writes a
result artifact.

It is a developer harness, not a serving mode, and not a pytest module: run it
from a GPU job. See
``docs/serving/nemotron/nemotron-labs-diffusion-14B.md`` for configuration.

AR execution contract under test:

* Causal prefill's final-position logits predict the first generated token.
* Each decode forward consumes the previous token at its absolute position,
  appends its KV to the cache, and predicts the next token.
* No decode call is made once EOS or the output budget is reached.

Oracle: the checkpoint's ``encoder`` plus ``diffusion_head`` with
``diffusion_lm=False``, which is the path ``ar_generate`` uses.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES = REPO_ROOT / "test" / "runtime" / "data" / "nemotron_ar_fixtures.json"

# Tolerances are derived from the calibration split and then frozen before the
# held-out split is evaluated; these are only the floors below which a derived
# tolerance is not allowed to shrink, so a suspiciously quiet calibration set
# cannot produce an unachievably tight gate.
MIN_TOLERANCE_MAX_ABS = 1e-3
MIN_TOLERANCE_NRMS = 1e-4
# Multiple of the measured reference-vs-reference noise floor.
TOLERANCE_SLACK = 4.0
# A same-weights, different-kernel comparison above this is not noise.
CALIBRATION_FLOOR_CEILING_NRMS = 0.05


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    """Absolute, relative and distributional errors for paired logits."""
    candidate = candidate.detach().to(torch.float32).cpu()
    reference = reference.detach().to(torch.float32).cpu()
    if candidate.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: candidate={tuple(candidate.shape)} "
            f"reference={tuple(reference.shape)}"
        )
    finite = bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all())
    difference = candidate - reference
    rms = float(difference.pow(2).mean().sqrt())
    reference_rms = float(reference.pow(2).mean().sqrt())

    if reference.shape[-1] >= 2:
        top2 = reference.topk(2, dim=-1).values
        margins = (top2[..., 0] - top2[..., 1]).reshape(-1)
    else:
        margins = torch.zeros(reference[..., 0].numel())
    candidate_top1 = candidate.argmax(-1).reshape(-1)
    reference_top1 = reference.argmax(-1).reshape(-1)
    agree = candidate_top1 == reference_top1
    disagreements = (~agree).nonzero(as_tuple=True)[0].tolist()

    return {
        "finite": finite,
        "max_abs": float(difference.abs().max()),
        "rms": rms,
        "nrms": rms / max(reference_rms, 1e-12),
        "reference_rms": reference_rms,
        "top1_agreement": float(agree.float().mean()),
        "positions": int(agree.numel()),
        "disagreement_positions": disagreements[:32],
        "disagreement_margins": [float(margins[i]) for i in disagreements[:32]],
        "reference_margin_min": float(margins.min()),
        "reference_margin_median": float(margins.median()),
    }


def worst(metrics: list[dict], key: str) -> float:
    return max((entry[key] for entry in metrics), default=0.0)


# ---------------------------------------------------------------------------
# Reference oracle
# ---------------------------------------------------------------------------


class ReferenceAR:
    """The checkpoint's own modeling code, driven in autoregressive mode."""

    def __init__(self, repo_id: str, device: str, revision: str | None = None):
        from transformers import AutoModel

        self.device = device
        common = {
            "revision": revision,
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }
        try:
            self.model = AutoModel.from_pretrained(
                repo_id, dtype=torch.bfloat16, **common
            )
        except TypeError:  # older transformers spell it torch_dtype
            self.model = AutoModel.from_pretrained(
                repo_id, torch_dtype=torch.bfloat16, **common
            )
        self.model = self.model.to(device).eval()
        # AR mode: strictly causal attention, exactly as `ar_generate` does.
        self.set_diffusion_lm(False)
        self.encoder_config = self.model.encoder.layers[0].self_attn.config

    def set_diffusion_lm(self, value: bool) -> None:
        for layer in self.model.encoder.layers:
            if hasattr(layer.self_attn, "diffusion_lm"):
                layer.self_attn.diffusion_lm = value

    def set_attn_implementation(self, name: str) -> None:
        # Every attention module shares the encoder's deep-copied config object.
        self.encoder_config._attn_implementation = name

    def attn_implementation(self) -> str:
        return self.encoder_config._attn_implementation

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, explicit_mask: bool = True):
        """Causal prefill. Returns ``(logits, cache)``.

        ``explicit_mask=False`` reproduces ``ar_generate``, which builds no mask
        and relies on SDPA's implicit ``is_causal``. That path is *not* causal
        under the eager implementation, so calibration always uses an explicit
        mask.
        """
        from transformers.cache_utils import DynamicCache

        length = input_ids.shape[1]
        cache_position = torch.arange(length, device=self.device)
        position_ids = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)
        kwargs = {"use_causal_mask": True} if explicit_mask else {}
        output = self.model.encoder(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            cache_position=cache_position,
            **kwargs,
        )
        logits = self.model.diffusion_head(output.last_hidden_state)
        return logits.float(), output.past_key_values

    @torch.no_grad()
    def decode_step(self, token: torch.Tensor, position: int, cache):
        cache_position = torch.tensor([position], device=self.device)
        position_ids = cache_position.unsqueeze(0).expand(token.shape[0], -1)
        output = self.model.encoder(
            input_ids=token,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        logits = self.model.diffusion_head(output.last_hidden_state[:, -1:, :])
        return logits.float().squeeze(1), output.past_key_values

    def rotary(self, positions: torch.Tensor):
        """Full-width cos/sin as the checkpoint builds them.

        A float32 probe tensor is used so the returned cache is compared at
        full precision; the checkpoint casts cos/sin to the activation dtype at
        run time, which is a separate (and in its favour, looser) question.
        """
        rotary = self.model.encoder.rotary_emb
        probe = torch.zeros(1, 1, 1, device=self.device, dtype=torch.float32)
        cos, sin = rotary(probe, positions.unsqueeze(0).to(self.device))
        return rotary.inv_freq.float().cpu(), cos.float().cpu(), sin.float().cpu()

    def layer_modules(self):
        return list(self.model.encoder.layers)

    @staticmethod
    def layer_output(output):
        return output if isinstance(output, torch.Tensor) else output[0]

    def release(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# FluxServe candidate
# ---------------------------------------------------------------------------


class FluxServeAR:
    """FluxServe's dense causal path, driven with the same AR contract."""

    def __init__(self, model, model_config, device: str, max_length: int):
        from fluxserve.backend.models.nemotron_diffusion import nemotron_head_dim

        self.device = device
        self.config = model_config
        self.model = model
        self.num_layers = int(model_config.num_hidden_layers)
        self.num_kv_heads = int(model_config.num_key_value_heads)
        self.head_dim = nemotron_head_dim(model_config)
        self.max_length = max_length
        # Follow the loaded weights rather than assuming bfloat16: the
        # dense attention path requires q, k and v to share a dtype.
        self.dtype = next(model.parameters()).dtype

    @classmethod
    def load(cls, model_config, device: str, max_length: int) -> "FluxServeAR":
        from fluxserve.backend.model_loader import get_model

        model = get_model(
            model_config=model_config, device=device, quant_config=None
        )
        return cls(model, model_config, device, max_length)

    def _new_cache(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            self.num_layers,
            2,
            batch_size,
            self.num_kv_heads,
            self.max_length,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def _store(self, cache: torch.Tensor, present, batch_size: int, length: int):
        stacked = torch.stack(present, dim=0).reshape(
            self.num_layers, 2, batch_size, self.num_kv_heads, length, self.head_dim
        )
        cache[:, :, :, :, :length] = stacked

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor):
        batch_size, length = input_ids.shape
        if length > self.max_length:
            raise ValueError(f"prompt length {length} exceeds cache {self.max_length}")
        mask = (
            torch.tril(torch.ones(length, length, dtype=torch.bool, device=self.device))
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        positions = torch.arange(length, device=self.device).unsqueeze(0).expand(
            batch_size, -1
        )
        output = self.model(
            input_ids=input_ids,
            position_ids=positions,
            past_key_values=None,
            use_cache=True,
            attention_mask=mask,
        )
        cache = self._new_cache(batch_size)
        self._store(cache, output.past_key_values, batch_size, length)
        return output.logits.float(), cache, length

    @torch.no_grad()
    def decode_step(self, token: torch.Tensor, position: int, cache, cache_length: int):
        batch_size = token.shape[0]
        length = cache_length + 1
        # The dense path writes the new token into the final slot of the window
        # it is given, so the window must already include it.
        window = cache[:, :, :, :, :length]
        positions = torch.full(
            (batch_size, 1), position, device=self.device, dtype=torch.long
        )
        output = self.model(
            input_ids=token,
            position_ids=positions,
            past_key_values=window,
            use_cache=True,
            attention_mask=None,
        )
        self._store(cache, output.past_key_values, batch_size, length)
        return output.logits.float()[:, -1], length

    def rotary(self, positions: torch.Tensor):
        """Half-width cos/sin, the layout FluxServe's rotary cache stores."""
        rotary = self.model.model.layers[0].self_attn.rotary_emb
        cached = rotary.cos_sin_cache.shape[0]
        if int(positions.max()) >= cached:
            raise ValueError(
                f"rotary position {int(positions.max())} is outside the "
                f"{cached}-entry cos/sin cache"
            )
        cos_sin = rotary.cos_sin_cache.index_select(
            0, positions.to(self.device)
        ).float()
        half = cos_sin.shape[-1] // 2
        return rotary.inv_freq.float().cpu() if hasattr(rotary, "inv_freq") else None, \
            cos_sin[..., :half].cpu(), cos_sin[..., half:].cpu()

    def layer_modules(self):
        return list(self.model.model.layers)

    @staticmethod
    def layer_output(output):
        return output if isinstance(output, torch.Tensor) else output[0]

    def release(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Shared AR procedures
# ---------------------------------------------------------------------------


@dataclass
class FixtureResult:
    name: str
    length: int
    prefill_logits: torch.Tensor | None = None
    decode_logits: torch.Tensor | None = None
    decode_positions: list[int] = field(default_factory=list)


def teacher_forced_split(length: int, steps: int) -> int:
    """Prefill boundary so that ``steps`` known tokens are fed one at a time."""
    return max(1, length - steps)


def run_reference_fixture(oracle: ReferenceAR, input_ids: list[int], steps: int,
                          explicit_mask: bool = True) -> FixtureResult:
    device = oracle.device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    length = ids.shape[1]
    prefill_logits, _ = oracle.prefill(ids, explicit_mask=explicit_mask)

    decode_logits = None
    positions: list[int] = []
    if length >= 2 and steps > 0:
        split = teacher_forced_split(length, steps)
        _, cache = oracle.prefill(ids[:, :split], explicit_mask=explicit_mask)
        collected = []
        for offset in range(split, length):
            token = ids[:, offset : offset + 1]
            logits, cache = oracle.decode_step(token, offset, cache)
            collected.append(logits.cpu())
            positions.append(offset)
        decode_logits = torch.stack(collected, dim=1)
    return FixtureResult(
        name="",
        length=length,
        prefill_logits=prefill_logits.cpu(),
        decode_logits=decode_logits,
        decode_positions=positions,
    )


def run_candidate_fixture(candidate: FluxServeAR, input_ids: list[int],
                          steps: int) -> FixtureResult:
    device = candidate.device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    length = ids.shape[1]
    prefill_logits, _, _ = candidate.prefill(ids)

    decode_logits = None
    positions: list[int] = []
    if length >= 2 and steps > 0:
        split = teacher_forced_split(length, steps)
        _, cache, cache_length = candidate.prefill(ids[:, :split])
        collected = []
        for offset in range(split, length):
            token = ids[:, offset : offset + 1]
            logits, cache_length = candidate.decode_step(
                token, offset, cache, cache_length
            )
            collected.append(logits.cpu())
            positions.append(offset)
        decode_logits = torch.stack(collected, dim=1)
    return FixtureResult(
        name="",
        length=length,
        prefill_logits=prefill_logits.cpu(),
        decode_logits=decode_logits,
        decode_positions=positions,
    )


def greedy_reference(oracle: ReferenceAR, input_ids: list[int], max_new_tokens: int,
                     eos_ids: tuple[int, ...]) -> dict:
    device = oracle.device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits, cache = oracle.prefill(ids)
    next_logit = logits[:, -1]
    tokens: list[int] = []
    margins: list[float] = []
    position = ids.shape[1]
    for _ in range(max_new_tokens):
        top2 = next_logit.topk(2, dim=-1).values
        margins.append(float(top2[0, 0] - top2[0, 1]))
        token_id = int(next_logit.argmax(-1))
        tokens.append(token_id)
        if token_id in eos_ids:
            break
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        next_logit, cache = oracle.decode_step(token, position, cache)
        position += 1
    return {"tokens": tokens, "margins": margins}


def greedy_candidate(candidate: FluxServeAR, input_ids: list[int],
                     max_new_tokens: int, eos_ids: tuple[int, ...]) -> dict:
    device = candidate.device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits, cache, cache_length = candidate.prefill(ids)
    next_logit = logits[:, -1]
    tokens: list[int] = []
    position = ids.shape[1]
    for _ in range(max_new_tokens):
        token_id = int(next_logit.argmax(-1))
        tokens.append(token_id)
        if token_id in eos_ids:
            break
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        next_logit, cache_length = candidate.decode_step(
            token, position, cache, cache_length
        )
        position += 1
    return {"tokens": tokens}


def causality_probe(oracle: ReferenceAR, input_ids: list[int]) -> dict:
    """Confirm the reference is genuinely causal under the current settings.

    Editing the final prompt token must leave every earlier position's logits
    untouched. This matters most for the eager calibration run: eager attention
    applies no causal masking when `attention_mask` is None, so if the
    `use_causal_mask` keyword ever stopped reaching the encoder, the calibration
    "noise floor" would silently become a causal-versus-bidirectional gap and
    the derived tolerances would be meaningless.
    """
    device = oracle.device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    baseline, _ = oracle.prefill(ids, explicit_mask=True)
    edited = ids.clone()
    edited[0, -1] = (int(edited[0, -1]) + 1) % int(oracle.model.config.vocab_size)
    changed, _ = oracle.prefill(edited, explicit_mask=True)
    prefix_delta = float((baseline[:, :-1] - changed[:, :-1]).abs().max())
    final_delta = float((baseline[:, -1] - changed[:, -1]).abs().max())
    return {
        "attn_implementation": oracle.attn_implementation(),
        "prefix_max_abs": prefix_delta,
        "final_position_max_abs": final_delta,
        "is_causal": prefix_delta < 1e-2 < final_delta,
    }


def select_probe_fixture(fixtures: list[dict], max_length: int = 128) -> dict:
    """Pick one fixture for the causality and self-consistency probes.

    Long enough that causality is meaningful, short enough to run several
    extra forwards over it. Falls back to the longest fixture available.
    """
    if not fixtures:
        raise ValueError("no fixtures to probe")
    affordable = [f for f in fixtures if f["length"] <= max_length and f["length"] >= 2]
    return max(affordable or fixtures, key=lambda fixture: fixture["length"])


def capture_layer_states(runner, input_ids: list[int]) -> list[torch.Tensor]:
    """Per-layer hidden states for one causal prefill, for divergence triage."""
    captured: list[torch.Tensor] = []
    handles = []

    def hook(_module, _inputs, output):
        captured.append(runner.layer_output(output).detach().float().cpu())

    for layer in runner.layer_modules():
        handles.append(layer.register_forward_hook(hook))
    try:
        ids = torch.tensor([input_ids], dtype=torch.long, device=runner.device)
        if isinstance(runner, ReferenceAR):
            runner.prefill(ids)
        else:
            runner.prefill(ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def environment_record(args) -> dict:
    def git(*command):
        try:
            return subprocess.check_output(
                ["git", *command], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:  # noqa: BLE001 - provenance is best effort
            return "unknown"

    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:  # noqa: BLE001
        transformers_version = "unknown"

    return {
        "fluxserve_commit": git("rev-parse", "HEAD"),
        "fluxserve_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "fluxserve_dirty": bool(git("status", "--porcelain")),
        "repo_id": args.model,
        "revision": args.revision,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "dtype": "bfloat16",
        "attention_backend": "dense (SDPA)",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/Nemotron-Labs-Diffusion-14B")
    parser.add_argument(
        "--revision", default="f8c3e2c078e193599b8882d965b1001c456ba738"
    )
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--layer-trace-fixture", default="len32")
    parser.add_argument(
        "--rope-positions",
        default="0,31,32,16383,16384,32768,65536,262143",
        help="absolute positions to compare cos/sin at, including YaRN boundaries",
    )
    parser.add_argument("--skip-greedy", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.fixtures) as handle:
        manifest = json.load(handle)
    if manifest["repo_id"] != args.model:
        raise ValueError(
            f"fixture manifest targets {manifest['repo_id']}, not {args.model}"
        )

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )

    from fluxserve.backend.model_loader.nemotron import (
        check_nemotron_context_limit,
        nemotron_decoding_ids,
    )

    ids_config = nemotron_decoding_ids(model_config)
    eos_ids = tuple(ids_config["eos_ids"])

    all_fixtures = manifest["calibration"] + manifest["held_out"]
    greedy_fixtures = [] if args.skip_greedy else manifest["greedy"]
    max_length = max(
        [fixture["length"] for fixture in all_fixtures]
        + [
            fixture["length"] + fixture["max_new_tokens"] + 1
            for fixture in greedy_fixtures
        ]
    )
    check_nemotron_context_limit(max_length)

    record: dict = {
        "environment": environment_record(args),
        "fixtures": {
            "path": str(args.fixtures),
            "revision": manifest["revision"],
            "calibration": [f["name"] for f in manifest["calibration"]],
            "held_out": [f["name"] for f in manifest["held_out"]],
            "greedy": [f["name"] for f in greedy_fixtures],
        },
        "decode_steps": args.decode_steps,
        "max_length": max_length,
        "eos_ids": list(eos_ids),
    }

    # ---------------- Stage A: reference oracle ----------------
    print("[stage A] loading reference model", flush=True)
    started = time.perf_counter()
    oracle = ReferenceAR(args.model, args.device, revision=args.revision)
    record["reference_load_seconds"] = round(time.perf_counter() - started, 1)
    print(f"[stage A] loaded in {record['reference_load_seconds']}s", flush=True)

    # RoPE and query-scale evidence (protocol item 1).
    rope_positions = torch.tensor(
        [int(value) for value in args.rope_positions.split(",")], dtype=torch.long
    )
    reference_inv_freq, reference_cos, reference_sin = oracle.rotary(rope_positions)

    # ar_generate relies on SDPA's implicit causality; prove it matches an
    # explicit causal mask before using either as the oracle.
    probe = select_probe_fixture(manifest["held_out"])["input_ids"]
    explicit = run_reference_fixture(oracle, probe, 0, explicit_mask=True)
    implicit = run_reference_fixture(oracle, probe, 0, explicit_mask=False)
    record["oracle_self_consistency"] = tensor_metrics(
        implicit.prefill_logits, explicit.prefill_logits
    )
    del implicit

    reference_results: dict[str, FixtureResult] = {}
    for fixture in all_fixtures:
        result = run_reference_fixture(
            oracle, fixture["input_ids"], args.decode_steps
        )
        result.name = fixture["name"]
        reference_results[fixture["name"]] = result
        print(f"[stage A] {fixture['name']} prefill+decode done", flush=True)

    # Calibration noise floor: same weights, same mask, different kernel.
    oracle.set_attn_implementation("eager")
    record["causality_probe"] = {
        "eager": causality_probe(oracle, probe),
    }
    calibration_floor = []
    for fixture in manifest["calibration"]:
        eager = run_reference_fixture(oracle, fixture["input_ids"], args.decode_steps)
        sdpa = reference_results[fixture["name"]]
        entry = {
            "name": fixture["name"],
            "prefill": tensor_metrics(eager.prefill_logits, sdpa.prefill_logits),
        }
        if eager.decode_logits is not None:
            entry["decode"] = tensor_metrics(eager.decode_logits, sdpa.decode_logits)
        calibration_floor.append(entry)
        print(f"[stage A] calibration floor {fixture['name']}", flush=True)
    oracle.set_attn_implementation("sdpa")
    record["causality_probe"]["sdpa"] = causality_probe(oracle, probe)
    record["calibration_floor"] = calibration_floor

    reference_greedy = {}
    for fixture in greedy_fixtures:
        reference_greedy[fixture["name"]] = greedy_reference(
            oracle, fixture["input_ids"], fixture["max_new_tokens"], eos_ids
        )
        print(f"[stage A] greedy {fixture['name']}", flush=True)

    trace_fixture = next(
        (f for f in all_fixtures if f["name"] == args.layer_trace_fixture), None
    )
    reference_layers = (
        capture_layer_states(oracle, trace_fixture["input_ids"])
        if trace_fixture
        else []
    )
    oracle.release()
    print("[stage A] reference released", flush=True)

    # Freeze tolerances before the held-out split is scored.
    floor_max_abs = max(
        [entry[key]["max_abs"] for entry in calibration_floor
         for key in ("prefill", "decode") if key in entry],
        default=0.0,
    )
    floor_nrms = max(
        [entry[key]["nrms"] for entry in calibration_floor
         for key in ("prefill", "decode") if key in entry],
        default=0.0,
    )
    tolerances = {
        "source": "reference sdpa vs reference eager on the calibration split",
        "floor_max_abs": floor_max_abs,
        "floor_nrms": floor_nrms,
        "slack": TOLERANCE_SLACK,
        "max_abs": max(TOLERANCE_SLACK * floor_max_abs, MIN_TOLERANCE_MAX_ABS),
        "nrms": max(TOLERANCE_SLACK * floor_nrms, MIN_TOLERANCE_NRMS),
        "ceiling_nrms": CALIBRATION_FLOOR_CEILING_NRMS,
    }
    record["tolerances"] = tolerances
    # A kernel-versus-kernel comparison of the same weights must be a small
    # effect. If it is not, the two runs are not comparing what we think, and
    # freezing a tolerance from it would produce a gate that passes anything.
    if floor_nrms > CALIBRATION_FLOOR_CEILING_NRMS or not all(
        entry["is_causal"] for entry in record["causality_probe"].values()
    ):
        record["checks"] = {"calibration_floor_sane": False}
        record["passed"] = False
        (output_dir / "metrics.json").write_text(json.dumps(record, indent=2))
        print(
            "[abort] calibration floor is not a noise floor: "
            f"nrms={floor_nrms:.3e} ceiling={CALIBRATION_FLOOR_CEILING_NRMS}, "
            f"causality={record['causality_probe']}",
            flush=True,
        )
        print("PHASE1 FAIL", flush=True)
        return 1
    print(f"[stage A] frozen tolerances {tolerances}", flush=True)

    # ---------------- Stage B: FluxServe candidate ----------------
    print("[stage B] loading FluxServe model", flush=True)
    started = time.perf_counter()
    candidate = FluxServeAR.load(model_config, args.device, max_length + 8)
    record["candidate_load_seconds"] = round(time.perf_counter() - started, 1)
    print(f"[stage B] loaded in {record['candidate_load_seconds']}s", flush=True)

    candidate_inv_freq, candidate_cos, candidate_sin = candidate.rotary(rope_positions)
    half = candidate_cos.shape[-1]
    record["rope"] = {
        "positions": rope_positions.tolist(),
        "layout_note": (
            "FluxServe stores half-width cos/sin; the checkpoint duplicates "
            "them to head_dim. The shared half is compared."
        ),
        "cos": tensor_metrics(candidate_cos, reference_cos[..., :half]),
        "sin": tensor_metrics(candidate_sin, reference_sin[..., :half]),
        "reference_duplicates_halves": bool(
            torch.equal(reference_cos[..., :half], reference_cos[..., half:])
        ),
        "inv_freq": tensor_metrics(candidate_inv_freq, reference_inv_freq)
        if candidate_inv_freq is not None
        else None,
    }

    comparisons: dict[str, dict] = {}
    for fixture in all_fixtures:
        name = fixture["name"]
        result = run_candidate_fixture(
            candidate, fixture["input_ids"], args.decode_steps
        )
        reference = reference_results[name]
        entry = {
            "split": "calibration"
            if name in record["fixtures"]["calibration"]
            else "held_out",
            "length": fixture["length"],
            "prefill": tensor_metrics(result.prefill_logits, reference.prefill_logits),
        }
        if result.decode_logits is not None:
            entry["decode"] = tensor_metrics(
                result.decode_logits, reference.decode_logits
            )
            entry["decode_positions"] = result.decode_positions
        comparisons[name] = entry
        print(f"[stage B] compared {name}", flush=True)
    record["comparisons"] = comparisons

    greedy_report = {}
    for fixture in greedy_fixtures:
        name = fixture["name"]
        produced = greedy_candidate(
            candidate, fixture["input_ids"], fixture["max_new_tokens"], eos_ids
        )
        expected = reference_greedy[name]
        divergence = None
        for index, (left, right) in enumerate(
            zip(produced["tokens"], expected["tokens"])
        ):
            if left != right:
                divergence = {
                    "index": index,
                    "candidate_token": left,
                    "reference_token": right,
                    "reference_margin": expected["margins"][index],
                }
                break
        greedy_report[name] = {
            "matched": divergence is None
            and len(produced["tokens"]) == len(expected["tokens"]),
            "candidate_tokens": produced["tokens"],
            "reference_tokens": expected["tokens"],
            "first_divergence": divergence,
            "reference_margin_min": min(expected["margins"]),
        }
        print(f"[stage B] greedy {name} matched={greedy_report[name]['matched']}",
              flush=True)
    record["greedy"] = greedy_report

    if trace_fixture and reference_layers:
        candidate_layers = capture_layer_states(candidate, trace_fixture["input_ids"])
        record["layer_trace"] = {
            "fixture": trace_fixture["name"],
            "layers": [
                {
                    "layer": index,
                    "nrms": tensor_metrics(produced, expected)["nrms"],
                    "max_abs": tensor_metrics(produced, expected)["max_abs"],
                }
                for index, (produced, expected) in enumerate(
                    zip(candidate_layers, reference_layers)
                )
            ],
        }
    candidate.release()

    # ---------------- Gate ----------------
    held_out = [
        entry for entry in comparisons.values() if entry["split"] == "held_out"
    ]
    stages = [
        entry[key] for entry in held_out for key in ("prefill", "decode") if key in entry
    ]
    unexplained = []
    for name, entry in comparisons.items():
        if entry["split"] != "held_out":
            continue
        for key in ("prefill", "decode"):
            if key not in entry:
                continue
            for position, margin in zip(
                entry[key]["disagreement_positions"],
                entry[key]["disagreement_margins"],
            ):
                if margin > tolerances["max_abs"]:
                    unexplained.append(
                        {"fixture": name, "stage": key, "position": position,
                         "reference_margin": margin}
                    )

    checks = {
        "all_finite": all(entry["finite"] for entry in stages),
        "max_abs_within_tolerance": worst(stages, "max_abs") <= tolerances["max_abs"],
        "nrms_within_tolerance": worst(stages, "nrms") <= tolerances["nrms"],
        "no_unexplained_top1_disagreement": not unexplained,
        "greedy_matches_reference": all(
            item["matched"] for item in greedy_report.values()
        ) if greedy_report else None,
        "oracle_self_consistent": record["oracle_self_consistency"]["top1_agreement"]
        == 1.0,
    }
    record["worst_held_out"] = {
        "max_abs": worst(stages, "max_abs"),
        "nrms": worst(stages, "nrms"),
        "min_top1_agreement": min((s["top1_agreement"] for s in stages), default=1.0),
    }
    record["unexplained_disagreements"] = unexplained
    record["checks"] = checks
    record["passed"] = all(value for value in checks.values() if value is not None)

    (output_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    (output_dir / "summary.md").write_text(render_summary(record))
    print(json.dumps(checks, indent=2), flush=True)
    print(f"PHASE1 {'PASS' if record['passed'] else 'FAIL'}", flush=True)
    return 0 if record["passed"] else 1


def render_summary(record: dict) -> str:
    environment = record["environment"]
    lines = [
        "# Nemotron-Labs-Diffusion Phase 1 — AR validation",
        "",
        f"**Result: {'PASS' if record['passed'] else 'FAIL'}**",
        "",
        "## Environment",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for key, value in environment.items():
        lines.append(f"| {key} | `{value}` |")
    lines += [
        "",
        "## Frozen tolerances",
        "",
        "Derived from the calibration split by comparing the reference against "
        "itself under two attention kernels (SDPA vs eager, identical explicit "
        "causal mask), then multiplied by the slack factor. Frozen before the "
        "held-out split was scored.",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for key, value in record["tolerances"].items():
        lines.append(f"| {key} | `{value}` |")

    lines += ["", "## Checks", "", "| Check | Result |", "| --- | --- |"]
    for key, value in record["checks"].items():
        mark = "skipped" if value is None else ("pass" if value else "**FAIL**")
        lines.append(f"| {key} | {mark} |")

    lines += [
        "",
        "## Held-out fixtures (FluxServe vs reference)",
        "",
        "| Fixture | Len | Stage | max_abs | nrms | top1 agree | ref margin min |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in record["comparisons"].items():
        if entry["split"] != "held_out":
            continue
        for stage in ("prefill", "decode"):
            if stage not in entry:
                continue
            metrics = entry[stage]
            lines.append(
                f"| {name} | {entry['length']} | {stage} | "
                f"{metrics['max_abs']:.3e} | {metrics['nrms']:.3e} | "
                f"{metrics['top1_agreement']:.4f} | "
                f"{metrics['reference_margin_min']:.3e} |"
            )

    positions = ", ".join(str(value) for value in record["rope"]["positions"])
    lines += ["", f"## RoPE (positions {positions})", "",
              f"- {record['rope']['layout_note']}", ""]
    for key in ("cos", "sin"):
        metrics = record["rope"][key]
        lines.append(
            f"- `{key}`: max_abs `{metrics['max_abs']:.3e}`, "
            f"nrms `{metrics['nrms']:.3e}`"
        )

    if record.get("greedy"):
        lines += ["", "## Greedy completions", ""]
        for name, item in record["greedy"].items():
            lines.append(
                f"- `{name}`: matched={item['matched']}, "
                f"{len(item['candidate_tokens'])} tokens, "
                f"reference margin min `{item['reference_margin_min']:.3e}`"
            )
            if item["first_divergence"]:
                lines.append(f"  - first divergence: `{item['first_divergence']}`")

    if record.get("layer_trace"):
        trace = record["layer_trace"]
        worst_layer = max(trace["layers"], key=lambda item: item["nrms"])
        lines += [
            "",
            f"## Layer trace ({trace['fixture']})",
            "",
            f"- worst layer {worst_layer['layer']}: nrms "
            f"`{worst_layer['nrms']:.3e}`, max_abs `{worst_layer['max_abs']:.3e}`",
            f"- first layer nrms `{trace['layers'][0]['nrms']:.3e}`, "
            f"last layer nrms `{trace['layers'][-1]['nrms']:.3e}`",
        ]

    if record["unexplained_disagreements"]:
        lines += ["", "## Unexplained top-1 disagreements", ""]
        for item in record["unexplained_disagreements"][:20]:
            lines.append(f"- `{item}`")

    if record.get("causality_probe"):
        lines += ["", "## Causality probe", "",
                  "| Implementation | prefix max_abs | final max_abs | causal |",
                  "| --- | --- | --- | --- |"]
        for name, item in record["causality_probe"].items():
            lines.append(
                f"| {name} | {item['prefix_max_abs']:.3e} | "
                f"{item['final_position_max_abs']:.3e} | {item['is_causal']} |"
            )

    lines += [
        "",
        "## Oracle self-consistency",
        "",
        "`ar_generate` builds no attention mask and relies on SDPA's implicit "
        "causality. This compares that path against an explicit causal mask on "
        "the same prompt; they must agree for either to serve as the oracle.",
        "",
        f"- top1 agreement `{record['oracle_self_consistency']['top1_agreement']}`, "
        f"max_abs `{record['oracle_self_consistency']['max_abs']:.3e}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
