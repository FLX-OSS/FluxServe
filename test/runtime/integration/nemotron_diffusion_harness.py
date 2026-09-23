"""Phase 2/3 diffusion validation for Nemotron-Labs-Diffusion.

Compares FluxServe's block contract against the checkpoint's own
``generate(causal_context=True)`` at temperature zero, and then compares the
paged FA4 path against the dense one.

Run as separate modes because two Python environments are involved: the dense
and reference lanes run in the conda environment, while FA4 needs the pinned
``.ci-artifacts/fa4-runtime`` virtualenv. Each mode writes an artifact; the
final ``compare`` mode reads them and writes the summary.

    --mode reference            the checkpoint's own generate(), instrumented
    --mode dense               NemotronDiffusionRunner
    --mode fa4                 NemotronFA4DiffusionRunner (run under the FA4 venv)
    --mode flashinfer          NemotronFlashInferDiffusionRunner (std FlashInfer)
    --mode selfspec_fa4        NemotronSelfSpecPagedRunner
    --mode selfspec_flashinfer NemotronFlashInferSelfSpecRunner
    --mode compare             read the artifacts, gate, write summary.md

Two options add passes to the reference and dense lanes rather than new modes,
because both reuse the weights already resident:

    --lora             also draft with the checkpoint's ``linear_spec_lora``.
                       The reference applies it through PEFT, unfused; FluxServe
                       pre-fuses it into a second o_proj weight. This lane is
                       the only thing that can tell those two apart, and the
                       only one that reads the real 189 MB adapter file.
    --thinking-budget  also run with a thinking budget, which the reference
                       enforces itself, so the forced end-of-thinking marker is
                       compared rather than assumed.

What is compared, beyond the final text: the block state after *every*
denoising step, each block's seed token, the number of denoising forwards per
block, and each block's committed keys and values per layer. A run that agrees
on the final tokens but disagrees on the committed KV has not reproduced the
model, it has been lucky.

See ``docs/serving/nemotron/nemotron-labs-diffusion-14B.md`` for configuration.
"""

from __future__ import annotations

import argparse
import gc
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


def load_fixtures(path: str) -> list[dict]:
    with open(path) as handle:
        manifest = json.load(handle)
    if "diffusion" not in manifest:
        raise ValueError(f"{path} has no 'diffusion' fixtures")
    return manifest["diffusion"]


def model_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True
    )


def kv_tail(tensor: torch.Tensor, count: int) -> torch.Tensor:
    return tensor[..., -count:, :].detach().float().cpu()


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


# ---------------------------------------------------------------------------
# Reference lane
# ---------------------------------------------------------------------------


def record_pass_error(results, key: str, exc: BaseException) -> None:
    """Note that an optional pass failed, without losing the whole lane.

    The lane writes one artifact at the end, so an exception in an extra pass
    would throw away the primary comparison too. The error is recorded per
    fixture instead, and the gate turns a recorded error into a failing check
    rather than a quietly missing one.
    """
    message = f"{type(exc).__name__}: {exc}"
    for entry in results.values():
        entry[f"{key}_error"] = message
    print(f"[lane] optional pass {key} FAILED -- {message}", flush=True)


def end_think_token_id(config):
    """The checkpoint's own ``</think>`` id, or ``None`` if it declares none."""
    from fluxserve.backend.model_loader.nemotron import resolve_end_think_token_id

    return resolve_end_think_token_id(config)


def attach_reference_adapter(model) -> int:
    """Attach ``linear_spec_lora`` to the reference model through PEFT.

    The model card's snippet is ``PeftModel.from_pretrained(model, repo,
    subfolder="linear_spec_lora")``, and ``linear_spec_generate`` then toggles
    the adapter itself by flipping ``_disable_adapters`` on every module that
    has one. So all this has to do is make the o_proj modules real tuner
    layers; the phase discipline is the reference's own.

    Left deliberately unfused, which is what makes the comparison worth
    running: PEFT evaluates ``B @ A @ x`` at adapter dtype and adds it to the
    base output, while FluxServe folds the same update into a second bfloat16
    weight once. Returns the number of adapted modules.
    """
    from peft import PeftModel

    from fluxserve.backend.model_loader.nemotron import (
        LORA_SUBFOLDER,
        resolve_nemotron_snapshot,
    )

    snapshot = resolve_nemotron_snapshot(model.config, local_files_only=True)
    path = Path(snapshot) / LORA_SUBFOLDER
    if not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"no draft adapter at {path}")
    # `model` is mutated in place: the wrapper is only a handle, and
    # linear_spec_generate has to be called on the base model.
    PeftModel.from_pretrained(model, str(path))
    adapted = sum(
        1 for module in model.modules() if hasattr(module, "_disable_adapters")
    )
    if not adapted:
        raise AssertionError("PEFT attached no adapter layers")
    return adapted


def run_reference(fixtures, device: str, *, lora=False, thinking=None) -> dict:
    from transformers import AutoModel

    common = {"revision": REVISION, "trust_remote_code": True,
              "attn_implementation": "sdpa"}
    try:
        model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16, **common)
    except TypeError:
        model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, **common)
    model = model.to(device).eval()

    results = {}
    for fixture in fixtures:
        block_length = int(fixture["block_length"])
        trace = {"forwards": [], "committed_kv": [], "seeds": []}
        original = model.forward

        def recording_forward(*args, **kwargs):
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            causal = bool(kwargs.get("use_causal_mask", False))
            use_cache = bool(kwargs.get("use_cache", False))
            output = original(*args, **kwargs)
            trace["forwards"].append({
                "tokens": input_ids[0].tolist(),
                "causal": causal,
                "use_cache": use_cache,
                "length": int(input_ids.shape[1]),
            })
            if use_cache and causal:
                # The first causal cache-writing forward is the prefill; only
                # the block commits after it are compared as committed KV.
                if trace["forwards"] and len(trace["seeds"]) > 0:
                    cache = output.past_key_values
                    trace["committed_kv"].append([
                        (kv_tail(layer.keys, block_length),
                         kv_tail(layer.values, block_length))
                        for layer in cache.layers
                    ])
                # Every causal forward's final logit is a seed, prefill's
                # included: it seeds the first block.
                trace["seeds"].append(int(output.logits[0, -1].argmax()))
            return output

        model.forward = recording_forward
        try:
            prompt = torch.tensor(
                [fixture["input_ids"]], dtype=torch.long, device=device
            )
            with torch.no_grad():
                tokens, nfe = model.generate(
                    prompt,
                    max_new_tokens=int(fixture["max_new_tokens"]),
                    block_length=block_length,
                    threshold=float(fixture["threshold"]),
                    causal_context=True,
                    temperature=0.0,
                    eos_token_id=11,
                )
        finally:
            model.forward = original
        results[fixture["name"]] = {
            "output_ids": tokens[0].detach().cpu().tolist(),
            "generated": tokens[0, prompt.shape[1]:].detach().cpu().tolist(),
            "nfe": int(nfe),
            "forwards": trace["forwards"],
            "seeds": trace["seeds"],
            "committed_kv": trace["committed_kv"],
        }
        print(f"[reference] {fixture['name']} nfe={nfe} "
              f"forwards={len(trace['forwards'])}", flush=True)

        # Self-speculation on the same already-loaded weights, no adapter --
        # the candidate runner is also built without one, so the comparison is
        # like for like.
        with torch.no_grad():
            spec_tokens, spec_nfe = model.linear_spec_generate(
                prompt,
                max_new_tokens=int(fixture["max_new_tokens"]),
                block_length=block_length,
                temperature=0.0,
                eos_token_id=11,
                threshold=0.0,
            )
        results[fixture["name"]]["selfspec"] = {
            "generated": spec_tokens[0, prompt.shape[1]:].detach().cpu().tolist(),
            "nfe": int(spec_nfe),
        }
        print(f"[reference] {fixture['name']} selfspec nfe={spec_nfe}", flush=True)

    if thinking is not None:
        try:
            marker = end_think_token_id(model.config)
            if marker is None:
                raise ValueError(
                    "the checkpoint declares no end-of-thinking token, so "
                    "--thinking-budget has nothing to compare against"
                )
            budget = {"max_thinking_tokens": int(thinking),
                      "end_think_token_id": int(marker)}
            for fixture in fixtures:
                prompt = torch.tensor(
                    [fixture["input_ids"]], dtype=torch.long, device=device
                )
                block_length = int(fixture["block_length"])
                with torch.no_grad():
                    think_tokens, think_nfe = model.generate(
                        prompt,
                        max_new_tokens=int(fixture["max_new_tokens"]),
                        block_length=block_length,
                        threshold=float(fixture["threshold"]),
                        causal_context=True,
                        temperature=0.0,
                        eos_token_id=11,
                        **budget,
                    )
                    spec_think_tokens, spec_think_nfe = model.linear_spec_generate(
                        prompt,
                        max_new_tokens=int(fixture["max_new_tokens"]),
                        block_length=block_length,
                        temperature=0.0,
                        eos_token_id=11,
                        threshold=0.0,
                        **budget,
                    )
                for key, tokens, count in (
                    ("thinking", think_tokens, think_nfe),
                    ("selfspec_thinking", spec_think_tokens, spec_think_nfe),
                ):
                    results[fixture["name"]][key] = {
                        "generated": tokens[0, prompt.shape[1]:]
                        .detach().cpu().tolist(),
                        "nfe": int(count),
                        "budget": int(thinking),
                        "marker": int(marker),
                    }
                print(f"[reference] {fixture['name']} thinking budget={thinking} "
                      f"nfe={think_nfe}/{spec_think_nfe}", flush=True)
        except Exception as exc:  # noqa: BLE001 - the primary lane must survive
            record_pass_error(results, "thinking", exc)

    if lora:
        # Last, because the adapter mutates the model in place, and only on the
        # speculation path, because drafting is the only phase that uses it.
        try:
            adapted = attach_reference_adapter(model)
            print(f"[reference] PEFT adapter on {adapted} modules", flush=True)
            for fixture in fixtures:
                prompt = torch.tensor(
                    [fixture["input_ids"]], dtype=torch.long, device=device
                )
                with torch.no_grad():
                    spec_tokens, spec_nfe = model.linear_spec_generate(
                        prompt,
                        max_new_tokens=int(fixture["max_new_tokens"]),
                        block_length=int(fixture["block_length"]),
                        temperature=0.0,
                        eos_token_id=11,
                        threshold=0.0,
                    )
                results[fixture["name"]]["selfspec_lora"] = {
                    "generated": spec_tokens[0, prompt.shape[1]:]
                    .detach().cpu().tolist(),
                    "nfe": int(spec_nfe),
                    "adapted_modules": int(adapted),
                }
                print(f"[reference] {fixture['name']} selfspec+lora nfe={spec_nfe}",
                      flush=True)
        except Exception as exc:  # noqa: BLE001
            record_pass_error(results, "selfspec_lora", exc)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# FluxServe lanes
# ---------------------------------------------------------------------------


def build_runner(config, fixtures, device: str, backend: str,
                 decoding: str = "threshold"):
    from fluxserve.backend.execution.forward_batch_info import RunnerConfig
    from fluxserve.backend.model_loader.nemotron import (
        apply_nemotron_runner_config,
    )
    from fluxserve.backend.utils.server_args import ServerArgs

    block_length = int(fixtures[0]["block_length"])
    max_length = max(
        fixture["length"] + fixture["max_new_tokens"] for fixture in fixtures
    ) + block_length
    runner_config = RunnerConfig(
        gen_length=max(f["max_new_tokens"] for f in fixtures),
        block_length=block_length,
        max_length=max_length,
        threshold=float(fixtures[0]["threshold"]),
        attention_backend=backend,
        kv_cache_layout="paged" if backend in ("fa4", "flashinfer") else "dense",
        page_size=block_length if backend in ("fa4", "flashinfer") else None,
        flashinfer_cache_mode="paged" if backend == "flashinfer" else "dense",
        flashinfer_prefill_mode="paged" if backend == "flashinfer" else "dense",
    )
    apply_nemotron_runner_config(runner_config, config)
    server_args = ServerArgs(
        model_name=MODEL, model_config=config, device=device,
        max_num_seqs=1, max_model_len=max_length,
    )
    from fluxserve.backend.execution.runners.nemotron import get_nemotron_runner

    runner_cls = get_nemotron_runner(backend, decoding)
    return runner_cls(
        model_config=config, server_args=server_args,
        runner_config=runner_config, device=device,
    )


def adapter_fingerprint(adapter) -> dict:
    """What the fused draft adapter actually changed, for the gate and summary.

    Read from the stored weight pair rather than the live module, so it does not
    depend on which way the pointer happens to be pointing.
    """
    _, base, draft = adapter.layers[0]
    reference = base.to(torch.float32)
    delta = draft.to(torch.float32) - reference
    return {
        "layers": len(adapter.layers),
        "o_proj_shape": list(base.shape),
        "layer0_delta_nrms": float(
            delta.pow(2).mean().sqrt() / reference.pow(2).mean().sqrt()
        ),
        "layer0_delta_max_abs": float(delta.abs().max()),
    }


def run_dense(fixtures, device: str, *, lora=False, thinking=None) -> dict:
    config = model_config()
    runner = build_runner(config, fixtures, device, "sdpa")
    results = {}
    for fixture in fixtures:
        trace = {"forwards": [], "committed_kv": [], "seeds": []}
        original_forward = runner._forward
        original_commit = runner._commit_kv

        def recording_forward(**kwargs):
            output = original_forward(**kwargs)
            mask = kwargs.get("attention_mask")
            trace["forwards"].append({
                "tokens": kwargs["input_ids"][0].tolist(),
                "causal": bool(mask is not None and not bool(mask.all())),
                "use_cache": bool(kwargs.get("use_cache")),
                "length": int(kwargs["input_ids"].shape[1]),
            })
            end = int(kwargs["position_ids"].max()) + 1
            if kwargs.get("use_cache") and end >= fixture["length"]:
                trace["seeds"].append(int(output.logits[0, -1].argmax()))
            return output

        def recording_commit(cache, present, start, end):
            original_commit(cache, present, start, end)
            if end <= fixture["length"]:
                return  # the prefill's write, not a block commit
            layers = len(present) // 2
            trace["committed_kv"].append([
                (kv_tail(present[2 * i], end - start),
                 kv_tail(present[2 * i + 1], end - start))
                for i in range(layers)
            ])

        runner._forward = recording_forward
        runner._commit_kv = recording_commit
        try:
            prompt = torch.tensor(
                [fixture["input_ids"]], dtype=torch.long, device=device
            )
            output = runner.generate(
                prompt,
                prompt_lengths=[fixture["length"]],
                generation_lengths=[int(fixture["max_new_tokens"])],
            )
        finally:
            runner._forward = original_forward
            runner._commit_kv = original_commit
        generated = output[0, fixture["length"]:].detach().cpu().tolist()
        results[fixture["name"]] = {
            "generated": generated,
            "stats": runner.last_stats[0],
            "forwards": trace["forwards"],
            "seeds": trace["seeds"],
            "committed_kv": trace["committed_kv"],
        }
        print(f"[dense] {fixture['name']} stats={runner.last_stats[0]}", flush=True)

    # Self-speculation shares the weights rather than loading a second copy:
    # 27 GB is worth avoiding, and the runner differs only in its block loop.
    spec = self_spec_view(runner)
    for fixture in fixtures:
        prompt = torch.tensor([fixture["input_ids"]], dtype=torch.long, device=device)
        output = spec.generate(
            prompt,
            prompt_lengths=[fixture["length"]],
            generation_lengths=[int(fixture["max_new_tokens"])],
        )
        results[fixture["name"]]["selfspec"] = {
            "generated": output[0, fixture["length"]:].detach().cpu().tolist(),
            "stats": spec.last_stats[0],
        }
        print(f"[dense] {fixture['name']} selfspec={spec.last_stats[0]}", flush=True)

    if thinking is not None:
        # Captured before anything can fail, because the budget is restored in a
        # finally clause: the plain passes above must not be affected by it.
        restore = (runner.thinking_budget, spec.thinking_budget)
        try:
            marker = end_think_token_id(config)
            if marker is None:
                raise ValueError(
                    "this checkpoint declares no end-of-thinking token, so a "
                    "thinking budget cannot be enforced"
                )
            from fluxserve.backend.execution.decoders.nemotron import ThinkingBudget

            budget = ThinkingBudget(max_thinking_tokens=int(thinking),
                                    end_think_token_id=int(marker))
            runner.thinking_budget = budget
            spec.thinking_budget = budget
            for fixture in fixtures:
                prompt = torch.tensor(
                    [fixture["input_ids"]], dtype=torch.long, device=device
                )
                for key, target in (("thinking", runner),
                                    ("selfspec_thinking", spec)):
                    produced = target.generate(
                        prompt,
                        prompt_lengths=[fixture["length"]],
                        generation_lengths=[int(fixture["max_new_tokens"])],
                    )
                    results[fixture["name"]][key] = {
                        "generated": produced[0, fixture["length"]:]
                        .detach().cpu().tolist(),
                        "stats": target.last_stats[0],
                        "budget": int(thinking),
                        "marker": int(marker),
                    }
                print(f"[dense] {fixture['name']} thinking budget={thinking}",
                      flush=True)
        except Exception as exc:  # noqa: BLE001 - the primary lane must survive
            record_pass_error(results, "thinking", exc)
        finally:
            runner.thinking_budget, spec.thinking_budget = restore

    if lora:
        try:
            spec.load_draft_adapter()
            if spec.lora is None:
                raise FileNotFoundError(
                    "no draft adapter beside the checkpoint; --lora has nothing "
                    "to load"
                )
            fingerprint = adapter_fingerprint(spec.lora)
            print(f"[dense] draft adapter {fingerprint}", flush=True)
            for fixture in fixtures:
                prompt = torch.tensor(
                    [fixture["input_ids"]], dtype=torch.long, device=device
                )
                produced = spec.generate(
                    prompt,
                    prompt_lengths=[fixture["length"]],
                    generation_lengths=[int(fixture["max_new_tokens"])],
                )
                results[fixture["name"]]["selfspec_lora"] = {
                    "generated": produced[0, fixture["length"]:]
                    .detach().cpu().tolist(),
                    "stats": spec.last_stats[0],
                    "adapter": fingerprint,
                }
                print(f"[dense] {fixture['name']} selfspec+lora="
                      f"{spec.last_stats[0]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            record_pass_error(results, "selfspec_lora", exc)
    return results


def self_spec_view(runner):
    """A self-speculation runner over an already-loaded diffusion runner."""
    from fluxserve.backend.execution.decoders.nemotron import (
        NemotronThresholdDecoder,
    )
    from fluxserve.backend.execution.runners.nemotron_selfspec import (
        NemotronSelfSpecRunner,
    )

    spec = object.__new__(NemotronSelfSpecRunner)
    spec.__dict__.update(runner.__dict__)
    spec.draft_threshold = 0.0
    spec.draft_decoder = NemotronThresholdDecoder(
        threshold=0.0,
        mask_id=runner.decoder.mask_id,
        eos_ids=runner.decoder.eos_ids,
    )
    spec.lora = None
    spec.last_stats = []
    return spec


def run_fa4(fixtures, device: str) -> dict:
    return run_paged(fixtures, device, "fa4")


def run_flashinfer(fixtures, device: str) -> dict:
    return run_paged(fixtures, device, "flashinfer")


def run_paged_selfspec(fixtures, device: str, backend: str, *, lora=False) -> dict:
    """Self-speculation on the paged path, adapter-free and then with the adapter.

    The comparison target is the dense self-speculation lane, not the reference:
    what is new here is the rollback, which is page accounting, and the dense
    lane is already tied to the reference token for token.
    """
    config = model_config()
    runner = build_runner(config, fixtures, device, backend,
                          decoding="self_speculation")
    results: dict = {}
    passes = [("selfspec", False)] + ([("selfspec_lora", True)] if lora else [])
    for key, with_adapter in passes:
        if with_adapter:
            runner.load_draft_adapter()
            if runner.lora is None:
                raise FileNotFoundError(
                    "no draft adapter beside the checkpoint; --lora has nothing "
                    "to load"
                )
        for fixture in fixtures:
            prompt = torch.tensor(
                [fixture["input_ids"]], dtype=torch.long, device=device
            )
            produced = runner.generate(
                prompt,
                prompt_lengths=[fixture["length"]],
                generation_lengths=[int(fixture["max_new_tokens"])],
            )
            entry = results.setdefault(fixture["name"], {})
            entry[key] = {
                "generated": produced[0, fixture["length"]:].detach().cpu().tolist(),
                "stats": runner.last_stats[0],
            }
            if with_adapter:
                entry[key]["adapter"] = adapter_fingerprint(runner.lora)
            print(f"[{backend} selfspec] {fixture['name']} {key}="
                  f"{runner.last_stats[0]}", flush=True)
    return results


def run_paged(fixtures, device: str, backend: str) -> dict:
    config = model_config()
    runner = build_runner(config, fixtures, device, backend)
    results = {}
    for fixture in fixtures:
        launches = []
        original = runner._paged_forward

        def recording(**kwargs):
            logits = original(**kwargs)
            launches.append({
                "rows": kwargs["seq_ids"].tolist(),
                "causal": bool(kwargs["causal"]),
                "prefill": bool(kwargs["is_prefill"]),
                "tokens": kwargs["tokens"][0].tolist(),
            })
            return logits

        runner._paged_forward = recording
        try:
            prompt = torch.tensor(
                [fixture["input_ids"]], dtype=torch.long, device=device
            )
            output = runner.generate(
                prompt,
                prompt_lengths=[fixture["length"]],
                generation_lengths=[int(fixture["max_new_tokens"])],
            )
        finally:
            runner._paged_forward = original
        results[fixture["name"]] = {
            "generated": output[0, fixture["length"]:].detach().cpu().tolist(),
            "stats": runner.last_stats[0],
            "launches": launches,
        }
        print(f"[{backend}] {fixture['name']} stats={runner.last_stats[0]}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def block_states(forwards: list[dict], block_length: int) -> list[list[int]]:
    """Every block forward's input tokens, in order, prefill excluded.

    Includes the commit forward, whose input is the block's final tokens, so a
    lane that resolves a block differently but lands on the same text still
    shows up as a mismatch.
    """
    return [
        item["tokens"]
        for item in forwards[1:]
        if item["length"] == block_length
    ]


def kv_metrics(candidate, reference) -> dict:
    worst_abs = 0.0
    worst_nrms = 0.0
    worst_layer = -1
    for layer, (produced, expected) in enumerate(zip(candidate, reference)):
        for side in range(2):
            difference = produced[side] - expected[side]
            max_abs = float(difference.abs().max())
            rms = float(difference.pow(2).mean().sqrt())
            reference_rms = float(expected[side].pow(2).mean().sqrt())
            nrms = rms / max(reference_rms, 1e-12)
            if nrms > worst_nrms:
                worst_nrms, worst_layer = nrms, layer
            worst_abs = max(worst_abs, max_abs)
    return {"max_abs": worst_abs, "worst_nrms": worst_nrms, "worst_layer": worst_layer}


def compare_flashinfer(output_dir: Path, dense: dict) -> dict:
    """Fail if a requested FlashInfer artifact is missing or incomplete."""
    path = output_dir / "flashinfer.pt"
    paged = torch.load(path, weights_only=False)
    checks = {"flashinfer_fixture_coverage": bool(dense["results"]) and (
        set(paged["results"]) == set(dense["results"])
    )}
    for name, expected in dense["results"].items():
        candidate = paged["results"].get(name)
        if candidate is None:
            checks[f"flashinfer_{name}_present"] = False
            continue
        checks[f"flashinfer_{name}_tokens"] = candidate["generated"] == expected["generated"]
        for counter in ("prefill_calls", "denoise_calls", "commit_calls"):
            checks[f"flashinfer_{name}_{counter}"] = (
                candidate["stats"][counter] == expected["stats"][counter]
            )
        launches = candidate["launches"]
        counts = {
            "prefill_calls": sum(item["prefill"] and item["causal"] for item in launches),
            "denoise_calls": sum(not item["prefill"] and not item["causal"] for item in launches),
            "commit_calls": sum(not item["prefill"] and item["causal"] for item in launches),
        }
        checks[f"flashinfer_{name}_causality"] = (
            bool(launches) and launches[0]["prefill"] and launches[-1]["causal"]
            and sum(counts.values()) == len(launches)
            and all(counts[key] == candidate["stats"][key] for key in counts)
        )
    return {"checks": checks, "provenance": paged["provenance"]}


EXTRA_REFERENCE_KEYS = ("thinking", "selfspec_thinking", "selfspec_lora")
SELFSPEC_PAGED_MODES = ("selfspec_fa4", "selfspec_flashinfer")


def first_difference(candidate: list, reference: list):
    """Index of the first differing token, or ``None`` when one is a prefix."""
    for index, (left, right) in enumerate(zip(candidate, reference)):
        if left != right:
            return index
    return None


def token_index(tokens: list, token: int):
    return tokens.index(token) if token in tokens else None


def compare_extra_pass(key, expected, produced, produced_fixture, block_length):
    """One reference-vs-dense pass that is not the plain threshold decode.

    Thinking-budget passes count denoise + commit forwards like the threshold
    lane; self-speculation passes count prefill + draft + verify, which is what
    the reference's own ``nfe`` counts. Where the reference forced an
    end-of-thinking marker, its position is checked rather than assumed: the
    budget bounds it to the block that carries the count past the limit.
    """
    reference_ids = expected["generated"]
    candidate_ids = produced["generated"]
    stats = produced["stats"]
    item = {
        "tokens_match": candidate_ids[: len(reference_ids)] == reference_ids,
        "first_divergence_index": first_difference(candidate_ids, reference_ids),
        "reference_generated": reference_ids,
        "candidate_generated": candidate_ids,
        "reference_nfe": expected["nfe"],
    }
    if key == "thinking":
        item["candidate_calls"] = stats["denoise_calls"] + stats["commit_calls"]
    else:
        item["candidate_calls"] = stats["total_calls"]
        item["mean_acceptance"] = stats["mean_acceptance"]
        item["tokens_per_forward"] = stats["tokens_per_forward"]
    item["nfe_match"] = item["candidate_calls"] == expected["nfe"]

    if "marker" in expected:
        marker, budget = int(expected["marker"]), int(expected["budget"])
        item["budget"] = budget
        item["marker"] = marker
        item["marker_index_reference"] = token_index(reference_ids, marker)
        item["marker_index_candidate"] = token_index(candidate_ids, marker)
        # The marker lands in the block that carries the count past the budget,
        # so the bound is the budget plus at most one block.
        item["marker_within_budget"] = (
            item["marker_index_candidate"] is not None
            and item["marker_index_candidate"] <= budget + block_length
        )
    if key == "selfspec_lora":
        item["adapter"] = produced.get("adapter")
        item["reference_adapted_modules"] = expected.get("adapted_modules")
        plain = produced_fixture.get("selfspec")
        if plain is not None:
            item["acceptance_without_adapter"] = plain["stats"]["mean_acceptance"]
            # Reported, not gated: the model card claims the adapter raises
            # acceptance, but a 32-token fixture is too short to hold that to.
            item["adapter_improves_acceptance"] = (
                stats["mean_acceptance"] >= plain["stats"]["mean_acceptance"]
            )
    return item


def compare_paged_selfspec(output_dir: Path, dense: dict, mode: str) -> dict:
    """Paged self-speculation against the dense lane, adapter state by adapter state."""
    paged = torch.load(output_dir / f"{mode}.pt", weights_only=False)
    checks = {
        f"{mode}_fixture_coverage": bool(dense["results"])
        and set(paged["results"]) == set(dense["results"])
    }
    for name, expected in dense["results"].items():
        candidate = paged["results"].get(name)
        if candidate is None:
            checks[f"{mode}_{name}_present"] = False
            continue
        for key in ("selfspec", "selfspec_lora"):
            if key not in expected or key not in candidate:
                continue
            checks[f"{mode}_{name}_{key}_tokens"] = (
                candidate[key]["generated"] == expected[key]["generated"]
            )
            checks[f"{mode}_{name}_{key}_calls"] = (
                candidate[key]["stats"]["total_calls"]
                == expected[key]["stats"]["total_calls"]
            )
            # Same acceptance sequence, not just the same total: two runs can
            # emit the same text having accepted it in different chunks, and
            # that difference is exactly what a rollback bug looks like.
            checks[f"{mode}_{name}_{key}_acceptance"] = (
                candidate[key]["stats"]["accepted_per_iteration"]
                == expected[key]["stats"]["accepted_per_iteration"]
            )
    return {"checks": checks, "provenance": paged["provenance"],
            "results": paged["results"]}


def compare(output_dir: Path, kv_nrms_tolerance: float, *, require_flashinfer=False) -> dict:
    reference = torch.load(output_dir / "reference.pt", weights_only=False)
    dense = torch.load(output_dir / "dense.pt", weights_only=False)
    fa4_path = output_dir / "fa4.pt"
    fa4 = torch.load(fa4_path, weights_only=False) if fa4_path.exists() else None

    record = {"fixtures": {}, "fa4_present": fa4 is not None,
              "kv_nrms_tolerance": kv_nrms_tolerance}
    for name, expected in reference["results"].items():
        produced = dense["results"][name]
        block_length = 32
        reference_blocks = block_states(expected["forwards"], block_length)
        dense_blocks = block_states(produced["forwards"], block_length)

        first_block_divergence = None
        for index, (left, right) in enumerate(zip(dense_blocks, reference_blocks)):
            if left != right:
                first_block_divergence = {"step": index, "candidate": left,
                                          "reference": right}
                break

        stats = produced["stats"]
        entry = {
            "tokens_match": produced["generated"][: len(expected["generated"])]
            == expected["generated"],
            "reference_generated": expected["generated"],
            "candidate_generated": produced["generated"],
            "block_state_steps_reference": len(reference_blocks),
            "block_state_steps_candidate": len(dense_blocks),
            "block_states_match": dense_blocks == reference_blocks,
            "first_block_divergence": first_block_divergence,
            "seeds_match": produced["seeds"] == expected["seeds"],
            "reference_seeds": expected["seeds"],
            "candidate_seeds": produced["seeds"],
            "reference_nfe": expected["nfe"],
            "candidate_stats": stats,
            # The reference counts denoise + commit calls, not the prefill.
            "nfe_match": stats["denoise_calls"] + stats["commit_calls"]
            == expected["nfe"],
            "commit_equals_blocks": stats["commit_calls"] == stats["blocks"],
        }
        kv_pairs = list(zip(produced["committed_kv"], expected["committed_kv"]))
        entry["committed_kv"] = [kv_metrics(a, b) for a, b in kv_pairs]
        entry["kv_within_tolerance"] = all(
            item["worst_nrms"] <= kv_nrms_tolerance for item in entry["committed_kv"]
        )
        if "selfspec" in expected and "selfspec" in produced:
            spec_expected = expected["selfspec"]
            spec_produced = produced["selfspec"]
            spec_stats = spec_produced["stats"]
            entry["selfspec"] = {
                "tokens_match": spec_produced["generated"][
                    : len(spec_expected["generated"])
                ] == spec_expected["generated"],
                "reference_nfe": spec_expected["nfe"],
                "candidate_calls": spec_stats["total_calls"],
                # The reference counts prefill, every draft and every verify.
                "nfe_match": spec_stats["total_calls"] == spec_expected["nfe"],
                "mean_acceptance": spec_stats["mean_acceptance"],
                "tokens_per_forward": spec_stats["tokens_per_forward"],
                "accepted_per_iteration": spec_stats["accepted_per_iteration"],
                "reference_generated": spec_expected["generated"],
                "candidate_generated": spec_produced["generated"],
            }
        for key in EXTRA_REFERENCE_KEYS:
            if key not in expected or key not in produced:
                continue
            entry[key] = compare_extra_pass(
                key, expected[key], produced[key], produced, block_length
            )
        if fa4 is not None:
            paged = fa4["results"][name]
            entry["fa4_tokens_match_dense"] = (
                paged["generated"] == produced["generated"]
            )
            entry["fa4_stats"] = paged["stats"]
            entry["fa4_stats_match_dense"] = (
                paged["stats"]["denoise_calls"] == stats["denoise_calls"]
                and paged["stats"]["commit_calls"] == stats["commit_calls"]
            )
            entry["fa4_causal_sequence"] = [
                (item["prefill"], item["causal"]) for item in paged["launches"]
            ]
        record["fixtures"][name] = entry

    checks = {
        "tokens_match_reference": all(
            item["tokens_match"] for item in record["fixtures"].values()
        ),
        "block_states_match_reference": all(
            item["block_states_match"] for item in record["fixtures"].values()
        ),
        "seeds_match_reference": all(
            item["seeds_match"] for item in record["fixtures"].values()
        ),
        "forward_accounting_matches": all(
            item["nfe_match"] and item["commit_equals_blocks"]
            for item in record["fixtures"].values()
        ),
        "committed_kv_within_tolerance": all(
            item["kv_within_tolerance"] for item in record["fixtures"].values()
        ),
    }
    spec_entries = [
        item["selfspec"] for item in record["fixtures"].values() if "selfspec" in item
    ]
    record["selfspec_present"] = bool(spec_entries)
    if spec_entries:
        checks["selfspec_matches_reference"] = all(
            item["tokens_match"] for item in spec_entries
        )
        checks["selfspec_forward_accounting_matches"] = all(
            item["nfe_match"] for item in spec_entries
        )
    if fa4 is not None:
        checks["fa4_matches_dense"] = all(
            item["fa4_tokens_match_dense"] and item["fa4_stats_match_dense"]
            for item in record["fixtures"].values()
        )
        checks["fa4_prefill_and_commit_are_causal"] = all(
            all(
                causal
                for prefill, causal in item["fa4_causal_sequence"]
                if prefill
            )
            for item in record["fixtures"].values()
        )
    # An optional pass that raised is recorded rather than lost, so that a
    # failed lane cannot look like a lane that was never asked for.
    errors = {}
    for key in EXTRA_REFERENCE_KEYS:
        for source, lane in (("reference", reference), ("dense", dense)):
            for name, item in lane["results"].items():
                if f"{key}_error" in item:
                    errors[f"{source}:{key}"] = item[f"{key}_error"]
    record["optional_pass_errors"] = errors
    if errors:
        checks["optional_passes_ran"] = False

    for key in EXTRA_REFERENCE_KEYS:
        items = [
            item[key] for item in record["fixtures"].values() if key in item
        ]
        record[f"{key}_present"] = bool(items)
        if not items:
            continue
        checks[f"{key}_matches_reference"] = all(
            item["tokens_match"] for item in items
        )
        checks[f"{key}_forward_accounting_matches"] = all(
            item["nfe_match"] for item in items
        )
        if key in ("thinking", "selfspec_thinking"):
            checks[f"{key}_marker_within_budget"] = all(
                item.get("marker_within_budget", False) for item in items
            )
        if key == "selfspec_lora":
            # The reference applies the adapter to every o_proj through PEFT and
            # FluxServe fuses one weight pair per layer, so the two counts have
            # to describe the same 40 layers.
            checks["lora_covers_every_layer"] = all(
                (item.get("adapter") or {}).get("layers") == 40
                and item.get("reference_adapted_modules", 0) >= 40
                for item in items
            )
            checks["lora_actually_changes_the_weights"] = all(
                (item.get("adapter") or {}).get("layer0_delta_nrms", 0.0) > 0.0
                for item in items
            )
    if require_flashinfer or (output_dir / "flashinfer.pt").exists():
        flashinfer = compare_flashinfer(output_dir, dense)
        checks.update(flashinfer["checks"])
        record["flashinfer_provenance"] = flashinfer["provenance"]
    record["paged_selfspec"] = {}
    for mode in SELFSPEC_PAGED_MODES:
        if not (output_dir / f"{mode}.pt").exists():
            continue
        paged_spec = compare_paged_selfspec(output_dir, dense, mode)
        checks.update(paged_spec["checks"])
        record[f"{mode}_provenance"] = paged_spec["provenance"]
        record["paged_selfspec"][mode] = {
            name: {
                key: {
                    "stats": value["stats"],
                    "adapter": value.get("adapter"),
                }
                for key, value in item.items()
            }
            for name, item in paged_spec["results"].items()
        }
    record["checks"] = checks
    record["passed"] = all(checks.values())
    record["reference_provenance"] = reference["provenance"]
    record["dense_provenance"] = dense["provenance"]
    if fa4 is not None:
        record["fa4_provenance"] = fa4["provenance"]
    return record


def render(record: dict) -> str:
    lines = [
        "# Nemotron-Labs-Diffusion Phase 2/3 — diffusion validation",
        "",
        f"**Result: {'PASS' if record['passed'] else 'FAIL'}**",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for key, value in record["checks"].items():
        lines.append(f"| {key} | {'pass' if value else '**FAIL**'} |")

    lines += [
        "",
        "## Per fixture",
        "",
        "| Fixture | tokens | block states | seeds | ref nfe | denoise+commit | "
        "worst KV nrms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, item in record["fixtures"].items():
        stats = item["candidate_stats"]
        worst = max(
            (entry["worst_nrms"] for entry in item["committed_kv"]), default=0.0
        )
        lines.append(
            f"| {name} | {item['tokens_match']} | {item['block_states_match']} "
            f"({item['block_state_steps_candidate']}/"
            f"{item['block_state_steps_reference']}) | {item['seeds_match']} | "
            f"{item['reference_nfe']} | "
            f"{stats['denoise_calls']}+{stats['commit_calls']} | {worst:.3e} |"
        )

    if record.get("selfspec_present"):
        lines += [
            "", "## Self-speculation vs reference", "",
            "Acceptance length is the number that decides whether this mode is "
            "worth anything: at one accepted token per iteration it is strictly "
            "worse than autoregressive decoding, having also paid for a draft.",
            "",
            "| Fixture | tokens | ref nfe | calls | mean acceptance | tokens/forward |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, item in record["fixtures"].items():
            spec = item.get("selfspec")
            if spec is None:
                continue
            lines.append(
                f"| {name} | {spec['tokens_match']} | {spec['reference_nfe']} | "
                f"{spec['candidate_calls']} | {spec['mean_acceptance']:.2f} | "
                f"{spec['tokens_per_forward']:.2f} |"
            )

    if record["fa4_present"]:
        lines += ["", "## Paged FA4 vs dense", "",
                  "| Fixture | tokens | denoise/commit |", "| --- | --- | --- |"]
        for name, item in record["fixtures"].items():
            paged = item["fa4_stats"]
            lines.append(
                f"| {name} | {item['fa4_tokens_match_dense']} | "
                f"{paged['denoise_calls']}+{paged['commit_calls']} |"
            )
    else:
        lines += ["", "_No FA4 artifact: the paged lane did not run._"]

    for name, item in record["fixtures"].items():
        if item["first_block_divergence"]:
            lines += ["", f"### First block divergence in {name}", "",
                      f"```\n{item['first_block_divergence']}\n```"]

    for key, title in (
        ("thinking", "Thinking budget — threshold decoding vs reference"),
        ("selfspec_thinking", "Thinking budget — self-speculation vs reference"),
        ("selfspec_lora", "Draft adapter (linear_spec_lora) vs reference"),
    ):
        if not record.get(f"{key}_present"):
            continue
        lines += ["", f"## {title}", "",
                  "| Fixture | tokens | ref nfe | calls | marker @ | extra |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, item in record["fixtures"].items():
            pass_item = item.get(key)
            if pass_item is None:
                continue
            marker = pass_item.get("marker_index_candidate")
            marker_cell = "n/a" if "marker" not in pass_item else (
                f"{marker} (<= {pass_item['budget']}+block)"
                if pass_item.get("marker_within_budget") else f"**{marker}**"
            )
            extra = ""
            if key == "selfspec_lora":
                adapter = pass_item.get("adapter") or {}
                extra = (
                    f"acc {pass_item.get('mean_acceptance', 0):.2f} vs "
                    f"{pass_item.get('acceptance_without_adapter', 0):.2f} "
                    f"without; {adapter.get('layers')} layers, "
                    f"delta nrms {adapter.get('layer0_delta_nrms', 0):.3e}"
                )
            elif "mean_acceptance" in pass_item:
                extra = f"acc {pass_item['mean_acceptance']:.2f}"
            lines.append(
                f"| {name} | {pass_item['tokens_match']} | "
                f"{pass_item['reference_nfe']} | {pass_item['candidate_calls']} | "
                f"{marker_cell} | {extra} |"
            )
            if not pass_item["tokens_match"]:
                lines.append(
                    f"| | first divergence at index "
                    f"{pass_item['first_divergence_index']} | | | | |"
                )

    for mode, fixtures in (record.get("paged_selfspec") or {}).items():
        lines += ["", f"## Paged self-speculation — {mode}", "",
                  "| Fixture | pass | calls | mean acceptance | accepted/iteration |",
                  "| --- | --- | --- | --- | --- |"]
        for name, item in fixtures.items():
            for key, value in item.items():
                stats = value["stats"]
                lines.append(
                    f"| {name} | {key} | {stats['total_calls']} | "
                    f"{stats['mean_acceptance']:.2f} | "
                    f"{stats['accepted_per_iteration']} |"
                )

    if record.get("optional_pass_errors"):
        lines += ["", "## Optional passes that failed", ""]
        for key, message in record["optional_pass_errors"].items():
            lines.append(f"- `{key}`: {message}")

    lines += ["", "## Provenance", ""]
    for key in ("reference_provenance", "dense_provenance", "fa4_provenance",
                "flashinfer_provenance", "selfspec_fa4_provenance",
                "selfspec_flashinfer_provenance"):
        if key in record:
            lines.append(f"- `{key}`: `{record[key]}`")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("reference", "dense", "fa4", "flashinfer",
                                 *SELFSPEC_PAGED_MODES, "compare"))
    parser.add_argument("--require-flashinfer", action="store_true",
                        help="Fail comparison if the FlashInfer artifact is absent")
    parser.add_argument("--lora", action="store_true",
                        help="also run self-speculation with linear_spec_lora")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="also run with this thinking-token budget")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kv-nrms-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        record = compare(output_dir, args.kv_nrms_tolerance,
                         require_flashinfer=args.require_flashinfer)
        (output_dir / "diffusion_metrics.json").write_text(
            json.dumps(
                {
                    key: value
                    for key, value in record.items()
                    if key != "fixtures"
                }
                | {
                    "fixtures": {
                        name: {
                            key: value
                            for key, value in item.items()
                            if key != "committed_kv"
                        }
                        | {"committed_kv": item["committed_kv"]}
                        for name, item in record["fixtures"].items()
                    }
                },
                indent=2,
            )
        )
        (output_dir / "diffusion_summary.md").write_text(render(record))
        print(json.dumps(record["checks"], indent=2), flush=True)
        print(f"PHASE23 {'PASS' if record['passed'] else 'FAIL'}", flush=True)
        return 0 if record["passed"] else 1

    fixtures = load_fixtures(args.fixtures)
    if args.mode in ("reference", "dense"):
        lane = run_reference if args.mode == "reference" else run_dense
        results = lane(fixtures, args.device, lora=args.lora,
                       thinking=args.thinking_budget)
    elif args.mode in SELFSPEC_PAGED_MODES:
        results = run_paged_selfspec(
            fixtures, args.device, args.mode.split("_", 1)[1], lora=args.lora
        )
    else:
        results = run_paged(fixtures, args.device, args.mode)
    torch.save(
        {"results": results, "provenance": provenance()},
        output_dir / f"{args.mode}.pt",
    )
    print(f"[{args.mode}] wrote {output_dir / (args.mode + '.pt')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
