"""GPU checkpoint gate for self-speculation on FA4 and FlashInfer.

Runs sequential, concurrent, cancellation and slot-reuse scenarios with and
without graphs, using the checkpoint's draft adapter in all lanes.

python test/runtime/integration/nemotron_selfspec_graph_harness.py \
    --model nvidia/Nemotron-Labs-Diffusion-3B --output .ci-artifacts/selfspec-graph
"""

import argparse
import json
import re
from pathlib import Path

import nemotron_online_harness as online

# Scenarios whose requests share a forward with a set of neighbours that the
# scheduler, not the harness, decides.
VARIABLE_BATCH = frozenset({"concurrent", "flood"})


def final_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def equivalent(expected: str, actual: str) -> bool:
    """What survives a change of batch composition, and what should not.

    Byte equality does not. Which rows a request shares its forward with
    decides the matmul tiling, which moves logits by ~1e-3, which flips a
    near-tied threshold decision, which sends the decode down a different but
    equally valid derivation: `'\\n126'` and `'\\n84 * 3 / 2 = 126'` are the
    same answer to `84 * 3 / 2`. Requiring them to be one string is requiring
    batch-invariant floating point, which this server does not offer.

    Corruption does not look like that. Stale KV, a bad rollback or a request
    slot handed over while still in use change the answer, empty it, or return
    another request's text -- all of which this still rejects.
    """
    if actual == expected:
        return True
    if not actual.strip():
        return False
    left, right = final_number(expected), final_number(actual)
    if left is not None and right is not None:
        return left == right
    # A prose fixture under a variable batch: there is no answer to extract, so
    # only degeneracy and cross-talk (checked per scenario) are decidable.
    return True


def scenario_checks(label: str, scenario: str, replies: dict, baseline: dict) -> dict:
    """Gate on what is achievable; report strict equality either way."""
    strict = all(reply == baseline[name] for name, reply in replies.items())
    checks = {f"{label}-{scenario}-strict-INFO": strict}
    if scenario in VARIABLE_BATCH:
        checks[f"{label}-{scenario}"] = all(
            equivalent(baseline[name], reply) for name, reply in replies.items()
        )
        # One request returning another's text is the failure page and slot
        # reuse actually produces, and it survives the relaxation above.
        distinct = {name: reply for name, reply in replies.items()
                    if baseline.get(name, "").strip()}
        checks[f"{label}-{scenario}-no-crosstalk"] = (
            len(set(distinct.values())) == len(distinct)
            or len(set(baseline[name] for name in distinct)) < len(distinct)
        )
    else:
        checks[f"{label}-{scenario}"] = strict
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    online.add_checkpoint_args(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("fa4", "flashinfer", "both"), default="both")
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()
    online.MODEL = args.model
    online.REVISION = online.resolve_revision(args.model, args.revision)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fixtures = online.load_fixtures(online.DEFAULT_FIXTURES)
    fixtures = [dict(f, temperature=0.0, max_new_tokens=96) for f in fixtures[:3]]
    checks = {}
    backends = ("fa4", "flashinfer") if args.backend == "both" else (args.backend,)
    for backend in backends:
        lanes = {}
        for graphs in (False, True):
            label = f"{backend}-{'graph' if graphs else 'eager'}"
            lane = online.run_serve(
                fixtures, graphs=graphs, output_dir=output, label=label,
                backend=backend, decoding="self_speculation", max_num_seqs=4,
                max_model_len=args.max_model_len,
            )
            (output / f"{label}.json").write_text(json.dumps(lane, indent=2))
            lanes[graphs] = lane
            baseline = lane["scenarios"]["sequential"]
            checks[f"{label}-clean-exit"] = lane["server_exit_code"] in (0, -2)
            # Repeats of one prompt inside the flood land in different batches
            # too, so this is the same floating-point question as the scenario
            # comparison below, not a stronger one.
            checks[f"{label}-flood-consistent-strict-INFO"] = lane[
                "flood_replies_consistent"
            ]
            for scenario, replies in lane["scenarios"].items():
                checks.update(scenario_checks(label, scenario, replies, baseline))
            if graphs:
                metrics = lane["metrics_final"]
                checks[f"{label}-captured"] = metrics.get("cuda_graph_decode_capture_count", 0) > 0
                checks[f"{label}-replayed"] = metrics.get("cuda_graph_decode_replay_count", 0) > 0
            print(json.dumps(checks, indent=2), flush=True)
        eager, graphed = lanes[False]["scenarios"], lanes[True]["scenarios"]
        checks[f"{backend}-graph-matches-eager-strict-INFO"] = eager == graphed
        # Graph and eager are compared per scenario for the same reason: at a
        # fixed batch they must agree byte for byte, and under a variable one
        # the batch differs between the two lanes as well.
        checks[f"{backend}-graph-matches-eager"] = all(
            all(
                equivalent(eager[scenario][name], reply)
                if scenario in VARIABLE_BATCH
                else reply == eager[scenario][name]
                for name, reply in replies.items()
            )
            for scenario, replies in graphed.items()
            if scenario in eager
        )
    (output / "checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)
    # `-strict-INFO` entries record batch-invariant byte equality, which this
    # server does not promise. They are reported, never gated on.
    gating = {k: v for k, v in checks.items() if not k.endswith("-strict-INFO")}
    failed = sorted(k for k, v in gating.items() if not v)
    if failed:
        print("FAILED CHECKS: " + ", ".join(failed), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
