"""FP16 size sweep; independent process passes should use distinct --pass-id."""
import argparse
import json
from pathlib import Path

from profile_fa4_llada import _parser, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pass-id", type=int, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--no-mixed", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [(phase, batch, length) for phase in ("prefill", "decode")
             for batch in (1, 4, 8, 16) for length in args.lengths]
    # Also remeasure the exact original heterogeneous workloads.
    if not args.no_mixed:
        cases += [(phase, None, None) for phase in ("prefill", "decode")]
    if args.pass_id % 2:
        cases.reverse()
    for phase, batch, length in cases:
        argv = ["--phase", phase, "--execution", "both", "--dtype", "float16",
                "--atol", ".005", "--rtol", ".005", "--warmup", "100",
                "--warmup-seconds", "0.5", "--iterations", str(args.iterations), "--repeats", "12"]
        if batch is not None:
            argv += ["--batch-size", str(batch), "--kv-length", str(length)]
        result = run(_parser().parse_args(argv))
        result["pass_id"] = args.pass_id
        name = f"{phase}_b{batch}_kv{length}" if batch else f"{phase}_mixed"
        path = args.output_dir / f"{name}_pass{args.pass_id}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(name, args.pass_id, {k: round(v["median_ms"] * 1000, 2)
                                  for k, v in result["timing"].items()}, flush=True)


if __name__ == "__main__":
    main()
