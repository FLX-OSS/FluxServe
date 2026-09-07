"""Render measured JSON as SVG line plots and a Markdown table (stdlib only)."""
import argparse
import json
import math
import statistics
from pathlib import Path

KEYS = ("fa4_eager", "fa4_graph", "batchprefill_eager", "batchprefill_graph")
COLORS = ("#1664c0", "#1664c0", "#c65d12", "#c65d12")
LENGTHS = (256, 512, 1024, 2048, 4096)
BATCHES = (1, 4, 8, 16)


def main():
    global LENGTHS
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--no-mixed", action="store_true")
    args = parser.parse_args()
    LENGTHS = tuple(sorted(set(args.lengths)))
    assert len(LENGTHS) >= 2 and all(v > 0 for v in LENGTHS)
    records = {}
    for path in args.input.glob("*.json"):
        record = json.loads(path.read_text())
        assert record["dtype"] == "float16"
        assert record["correctness_gate"]["passed"] and record["graph_parity"]
        name = path.stem.rsplit("_pass", 1)[0]
        records.setdefault(name, {})[record["pass_id"]] = record
    expected = [f"{p}_b{b}_kv{l}" for p in ("prefill", "decode")
                for b in BATCHES for l in LENGTHS]
    if not args.no_mixed:
        expected += ["prefill_mixed", "decode_mixed"]
    assert set(records) == set(expected), "Incomplete or unexpected cases"
    assert all(set(v) == {0, 1} for v in records.values()), "Need both independent passes"
    args.output.mkdir(parents=True, exist_ok=True)

    def values(name, key):
        return [r["timing"][key]["median_ms"] * 1000 for r in records[name].values()]

    for phase in ("prefill", "decode"):
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="790" viewBox="0 0 1200 790">',
               '<rect width="1200" height="790" fill="white"/>',
               '<g font-family="sans-serif" fill="#222">',
               f'<text x="600" y="28" text-anchor="middle" font-size="22">FP16 GH200 — {phase}</text>',
               '<text x="600" y="51" text-anchor="middle" font-size="13">Mean of two process medians; whiskers: min/max process median. Log axes; attention API only.</text>']
        for ki, key in enumerate(KEYS):
            x = 115 + ki * 275
            dash = ' stroke-dasharray="6 4"' if key.endswith("graph") else ""
            svg += [f'<line x1="{x}" y1="78" x2="{x+32}" y2="78" stroke="{COLORS[ki]}" stroke-width="2"{dash}/>',
                    f'<text x="{x+40}" y="82" font-size="13">{key}</text>']
        for i, batch in enumerate(BATCHES):
            left = 85 + (i % 2) * 595
            top = 135 + (i // 2) * 325
            width, height = 470, 225
            all_values = [v for length in LENGTHS for key in KEYS
                          for v in values(f"{phase}_b{batch}_kv{length}", key)]
            lo = math.floor(math.log10(min(all_values)) * 2) / 2
            hi = math.ceil(math.log10(max(all_values)) * 2) / 2
            if hi == lo:
                hi += .5

            def ypos(v):
                return top + height * (hi - math.log10(v)) / (hi - lo)

            svg.append(f'<text x="{left+width/2}" y="{top-15}" text-anchor="middle" font-size="17">Batch = {batch}</text>')
            for tick in range(int(lo * 2), int(hi * 2) + 1):
                v = 10 ** (tick / 2)
                y = ypos(v)
                svg += [f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" stroke="#ddd"/>',
                        f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-size="12">{v:.0f}</text>']
            svg.append(f'<text x="{left-58}" y="{top+height/2}" font-size="12" transform="rotate(-90 {left-58} {top+height/2})">Latency (us, log)</text>')
            for j, length in enumerate(LENGTHS):
                x = left + width * math.log(length / LENGTHS[0]) / math.log(LENGTHS[-1] / LENGTHS[0])
                svg.append(f'<text x="{x}" y="{top+height+20}" text-anchor="middle" font-size="12">{length}</text>')
            svg.append(f'<text x="{left+width/2}" y="{top+height+43}" text-anchor="middle" font-size="13">Total KV length (log2)</text>')
            for ki, key in enumerate(KEYS):
                points = []
                for j, length in enumerate(LENGTHS):
                    vv = values(f"{phase}_b{batch}_kv{length}", key)
                    x = left + width * math.log(length / LENGTHS[0]) / math.log(LENGTHS[-1] / LENGTHS[0])
                    y = ypos(statistics.mean(vv))
                    points.append(f"{x},{y}")
                    svg += [f'<line x1="{x}" x2="{x}" y1="{ypos(min(vv))}" y2="{ypos(max(vv))}" stroke="{COLORS[ki]}" stroke-width="2"/>',
                            f'<circle cx="{x}" cy="{y}" r="3" fill="{COLORS[ki]}"/>']
                dash = ' stroke-dasharray="6 4"' if key.endswith("graph") else ""
                svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[ki]}" stroke-width="2"{dash}/>')
        svg.append('</g></svg>')
        (args.output / f"fa4_fp16_{phase}.svg").write_text("\n".join(svg))

    rows = ["# Expanded FP16 measured results", "", f"Generated from {2 * len(expected)} JSON records: {len(expected)} cases × 2 independent process passes.",
            "Values: mean of the two process medians, in microseconds. Each process median is over 12 round averages, not individual-call latencies.", "",
            "![Prefill](fa4_fp16_prefill.svg)", "", "![Decode](fa4_fp16_decode.svg)", "",
            "| Case | FA4 eager | FA4 graph | BatchPrefill eager | BatchPrefill graph |",
            "|---|---:|---:|---:|---:|"]
    for name in expected:
        rows.append("| " + name + " | " + " | ".join(f"{statistics.mean(values(name, k)):.2f}" for k in KEYS) + " |")
    max_error = max(r["correctness_gate"][key]["max_abs_error"]
                    for runs in records.values() for r in runs.values()
                    for key in ("fa4_vs_fp32_sdpa", "batchprefill_vs_fp32_sdpa"))
    rows += ["", f"All correctness gates passed; maximum absolute FP32 SDPA error: {max_error:.9f}.", "",
             "Whiskers are the range of two process medians, NOT confidence intervals. Raw per-round variability remains in JSON.", ""]
    (args.output / "fa4_fp16_expanded_results.md").write_text("\n".join(rows))


if __name__ == "__main__":
    main()
