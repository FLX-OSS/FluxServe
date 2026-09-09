"""Reject missing or failed perf runs even if EvalScope exited successfully."""
import json
from pathlib import Path
import sys

reports = list(Path(sys.argv[1]).rglob("benchmark_summary.json"))
if len(reports) != 1:
    raise SystemExit(f"Expected one perf summary, found {len(reports)}")
summary = json.loads(reports[0].read_text())
if (summary.get("Total Requests") != 1000
        or summary.get("Success Requests") != 1000
        or summary.get("Failed Requests") != 0):
    raise SystemExit(f"Incomplete/failed benchmark: {summary}")
print(json.dumps(summary, indent=2))
