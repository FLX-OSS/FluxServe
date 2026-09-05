from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys


def summarize_trace(path: str | Path) -> dict:
    events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen_admit: set[str] = set()
    planned: set[str] = set()
    terminal: dict[str, str] = {}
    errors: list[str] = []
    skipped_decode = 0
    batch_sizes: list[int] = []
    for event in events:
        kind = event.get("event")
        ids = [str(rid) for rid in event.get("request_ids", [])]
        if kind == "admission":
            seen_admit.update(ids)
        elif kind == "plan":
            selected = set(event.get("selected_request_ids", ids))
            active = set(event.get("active_request_ids", ()))
            if active and not selected.issubset(active):
                errors.append("plan-selected-not-active")
            skipped_decode += len(event.get("skipped_decode_request_ids", ()))
            if "batch_size" in event:
                batch_sizes.append(int(event["batch_size"]))
            for rid in ids:
                if rid in terminal:
                    errors.append(f"plan-after-terminal:{rid}")
                planned.add(rid)
        elif kind in ("finish", "abort"):
            for rid in ids:
                if rid not in seen_admit:
                    errors.append(f"terminal-before-admission:{rid}")
                if rid in terminal:
                    errors.append(f"duplicate-terminal:{rid}")
                terminal[rid] = kind
    errors.extend(f"admitted-never-planned:{rid}" for rid in sorted(seen_admit - planned))
    return {"events": len(events), "admitted": sorted(seen_admit),
            "planned": sorted(planned), "terminal": terminal, "errors": errors,
            "skipped_decode_requests": skipped_decode,
            "batch_sizes": batch_sizes}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m fluxserve.backend.engine.trace_parser TRACE.jsonl")
    summary = summarize_trace(sys.argv[1])
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(1 if summary["errors"] else 0)
