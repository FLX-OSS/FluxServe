# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
    Scheduler and decode-block metrics and tracing.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fluxserve.backend.metrics.perf import (
    DecodeBlockMetric,
    summarize_decode_block_metrics,
)

logger = logging.getLogger(__name__)


class SchedulerTrace:
    """Best-effort, opt-in JSONL scheduler trace."""

    schema_version = 1

    def __init__(self, path: str | None, *, metadata: dict[str, Any] | None = None):
        self.path = path
        self._fh = None
        self._seq = 0
        self._start_ns = time.monotonic_ns()
        self._disabled = not path
        if path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                self._fh = open(path, "a", encoding="utf-8")
                self.emit("run", request_ids=[], metadata=metadata or {})
            except OSError:
                logger.exception("Unable to open scheduler trace %s", path)
                self._disabled = True

    @property
    def enabled(self) -> bool:
        return self._fh is not None and not self._disabled

    def emit(self, event: str, *, request_ids=(), **fields: Any) -> None:
        if not self.enabled:
            return
        record = {"schema_version": self.schema_version, "sequence": self._seq,
                  "event": event, "timestamp_ns": time.time_ns(),
                  "elapsed_ns": time.monotonic_ns() - self._start_ns,
                  "request_ids": sorted(str(x) for x in request_ids), **fields}
        self._seq += 1
        try:
            self._fh.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
            self._fh.flush()
        except (OSError, ValueError):
            logger.exception("Disabling scheduler trace after write failure")
            self._disabled = True

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def summarize_trace(path: str | Path) -> dict:
    events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen_admit, planned, terminal, errors = set(), set(), {}, []
    skipped_decode, batch_sizes = 0, []
    for event in events:
        kind = event.get("event")
        ids = [str(rid) for rid in event.get("request_ids", [])]
        if kind == "admission":
            seen_admit.update(ids)
        elif kind == "plan":
            selected = set(event.get("selected_request_ids", ids))
            active = set(event.get("active_request_ids", ()))
            if active and not selected.issubset(active): errors.append("plan-selected-not-active")
            skipped_decode += len(event.get("skipped_decode_request_ids", ()))
            if "batch_size" in event: batch_sizes.append(int(event["batch_size"]))
            for rid in ids:
                if rid in terminal: errors.append(f"plan-after-terminal:{rid}")
            planned.update(ids)
        elif kind in ("finish", "abort"):
            for rid in ids:
                if rid not in seen_admit: errors.append(f"terminal-before-admission:{rid}")
                if rid in terminal: errors.append(f"duplicate-terminal:{rid}")
                terminal[rid] = kind
    errors.extend(f"admitted-never-planned:{rid}" for rid in sorted(seen_admit - planned))
    return {"events": len(events), "admitted": sorted(seen_admit), "planned": sorted(planned),
            "terminal": terminal, "errors": errors, "skipped_decode_requests": skipped_decode,
            "batch_sizes": batch_sizes}

__all__ = [
    "DecodeBlockMetric",
    "SchedulerTrace",
    "summarize_decode_block_metrics",
    "summarize_trace",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m fluxserve.backend.metrics.trace TRACE.jsonl")
    print(json.dumps(summarize_trace(sys.argv[1]), indent=2, sort_keys=True))
