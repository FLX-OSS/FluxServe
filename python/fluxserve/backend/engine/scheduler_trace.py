from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

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
        now = time.time_ns()
        record = {
            "schema_version": self.schema_version,
            "sequence": self._seq,
            "event": event,
            "timestamp_ns": now,
            "elapsed_ns": time.monotonic_ns() - self._start_ns,
            "request_ids": sorted(str(x) for x in request_ids),
            **fields,
        }
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

