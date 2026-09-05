import json

from fluxserve.backend.engine.scheduler_trace import SchedulerTrace
from fluxserve.backend.engine.trace_parser import summarize_trace


def test_trace_writes_jsonl_and_flushes(tmp_path):
    path = tmp_path / "nested" / "trace.jsonl"
    trace = SchedulerTrace(str(path), metadata={"policy": "dynamic"})
    trace.emit("plan", request_ids=["b", "a"])
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["event"] == "run"
    assert rows[1]["request_ids"] == ["a", "b"]
    assert rows[1]["sequence"] > rows[0]["sequence"]
    trace.close()


def test_disabled_trace_does_not_create_file(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = SchedulerTrace(None)
    trace.emit("plan", request_ids=["a"])
    assert not path.exists()


def test_trace_parser_detects_lifecycle_errors(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"event": "admission", "request_ids": ["a"]},
        {"event": "finish", "request_ids": ["b"]},
        {"event": "plan", "request_ids": ["a"]},
        {"event": "finish", "request_ids": ["a"]},
        {"event": "plan", "request_ids": ["a"]},
    ]))
    summary = summarize_trace(path)
    assert any("terminal-before-admission:b" in error for error in summary["errors"])
    assert any("plan-after-terminal:a" in error for error in summary["errors"])

