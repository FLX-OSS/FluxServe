import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))
from fluxserve.diagnostics import sgl_kernel_trace as trace


def test_trace_phase_restores_context():
    @trace.trace_phase("request_eager")
    async def current_phase():
        return trace._phase.get()

    assert asyncio.run(current_phase()) == "request_eager"
    assert trace._phase.get() == "startup"


def test_tensor_metadata_is_value_free():
    torch = pytest.importorskip("torch")
    metadata = trace._tensor_metadata({"x": torch.tensor([[123.0]])})
    assert metadata["x"] == {"type": "tensor", "shape": [1, 1], "dtype": "torch.float32", "device": "cpu"}
    assert "123" not in str(metadata)


def test_wrapper_preserves_result_and_exception(monkeypatch):
    events = []
    monkeypatch.setattr(trace, "_write", events.append)
    module = types.SimpleNamespace(add=lambda value: value + 1)
    trace._wrap(module, "add", "sgl_kernel.add")
    assert module.add(2) == 3
    assert events[-1]["symbol"] == "sgl_kernel.add"

    def fail():
        raise LookupError("expected")

    module.fail = fail
    trace._wrap(module, "fail", "sgl_kernel.fail")
    with pytest.raises(LookupError, match="expected"):
        module.fail()
    assert events[-1]["exception"] == "LookupError"
