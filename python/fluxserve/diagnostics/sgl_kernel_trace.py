"""Low-overhead, opt-in tracing of calls into :mod:`sgl_kernel`.

Set ``FLUXSERVE_SGL_KERNEL_TRACE=1`` before importing FluxServe. Each process
writes an independent JSONL stream so distributed workers never share a file.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib
import inspect
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

_phase = contextvars.ContextVar("sgl_kernel_trace_phase", default="startup")
_lock = threading.Lock()
_sequence = 0
_bootstrapped = False


def enabled() -> bool:
    return os.getenv("FLUXSERVE_SGL_KERNEL_TRACE", "").lower() in {"1", "true", "yes"}


@contextlib.contextmanager
def phase(name: str):
    token = _phase.set(name)
    try:
        yield
    finally:
        _phase.reset(token)


def trace_phase(name: str):
    """Decorate a sync or async entry point with a trace phase."""

    def decorate(func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with phase(name):
                    return await func(*args, **kwargs)
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with phase(name):
                return func(*args, **kwargs)
        return wrapper
    return decorate


def _tensor_metadata(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return {"type": type(value).__name__}
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return {
                "type": "tensor", "shape": list(value.shape),
                "dtype": str(value.dtype), "device": str(value.device),
            }
    except ImportError:
        pass
    if isinstance(value, (list, tuple)):
        return [_tensor_metadata(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): _tensor_metadata(item, depth + 1) for key, item in value.items()}
    return {"type": type(value).__name__}


def _is_capturing() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except (ImportError, RuntimeError, AttributeError):
        return False


def _write(event: dict[str, Any]) -> None:
    global _sequence
    output_dir = Path(os.getenv("FLUXSERVE_SGL_KERNEL_TRACE_DIR", ".sgl-kernel-audit/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "unknown"))
    path = output_dir / f"trace-rank{rank}-pid{os.getpid()}.jsonl"
    with _lock:
        _sequence += 1
        event.update(sequence=_sequence, timestamp_ns=time.time_ns(), pid=os.getpid(), rank=rank)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")


def record_metadata(**metadata: Any) -> None:
    if enabled():
        _write({"kind": "metadata", "phase": _phase.get(), "metadata": metadata})


def _wrap(module: Any, name: str, qualified_name: str) -> None:
    original = getattr(module, name, None)
    if (
        not callable(original)
        or inspect.isclass(original)
        or getattr(original, "__fluxserve_sgl_trace__", False)
    ):
        return

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        capture = _is_capturing()
        event = {
            "kind": "call", "symbol": qualified_name,
            "phase": "graph_capture" if capture else _phase.get(),
            "context_phase": _phase.get(), "cuda_graph_capture": capture,
            "inputs": _tensor_metadata(args), "kwargs": _tensor_metadata(kwargs),
        }
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            event["exception"] = type(exc).__name__
            _write(event)
            raise
        event["outputs"] = _tensor_metadata(result)
        _write(event)
        return result

    wrapper.__fluxserve_sgl_trace__ = True
    setattr(module, name, wrapper)


def bootstrap() -> None:
    """Wrap sgl-kernel exports before FluxServe imports aliases from them."""
    global _bootstrapped
    if _bootstrapped or not enabled():
        return
    _bootstrapped = True
    try:
        root = importlib.import_module("sgl_kernel")
    except ImportError:
        return
    modules = [(root, "sgl_kernel")]
    for submodule in ("gemm",):
        try:
            modules.append((importlib.import_module(f"sgl_kernel.{submodule}"), f"sgl_kernel.{submodule}"))
        except ImportError:
            pass
    for module, prefix in modules:
        for name in dir(module):
            if not name.startswith("_"):
                _wrap(module, name, f"{prefix}.{name}")
    record_metadata(event="bootstrap", sgl_kernel_file=getattr(root, "__file__", None))
