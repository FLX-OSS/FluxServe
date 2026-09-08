"""Benchmark probes; timed runs have no per-forward hooks or model changes."""
import os
import json
from pathlib import Path

from fluxserve.backend.engine.async_llm import AsyncLLM
from fluxserve.cli import main


original_snapshot = AsyncLLM.get_metrics_snapshot
profile_reads = 0
measuring = False
decode_profiled = False


def profile_first_decode(method):
    def wrapped(*args, **kwargs):
        global decode_profiled
        if not measuring or decode_profiled:
            return method(*args, **kwargs)
        import torch
        decode_profiled = True
        torch.cuda.cudart().cudaProfilerStart()
        try:
            return method(*args, **kwargs)
        finally:
            torch.cuda.cudart().cudaProfilerStop()
    return wrapped


def snapshot_with_forwards(self):
    global profile_reads, measuring
    snapshot = original_snapshot(self)
    executor = getattr(self.executor, "base_executor", self.executor)
    runner = getattr(executor, "runner", None)
    if runner is not None:
        snapshot["benchmark_model_forwards"] = int(runner.num_forwards)
        import torch
        snapshot["benchmark_gpu_memory_allocated_bytes"] = torch.cuda.memory_allocated()
        snapshot["benchmark_gpu_memory_reserved_bytes"] = torch.cuda.memory_reserved()
        snapshot["benchmark_gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    profile_mode = os.getenv("FLUXSERVE_BENCH_PROFILE_API")
    if profile_mode:
        measuring = profile_reads == 0
    if profile_mode == "nsys":
        import torch
        if profile_reads == 0:
            torch.cuda.cudart().cudaProfilerStart()
        elif profile_reads == 1:
            torch.cuda.cudart().cudaProfilerStop()
    profile_reads += 1
    return snapshot


def install_probes():
    AsyncLLM.get_metrics_snapshot = snapshot_with_forwards
    if os.getenv("FLUXSERVE_BENCH_PROFILE_API") == "ncu":
        # Diagnostic only: capture one decode attention kernel, not the first
        # prefill kernel of an HTTP request. Timed benchmark runs have no hooks.
        from fluxserve.backend.execution.runners.fa4_diffusion import FA4DiffusionRunner
        from fluxserve.backend.execution.runners.flashinfer_diffusion import FlashInferDiffusionRunner
        for runner_class in (FA4DiffusionRunner, FlashInferDiffusionRunner):
            runner_class._decode_selected_batch = profile_first_decode(runner_class._decode_selected_batch)

if __name__ == "__main__":
    if os.getenv("FLUXSERVE_BENCH_PID_FILE"):
        start_time = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        Path(os.environ["FLUXSERVE_BENCH_PID_FILE"]).write_text(json.dumps({
            "pid": os.getpid(), "start_time": start_time,
        }))
    install_probes()
    main()
