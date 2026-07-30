# sgl-kernel microbenchmarks

These standalone scripts measure the four `sgl_kernel` calls observed on the
LLaDA2 serving critical path. Defaults reproduce the shapes captured from a
23-token TP4/EP4 request.

Run all benchmarks inside `flux_workspace`:

```bash
python test/sgl_kernel/run_all.py --output-dir .sgl-kernel-benchmarks
```

Run one kernel or sweep token count:

```bash
python test/sgl_kernel/benchmark_moe_fused_gate.py --tokens 64
python test/sgl_kernel/benchmark_rope.py --tokens 64 --json-output rope-64.json
```

Timing uses CUDA events. Each reported sample is the average latency of one
kernel call over an iteration round; the JSON reports min, median, mean, p90,
and max across rounds. Setup/allocation and correctness checks are outside the
timed region. These are isolated kernel latencies, not end-to-end serving time.
