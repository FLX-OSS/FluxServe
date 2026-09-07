# LLaDA2.x attention: FP16 eager vs CUDA Graph

## Scope and method

GH200 120GB (SM90), Slurm benchmark/profiler job 3089394; additional
node-level NSYS capture job 3089415. Container: `flux.sif`, PyTorch 2.8/CUDA
12.9, flash-attn-4 4.0.0b29, custom FlashInfer BatchPrefillBlockExtend.

This measures the attention API, **not full FluxServe adapter, model or server
latency**. It does not enable the online runner's CUDA Graph path.
Both backends use the same FP16 Q, physical paged K/V storage and shuffled
page IDs. Hq/Hkv/D = 16/4/128; block length and page size = 64.
Prefill lengths: 128, 256, 512, 1024. Decode: 16 requests, 64 queries each;
KV lengths are recorded in the JSON (256 through 8192).

These two phase workloads are not matched in batch size or context length.
Their valid Q–K pairs per head are 757,760 (prefill) and 3,145,728
(decode), respectively: decode has 4.15x as many pairs despite fewer Q tokens.
Here decode means a 64-token diffusion block, not single-token AR decode.
Consequently this table cannot establish that decode is intrinsically slower
than prefill. It compares backends/execution modes within each workload only.

One attention call is captured per replay. Input allocation/population,
metadata planning, JIT, capture and input updates are excluded from timing.
All four combinations run in the same process with rotated measurement order.
Warmup = 100; each of 20 rounds measures 200 calls with CUDA events. These
are average stream elapsed times per call, including GPU idle gaps caused by
host submission; they are not pure kernel time or per-request latency percentiles.
Profilers run separately from benchmark timing.

## Correctness

Before timing, both backends pass all-query FP32 math SDPA checks with
atol=rtol=0.005. Maximum absolute errors are 0.00047815 (prefill) and
0.00021696 (decode). Graph also passes the reference check. Each backend's
graph output is bitwise equal to its eager output for two changed Q inputs
(0.75 and -0.5 times original Q), checking that replay reads updated storage.
This benchmark does not establish full-model/logit parity or arbitrary dynamic
shape/cache-update correctness.

## Results

Times in microseconds; speedup = eager mean / graph mean.

| Phase | Backend | Eager mean | Graph mean | Speedup | Eager median | Graph median |
|---|---|---:|---:|---:|---:|---:|
| Prefill | FA4 | 61.11 | 28.35 | 2.155x | 56.19 | 28.40 |
| Prefill | BatchPrefill | 43.86 | 43.23 | 1.015x | 43.70 | 43.29 |
| Decode | FA4 | 121.83 | 121.06 | 1.006x | 122.55 | 121.34 |
| Decode | BatchPrefill | 192.82 | 193.03 | 0.999x | 192.63 | 192.92 |

FA4 prefill eager has outliers (round-average range 50.29–116.41 us,
standard deviation 15.07 us). Its median-based graph speedup is 1.98x;
the mean-based 2.155x should not be treated as a stable universal gain.
With both backends graphed, FA4 is 1.525x faster for prefill and 1.595x
for decode on these workloads. Decode graph differences and BatchPrefill
graph gains are small and are not evidence of a meaningful improvement.

The prefill result supports substantial avoidable host submission overhead
in this FA4 API path. It does not identify every saved microsecond as CUDA
API execution, or imply graph accelerates the attention kernel itself.
Do not add CUDA API and GPU kernel durations: they overlap, and synchronization
API duration can represent waiting for already submitted GPU work.

The original decode FA4 NSYS kernel means are 114.27 us (eager) and
114.58 us (graph, node-level trace), compared with roughly 121–122 us
in the separate unprofiled benchmark. This supports GPU execution dominating
that workload; graph does not remove its QK/softmax/AV work. CUDA events around
a loop measure queued stream elapsed time, not synchronized single-call wall
latency. CPU submission can overlap a long kernel, so reducing host overhead
need not reduce this metric. These separate traces cannot be subtracted to
calculate an exact CPU overhead budget.

## Artifacts and reproduction

All files are under `test/runtime/integration/profiles/fa4_llada_fp16_graph/`:

- `prefill_benchmark.json`, `decode_benchmark.json`: all timing samples and checks.
- `{prefill,decode}_{fa4,batchprefill}_{eager,graph}_nsys.nsys-rep` and `.sqlite`:
  8 initial traces, 20 profiled calls each. Initial graph traces are graph-level.
- `{prefill,decode}_{fa4,batchprefill}_graph_nodes_nsys.nsys-rep` and `.sqlite`:
  4 additional node-level graph traces; prefer these to inspect internal kernels.
- `{prefill,decode}_{fa4,batchprefill}_{eager,graph}_ncu.ncu-rep`:
  8 reports, one attention call each, SpeedOfLight, MemoryWorkloadAnalysis,
  LaunchStats and Occupancy sections. NCU replay/clock/cache conditions can
  change measured kernel duration; use benchmark JSON for speed comparisons.

Inside the GPU environment with the standalone FA4 dependencies installed:

```bash
python test/runtime/integration/profile_fa4_llada.py \
  --phase prefill --execution both --dtype float16 --atol .005 --rtol .005 \
  --warmup 100 --iterations 200 --repeats 20 --output prefill_graph.json

PHASE=prefill BACKEND=fa4 REPORT=/tmp/prefill_fa4_graph_nsys \
  bash test/runtime/integration/profile_fa4_llada_nsys.sh \
  --execution graph --dtype float16 --atol .005 --rtol .005

PHASE=prefill BACKEND=fa4 REPORT=/tmp/prefill_fa4_graph_ncu \
  bash test/runtime/integration/profile_fa4_llada_ncu.sh \
  --execution graph --dtype float16 --atol .005 --rtol .005
```

Repeat with phase `decode`, backend `batchprefill` and execution `eager`.
Set distinct `REPORT` paths to preserve previous captures. The NSYS script
now requests `--cuda-graph-trace=node` for internal kernel visibility.
