# FP16 FA4 vs BatchPrefillBlockExtend on GH200

Slurm job: 3086895. NVIDIA GH200 120GB, SM90. Standalone FA4 4.0.0b29;
the existing container FlashInfer block-extend path. Geometry: Q heads 16,
KV heads 4, head dimension 128, block/page size 64. Seed 0.

These are eager attention API timings, not full-adapter or CUDA Graph timings.
Both backends share Q/K/V storage and physical page IDs. Planning, JIT, input
allocation and KV population are outside timing. API output initialization and
host submission gaps remain inside timing. Warmup: 100 calls; measurements:
20 alternating-order rounds, 200 calls per backend per round.

| Phase | FA4 mean | BatchPrefill mean | FA4 result |
|---|---:|---:|---|
| Prefill, lengths 128/256/512/1024 | 55.0444 us | 42.8742 us | 28.39% slower |
| Decode, batch 16, Q=64, KV=256–8192 | 121.1059 us | 192.2896 us | 1.5878x speedup, 37.02% latency reduction |

All queries were checked against FP32 math SDPA before timing and profiling,
using atol=rtol=0.005. Max absolute errors:

| Phase | FA4 vs SDPA | BatchPrefill vs SDPA | FA4 vs BatchPrefill |
|---|---:|---:|---:|
| Prefill | 0.0004781485 | 0.0004781485 | 0.0004882813 |
| Decode | 0.0002169609 | 0.0002169609 | 0.0002441406 |

Artifacts are under `test/runtime/integration/profiles/fa4_llada_fp16/`:
`prefill_benchmark.json`, `decode_benchmark.json`, and reports named
`{prefill,decode}_{fa4,batchprefill}_nsys.nsys-rep` (with SQLite exports) and
`{prefill,decode}_{fa4,batchprefill}_ncu.ncu-rep`.

NSYS captures 20 calls after 50 warmup calls. NCU captures one call after 50
warmup calls, with SpeedOfLight, MemoryWorkloadAnalysis, LaunchStats, and
Occupancy sections. Use the uninstrumented benchmark for latency comparisons.

Reproduce inside the GPU environment:

```bash
export PYTHONNOUSERSITE=1 CC=/usr/bin/gcc CXX=/usr/bin/g++
python test/runtime/integration/profile_fa4_llada.py --phase prefill \
  --dtype float16 --atol 0.005 --rtol 0.005 \
  --warmup 100 --iterations 200 --repeats 20
export PROFILE_DIR=test/runtime/integration/profiles/fa4_llada_fp16
PHASE=prefill BACKEND=fa4 bash test/runtime/integration/profile_fa4_llada_nsys.sh \
  --dtype float16 --atol 0.005 --rtol 0.005
PHASE=prefill BACKEND=fa4 bash test/runtime/integration/profile_fa4_llada_ncu.sh \
  --dtype float16 --atol 0.005 --rtol 0.005
```

Repeat with `--phase decode` / `PHASE=decode` and `BACKEND=batchprefill`.
