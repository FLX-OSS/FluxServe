# LLaDA2.x FA4 correctness and GH200 profile

## Scope

The comparison uses LLaDA2.x production attention geometry: BF16, 16 query
heads, 4 KV heads, head dimension 128, block length 64, and page size 64. Every
backend receives the same Q storage, physical K/V cache storage, shuffled page
IDs, softmax scale, visible Q–K pairs, and output token order.

The performance timing measures eager attention API calls, including host
submission gaps and any output allocation/initialization performed by the API.
Input allocation, metadata construction, FlashInfer planning, JIT compilation,
and KV-cache population are performed before timing. It does not measure the
full FluxServe adapter. The runners alternate measurement order between
repeats. Every benchmark/profile process runs a cross-backend correctness gate
before it enters a timed or profiler range.

Hardware: NVIDIA GH200 120GB, compute capability 9.0. Profilers: Nsight Systems
2025.5.1 and Nsight Compute 2025.2.1.

## Correctness

The test ran prefill and decode with seeds 0, 1, and 7. All six cases passed at
`atol=rtol=3e-2` against an FP32 PyTorch SDPA block-causal reference.

| Phase | Comparison | Largest max abs error | Mean abs error range |
|---|---:|---:|---:|
| prefill | FA4 vs SDPA | 0.00781250 | 0.00018727–0.00018913 |
| prefill | BatchPrefillBlockExtend vs SDPA | 0.00781250 | 0.00018550–0.00018719 |
| prefill | FA4 vs BatchPrefillBlockExtend | 0.00390625 | 0.00002310–0.00002424 |
| decode | FA4 vs SDPA | 0.00390625 | 0.00012091–0.00012155 |
| decode | BatchPrefillBlockExtend vs SDPA | 0.00390625 | 0.00012032–0.00012069 |
| decode | FA4 vs BatchPrefillBlockExtend | 0.00390625 | 0.00005885–0.00006215 |

## CUDA-event benchmark

Each sample is the average of 100 calls. The table reports the mean of 10
balanced-order samples after 50 warmup calls.

| Workload | Query/KV shape | FA4 | BatchPrefillBlockExtend | Result |
|---|---|---:|---:|---:|
| prefill | Q lengths 128, 256, 512, 1024; 1,920 Q tokens | 0.05436 ms | 0.04323 ms | FA4 is 25.75% slower (0.795x) |
| decode | batch 16; 64 Q tokens/request; KV 256–8192 | 0.11719 ms | 0.18531 ms | FA4 is 36.76% faster (1.581x) |

The prefill result is consistent with host submission limiting this short
workload. Subtracting the NSYS kernel duration from the separately measured
CUDA-event average does not isolate CuTe overhead; attribution needs CPU/CUDA
correlation within a single trace. BatchPrefillBlockExtend's extra output-fill
kernel is included in both event timing and profiler summaries. These historical
measurements do not establish full-adapter or CUDA Graph performance.

## Nsight Systems

Twenty steady-state calls were captured per report.

| Workload/backend | Main attention kernel avg | Auxiliary GPU kernels | Total GPU kernel time/call |
|---|---:|---:|---:|
| prefill / FA4 | 27.050 us | none | 27.050 us |
| prefill / BatchPrefillBlockExtend | 37.800 us | output fill 3.301 us | 41.101 us |
| decode / FA4 | 112.869 us | none | 112.869 us |
| decode / BatchPrefillBlockExtend | 179.110 us | output fill 2.197 us | 181.307 us |

## Nsight Compute

The table reports the main attention kernel. BatchPrefillBlockExtend also
launches an output-fill kernel (4.51 us in prefill and 3.42 us in decode in the
instrumented NCU runs). NCU duration includes metric-replay/instrumentation and
should not replace the CUDA-event or NSYS latency measurements.

| Workload/backend | NCU duration | Compute | DRAM | Memory throughput | Achieved occupancy | Grid | Registers/thread | Dynamic shared memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill / FA4 | 37.86 us | 23.59% | 7.79% | 313.00 GB/s | 17.45% | 356 | 168 | 164.86 kB |
| prefill / BatchPrefillBlockExtend | 51.26 us | 21.18% | 5.77% | 232.07 GB/s | 18.40% | 132 | 168 | 131.15 kB |
| decode / FA4 | 149.18 us | 23.57% | 18.30% | 736.11 GB/s | 18.47% | 188 | 168 | 164.86 kB |
| decode / BatchPrefillBlockExtend | 226.85 us | 33.10% | 12.46% | 501.21 GB/s | 18.68% | 132 | 168 | 131.15 kB |

## Reproduction

Correctness:

```bash
FLUXSERVE_RUN_FA4_THREE_WAY=1 pytest -q -s \
  test/runtime/integration/test_fa4_llada_three_way.py
```

Benchmark:

```bash
python test/runtime/integration/profile_fa4_llada.py --phase prefill
python test/runtime/integration/profile_fa4_llada.py --phase decode
```

Profiler scripts select a backend with `BACKEND=fa4` or
`BACKEND=batchprefill`, and select the workload with `PHASE=prefill` or
`PHASE=decode`:

```bash
PHASE=decode BACKEND=fa4 \
  test/runtime/integration/profile_fa4_llada_nsys.sh
PHASE=decode BACKEND=fa4 \
  test/runtime/integration/profile_fa4_llada_ncu.sh
```
