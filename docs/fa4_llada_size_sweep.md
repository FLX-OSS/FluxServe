# FP16 attention size sweep

For KV=8192/16384 and a same-allocation KV=4096 anchor, see the
[long-KV FP16 recheck and line plots](fa4_llada_long_kv.md).

## Expanded, sustained-warmup recheck

The original and expanded sweeps both explicitly use `--dtype float16`.
The extended experiment (GH200 job 3091779) adds B=8 and KV=512/2048:
B={1,4,8,16}, KV={256,512,1024,2048,4096}, both phases (40 cases),
plus the two original heterogeneous workloads. Each case is run in two
independent processes (one process per full pass; reverse case order on pass 1).
Every runner gets 100 warmup calls plus 0.5 seconds of synchronized warmup;
timing then uses 12 rotated rounds of 100 calls. All four methods share the
same workload tensors; JIT, planning, capture and reference checks are outside
timing. Clocks are not locked and GPU exclusivity beyond the allocated GPU
does not imply CPU/system isolation. Sustained warmup reduces, but does not
prove absence of, measurement drift.

See [expanded tables and line plots](fa4_fp16_expanded_results.md).
Plot points are the mean of the two independent process medians; whiskers
show the two-median range, not a confidence interval. Both axes are logarithmic.
Raw results: `test/runtime/integration/profiles/fa4_llada_fp16_expanded/`.

![Expanded FP16 prefill](fa4_fp16_prefill.svg)

![Expanded FP16 decode](fa4_fp16_decode.svg)

### Why the previous numbers differ

The original prefill B=4/KV=4096 graph median (719.24 us) exceeded eager
(715.90 us) by only 3.34 us / 0.47%. Round-average standard deviations were
5.87 and 11.39 us, respectively, with overlapping ranges. This does not
establish a stable regression, nor does it prove the entire difference is
random noise. Graph reduces submission overhead; it does not guarantee a
faster GPU-bound kernel. Small differences require repeated same-shape tests
and, if persistent, targeted kernel/clock/memory-layout investigation.

The sustained-warmup recheck also shows a small graph slowdown at this exact
size, so it must not simply be dismissed as noise:

| Prefill B=4, KV=4096 | FA4 eager (us) | FA4 graph (us) | Graph latency increase |
|---|---:|---:|---:|
| Pass 0 | 696.79 | 706.49 | 1.39% |
| Pass 1 | 692.03 | 698.90 | 0.99% |

This is a small observed regression in both passes, not a proven mechanism.
The defensible conclusion is that this size does not benefit from graph;
the benchmark has not isolated whether the residual difference is due to
clock/cache state, captured-buffer placement, device scheduling or another
effect. No claim of a specific CUDA Graph kernel overhead is established.

The old ~122 us decode measurement used heterogeneous KV lengths
256,512,768,1024,1280,1536,1792,2048,2560,3072,3584,4096,5120,6144,7168,8192.
The 89.63 us row used 16 requests all at KV=4096. Both use 64 Q tokens per
request, FP16 and the same head geometry, but they are different workloads:
total KV = 49152 vs 65536, maximum KV = 8192 vs 4096. Thus even total FLOPs
alone cannot explain the ordering. Per-request work distribution and the
maximum-length argument passed to the kernel differ; scheduling/configuration
effects are plausible, but are not isolated by this benchmark. The expanded
run includes both exact workloads so comparisons no longer mix separate jobs.

| Decode workload | Pass 0 FA4 eager / graph (us) | Pass 1 FA4 eager / graph (us) |
|---|---:|---:|
| Original mixed KV, max 8192 | 120.96 / 118.98 | 120.74 / 118.70 |
| B=16, all KV=4096 | 88.95 / 85.75 | 89.17 / 85.59 |

The large difference reproduces within this one allocation. Comparing the
same workload across passes is much more stable than comparing different
length distributions under the generic label "decode".

Reproduce the expanded sweep in the configured FA4 GPU container:

```bash
for pass_id in 0 1; do
  python test/runtime/integration/sweep_fa4_llada.py \
    --pass-id "$pass_id" \
    --output-dir test/runtime/integration/profiles/fa4_llada_fp16_expanded
done
python3 test/runtime/integration/plot_fa4_llada_sweep.py \
  test/runtime/integration/profiles/fa4_llada_fp16_expanded docs
```

The plotter rejects missing cases, missing passes, non-FP16 records and failed
correctness gates. Keep output directories distinct to preserve older runs.

## Original 18-case exploratory sweep

GH200 job 3091532. This supplements, rather than replaces, the original
heterogeneous workloads in `fa4_llada_cuda_graph_results.md`.

Batch sizes: 1, 4, 16; uniform total KV lengths: 256, 1024, 4096.
Prefill Q length equals KV length and uses block-causal attention; decode
Q length is 64 with the remaining tokens as prefix. Heads = 16/4, D=128,
block/page=64. Matching B and KV does not mean equal prefill/decode FLOPs:
prefill produces the entire sequence; decode produces only the last block.

Each of 18 cases checks FA4 and BatchPrefill against all-query FP32 math SDPA
(atol=rtol=0.005), checks graph reference output, and tests bitwise graph/eager
agreement after two Q updates. Timing uses the same tensors for both backends,
50 warmup calls, 12 rounds of 100 calls, rotating all four measurement positions.
The 12 raw round averages, means, medians and standard deviations are retained
under `test/runtime/integration/profiles/fa4_llada_fp16_size_sweep/`.

All 18 cases completed successfully; maximum absolute error versus FP32
SDPA across both backends and all cases was 0.00059271. All graph checks passed.

These are attention API, warm/reused-buffer, queued CUDA-event measurements,
not server latency, cold-cache performance or per-call latency percentiles.
Small-case eager samples show downward drift over rounds: 50 warmup calls did
not establish fully stationary timings. Treat this sweep as size-dependence
evidence, not a precise universal graph speedup claim. A publication-quality
speedup estimate needs sustained warmup, repeated independent runs and CPU/GPU
clock/environment controls. Only B and KV are swept; head geometry, block
length, page size, irregular lengths and cache-update costs remain out of scope.

Results (median of round-average times, microseconds):

| Phase | B | KV | FA4 eager | FA4 graph | BatchPrefill eager | BatchPrefill graph |
|---|---:|---:|---:|---:|---:|---:|
| Prefill | 1 | 256 | 62.61 | 9.91 | 36.90 | 12.87 |
| Prefill | 1 | 1024 | 63.58 | 19.87 | 45.18 | 25.69 |
| Prefill | 1 | 4096 | 180.93 | 180.80 | 258.95 | 259.06 |
| Prefill | 4 | 256 | 60.19 | 10.33 | 35.49 | 14.49 |
| Prefill | 4 | 1024 | 65.13 | 64.30 | 81.09 | 80.47 |
| Prefill | 4 | 4096 | 715.90 | 719.24 | 862.57 | 865.00 |
| Prefill | 16 | 256 | 57.89 | 35.69 | 41.93 | 41.32 |
| Prefill | 16 | 1024 | 292.94 | 290.18 | 312.97 | 311.38 |
| Prefill | 16 | 4096 | 2863.93 | 2861.87 | 3303.33 | 3314.66 |
| Decode | 1 | 256 | 62.77 | 9.72 | 38.09 | 12.12 |
| Decode | 1 | 1024 | 61.67 | 18.70 | 38.05 | 22.41 |
| Decode | 1 | 4096 | 60.87 | 53.59 | 61.73 | 61.28 |
| Decode | 4 | 256 | 61.21 | 9.84 | 39.92 | 12.96 |
| Decode | 4 | 1024 | 60.71 | 18.76 | 38.31 | 23.42 |
| Decode | 4 | 4096 | 58.57 | 56.24 | 67.73 | 67.54 |
| Decode | 16 | 256 | 58.79 | 11.08 | 37.77 | 21.45 |
| Decode | 16 | 1024 | 59.28 | 21.44 | 48.49 | 47.57 |
| Decode | 16 | 4096 | 89.63 | 86.97 | 190.10 | 187.85 |

The original conclusion must be workload-specific: graph can help short
decode calls substantially, and can have negligible effect on long prefill
calls. At B=1/KV=4096, prefill graph takes 180.80 us versus decode 53.59 us;
the original heterogeneous phase comparison had reversed workload sizes.

Reproduce in the configured GPU environment:

```bash
for phase in prefill decode; do
  for batch in 1 4 16; do
    for length in 256 1024 4096; do
      python test/runtime/integration/profile_fa4_llada.py \
        --phase "$phase" --batch-size "$batch" --kv-length "$length" \
        --execution both --dtype float16 --atol .005 --rtol .005 \
        --warmup 50 --iterations 100 --repeats 12 \
        --output "/tmp/${phase}_b${batch}_kv${length}.json"
    done
  done
done
```

The same size arguments work with the NSYS and NCU wrapper scripts for
targeted profiling. This sweep itself is unprofiled; original mixed-workload
NSYS/NCU artifacts are unchanged.
