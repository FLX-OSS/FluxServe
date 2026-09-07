# Expanded FP16 measured results

Generated from 48 JSON records: 24 cases × 2 independent process passes.
Values: mean of the two process medians, in microseconds. Each process median is over 12 round averages, not individual-call latencies.

![Prefill](fa4_fp16_prefill.svg)

![Decode](fa4_fp16_decode.svg)

| Case | FA4 eager | FA4 graph | BatchPrefill eager | BatchPrefill graph |
|---|---:|---:|---:|---:|
| prefill_b1_kv4096 | 185.03 | 181.15 | 253.42 | 252.59 |
| prefill_b1_kv8192 | 624.14 | 623.90 | 939.53 | 942.57 |
| prefill_b1_kv16384 | 2259.65 | 2266.65 | 3640.23 | 3643.15 |
| prefill_b4_kv4096 | 724.18 | 711.07 | 846.84 | 847.43 |
| prefill_b4_kv8192 | 2521.93 | 2531.99 | 3014.62 | 3013.13 |
| prefill_b4_kv16384 | 9686.01 | 9635.88 | 11738.84 | 11772.37 |
| prefill_b8_kv4096 | 1403.70 | 1391.26 | 1605.74 | 1617.71 |
| prefill_b8_kv8192 | 5016.12 | 5001.72 | 5955.99 | 5943.06 |
| prefill_b8_kv16384 | 19104.65 | 19115.97 | 23309.25 | 23361.15 |
| prefill_b16_kv4096 | 2856.10 | 2839.26 | 3249.81 | 3244.28 |
| prefill_b16_kv8192 | 10239.41 | 10184.90 | 12064.66 | 12126.03 |
| prefill_b16_kv16384 | 37999.36 | 38003.64 | 46942.42 | 46952.33 |
| decode_b1_kv4096 | 57.28 | 54.11 | 63.11 | 61.53 |
| decode_b1_kv8192 | 104.92 | 101.45 | 116.56 | 114.80 |
| decode_b1_kv16384 | 198.82 | 196.22 | 222.71 | 221.75 |
| decode_b4_kv4096 | 59.44 | 56.31 | 69.10 | 67.97 |
| decode_b4_kv8192 | 110.29 | 107.38 | 129.61 | 127.72 |
| decode_b4_kv16384 | 211.61 | 207.72 | 247.55 | 243.44 |
| decode_b8_kv4096 | 76.66 | 73.86 | 92.84 | 91.08 |
| decode_b8_kv8192 | 144.98 | 139.72 | 178.39 | 174.40 |
| decode_b8_kv16384 | 279.77 | 269.01 | 339.34 | 332.17 |
| decode_b16_kv4096 | 92.10 | 87.99 | 191.32 | 188.10 |
| decode_b16_kv8192 | 172.71 | 167.09 | 362.34 | 354.00 |
| decode_b16_kv16384 | 331.76 | 323.64 | 703.06 | 685.50 |

All correctness gates passed; maximum absolute FP32 SDPA error: 0.000545382.

Whiskers are the range of two process medians, NOT confidence intervals. Raw per-round variability remains in JSON.
