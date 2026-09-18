# Benchmark summary

Medians across repetitions. TPS is output-token throughput; TPOT and TTFT are milliseconds.

| Config | Workload | In/out | C | Temp | N | Output TPS | Total TPS | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Spec accept % | CV % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| safe-wperm-nt-postreboot | context_quick | 8128/64 | 1 | 0 | 1 | 27.19 | 3480.34 | 1827.16 | 1827.16 | 8.36 | 8.36 | 66.23 | 0.00 |
| safe-wperm-nt-postreboot | correctness_fixed | / |  | 0 | 1 |  |  |  |  |  |  |  |  |
| safe-wperm-nt-postreboot | decode | 256/256 | 1 | 0 | 2 | 195.16 | 390.32 | 75.39 | 75.88 | 4.25 | 8.53 | 54.70 | 30.28 |
| safe-wperm-nt-postreboot | decode | 256/256 | 2 | 0 | 2 | 231.10 | 462.20 | 129.40 | 174.24 | 7.24 | 12.51 | 36.81 | 20.08 |
| safe-wperm-nt-postreboot | decode | 256/256 | 4 | 0 | 2 | 355.25 | 710.51 | 161.14 | 279.28 | 7.41 | 16.00 | 40.28 | 32.46 |
| safe-wperm-nt-postreboot | decode | 256/256 | 8 | 0 | 2 | 556.43 | 1112.86 | 230.72 | 504.38 | 8.39 | 23.31 | 59.92 | 3.65 |
| safe-wperm-nt-postreboot | prefill | 2048/64 | 1 | 0 | 1 | 55.45 | 1829.93 | 442.33 | 449.07 | 11.30 | 13.80 | 18.80 | 0.00 |
| safe-wperm-nt-postreboot | prefill | 2048/64 | 4 | 0 | 1 | 89.77 | 2962.32 | 525.22 | 1837.49 | 24.67 | 57.35 | 27.14 | 0.00 |
| safe-wperm-nt-postreboot | prefill | 2048/64 | 8 | 0 | 1 | 106.43 | 3512.08 | 1005.53 | 3673.55 | 47.21 | 89.29 | 45.58 | 0.00 |
| safe-wperm-nt-postreboot | warmup | 128/64 | 1 | 0 | 1 | 173.48 | 520.45 | 62.33 | 62.45 | 4.86 | 5.38 | 53.44 | 0.00 |
| tool-schema-gate | unknown | / |  |  | 30 |  |  |  |  |  |  |  |  |
