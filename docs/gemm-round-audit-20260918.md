# Compiled GEMM and target-head audit — 18 September 2026

The GEMM backport saves **13.91 ms per complete round** on the existing 60K Pi
fixture with global top-256 held constant. The final configuration measures
**42.85 ms median rounds and 121.15 t/s**. The earlier approximately 17 ms figure
was an extrapolation from one isolated GEMM, not an end-to-end measurement.

## Controlled results

All three configurations use the corrected arithmetic, FP8 norm/quant backport,
existing nine-slot state layout and compiled piecewise graphs. The actual GEMM
binary is selected before graph capture; native traces confirm the dispatch.
Lazy GDN is disabled. Each configuration generates three natural responses on
the same existing 60,000-token Pi prefix, with seeds 0/17/42, temperature 0.6,
top-p 0.95 and top-k 20. No chat text or token arrays were inspected.

| GEMM dispatch | Target head | Median round ms | Mean round ms | Post-first t/s | Natural output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Original | Global top-256 | 56.76 | 57.95 | 92.27 | 2,375 |
| Backported | Corrected full BF16 | 45.93 | 47.04 | 115.73 | 2,375 |
| Backported | Global top-256 | **42.85** | **43.95** | **121.15** | 2,375 |

All paired output hashes and output counts match. This is sampled agreement,
not a proof of the approximate top-256 head's equivalence to full BF16.

Round statistics exclude each response's first eight output rounds; the median
column is the median of its three response medians and the mean column is the
mean of three response means. Throughput pools every output after the first
observed output and includes warm-up rounds. It therefore need not equal a
simple tokens-per-round divided by the median round time. These are direct-engine
results; HTTP serving, snapshots and chat scheduling are not included.

## Native dispatch and the 17 ms estimate

A separate 256-token pass profiles eight rounds. Its elapsed time is excluded
from throughput above. Pairing all 2,048 MXFP4 dispatches verifies that the 512
folded gate/up kernels become decode kernels, with every remaining dispatch
name unchanged.

| GPU work per round | Original GEMM | Backported GEMM | Change |
| --- | ---: | ---: | ---: |
| Gate/up GEMMs across 64 layers | 25.117 ms | 11.331 ms | −13.786 ms |
| Remaining MXFP4 GEMMs | 12.540 ms | 12.889 ms | +0.348 ms |
| All GPU kernel durations | 54.798 ms | 41.281 ms | −13.517 ms |
| Unprofiled complete round, median | 56.764 ms | 42.852 ms | −13.912 ms |

The isolated rebuilt-control M8 timing was 0.418448 → 0.155914 ms.
Multiplying by 64 predicted 16.802 ms. In the model the same stage measures
0.392447 → 0.177047 ms per layer, saving 13.786 ms. Isolated repetition does
not reproduce the full model's execution conditions; the direct measurement
replaces that estimate. The trace proves this is not a missing GEMM backport.

Holding GEMM constant, global top-256 reduces the median round by **3.080 ms**
against the corrected full BF16 head. The latter's native head kernel measures
4.139 ms per round. Total GPU kernel time falls from 44.425 to 41.281 ms when
top-256 is enabled.

The earlier serving comparison measured 57.65 → 44.79 ms mean round intervals.
Its full-head arm had additional overhead absent from these controls. That
overhead's cause remains unclassified; the earlier 12.86 ms deployment change
must not be presented as an isolated target-head saving or added to the GEMM gain.

## Evidence and reproduction

- Aggregate: [gemm-round-audit-20260918.json](gemm-round-audit-20260918.json).
- Local receipts: `artifacts/deployment/20260918-gemm-round-audit/`.
- Remote traces and receipts: `~/.local/state/qwen-r9700/conformance/20260915/preflight/gemm-round-audit-20260918/` on the GPU host.
- Runner: `experiments/radiance-public/benchmark_gemm_round_audit.py` with
  `--variant original|candidate`, `--target-head global256|full-bf16`, and a fresh
  `--output` directory inside the pinned qualification container.
- `gemm_round_audit_worker.py` selects the original or authenticated candidate
  extension before compilation/capture, rebinds its scratch, and records the
  selected binary hash. The performance payload's installation receipt alone
  is not used to infer which variant the diagnostic selected.

The test processes were shut down before restoring production with both
correctness repairs, performance recovery, GEMM/FP8 backports and global top-256.
No test prompt is submitted after the final restart, leaving no benchmark GPU bank
that a real chat would first have to evict.
