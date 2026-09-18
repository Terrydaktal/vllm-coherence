# GEMM dispatch backport — 18 September 2026

Backported the decode-width dispatch from
[GGZ14 commit 3f542b7](https://github.com/GGZ14/vllm-mxfp4/commit/3f542b7cbfce3fa0d01dc55665af4c77a7093ce8)
onto the pinned Radiance 1.0.16 kernel. Qwen's TP1 gate/up projection has 34,816
output columns; the old 32,768-column limit sent it through a larger prefill tile.
The new limit is 36,864, using the existing decode kernel. Scratch increases from
32 to 36 MiB. No GDN, activation precision, normalization, or attention changes
are included.

The backport additionally retains the old fallback above 32,768 columns when
split-K is forced above one. The unguarded upstream change produced four unequal
BF16 values in a forced-four-way test; the guarded backport preserves the old
result. The normal automatic policy already selects one split at these widths.

## Quick correctness result

**Zero mismatches in 115,841,664 candidate/reference BF16 element comparisons.**
All 55 checks passed:

- 320 deterministic input rows each, using actual checkpoint weights from layers
  0, 31 and 63, through both M1 and M8.
- Exact comparison against the frozen shipped binary and an unchanged source
  control rebuilt with the same compiler as the candidate.
- M1/M8 equality, native calls and captured graph replay. Output poisoning before
  replay verifies that the graph actually computes the results.
- Row counts 1, 8, 64 and 65; column counts 32,768, 32,784, 34,816, 36,864 and
  36,880, spanning both dispatch limits. The shared dimension is 5,120.
- Output/scratch/counter guard regions, settled counters, finite outputs, and a
  deliberately flipped BF16 bit to verify mismatch detection.
- A separate forced-four-way split-K pass.

These are **GEMM-stage checks using synthetic activations and actual weights**.
No chat contents were read. This is not a new 10K whole-model comparison or proof
over arbitrary inputs. Sixteen CPU admission/regression tests also passed.
The runtime adapter also passed a TP1 initialization and actual Radiance custom-op
graph replay smoke test, with exact agreement across all 278,528 output values.

## Graph-enabled stage timings

GPU-event medians, nine alternating batches of twenty graph replays. Each entry
is one gate/up GEMM, not a whole decoder layer or generation round.

| Rows | Frozen original | Rebuilt control | Backport | Time reduction vs original |
| --- | ---: | ---: | ---: | ---: |
| M1 | 0.433 ms | 0.430 ms | 0.151 ms | 65.1% |
| M8 | 0.424 ms | 0.418 ms | 0.156 ms | 63.3% |

Multiplying the isolated M8 saving by 64 layers predicted about 16.8 ms per
round. The later [compiled whole-model comparison](gemm-round-audit-20260918.md)
measured **13.79 ms** in gate/up GEMMs and **13.91 ms** in the complete round:
56.76 → 42.85 ms with global top-256 held constant. The isolated estimate is
superseded by that direct measurement; it was not an observed 17 ms round saving.

## Reproducible integration

- `experiments/radiance-public/build_mxfp4_dispatch.py` checks the pinned native
  and Python source hashes and builds an unchanged control plus the backport.
- `probe_mxfp4_dispatch.py` checks the built binaries against the frozen installed
  extension and writes numerical, memory-safety and graph-timing evidence.
- `prepare_mxfp4_dispatch.py` validates both test receipts and creates a separate
  corrected runtime bundle, retaining the previous bundle.
- `mxfp4_dispatch.py` verifies artifact/evidence hashes, then installs the candidate
  and its scratch after weight loading and before compilation/graph capture.
  `optimized_d7_performance.py` invokes this adapter when the bundle includes it.

The prepared bundle is in the qualification container at
`/qualification/preflight/mxfp4-dispatch-v1/qualified-runtime/performance.json`;
its Python files are in the adjacent `runtime/` directory. The existing repair
manifest remains unchanged. No production backend was restarted for these checks.

Aggregate evidence: [gemm-dispatch-backport-20260918.json](gemm-dispatch-backport-20260918.json).
Full local receipts: `artifacts/conformance/20260918-gemm-dispatch/`.
