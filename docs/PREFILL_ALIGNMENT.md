# Prefill and continuation arithmetic alignment

Cold reconstruction of a token history and continuing its generated state used
different floating-point operations. This was a numerical defect, separate from
snapshot serialization. It also reproduced with response-end reuse disabled and
with the full BF16 vocabulary head.

The repair preserves the corrected decode arithmetic during prefill. It changes
neither the checkpoint, quantizers, sampling settings nor the physical nine-slot
GDN state layout. No loop guard or invisible steering message is added.

## Measured disagreement

The comparison forces both paths through the same synthetic token history. A
single full BF16 head scores both captured final hidden tensors. Thus an early
different prediction cannot change the later test inputs, and approximate
Global-512 candidate selection cannot explain these differences. Private chats
were not read or decoded.

At a **60,000-token prefix, over 1,000 subsequent token positions**:

| Comparison | Before repair | After repair |
| --- | ---: | ---: |
| Top-1 token | 987 / 1,000 (98.7%) | 1,000 / 1,000 (100%) |
| Top-10 token set | 648 / 1,000 (64.8%) | 1,000 / 1,000 (100%) |
| Top-10 ordering | 169 / 1,000 (16.9%) | 1,000 / 1,000 (100%) |
| Top-20 token set | 362 / 1,000 (36.2%) | 1,000 / 1,000 (100%) |
| Top-20 ordering | 3 / 1,000 (0.3%) | 1,000 / 1,000 (100%) |
| Entire hidden vector, byte exact | 0 / 1,000 | 1,000 / 1,000 |
| Entire vocabulary-logit vector, byte exact | 0 / 1,000 | 1,000 / 1,000 |

After repair, all **2,320 sampled positions** agree exactly: 1,000 following a
1,651-token prefix, 1,000 following a 60,000-token prefix, and 320 following a
200,000-token prefix. The short-prefix baseline changed 14 of 1,000 top-1
predictions; the 60K baseline changed 13. These are agreement measurements on
synthetic inputs, not percentages of intelligence loss or evidence that a
particular fraction of reasoning loops has been eliminated.

## First-divergence localization

| Cause | Evidence | Repair |
| --- | --- | --- |
| Attention reduction depends on query batch size | First original mismatch at layer 3, the first full-attention layer. Queries and logical KV bytes already agree; 158 of 49,152 attention-output values differ across eight shared query rows. | Use decode's per-query TILE16 processing, split boundaries and FP32 merge order for every prefill query. Queries remain parallel and share KV loads. |
| Output projection changes accumulation partitions | After attention alignment, the first remaining mismatch is layer 20's MLP down projection. Decode uses four K partitions; the large-row prefill path accumulates differently. | Keep the same four contiguous, 128-coefficient-aligned K ranges, ordered FP32 merge, scaling and final BF16 rounding. Preserve folded and tiled layouts. |
| Final residual normalization selects a different reduction | After both repairs, 16 of 1,000 final hidden vectors still differ, involving 22 elements. Top-1/10/20 already agree, but full vectors do not. | Admit prefill widths 9–2,048 to the qualified residual-normalization kernel already used by decode. |

Stage capture identifies the earliest unequal operation using identical inputs;
later unequal GDN values alone do not establish a GDN defect. No additional GDN
recurrence defect was established by this investigation. Partial-hook diagnostic
attempts and interrupted captures are not qualifying results.

## Operator, cache and release checks

- Eleven native attention cases compare against independent M1 workgroups,
  covering page boundaries, shuffled physical page locations, varied KV scales,
  BF16/FP8 storage and contexts up to 253,792 tokens.
- Ten complete projection-output comparisons cover captured model activations
  at widths 8, 65, 256, 1,003 and 1,648 in both normal and tiled layouts.
- The original cold-recompute versus response-end tool-continuation regression
  passes six greedy cases with the full BF16 head and six with production Global-512.
  Each continuation reuses its exact endpoint and processes only the 41 appended
  synthetic tokens.
- After freezing the runtime and restarting with hooks installed before
  compilation, the short-prefix 1,000-position comparison passes again. This
  repeats an existing sample; it does not increase the 2,320 distinct positions.
  All eleven packaged lifecycle cases also pass, including cold-prefill and
  queued cancellation, handover, priority changes and sampled replay.
- Qualification authenticates runtime sources, native binaries and aggregate
  evidence. Missing, modified, divergent or unsupported evidence fails release
  admission. Unsupported prefill shapes fail explicitly instead of silently
  selecting the old arithmetic.

The serving scope is one R9700, TP1, one scheduled sequence, the pinned model and
its full causal attention, and prefill widths through 2,048. Source and binary
identities, individual measurements and the final packaging checks are recorded
in [the aggregate evidence](../benchmarks/results/prefill-alignment-20260927.json).
Captured tensors and token fixtures remain private.

The [production deployment receipt](../benchmarks/results/prefill-alignment-deployment-20260927.json)
records installation before compilation, five passing synthetic API smoke
requests, deterministic fresh-prompt replay and rejection of the old snapshot
identity. Round-time and acceptance telemetry remain enabled. These checks are
not a new throughput benchmark.

The diagnostic tools live under `experiments/radiance-public/`:
`diagnose_prefill_decode.py` drives cold and forced-continuation capture;
`prefill_divergence_probe.py` records boundaries and compares hidden/logit rows;
`prefill_attention_alignment.py` and `prefill_gemm_alignment.py` build the native
repairs; `collect_prefill_alignment_evidence.py` validates and seals aggregate
results; `prepare_prefill_alignment_release.py` freezes those exact artifacts
and updates their arithmetic identity. The probe is an isolated-test worker
extension, not enabled in production. Its fixture and capture paths are explicit
inputs; never substitute a user's transcript for the synthetic fixture.

## Performance and memory

This repair has a measured prefill cost. For one captured attention invocation
with 1,003 query rows, five GPU-event samples gave:

| Cached context | Previous prefill attention | Aligned prefill attention |
| --- | ---: | ---: |
| 60K | 20.17 ms | 34.50 ms |
| 200K | 63.49 ms | 112.08 ms |

A single whole-model cold 61K diagnostic request took 36.56 seconds before and
45.34 seconds after alignment, approximately 24% longer. This is a diagnostic
comparison, not a repeated production benchmark. Decode kernels are unchanged;
the README's earlier decode timing table is not a new timing qualification for
this release.

Scratch allocation is bounded: 48.4 MiB for attention and 20 MiB for projections,
about 68.4 MiB together. Projection scratch is reused in 256-row windows rather
than allocating roughly 160 MiB for a 2,048-row batch.

## Snapshot migration and limits

The snapshot arithmetic identity changes. Old numerical snapshots must not be
restored into the repaired execution path; their token histories are still
valid. The first request for an existing chat therefore rebuilds its cache.
Previous snapshot files are retained for rollback, in their separate ABI
namespace. Subsequent tool continuations still use exact response-end reuse.

Sampled equality is not a universal proof that arbitrary inputs, every prefill
partition, all session transitions or the bare weight-quantized model agree.
Global-512 remains an approximate head. A conforming model can still repeat
itself. These repairs remove demonstrated numerical differences; measuring a
change in loop frequency would require a separate controlled behavioral study.
