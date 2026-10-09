# Prefill speed

The current cold-prefill measurements are from **9 October 2026**, using frozen
source [`99d5fcf`](https://github.com/Terrydaktal/vllm-coherence/commit/99d5fcfd1bcd0fe6f70445d154e63b7092b02f24).
The implementation retains the corrected prefill/decode arithmetic and the work
sharing, scratch-traffic and compilation-reuse optimizations introduced on
28 September. The October refresh measured speed only; the numerical and cache
lifecycle qualifications below retain their original dates.

## Changes

| Operation | Optimization | Preserved numerical boundary |
| --- | --- | --- |
| Attention | Pack eight queries into three WMMA waves, share renormalization votes with the existing V barrier, and stage the next key tile after current readers finish; retain the short-context kernel | Per-query six-head renormalization decision, TILE16 traversal, M1 split boundaries, FP32 partials and ordered merge |
| Output projections | Keep four partial accumulators in registers instead of writing and reloading four global matrices | Same four K partitions, FP32 accumulation/merge order and final BF16 rounding |
| GDN recurrence | Calculate query/key normalization and gates once per row, then reuse FP32 prepared inputs | Chronological recurrence, original reduction order, beta rounding and final FP32 state |
| Input projections | Use the existing qualified activation layout for the actual merged GDN shape and intermediate prompt lengths | Same packed weights, activation bytes/scales, accumulation and output rounding |
| Changing prompt lengths | Reuse compiled packing, preparation and convolution kernels across lengths | Integer lengths become runtime inputs; numerical operations and masks remain unchanged |
| Chunk size | Admit up to 4,096 rows, allowing the scheduler to use 3,296-row chunks | Each query/row retains its arithmetic and causal boundary; recurrent transitions remain chronological |

The convolution adapter targets the separately pinned stock module used by the
corrected GDN path. An ordinary vLLM import is a different module. Qualification
requires counters showing that the intended optimized paths actually ran, in
addition to comparing their outputs.

Attention uses a 96.75 MiB bounded workspace. Projection partials no longer need
the old 20 MiB global workspace. Together this is about 28.4 MiB more persistent
scratch than the previous corrected implementation. GDN preparation also uses
temporary FP32 input arrays, now bounded by the qualified 4,096-row prefill limit.
The wider batch increases temporary activation memory. The actual scheduled
chunk width is recorded, rather than inferred from the scheduler setting.
The existing nine-slot recurrent layout and decode kernels are unchanged.

## Current measurements: 9 October 2026

The [current capture](../benchmarks/results/prefill-speed-timings-20261009.json)
records two unprofiled cold synthetic requests per context in one server process,
using fresh chat generations and zero reused prompt tokens:

| Context | Mean backend prefill | Backend prefill range | Mean request-to-first-data |
| --- | ---: | ---: | ---: |
| 60,000 tokens | **33.249 s** | 33.046–33.453 s | 33.497 s |
| 200,000 tokens | **200.108 s** | 199.941–200.275 s | 200.666 s |

Greedy requests generate one token and request log probabilities to select the
full BF16 head explicitly. They measure cold-prefill latency, not generation
throughput. Normal compilation, caching and scheduling remain enabled; no stage
profiler is active. Request-to-first-data also includes admission, cache
preparation and first output. Source/build identities and prompt hashes are
recorded in the capture. These are current-build measurements, not a newly
paired comparison against an older release.

Use `experiments/radiance-public/benchmark_prefill_speed.py` on an isolated
server with an explicitly synthetic fixture. Counterbalanced repeats are
supported when performing a new paired comparison.

## Historical measurements: 28 September 2026

The [September 28 capture](../benchmarks/results/prefill-speed-timings-20260928.json)
recorded optimized cold-prefill means of 35.500 s at 60K and 208.848 s at 200K,
averaging two requests each. Those are historical measurements, not the current
speeds above. The [initial corrected comparison](../benchmarks/results/prefill-speed-initial-timings-20260928.json)
and [paired packed-attention comparison](../benchmarks/results/prefill-packed-timings-20260928.json)
retain the original controls; the latter isolates attention at the old chunk
size. The initial corrected record lacks per-prompt hashes, so it does not
establish a fully authenticated paired comparison with the current capture.

`profile_prefill_extension.py` can attribute a subsequent extension to kernels,
reusing an explicitly supplied synthetic bank with `--resume-chat`. It checks
prefix reuse before enabling the profiler. A 1,649-token extension near 202K
attributed roughly 79% of summed kernel duration to aligned attention traversal;
the [attribution](../benchmarks/results/prefill-speed-attribution-20260928.json)
includes instrumentation and is not used as a production timing.
That profile predates the packed attention change. The packed kernel's separate
11-repeat operator comparison reduced a 1,648-row invocation from 48.613 to
44.481 ms at 60K, and from 156.158 to 138.927 ms at 200K. These isolated numbers
are not whole-request prefill times.

## Historical numerical qualification: 28 September 2026

`qualify_prefill_speed.py` compares the entire final hidden vector and full BF16
vocabulary-logit vector with retained corrected decode captures at 60K and 200K.
It checks the forced token identities and positions and records source, binary
and fixture hashes. Native attention tests cover randomized physical pages,
split/adaptive boundaries, both admitted KV formats and the 253,792-token limit.
Projection, GDN and convolution checks compare complete outputs; recurrence and
convolution also compare their retained state.

The [September 28 qualification](../benchmarks/results/prefill-speed-qualification-20260928.json)
matched all 1,000 positions at 60K and all 320 positions at 200K: complete hidden
vectors, complete BF16 logits, and top-1/10/20 sets and ordering. It also passed
68 attention cases (635,731,968 output elements), 108 output-projection cases,
52 GDN cases, 54 input-projection cases and 52 convolution cases. Normalization
matched 1,000 rows at each of the 48 GDN sites for M1 and M8, plus the wider
prefill boundaries through 4,096 rows. Cached tool continuations matched fresh full
prompts in six cases each with the full BF16 and Global-512 heads. These vector,
operator and continuation comparisons were not rerun for the October 9 timing
capture.

The ordinary Global-512 serving head is unchanged. The full BF16 head in this
qualification exposes backbone differences that a shortlist or matching winner
could hide. Exact samples establish no observed additional numerical change;
they do not prove all arbitrary inputs or that the underlying quantized model
cannot loop.

## Release admission and historical September 28 deployment

`collect_prefill_alignment_evidence.py` rejects missing, divergent or incorrectly
bound evidence. `prepare_prefill_alignment_release.py` packages those exact
artifacts. Its `--preserve-snapshot-contract` option is restricted to an already
corrected parent with unchanged decode arithmetic, successful native and
full-model comparisons, and exact tool-continuation evidence. Implementation
hashes change independently of the retained numerical-reference identity.
Widening the two normalization dispatch guards requires a byte-for-byte check
that their only changes are the upper row bounds, plus fresh normalization
evidence. The new domain must pass native boundary checks before admission.

Deployment also requires restarting an isolated server on the frozen artifact,
restoring snapshots produced by the corrected parent, and checking cancellation
and concurrent-chat lifecycle behavior. Historical snapshots made with the
uncorrected prefill arithmetic remain incompatible.

The [September 28 frozen-release checks](../benchmarks/results/prefill-speed-lifecycle-20260928.json)
passed six corrected-parent snapshot restores and all nine greedy/control
cancellation and scheduling cases, including cancellation during prefill and
decode, queue cancellation, response handover and priority preemption. The
seeded stochastic replay assertion did not pass; its scope is explained below.
The changed activation packer also passed all
twelve original [packing/graph-replay checks](../benchmarks/results/prefill-speed-packing-20260928.json).
The runtime identity changes; the corrected snapshot data identity is retained.
The [deployment receipt](../benchmarks/results/prefill-speed-deployment-20260928.json)
records the installed sources, startup checks and five passing synthetic API
requests. It records that deployment, rather than establishing the identity of
the October 9 measured server.

That deployment's locally prepared archive was
`coherence-runtime-prefill-speed-20260928-linux-amd64-gfx1201.tar.xz`.
For current packaging and source verification, use [BUILDING.md](BUILDING.md)
and the measured source/manifest identities in the current capture.

Additional key prefetching, double-buffered attention and several other layouts
were explored and rejected. Their measurements remain experiment evidence.
`qualify_prefill_wide.py` reruns the wider operator checks; `--stages` can resume
unfinished groups without overwriting completed evidence.

## Historical seeded sampling comparison: 28 September 2026

The wider chunks changed a seeded DFlash continuation after cancellation at
output token 244. This also happened with the full BF16 head, while fresh
repeats matched. The previous 2,048-row release passed that sample-path test.
The failed assertion remains in the lifecycle evidence; it is not reported as
a passing deterministic replay test.

The [short-prefix comparison](../benchmarks/results/prefill-short-context-equality-20260928.json)
then checked 321 complete hidden/logit rows each around 2K and 4K: all matched
the previous release. A [live cancellation capture](../benchmarks/results/prefill-sampled-target-equality-20260928.json)
matched all 244 target hidden states and full logits through the first different
sample, with no differing element. Its draft proposals differed earlier, at
position 2,059, and the two executions reached that sample in different numbers
of speculative rounds.

Different proposals can change the realization of a seeded speculative sample
without changing the target distribution. These observations isolate this
failure from target prefill arithmetic; they do not prove the entire sampler's
distribution for arbitrary inputs. Exact seeded stochastic replay across
cancellation/chunk layouts is not an admitted guarantee of this release.

The earlier timing driver requested greedy `top_k=129`; that alone does not
force the full head because greedy head admission ignores `top_k`. Its stored
timings are backend prefill phases with one generated token, not full-head
latency measurements. The driver and response-end qualifier now request log
probabilities when a full-head control is intended. The
[explicit rerun](../benchmarks/results/prefill-explicit-full-head-tools-20260928.json)
passed all six tool continuations;
[dispatch counters](../benchmarks/results/prefill-explicit-full-head-dispatch-20260928.json)
verify zero Global-512 calls and 374 full-head calls. Full-vocabulary numerical
comparisons above compute the complete reference head directly and are unaffected.
