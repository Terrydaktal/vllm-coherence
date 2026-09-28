# Prefill speed recovery

The 28 September build preserves the corrected 27 September prefill/decode
arithmetic. It changes work sharing, scratch traffic and compilation reuse. It
does not restore the older, numerically different attention or projection path.

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

## Measurement

The [prefill measurements](../benchmarks/results/prefill-speed-timings-20260928.json)
record these backend prefill phases:

| Context | Older, numerically inconsistent path | Corrected before optimization | Current corrected path |
| --- | ---: | ---: | ---: |
| 60K | 36.636 s | 48.300 s | 35.500 s |
| 200K | 207.078 s | 280.404 s | 208.848 s |

Current values average two cold requests at each context: 35.448/35.553 s at
60K and 206.951/210.745 s at 200K. Their mean request-to-first-data times were
35.858/209.553 s. The earlier columns retain one request per context from their
separate recorded runs. The older inconsistent controls and current requests
have matching prompt hashes; the initial corrected record did not retain
per-prompt hashes, so its time reduction is an indicative historical comparison.
This is about 3.1%
faster than the old path at 60K and 0.9% slower at 200K, with the 200K repeats
spanning the old result. It is not evidence of a speedup on every workload.
The time reductions against the initial corrected path are 26.5%/25.5%.

The [initial corrected comparison](../benchmarks/results/prefill-speed-initial-timings-20260928.json)
and the [paired packed-attention comparison](../benchmarks/results/prefill-packed-timings-20260928.json)
remain separate evidence. The latter isolates the attention change at the old
chunk size. The final release packages the measured native files; its startup,
snapshot compatibility and lifecycle are checked separately.

Use `experiments/radiance-public/benchmark_prefill_speed.py` on the isolated
server with an explicitly synthetic fixture. It compares fresh generations in
one process, requires zero reused prompt tokens and records both first-data wall
time and backend phase durations. Normal compilation, caching and scheduling
remain enabled; no stage profiler is active. One generated token measures
prefill latency, not generation throughput. Counterbalanced repeats are supported.

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

`qualify_prefill_speed.py` compares the entire final hidden vector and full BF16
vocabulary-logit vector with retained corrected decode captures at 60K and 200K.
It checks the forced token identities and positions and records source, binary
and fixture hashes. Native attention tests cover randomized physical pages,
split/adaptive boundaries, both admitted KV formats and the 253,792-token limit.
Projection, GDN and convolution checks compare complete outputs; recurrence and
convolution also compare their retained state.

The [final candidate qualification](../benchmarks/results/prefill-speed-qualification-20260928.json)
matched all 1,000 positions at 60K and all 320 positions at 200K: complete hidden
vectors, complete BF16 logits, and top-1/10/20 sets and ordering. It also passed
68 attention cases (635,731,968 output elements), 108 output-projection cases,
52 GDN cases, 54 input-projection cases and 52 convolution cases. Normalization
matched 1,000 rows at each of the 48 GDN sites for M1 and M8, plus the wider
prefill boundaries through 4,096 rows. Cached tool continuations matched fresh full
prompts in six cases each with the full BF16 and Global-512 heads.

The ordinary Global-512 serving head is unchanged. The full BF16 head in this
qualification exposes backbone differences that a shortlist or matching winner
could hide. Exact samples establish no observed additional numerical change;
they do not prove all arbitrary inputs or that the underlying quantized model
cannot loop.

## Release and snapshot admission

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

The [frozen-release checks](../benchmarks/results/prefill-speed-lifecycle-20260928.json)
passed six corrected-parent snapshot restores and all nine greedy/control
cancellation and scheduling cases, including cancellation during prefill and
decode, queue cancellation, response handover and priority preemption. The
seeded stochastic replay assertion did not pass; its scope is explained below.
The changed activation packer also passed all
twelve original [packing/graph-replay checks](../benchmarks/results/prefill-speed-packing-20260928.json).
The runtime identity changes; the corrected snapshot data identity is retained.
The [deployment receipt](../benchmarks/results/prefill-speed-deployment-20260928.json)
records the installed sources, startup checks and five passing synthetic API
requests. Ordinary Pi and Pi-opsec use the same restarted model server.

The matching portable archive is prepared locally as
`coherence-runtime-prefill-speed-20260928-linux-amd64-gfx1201.tar.xz`. Use
`tools/coherence prepare --archive PATH` with that archive. Its new asset name
has not been published; the preceding public archive is not this build.

Additional key prefetching, double-buffered attention and several other layouts
were explored and rejected. Their measurements remain experiment evidence.
`qualify_prefill_wide.py` reruns the wider operator checks; `--stages` can resume
unfinished groups without overwriting completed evidence.

## Seeded sampling is a separate comparison

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
