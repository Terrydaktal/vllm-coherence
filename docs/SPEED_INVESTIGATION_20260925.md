# Preserved speed investigation — 25 September 2026

The investigation stopped at the user's request. Three experimental changes
passed sampled correctness checks and improved measured round time. They are
saved here for later integration; the production release remains unchanged.
The requested 5–10 ms saving was **not established**.

## Measured improvements

These are compiled, natural-completion coding runs using the same private
60,000-token Pi prefix, Global-512 target head, D7/M8, temperature 1.0,
top-p 0.95, top-k 40 and seed 113. Each measured completion produced 4,488
tokens over 1,020 generation rounds. No profiler was active in these timings.

| Change | Control ms/round | Candidate ms/round | Saving | Comparison |
| --- | ---: | ---: | ---: | --- |
| Capture the corrected target path as a full graph | 43.509 | 42.843 | 0.666 ms | Original piecewise versus full graph, separate server arms |
| Tune drafter attention launch and specialize unit scales | 42.924 | 41.616 | 1.308 ms | Full-graph baseline; same-process control/candidate/candidate/control |
| Use local split-K reduction for down/output projections | 41.521 | 41.081 | 0.440 ms | Full graph plus tuned drafter; same-process control/candidate/candidate/control |

The last candidate contains all three changes and measured about 107.16 t/s
on this workload. Its matched control already contains the first two changes.
The initial comparison of all three against the original baseline was stopped
before running. Adding the three savings is **not** a measured combined result.
The later production-integration comparison is recorded below.
The first comparison also has weaker drift control than the two ABBA pairs.

The power cut reset the GPU to auto mode with a 0 mV offset. Those settings
were retained throughout these comparisons. These results do not measure an
undervolt or compare software under different hardware settings.

## Correctness scope

Each retained candidate matched **320/320 complete full-vocabulary logit
vectors**, including exact top-1/10/20 sets, ordering and retained scores,
plus the initial prefill prediction. Natural timing arms also matched the
output token hash and accepted/drafted token counts. This is sampled evidence,
not a proof for arbitrary inputs.

The untimed forced-token replay requests top-k 129 to force the full BF16 head
and reject masked/truncated vocabulary results. Serving measurements retain
Global-512/top-k 40. The numerical qualification is not a BF16 serving benchmark.

The drafter probe additionally compares complete output bytes on 320 positions
for each of six causal/context cases at 1K, 60K and 200K, with scales 1.0, 0.5,
1.25 and 0.003 and rotating physical windows. Non-unit scales retain their
original multiplication and rounding. The target attention implementation
remains unchanged in the retained combination.

The matrix probe checks complete M1 and M8 outputs and their agreement, with
real checkpoint weights, seeded activations, scratch canaries and injected
fault controls. The retained dispatch admits only M=1/8, N=5120 and
K=6144/17408. Its four K partitions, WMMA accumulation order, ordered FP32
partial reduction and BF16 epilogue are preserved; partials move through local
shared memory rather than global publication and counters. Other shapes use
the original implementation. The existing nine-slot GDN state layout remains.

The disposable worker rebinds the already repaired GDN/attention functions
only while capturing admitted FULL descriptors. This prevents capturing the
different startup-only operators. Its changes are experimental and opt-in.

## Rejected and deferred paths

| Experiment | Outcome |
| --- | --- |
| Packed three-wave target attention | Sample-exact operator improvement, but no established additional whole-model gain. Excluded from the retained combination. |
| Wider attention staging and separate probability/PV passes | Sample-exact outputs and partials, but slower. |
| GDN launch geometry | Exact full states/outputs; projected gain only about 0.042 ms across 48 layers. Not integrated. |
| Separate GDN gate preparation | The exact four-warp version was slower; one-warp normalization changed state/output and was rejected. |
| Target-head graph and full head/sampler graph | Exact sampled checks; no material full-model improvement. The sampler experiment did exercise all 2,178 eligible replays. |
| W4 drafter projections | Target forced logits matched, but proposal acceptance fell and natural throughput decreased to about 97.97 t/s. |
| Head selection/launch sweeps | No worthwhile gain while retaining the selected candidates. |
| Other matrix staging, folding and prefetch sweeps | Mostly slower or too small. An unguarded wider slab changed 21 output elements; the retained builder preserves the original split boundaries. |

These sources remain to make the negative findings reproducible and avoid
repeating failed work. Their presence does not enable them in production.

## Sources and evidence

| File in `experiments/radiance-public/` | Role |
| --- | --- |
| `speed_candidate_worker.py` | Experimental worker, authenticated candidate loaders, paired graph selection and forced replay hook |
| `benchmark_speed_candidate_pairs.py` | Natural-completion ABBA runner; rejects changed output/acceptance or an early length stop and preserves reports atomically |
| `benchmark_speed_forced.py` | 320-position forced replay against a sealed fixture in the disposable lane |
| `probe_draft_attention_tuning.py` | Drafter launch/scale variants, full-output comparison and rotating-window timing |
| `build_mxfp4_register_decode.py` | Builds matrix variants; sweep 9 contains the retained `local_split` implementation |
| `probe_mxfp4_register_decode.py` | Matrix output/state-canary checks and graph timing with weights exceeding cache capacity |
| `probe_attention_memory_tuning.py` | Target attention variants and exact output/FP32 partial checks |
| `probe_gdn_precompute.py` | GDN preparation/geometry trials and complete recurrent-state comparisons |
| `probe_head_launch_tuning.py` | Head projection/selection variants and sampled output comparison |
| `speed_graph_head.py`, `speed_graph_sampler.py` | Deferred graph experiments and comparison gates |

[Aggregate results](../benchmarks/results/speed-investigation-20260925.json)
record the individual arms, qualification totals, actual candidate identities,
and hashes of the private source reports. No chat text, token arrays, hidden
states or snapshots are committed. Private captures and exact tested binaries
remain under the lab's `artifacts/speed-investigation-20260925/` and the remote
state directory of the same name.

The full-head adapter used in the isolated worker is the tracked
`benchmark_d7_equivalence.py`, staged into that disposable interpreter under
the name `speed_equivalence_adapter.py`. Its SHA-256 is
`647971a460eaa5cd54a824f5d205a65636066edd17f1ef141dab0168ba6d4991`.
That distinct import name prevents the frozen release's older adapter from
being silently reused. It is not a second maintained implementation.

To reproduce a matrix trial inside the pinned test image, provide the
authenticated reference source/binary directory and model checkpoint:

```bash
python experiments/radiance-public/build_mxfp4_register_decode.py \
  --source "$REFERENCE_GEMM" --output "$TRIAL/gemm" --sweep 9
python experiments/radiance-public/probe_mxfp4_register_decode.py \
  --source "$REFERENCE_GEMM" --build "$TRIAL/gemm" \
  --model "$MODEL" --output "$TRIAL/gemm/probe-001"
```

The worker's `QWEN_SPEED_TARGET_GEMM_BUILD` must point to that build, and the
loader verifies its source, binary, parent and qualification identities.
Drafter source/generator paths similarly require the matching qualification
report. Serving uses `FULL_AND_PIECEWISE` with the admitted eight-row capture;
this worker rejects unqualified capture domains. The benchmark runners are
for an isolated loopback test lane, not a live multi-chat server.

## Full-graph cache-preparation regression and repair

The subsequent benchmark refresh included the production scheduler, snapshots
and ROCr polling-backoff overlay. Its first combined comparison regressed to
59.44 ms, despite the isolated improvements above. The full-graph execution
branch was missing `after_forward_prepare()`: the installer matched only the
more deeply indented piecewise/eager call to `kv_connector.pre_forward()`.
The scheduler armed the existing transition/recovery fence, but FULL replay
never executed it. The old global idempotence check also prevented repairing
an installation where only the piecewise branch had already been patched.

The repair installs the hook after **each** connector preparation call,
preserving that call's indentation and checking idempotence independently.
Executable regression tests cover FULL, PIECEWISE and eager routes, including
upgrades from the partially patched runner. The hook retains its existing
first-decode and adaptive-recovery conditions; this adds no unconditional
synchronization to every round. Arithmetic and the snapshot data ABI are
unchanged.

| Matched 60K check | Control ms/round | All three candidates ms/round |
| --- | ---: | ---: |
| Before repair, control/candidate/candidate/control | 43.335 | 59.443 |
| After repair, control/candidate/candidate/control | 43.383 | 41.139 |
| After repair, candidates enabled before a fresh chat's warmup | — | 41.121 / 41.513 |

Each natural completion produced the same 4,488 tokens over 1,020 rounds,
with identical output hashes and acceptance (48.5714%). The repaired paired
comparison measures a **2.245 ms combined saving**, about 5.2% of round time.
These are unprofiled measurements on the fixed disposable integration backend,
not a claim that the new candidates have been deployed to Pi.

An independent reproduction on the unfixed integration measured
59.228 ms before one stream synchronization and 40.967 ms afterwards.
A short native trace captured the slow state without first synchronizing it:
the same 1,005 target kernels had mean internal gaps of **13.977 ms** in three
slow graphs versus **3.885 ms** in four recovered graphs. Kernel durations also
changed. These traces diagnose the slow state; their small, instrumented samples
do not replace the stage benchmark. The backoff overlay alone was not causal:
both slow and fast operation occurred with that same library enabled.

This establishes a missing integration hook and verifies its repair on the
reproduction. It does not prove that all underlying HIP/ROCr queue stalls are
eliminated. [Numeric evidence and source hashes](../benchmarks/results/full-graph-cache-prepare-20260925.json).

## Remaining work before deployment

The September 25 refresh passes A/B/A cache-bank handover, disk restore and three verified snapshot-generation replacements, plus all four 320-token whole-model comparisons, the 22-boundary stage replay and the 2,817-check independent operator audit. The shared 0K/60K/200K performance suite and chained tasks also completed; the 0K and 200K coding tasks naturally stopped below the requested 5K length. [Current confirmations](CURRENT_CONFIRMATIONS.md) and [shared-suite method](BENCHMARK_SUITE.md).

Before switching the production launcher to this experimental worker, cancellation and all supported multi-request capture shapes still need targeted integration qualification. The initial tests used a disposable server without production
snapshot I/O or the production ROCr polling-backoff overlay. A production
speedup is therefore not promised by those initial results. The repaired
integration tests include both systems. The hook repair changes the runtime
ABI; it preserves the snapshot data ABI and numerical release.

The normal Pi `optimized_d7_worker.OptimizedWorker` has been restored with
Global-512 and the repaired runtime hooks. Startup authentication passed, and
two non-session warmups each generated 86 tokens; the second observed no new
inference-time compilation. The numerical release and durable snapshot data ABI
are unchanged. The experimental FULL-graph speed worker remains separate from
this production deployment.
