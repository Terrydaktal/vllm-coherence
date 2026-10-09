# Shared benchmark captures

Use `benchmark_pi_coding_contexts.py --suite` when refreshing the workload tables,
context comparison, latency histogram and compiled stage timings together.
The standalone commands remain available for focused measurements.

For each of 0K, 60K and 200K, the suite runs the existing coding task in this order:

1. Warmup.
2. Unprofiled `control_before`.
3. Kernel trace with the existing matched-stage profiler.
4. Unprofiled `control_after`.

`control_after` is selected before any results are available. Its single capture
feeds the coding row, context-length table and histogram. At 60K, that exact
generated continuation is immediately extended with prose about code, JSON,
thinking/prose and compaction, using the same cache identity. The suite then moves
to 200K. It does not regenerate the 60K code for the chained table.

Running the three previous entry points separately required 16 coding generations
(including warmups and traces), plus four subsequent tasks. The combined run uses
12 coding generations plus those four tasks: four redundant generations removed.
These are shared observations, not independent benchmark repetitions.

## Run

Use an isolated `speed_matched_stage_worker.SpeedMatchedStageWorker` server with
the current frozen numerical release, compiled FULL target graphs and Global-512.
It records the target GEMM binary, drafter attention implementation and graph
capture identities. Its analyzer requires one target graph per retained M8
round. The historical PIECEWISE baseline used
`matched_stage_profile_worker.MatchedStageWorker` and required 65 graphs per
round. These profiling workers are not installed into the normal Pi server.
The command validates the live head selection and compiled mode; it does not
start, stop or change a backend. A busy backend is rejected by the request helper.

The output directory and round feed must be accessible to the runner and worker
at the same absolute paths. Use a fresh private directory outside Git checkouts;
it must be mode 0700. Private fixture token IDs are never decoded or printed.

```bash
UV_CACHE_DIR=/data/.cache/uv uv run --locked python \
  experiments/radiance-public/benchmark_pi_coding_contexts.py --suite \
  --output ~/.local/state/vllm-coherence/benchmarks/run-YYYYMMDD \
  --fixture-60k PATH_TO_60K_FIXTURE \
  --fixture-200k PATH_TO_200K_FIXTURE \
  --tokenizer-json PATH_TO_TOKENIZER \
  --runtime-manifest PATH_TO_MEASURED_DEPLOYMENT_MANIFEST \
  --abi SNAPSHOT_DATA_ABI \
  --base-url http://127.0.0.1:8081 \
  --round-log /dev/shm/qwen-stage-timing-rounds.jsonl \
  --head global512 --temperature 1 --top-p 0.95 --top-k 40 --seed 0
```

`--abi` is the storage data ABI accepted by the chat snapshot request guard,
not the separately recorded deployment/runtime namespace ABI. Keep both
identities in the capture; substituting one for the other rejects admission.

EOS remains enabled. The coding prompt targets about 5–6K output tokens, with a
10K safety ceiling. Short natural completions and incomplete round feeds remain
visible validation failures; they are not padded or retried to improve results.

## Outputs and resume

| Path under the private run directory | Purpose |
| --- | --- |
| `checkpoint.json` | Run identity, settings, artifact hashes and completed work |
| `runs/<context>/report.json` | All four arms and their numeric telemetry |
| `runs/<context>/<context>-<arm>/` | Existing worker round boundaries and trace chunks |
| `runs/<context>/<context>-<arm>/diagnostics/` | Private raw round/cache-job/API feeds, recorder health and archive manifest, retained after each arm, including failed arms |
| `diagnostics/chain/<stage>-<attempt>/` | The same diagnostic snapshot after each chained task, including failed requests |
| `continuations/` | Private generated token IDs for resuming the 60K chain, mode 0600 |
| `public/pi-coding-contexts.json` | Context rows and every captured round, including outliers |
| `public/pi-round-histogram.json` | Histogram view of those same selected coding controls, with shared capture identity; no additional requests |
| `public/pi-coding-json-compaction.json` | Chained workload rows; coding has the same capture ID as the context row |
| `analysis.json` | Existing matched-stage reducer's stage timings and paired-control residual |
| `abandoned/` | Preserved partial groups from interrupted attempts |

Repeating the command resumes completed context groups and chain stages. Reuse
requires identical fixture, tokenizer/template, sampling, limits, measurement
source, release manifest and observed worker/configuration identities. A worker
restart requires a new run directory: it changes the cache/warmup conditions.
Missing or changed evidence is an error, not permission to regenerate it silently.
An interrupted profiling group gets a fresh warmup and complete before/trace/after
sequence; its partial evidence is preserved. An active unfinished worker arm must
be closed before resuming. Only one coordinator may own a run directory.

The runner invokes `analyze_matched_stage_timings.py` after capture. A schedule,
output or trace-validation failure stops stage analysis while preserving the
captures. Repeating offline analysis does not require another GPU run. Existing
`package_matched_stage_timings.py` handles publication with the deployment's
`production-inspect.json` and runtime-binding evidence, as before. Publish only
reviewed numeric reports; never publish `checkpoint.json`, continuations or raw
private capture directories.

Publish the coding and histogram views together. Each suite refresh replaces the
histogram view, including while its status is `running`, so a previous focused
histogram-only rerun cannot silently remain beside newer workload measurements.
The renderer distinguishes a shared-suite histogram from a separate focused run.

The packager authenticates the declared source commit against every recorded
measurement and host-runtime source hash. When packaging from an exported tree,
pass `--source-repository PATH_TO_GIT_CHECKOUT` so it can read that commit's Git
objects. Only a complete match produces `source_artifact_commit`; unavailable or
different files remain an explicitly unverified source binding. The measured
source commit is distinct from the later commit that publishes the reports.

## Measurement boundaries

The histogram includes every timed round in the selected coding response. The
stage residual uses **both controls at exactly the retained M8 indices** from the
trace, after output, acceptance and scheduled-shape equality checks. It subtracts
the GPU interval union from those matched control intervals; it does not subtract
stages from the whole-response coding mean. Traced t/s never feeds a workload row.

Controls retain ordinary production telemetry and one host boundary clock/record
per decode round. They have no stage probes or added synchronization. Profiling
can still perturb individual kernel timings; the paired controls expose observer
effects, not a proof that those effects are zero.

The matched timing workers now reclaim completed profiler cycles at each excluded
trace-teardown boundary. After stopping and exporting a trace, the benchmark-only
observer clears its profiler reference and performs cyclic garbage collection.
This prevents unreachable PyTorch profiler objects and their native trace results
from surviving into an unprofiled control. Cleanup duration and collected-object
counts appear in each chunk's `observation.profile_cleanup`; they are not included
in retained stage or control intervals. Collection is never added to control
rounds, and the live worker and its GC settings are unchanged. A stop or export
failure still aborts the arm rather than permitting a control to run after a
failed teardown.

Diagnostic snapshots copy the recorder's existing bounded files while the
container is still available, outside timed requests. Raw request/round identities
and monotonic fields remain available for correlation in the private archive;
they are not added to public reports. The manifest records file hashes, recorder
health and explicit gaps for missing feeds, malformed or incomplete records,
recorder loss, exhausted bounds or unresolved rotation. It is a bounded snapshot,
not a claim that all lifetime records survived. Copies are limited to 32 files,
128 MiB total and 16 MiB per file, with three attempts to resolve rotation.

The first October 9 capture predates these safeguards. Its 591.203 ms 0K and
751.876 ms 60K intervals remain measured outliers with an unproved cause because
their detailed temporary feeds were not archived. A CPU-only PyTorch check
confirmed the profiler lifetime hazard; the histogram-only GPU rerun after the
cleanup captured all 1,091 / 1,383 / 1,714 timed rounds at 0K / 60K / 200K with
maxima of 42.639 / 43.133 / 52.475 ms and none at least 90 ms. This bounded result
does not establish the cause of the earlier pauses or guarantee no future stall.
Its [dedicated numeric artifact](../benchmarks/results/pi-round-histogram.json)
identifies the captured source as base `a16fa00` plus benchmark changes by exact
file hashes. The diagnostic filename-prefix repair was added after this source
was frozen. Existing coding, chained workload and stage timing measurements are
unchanged. The 0K natural
completion remains below the requested 5,000-token minimum. Original per-arm
diagnostic prefix gaps remain explicit; later bounded supplemental copies retain
the correct feeds and match CPU/HIP records to all 4,188 selected timed rounds,
with no missing or invalid selected records. One generic dropped-record count
remains separate from that verified round coverage. All 26 profiler cleanup
operations took 103.33–168.97 ms at excluded boundaries, outside the controls.
Neither the bounded copies nor those joins claim full lifetime diagnostic coverage.

CPU tests cover shared provenance, exact continuation, complete histograms,
interruption recovery, invalidation and privacy boundaries. The September 25
GPU run completed all three contexts and the chained tasks using the speed
candidate plus the FULL-graph cache-preparation hook repair. Its 0K and 200K
coding responses stopped naturally below the 5K target; those length checks
remain explicit failures. All three before/profile/after groups reproduced
their output and accepted-token schedules, allowing matched stage analysis.
The public reports share capture ID `6ec96682eeba444b9c1cddaa0cdb5cc2` and bind
the exact measured source hashes. This benchmark does not itself deploy the
candidate or establish universal numerical equivalence.

## October 9 refresh

The completed shared capture used frozen source
[`99d5fcf`](https://github.com/Terrydaktal/vllm-coherence/commit/99d5fcfd1bcd0fe6f70445d154e63b7092b02f24),
compiled FULL target graphs and Global-512. Its capture ID is
`7210aa09111946adb3d80462bdcfbce4`.
This refresh measures speed. The earlier 320-token arithmetic, independent
operator and cache-state correctness studies retain their original source
identities and dates; they were not rerun for this refresh.

| Starting context | Natural output tokens per coding arm | Timed rounds in the selected clean control | Retained M8 cycles for stage attribution |
| --- | ---: | ---: | ---: |
| 0K | 4,912 | 1,091 | 1,003 |
| 60K | 5,675 | 1,383 | 1,134 |
| 200K | 7,948 | 1,714 | 1,134 |

Every context also retains its first untimed scheduler event. Before/profile/after
outputs and accepted-token schedules matched. The trace budget is 1,152 decode
calls after 64 warmup calls, in 128-call chunks. Only complete M8 cycles enter
the stage means; closing boundaries and the first two trace-activation cycles
per chunk are excluded. The complete control feeds remain the histogram source,
including stalls outside the retained profile indices.

Only the 0K coding completion failed the 5K minimum-length check; its natural
4,912-token stop remains an explicit validation failure. The 60K and 200K arms
passed. All four chained tasks completed, and the 4,730-token checkpoint passed
its section and completion-marker checks. This checks checkpoint generation
and requested cache-tail flushing, not a complete Pi transcript commit.

[Context results](../benchmarks/results/pi-coding-contexts.json),
[chained results](../benchmarks/results/pi-coding-json-compaction.json), and the
[matched stage capture](../benchmarks/results/compiled-global512-stage-profile-20261009.json)
retain numeric records and source identities. Generated text, raw token arrays
and private fixtures remain outside the repository.
