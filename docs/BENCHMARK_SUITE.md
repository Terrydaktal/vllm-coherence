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

Use an isolated `matched_stage_profile_worker.MatchedStageWorker` server with the
current frozen numerical release, its compiled PIECEWISE path and Global-512
target head. To qualify the banked speed candidate, use
`speed_matched_stage_worker.SpeedMatchedStageWorker`: it records the target GEMM
binary, drafter attention implementation and FULL graph capture identities. Its
analyzer requires one target graph per retained M8 round; the PIECEWISE release
requires 65. These diagnostic workers are not installed into the normal Pi server.
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
  --abi SNAPSHOT_ABI \
  --base-url http://127.0.0.1:8081 \
  --round-log /dev/shm/qwen-stage-timing-rounds.jsonl \
  --head global512 --temperature 1 --top-p 0.95 --top-k 40 --seed 0
```

EOS remains enabled. The coding prompt targets about 5–6K output tokens, with a
10K safety ceiling. Short natural completions and incomplete round feeds remain
visible validation failures; they are not padded or retried to improve results.

## Outputs and resume

| Path under the private run directory | Purpose |
| --- | --- |
| `checkpoint.json` | Run identity, settings, artifact hashes and completed work |
| `runs/<context>/report.json` | All four arms and their numeric telemetry |
| `runs/<context>/<context>-<arm>/` | Existing worker round boundaries and trace chunks |
| `continuations/` | Private generated token IDs for resuming the 60K chain, mode 0600 |
| `public/pi-coding-contexts.json` | Context rows and every captured round, including outliers |
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
