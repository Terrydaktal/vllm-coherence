# Development and the private lab

Coherence is the authoritative checkout for the backend, Pi integrations,
instrumentation and current benchmarks. Make and test changes here. The private
`qwen` lab holds captured evidence, private fixtures and historical experiments.
Its shared source files link to Coherence; they are no longer independently
maintained copies. Existing lab command paths continue to work.

## Source and data locations

| Location | Purpose |
| --- | --- |
| `src/qwen_r9700_lab/` | Cache, telemetry, conformance and diagnostic Python modules |
| `experiments/radiance-public/` | Backend integration, correctness probes and performance drivers |
| `integrations/pi/`, `scripts/` | Pi extensions, runtime patchers and launch commands |
| `tests/` | CPU regressions and synthetic Pi integration tests |
| `tools/` | Packaging, publication checks, documentation rendering and lab reconciliation |
| `benchmarks/results/` | Publishable aggregate measurements, never chat contents |
| Private lab `artifacts/` and retained archives | Captures and private fixtures; pass their paths explicitly to benchmark drivers |

Files present only in the lab remain there. Historical launchers are not a new
release path. The legacy `pi-remote-qwen-radiance` launcher retains its remote
host, deployment and cache ABI; the portable Coherence launcher uses its own
deployment. Both select the backend container explicitly for telemetry.

For rare round stalls, use the bounded [cache-job recorder and correlation
tool](CACHE_ROUND_DIAGNOSTICS.md). It records cache/lock/GC timings without
adding GPU waits and retains numeric evidence only.

The legacy launcher's `--reuse-existing` mode authenticates the pinned backend
files on the GPU host and the running container. Local edits to backend source
are not loaded by that mode and do not prevent attaching to the deployed release.
Starting a backend also requires the local backend sources to match their release
manifest. Both modes still authenticate the local ABI contract and Pi dependencies.

Startup inference warmup is cached in the GPU host's private
`$REMOTE_CACHE/startup-warmup/` directory. Subsequent launches skip the synthetic
request and JIT-log polling when the live container, engine/worker process start
identities, model, release ABI, configuration and warmup policy still match.
Backend/worker restarts or configuration changes require another clean warmup;
failed or explicitly skipped warmups never create a success receipt. Health,
model identity and authenticated-release checks still run on every launch.

For the current Qwen model, interrupted reasoning and prose remain in the next
request and compaction input. The transcript still records the attempt as aborted.
Unexecuted tool calls are removed from that request-only copy, with no invented
tool results. Explicit `/context` exclusions and `/purge-thinking` still apply;
failed requests and other models retain Pi's normal discard behavior.

During Radiance compaction, **Esc cancels** and retains the original active
conversation. **Alt+C finishes now** by stopping the current checkpoint stream and
using the summary text already generated. It sends no replacement inference
request. This explicit user cutoff may leave an incomplete summary; it does not
claim that generation finished naturally. The progress display shows
`Esc cancel · Alt+C finish now` while cutoff is available, and Esc remains available
while saving. Repeated Alt+C presses send no additional requests. The normal
snapshot flush, conversation commit and old-snapshot retirement ordering still
applies; the original conversation remains authoritative until commit succeeds.

Normal compaction now appends a bounded plan, source-linked continuity evidence,
fresh Git state, relevant file contents and recorded task handles, and protects
complete tool groups in its recent tail. Loaded project instructions remain in
Pi's separate system prompt. It still makes one checkpoint inference request and preserves the existing
provider prefix. `/context` and thinking exclusions are applied before memory
extraction. Alt+C keeps its exact partial text, with memory stored only in metadata.
See [COMPACTION_MEMORY.md](COMPACTION_MEMORY.md) for budgets, CPU/SDK checks,
known limits and activation.

The bundled [plan extension](PLAN_MODE.md) stores structured state in the selected
session branch. `/plan` enters read-only planning; `/plan execute` resumes
implementation. Normal execution can maintain a plan through `qwen_plan` without
visible plan-file reads. Both host launchers and the Pi-opsec bundle load it.

## Development, verification and publication

1. Edit Coherence directly. Run focused CPU tests here with
   `UV_CACHE_DIR=/data/.cache/uv uv run pytest tests/test_NAME.py`.
   Pi extension tests use `node --test tests/test_NAME.mjs`; installed-SDK tests
   use the pinned Pi runtime and synthetic providers.
2. Build and qualify the candidate using [BUILDING.md](BUILDING.md) and
   [VERIFICATION.md](VERIFICATION.md). Keep private inputs in the lab. Record the
   actual source, configuration and runtime-artifact hashes with the results.
3. Organize reviewed changes into logical commits in this checkout. There is no
   source port back from the lab. A commit/squash changes Git identity; it does
   not establish a new GPU qualification. Reuse evidence only when its complete
   tested source, configuration and binary identities still match.
4. Publish the already qualified runtime artifact with its manifest. Rebuilding
   or changing a dependency/configuration requires the applicable verification;
   source similarity alone does not make it the tested binary.

Linking source does not replace modules already loaded by a running backend or
Pi process. Normal deployment/restart and Pi reload rules still apply.

For a full benchmark refresh, use the [shared capture suite](BENCHMARK_SUITE.md).
It supplies the coding, chained-workload and histogram views from the existing
stage controls, preserving the paired measurements needed for the stage residual.

## Maintaining compatibility links

`tools/reconcile_lab.py` is a CPU-only utility. From the Coherence checkout:

```bash
python tools/reconcile_lab.py check --lab ../qwen
python tools/reconcile_lab.py plan --lab ../qwen --output /tmp/coherence-lab-plan.json
# Review the reported differing paths before applying the plan.
python tools/reconcile_lab.py apply --plan /tmp/coherence-lab-plan.json
```

`plan` records source hashes, modes and existing lab paths. `apply` rejects a
stale plan, saves originals under
`~/.local/state/vllm-coherence/reconciliation/`, then atomically installs relative
links. The printed manifest records every replacement and the original Git
state. It never stages, commits, pushes, restarts a service or touches GPU state.

`check` detects files copied back over links and newly added canonical files that
need compatibility links. Run it after adding shared source files. Prefer editing
from Coherence: some editors replace symlinks instead of following them.

To undo an application, use `restore --manifest /path/to/manifest.json`. Restore
verifies backups and refuses to overwrite subsequent independent lab edits.
Original private data and files outside the selected source trees are never moved.
