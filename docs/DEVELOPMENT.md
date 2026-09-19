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
