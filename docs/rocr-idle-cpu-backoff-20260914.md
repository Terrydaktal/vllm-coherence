# ROCr idle event polling backoff

The idle backend consumed one CPU core in ROCr's
`Runtime::AsyncEventsLoop`: all 149 user instruction samples landed in its
signal-scanning loop while the scheduler had no requests. The main engine
thread was asleep. The exact signal that originally forced polling was not
captured; the event-loop mechanism was confirmed from sampled instructions and
the loaded library's disassembly.

The deployment backports [ROCm change 46558b7a](https://github.com/ROCm/rocm-systems/commit/46558b7af4dc79b8b8014619c1afdb82db079a9f)
onto the image's exact source revision, `2b22ab0195cc1461cd9abf3b969e9dd7c10af350`.
Only the backoff header and the event-loop additions are taken. The older
source's `init_age` bookkeeping remains intact. Other later ROCm changes are
not included.

When a signal requires polling, the loop sleeps for 20 microseconds initially,
doubling up to 200 microseconds when interrupt-backed handlers can share the
thread. The ceiling is 2 milliseconds only when interrupt waiting is globally
unavailable. Each new wait batch resets the backoff. These are requested sleep
durations; OS scheduling can delay wakeup beyond them.

## Build and deployment

The [recipe directory](../experiments/radiance-public/rocr-poll-backoff) contains:

- `async-events-backoff.patch`: the isolated source backport.
- `CMakeLists.txt` and `toolchain/`: build ROCr and its statically linked thunk
  using the pinned image's compiler and existing shared dependencies. The
  small tool imports avoid referring to LLVM development archives removed
  from the serving image.
- `build.sh`: checks the patched source/header hashes, builds with six CPU
  jobs, and runs the CPU backoff checks.
- `poll_backoff_test.cpp`: checks saturation, both ceilings, and overflow
  boundaries with checks enabled in Release builds.
- `async_event_smoke.cpp`: compares a genuinely imported IPC polling signal
  with an interrupt-backed signal on the same runtime event thread. It forks
  before HSA initialization and uses no model or GPU kernels.
- `inference_smoke.py`: three bounded synthetic greedy generations and one
  forced tool call. Reports contain token hashes and numeric metadata.
- `runtime.json`: source, patch, compiler and binary identities used by the
  launcher and runtime manifest.

Source came from the pinned ROCr subtree
`28542580ed0e1104bfc1795cc7f1a6af03eeca52`: all 768 files were checked against
their Git blob identities before applying the patch. Build tools were CMake
3.31.10, the image's Ninja and AMD Clang 23, and Ubuntu's pkgconf 1.8.1. The
build container had no GPU devices or network. Given that source and those
tools on PATH, run `build.sh SOURCE BUILD_DIRECTORY` inside the pinned image;
set `PKG_CONFIG_PATH` to its `core-7.14/lib/rocm_sysdeps/lib/pkgconfig` directory.

The resulting library retains all 276 exported symbols. The launcher verifies
its SHA-256 and mounts it read-only over the image's original library. The
image itself remains available for rollback. The runtime ABI changes to
`5050972c884aefe227c416fc824fb8dae6cc2165c3bd36f7274ffa8e7d70d2d3`;
the snapshot data ABI remains `2ec52392be20945df93b37ee13dc2f605d29bb010b6e0e36d40f5b75d95ac560`.
Model weights, numerical kernels, quantization, and snapshot layout are unchanged.

## Qualification

The imported-IPC test measured:

| Measurement | Original | Backoff |
| --- | ---: | ---: |
| Idle process CPU, one core = 100% | 99.6371% | 0.523704% |
| Polling callback p99 | 6.38 µs | 241.214 µs |
| Interrupt callback p99 in the mixed batch | 0.53 µs | 231.914 µs |
| Largest callback delay | 7.69 µs | 251.635 µs |
| Delivered callbacks | 400/400 | 400/400 |

An earlier CPU-copy test was unsuitable for this mechanism: it exercised a
separate `CpuAgent::DmaCopy` worker's `WaitRelaxed`, so its unchanged CPU usage
does not measure `AsyncEventsLoop`. That result was not used to qualify the fix.

The backoff arithmetic test and 27 existing frontend/ABI/admission tests passed.
ShellCheck, shell formatting, Ruff and manifest/hash consistency checks passed.
The pending production tail was flushed through 195,862 tokens before restart.
Private transcripts, summaries and token IDs were not read.

The short inference checks establish scoped functional regression evidence,
not a long-context performance comparison or universal numerical equivalence.
Baseline request times include concurrent-chat queueing and must not be used
to claim a speed improvement.

After restart, the expected library hash was confirmed in the live EngineCore
and HTTP health returned 200. A two-second measurement with no scheduled
requests recorded 1.0% total engine CPU: the former polling thread used 0.5%
and was sleeping in `hrtimer_nanosleep`; a second thread used 0.5%.

Three 128-token generations and the forced tool call completed after restart.
Greedy output hashes matched within each set of three, but differed between
the original and restarted processes. Exact output equivalence is therefore
**unproved**. Concurrent-chat activity and a fresh compilation namespace were
not controlled in this short smoke check. The difference remains unresolved;
it is neither evidence attributing a numerical regression to backoff nor a
demonstration that the difference is harmless. The numeric
[qualification record](../experiments/radiance-public/rocr-poll-backoff/qualification.json)
retains both hashes and this limitation.

Raw build, callback and restart evidence is retained on `ai` under
`/tmp/qwen-rocr-backoff-20260914`; the deployed binary is under
`~/.local/share/qwen-r9700/overlays/rocr-poll-backoff/38ec92308ff664afec6b7e58299c2b487c5dd849f957633b26fd959cb631bbda/`.
