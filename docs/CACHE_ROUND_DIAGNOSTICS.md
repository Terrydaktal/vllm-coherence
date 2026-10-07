# Cache jobs and isolated slow rounds

The cache-job recorder investigates rare long rounds without changing cache
contents, arithmetic, flush intervals or scheduling. It starts with the serving
scheduler and is installed by `patch_chat_snapshot.py`; the release package
includes `qwen_radiance_cache_telemetry.py`. Set `QWEN_CACHE_JOB_TELEMETRY=0` in
the **container environment before startup** to disable it. Editing the source
or restarting Pi does not activate it in an already running backend: the backend
must reload the runtime. The stage profiler is not needed.

## Recorded operations

| Operation | Evidence |
| --- | --- |
| Round start/completion | Hashed request/chat identity, round number, scheduled/input/computed token counts, monotonic timestamp and recorder lifecycle ID. Computed tokens count processed state, not an emitted pending token. |
| Main generation thread | Thread CPU, user/system CPU, minor/major page faults, voluntary/involuntary context switches and wall-minus-CPU time across consecutive completed rounds. This covers scheduler work outside the worker forward. Handover, another request, a different thread or a reset generation clock starts a new baseline. |
| Generation GPU timing | Two HIP markers bracket `execute_model` entry through `sample_tokens` return. Completed current-stream duration and the previous end-to-next-start gap are collected asynchronously. The gap is included only for consecutive successful rounds of the same request, generation and stream. Worker host/thread counters are included separately. |
| GPU transfer submission | Host wall/thread CPU time, thread page faults, job ID, direction, bytes, per-group logical ranges `[first GPU-block index, count]`. These are logical indices, not physical pointers. |
| GPU transfer completion | Native completed HIP start/end duration, submission-to-observed-completion lifetime, success and originating request/round. Lifetime includes dependency/queue waits and completion polling delay; it is not DMA duration. |
| Existing GPU waits | Duration of the existing offload wait and decode-transition/recovery fence, with job IDs/reason. No new wait is inserted. |
| Startup RAM and chat handover | Individual pinned mappings, huge-page policy, parallel prefault, HIP registration, copy submission, existing device waits and RAM-to-RAM stage copies. Host wall/thread CPU time and page faults distinguish allocation/registration from copying and completion waiting. No device event or synchronization is added. The page-aligned backing size is checked against available RAM; production prepares the pool before readiness. Main-thread counters exclude prefault helper threads. |
| Filesystem jobs | Submission, queue delay, worker wall/thread CPU time, page faults, completion acknowledgement and bytes. GPU and filesystem job IDs occupy separate namespaces. |
| Snapshot processing | Tail and restore RAM copies, compression, decompression, checksum, encoded-buffer copy, file write, file/directory fsync and atomic rename. Payload and metadata writes both appear, distinguished by byte count. |
| Response-end reuse | Local GPU and offload endpoint decisions, explicit rejection reason, endpoint/input/computed token counts, hash block size and missing/pending dependency counts. Ordinary GPU prefix fallback records its matched token count. |
| Shared state | Contended tier/block/manifest/I/O-counter lock acquisition (at least 0.05 ms), slow completion polling (at least 1 ms), tail publication and old-namespace collection. |
| Python GC | Generation, duration, collected/uncollectable counts and executing thread. Existing GC callbacks and collection policy are retained. |
| Recorder health | Dropped records/context, write errors, pending queue size, writer batch time, transfer-hook installation and installed source hashes. Slow recorder batches are recorded so the observer's own work is visible. |

The qualified TP1 lane runs scheduler and worker in one process. Jobs retain
their originating identity when another chat takes the GPU. Separate `active`
fields identify the chat active **when the event is recorded**, not the owner
of a background job. An operation can overlap several rounds; use interval
overlap rather than that one snapshot. Unsupported process layouts must not be
joined by guesswork.

No prompt text, token values, cache keys, tensors, paths, pointers or exception
messages are included. Logical ranges are capped at 16 groups; truncation is
explicit. GPU job contexts are capped at 256, with evictions counted.

## Files and overhead

For status prefix `/dev/shm/qwen-radiance-fair-public`, the recorder writes:

- `-cache-jobs.jsonl` and `.jsonl.1`: up to 16 MiB each, retaining one older file.
- `-cache-jobs-health.json`: health snapshot updated about once a second.
- The existing `-rounds.jsonl` gains lifecycle/monotonic timestamps and input and
  computed context counts; its existing retention is unchanged.

Producers take clocks and enqueue numeric records. JSON formatting and file
writes run on one background thread, draining at most 128 records per 100 ms.
The 4,096-record queue drops instead of waiting when full or contended. GC
callbacks cannot wait on a producer's lock. Diagnostic write failures do not
change cache results or the durability protocol.

The cache-transfer adapter reuses timings from the handler's **existing completed
HIP events**. It adds no transfer event creation, event query, synchronization or
tensor read. Existing blocking calls are measured rather than newly introduced.

Generation timing adds two HIP marker records per measured round. Its 64-pair
pool is bounded; unfinished markers are never recycled, and exhaustion drops
measurements instead of waiting. The writer queries completion and reads timings
every 100 ms; the serving thread never synchronizes or queries these markers.
The last completed pair is retained until the next measurement reads its gap.
An unexpected stream change declines the GPU duration rather than inserting a
cross-stream dependency. Failed driver queries quarantine pairs. Health reports
completed/dropped/error counts and pending/free/quarantined capacity.

Set `QWEN_GENERATION_ROUND_TELEMETRY=0` in the **container environment before
startup** to disable the CPU/GPU round measurements while retaining cache-job
diagnostics. The worker hook is installed on the actual V2 runner independently
of stage profiling; dummy runs, barrier frames, unowned/batched calls and forwards
larger than 16 rows are excluded. An early runner import failure is retried at
most once a second rather than permanently abandoning round diagnostics.
This is the pinned single-request TP1 lane,
not a claim that arbitrary asynchronous/multi-rank streams are covered.

CPU/GIL, resource-counter syscalls, marker submission and writer costs still
exist; throughput impact needs a matched native measurement. This is not a claim
of zero overhead. Slow background marker queries appear as `gpu_observer_poll`.

## Response-end rejection reasons

Every distinct response-end lookup decision is written as a `Response cache
decision:` JSON entry in the ordinary backend log. This remains enabled when
the optional cache-job recorder is disabled, its queue is full or writing fails.
The same fields enter the recorder as `response_end_lookup` events when available.
Identical polling decisions for the same request/source/tier are logged once;
changed reasons or counts are recorded again, including loading-to-hit transitions.

Local decisions distinguish `no_endpoint`, `prompt_does_not_extend_endpoint`,
unsupported request features, disabled prefix caching, unavailable identity,
`cache_salt_changed`, `prefix_hash_changed` and `partial_prefix_changed`. Offload
decisions also distinguish incompatible schema/block/group metadata,
`prefix_identity_changed`, `missing_dependencies` and `pending_dependencies`.
An ordinary GPU-block fallback records `cached_tokens`, making the replay size
recoverable from the input token count even after the live phase row is replaced.

Request/chat identifiers are hashed; prefix hashes, token IDs, message contents,
cache salts and exception text are not logged. Decisions inspect existing CPU
bookkeeping at admission; they add no tensor reads or GPU events/synchronizations.
They do not change acceptance checks or cache ownership. Backend modules must be
reloaded before these additions appear in a running process.

## Handover allocation

`radiance_pinned_memory.py` maps exactly the requested bytes rounded to a system
page, disables huge-page promotion before touching or registering pages, prefaults
large mappings using eight CPU helpers, then registers the memory with the loaded
HIP runtime. `torch.frombuffer` retains the registration through the underlying
buffer object, including tensor views. Existing handover completion fences keep
that storage alive until DMA finishes. Registration failure aborts preparation;
there is no fallback to a larger PyTorch pinned allocation.

The startup worker hook prepares the two parking images and swap stage after GPU
warmup and before API readiness. The first chat switch reuses that pool. A stock
scheduler does not allocate it. The headroom check includes all actual backing
pages and an 8 GiB system reserve. Cache layout, saved ranges, snapshot data ABI
and numerical execution are unchanged. Disk restoration remains a separate cost.

The older first-switch capture allocated 32.25 GiB for 18.858 GiB of requested
buffers and took 86.843 seconds. The exact-size pool took 2.204, 2.206 and 2.064
seconds in three subsequent startups, with 18.858 GiB of backing memory. The
[deployment receipt](../benchmarks/results/exact-pinned-handover-deployment-20261007.json)
records these observations and the synthetic DMA/handover checks. These are pool
preparation timings; disk restore and model prefill remain separate costs.

## Bounded driver events

An explicitly approved capture can use `radiance_kfd_trace.py` when ROCm SDK
attachment is unavailable. On the backend host, pre-create an owner-only capture
directory and set `QWEN_KFD_CAPTURE_PATH` to a new file inside its container path
before starting the launcher. `QWEN_KFD_CAPTURE_SECONDS` defaults to 300 and
cannot exceed 300. Without a destination, no KFD subscription or reader starts.

The first owned, non-dummy forward of at most 16 scheduled tokens starts one
background reader. It subscribes to the process's KFD migration, page-fault,
queue eviction/restoration and unmapping events. It never requests all-process
events, reads GPU tensors or adds GPU launches/events/synchronization. The reader
closes its descriptors after five minutes, shutdown or the 64 MiB capture limit,
then seals the raw events and an identity/source/clock manifest. Recorder health
reports its state and retained bytes. Activation requires a backend restart;
reloading Pi does not enable it.

KFD queue timestamps use the driver's boot clock; start/end boot and monotonic
clock pairs allow comparison with round telemetry. Event formats and other clock
sources must be checked against the running driver. Its finite FIFO can drop
events without a loss counter, so an empty trace does not prove absence of a
driver stall. A separate synthetic registration control recorded five unmapping
events from five unregister operations on this host.

### Captured queue interruptions on 7 October

The [numeric diagnosis](../benchmarks/results/queue-eviction-round-diagnosis-20261007.json)
joins the five-minute process-local KFD capture to 2,050 live generation rounds
at 205,155–213,540 context tokens. No chat text, token IDs or tensors were inspected.
The median round was 51.536 ms. Of 28 rounds at least 65 ms, 26 overlapped driver
queue eviction intervals. Those 26 leave approximately 52.4–62.4 ms after
subtracting the clipped eviction intervals. The other two took 68.277 and
68.583 ms and have no matching recorded eviction.

| Round | Observed round ms | Eviction interval overlap ms | Remaining ms | CPU submission ms |
| --- | ---: | ---: | ---: | ---: |
| 1,838 | 394.146 | 339.180 | 54.966 | 5.248 |
| 2,023 | 363.172 | 308.885 | 54.287 | 4.557 |
| 1,502 | 355.683 | 300.860 | 54.823 | 8.373 |
| 1,484 | 297.649 | 243.752 | 53.897 | 6.059 |

The driver recorded 65 SVM-triggered and 46 USERPTR-triggered evictions, 111
non-retry restore requests and 90 restore retries. Nested evictions are counted
until every request has a matching non-retry restore; an `R` retry does not end
the interval. The current kernel's UAPI marks queue PIDs with a literal `-`
delimiter. The host PID matched the container worker, and boot/monotonic clock
offset drift was 1.24 microseconds.

This identifies memory-mapping invalidation and GPU queue restoration as the
mechanism behind the captured large pauses. KFD stops process queues while
restoring mapped CPU memory; its [SVM implementation](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdkfd/kfd_svm.c)
also records retries when validation cannot complete. The event timestamps mark
eviction/restore requests before the queue operations, so their intervals are
an approximation to the time queues were unavailable, not active kernel timing.

The specific mapping and CPU action that caused each invalidation remain
unidentified. Huge-page promotion remains a candidate: the worker still has
eligible heap/anonymous mappings even though the new parking buffers exclude it.
The trace contains no page-fault/migration records; its finite FIFO prevents
interpreting that absence as proof. No round reached 500 ms in this capture
(maximum 407.783 ms), so the older 2,132.826 ms incident has not been directly
attributed to a driver event. The captured mechanism is a strong candidate for
that older incident, not a claim that all long pauses are repaired.

## Capture and correlation

Let normal Pi work produce around 5–10K rounds: roughly 4–8 minutes at 50 ms
per round. Preserve current and rotated numeric feeds and health before further
rotation overwrites them. No extra model workload or chat inspection is needed.

```bash
UV_CACHE_DIR=/data/.cache/uv uv run python tools/analyze_cache_job_telemetry.py \
  --round-log /tmp/capture/rounds.jsonl.1 \
  --round-log /tmp/capture/rounds.jsonl \
  --cache-log /tmp/capture/cache-jobs.jsonl.1 \
  --cache-log /tmp/capture/cache-jobs.jsonl \
  --health /tmp/capture/cache-jobs-health.json \
  --threshold-ms 65 --output /tmp/capture/correlation.json
```

Omit a rotated input if absent. The analyzer joins process **and lifecycle**,
clips spans to each slow round, and reports per-stage time unions, job metadata
and coverage gaps. Nested/concurrent durations are not a sum of round cost.
Rotation, partial records, drops and jobs spanning a capture boundary limit
attribution. Overlap is evidence for a controlled test, not proof of causation.

A long `gpu_submit` with page faults points towards buffer preparation; a long
existing wait identifies where the engine blocks. A long GPU event duration
indicates device transfer work, whereas long observed lifetime alone can mean
queue dependencies or delayed polling. Copy, compression, filesystem and GC
spans expose other concurrent work. If these do not explain a stall, OS/GIL or
driver tracing may still be needed; not every HIP instruction is instrumented.

`main_thread_round.off_cpu_ms` is wall time minus thread CPU, not an identified
blocking cause: it can include GPU/I/O waits, GIL contention and OS descheduling.
Major page faults show disk-backed page-in activity, not necessarily swap.
Generation `gpu_elapsed_ms` is the current-stream marker span, including idle or
dependency gaps within it; it is not a sum of active kernel times and does not
measure unrelated streams. Neither metric by itself establishes causation.

CPU tests cover snapshot bytes, locks, propagation of original results/errors,
bounded drops/rotation, real writing, GC callbacks, native transfer-result reuse
and lifecycle correlation. Installer/release bindings authenticate the helper.
The [7 October activation receipt](../benchmarks/results/cache-job-diagnostics-deployment-20261007.json)
records the initial cache-job recorder's idle restart with the buffered tail flushed, installed source hashes,
and five startup/activation checks. Its unlabelled synthetic request used 2,229
prompt tokens and stopped naturally after two output tokens. Native GPU
submission/completion records and the round-log correlation worked, with zero
dropped records or write errors. It created no durable test-chat snapshot.

Native cancellation, handover stress and matched observer-overhead measurements
remain separate qualification; this smoke check does not establish them or
identify the rare stall's blocking cause. That receipt predates the generation
CPU/GPU additions and does not qualify or activate them.

The [generation-round activation receipt](../benchmarks/results/generation-round-diagnostics-deployment-20261007.json)
records their subsequent idle restart and a short, unlabelled synthetic request
that stopped naturally after 90 output tokens. The live feed contained 12
main-thread intervals, 12 completed GPU marker spans and 11 inter-round gaps,
with no dropped/error records or unmatched lifecycle joins. The 274 focused
CPU checks preceded activation; the deployment-receipt check passed afterwards.
This verifies the recording and correlation paths, not native observer overhead
or the cause of a rare stall. Restarting Pi is unnecessary for these backend-only
additions.

The [response-end logging activation receipt](../benchmarks/results/response-end-reuse-diagnostics-deployment-20261007.json)
records the next idle restart, 383 passing CPU checks, matching installed source
hashes and an installed CPU logger check. It also activates the first-use
allocation/copy/wait spans described below. The live recorder has the new source
identity; the CPU logger check confirms ordinary logging and duplicate-poll
suppression without a model prompt. The next natural cache lookup supplies the
incident-specific rejection reason. This is not new numerical or observer-overhead
qualification. The local Pi launcher's compatible-release list now travels as
one comma-separated SSH argument, fixing rejection of an authenticated runtime
when its ABI was later in that list.

## First request after restart

The [7 October first-request diagnosis](../benchmarks/results/first-chat-restart-diagnosis-20261007.json)
records 61.058 seconds to first output: 17.950 seconds allocating parked-chat
RAM, 27.506 seconds handing over from the synthetic activation request, 15.111
seconds loading the checkpoint into the CPU tier, 0.242 seconds restoring the
GPU cache and 0.249 seconds prefilling 31 new tokens. It reused 182,527 of
182,558 input tokens. The lookup phase includes asynchronous filesystem loading;
it is not just a manifest check.

The two requested 9.304 GiB parked-chat buffers were backed by two fully
resident 16 GiB ROCm arenas, plus a 0.25 GiB staging arena. Thus the requested
18.858 GiB allocation retained 32.25 GiB of physical host memory. The guard
currently counts requested bytes; that is not a measurement of the larger
allocator arenas. Normal virtual-address reservation would not pay this same
backing/registration cost. Existing huge pages can remain despite the `nh`
mapping flag preventing future promotion.

Only the first bank switch allocates these buffers. Each chat's first use after
a restart may still require disk-to-CPU restoration until its state is resident.
For this request, the filesystem job took 15.091 seconds, including 3.957 seconds
reading, 5.727 seconds decompressing, 2.741 seconds verifying checksums and 1.553
seconds copying to the CPU tier. Fault counts corroborate memory preparation
work but do not identify a particular swap or driver blocking mechanism.

The allocation and handover's internal blocking cause was not captured. New
`pinned_allocation` and `pinned_page_policy` spans separate allocator work from
mapping policy; `handover_copy_submit`, `handover_prepare_wait`,
`handover_copy_wait` and `handover_ram_copy` separate dispatch, existing waits
and RAM staging. All include CPU clocks and fault deltas, retain the original
copy order and wait count, and add no device events. They are prepared for the
next runtime reload; the incident runtime was left running and no reproduction
workload was submitted. This is instrumentation, not a repair of the delay.
