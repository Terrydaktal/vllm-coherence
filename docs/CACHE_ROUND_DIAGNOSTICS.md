# Cache jobs and isolated slow rounds

The cache-job recorder investigates rare long rounds without changing cache
contents, arithmetic, flush intervals or scheduling. API admission, model warmup
and serving-scheduler hooks each start a process-scoped recorder and are installed
by `patch_chat_snapshot.py`; the release package
includes `qwen_radiance_cache_telemetry.py`. Set `QWEN_CACHE_JOB_TELEMETRY=0` in
the **container environment before startup** to disable it. Editing the source
or restarting Pi does not activate it in an already running backend: the backend
must reload the runtime. The stage profiler is not needed.

## Recorded operations

| Operation | Evidence |
| --- | --- |
| Round start/completion | Hashed request/chat identity, round number, scheduled/input/computed token counts, monotonic timestamp and recorder lifecycle ID. Computed tokens count processed state, not an emitted pending token. |
| Main generation thread | Thread CPU, user/system CPU, minor/major page faults, voluntary/involuntary context switches and wall-minus-CPU time across consecutive completed rounds. This covers scheduler work outside the worker forward. Handover, another request, a different thread or a reset generation clock starts a new baseline. |
| Generation and prefill GPU timing | Two HIP markers bracket `execute_model` entry through `sample_tokens` return. Completed current-stream duration and the previous end-to-next-start gap are collected asynchronously. Prefill records include scheduled and processed ranges; gaps join only adjacent chunks of the same request, generation and stream. These elapsed intervals can include host dispatch starvation; they are not pure kernel busy time. Worker host/thread counters are included separately. |
| GPU transfer submission | Host wall/thread CPU time, thread page faults, job ID, direction, bytes, per-group logical ranges `[first GPU-block index, count]`. These are logical indices, not physical pointers. |
| GPU transfer completion | Native completed HIP start/end duration, submission-to-observed-completion lifetime, success and originating request/round. Lifetime includes dependency/queue waits and completion polling delay; it is not DMA duration. |
| Existing GPU waits | Duration of the existing offload wait and decode-transition/recovery fence, with job IDs/reason. Prefill and first-output results also retain the wall/thread CPU time and page faults of their existing `AsyncOutput.get_output` completion wait. Context belongs to the actual result object, even if results are collected out of order. No new wait is inserted. |
| Startup RAM and chat handover | Individual pinned mappings, huge-page policy, parallel prefault, HIP registration, copy submission, existing device waits and RAM-to-RAM stage copies. Host wall/thread CPU time and page faults distinguish allocation/registration from copying and completion waiting. No device event or synchronization is added. The page-aligned backing size is checked against available RAM; production prepares the pool before readiness. Main-thread counters exclude prefault helper threads. |
| Filesystem jobs | Submission, queue delay, worker wall/thread CPU time, page faults, completion acknowledgement and bytes. GPU and filesystem job IDs occupy separate namespaces. |
| Snapshot processing | Tail and restore RAM copies, compression, decompression, checksum, encoded-buffer copy, file write, file/directory fsync and atomic rename. Payload and metadata writes both appear, distinguished by byte count. |
| Response-end reuse | Local GPU and offload endpoint decisions, explicit rejection reason, endpoint/input/computed token counts, hash block size and missing/pending dependency counts. Ordinary GPU prefix fallback records its matched token count. |
| Prefix construction | Content-free comparisons of generated-token roundtrips, delivered versus incoming assistant fields, template normalization, rendered prompt prefixes and input processing; unsupported, lost or evicted predecessor evidence remains explicit. |
| Shared state | Contended tier/block/manifest/I/O-counter lock acquisition (at least 0.05 ms), slow completion polling (at least 1 ms), tail publication and old-namespace collection. |
| Python GC | Generation, duration, collected/uncollectable counts and executing thread. Existing GC callbacks and collection policy are retained. |
| Recorder health | Dropped records/context, write errors, pending queue size, writer batch time, transfer-hook installation and installed source hashes. Slow recorder batches are recorded so the observer's own work is visible. |
| First-output request timeline | HTTP admission/body receive, render/template work, asynchronous input processing, engine submission, scheduler admission and phase transitions, worker execution/sampling, first generated engine output, first serialized API content, HTTP headers/first body and terminal completion. Explicit hashed-ID bridges connect API and internal engine request identities. |
| Startup lifecycle | Process-scoped recorder start and full worker model warmup before readiness, with monotonic timestamps, process/lifecycle identity and a hashed boot identity. Startup spans are retained separately from request work. |

The qualified TP1 lane runs scheduler and worker in one process. Jobs retain
their originating identity when another chat takes the GPU. Separate `active`
fields identify the chat active **when the event is recorded**, not the owner
of a background job. An operation can overlap several rounds; use interval
overlap rather than that one snapshot. Unsupported process layouts must not be
joined by guesswork.

No prompt text, token values, cache keys, tensors, paths, pointers or exception
messages are included. Logical ranges are capped at 16 groups; truncation is
explicit. GPU job contexts are capped at 256, with evictions counted.

## Long first-output delays

The first tiny request in the [process-policy deployment receipt](../benchmarks/results/process-wide-thp-deployment-20261007.json)
took 61.213 seconds **to complete a nonstreaming HTTP request**. That capture
did not retain a matching request phase timeline or the first generated-output
timestamp. It therefore does not measure 61.213 seconds to first token, or show
which operation consumed the delay. Subsequent request completion times cannot
supply the missing phase attribution retrospectively.

The request recorder now starts at HTTP admission, before body processing and
render/tokenization, rather than relying solely on scheduler admission. It
records points or host spans at the existing boundaries and bridges the random
HTTP request identity to the hashed API and internally assigned engine IDs.
Scheduler phase-enter markers and completed phase spans persist independently
of the footer's latest status. Worker host execution and sampling spans include
prefill and the first generated output. These API and host timeline hooks add
no GPU event, synchronization, tensor read or model request. The separate GPU
timing recorder adds the two asynchronous HIP markers described above; clocks,
thread counters and bounded recorder work also have a cost.

The output boundaries have different meanings:

| Boundary | Meaning |
| --- | --- |
| `first_engine_output` | First engine output containing generated token IDs; its timestamp is retained, with no token values. |
| `first_api_content` | First serialized streaming chunk containing content, reasoning or tool metadata, before yielding it to the HTTP writer. |
| `http_first_body` | First nonempty response body successfully handed to the ASGI sender. An early role-only SSE event can satisfy this without a generated token. |
| `http_end` | Completed HTTP wall time; a cancelled or failed request remains an explicit failed timeline. |

These are server boundaries. Network/tunnel/client buffering after ASGI send
is not measured as GPU generation time. A nonstreaming response may have no
first-content stream marker; its engine first-output boundary remains distinct
from response completion.

Preserve each participating API/engine process's numeric log, rotated log and
health snapshot. All timestamps must share the same hashed boot identity;
the analyzer refuses cross-process subtraction when the clock domain is missing
or differs. An explicit internal-ID bridge is required to join request identities.
Matching chat identity, active-request snapshots and nearby timestamps are not
substitutes. Use any of the recorded HTTP/API/internal request SHA-256 digests:

```bash
UV_CACHE_DIR=/data/.cache/uv uv run python tools/analyze_cache_job_telemetry.py \
  --request-id REQUEST_SHA256 \
  --cache-log /tmp/capture/api-cache-jobs.jsonl \
  --cache-log /tmp/capture/engine-cache-jobs.jsonl \
  --health /tmp/capture/api-cache-jobs-health.json \
  --health /tmp/capture/engine-cache-jobs-health.json \
  --output /tmp/capture/first-output.json
```

Request mode does not need a round log. It reports engine, serialized-content,
first-body and completed-request latencies separately. The attribution window
ends at first serialized content when available, otherwise first engine output.
Every observed span is clipped to that window. `covered_union_ms` merges nested
and concurrent spans; `unattributed_ms` retains the rest of the elapsed wall time.
Their sum equals the latency window. Per-stage overlap times can exceed that
window when added and must never be presented as a serial total.

`INCOMPLETE` is reported for missing/ambiguous start, bridge or output boundaries,
missing terminal completion, failed/cancelled phases, unmatched lifecycle health,
invalid records, sequence gaps, recorder loss or unclosed scheduler phases.
Unfinished phases remain explicit rather than receiving invented durations.
The legacy `status` and `timing_complete` describe the retained timing boundaries
and recorder checks. `prefix_diagnosis_complete` is a separate result: a complete
timing trace can still lack the comparisons needed to explain a prefix rejection.
Neither result certifies every internal instruction. Ordinary rotation and a full
recorder queue can still limit what a later capture proves.

The [7 October activation receipt](../benchmarks/results/first-output-timeline-deployment-20261007.json)
records 270 CPU checks and five complete synthetic request timelines after an
idle restart, with no dropped records or write errors. The first request's
serialized-content latency was 1,779 ms: 481 ms in the first worker execute call
and 1,247 ms in sampling, with 29 ms explicitly unattributed. These nested host
spans are diagnostic attribution, not independent GPU durations. Later requests
reached serialized content in 82–93 ms. This confirms recorder activation; it
does not qualify real-chat restoration or establish that the old delay is fixed.

## Files and overhead

For status prefix `/dev/shm/qwen-radiance-fair-public`, the recorder writes:

- `-cache-jobs.jsonl` and `.jsonl.1`: up to 16 MiB each, retaining one older file.
- `-cache-jobs-health.json`: health snapshot updated about once a second.
- API writers use `-api-PID-cache-jobs.jsonl` (and `.1`) and a matching health
  file. They cannot rotate the engine's log. Startup retains live API writers
  and the two most recent dead-PID sets; older API sets are retired.
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
startup** to disable the CPU/GPU round and first-work measurements while retaining
cache-job diagnostics. The worker hook is installed on the actual V2 runner,
independently of stage profiling. Dummy runs, barrier frames and unowned/batched
calls are excluded. GPU round markers cover at most 16 rows; the host first-work
spans also cover larger prefill chunks until the first generation step completes.
`worker_first_work_hooks` in recorder health confirms that hook installation
succeeded; an early setup failure is retried at most once a second.
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

### Prefix lineage and diagnostic completeness

`prefix_lineage` records compare the previous output with the actual next request
at five boundaries: original tokens versus their decode/encode roundtrip,
delivered assistant fields versus incoming message fields, template normalization,
the rebuilt prompt prefix and input processing. Pi also records context conversion,
payload hooks, final SDK/wire
payload, response identity and assembled output, so a provider-side rewrite can
be distinguished from a backend rewrite. These are comparisons of existing CPU
data; no prompt, token value, tensor or GPU synchronization is added to the log.

The analyzer reports each comparison's status, equality and counts. Token and
character offsets retain their units; an aggregate message-field comparison does
not invent a token offset. Sanitized `template_operations` retain the character
comparison scope and whether a recognized terminal marker was removed. Their
individual results do not substitute for the final normalization aggregate.
A failed comparison identifies an observed
change; it does not, by itself, prove that change caused a particular cache miss.

A recorded prefix-identity rejection requires all five comparisons before
`prefix_diagnosis_complete` can be true. Missing hooks or predecessor data,
unsupported inputs, restart, bounded retention eviction, partial output, recorder
loss and invalid or conflicting evidence remain explicit gaps. A recorded cache
hit can make those comparisons inapplicable; an absent lookup record cannot be
interpreted as a hit. The [focused requirements inventory](../tests/prefix_lineage_requirements.json)
links these boundaries and negative controls to their actual tests. This is a
prefix-reuse diagnostic contract, not a claim of complete hardware instrumentation
or mathematical model equivalence.

Endpoint scope is checked against the lookup's actual processed-token count.
The retained raw output can include an emitted token that was not processed into
the checkpoint. A difference confined to that pending suffix does not explain a
checkpoint-prefix rejection; it leaves an explicit `emitted_unprocessed_boundary`
gap. Missing or insufficient endpoint coverage is `exact_endpoint_missing`.

For client evidence, preserve the Pi process's private
`~/.local/state/qwen-r9700/diagnostics/pi-prefix-lineage-PID-TRACE.jsonl` and `.1`
files and pass each available file as `--pi-lineage-log FILE` to the request-mode
analyzer. The hash of the streamed response ID joins Pi's producer/ordinal group
to the backend's explicit external request ID. Earlier context and wire records
join through that group; chat identity and nearby timestamps are never substitutes.
Client wall clocks may be on a different machine, so no cross-host duration is
calculated from these records.

`pi_prefix_diagnosis` reports client boundary coverage and observed conversion,
payload, serialization, configuration or history changes separately from backend
comparisons. `production_path_diagnosis_complete` requires complete timing,
backend prefix evidence and client evidence for this admitted construction/reuse
route. It remains false when client records, prior output, required hooks, source
identities or recorder health are missing. This result does not cover every model
kernel or hardware event. A lazy token roundtrip can be explicitly inapplicable
when the rendered prefix matches and a covered input-processor change already
identifies the difference inside the checkpoint.

The healthy path retains bounded CPU token arrays and private in-memory message
fingerprints, then checks the actual prefix. Backend diagnostic payload retention
defaults to 32 MiB across at most eight completed chats and eight pending requests;
working copies and Python object overhead are additional. Pi retains at most
32 prior chat fingerprints, with 1,024-message snapshots and a bounded log queue.
The extra tokenizer decode/encode
roundtrip is deferred until a changed prefix needs explanation. This avoids
performing that additional tokenization on every successful reuse; CPU comparisons,
hashing and bounded bookkeeping still cost time and RAM. Observer failure must not
change the request, model output, cache acceptance or cancellation behavior.
Source tests establish the implementation contract, not a native overhead result.
Observer work has its own `prefix_lineage_observer` span and is not attributed as
a cache/prefill blocking operation. An enclosing render span still measures its
actual elapsed time; the analyzer does not subtract an invented observer cost.

Activation requires packaging these modules and provider hooks into the pinned
runtime. An already running backend or Pi process does not acquire new hooks from
a changed checkout. Lifecycle/source bindings and producer health must accompany
the captured request; missing activation evidence is not a successful diagnosis.

### Stable Pi prompt asset paths

Pi's built-in system prompt includes its package's README, documentation and
example paths. In the VM, those paths previously included the runtime bundle
hash. A runtime-only update therefore changed the model input even when the
operating prompt and provider settings were unchanged. The checkpoint correctly
rejected that changed prefix; weakening the prefix check would reuse the wrong
model state.

The VM installer now keeps a validated package-asset anchor for each Pi version
and sets `PI_PACKAGE_DIR` to that fixed target while executing the new runtime.
For existing installations, it preserves the currently used package path, avoiding
another migration-induced cold fill. The anchored release must remain available.
An actual installed-Pi test verifies identical public system-prompt bytes across
runtime updates; unsafe anchors and mismatched package versions fail validation.
The normal host launcher already installs Pi under a stable version directory.

Known observer limitation: a delivered response whose JSON escaping exceeds the
8 MiB fingerprint limit can abort diagnostic finalization after removing the
pending turn but before releasing its callback entry. The production callback
contains no response text and the serving wrapper preserves model output, but
repeated exceptional completions can grow callback bookkeeping and leave the
lineage evidence incomplete. Cleanup on this failure path requires a follow-up
repair and requalification; the normal-path retention limits above are not a
claim that this exceptional path is bounded.

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

### Process-wide huge-page protection

Production startup now applies `PR_SET_THP_DISABLE=1` and requires
`PR_GET_THP_DISABLE=1` before importing release installers or model/GPU code.
The existing qualification supervisor uses the same helper in
`radiance_memory.py`. Linux inherits this policy across fork and exec, covering
the API server and its model workers as well as the dedicated parking buffers.
Failure to set or verify the policy aborts startup. The startup log retains the
numeric policy receipt; `/proc/PID/status` exposes `THP_enabled: 0` for each worker.

This does not change the host's global huge-page settings, model arithmetic,
snapshot data layout or tensor contents, and adds no per-round call. It removes
huge-page promotion as one source of mapping invalidation in this process tree;
ordinary unmapping, registration lifetimes and other driver causes can still
interrupt queues. CPU fork/exec and failure-injection checks establish the
startup behavior. Whether it eliminates the remaining long rounds requires a
matched live capture; allocation and CPU-transfer performance must also be
checked before claiming a performance benefit.

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
Major page faults count serviced faults, not disk traffic: swap may be compressed
RAM, and zero-page swap restoration need not read storage at all.
Generation `gpu_elapsed_ms` is the current-stream marker span, including idle or
dependency gaps within it; it is not a sum of active kernel times and does not
measure unrelated streams. Neither metric by itself establishes causation.

The same bounded asynchronous marker pool also records every owned prefill
chunk as `gpu_prefill`, including its starting computed-token count and scheduled
width. `gpu_inter_prefill_gap_ms` is emitted only between adjacent prompt ranges
with the same request, chat, generation, prompt length and stream. Decode and
prefill gaps cannot bridge each other. Marker queries and elapsed-time reads run
on the existing writer; prefill adds no synchronization or serving-thread wait.
These spans separate queued GPU work from gaps between chunks, while the
existing `worker_execute` and `worker_sample` spans retain host CPU/fault costs.
`worker_get_output` observes the asynchronous result's existing completion wait
and output reconstruction, with wall/thread CPU time and page faults. Its
numeric context is attached to that result, so a later request cannot relabel
it. It introduces no additional GPU query, wait, event, or synchronization.
Marker elapsed time includes any host-dispatch starvation between its markers;
it is not a pure kernel-busy measurement.

The writer also samples host and own-cgroup memory pressure once a second as
`host_memory_pressure`, after the worker hooks are installed. It records available
memory, file/anonymous/shared-memory accounting, swap-in/out, zero-page swap-in,
anonymous refaults and memory-pressure totals. The six bounded reads stay off the
model-serving thread. Missing fields and read/parse failures remain explicit in
each sample and recorder health; unavailable counters are not reported as zero.
These cumulative counters can be differenced across a GPU interval. They identify
reclaim/refault activity, not the exact driver instruction responsible for a wait.

### Prefill pauses from redundant checkpoint file cache

On the 64 GiB AI host, a fresh-process 60K/60K/120K synthetic sequence reproduced
a 76.117-second second 60K prefill. One chunk spent 31.658 seconds in the existing
asynchronous output wait while GPU clocks fell nearly idle. During the request,
the backend cgroup recorded 2,883,378 anonymous refaults: 776,628 swap-ins and
2,106,750 zero-page swap-ins. The host uses compressed zram, so these are not
NVMe reads. A separate first-use 120K request took 158.792 seconds; its matched
repeat took 97.481 seconds without renewed swap-out. Output hashes agreed.

Redundant checkpoint pages contributed avoidable pressure alongside the roughly
38 GiB primary-offload and parked-chat arenas. Snapshot writes, restores **and
publication-verification reads** now advise their copied payload pages as
`POSIX_FADV_DONTNEED`. After target/drafter warmup and before preparing the parking
pool, startup also advises regular safetensors files in the configured local
model directories. No global cache drop, file deletion, model-state change or
precision change is involved. Advice failures are nonfatal and recorded; it is
a cache hint rather than a guarantee that every mapped page can be reclaimed.
The primary offload tier and parked chat images retain their own copies. A later
disk-only restore may read the NVMe instead of an extra OS copy of the same file.

The [matched repair check](../benchmarks/results/prefill-memory-pressure-repair-20261008.json)
repeated the fresh-process sequence with unchanged numerical settings and one
greedy output token per request:

| Cold synthetic request | Before complete repair | Complete repair |
| --- | ---: | ---: |
| First 60K | 33.553 s | 33.025 s |
| Second 60K after chat handover | 76.117 s | 33.390 s |
| First 120K | 98.263 s | 89.848 s |

All three repaired requests recorded zero swap-in/out, zero-page swap-in,
anonymous refaults and cgroup major faults. All 75 prompt GPU intervals and
completion waits were captured, without recorder drops/errors; the 1 Hz memory
samples were complete. Both checkpoint-file groups had zero resident file-cache
bytes before and after the sequence. Output hashes matched the earlier runs.
A separate completed-response continuation reused exactly 4,120 of 4,136 input
tokens and prefilled only its 16 new tokens in 73.271 ms, with first data at
257.210 ms. This qualifies the reproduced failure and cache-reuse paths, not
every future workload or a universal numerical-equivalence claim.
The [deployment receipt](../benchmarks/results/prefill-memory-repair-deployment-20261008.json)
binds those five native requests to the verified installed source hashes and
unchanged snapshot data ABI. Earlier activation receipts remain historical evidence.

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
