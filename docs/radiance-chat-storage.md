Radiance chat snapshots now have a stable chat identity, a separate generation
for each committed compaction, and lossless compression. Run
`qwen-radiance-cache list` in a terminal or `/cache` inside Pi. `/cache flush`
forces the current chat's buffered tail to disk. The launcher
`pi-remote-qwen-radiance` loads the storage extension automatically.
See [Pi configuration](pi-radiance-config.md) for its tools and operating preferences.
For block completeness, per-group sizes, missing snapshots, duplicate detection,
and cleanup diagnostics, use `qwen-radiance-cache status`, `show CHAT`, or `audit`.
The [cache CLI guide](radiance-cache-cli.md) explains coverage and watch mode.

The working spinner distinguishes GPU contention from checking reusable context,
finishing the previous response's cache update, loading cached state, allocating
RAM for a handover, moving cache banks, and processing uncached prompt tokens.
Phase transitions publish immediately through one shared event-driven tmpfs/SSH
monitor; ordinary numeric counters still refresh every 0.5 seconds and disk
inventory verification stays at once per second. Existing Pi windows can keep
using the older numeric feed until restarted.

Use `/qwen-timing` inside Pi for the current or last response's numeric timings.
The dismissible view includes the gap after tool completion, HTTP headers, first
streamed output, and backend cache/queue/prefill phases. Preparation, network and
stream delivery are reported together when they cannot be measured separately.
These are elapsed phase times, not isolated GPU kernel measurements. Compaction
also uses backend phase durations, including steps shorter than a display tick.
Diagnostics retain only hashes, counts and durations in memory/tmpfs: at most the
latest response for each of 16 generations, without prompts or tool contents.

The listing shows the chat name or working directory, its identifier, compressed
file size, original cache size, cached request token count, and publication state.
`qwen-radiance-cache list --json` also includes the session file, generation,
update time, file count, and allocated filesystem blocks. File size measures the
compressed snapshot objects; filesystem block accounting can differ because of
Btrfs allocation, compression, or filesystem snapshots. Token counts include the
completed request; the restorable prefix is rounded to the model's chunk boundary.

```bash
qwen-radiance-cache list
qwen-radiance-cache list --json
watch -n 10 qwen-radiance-cache list
```

The command defaults to SSH host `ai`. Override it with `--host HOST`, or use
`--host local --cache-root PATH` before `list` to inspect another cache locally.
The command is installed as a symlink to `scripts/qwen-radiance-cache`.

Pi hashes the saved session's path and UUID to identify a chat. Resuming that
session uses the same identity; a fork or copied session gets its own identity.
The latest compaction entry in the session determines the generation, including
when navigating older transcript branches. Chat and generation also enter vLLM's
cache salt, so a block cannot accidentally belong to another chat's GPU, CPU, or
disk cache. Independent chats consequently do not deduplicate their common
system prompt.

Before Pi is allowed to commit a compaction entry, the compactor forces the old
generation's RAM tail to disk and waits for full-head verification. It then flushes
the committed transcript and its parent directory and retires that chat's previous
disk generation. Retirement
waits for current disk readers and writers, durably changes the active generation,
and keeps the preceding complete head as a fallback while the compacted prompt
creates its replacement. The fallback is removed only after every replacement
block passes checksum verification and the new head commits atomically.
Queued old writes check the generation and cannot recreate it. A failed SSH cleanup
is reported and retried when the next tagged model request activates the generation.
Before that request, Pi flushes the transcript again if necessary. If the model is
not called after a manual compaction, its disk snapshot remains empty until the
next request. A stale client request naming a tombstoned generation keeps its unique
cache salt but bypasses the durable tier, so it cannot reactivate or write that
generation and cannot terminate the shared engine through a connector exception.

Between compactions, immutable full-attention blocks are compressed and written
once under their content-addressed names. The changing settled recurrent/draft tail
contains at most 15 blocks and stays in a private system-RAM journal. A newer
complete tail replaces the older RAM copy; incomplete or failed requests cannot
supersede it. The tier flushes and publishes that tail after at least 8,192 newer
tokens. It also forces a flush before its 6 GiB / five-chat RAM journal evicts a
chat, during clean backend shutdown, before compaction commits, and for an explicit
`/cache flush` request. The general 18 GiB CPU offload cache may evict independently;
the journal keeps its own acknowledged copy so eviction does not require another
GPU prefill.

Until a flush completes, `chat.json` continues to name the preceding verified disk
head. Tail objects are written under new immutable names, the whole candidate is
verified, and only then does one atomic manifest update make it current and collect
the old tail. A failed write or incomplete successor keeps the preceding head and
removes the abandoned candidate blocks after all requests and disk jobs for that
chat finish. The manifest records missing/invalid object keys and the attempted
token count. Collection records durable pending/failed/completed state;
failed deletion retries every five seconds, including while the model is idle.
The engine also collects abandoned writes at startup under its exclusive engine
lock. `qwen-radiance-cache show CHAT` displays publication and collection results;
`GC_PENDING` and `GC_FAILED` remain warnings even when the old manifest is 100% intact.
Publication verification remembers immutable file fingerprints, including across
backend restarts, so normal continuations verify only new or changed objects. RAM
hits request a complete store cascade: a missing immutable block is repaired from
RAM rather than leaving the manifest silently incomplete. A crash discards only the
unpublished RAM tail, so normal recovery computes at most about one 8K interval.
At the current block sizes, the policy is intended to reduce a 250K chat to roughly
15–20 GiB of snapshot payload traffic instead of rewriting a tail after every tool
call; an end-to-end 250K measurement is still required to confirm that projection.

The primary offload arena is 18 GiB, enough for roughly two current long-chat
snapshots at the measured tensor sizes. It remains an LRU cache and does not add
disk durability. The separate tail journal can retain up to five changing tails in
6 GiB of ordinary system RAM and flushes before exceeding either bound. Two
concurrent chats run one complete model response at a time. The GPU changes chats
after response completion or cancellation, so handover can overlap client tool
execution. Thinking, tool-argument generation, and final answers are never paused
to serve another chat. Their allocator metadata stays separate, and one inactive
packed GPU-cache image is held in pinned RAM. A zero-token barrier drains outstanding snapshot stores
before each swap. The pinned image is allocated only when a second chat actually
contends for the GPU, so a single chat does not pay its allocation cost.

Pinned handover buffers apply `MADV_NOHUGEPAGE` to their complete anonymous
backing mappings before any cache transfer, including space reserved beyond the
tensor by the allocator. This prevents background huge-page promotion from
invalidating the HSA registration and pausing the process's GPU queues. PyTorch
still owns and pins the buffers; model arithmetic, cache bytes and disk-snapshot
compatibility are unchanged. Worker status reports `host_page_policy` and
`host_page_policy_bytes`. This policy is set during allocation, with no per-round
scan or syscall, and does not change the machine's global huge-page settings.

Zstd level 1 compresses complete blocks without changing any cache values.
Incompressible blocks use a raw representation. Every object has its original
length and a SHA256 checksum, and decoding verifies the entire block before
writing to the restore buffer. Publication uses temporary files, file fsync,
atomic rename, and directory fsync. The existing FP8 cache precision and model
kernels are unchanged. Samples from all nine legacy tensor groups saved roughly
7–16% with Zstd; higher compression levels provided little benefit. Keeping only
the required recurrent tail saves much more than raising the compression level.

The old cache had no chat ownership records. Its 873 blocks totaling
23,571,726,336 bytes (21.95 GiB) were removed on 7 September 2026 at the user's
request, after verifying that the running backend used the new cache namespace.
Chat transcripts and the active cache were preserved. Existing sessions perform
one prefill to establish a snapshot in their new chat namespace; subsequent
resumes reuse it. Legacy blocks cannot be retroactively attributed to individual
chats. The previous prune utility's deletion modes now refuse to run: its
assumption that groups 0–5 were full attention was incorrect for the actual
model, and timestamps cannot establish whether another chat still needs a block.
Unlabelled requests now always miss the disk tier and their store cascades are
acknowledged without creating `.bin` files.

The storage lives on `ai` under:

```text
~/.cache/qwen-radiance-public-clean-snapshot-v1/snapshots/<ABI>/data/
└── qwen-chat-cache-v1/
    ├── .engine.lock
    └── <chat SHA256>/
        ├── .lock
        ├── .io.lock
        ├── io.json
        ├── chat.json
        └── generations/
            ├── <current generation SHA256>/g<group>-<block hash>.qkv
            └── <temporary verified fallback SHA256>/g<group>-<block hash>.qkv
```

`src/qwen_r9700_lab/radiance_cache.py` implements the object format, locking,
retirement, inventory, and owner-only tmpfs flush command.
`integrations/pi/qwen-radiance-cache.mjs` tags requests, gates compaction cleanup,
and provides `/cache` and `/cache flush`. The existing
Radiance compactor forwards those same tags to its summary request.
`experiments/radiance-public/radiance_chat_tier.py` connects the store to vLLM's
filesystem worker threads. `patch_chat_snapshot.py` installs the tier and two
scheduler hooks during the existing snapshot bootstrap.
`radiance_fair_scheduler.py` implements single-GPU response scheduling and writes
content-free state/transfer telemetry to `/dev/shm`. The shared residency monitor
discovers the snapshot namespace from the running backend's cache mount and serving
arguments, rather than inheriting the first Pi window's snapshot version. Discovery
is cached until the backend's memory-report instance changes; unavailable discovery
retries at most once per ten seconds. All windows continue to share one monitor.
Prefill status shows completed uncached tokens out of the uncached workload, with
reused tokens separately, instead of displaying the fixed input split as progress.
The snapshot tier separately
reports per-chat tail bytes, blocks, durable tokens, and current tokens there; it
never reports cache payloads or transcript text. vLLM's V2 runner has two
request-state slots so one request can remain parked, while the scheduler dispatches
only one sequence to the GPU. Worker telemetry identifies the active GPU bank and
reports both useful bytes and full buffer allocation for each pinned handover image;
the first bank publishes a zero-allocation status before any handover. A different
generation of the same chat replaces its old GPU bank after the store-flush barrier:
the scheduler discards the obsolete bank without copying it to RAM or allocating the
pinned handover arena. Only a genuinely different cached chat activates the RAM swap
path. Allocation time and GPU/RAM transfer time are reported separately, along with
the number of in-place generation replacements. On startup,
the launcher retires exact, unowned offload arenas from prior runtime ABIs after
checking their owner, mode, link count, and live file descriptors, so an old 18 GiB
arena cannot prevent the replacement runtime from starting. The new
`snapshot-abi-chat-cache-v1.json` pins the modules and serving contract; the older
ABI and its data remain separate. After reviewing future runtime changes, run
`python3 experiments/radiance-public/update_chat_snapshot_bindings.py` to refresh
these hashes, then validate and deploy all corresponding modules together.

The [September 13 release upgrade](radiance-release-upgrade-20260913.md) moves to
Radiance 1.0.16 and a separate data ABI. Old snapshots remain available for
rollback; resumed Pi windows use the new namespace after an initial prefill.

The September 8 cleanup repair kept data ABI `74aef307` while authenticating the
updated runtime with a new manifest hash. Tensor layout, model weights, cache-key
hashes, and block encoding are unchanged; both new and already-running Pi clients
continue to use the same data directory. The Mamba manager now retains three prior
aligned state blocks per group alongside its working state, keeping the snapshot's
two settled states alive until final offload. Both normal prefix release and the
align-mode state recycling path honor this bounded tail; allocation is unchanged.
The existing conservative draft lookup and disk retention rules are unchanged.
`qualify_chat_cleanup.py` tests old-runtime disk reuse,
a changed-prefix 208,633-token prefill with 2,519 generated tokens, collection,
another chat's isolation, and long-context disk restoration after a restart.
The [September 8 repair check](../experiments/radiance-public/results/chat-cache-gc-20260908/qualification.json)
passed: the full prefill left no unreferenced blocks, and a restarted backend
restored 207,648 tokens with matching greedy output. Startup recovery removed an
injected orphan; the other synthetic chat stayed unchanged. Both test caches were
removed afterward. The repair also passed 60 focused Python tests, four Pi extension
tests, real filesystem-thread checks, Ruff, ShellCheck, and shfmt.

Validation includes lossless round trips, corruption checks, failed publication,
compaction versus in-flight writes, repeated compactions, preservation of other
chats, scheduler patch idempotence, and the Pi commit hook. The
`qualify_chat_storage.py` script additionally exercises real vLLM filesystem
threads, RAM tail flushing, fallback retention, and restoration through a fresh
manager inside the pinned image. `qualify_tail_journal.py` uses only synthetic token
IDs against the live HTTP endpoint and checks that a second sub-interval turn adds no
snapshot payload traffic before an explicit tail flush.
`qualify_chat_model.py` seeds two synthetic chats, then after a backend restart
checks disk reuse, matching greedy output, and three isolated compaction cycles.

The [recorded model check](../experiments/radiance-public/results/chat-cache-v1/qualification.json)
passed on 7 September 2026. A 10,008-token request occupied 614,895,292 compressed
bytes versus 729,022,464 raw bytes, a 15.7% saving. After a complete backend restart,
6,592 of the 10,000 prompt tokens were restored from disk and greedy output matched.
The remaining suffix follows this runtime's conservative hybrid/draft boundary.
Three successive 4K-token compactions kept exactly one generation, each around
249–251 MB, while the other chat's 249,201,827 bytes remained unchanged. These
model checks used 10K and 4K prompts; they do not constitute a new 250K qualification.
The test snapshots were removed afterward. The final automated checks passed
52 Python tests and 15 Node tests, plus ShellCheck and shfmt verification.

The actual Pi launcher was also checked in RPC mode: `/cache` registered and
displayed the 21.95 GiB legacy inventory before its removal. Start a new Pi
process through `pi-remote-qwen-radiance` to load the new extension and environment.

## Whonix client transport

The Whonix Pi bundle uses the same cache, compaction and progress extensions
through `QWEN_RADIANCE_BRIDGE_URL=http://127.0.0.1:18080/qwen-radiance/control`.
When this variable is absent, host Pi retains its existing SSH transport.
Inside Whonix, cache inventory, tail flush, generation retirement and sanitized
backend diagnostics use the fixed serial relay; an unavailable relay produces
an error and never falls back to guest SSH. The relay consistently namespaces
VM chat identities for both inference snapshots and cache management.

The guest reads one shared telemetry mirror every 0.5 seconds. Its poller shares
the existing host samplers, including the once-per-second temperature probe,
and preserves source timestamps. All VM windows use the same mirror. The
operating prompt and patched Pi runtime are installed in the VM, together with
the VM's bespoke search integration. Deployment is maintained in the adjacent
`opsec` repository's `vm/radiance_bundle.py` and `vm/whonix/install-radiance.py`.
