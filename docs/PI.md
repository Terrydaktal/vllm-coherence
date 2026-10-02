# Pi and persistent sessions

Coherence supplies a patched Pi runtime, extensions and matching backend support.
The [Pi feature overview](../README.md#what-changes-in-pi) describes the user-facing
changes; this guide explains setup, counters and session behavior.

## Start or resume a workspace

Run these from the development folder whose history you want to use:

```sh
/path/to/vllm-coherence/tools/coherence pi -- --thinking xhigh
/path/to/vllm-coherence/tools/coherence pi -- --session last
```

The first command starts a new chat. The second resumes the latest session in
that workspace. Transcripts are model-neutral files in `WORKSPACE/.pi/sessions`;
changing models can reuse the transcript, but incompatible model/KV state needs
recomputation. Add `.pi/` to the workspace's `.gitignore` to keep transcripts private.

The launcher verifies the Pi 0.84.2 lock and patches exact known runtime bytes.
It installs to `STATE/pi/0.84.2` and exposes `STATE/bin/pi` as a symlink, preserving
unrelated Pi installations. Reuse checks authenticate the patched runtime again.
Progress, temperatures, tool condensation/rehydration, compaction, cache identity
and priority support load with the operating prompt. `--no-context-files` keeps
AGENTS files out of Pi's context. Internal loop interruption is not enabled by
the release launcher.

For a remote GPU host, use `tools/coherence pi --ssh gpu-host -- --session last`.
The launcher reserves a free forwarding port and uses SSH directly, without
nesting tmux. SSH authentication must already work noninteractively. Supply
`--search-extension /path/to/index.ts` before the `--` separator to use your own
search integration, including one specific to a VM. Terminal keybindings remain
the terminal owner's configuration.

## Read the footer

The footer is one continuous line that wraps to the available width. Its order is
workspace/branch → context → cache → GPU temperatures/fan → Pi usage totals →
model/thinking level. Context usage and capacity use full comma-separated numbers,
followed by the percentage; there is no zero-padding or abbreviated context limit.

| Cache counter | Meaning |
| --- | --- |
| `GPU` | Estimated tokens currently reusable from this chat's GPU cache. |
| `RAM` | Additional cached tokens available from system RAM. |
| `Disk` | Tokens covered by the current verified disk snapshot. This is durable backup coverage and may overlap GPU/RAM, rather than only the tokens that must be read from disk next. |
| `Cold` | Estimated tokens not covered by reusable GPU/RAM/disk state that need model computation. |

Consequently, **do not add all four counters together**. A memory-only tail can
make GPU coverage exceed Disk coverage. Compaction changes the generation being
reported; saved coverage for the previous generation is not credited to the new
checkpoint. Unknown or stale measurements appear as unavailable, not zero.

The two temperatures are **junction, then edge**, followed by fan percentage.
One file-locked temperature probe per host/state serves all windows once per
second. Scheduler/cache metadata refreshes every 0.5 seconds without copying GPU
buffers. Pi's existing input/output/cache-usage totals follow the temperatures.

## Read the working spinner

The spinner follows the request's actual backend phase when telemetry is present:
preparing/submitting → admission or GPU queue → handover/cache lookup/restore →
uncached prompt prefill → reasoning, answer or tool-argument generation → tool
execution. Phases that are unnecessary can be skipped.

Queue messages identify the blocking chat and what it is doing. Allocation of
handover RAM, loading cached context, finishing a cache update and computing
uncached tokens have separate labels and timers. Waiting for admission does not
mean a cached prompt is empty; reuse is reported once observed. If telemetry is
missing, the display says the wait is unknown rather than guessing.

Prefill progress is published after each completed GPU chunk, using the same
phase feed for ordinary turns and compaction. A chunk that has only been
scheduled is not counted as processed. Pi reads the feed every 100 ms; the count
stays still while a chunk runs, then advances by the completed amount. Bulk cache
and scheduler snapshots retain their half-second cadence. None of these updates
adds GPU synchronization or interpolates an estimated processed-token count.

During generation, `x t/s, y t/s avg` means a three-second rolling rate followed
by the average after first data. The rate clock excludes observed waits for
another chat, while elapsed time continues to show the full wait. `first data`
is the time from starting the provider request to its first observed output,
including any admission, restoration or prefill before that output.

Usage updates continue while tool arguments are buffered. The spinner distinguishes
`generating edit arguments` from `applying edit`, and times the tool execution.
Reasoning counts appear only when a positive count is known. Backend wait/tool
states replace the generation-rate display when the model is no longer emitting
output.

## Purge earlier thinking, retain future thinking

This is useful when runaway thinking has poisoned a chat with repetitive
reasoning that later turns keep echoing. Purging those earlier thinking blocks
removes that loop-reinforcing history from future model inputs while retaining
new thinking.

Run `/purge-thinking` while the chat is idle to exclude its current thinking
blocks from subsequent prompts. New thinking is retained from that point on,
including across later user messages. The command overrides
`preserve_thinking=false` for this chat branch without changing the global model
configuration or whether the model generates fresh thinking. Use
`/purge-thinking status` to inspect the saved policy; running the purge again
excludes thinking accumulated since the previous purge.

The cutoff is saved as session entry IDs, so it survives restart, resume and
compaction. Normal requests and checkpoint generation apply the same filter.
Prose, tool calls, tool results and user messages remain in context. Original
thinking remains in the saved transcript; existing compaction summaries are not
rewritten. Navigating to a branch before the purge restores that branch's policy.
The next request may need to prefill the suffix after the first removed thinking
block. Later requests can reuse the new prefix normally. No backend restart is
required.

## Choose what stays in context

Run `/context` while the chat is idle to choose individual user or assistant
messages, whole turns, or a range of older messages. This changes what the model
receives without deleting anything from the saved transcript. Excluded messages
remain visible in the picker so they can be restored.

| Key | Action |
| --- | --- |
| Up/Down, Home/End, Page Up/Down | Move through messages, oldest first |
| Space | Select or deselect the highlighted message |
| `t` | Select or deselect the highlighted message's whole turn |
| `r`, move, `r` | Select a range |
| `o` | Select everything from the oldest message through the cursor |
| `a`, `c` | Select all / clear the selection |
| `e`, `s` | Exclude from context / restore to context |
| `h`, `H` | Exclude / restore selected assistant thinking only |
| `u` | Undo the last pending change |
| `p` | Count the exact rendered prompt through the CPU tokenizer |
| Enter, Escape | Save / cancel all pending changes |

Actions use the highlighted message when none are selected. Selecting a tool call
or result for exclusion includes its entire assistant message and all of that
message's tool results. Restoring any member restores the group. Thinking-only
changes leave the calls, results and prose intact. An explicit thinking restore
can undo an earlier `/purge-thinking` decision for that message; purging again
excludes it again. Future thinking remains retained on the Radiance provider.
An interrupted call with missing recorded results must stay excluded until those
results exist; the picker does not synthesize tool results to make it fit.

The picker shows approximate message token counts, clearly excluding the system
prompt, tool schemas and template. Press `p` for exact before/after prompt counts,
including those components. Previewing performs tokenization without generation
or GPU prefill. An unavailable tokenizer leaves selections unchanged.

`/context undo` reverses the last saved picker change; repeat it to step back
through earlier changes. `/context status` reports the current branch's active
exclusions. Decisions are saved as entry IDs, survive reload/resume and apply to
both ordinary requests and checkpoint generation. Navigating to an earlier
branch uses that branch's decisions. Earlier history already represented by a
compaction checkpoint is not available as individual messages in this picker;
restoring an old decision cannot unpack an existing checkpoint.

The first request after a change may need to rebuild the changed cache prefix.
The input draft is not cleared. Invalid saved decisions or unsafe tool pairing
block the provider request rather than silently restore excluded history.
Restart Pi once after installing this feature to load its context-safety runtime
patch; `/reload` alone cannot replace a running runtime. No backend restart is
needed. On `pi-opsec`, exit Pi with `/quit` and reconnect; detaching its terminal
leaves the old Pi process running.

## Compaction transaction

Manual `/compact` and automatic compaction use the same checked transaction:

1. Prepare the checkpoint prompt and verify that the historical token prefix is
   unchanged, allowing cached context to be reused.
2. Calculate the available context capacity for checkpoint output. There is no
   separate fixed summary-token cap; the model's remaining context is the limit.
3. Submit the request and time admission, GPU queue ownership, handover, cache
   lookup/restore, prefill and checkpoint generation separately when observed.
4. Validate required sections, the completion marker and finish status, then
   save the checkpoint and flush the pending KV tail.
5. Commit the conversation's new generation before retiring old snapshots.

A failed validation retains the original transcript. Cleanup failures are shown
and retirement is retried on a subsequent request. Compaction is a model-generated
summary; transactional safety does not make that summary semantically lossless.

The progress widget updates in place, with checkpoint-token counts, a three-second
generation rate and per-phase timers. It disappears when finished; the durable
`Compacted from … tokens` entry records total time. Text already typed in the
editor is preserved across completion of compaction.

## Snapshot reuse and disk writes

Snapshots bind chat identity, compaction generation, token prefix and compatible
runtime identity. The backend reuses compatible GPU state, restores parked RAM
state or loads compressed disk state before computing the missing suffix. An
incompatible snapshot cannot substitute for fresh computation.

Immutable full-attention blocks are written once and reused. The changing tail
stays in GPU/system RAM and normally flushes after roughly 8,192 new tokens, or
before RAM eviction, clean shutdown, successful compaction or an explicit flush.
The previous complete disk head stays valid until its replacement is verified.
A crash can require recomputing the unflushed tail; clean shutdown must be allowed
to finish flushing it.

Compaction supersedes the old generation rather than accumulating generations
indefinitely. Stale requests cannot reactivate retired generations, and the old
GPU bank is discarded without a redundant save to handover RAM.

On the GPU host, inspect storage with:

```sh
tools/coherence cache -- status --details
tools/coherence cache -- audit
qwen-radiance-cache purge-tests --dry-run
qwen-radiance-cache purge-tests
tools/coherence cache -- watch --interval 5
```

The inventory separates current disk usage from cumulative **disk traffic**, and
shows saved token/block coverage, handover/buffered-tail RAM, active Pi processes
and ports. Audit exposes incomplete/duplicate snapshots and failed cleanup. Its
`--help` documents explicit flush and generation-cleanup operations.

`purge-tests` removes snapshots labelled **Synthetic release smoke** or
**Synthetic relay probe** in a qualification directory. It skips tests with
active requests, a GPU/RAM bank or a pending snapshot tail, as well as busy chat
locks; real chat snapshots and transcripts are preserved. A tiny deletion marker,
lock and write counter prevent late writes from recreating purged generations.
An interrupted purge can be retried safely.

The header reports **lifetime disk traffic** across every data ABI, including
deleted chats. It combines each chat's durable `io.json` counter with numeric
history in `snapshot-retirements.json`, counting overlapping copies only once.
Deleting a test cache, retiring an ABI or compacting a chat does not reset this
total. Counts start when write tracking began, exclude metadata/unfinished writes,
and cannot recover unrecorded history from a cache deleted manually. Reading the
history adds no new recording or synchronization to backend generation rounds.

## Concurrent chats and priority

Normal scheduling lets the active chat finish a model response. When it invokes
a tool, a two-second grace period allows a fast result to continue on the same
GPU bank; a longer tool gives a waiting chat a chance to run. This reduces
handover delays during short tool calls. A handover preserves compatible cache
state in RAM when available, and each waiting window reports the owner/phase.

`/priority` shows the chat's setting. It defaults to 0 and is saved per chat:

- `/priority 0`: normal scheduling.
- `/priority 1`: acquire at a normal response/tool boundary, then hold the GPU
  through tools until the answer finishes, ahead of lower-priority chats.
- `/priority 2`: request preemption of a lower-priority owner at the next safe
  GPU step, then hold through tools until the answer finishes.

Equal priorities retain normal scheduling. An answer includes all its provider
requests and intervening tools. Ownership is released after the whole answer
settles, and does not end at each individual tool call or compaction.

## Tool-output archives and backend errors

Oversized tool outputs get bounded context views after the full result is saved
to an authenticated archive. `qwen_rehydrate_tool_turn` retrieves exact line ranges
or literal matches, preserving access to evidence without resending every large
tool output on every request.

Recorded EngineCore errors are attached as expandable diagnostics: `Ctrl+O`
reveals the traceback, and `/backend-error` retrieves the latest recorded failure.
A missing traceback is reported explicitly; a forced kill may leave none.

Completed backend requests also append one content-free termination record to
`/dev/shm/qwen-radiance-fair-public-stops.jsonl` on the GPU host. It captures the
core status before cleanup: EOS, configured stop token/string, output/context
limit, abort, error, repetition, or an explicit unknown cause. Records contain
hashed request/chat/generation identities, the completion timestamp, token
counts, limits, sampling settings and final round/acceptance counts. They never
contain token IDs, stop strings, prompts, answers or tool arguments.

To inspect recent terminations without opening a transcript:

```sh
jq -s '.[-10:] | map({finished_at_ms, request_id, chat_id, cause,
  finish_reason, output_tokens, total_tokens, generation_rounds, sampling})' \
  /dev/shm/qwen-radiance-fair-public-stops.jsonl
```

The last termination per generation is also in the completion-only
`qwen-radiance-fair-public-stops-status.json` file, with at most 16 generations.
Its hashed request ID and `finished_at_ms` correlate with
the existing round log to establish whether a response ended before another chat
took the GPU. This explains the backend's stop decision; it does not establish
why the model chose EOS or whether an answer is semantically complete. `aborted`
is the core status and can include an upstream stop-string decision, not only a
client cancellation. API/detokenizer-only causes and a crash before request
cleanup are not inferred from a missing record.

The recorder does no per-token work, GPU readback, synchronization or filesystem
sync. It writes only at completion, using existing CPU metadata. The RAM-backed
log rotates at 1 MiB, retains one previous file and creates files with mode 0600;
it is not durable across reboot. Pending records are bounded to 64. If a capture
or write fails, inference continues and the stop-status file reports
`stop_capture_failures` or `stop_log_dropped`; failed writes do not trigger retry
work on subsequent generation rounds. The existing phase feed remains unchanged,
so currently running Pi and VM readers keep their round/acceptance counters.
A running backend needs the normal
runtime-package update and restart to load this recorder; Pi needs no reload.

An incompatible retained GPU endpoint can no longer prevent a replacement prompt
from being admitted indefinitely at “Checking reusable context”. If allocation
fails after endpoint reuse was rejected, the scheduler releases that optional
checkpoint's pins and retries. Matching continuations, outstanding copies and
transfer-owned pages stay protected; disk snapshots are unchanged. The numeric
cache-pressure record identifies when an unused endpoint was released.

Connection failures also retain an expandable report in the session. It captures
the original transport error codes before the SDK replaces them with a generic
`Request timed out.`, plus the endpoint, attempt duration, response-header timing
and bounded cause chain. `Ctrl+O` shows those details automatically beneath the
failed response. The report is a custom session entry, excluded from model context;
it contains no request/response bodies, headers, URL credentials or free-form
exception text. Local JSON inference requests retry one failed TCP connection
before any HTTP request is sent. Header/stream timeouts, resets, HTTP errors and
consumed request bodies are not replayed; cancellation also stops the retry.
If both connections fail, the report retains both attempts. Restart Pi
after installing the provider patch; `/reload` alone does not reload that module.

Start `tools/coherence serve` on the GPU host if its API is unavailable. Retain
failing artifacts when verification fails; prepare into a fresh `--state` rather
than bypassing checks.

A new Coherence installation uses its own snapshot namespace. It does not
automatically import another backend's KV files. State/arithmetic compatibility
must be established before such a migration can reuse model state.
