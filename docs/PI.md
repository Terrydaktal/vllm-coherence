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
tools/coherence cache -- watch --interval 5
```

The inventory separates current disk usage from cumulative **disk traffic**, and
shows saved token/block coverage, handover/buffered-tail RAM, active Pi processes
and ports. Audit exposes incomplete/duplicate snapshots and failed cleanup. Its
`--help` documents explicit flush and generation-cleanup operations.

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

Start `tools/coherence serve` on the GPU host if its API is unavailable. Retain
failing artifacts when verification fails; prepare into a fresh `--state` rather
than bypassing checks.

A new Coherence installation uses its own snapshot namespace. It does not
automatically import another backend's KV files. State/arithmetic compatibility
must be established before such a migration can reuse model state.
