# Compaction memory and continuation

Coherence keeps Pi's original JSONL session as the archive and replaces older
active history with a checkpoint plus a recent message tail. The handoff includes
source-linked evidence, a persistent task plan, current referenced files and
repository state, and historical execution reminders. It
uses one existing Qwen checkpoint request, with no second model or background
GPU work.

## What survives the handoff

1. **Working-state checkpoint.** The model leads with the immediate objective,
   latest correction, unfinished action and next supported action. Separate
   sections preserve active constraints, decisions, measured results, rejected
   approaches and uncertainty. Subsequent checkpoints update this ledger from
   the previous checkpoint and recent observations, explicitly marking
   superseded conclusions.
2. **Source evidence.** A deterministic packet quotes up to three recent user
   requests and one assistant statement, and records up to four completed tool
   groups. Quotes are bounded exact excerpts labelled with source entry IDs;
   tool records contain names, result IDs, error flags and text hashes, without
   treating a successful command as proof that the task is correct. A prior
   checkpoint gets a recovery pointer and hash, rather than a second copy of
   its summary. The packet is supplied to the summarizer and appended to a
   normally completed checkpoint.
3. **Original recent messages.** The largest recent suffix fitting the configured
   `keepRecentTokens` estimate is retained, up to one quarter of the context
   window. A tool call and all its results are an indivisible interval, including
   interleaved calls. Message bodies and transcript entries are not rewritten.
4. **Persistent plan.** The selected branch's structured goal, ordered steps,
   constraints, notes and source/evidence IDs survive in session metadata. A
   bounded excerpt prioritizes unfinished steps. Completion remains
   model-reported. Normal task tracking does not require read-only planning or
   approval; `/plan on` explicitly enables read-only planning. See
   [PLAN_MODE.md](PLAN_MODE.md) for commands and recovery.
5. **Execution reminders.** Selected tool metadata can preserve recorded
   `session_id`/PID handles and unmatched calls, with source IDs and explicit
   `host`, `vm` or `unknown` scope. No process is inspected; every handle remains
   unverified, including after resume. An unmatched call means its result was
   not observed, not that a process is running. The pinned Pi 0.84.2 builtin Bash
   results do not expose persistent process handles; adapters must supply
   structured metadata.
6. **Fresh repository state.** A bounded, read-only Git status captures the branch,
   HEAD and tracked-file changes in the Pi workspace. It does not run tests or
   claim work is complete. Non-repositories, oversized results, configured
   clean/process filters and timeouts are recorded
   as unavailable; they do not prevent compaction. A config preflight and status
   share a 750 ms deadline. Repository filters, filesystem monitoring and optional
   Git writes cannot run through this capture.
7. **Referenced files.** At most five files are considered, starting with explicit
   plan paths, then recent successful edits/writes and reads from complete tool
   groups. Only regular UTF-8 files inside the active workspace are eligible.
   Each included body is complete and carries its exact byte SHA-256 and source
   entry IDs. Large files and bodies that do not fit become explicit path
   references with provenance and bounded targeted reread hints, rather than
   silently truncated content. Missing, changed, binary, inaccessible and unsafe
   files are reported. Credential/dotenv/key paths are references by default.
   There is no directory scan or automatic reopening of archived transcripts.
8. **Historical recovery.** `pi_session_search` retrieves original entries using the
   recorded IDs or keywords. Instructions tell the resumed model to retrieve
   missing evidence before repeating completed work, reversing a decision, or
   asserting an unverified historical result. See [pi-search-tools SESSION_SEARCH.md](https://github.com/Terrydaktal/pi-search-tools/blob/main/docs/SESSION_SEARCH.md).

The entire restoration appendix shares a maximum of **20,000 characters**,
reduced when context headroom is small. Headers, quoted paths, source IDs,
digests, JSON escaping and omission notices count toward that limit. Plans,
execution reminders, source evidence and Git state consume their bounded shares
before files use the remaining space, up to 10,000 characters. Reports retain
explicit omissions when even a reference cannot fit. File capture normally caps
each file at 64 KiB and total file reads at 256 KiB, with a shared 750 ms file
deadline; cancellation or timeout ends its worker before returning.

Source evidence has a ceiling of 7,200 characters, about 1,800 estimated tokens.
Restoration and tail estimates use characters/4, not exact Qwen token counts.
Exact server tokenization still checks the checkpoint request and its remaining
output allowance. An oversized newest message or tool group is
reported in `details.continuity.tail`; when no complete suffix fits, Pi's prepared
boundary is retained. This fallback can exceed the requested tail estimate;
there is no silent slicing of a message or a fabricated tool result.

The completed notice's **Compacted from N tokens** uses the exact tokenized
selected history after `/context` and thinking exclusions, including the existing
system prompt, tool schemas and template. It excludes the new checkpoint
instruction and generation suffix. This count is already obtained during
compaction; no extra tokenization or inference is needed. The old preparation
estimate remains in `details.preparedTokensBefore` for diagnostics, with
`tokensBeforeSource=selected_history_tokenization`. Authenticated recovery
receipts use their saved historical count without rewriting the receipt.
When reopening older Radiance compactions, Pi's display also uses that recorded
historical count when valid. The original JSONL entry remains unchanged.

## Checkpoint prompt reuse

Snapshot-backed compaction prepares its prompt through the authenticated
`POST /qwen/prepare-compaction` endpoint. It uses the normal chat renderer and
the same generated-token journal as ordinary replies. The selected history
receives the same context exclusions, plan filtering and tool-name migration
before provider conversion. Only the checkpoint instruction and its closed
thinking header are appended to that history.

Previously, compaction separately re-tokenized all history through `/tokenize`.
Decoding generated IDs to text and encoding that text again can produce different
IDs, even when the text is identical. Bypassing the normal continuation path
could therefore invalidate the whole KV/recurrent prefix. The new preparation
preserves authenticated historical IDs when the history, output, configuration
and rendered text still match. Changes to old messages remain real cache misses;
the endpoint never treats different token sequences as interchangeable model
state. A cancelled answer does not become a completed journal entry, but an
earlier valid completed prefix can still be reused.

Preparation performs no inference, GPU wait or journal commit. The normal cache
checks still determine how much prepared state is reusable. Its receipt records
the preparation reason and the number of preserved IDs; that count is not a
claim that all of those tokens are resident in the GPU. A missing endpoint or
invalid preparation stops compaction with the original transcript retained,
instead of silently falling back to a full-history prefill. Chats without a
snapshot identity retain their prefix-checked canonical tokenization path.

The serving process must load the updated preparation endpoint, and Pi must
load the updated extension. The snapshot data format and model arithmetic are
unchanged.

The [27 October qualification](../benchmarks/results/compaction-prefix-preparation-20261027.json)
checks the actual installed Pi SDK and pinned Qwen tokenizer with two different
token encodings of the same text. The unchanged generated prefix survives;
editing an earlier message rejects that reuse. A native streaming-chat check
then reused 3,042 of a 3,064-token checkpoint prompt, processing only the 22-token
suffix in a 77.5 ms one-output-token request. This is a small cache-reuse check,
not a full-length summary-generation benchmark. The recorded 235,369-token
incident had reused zero tokens and spent 261 seconds prefilling. Its first
changed token was not retained, so the capture does not identify which private
history transformation first broke that prefix. The VM relay and installed
Pi-opsec bundle also passed preparation and SDK checks.

## Automatic compaction during generation

The Qwen/Radiance harness watches fresh provider usage while an answer is
streaming. The threshold is
`min(window - reserve, max(240000, window - 13792))`, where `reserve` is
Pi's configured `compaction.reserveTokens`.

With a **327,952-token** context window and the default **16,384-token** reserve,
streaming compaction triggers at **311,568 tokens**.

The existing checks between tool batches and completed answers still apply.
Disabling automatic compaction also disables the streaming guard. The guard
applies to the configured Qwen/Radiance model.

The interrupted assistant message is saved to the original session JSONL before
compaction begins. Its already received thinking and prose remain intact, with
an interrupted status. Unfinished tool calls remain archival data and are never
executed or given invented results. Completed messages and tool results earlier
in the turn remain in the archive. The active context receives the checkpoint
and a bounded recent tail, rather than the entire potentially enormous turn.
If the newest interrupted message exceeds that tail budget, it is summarized
instead of retained whole in the active context. The original message remains
intact in JSONL and can be recovered with transcript search.

After the stream has stopped and the checkpoint has committed, the harness
continues the original task without inserting a user message. The transcript is
flushed before checkpoint generation and again before automatic continuation.
Cancellation or
failed compaction leaves the saved partial answer in place and does not
automatically restart generation. The guard checks existing streamed usage; it
does not add tokenization, inference, GPU synchronization or per-token disk
writes. A provider that does not report fresh usage cannot provide this measured
streaming limit; the normal boundary checks still apply. The reserve provides
headroom, not a guarantee that a model will obey the checkpoint length target.

## Scope and transaction safety

Only Pi's selected, active context is used. `/context` exclusions and
`/purge-thinking` apply before evidence extraction. Summarized ancestors,
sibling branches and excluded messages are not reopened to populate source or
execution evidence. Source-linked plan fields are filtered against exclusions,
including compacted ancestors; permitted explicit plan paths can remain useful
after their original messages are summarized.
Thinking is not copied into its source quotes. The recent original tail follows
the existing thinking policy. Archived source strings are JSON-quoted, with
chat-template controls escaped, and labelled as historical data.

Loaded project instructions remain in Pi's existing system prompt. Restoration
records do not replace or duplicate that instruction source. The canonical
provider history remains the existing token prefix; the memory instruction is
appended after it. The normal ten-section and completion checks,
authenticated token accounting, snapshot flush, transcript append and old-cache
retirement remain in place. A receipt is bound to its source session/leaf,
packet digest, tail boundary, restoration digest and repository-state digest.
A fresh capture timestamp alone does not invalidate a reusable receipt. Recovery
receipts are capped at 2 MiB; that disk limit does not increase the context budget.

**Esc cancels. Alt+C accepts the exact text already generated.** In Termux, tap
the extra-row ALT key, then C; no function key or extended keyboard protocol is
needed. The progress display shows `Esc cancel · Alt+C finish now` once checkpoint
text exists. The finish action is inactive outside its current compaction.
Alt+C starts no replacement request and appends no restoration text to that explicitly chosen
partial summary. Continuity metadata remains attached to its session entry for
inspection, with `appendedToSummary=false`; the appendix is not active model
context in that case. The independently saved plan remains recoverable by its
extension. Prefer natural completion when the restoration appendix is needed.

## Verification and limits

The CPU tests use synthetic histories and the real pinned Pi SDK. They check
atomic tool tails, bounded excerpts, mutation-free persistence, repeated
compaction, branch isolation, exclusions, receipt binding, exact Alt+C behavior,
plan replay and read-only gating, unverified execution metadata, safe file reads
and shared restoration budgets. They make no inference requests to a real model and
read no private chats.

```bash
node --test tests/test_compaction_memory.mjs tests/test_compaction_workspace.mjs \
  tests/test_compaction_files.mjs tests/test_compaction_tasks.mjs \
  tests/test_compaction_restoration.mjs tests/test_task_plan.mjs \
  tests/test_compaction_continuity_sdk.mjs tests/test_radiance_compaction.mjs \
  tests/test_radiance_compaction_finish.mjs tests/test_radiance_compaction_sdk.mjs
```

The requirements inventory is
[`tests/compaction_continuity_requirements.json`](../tests/compaction_continuity_requirements.json).
Semantic retention still depends on the summarizing model. Recent excerpts do
not mechanically preserve every old constraint, and access to search does not
guarantee the model chooses to search. A paired continuation evaluation with
the same model and restored workspace is needed to measure lost constraints,
repeated work and the correctness of the first actions after compaction.

One pinned Pi SDK preparation edge remains: if the newest entry is a tool result
larger than `keepRecentTokens`, its cut-point search can report “Nothing to compact
(session too small)” before this extension runs. The transcript and active
context remain intact, with no inference request. The usual condensed tool output
is smaller than the default tail budget, but a deliberately small budget or a
large uncondensed result can still reach this case; the new tail selector cannot
repair a compaction hook that the SDK never calls.

Host Pi loads the updated source on relaunch. Pi-opsec needs its frontend bundle
installed and a new Pi process; no backend restart or model change is required.
