# Archived tool-result retrieval

`rehydrate_tool_result` retrieves original text retained by the tool-output
condensing extension. It is separate from `pi_session_search`, which searches saved
conversation messages. Retrieval uses local CPU/file operations and no model or
GPU inference.

Saved history named `qwen_rehydrate_tool_turn` or `read_archived_tool_result` is
normalized in outgoing model context. Those names are not registered tools;
new calls use `rehydrate_tool_result`. Stored transcripts and archive bytes
remain unchanged.

## Archive and source contract

Oversized text-only results are saved before their compact context view is
returned. The default inline budget is 8 KiB. Archives use SHA-256 names under
`$XDG_STATE_HOME/qwen-r9700/pi-tool-results`, falling back to
`~/.local/state/qwen-r9700/pi-tool-results`. `QWEN_PI_TOOL_RESULT_DIR` selects an
absolute alternative; writer and reader normalize it identically. Symlinked
archive paths are rejected. Directories are owned mode `0700`, payloads owned
mode `0400` with one link after publication or interrupted-publication recovery.

The retained text is the exact available tool result. For Pi's truncated Bash
results, the writer can retain the complete backing output instead: it checks
the actual truncation metadata, byte counts, displayed tail and file identity.
Arbitrary `fullOutputPath` fields cannot replace a tool's returned text. If a
tool or an upstream exception omits that metadata, the writer retains the text
it actually received, not an invented reconstruction of hidden command output.

The reader verifies permissions, file identity and SHA-256 before returning
text. The optional `QWEN_PI_TOOL_TURN_ARCHIVE_ROOT` also supports historical
`qwen-pi-archived-tool-turn-v1` JSON archives with text-only tool results. Current
Radiance launches use the live text archive by default.

Archives survive frontend releases and backend restarts. Host and Pi-opsec stores
are separate: moving a transcript alone does not move its archived tool payloads.

## Selection and output limits

```json
{"sha256":"DIGEST_FROM_CONDENSED_RESULT","pattern":"exact diagnostic text","context":2,"max_lines":20}
```

```json
{"sha256":"DIGEST_FROM_CONDENSED_RESULT","start_line":120,"end_line":135}
```

Lines are numbered from one and range endpoints are inclusive. Patterns are
case-insensitive literal text, not regular expressions. Matching lines take
priority over surrounding context when the line budget is small; returned lines
remain in source order. Context is limited to 20 neighboring lines on either
side and selected output to 200 lines, with a default of 80 for explicit queries.

A digest-only call previews at most 12 lines in 1,536 bytes. Explicit retrieval
has a 32 KiB rendered-output limit. Oversized lines are excerpted around their
literal matches so a late match remains visible. When necessary, matching lines
also take priority over context under the byte limit. UTF-8 characters are not
split. Ellipses and notices identify clipped text; the immutable archive remains
unchanged. Tool metadata distinguishes selected, returned and clipped lines and
records line-limit and output truncation separately.

Rehydrated results are marked untrusted data and are excluded from condensation,
preventing recursive rehydrate/condense calls. A missing, unsafe or corrupt
archive produces an error. Malformed JSON errors do not quote archived contents.

The output cap does not cap the archive read: authentication currently reads and
hashes the full payload synchronously. Very large archives can therefore take
longer than their small returned excerpts suggest. Cancellation is checked before
retrieval; it does not interrupt an in-progress synchronous filesystem read.

## Audit and verification

The focused audit reproduced and repaired match omission under small line and
byte limits, inconsistent archive-root normalization, unverified backing-file
substitution, Unicode-tail overflow at small condensation budgets, an invented
blank line for an empty result, and parser diagnostics quoting malformed payloads.

Checks use synthetic data, including the actual pinned Pi SDK and Bash tool
protocol. They cover live and historical lookup, exact archived bytes, line/range
boundaries, Unicode clipping, corruption and unsafe paths, interrupted publishing,
deduplication, cancellation, activation, persistence and recursion prevention.

```sh
node --test tests/test_tool_turn_rehydrate.mjs tests/test_tool_turn_rehydrate_sdk.mjs
UV_CACHE_DIR=/data/.cache/uv uv run pytest -q tests/test_pi_tool_output_condense.py
```

The installed host and VM artifacts receive separate offline synthetic checks.
Passing these checks establishes the tested behavior, not absence of every
possible filesystem race or unlimited-memory handling.
