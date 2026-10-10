# Original transcript search

`pi_session_search` gives the Pi agent access to original saved conversation text
after compaction. JSONL remains authoritative. The derived SQLite FTS5 database
stores searchable words, their positions, document lengths, entry relationships
and source offsets; it does not store another copy of the message bodies.
The index is private data despite being smaller: its vocabulary and metadata can
still disclose conversation information.

Search, index refresh and JSONL parsing run on the Pi computer's CPU in a worker
thread. They require no additional model or GPU memory and add no per-token or
per-generation-round hook. The first search builds the selected index; later
searches refresh changed sources before querying. Those refresh costs are
separate from the small warm-index lookup times measured on synthetic data.

## Using the tool

The model can call these operations:

```json
{"query":"top-20 mismatch"}
```

```json
{"query":"\"top 20\" AND mismatch","mode":"fts","limit":3}
```

```json
{"query":"UND_ERR_SOCKET","mode":"literal","roles":["toolResult"]}
```

```json
{"query":"accepted arithmetic contract","scope":"project"}
```

```json
{"around_entry_id":"ENTRY_ID_FROM_RESULT","session_file":"SESSION_FILE_FROM_RESULT","window":2}
```

The default `words` mode requires all query words and uses BM25 relevance
ranking. `fts` accepts SQLite phrases, Boolean operators and proximity queries.
`literal` checks an exact, case-sensitive substring in original text; use it for
punctuation-sensitive symbols, paths or errors. Literal search may scan more text
than an indexed word search.

Search defaults to the current session's selected branch, including its original
pre-compaction ancestors. `scope:"project"` searches the JSONL files in the same
session directory. It does not search unrelated project directories or the remote
model host. An explicit `session_file` must remain in this directory.
`include_branches:true` additionally permits alternate branches and labels their
results. Entry IDs and their parent relationships determine branch membership;
adjacent JSONL lines alone do not establish a conversation branch.

Original user messages, assistant prose, unredacted thinking, tool results and
saved summaries are searchable. Images and redacted thinking are not invented
or decoded as text. `/context` and `/purge-thinking` exclusions affect the active
prompt; historical retrieval can still recover saved excluded text. Returned
history is labelled untrusted and potentially superseded, and never becomes a
new standing instruction merely because a search found it.

The default is three matches with a shared 6,000-character excerpt budget. Limits
are ten matches, 16,000 excerpt characters and a 24 KiB rendered tool response.
These are character/byte budgets, not exact model-token counts. Excerpts are
centred on the query where possible; the response identifies truncation and the
source entry for a narrower search or expansion. Expansion permits up to three
neighbouring messages on either side. The tool does not automatically replace
compaction, run a second summarizer or search after every compaction.

## Persistence and source changes

The default database directory is
`$XDG_STATE_HOME/qwen-r9700/transcript-index`, falling back to
`~/.local/state/qwen-r9700/transcript-index`. Each session directory receives a
separate database. `QWEN_SESSION_SEARCH_DIR` overrides the private index directory.
Deleting a derived index permits rebuilding it from JSONL; it does not remove
conversation history. Do not delete the original transcript to clear an index.

The index authenticates retrieved source records with their byte offsets and
hashes. It handles appended records, rewritten or replaced files, truncation and
deleted sources. A partial final JSONL line waits until complete. A source that
changes during retrieval is refreshed once; remaining stale records are withheld
with an explicit notice. Malformed completed records fail indexing without
silently publishing an incomplete replacement index.

The worker serializes its own SQLite operations; SQLite transactions protect
concurrent Pi processes using the same derived database. Cancellation or the
60-second deadline terminates the worker and leaves the original transcript
untouched. The next call can start a fresh worker. Database directories must be
private and owned by the user; symlinked sources and unsafe database paths are
rejected. Existing group-writable transcripts are accepted only inside a private
session directory; world-writable sources remain rejected. Search never changes
transcript permissions. The required runtime is Node with built-in SQLite and FTS5
contentless-delete support, as provided by the pinned host/VM runtime.

## Source and verification

| File | Responsibility |
| --- | --- |
| `integrations/pi/qwen-session-index.mjs` | Contentless FTS5 index, source verification, branches, search and expansion. |
| `integrations/pi/qwen-session-search-worker.mjs` | Index refresh and query execution outside the UI thread. |
| `integrations/pi/qwen-session-search.mjs` | Pi tool schema, lifecycle, cancellation and bounded rendering. |
| `tests/test_session_index.mjs` | Synthetic storage, retrieval and filesystem regressions. |
| `tests/test_session_search.mjs` | Worker, tool, cancellation, scope and output regressions. |
| `tests/test_session_search_sdk.mjs` | Actual pinned Pi registration, compaction and tool execution. |

Run the focused CPU checks with:

```sh
node --test tests/test_session_index.mjs tests/test_session_search.mjs tests/test_session_search_sdk.mjs
```

These checks use synthetic sessions, not private chats or GPU inference. The
normal launcher and Pi-opsec package load the tool. Relaunch an existing Pi
process to load the installed extension; no model-server restart is needed.
