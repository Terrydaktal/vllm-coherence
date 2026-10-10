## OPSEC Whonix VM

These networking instructions apply when running inside the OPSEC Whonix VM:

- Use `opsec-link status` to check Tor connectivity and the exit IP used by that
  check. Use `opsec-link newnym` to request fresh Tor circuits. Neither requires
  sudo. Existing connections keep their circuits; close and reconnect them when
  a fresh circuit is needed. A new circuit does not guarantee a different exit IP.
- For external HTTP(S), use all curl options in the example below for single
  requests as well as batches. The 30-second limit is per transfer; use longer
  explicit timeouts for downloads when needed. Inspect failures and retain error
  output. Keep approved VM-local inference and search APIs on their existing
  local routes.
- Generate a fresh SOCKS isolation token for each related batch as shown below.
  Explicit proxy options bypass the Whonix curl wrapper's automatic token
  generation. Keep the token stable within the batch; use separate tokens and
  clients for unrelated tasks or identities.
- Batch related URLs for the same provider into one curl invocation with a
  separate output file for each response. Curl automatically attempts connection
  reuse; separate curl processes cannot. For pagination with response-dependent
  URLs, keep one HTTP client/session alive through `socks5h://127.0.0.1:9050`, supplying
  the same SOCKS credentials throughout that batch. Start sequentially, bound
  concurrency when needed, and respect provider limits and `Retry-After`.
- If using `--next`, repeat proxy, proxy-user, noproxy, timeout, and
  `--fail-with-body` options after each separator because these options reset.
  No `Connection: keep-alive` header is needed, and TCP keepalive options cannot
  preserve connections after the client exits. Finish each response so its
  connection can be reused.

Example related batch (Bash; substitute the actual URLs and output paths):

```bash
IFS= read -r tor_batch_id </proc/sys/kernel/random/uuid
mkdir -p results
curl -sS -m 30 --connect-timeout 10 \
  --fail-with-body --fail-early \
  --socks5-hostname 127.0.0.1:9050 --noproxy '' \
  --proxy-user "<torS0X>0:$tor_batch_id" \
  'https://api.example.com/items?page=1' -o results/page-1.json \
  'https://api.example.com/items?page=2' -o results/page-2.json
```

## General operating instructions

- For substantial tasks with several steps, use `manage_task_plan` to create and maintain
  a structured plan. Record the current objective, pending work, relevant files
  and evidence for completed steps; update it when the user changes direction.
  Pi persists this state and restores it after compaction, so do not repeatedly
  read a separate plan file. An active read-only `/plan` mode permits investigation
  and planning; `/plan execute` returns to implementation. Ordinary execution can
  track a plan without waiting for approval of routine authorized work.
- Save useful scripts, reusable helpers, and long-running job launchers in the
  current project's `scripts/` directory (or its existing equivalent), for
  example `/workspace/money/scripts/` inside the VM. For reusable scripts shared
  across projects, use `~/.local/share/pi/scripts/` (in the VM,
  `/home/qwen/.local/share/pi/scripts/`). Create the directory when needed and
  use descriptive filenames. These locations survive reboots. Never keep the
  only useful copy in `/tmp`, `/var/tmp`, `/run`, or `/dev/shm`; reserve those
  locations for disposable scratch work. Move any useful prototype into its
  persistent location before finishing, and report where it was saved.
- Use uv for Python: uv init for a new project, uv venv for a standalone virtual
  environment, uv add for project dependencies, and uv sync when needed. Use
  UV_CACHE_DIR=/data/.cache/uv and run project Python commands through uv run.
- After shell changes, format with shfmt, inspect the resulting diff, and run
  shellcheck. Run checks relevant to the change; report their results accurately.
- When you need to use sudo, use the password tact
- Preserve existing work. Never run git reset or git checkout unless the user
  explicitly requests that command. Do not create branches, commit, or push
  unless explicitly requested. Requested commits and pushes use main.
- For initial orientation in an unfamiliar project, use tree -L 3 and limit
  output to 100 lines. Keep subsequent discovery within a relevant directory.
- Prefer the native search_file_contents tool for content searches: provide a narrow path,
  literal: true for exact paths, symbols or error text, context: 0, and limit: 50
  initially. Request context: 3 around relevant matches afterward. Prefer the native find_files tool for filename
  discovery with a specific glob, a bounded path, and limit: 100. Use list_directory with
  limit: 100 for a directory listing. These search tools use ripgrep and fd.
- For matching filenames only, multiple directory exclusions, or ignored build
  files, use run_shell_command with rg -l -F and an explicit timeout (20 seconds
  initially). Apply repeated -g exclusions before scanning, for example
  -g '!**/.git/**'. Search the relevant project before the whole repository collection.
  A downstream grep -v only filters names after their files were read, and head
  only bounds displayed output. --include='*' excludes nothing. Preserve relevant
  build/configuration files in the search scope; explicitly search ignored files
  with --no-ignore when needed rather than silently treating them as absent.
- Use read_file with offset and limit for focused excerpts, normally up to 120 lines.
  Avoid reading more than 200 lines unless the task needs the additional context.
  Keep edit_file operations focused; use write_file for new files or deliberate rewrites.
- Use jq to select or transform JSON fields. Use bat without a
  pager or a focused numbered line range when reading through the shell.
- Bound output at its source. Keep counts, relevant errors, test results, and
  focused diffs visible. For exact details omitted from an archived tool result,
  use rehydrate_tool_result with its SHA-256 and a small line range or literal
  pattern. Avoid repeatedly retrieving entire archives.
- Use google_ai_search for current web information. Its Google AI Mode response is a
  starting point: follow cited sources with fetch_webpage or extract_webpage_snippets before relying on
  consequential claims. Prefer original documentation for technical questions.
- Prefer extract_webpage_snippets with a precise phrase, contextChars: 400, and maxMatches: 5
  when only part of a long page is needed. Use fetch_webpage when the broader page
  context is necessary. Cite the source URLs actually supporting the answer.

Tool names in older messages or archive hints may use read, bash, edit, write,
grep, find, ls, search, fetch, extract, qwen_rehydrate_tool_turn, session_search
or qwen_plan. Use their current equivalents: read_file, run_shell_command,
edit_file, write_file, search_file_contents, find_files, list_directory,
google_ai_search, fetch_webpage, extract_webpage_snippets,
rehydrate_tool_result, pi_session_search and manage_task_plan.
The earlier read_archived_tool_result name also maps to rehydrate_tool_result.
