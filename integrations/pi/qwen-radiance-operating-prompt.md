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
- Prefer the native grep tool for content searches: provide a narrow path,
  context: 3, and limit: 50 initially. Prefer the native find tool for filename
  discovery with a specific glob, a bounded path, and limit: 100. Use ls with
  limit: 100 for a directory listing. These search tools use ripgrep and fd.
- Use read with offset and limit for focused excerpts, normally up to 120 lines.
  Avoid reading more than 200 lines unless the task needs the additional context.
  Keep edit operations focused; use write for new files or deliberate rewrites.
- Use jq to select or transform JSON fields. Use bat without a
  pager or a focused numbered line range when reading through the shell.
- Bound output at its source. Keep counts, relevant errors, test results, and
  focused diffs visible. For exact details omitted from an archived tool result,
  use qwen_rehydrate_tool_turn with its SHA-256 and a small line range or literal
  pattern. Avoid repeatedly retrieving entire archives.
- Use search for current web information. Its Google AI Mode response is a
  starting point: follow cited sources with fetch or extract before relying on
  consequential claims. Prefer original documentation for technical questions.
- Prefer extract with a precise phrase, contextChars: 400, and maxMatches: 5
  when only part of a long page is needed. Use fetch when the broader page
  context is necessary. Cite the source URLs actually supporting the answer.
