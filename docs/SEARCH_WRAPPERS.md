# Pi filesystem search

Qwen's `run_shell_command` uses the native Rust engines from
[grep2rg](https://github.com/Terrydaktal/grep2rg) and
[find2fd](https://github.com/Terrydaktal/find2fd). Qualified searches run inside
one Rust process, with sequential or parallel execution and GNU-compatible
ordering. They launch neither Node nor rg/fd. Unsupported semantics delegate
to the original GNU tool before emitting search results. These finite supported
classes and measurements do not establish universal acceleration or correctness.

| Entry point | Current behavior |
| --- | --- |
| `search_file_contents` | Uses Pi's native content search; `files_only: true` returns matching filenames once each. |
| Model-issued Bash `grep` for `qwen-r9700` | Uses the built Rust grep2rg engine. If it is unavailable, the retained C dispatcher or GNU grep remains usable. |
| Model-issued Bash `find` for `qwen-r9700` | Uses the built Rust find2fd engine, or GNU find if the build is unavailable. |
| User `!` commands, other providers, child scripts, explicit bypasses | Keep ordinary system executables. |

The native tools remain the preferred interface for repository inspection.
Narrow the directory, use `literal: true` for exact paths/symbols/errors and
request context only after finding the relevant matches. Multiple traversal
exclusions or ignored build files can use explicit shell `rg` with scoped paths
and a deadline. Filtering results afterward does not avoid scanning the files.

## Scope and fallback

`qwen-shell-search-wrappers.mjs` composes the two standalone packages' actual
private Bash functions. The shell expands quotes and globs normally; Coherence
does not parse or rewrite the user's command text. Pipes, loops and Bash
subshells retain the functions. The functions are not exported, so child
scripts, `command grep/find`, explicit `/usr/bin/grep/find`, user-issued `!`
commands and other providers retain their normal execution. Non-Bash shells
receive no hook. No global alias, system executable or user `PATH` is replaced.

Use `GREP2RG_ACCELERATION=gnu` or `FIND2FD_ACCELERATION=off` to force GNU.
`GREP2RG_ENGINE=c` explicitly selects grep's older C dispatcher.
`GREP2RG_EXPERIMENTAL_RG=1` and `FIND2FD_EXPERIMENTAL_FD=1` select the retained
JavaScript/rg/fd experiments; these are not the default Rust engines and do not
promise a speedup. GNU remains necessary for unsupported options, descriptors,
actions and other compatibility boundaries.

The `files_only` addition preserves the native search schema and default
execution. It uses rg's hidden/ignore/glob/literal/case behavior, returns at most
1,000 filenames or 50 KiB, and has a 20-second deadline. `limit` counts filenames
(default 100); `context` is ignored. Cancellation, no matches, invalid input,
search errors and output truncation remain distinct outcomes.

## Source pinning and builds

The standalone repositories own the search engines and their operator suites.
Coherence bundles their exact runtime sources and attribution under
`integrations/pi/search-wrappers`; it does not maintain another implementation.
`source-manifest.json` records each source file's SHA-256 and the upstream base
commit. Generated ELF files and Cargo targets are excluded from Git.

After updating the standalone sources, refresh and check the snapshot:

```sh
python3 tools/sync_search_wrappers.py
python3 tools/sync_search_wrappers.py --check
```

These maintenance commands expect the standalone checkouts at `~/Dev/grep2rg`
and `~/Dev/find2fd`, or explicit `--grep-root` and `--find-root` paths. Normal
clones and runtime builds do not need either external checkout. Build the locked
Rust sources on each machine that executes Pi tools, using Rust 1.90 or newer:

```sh
node integrations/pi/search-wrappers/grep2rg/tools/build-rust.mjs
node integrations/pi/search-wrappers/find2fd/tools/build-rust.mjs
```

The optional legacy C adapter builds with
`node integrations/pi/search-wrappers/grep2rg/tools/build-native.mjs`.
Missing builds retain compatibility fallback. Installation and deployment must
verify both source and executable identities; a source manifest alone does not
prove that an already running Pi process loaded the new module. Existing Pi
processes retain loaded extensions until quit/relaunch. No model backend restart
is required for these search changes.

## Verification and measurements

The standalone repositories contain Rust units, generated independent GNU
comparisons, CLI/Pi integration and extracted-package checks. Run their published
`verify:rust` and normal/deep verification commands there for engine qualification.
The vendored snapshot contains runtime sources and Rust unit checks; the full
release-oracle campaigns stay in their authoritative repositories.

Coherence's focused checks cover bounded filename output and failures, manifest
integrity, combined grep/find GNU bytes/status/order, explicit C/JS experiments,
and actual `execve` proof that the default hook launches both Rust engines:

```sh
cargo test --locked --manifest-path integrations/pi/search-wrappers/grep2rg/rust/Cargo.toml --bin grep2rg-rust
cargo test --locked --manifest-path integrations/pi/search-wrappers/find2fd/rust/Cargo.toml --bin find2fd-rust
node --test tests/test_shell_search_wrappers.mjs tests/test_files_only_search.mjs
node --test tests/test_tool_names_sdk.mjs
```

The shell tests require GNU grep/find, Bash, rg, fd/fdfind, `strace` and the built
Rust and legacy C artifacts. The schema and complete SDK checks require Pi
0.84.2; the complete SDK test also needs the Searchtool source. Set
`QWEN_TEST_PI_ROOT` to `node_modules/@earendil-works` and
`QWEN_TEST_SEARCHTOOL_ROOT` to its checkout when using nondefault locations.
Inspect TAP skip counts before claiming SDK qualification. CI provisions the
pinned schema SDK and tests Rust units plus CPU/shell/schema behavior; it does
not claim a complete Searchtool or live inference test.

For an isolated qualification build, `COHERENCE_SEARCH_TEST_ROOT` can point the
combined-hook and SDK naming-extension tests at an exact disposable source
copy, preserving binaries used by installed sessions. The
[integration capture](../benchmarks/results/pi-rust-search-integration-20261020.json)
records the executed source and build identities and test outcomes.

`node tools/benchmark_search_dispatch.mjs [SOURCE_ROOT]` measures seven owned
fixture classes through the combined Bash hook. One warmup and seven alternating
GNU/automatic pairs compare full stdout/stderr/status on every invocation. The
timed interval includes Bash startup, dispatch, search and complete output;
fixture creation and comparisons are outside it. Direct engine class tables
remain in the standalone READMEs and must not be presented as Pi shell timings.

The October 12 [host](../benchmarks/results/search-wrapper-routing-20261012/host-dispatch-performance.json)
and [VM](../benchmarks/results/search-wrapper-routing-20261012/vm-dispatch-performance.json) measurements,
and [VM qualification](../benchmarks/results/search-wrapper-routing-20261012/vm-installed-qualification.json)
describe the earlier C/Node/rg/fd dispatchers. Their source identities and raw
numeric results are retained as historical evidence, not current Rust timings.
All fixtures and captures described here use synthetic data, with no model
inference, Google query or private transcript.
