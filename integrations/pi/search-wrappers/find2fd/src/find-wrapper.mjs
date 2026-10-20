import { basename } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { runFind } from "./find-runtime.mjs";
export { planFind, runFind } from "./find-runtime.mjs";

const quote = (value) => `'${value.replaceAll("'", "'\\''")}'`;
const wrapperPath = fileURLToPath(import.meta.url);
const rustPath = fileURLToPath(new URL("../rust/target/release/find2fd-rust", import.meta.url));

// Bash expands quotes and globs first. This function is intentionally private to
// its shell; user commands and build scripts retain their native executable.
export function withFindWrapper(context, shellPath) {
  if (shellPath && !/^bash(?:\.exe)?$/.test(basename(shellPath))) return context;
  const prefix = `if [ -n "\${BASH_VERSION-}" ]; then
\tfind() {
\t\tif [[ \${FIND2FD_ACCELERATION-} == off || \${FIND2FD_ACCELERATION-} == 0 || (\${FIND2FD_EXPERIMENTAL_FD+x} && \${FIND2FD_EXPERIMENTAL_FD-} != 1) ]]; then
\t\t\tcommand find "$@"
\t\t\treturn
\t\tfi
\t\tif [[ ! -e /proc/"$BASHPID"/fd/0 || ! -e /proc/"$BASHPID"/fd/1 || ! -e /proc/"$BASHPID"/fd/2 ]]; then
\t\t\tcommand find "$@"
\t\t\treturn
\t\tfi
\t\tif [[ \${FIND2FD_EXPERIMENTAL_FD-} != 1 ]]; then
\t\t\tlocal _find2fd_rust=\${FIND2FD_RUST_BINARY:-${quote(rustPath)}}
\t\t\tif [[ -f \${_find2fd_rust} && -x \${_find2fd_rust} ]]; then
\t\t\t\tcommand "\${_find2fd_rust}" "$@"
\t\t\telse
\t\t\t\tcommand find "$@"
\t\t\tfi
\t\t\treturn
\t\tfi
\t\tif ! (
\t\t\tLC_ALL=C
\t\t\t_find2fd_nonascii=$'[^\\001-\\177]'
\t\t\tfor _find2fd_arg; do
\t\t\t\tif [[ $_find2fd_arg =~ $_find2fd_nonascii ]]; then
\t\t\t\t\tcommand ${quote(process.execPath)} --input-type=module -e 'import {isUtf8} from "node:buffer";import {readFileSync} from "node:fs";try{process.exit(isUtf8(readFileSync("/proc/self/cmdline"))?0:1)}catch{process.exit(1)}' -- "$@"
\t\t\t\t\texit $?
\t\t\t\tfi
\t\t\tdone
\t\t); then
\t\t\tcommand find "$@"
\t\t\treturn
\t\tfi
\t\tlocal _find2fd_fds='' _find2fd_info _find2fd_fd _find2fd_flags _find2fd_key _find2fd_value
\t\tlocal _find2fd_fdinfo_dir=/proc/$BASHPID/fdinfo
\t\tfor _find2fd_info in "\${_find2fd_fdinfo_dir}"/[0-9]*; do
\t\t\t_find2fd_fd=\${_find2fd_info##*/}
\t\t\t[[ \${_find2fd_fd} =~ ^[0-9]+$ && \${_find2fd_fd} -ge 3 ]] || continue
\t\t\t_find2fd_flags=
\t\t\twhile read -r _find2fd_key _find2fd_value; do
\t\t\t\tif [[ \${_find2fd_key} == flags: ]]; then
\t\t\t\t\t_find2fd_flags=\${_find2fd_value}
\t\t\t\t\tbreak
\t\t\t\tfi
\t\t\tdone 2>/dev/null <"\${_find2fd_info}"
\t\t\tif [[ \${_find2fd_flags} =~ ^[0-7]+$ ]] && (((8#\${_find2fd_flags} & 8#2000000) == 0)); then
\t\t\t\t_find2fd_fds+=\${_find2fd_fds:+,}\${_find2fd_fd}
\t\t\tfi
\t\tdone
\t\tFIND2FD_EXPERIMENTAL_FD=1 _FIND2FD_CALLER_FDS=\${_find2fd_fds} command ${quote(process.execPath)} ${quote(wrapperPath)} "$@"
\t}

\texport -n -f find
fi
`;
  return { ...context, command: prefix + context.command };
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  process.exitCode = await runFind(process.argv.slice(2));
}
