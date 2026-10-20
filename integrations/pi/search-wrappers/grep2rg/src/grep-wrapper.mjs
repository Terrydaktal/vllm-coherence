// A compatibility adapter for model-issued filesystem searches, not a global
// grep replacement. Bash expands quotes/globs before passing argv here.
import { basename } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const quote = (value) => `'${value.replaceAll("'", "'\\''")}'`;
const wrapperPath = fileURLToPath(import.meta.url);
const nativePath = fileURLToPath(new URL("../bin/grep2rg-native", import.meta.url));
const rustPath = fileURLToPath(new URL("../rust/target/release/grep2rg-rust", import.meta.url));
import { runGrep } from "./grep-runtime.mjs";
export { planGrep, runGrep } from "./grep-runtime.mjs";

// Non-exported function: child build scripts, user shell commands and system
// grep remain untouched. No textual parsing/replacement of the command itself.
export function withGrepWrapper(context, shellPath) {
  if (shellPath && !/^bash(?:\.exe)?$/.test(basename(shellPath))) return context;
  // A Node argv string cannot preserve an invalid UTF-8 byte. Bash identifies
  // non-ASCII arguments; a raw /proc read distinguishes valid Unicode from
  // invalid bytes before either can pass through JavaScript argv decoding.
  // The locale change stays in one subshell and never reaches either backend.
  const prefix = `if [ -n "\${BASH_VERSION-}" ]; then
\tgrep() {
\t\tif [[ \${GREP2RG_EXPERIMENTAL_RG-} != 1 ]]; then
\t\t\tif [[ \${GREP2RG_ACCELERATION-} == gnu || ! -e /proc/"$BASHPID"/fd/0 || ! -e /proc/"$BASHPID"/fd/1 || ! -e /proc/"$BASHPID"/fd/2 ]]; then
\t\t\t\tcommand grep "$@"
\t\t\telif [[ \${GREP2RG_ENGINE-} != c && -x ${quote(rustPath)} ]]; then
\t\t\t\tcommand ${quote(rustPath)} "$@"
\t\t\telif [[ -x ${quote(nativePath)} ]]; then
\t\t\t\tcommand ${quote(nativePath)} "$@"
\t\t\telse
\t\t\t\tcommand grep "$@"
\t\t\tfi
\t\t\treturn
\t\tfi
\t\tlocal _grep2rg_info _grep2rg_fd _grep2rg_key _grep2rg_flags _grep2rg_caller_fds=
\t\tif [[ ! -e /proc/"$BASHPID"/fd/0 || ! -e /proc/"$BASHPID"/fd/1 || ! -e /proc/"$BASHPID"/fd/2 ]]; then
\t\t\tcommand grep "$@"
\t\t\treturn
\t\tfi
\t\tif ! (
\t\t\tLC_ALL=C
\t\t\t_grep2rg_nonascii=$'[^\\001-\\177]'
\t\t\tfor _grep2rg_arg; do
\t\t\t\tif [[ $_grep2rg_arg =~ $_grep2rg_nonascii ]]; then
\t\t\t\t\tcommand ${quote(process.execPath)} --input-type=module -e 'import {isUtf8} from "node:buffer";import {readFileSync} from "node:fs";try{process.exit(isUtf8(readFileSync("/proc/self/cmdline"))?0:1)}catch{process.exit(1)}' -- "$@"
\t\t\t\t\texit $?
\t\t\t\tfi
\t\t\tdone
\t\t); then
\t\t\tcommand grep "$@"
\t\t\treturn
\t\tfi
\t\tfor _grep2rg_info in /proc/"$BASHPID"/fdinfo/[0-9]*; do
\t\t\t_grep2rg_fd=\${_grep2rg_info##*/}
\t\t\tif ! [[ $_grep2rg_fd =~ ^[0-9]+$ ]] || ((_grep2rg_fd <= 2)); then continue; fi
\t\t\twhile IFS=$' \\t' read -r _grep2rg_key _grep2rg_flags; do
\t\t\t\tif [[ $_grep2rg_key == flags: && $_grep2rg_flags =~ ^[0-7]+$ ]] && (((8#$_grep2rg_flags & 8#2000000) == 0)); then
\t\t\t\t\t_grep2rg_caller_fds+=\"\${_grep2rg_caller_fds:+,}$_grep2rg_fd\"
\t\t\t\tfi
\t\t\tdone 2>/dev/null <"$_grep2rg_info"
\t\tdone
\t\t_GREP2RG_CALLER_FDS=$_grep2rg_caller_fds command ${quote(process.execPath)} ${quote(wrapperPath)} "$@"
\t}

\texport -n -f grep
fi
`;

  return { ...context, command: prefix + context.command };
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  const status = await runGrep(process.argv.slice(2));
  process.exitCode = status;
  // The cleanup has completed. A cancelled stdout write may otherwise keep
  // Node's stdio request alive indefinitely when its consumer stops reading.
  if ([129, 130, 143].includes(status)) process.exit(status);
}
