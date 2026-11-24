import { createRunningBuildMonitor, strikeAppTitle } from "./qwen-running-build.mjs";
import { createRequire } from "node:module";
import { readdir, realpath } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const GUARD = Symbol.for("coherence.pi.running-build-title.v1");
let runningRuntime;

function runningTerminalRuntime() {
  // Pin the executing SDK once. A later session switch or /reload must not
  // resolve a launcher that may now point to a different installation.
  return runningRuntime ??= (async () => {
    const entrypoint = await realpath(process.argv[1]);
    const runtimeRequire = createRequire(pathToFileURL(entrypoint));
    const { ProcessTerminal } = await import(pathToFileURL(runtimeRequire.resolve("@earendil-works/pi-tui")));
    return { entrypoint, ProcessTerminal };
  })();
}

// Replace only the SDK's trailing cwd basename; keep session names and custom
// extension titles intact. The terminal itself supplies its " - Terminal" suffix.
export function expandTitleCwd(title, cwd, home = homedir()) {
  if (!cwd || !/^(?:π|Pi|pi) - /u.test(title)) return title;
  const directory = resolve(cwd), homeDirectory = resolve(home);
  const name = basename(directory);
  if (!title.endsWith(` - ${name}`)) return title;
  const display = directory === homeDirectory ? "~"
    : directory.startsWith(`${homeDirectory}${sep}`) ? `~${directory.slice(homeDirectory.length)}` : directory;
  return title.slice(0, title.length - name.length) + display;
}

function extensionRoots(args = process.argv) {
  const roots = [fileURLToPath(import.meta.url)];
  for (let index = 0; index < args.length; index++) {
    if (["--extension", "-e"].includes(args[index]) && args[index + 1]) roots.push(args[++index]);
    else if (args[index].startsWith("--extension=")) roots.push(args[index].slice("--extension=".length));
  }
  return roots;
}

async function discoveredExtensions() {
  if (process.argv.includes("--no-extensions")) return [];
  const directory = resolve(process.env.PI_CODING_AGENT_DIR ?? `${homedir()}/.pi/agent`, "extensions");
  let entries;
  try { entries = await readdir(directory, { withFileTypes: true }); }
  catch (error) { if (error.code === "ENOENT") return []; throw error; }
  if (entries.length > 256) throw new Error("Extension directory exceeds identity bound");
  return entries.filter((entry) => !entry.name.startsWith(".") &&
    (entry.isDirectory() || entry.isSymbolicLink() || /\.(?:ts|mjs|cjs|js)$/.test(entry.name)))
    .map((entry) => resolve(directory, entry.name)).sort();
}

// One guard per terminal class, including across /reload and session switches.
// Recapturing the baseline on reload would bless code the process never loaded.
export function installRunningBuildTitle(TerminalClass, {
  monitorOptions = {}, monitorFactory = createRunningBuildMonitor, registerExitHook = true,
} = {}) {
  const prototype = TerminalClass.prototype;
  if (Object.hasOwn(prototype, GUARD)) return prototype[GUARD];
  const original = prototype.setTitle;
  if (typeof original !== "function") throw new Error("Pi terminal has no title setter");
  let terminal, rawTitle, renderedTitle, renderedTerminal, workingDirectory, disposed = false;
  const render = () => {
    if (disposed || !terminal || rawTitle === undefined) return;
    const next = strikeAppTitle(expandTitleCwd(rawTitle, workingDirectory), monitor.outdated);
    if (renderedTitle === next && renderedTerminal === terminal) return;
    original.call(terminal, next);
    renderedTitle = next;
    renderedTerminal = terminal;
  };
  const monitor = monitorFactory({ extensionRoots: extensionRoots(), ...monitorOptions, onChange: render });
  const wrapper = function (title) {
    terminal = this;
    rawTitle = String(title);
    render();
  };
  const onExit = () => monitor.stop();
  const guard = {
    monitor,
    start: () => monitor.start(),
    setCwd(cwd) { workingDirectory = cwd; render(); },
    dispose() {
      if (disposed) return;
      disposed = true;
      monitor.stop();
      if (prototype.setTitle === wrapper) prototype.setTitle = original;
      if (prototype[GUARD] === guard) delete prototype[GUARD];
      process.removeListener("exit", onExit);
    },
  };
  Object.defineProperty(prototype, GUARD, { value: guard, configurable: true });
  prototype.setTitle = wrapper;
  if (registerExitHook) process.once("exit", onExit);
  return guard;
}

export default function runningBuildTitle(pi) {
  pi.on("session_start", async (_event, ctx) => {
    // Print and RPC sessions have no terminal title to decorate.
    if (!ctx.hasUI) return;
    // Resolve from the executing CLI, not the extension checkout or frozen
    // documentation assets: .mjs extensions can load outside Jiti's aliases.
    // Node can preserve the launcher's symlink in argv[1]; resolve it before
    // locating dependencies, and pin the monitor to this running CLI target.
    const { entrypoint, ProcessTerminal } = await runningTerminalRuntime();
    const guard = installRunningBuildTitle(ProcessTerminal, { monitorOptions: {
      entrypoint, recordBuild: true, extensionRootProvider: discoveredExtensions,
      profileFiles: process.env.QWEN_PI_BUILD_PROFILE ? [process.env.QWEN_PI_BUILD_PROFILE] : [],
    } });
    guard.setCwd(ctx.sessionManager?.getCwd?.() ?? ctx.cwd);
    // Identity I/O never blocks session admission or model generation.
    void guard.start().catch(() => {});
  });
}
