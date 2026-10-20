import { spawn, spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { constants, closeSync, lstatSync, mkdtempSync, openSync, opendirSync, readSync, realpathSync, rmSync, statfsSync, writeFileSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve, sep } from "node:path";
import { inheritedStdio, unusedInputFd } from "./find-descriptors.mjs";
import { displayPath, MAX_DIRECTORY_ENTRIES } from "./find-order.mjs";

export const INCREMENTAL_START_TIMEOUT_MS = 2000;
export const INCREMENTAL_CAPTURE_BYTES = 32 * 1024 * 1024;
export const MAX_LIVE_INCREMENTAL_CHILDREN = 100000;
export const MAX_LIVE_INCREMENTAL_NAME_BYTES = 32 * 1024 * 1024;
export const MAX_INCREMENTAL_PATH_BYTES = 4095;
export const MAX_INCREMENTAL_COMPONENT_BYTES = 255;
export const MAX_INCREMENTAL_MOUNT_INFO_BYTES = 1024 * 1024;
export const INCREMENTAL_FILESYSTEMS = ["ext2", "ext3", "ext4", "btrfs", "xfs", "tmpfs", "ramfs", "f2fs"];
const FILESYSTEM_MAGICS = new Set([0xef53, 0x9123683e, 0x58465342, 0x01021994, 0x858458f6, 0xf2f52010]);
const delay = (milliseconds) => new Promise((done) => setTimeout(done, milliseconds));
const globLiteral = (value) => value.replace(/[\\*?\[]/g, "\\$&");

function knownComponentFilesystems(roots) {
  if (roots.some((root) => !FILESYSTEM_MAGICS.has(Number(statfsSync(root).type) >>> 0))) return false;
  const descriptor = openSync("/proc/self/mountinfo", "r"), data = Buffer.alloc(MAX_INCREMENTAL_MOUNT_INFO_BYTES + 1);
  let length = 0;
  try {
    let count;
    while (length < data.length && (count = readSync(descriptor, data, length, data.length - length)) > 0) length += count;
  } finally { closeSync(descriptor); }
  if (length > MAX_INCREMENTAL_MOUNT_INFO_BYTES) return false;
  const buffers = roots.map((root) => Buffer.from(root));
  for (const line of data.subarray(0, length).toString("latin1").split("\n")) {
    if (!line) continue;
    const halves = line.split(" - ");
    if (halves.length !== 2) return false;
    const path = halves[0].split(" ")[4], filesystem = halves[1].split(" ")[0];
    if (!path || !filesystem) return false;
    const mount = Buffer.from(path.replace(/\\([0-7]{3})/g, (_, octal) => String.fromCharCode(Number.parseInt(octal, 8))), "latin1");
    if (!INCREMENTAL_FILESYSTEMS.includes(filesystem) && buffers.some((root) => root.equals(Buffer.from("/")) || mount.equals(root) ||
      (mount.subarray(0, root.length).equals(root) && mount[root.length] === 47))) return false;
  }
  return true;
}

/** The per-root GNU broker cannot retain FTS anchors beyond Linux PATH_MAX. */
export function incrementalPathReason(options, searches, cwd) {
  if (options.maxDepth === null) return "incremental controls require a finite maximum depth";
  const suffix = options.maxDepth * (MAX_INCREMENTAL_COMPONENT_BYTES + 1);
  const anchor = `/proc/${process.pid}/fd/2147483647`;
  if (Buffer.byteLength(cwd) > MAX_INCREMENTAL_PATH_BYTES || Buffer.byteLength(anchor) > MAX_INCREMENTAL_PATH_BYTES)
    return "incremental current-directory or descriptor-anchor pathname bound";
  try {
    const canonicalDirectories = [];
    for (const search of searches) {
      const paths = [search.root, search.path];
      if (search.directory) {
        const canonical = realpathSync(search.path); paths.push(canonical); canonicalDirectories.push(canonical);
      }
      if (paths.some((path) => Buffer.byteLength(path) + suffix > MAX_INCREMENTAL_PATH_BYTES))
        return "incremental displayed, absolute or canonical pathname bound";
    }
    if (options.maxDepth && canonicalDirectories.length && !knownComponentFilesystems(canonicalDirectories))
      return "incremental filesystem component-size or mount qualification";
  } catch { return "incremental starting-path canonicalization"; }
  return null;
}

/** GNU owns expression truth and output; only its descent decisions are observed. */
export function incrementalArguments(options, input, control, sentinel, token) {
  const controls = new Map(options.controls.map((item) => [item.index, item.kind]));
  const expression = options.evaluationArgs.flatMap((value, index) => {
    const kind = controls.get(index);
    if (kind === "prune") return ["(", "-prune", "-a", "-fprintf", control, `${token}:P\\n`, ")"];
    if (kind === "quit") return ["(", "-fprintf", control, `${token}:Q\\n`, "-a", "-quit", ")"];
    return [value];
  });
  const original = ["(", ...expression, ")", ...(options.implicitPrint ? ["-a", "-print"] : [])];
  return ["-P", ...(options.optimization ? [options.optimization] : []), "-files0-from", input, "-maxdepth", "0",
    "(", "-path", globLiteral(sentinel), "-a", "-fprintf", control, `${token}:A\\n`, ")", "-o", "(", ...original, ")"];
}

function fdProgram() {
  for (const program of ["fd", "fdfind"]) {
    const result = spawnSync(program, ["--version"], { stdio: ["ignore", "pipe", "pipe"], timeout: 2000, maxBuffer: 65536 });
    if (result.error?.code === "ENOENT") continue;
    return !result.error && !result.signal && result.status === 0 ? program : null;
  }
  return null;
}

function inside(path, root) { return root === sep || path === root || path.startsWith(`${root}${sep}`); }

/** A private directory must not become an extra entry in this traversal. */
function privateDirectory(searches) {
  const parent = realpathSync(tmpdir());
  const roots = searches.filter((search) => search.directory).map((search) => realpathSync(search.path));
  if (roots.some((root) => inside(parent, root))) return null;
  const directory = mkdtempSync(join(parent, "find2fd-incremental-"));
  try {
    if (roots.some((root) => inside(realpathSync(directory), root))) {
      rmSync(directory, { recursive: true, force: true }); return null;
    }
    return directory;
  } catch (error) {
    rmSync(directory, { recursive: true, force: true }); throw error;
  }
}

function orderedChildren(data, canonicalRoot, actualPath) {
  if (data.length && data.at(-1) !== 0) throw new Error("fd unterminated child record");
  const prefix = canonicalRoot.equals(Buffer.from("/")) ? canonicalRoot : Buffer.concat([canonicalRoot, Buffer.from("/")]);
  const names = new Map();
  let start = 0;
  for (let index = 0; index < data.length; index++) {
    if (data[index] !== 0) continue;
    let path = data.subarray(start, index); start = index + 1;
    if (!path.subarray(0, prefix.length).equals(prefix)) throw new Error("fd child escaped directory");
    const directory = path.at(-1) === 47;
    if (directory) path = path.subarray(0, -1);
    const relative = path.subarray(prefix.length), key = relative.toString("hex");
    if (!relative.length || relative.includes(47) || relative.equals(Buffer.from(".")) || relative.equals(Buffer.from("..")) || names.has(key))
      throw new Error("fd immediate-child protocol");
    names.set(key, { relative, directory });
  }
  const directory = opendirSync(actualPath, { encoding: "buffer" });
  const ordered = [];
  try {
    let entry;
    while ((entry = directory.readSync()) !== null) {
      if (ordered.length >= MAX_DIRECTORY_ENTRIES) throw new Error("GNU large-directory inode ordering");
      const candidate = names.get(entry.name.toString("hex"));
      if (!candidate || entry.isDirectory() !== candidate.directory) throw new Error("fd child set changed or incomplete");
      names.delete(entry.name.toString("hex"));
      ordered.push({ name: entry.name, directory: entry.isDirectory() });
    }
    if (names.size) throw new Error("fd child disappeared");
  } finally { directory.closeSync(); }
  return ordered;
}

/** Persistent FIFO/PTY evaluator with immediate, genuinely used fd listings. */
export async function runIncremental(plan, state, helpers) {
  const program = fdProgram();
  if (!program || state.signal) return state.signal ? helpers.signalStatus(state.signal) : null;
  let directory;
  try { directory = privateDirectory(plan.searches); } catch { return null; }
  if (!directory) return null;
  const input = join(directory, "paths"), sentinel = join(directory, "ack");
  const token = `F2FD${randomBytes(16).toString("hex")}`;
  let reader, writer, allocator, broker, active, started = false, cleanup;
  let allocatorFailed = false, brokerClosed = false, brokerStatus = 0, discoveryStatus = 0;
  const pending = [], waiting = [];
  let startupResolve, startupReject, controlText = "";
  const identities = new WeakMap();
  const track = (child) => { identities.set(child, child.pid ? helpers.processIdentity(child.pid) : null); return child; };
  const startup = new Promise((done, fail) => { startupResolve = done; startupReject = fail; });
  const publish = (message) => waiting.length ? waiting.shift().resolve(message) : pending.push(message);
  const failWaiters = (error) => { for (const waiter of waiting.splice(0)) waiter.reject(error); };
  const next = () => pending.length ? Promise.resolve(pending.shift()) : brokerClosed
    ? Promise.reject(new Error("GNU evaluator closed")) : new Promise((done, fail) => waiting.push({ resolve: done, reject: fail }));
  const stopOwned = (child, signal, detached = false) => {
    if (!child?.pid || child.exitCode !== null || child.signalCode) return [];
    const identity = identities.get(child);
    if (!identity || helpers.processIdentity(child.pid) !== identity) return [];
    try { child.kill("SIGSTOP"); } catch { /* Exited. */ }
    const snapshot = helpers.ownedDescendants(child.pid);
    helpers.signalSnapshot(snapshot, signal);
    if (detached) try { process.kill(-child.pid, signal); } catch { /* Gone group. */ }
    try { child.kill(signal); child.kill("SIGCONT"); } catch { /* Exited. */ }
    return snapshot;
  };
  const stop = (signal) => {
    if (cleanup) return;
    active?.stop?.(signal);
    const snapshot = [...stopOwned(broker, signal), ...stopOwned(allocator, signal, true)];
    failWaiters(new Error("cancelled")); startupReject(new Error("cancelled"));
    cleanup = new Promise((done) => setTimeout(() => {
      helpers.signalSnapshot(snapshot, "SIGKILL"); done();
    }, 250));
  };
  const closeWriter = () => { if (writer !== undefined) { closeSync(writer); writer = undefined; } };
  const write = async (data) => {
    let offset = 0;
    while (offset < data.length) {
      if (state.signal || brokerClosed || allocatorFailed) throw new Error("GNU evaluator unavailable");
      try { offset += writeSync(writer, data, offset); }
      catch (error) { if (error.code !== "EAGAIN") throw error; await delay(2); }
    }
  };
  let first = true;
  const evaluate = async (path) => {
    await write(Buffer.concat([path, Buffer.from([0]), Buffer.from(sentinel), Buffer.from([0])]));
    let pruned = false;
    for (;;) {
      const message = await next();
      if (message === "P") pruned = true;
      if (message === "Q") return { quit: true, pruned };
      if (message === "A") {
        if (first) { closeSync(reader); reader = undefined; first = false; }
        return { quit: false, pruned };
      }
    }
  };
  let brokerCompletion;
  let liveChildren = 0, liveNameBytes = 0;
  try {
    writeFileSync(sentinel, "", { mode: 0o600 });
    const fifo = spawnSync("mkfifo", ["-m", "600", input], { stdio: "ignore", timeout: 2000 });
    if (fifo.error || fifo.status !== 0) return null;
    reader = openSync(input, constants.O_RDONLY | constants.O_NONBLOCK);
    writer = openSync(input, constants.O_WRONLY | constants.O_NONBLOCK);
    // script allocates only our control terminal. GNU remains in the caller's
    // session, with its original stdin, output, and controlling terminal.
    const command = `printf '${token}:S%s\\n' "$(readlink /proc/self/fd/0)"
IFS= read -r _find2fd_hold`;
    allocator = track(spawn("script", ["-q", "-e", "-f", "-E", "never", "-c", command, "/dev/null"], {
      stdio: ["pipe", "pipe", "pipe"], detached: true, env: { ...process.env, SHELL: "/bin/bash", BASH_ENV: "/dev/null" },
    }));
    allocator.stdin.on("error", () => {});
    allocator.stdout.on("data", (data) => {
      controlText += data.toString("ascii");
      if (controlText.length > 65536) { allocatorFailed = true; stop("SIGTERM"); return; }
      let newline;
      while ((newline = controlText.indexOf("\n")) >= 0) {
        const line = controlText.slice(0, newline).replace(/\r$/, ""); controlText = controlText.slice(newline + 1);
        if (line.startsWith(`${token}:S`)) startupResolve(line.slice(token.length + 2));
        else if ([`${token}:A`, `${token}:P`, `${token}:Q`].includes(line)) publish(line.at(-1));
      }
    });
    allocator.stderr.on("data", () => { allocatorFailed = true; startupReject(new Error("control allocator error")); failWaiters(new Error("control allocator error")); });
    allocator.once("error", () => { allocatorFailed = true; startupReject(new Error("control allocator unavailable")); failWaiters(new Error("control allocator unavailable")); });
    allocator.once("close", () => { allocatorFailed = true; startupReject(new Error("control allocator closed")); failWaiters(new Error("control allocator closed")); });
    state.stop = stop;
    const timer = setTimeout(() => startupReject(new Error("control allocator timeout")), INCREMENTAL_START_TIMEOUT_MS);
    let control;
    try { control = await startup; } finally { clearTimeout(timer); }
    if (!/^\/dev\/pts\/\d+$/.test(control)) return null;
    const args = incrementalArguments(plan.options, input, control, sentinel, token);
    const validation = spawnSync("find", incrementalArguments(plan.options, "/dev/null", control, sentinel, token), {
      cwd: plan.cwd, stdio: inheritedStdio(["ignore", "pipe", "pipe"], state.callerDescriptors), timeout: 2000, maxBuffer: 1024 * 1024,
    });
    if (validation.error || validation.signal || validation.status !== 0) return null;
    await delay(0);
    if (allocatorFailed) return null;
    broker = track(spawn("find", args, { cwd: plan.cwd, stdio: inheritedStdio(undefined, state.callerDescriptors) }));
    started = true;
    brokerCompletion = new Promise((done) => {
      broker.once("error", () => { brokerStatus = 1; brokerClosed = true; failWaiters(new Error("GNU evaluator unavailable")); done(1); });
      broker.once("close", (code, signal) => {
        brokerStatus = code ?? helpers.signalStatus(signal); brokerClosed = true;
        // GNU -quit can exit before script forwards its already-written marker.
        // With our FIFO writer still open, a successful evaluator exit is an
        // authoritative quit. Do not turn this pipe scheduling race into failure.
        if (brokerStatus === 0) publish("Q");
        else failWaiters(new Error("GNU evaluator closed"));
        done(brokerStatus);
      });
    });
    // Establish the FIFO reader before closing its bootstrap descriptor or
    // handling a search whose mindepth suppresses every expression evaluation.
    await write(Buffer.concat([Buffer.from(sentinel), Buffer.from([0])]));
    const ready = await next();
    if (ready !== "A") return await brokerCompletion;
    closeSync(reader); reader = undefined; first = false;

    const discovery = async function* (path) {
      const source = join(directory, "discovery");
      writeFileSync(source, Buffer.concat([path, Buffer.from([0])]), { mode: 0o600 });
      const descriptor = openSync(source, "r"), stdio = inheritedStdio(["inherit", "pipe", "inherit"], state.callerDescriptors);
      const inputFd = unusedInputFd(stdio); stdio[inputFd] = descriptor;
      const child = track(spawn("find", ["-P", "-files0-from", `/proc/self/fd/${inputFd}`, "-mindepth", "1", "-maxdepth", "1", "-print0"], { cwd: plan.cwd, stdio }));
      closeSync(descriptor);
      const completion = new Promise((done) => {
        child.once("error", () => done(1)); child.once("close", (code, signal) => done(code ?? helpers.signalStatus(signal)));
      });
      active = { stop: (signal) => {
        const snapshot = stopOwned(child, signal);
        setTimeout(() => helpers.signalSnapshot(snapshot, "SIGKILL"), 250);
      } };
      let buffered = Buffer.alloc(0);
      try {
        for await (const chunk of child.stdout) {
          buffered = buffered.length ? Buffer.concat([buffered, chunk]) : chunk;
          let start = 0, end;
          while ((end = buffered.indexOf(0, start)) >= 0) {
            const childPath = buffered.subarray(start, end); start = end + 1;
            const slash = childPath.lastIndexOf(47), name = childPath.subarray(slash + 1);
            let isDirectory = false;
            const actual = childPath[0] === 47 ? childPath : Buffer.concat([Buffer.from(`${plan.cwd}/`), childPath]);
            try { isDirectory = lstatSync(actual).isDirectory(); } catch { /* Broker reports missing entries. */ }
            yield { name: Buffer.from(name), directory: isDirectory };
          }
          buffered = buffered.subarray(start);
        }
        if (buffered.length) throw new Error("GNU child enumeration protocol");
        discoveryStatus ||= await completion;
      } finally {
        if (child.exitCode === null && !child.signalCode) stopOwned(child, "SIGTERM");
        await completion; active = null;
      }
    };
    const children = async function* (record) {
      let descriptor, ordered, nameBytes = 0;
      try {
        descriptor = openSync(record.actual, constants.O_RDONLY | constants.O_DIRECTORY);
        const queryRoot = `/proc/${process.pid}/fd/${descriptor}`;
        const canonicalRoot = realpathSync(queryRoot, { encoding: "buffer" });
        const args = ["--hidden", "--no-ignore", "--no-ignore-parent", "--show-errors", "--print0", "--color", "never", "--threads", "8", "--absolute-path", "--max-depth", "1", "--search-path", queryRoot, "--", ""];
        active = { signal: state.signal, stop: null };
        const data = await helpers.captureFd(args, plan.cwd, active, INCREMENTAL_CAPTURE_BYTES, program);
        active = null;
        if (state.signal) return;
        if (data === null || data === undefined) throw new Error("fd immediate enumeration failed");
        ordered = orderedChildren(data, canonicalRoot, record.actual);
        nameBytes = ordered.reduce((bytes, child) => bytes + child.name.length, 0);
        if (liveChildren + ordered.length > MAX_LIVE_INCREMENTAL_CHILDREN || liveNameBytes + nameBytes > MAX_LIVE_INCREMENTAL_NAME_BYTES)
          ordered = undefined;
      } catch { /* Continue this directory only; original expression is never replayed. */ }
      finally { if (descriptor !== undefined) closeSync(descriptor); }
      if (ordered) {
        liveChildren += ordered.length; liveNameBytes += nameBytes;
        try { for (const child of ordered) yield child; }
        finally { liveChildren -= ordered.length; liveNameBytes -= nameBytes; }
      }
      else yield* discovery(record.path);
    };
    const visit = async (record) => {
      if (state.signal) return true;
      let result = { quit: false, pruned: false };
      if (record.depth >= plan.options.minDepth) result = await evaluate(record.path);
      if (result.quit) return true;
      if (!record.directory || result.pruned || (plan.options.maxDepth !== null && record.depth >= plan.options.maxDepth)) return false;
      for await (const child of children(record)) {
        const relative = record.relative.length ? Buffer.concat([record.relative, Buffer.from("/"), child.name]) : child.name;
        if (await visit({ path: displayPath(record.root, relative), actual: Buffer.concat([record.actual, Buffer.from("/"), child.name]),
          root: record.root, relative, depth: record.depth + 1, directory: child.directory })) return true;
      }
      return false;
    };
    for (const search of plan.searches) {
      if (await visit({ path: Buffer.from(search.root), actual: Buffer.from(search.path), root: search.root,
        relative: Buffer.alloc(0), depth: 0, directory: search.directory })) break;
    }
    closeWriter();
    return state.signal ? helpers.signalStatus(state.signal) : (await brokerCompletion) || discoveryStatus;
  } catch {
    if (!started) return state.signal ? helpers.signalStatus(state.signal) : null;
    if (!brokerClosed) stop("SIGTERM");
    if (brokerCompletion) await brokerCompletion;
    return state.signal ? helpers.signalStatus(state.signal) : brokerStatus || 1;
  } finally {
    closeWriter();
    if (reader !== undefined) closeSync(reader);
    if (!cleanup) {
      const snapshot = stopOwned(allocator, "SIGTERM", true);
      cleanup = new Promise((done) => setTimeout(() => { helpers.signalSnapshot(snapshot, "SIGKILL"); done(); }, 250));
    }
    await cleanup;
    allocator?.stdout?.destroy(); allocator?.stderr?.destroy(); allocator?.stdin?.destroy();
    rmSync(directory, { recursive: true, force: true });
    state.stop = null;
  }
}
