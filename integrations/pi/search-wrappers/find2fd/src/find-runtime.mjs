import { spawn, spawnSync } from "node:child_process";
import { isUtf8 } from "node:buffer";
import { closeSync, lstatSync, mkdtempSync, openSync, readFileSync, readlinkSync, realpathSync, rmdirSync, statSync, unlinkSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { evaluateFind, fdArguments, parseFindOptions, rootName } from "./find-options.mjs";
import { decodeCandidates, displayPath, MAX_CANDIDATES, orderCandidates } from "./find-order.mjs";
import { createEvaluationRequests, MAX_EVALUATION_RECORDS } from "./find-evaluate.mjs";
import { inheritedStdio } from "./find-descriptors.mjs";
import { incrementalPathReason, runIncremental } from "./find-incremental.mjs";

export const MAX_CAPTURE_BYTES = 32 * 1024 * 1024;
export const MAX_OUTPUT_RECORDS = MAX_CANDIDATES;
export const FD_TIMEOUT_MS = 10000;
const nativePlan = (argv, cwd, reason) => ({ backend: "find", args: argv, cwd, reason });

function currentDirectory() {
  try { return process.cwd(); } catch { return undefined; }
}

export function planFind(argv, configuration = {}) {
  if (["off", "0"].includes(process.env.FIND2FD_ACCELERATION))
    return nativePlan(argv, configuration.cwd ?? currentDirectory(), "explicit GNU routing");
  if (process.env.FIND2FD_EXPERIMENTAL_FD !== "1") {
    if (Object.hasOwn(process.env, "FIND2FD_EXPERIMENTAL_FD") || ["off", "0"].includes(process.env.FIND2FD_ACCELERATION))
      return nativePlan(argv, configuration.cwd ?? currentDirectory(), "explicit GNU routing");
    const policy = fileURLToPath(new URL("./find-auto.sh", import.meta.url));
    const decision = spawnSync("/usr/bin/bash", ["--noprofile", "--norc", "-c", 'source "$1"; shift; _find2fd_auto_eligible "$@"', "find2fd-auto", policy, ...argv],
      { cwd: configuration.cwd ?? currentDirectory(), stdio: "ignore", timeout: 10000 });
    if (decision.error || decision.status !== 0)
      return nativePlan(argv, configuration.cwd ?? currentDirectory(), "GNU wins small/dense/shallow or unqualified automatic class");
  }
  const actualCwd = currentDirectory();
  const cwd = configuration.cwd ?? actualCwd;
  const parsed = parseFindOptions(argv, { cwd, callerDescriptors: configuration.callerDescriptors });
  if (!parsed.eligible) return nativePlan(argv, cwd, parsed.reason);
  if (process.platform !== "linux") return nativePlan(argv, cwd, "platform is outside the Linux qualification");
  if (cwd === undefined) return nativePlan(argv, cwd, "current directory path is unavailable");
  if (cwd === actualCwd) {
    try {
      if (!isUtf8(readlinkSync("/proc/self/cwd", { encoding: "buffer" })))
        return nativePlan(argv, cwd, "current directory contains non-UTF-8 bytes");
    } catch { return nativePlan(argv, cwd, "cannot qualify current directory bytes"); }
  }
  const options = parsed.options;
  if (options.contextualPrintf && process.stdout.isTTY)
    return nativePlan(argv, cwd, "GNU terminal quoting of contextual printf fields");
  if (!["gnu-evaluate", "gnu-incremental"].includes(options.mode) && options.action === "print" && process.stdout.isTTY)
    return nativePlan(argv, cwd, "GNU terminal pathname quoting");
  if (options.roots.length > 256) return nativePlan(argv, cwd, "starting root count bound");
  const searches = [];
  try {
    for (const root of options.roots) {
      let path = resolve(cwd, root);
      let entry = lstatSync(path);
      if (["L", "H"].includes(options.dereference)) {
        try { entry = statSync(path); }
        catch (error) { if (!["ENOENT", "ENOTDIR"].includes(error.code)) throw error; }
        if (entry.isDirectory()) path = realpathSync(path);
      }
      const directory = entry.isDirectory();
      searches.push({ root, path, entry, directory, args: directory ? fdArguments(path, options) : [] });
    }
  } catch { return nativePlan(argv, cwd, "GNU starting path diagnostics"); }
  if (options.mode === "gnu-incremental") {
    const reason = incrementalPathReason(options, searches, cwd);
    if (reason) return nativePlan(argv, cwd, reason);
  }
  return { backend: "fd", mode: options.mode === "gnu-incremental" ? "incremental" : "listing", args: searches.find((search) => search.directory)?.args ?? [], searches, options, cwd };
}

const signalStatus = (signal) => ({ SIGINT: 130, SIGTERM: 143, SIGPIPE: 141, SIGHUP: 129 })[signal] ?? 1;

function processIdentity(pid) {
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
    return stat.slice(stat.lastIndexOf(")") + 2).split(" ")[19];
  } catch { return null; }
}

function ownedDescendants(pid) {
  const pending = [pid], seen = new Set(), records = [];
  while (pending.length && records.length < 4096) {
    const next = pending.pop();
    if (seen.has(next)) continue;
    seen.add(next);
    const identity = processIdentity(next);
    if (!identity) continue;
    records.push({ pid: next, identity });
    try {
      const children = readFileSync(`/proc/${next}/task/${next}/children`, "ascii").trim().split(/\s+/);
      for (const child of children) if (/^\d+$/.test(child) && Number(child) > 0) pending.push(Number(child));
    } catch { /* An exited or inaccessible process has no capturable children. */ }
  }
  return records;
}

function signalSnapshot(records, signal) {
  for (const record of [...records].reverse()) {
    // Start time prevents a PID reused after exit from receiving our signal.
    if (processIdentity(record.pid) !== record.identity) continue;
    try { process.kill(record.pid, signal); } catch { /* Already exited. */ }
  }
}

function runNative(argv, cwd, state, input, inputFd = 3) {
  return new Promise((done) => {
    // Evaluation receives paths on a private descriptor. Original stdin and
    // terminal streams remain inherited, including GNU's pathname quoting.
    let descriptor, directory, filename, child;
    try {
      if (input !== undefined) {
        // Node's "pipe" is a Unix socket and cannot be reopened through /proc.
        // An unlinked private regular file provides a seekable fd without
        // taking over the caller's stdin or exposing a persistent pathname.
        directory = mkdtempSync(join(tmpdir(), "find2fd-input-"));
        filename = join(directory, "paths");
        descriptor = openSync(filename, "wx+", 0o600);
        unlinkSync(filename); filename = undefined;
        rmdirSync(directory); directory = undefined;
        let offset = 0;
        while (offset < input.length) offset += writeSync(descriptor, input, offset, input.length - offset, offset);
      }
      const stdio = inheritedStdio(undefined, state.callerDescriptors);
      if (descriptor !== undefined) stdio[inputFd] = descriptor;
      child = spawn("find", argv, { cwd, stdio });
    } catch (error) {
      process.stderr.write(`find: could not prepare evaluation input: ${error.code ?? "input failure"}\n`);
      done(1); return;
    } finally {
      if (descriptor !== undefined) closeSync(descriptor);
      if (filename) try { unlinkSync(filename); } catch { /* Owned partial input. */ }
      if (directory) try { rmdirSync(directory); } catch { /* Owned directory. */ }
    }
    let cleanup;
    state.stop = (signal) => {
      if (cleanup) return;
      // Preserve native sessions/groups normally. On cancellation only, pause
      // our find child while snapshotting its current descendants, so killing
      // an -exec child cannot make find begin another action before it stops.
      child.kill("SIGSTOP");
      const descendants = child.pid ? ownedDescendants(child.pid) : [];
      signalSnapshot(descendants, signal);
      child.kill(signal); child.kill("SIGCONT");
      cleanup = new Promise((resolveCleanup) => setTimeout(() => {
        signalSnapshot(descendants, "SIGKILL"); resolveCleanup();
      }, 250));
    };
    child.once("error", (error) => {
      process.stderr.write(`find: could not start find: ${error.code ?? "spawn failure"}\n`);
      state.stop = null; done(1);
    });
    child.once("close", async (code, signal) => {
      state.stop = null;
      if (cleanup) await cleanup;
      done(state.signal ? signalStatus(state.signal) : code ?? signalStatus(signal));
    });
  });
}

function captureFd(args, cwd, state, remainingBytes, program = "fd") {
  return new Promise((done) => {
    // fd never needs stdin for this subset. Its private process group permits
    // cancellation of owned helper descendants without touching the caller.
    const child = spawn(program, args, { cwd, detached: true, stdio: ["ignore", "pipe", "pipe"] });
    const groupIdentity = child.pid ? processIdentity(child.pid) : null;
    const buffers = [];
    let bytes = 0, failed = false, missing = false, cleanup;
    const killGroup = (signal) => {
      if (!child.pid) return;
      const identity = processIdentity(child.pid);
      // A surviving original group retains its PGID after the leader exits.
      // Do not signal a new leader if that ID has since been reused.
      if (groupIdentity && identity && identity !== groupIdentity) return;
      try { process.kill(-child.pid, signal); } catch { child.kill(signal); }
    };
    const stop = (signal = "SIGTERM") => {
      failed = true;
      killGroup(signal);
      if (!cleanup) cleanup = new Promise((resolveCleanup) => setTimeout(() => {
        killGroup("SIGKILL"); resolveCleanup();
      }, 250));
    };
    state.stop = stop;
    const timeout = setTimeout(stop, FD_TIMEOUT_MS);
    child.stdout.on("data", (data) => {
      bytes += data.length;
      if (bytes > remainingBytes) stop();
      else if (!failed) buffers.push(data);
    });
    child.stderr.on("data", (data) => { if (data.length) stop(); });
    child.once("error", (error) => { failed = true; missing = error.code === "ENOENT"; });
    child.once("close", async (code, signal) => {
      clearTimeout(timeout);
      // Leader exit does not prove the private group is empty: a helper may
      // ignore termination and hold no pipes. Finish its scheduled cleanup.
      if (cleanup) await cleanup;
      state.stop = null;
      done(missing ? undefined : !failed && code === 0 && !signal ? Buffer.concat(buffers, bytes) : null);
    });
  });
}

function writeOutput(data, state) {
  return new Promise((done) => {
    let settled = false;
    const finish = (status) => {
      if (settled) return;
      settled = true; state.stop = null;
      // Node can call the write callback before emitting the corresponding
      // stream error. Keep the handler through that turn to consume EPIPE.
      setImmediate(() => { process.stdout.removeListener("error", onError); done(status); });
    };
    const onError = (error) => finish(error.code === "EPIPE" ? 141 : 1);
    process.stdout.on("error", onError);
    state.stop = () => { process.stdout.destroy(); finish(signalStatus(state.signal)); };
    process.stdout.write(data, (error) => error ? onError(error) : finish(0));
  });
}

export async function runFind(argv, configuration = {}) {
  const cwd = configuration.cwd ?? currentDirectory();
  // An inherited OS cwd retains bytes that process.cwd() cannot represent.
  const nativeCwd = configuration.cwd === undefined ? undefined : cwd;
  const state = { signal: null, stop: null, callerDescriptors: configuration.callerDescriptors };
  const interrupted = (signal) => { state.signal = signal; state.stop?.(signal); };
  const onInt = () => interrupted("SIGINT"), onTerm = () => interrupted("SIGTERM"), onHup = () => interrupted("SIGHUP");
  process.on("SIGINT", onInt); process.on("SIGTERM", onTerm); process.on("SIGHUP", onHup);
  try {
    const plan = planFind(argv, { cwd, callerDescriptors: configuration.callerDescriptors });
    if (plan.backend === "find") return await runNative(argv, nativeCwd, state);
    const { options } = plan;
    if (options.mode === "gnu-incremental") {
      const status = await runIncremental(plan, state, { captureFd, processIdentity, ownedDescendants, signalSnapshot, signalStatus });
      return status === null ? await runNative(argv, nativeCwd, state) : status;
    }
    const records = [], terminators = { print: Buffer.from([10]), print0: Buffer.from([0]) };
    const evaluationRecords = [];
    let evaluationBytes = 0;
    const appendEvaluation = (record) => {
      evaluationBytes += record.path.length + 1;
      if (evaluationRecords.length >= MAX_EVALUATION_RECORDS || evaluationBytes > MAX_CAPTURE_BYTES)
        throw new Error("GNU evaluation input bound");
      evaluationRecords.push(record);
    };
    const budget = { entries: 0, parents: 0 };
    let captured = 0, outputBytes = 0, totalCandidates = 0, outputRecords = 0;
    const append = (path, action) => {
      outputBytes += path.length + 1;
      if (++outputRecords > MAX_OUTPUT_RECORDS) throw new Error("formatted output record bound");
      if (outputBytes > MAX_CAPTURE_BYTES) throw new Error("formatted output byte bound");
      records.push(path, terminators[action]);
    };
    try {
      for (const search of plan.searches) {
        let data = search.directory ? await captureFd(search.args, cwd, state, MAX_CAPTURE_BYTES - captured) : Buffer.alloc(0);
        if (search.directory && data === undefined && !state.signal)
          data = await captureFd(search.args, cwd, state, MAX_CAPTURE_BYTES - captured, "fdfind");
        if (state.signal) return signalStatus(state.signal);
        if (data === null || data === undefined) throw new Error("fd dependency, error, timeout or response bound");
        captured += data.length;
        const candidates = decodeCandidates(data, search.path);
        totalCandidates += candidates.length;
        if (totalCandidates > MAX_CANDIDATES) throw new Error("total candidate count bound");
        const ordered = orderCandidates(search.path, candidates, options, budget);
        if (options.xdev && options.dereference === "L" && ordered.some((candidate) => candidate.entry.isDirectory() &&
          statSync(Buffer.concat([Buffer.from(search.path === "/" ? "/" : `${search.path}/`), candidate.relative])).dev !== search.entry.dev))
          throw new Error("GNU logical cross-filesystem boundary evaluation");
        if (options.mode === "gnu-evaluate") {
          const rootRecord = { path: Buffer.from(search.root), root: search.root, relative: Buffer.alloc(0), depth: 0 };
          if (!options.minDepth && !options.depth) appendEvaluation(rootRecord);
          for (const candidate of ordered)
            appendEvaluation({ path: displayPath(search.root, candidate.relative), root: search.root, relative: candidate.relative, depth: candidate.parts.length });
          if (!options.minDepth && options.depth) appendEvaluation(rootRecord);
          continue;
        }
        const appendRoot = () => {
          if (options.minDepth) return;
          const path = Buffer.from(search.root);
          for (const action of evaluateFind(options.expression, { name: rootName(search.root), path, entry: search.entry }).actions) append(path, action);
        };
        if (!options.depth) appendRoot();
        for (const candidate of ordered) {
          const path = displayPath(search.root, candidate.relative);
          for (const action of evaluateFind(options.expression, { name: candidate.parts.at(-1), path, entry: candidate.entry }).actions) append(path, action);
        }
        if (options.depth) appendRoot();
      }
    } catch {
      if (state.signal) return signalStatus(state.signal);
      return await runNative(argv, nativeCwd, state);
    }
    if (state.signal) return signalStatus(state.signal);
    if (options.mode === "gnu-evaluate") {
      let requests;
      try { requests = createEvaluationRequests(options, evaluationRecords); }
      catch { return await runNative(argv, nativeCwd, state); }
      for (const request of requests) {
        if (state.signal) return signalStatus(state.signal);
        const status = await runNative(request.args, cwd, state, request.input, request.inputFd);
        if (status) return status;
      }
      return 0;
    }
    if (!outputBytes) return 0;
    return await writeOutput(Buffer.concat(records, outputBytes), state);
  } finally {
    process.removeListener("SIGINT", onInt); process.removeListener("SIGTERM", onTerm); process.removeListener("SIGHUP", onHup);
  }
}
