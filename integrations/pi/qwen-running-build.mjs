/** Local installed-code identity. No network, session, provider-config or token work. */
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { watch } from "node:fs";
import { lstat, readdir, readFile, readlink, realpath, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createBuildProcessRecord } from "./qwen-build-process-record.mjs";

const CODE = new Set([".js", ".mjs", ".cjs", ".ts"]);
const SKIP = new Set(["node_modules", "target", "tests", "test", "captures", "artifacts", ".git"]);
const LEGACY_PATCH_SOURCES = [
  "streaming-usage", "footer", "tool-call-integrity", "internal-guard-retry",
  "precontinuation-compaction", "project-history",
].map((name) => `patch-pi-${name}`);
const STREAMING_PATCH_SOURCES = [
  ...LEGACY_PATCH_SOURCES, "patch-pi-streaming-render",
  "pi_streaming_markdown_patch.py", "pi_streaming_screen_patch.py",
  "pi-streaming-markdown.mjs", "pi-streaming-screen.mjs",
];
const cleanTitle = (text) => String(text ?? "").replace(/[\u0000-\u001f\u007f-\u009f\u061c\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]/gu, "").trim();
const strike = (text) => [...text].map((c) => `${c}\u0336`).join("");

/** Strike the application label; preserve the useful dynamic title and disabled titles. */
export function strikeAppTitle(title, outdated) {
  const text = cleanTitle(title);
  if (!text || !outdated) return text;
  const leading = /^(π|Pi|pi)(?=$|[\s:|/—–-])/u;
  return leading.test(text) ? text.replace(leading, strike) : `${text} - ${strike("π")}`;
}

const stamp = (s) => `${s.dev}:${s.ino}:${s.size}:${s.mtimeNs}:${s.ctimeNs}:${s.mode}`;
const asPath = (path) => path instanceof URL || String(path).startsWith("file:") ? fileURLToPath(path) : resolve(String(path));

/**
 * Snapshot actual loaded code once, then compare the installed launch target.
 * Unknown candidates retain the last established state. Call start before changing
 * installed files; preserve this object across extension reloads. SDK scope is the
 * compiled dist JS of sibling @earendil-works packages, plus explicit extension
 * roots and their literal local imports. Other dependencies are outside this scope.
 */
export function createRunningBuildMonitor({
  entrypoint = process.argv[1],
  launcher = process.env.QWEN_PI_LATEST_LAUNCHER ?? join(homedir(), ".local", "bin", "pi"),
  releaseRoot = process.env.QWEN_PI_RUNNING_RELEASE,
  extensionRoots = [],
  extensionRootProvider,
  profileFiles = [],
  recordBuild = false,
  recordOptions = {},
  onChange = () => {},
  debounceMs = 1000,
  pollMs = 30000,
  maxFiles = 4096,
  maxDirectories = 2048,
  maxBytes = 64 * 1024 * 1024,
  watchFactory = watch,
} = {}) {
  let outdated = false;
  let stopped = false;
  let started = false;
  let baseline;
  let baselineRelease;
  let readyPromise;
  let inFlight;
  let debounceTimer;
  let pollTimer;
  let validationChild;
  const cache = new Map();
  const syntaxCache = new Map();
  const watchers = new Map();
  let wantedWatches = new Map();
  const loadedEntry = entrypoint ? asPath(entrypoint) : undefined;
  const stableLauncher = asPath(launcher);
  const configuredRoots = extensionRoots.map(asPath);
  let roots = [];
  let runningIdentity;
  let latestEntry;
  let identityError;
  const failure = (stage, error) => `${stage}: ${error.code ?? error.message}${error.path ? ` (${error.path})` : ""}`.slice(0, 1024);
  const profiles = profileFiles.map(asPath);
  const recorder = recordBuild ? createBuildProcessRecord(recordOptions) : undefined;

  async function loadedExtensionRoots() {
    const selected = [];
    const paths = [...new Set([...configuredRoots, ...(extensionRootProvider ? await extensionRootProvider() : [])].map(asPath))];
    for (const path of paths) {
      const info = await stat(path);
      if (!info.isDirectory()) { selected.push(path); continue; }
      // Pi also accepts extension package directories. Resolve their conventional
      // index module only; unsupported directories never suppress SDK monitoring
      // or trigger recursive reads of unrelated/private files.
      for (const name of ["index.ts", "index.js", "index.mjs", "index.cjs"]) {
        const index = join(path, name);
        try {
          if ((await stat(index)).isFile()) { selected.push(index); break; }
        } catch (error) { if (error.code !== "ENOENT") throw error; }
      }
    }
    return selected;
  }

  function addWatch(path, names, all = false) {
    const directory = all ? path : dirname(path);
    const name = all ? "*" : relative(directory, path);
    if (!names.has(directory)) names.set(directory, new Set());
    names.get(directory).add(name);
  }

  async function traceLinks(path, names) {
    // Observe directory symlinks as well as the final launcher symlink.
    let current = resolve(path);
    for (let hop = 0; hop < 40; hop += 1) {
      const parts = current.split("/").filter(Boolean);
      let prefix = "/";
      let changed = false;
      for (let i = 0; i < parts.length; i += 1) {
        prefix = join(prefix, parts[i]);
        addWatch(prefix, names);
        const info = await lstat(prefix);
        if (info.isSymbolicLink()) {
          const target = await readlink(prefix);
          current = resolve(dirname(prefix), target, ...parts.slice(i + 1));
          changed = true;
          break;
        }
      }
      if (!changed) return current;
    }
    throw new Error("launcher symlink depth exceeded");
  }

  async function content(path, names, budget) {
    addWatch(path, names);
    const actual = await traceLinks(path, names);
    const before = await stat(actual, { bigint: true });
    if (!before.isFile() || before.size <= 0n || before.size > BigInt(maxBytes)) throw new Error("unknown code file");
    budget.files += 1;
    budget.bytes += Number(before.size);
    if (budget.files > maxFiles || budget.bytes > maxBytes) throw new Error("code identity budget exceeded");
    const key = stamp(before);
    let entry = cache.get(actual);
    if (!entry || entry.stamp !== key) {
      const bytes = await readFile(actual);
      const text = bytes.toString("utf8");
      if (stamp(await stat(actual, { bigint: true })) !== key) throw new Error("code changed while reading");
      entry = { stamp: key, text, hash: createHash("sha256").update(bytes).digest("hex") };
      cache.set(actual, entry);
    }
    return entry;
  }

  async function snapshot(entry, names, selectedRoots = roots, selectedProfiles = profiles) {
    const budget = { files: 0, bytes: 0, directories: 0 };
    const pieces = [];
    const sdkFiles = [];
    const extensionPieces = [];
    const seen = new Map();
    const actual = await traceLinks(entry, names);
    async function put(path, label) {
      const prior = seen.get(path);
      const item = prior ?? await content(path, names, budget);
      if (label.startsWith("sdk/")) sdkFiles.push({ path: label.slice(4), sha256: item.hash });
      if (label.startsWith("extension/") || label.startsWith("profile/")) extensionPieces.push(`${label}\0${item.hash}`);
      if (prior) return undefined;
      seen.set(path, item);
      pieces.push(`${label}\0${item.hash}`);
      return item.text;
    }
    await put(actual, "entrypoint");
    const marker = `${join("node_modules", "@earendil-works")}/`;
    const scopedAt = actual.lastIndexOf(marker);
    if (scopedAt !== -1) {
      const scope = actual.slice(0, scopedAt + marker.length - 1);
      addWatch(scope, names, true);
      async function walk(directory, label) {
        budget.directories += 1;
        if (budget.directories > maxDirectories) throw new Error("code directory budget exceeded");
        addWatch(directory, names, true);
        const entries = await readdir(directory, { withFileTypes: true });
        for (const item of entries.sort((a, b) => a.name.localeCompare(b.name))) {
          if (SKIP.has(item.name)) continue;
          const path = join(directory, item.name);
          if (item.isDirectory()) await walk(path, `${label}/${item.name}`);
          else if ((item.isFile() || item.isSymbolicLink()) && [".js", ".mjs", ".cjs"].includes(extname(item.name))) await put(path, `${label}/${item.name}`);
        }
      }
      for (const item of (await readdir(scope, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
        if (!item.isDirectory() && !item.isSymbolicLink()) continue;
        const dist = join(scope, item.name, "dist");
        addWatch(dist, names);
        try { await stat(dist); } catch (error) { if (error.code === "ENOENT") continue; throw error; }
        await walk(dist, `sdk/${item.name}/dist`);
      }
    }
    async function extension(path, label) {
      const actualPath = await traceLinks(path, names);
      const text = await put(actualPath, label);
      if (text === undefined) return;
      // Literal local module dependencies only; never follow package imports or
      // scan a workspace, private sessions, generated targets or capture trees.
      const imports = text.matchAll(/(?:\b(?:import|export)\s+(?:[^;\n]*?\s+from\s*)?|\bimport\s*\(\s*)["'](\.[^"']+)["']/gu);
      for (const match of imports) {
        const target = resolve(dirname(actualPath), match[1]);
        if (match[1].split("/").some((part) => SKIP.has(part)) || !CODE.has(extname(target))) continue;
        await extension(target, `${label}/${match[1]}`);
      }
    }
    for (let i = 0; i < selectedRoots.length; i += 1) {
      extensionPieces.push(`root/${i}\0${selectedRoots[i]}`);
      await extension(selectedRoots[i], `extension/${i}`);
    }
    for (let i = 0; i < selectedProfiles.length; i += 1) await put(selectedProfiles[i], `profile/${i}`);
    const digest = (text) => createHash("sha256").update(text).digest("hex");
    sdkFiles.sort((a, b) => Buffer.compare(Buffer.from(a.path), Buffer.from(b.path)));
    return {
      fingerprint: digest(pieces.sort().join("\n") + JSON.stringify(selectedRoots)),
      sdk_fingerprint: digest(sdkFiles.map((file) => `${file.path}\0${file.sha256}\n`).join("")),
      extensions_fingerprint: digest(extensionPieces.sort().join("\n")),
      extensions: [...selectedRoots],
    };
  }

  function schedule() {
    if (stopped) return;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { void check(); }, debounceMs);
    debounceTimer.unref?.();
  }

  function updateWatches(names) {
    if (stopped) return;
    wantedWatches = names;
    for (const [directory, watcher] of watchers) {
      if (!names.has(directory)) { watcher.close(); watchers.delete(directory); }
    }
    for (const directory of names.keys()) {
      if (watchers.has(directory)) continue;
      try {
        const watcher = watchFactory(directory, { persistent: false }, (_event, filename) => {
          const watched = wantedWatches.get(directory);
          if (!filename || watched?.has("*") || watched?.has(String(filename))) schedule();
        });
        watcher.on?.("error", () => { watcher.close(); watchers.delete(directory); });
        watchers.set(directory, watcher);
      } catch { /* Missing/unwatchable parents are covered by periodic rechecks. */ }
    }
  }

  async function validCli(path, names) {
    const item = await content(path, names, { files: 0, bytes: 0 });
    if (syntaxCache.has(item.hash)) return syntaxCache.get(item.hash);
    // Syntax-check changed CLI bytes through stdin. No import, candidate execution,
    // version lookup, NODE_OPTIONS preload or unbounded output is allowed here.
    const valid = await new Promise((done) => {
      const env = { ...process.env };
      delete env.NODE_OPTIONS;
      const child = spawn(process.execPath, ["--input-type=module", "--check"], { env, stdio: ["pipe", "ignore", "pipe"] });
      validationChild = child;
      let output = 0;
      let transientFailure = false;
      const timer = setTimeout(() => { transientFailure = true; child.kill("SIGKILL"); }, 2000);
      timer.unref?.();
      child.stderr.on("data", (chunk) => { output += chunk.length; if (output > 8192) { transientFailure = true; child.kill("SIGKILL"); } });
      child.on("error", () => { clearTimeout(timer); done(undefined); });
      child.on("close", (code) => { clearTimeout(timer); if (validationChild === child) validationChild = undefined; done(!transientFailure && [0, 1].includes(code) ? code === 0 : undefined); });
      child.stdin.on("error", () => {});
      child.stdin.end(item.text);
    });
    // A resource timeout or failed spawn is unknown, not a permanently invalid
    // digest: retry after the next filesystem event or periodic check.
    if (valid !== undefined) syntaxCache.set(item.hash, valid);
    if (syntaxCache.size > 128) syntaxCache.delete(syntaxCache.keys().next().value);
    return valid === true;
  }

  async function packageInfo(entry, names) {
    const manifest = join(dirname(dirname(entry)), "package.json");
    const item = await content(manifest, names, { files: 0, bytes: 0 });
    if (item.text.length > 128 * 1024) throw new Error("Package manifest exceeds bound");
    const value = JSON.parse(item.text);
    if (value.name !== "@earendil-works/pi-coding-agent" || typeof value.version !== "string" || value.version.length > 128) throw new Error("Unknown Pi package");
    return { package: value.name, version: value.version, package_manifest: manifest };
  }

  async function installedStatus(entry, identity, names) {
    const manifestPath = join(dirname(dirname(entry)), "coherence-build.json");
    try {
      const item = await content(manifestPath, names, { files: 0, bytes: 0 });
      if (item.text.length > 512 * 1024) throw new Error("Installed build manifest exceeds bound");
      const manifest = JSON.parse(item.text);
      if (manifest.schema !== "coherence.pi.installed-build.v1" || manifest.package !== "@earendil-works/pi-coding-agent"
        || manifest.entry_point !== "dist/cli.js" || manifest.patches_verified !== true || !/^[a-f0-9]{64}$/.test(manifest.sdk_fingerprint)
        || !Array.isArray(manifest.patches)) throw new Error("Unknown installed build manifest");
      const sources = manifest.patches.length === LEGACY_PATCH_SOURCES.length
        ? LEGACY_PATCH_SOURCES : STREAMING_PATCH_SOURCES;
      if (manifest.patches.length !== sources.length) throw new Error("Unknown installed patch set");
      const expected = new Set(sources);
      let changed = manifest.sdk_fingerprint !== identity.sdk_fingerprint;
      for (const patch of manifest.patches) {
        if (typeof patch.path !== "string" || patch.path.length > 4096 || !/^[a-f0-9]{64}$/.test(patch.sha256)) throw new Error("Unknown patch identity");
        const path = resolve(dirname(manifestPath), patch.path);
        if (!expected.delete(path.split("/").at(-1))) throw new Error("Unknown patch source");
        const source = await content(path, names, { files: 0, bytes: 0 });
        if (source.hash !== patch.sha256) changed = true;
      }
      return { manifest_path: manifestPath, patches_status: changed ? "outdated" : "verified" };
    } catch {
      return { manifest_path: manifestPath, patches_status: "unknown" };
    }
  }

  async function publish(identity, names, error = false) {
    if (!recorder || !runningIdentity || stopped) return;
    const installed = identity && !error ? await installedStatus(latestEntry, identity, names)
      : { manifest_path: latestEntry ? join(dirname(dirname(latestEntry)), "coherence-build.json") : "", patches_status: "unknown" };
    const changed = identity && (identity.sdk_fingerprint !== baseline.sdk_fingerprint || identity.extensions_fingerprint !== baseline.extensions_fingerprint || outdated);
    const status = error || installed.patches_status === "unknown" ? "unknown"
      : installed.patches_status === "outdated" ? "installation_update_required" : changed ? "restart_required" : "current";
    const reason = status === "unknown" ? "Current installation could not be verified"
      : status === "installation_update_required" ? "Installed code or patch sources differ from the verified installation manifest"
        : status === "restart_required" ? "Running code, selected extensions or launch profile differs from the current installation"
          : "Verified installation and captured startup identity agree";
    try {
      await recorder.publish({ ...runningIdentity, latest: { ...installed, entry_point: latestEntry ?? "",
        sdk_fingerprint: identity?.sdk_fingerprint ?? "", extensions_fingerprint: identity?.extensions_fingerprint ?? "",
        checked_at_unix_ms: Date.now(), status, reason } });
    } catch (error) { identityError = failure("record publication", error); }
  }

  async function inspect() {
    const names = new Map();
    // Keep launch indirections under observation even while a replacement is absent.
    addWatch(stableLauncher, names);
    try {
      const target = await traceLinks(stableLauncher, names);
      let candidate = target;
      let releaseChanged = false;
      if (releaseRoot) {
        const candidateRoot = dirname(dirname(target));
        if (target !== join(candidateRoot, "scripts", "pi-radiance.py")) throw new Error("unknown release launcher");
        await content(target, names, { files: 0, bytes: 0 });
        candidate = join(candidateRoot, "runtime", "node_modules", "@earendil-works", "pi-coding-agent", "dist", "cli.js");
        releaseChanged = candidateRoot !== baselineRelease;
      }
      latestEntry = candidate;
      const currentRoots = extensionRootProvider ? await loadedExtensionRoots() : roots;
      const remap = (path) => releaseRoot && baselineRelease && path.startsWith(`${baselineRelease}/`)
        ? join(dirname(dirname(target)), relative(baselineRelease, path)) : path;
      const identity = await snapshot(candidate, names, currentRoots.map(remap), profiles.map(remap));
      if (!await validCli(candidate, names)) throw new Error("incomplete CLI candidate");
      const next = releaseChanged || identity.fingerprint !== baseline.fingerprint;
      if (!stopped && next !== outdated) {
        outdated = next;
        try { onChange(outdated); } catch { /* Display failures cannot stop monitoring. */ }
      }
      await publish(identity, names);
    } catch (error) {
      identityError = failure("current installation", error);
      // Unknown candidates preserve the title marker but publish explicit unknown
      // evidence rather than letting an old "current" receipt remain authoritative.
      await publish(undefined, names, true);
    }
    updateWatches(names);
    return outdated;
  }

  function start() {
    if (readyPromise) return readyPromise;
    started = true;
    readyPromise = (async () => {
      if (!loadedEntry || stopped) return;
      const names = new Map();
      try {
        roots = await loadedExtensionRoots();
        baselineRelease = releaseRoot ? await realpath(asPath(releaseRoot)) : undefined;
        baseline = await snapshot(loadedEntry, names);
        const running = await content(loadedEntry, names, { files: 0, bytes: 0 });
        syntaxCache.set(running.hash, true);
        if (recorder) {
          try {
            const info = await packageInfo(loadedEntry, names);
            runningIdentity = { ...info, entry_point: loadedEntry, captured_at_unix_ms: Date.now(),
              running: { sdk_fingerprint: baseline.sdk_fingerprint, extensions_fingerprint: baseline.extensions_fingerprint, extensions: baseline.extensions },
              scope: "Compiled Pi SDK, declared extensions and their literal local imports, and launch profile" };
          } catch (error) { identityError = failure("running package", error); }
        }
      } catch (error) { identityError = failure("startup snapshot", error); updateWatches(names); return; }
      if (stopped) return;
      updateWatches(names);
      await inspect();
      if (stopped) return;
      pollTimer = setInterval(() => { void check(); }, pollMs);
      pollTimer.unref?.();
    })();
    return readyPromise;
  }

  async function check() {
    if (stopped) return outdated;
    if (!started) await start();
    else await readyPromise;
    if (stopped || !baseline) return outdated;
    if (!inFlight) inFlight = inspect().finally(() => { inFlight = undefined; });
    return inFlight;
  }

  function stop() {
    stopped = true;
    clearTimeout(debounceTimer);
    clearInterval(pollTimer);
    validationChild?.kill("SIGKILL");
    recorder?.stop();
    for (const watcher of watchers.values()) watcher.close();
    watchers.clear();
    cache.clear();
    syntaxCache.clear();
  }

  return { start, check, stop, get ready() { return readyPromise ?? start(); }, get outdated() { return outdated; }, get recordPath() { return recorder?.path; }, get identityError() { return recorder?.error ?? identityError; } };
}
