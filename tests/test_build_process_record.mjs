import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { chmod, lstat, mkdir, mkdtemp, readFile, readdir, rename, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import test from "node:test";
import { createBuildProcessRecord, processStartTicks } from "../integrations/pi/qwen-build-process-record.mjs";
import { createRunningBuildMonitor } from "../integrations/pi/qwen-running-build.mjs";

const hash = (text) => createHash("sha256").update(text).digest("hex");
const legacyPatchSources = [
  "streaming-usage", "footer", "tool-call-integrity", "internal-guard-retry",
  "precontinuation-compaction", "project-history",
].map((name) => `patch-pi-${name}`);
const streamingPatchSources = [
  ...legacyPatchSources, "patch-pi-streaming-render",
  "pi_streaming_markdown_patch.py", "pi_streaming_screen_patch.py",
  "pi-streaming-markdown.mjs", "pi-streaming-screen.mjs",
];
async function write(path, text) { await mkdir(dirname(path), { recursive: true }); await writeFile(path, text); }
async function rootFor(t) {
  const root = await mkdtemp(join(tmpdir(), "pi-build-record-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}
const sample = (latest = { status: "current" }) => ({
  package: "@earendil-works/pi-coding-agent", version: "0.84.2", entry_point: "/owned/pi/dist/cli.js",
  captured_at_unix_ms: 1234, running: { sdk_fingerprint: "a".repeat(64), extensions_fingerprint: "b".repeat(64), extensions: ["/owned/extension.mjs"] }, latest,
});
const json = async (path) => JSON.parse(await readFile(path, "utf8"));

test("process start identity parses comm names containing spaces and closing parentheses", async () => {
  const stat = await readFile(`/proc/${process.pid}/stat`, "utf8");
  const expected = stat.slice(stat.lastIndexOf(")") + 2).trim().split(/\s+/)[19];
  assert.match(expected, /^\d+$/);
  assert.equal(processStartTicks(stat), expected);
  assert.equal(processStartTicks(stat.replace(/\([^]*\)/, "(pi ) private name)")), expected);
  assert.throws(() => processStartTicks("invalid"), /start identity/);
});

test("private record captures real PID/start ticks, modes and immutable startup identity", async (t) => {
  const runtimeDir = await rootFor(t);
  const recorder = createBuildProcessRecord({ runtimeDir });
  t.after(() => recorder.stop());
  const first = sample();
  await recorder.publish(first);
  const value = await json(recorder.path);
  const ticks = processStartTicks(await readFile(`/proc/${process.pid}/stat`, "utf8"));
  assert.equal(recorder.path, join(runtimeDir, "pi", "processes", `${process.pid}-${ticks}.json`));
  assert.equal(value.schema, "coherence.pi.process-build.v1");
  assert.equal(value.pid, process.pid);
  assert.equal(value.uid, process.getuid());
  assert.equal(value.start_ticks, ticks);
  assert.equal((await lstat(recorder.path)).mode & 0o777, 0o600);
  for (const directory of [join(runtimeDir, "pi"), join(runtimeDir, "pi", "processes")]) assert.equal((await lstat(directory)).mode & 0o777, 0o700);
  first.running.extensions.push("/mutated-after-publish.mjs");
  await recorder.publish({ ...sample({ status: "restart_required", checked_at_unix_ms: 9000 }), version: "changed", captured_at_unix_ms: 9999, running: { sdk_fingerprint: "c".repeat(64) } });
  const next = await json(recorder.path);
  assert.equal(next.version, "0.84.2");
  assert.equal(next.captured_at_unix_ms, 1234);
  assert.deepEqual(next.running.extensions, ["/owned/extension.mjs"]);
  assert.equal(next.running.sdk_fingerprint, "a".repeat(64));
  assert.deepEqual(next.latest, { status: "restart_required", checked_at_unix_ms: 9000 });
});

test("serialized atomic publication never exposes partial JSON and removes its own record", async (t) => {
  const runtimeDir = await rootFor(t);
  const recorder = createBuildProcessRecord({ runtimeDir });
  t.after(() => recorder.stop());
  await recorder.publish(sample());
  const writes = Array.from({ length: 24 }, (_, sequence) => recorder.publish(sample({ status: "current", sequence, padding: "x".repeat(8192) })));
  let reads = 0;
  const readChecks = (async () => {
    for (let index = 0; index < 40; index += 1) {
      const value = await json(recorder.path);
      assert.equal(value.pid, process.pid);
      assert.equal(value.running.sdk_fingerprint, "a".repeat(64));
      reads += 1;
    }
  })();
  await Promise.all([...writes, readChecks]);
  assert.equal(reads, 40);
  assert.equal((await json(recorder.path)).latest.sequence, 23);
  assert.equal((await readdir(dirname(recorder.path))).length, 1);
  const ownPath = recorder.path;
  recorder.stop();
  await assert.rejects(lstat(ownPath), { code: "ENOENT" });
});

test("cleanup does not unlink another writer's replacement inode", async (t) => {
  const runtimeDir = await rootFor(t);
  const recorder = createBuildProcessRecord({ runtimeDir });
  await recorder.publish(sample());
  const replacement = join(runtimeDir, "replacement.json");
  await writeFile(replacement, "replacement owner\n", { mode: 0o600 });
  await rename(replacement, recorder.path);
  recorder.stop();
  assert.equal(await readFile(recorder.path, "utf8"), "replacement owner\n");
});

test("unsafe and symlink runtime directories disable publication without following links", async (t) => {
  const root = await rootFor(t);
  const publicDirectory = join(root, "public");
  await mkdir(publicDirectory, { mode: 0o777 }); await chmod(publicDirectory, 0o777);
  const linked = join(root, "linked"); await symlink(publicDirectory, linked);
  for (const runtimeDir of [publicDirectory, linked, "", "relative/path", join(root, "missing")]) {
    const recorder = createBuildProcessRecord({ runtimeDir });
    await recorder.publish(sample());
    assert.equal(recorder.path, undefined);
    recorder.stop();
  }
  assert.deepEqual(await readdir(publicDirectory), []);
});

test("symlink or unsafe child directory cannot become a build-record store", async (t) => {
  for (const kind of ["symlink-pi", "public-pi", "symlink-processes"]) {
    const root = await rootFor(t), runtimeDir = join(root, "runtime"), target = join(root, "target");
    await mkdir(runtimeDir, { mode: 0o700 }); await mkdir(target, { mode: 0o700 });
    if (kind === "symlink-pi") await symlink(target, join(runtimeDir, "pi"));
    if (kind === "public-pi") { await mkdir(join(runtimeDir, "pi")); await chmod(join(runtimeDir, "pi"), 0o755); }
    if (kind === "symlink-processes") { await mkdir(join(runtimeDir, "pi"), { mode: 0o700 }); await symlink(target, join(runtimeDir, "pi", "processes")); }
    const recorder = createBuildProcessRecord({ runtimeDir });
    await recorder.publish(sample()); recorder.stop();
    assert.equal(recorder.path, undefined);
    assert.deepEqual(await readdir(target), []);
  }
});

test("existing symlink or public record is refused without modifying its target", async (t) => {
  for (const kind of ["symlink", "public"]) {
    const root = await rootFor(t), directory = join(root, "pi", "processes");
    await mkdir(join(root, "pi"), { mode: 0o700 }); await mkdir(directory, { mode: 0o700 });
    const path = join(directory, `${process.pid}-456.json`), target = join(root, "target");
    if (kind === "symlink") { await writeFile(target, "do not change"); await symlink(target, path); }
    else { await writeFile(path, "do not change", { mode: 0o644 }); await chmod(path, 0o644); }
    const recorder = createBuildProcessRecord({ runtimeDir: root, startTicks: "456" });
    await assert.rejects(recorder.publish(sample()), /Unsafe existing build record/);
    recorder.stop();
    assert.equal(await readFile(kind === "symlink" ? target : path, "utf8"), "do not change");
  }
});

test("stop prevents pending or later publication and bounded records reject excess output", async (t) => {
  const root = await rootFor(t), recorder = createBuildProcessRecord({ runtimeDir: root });
  const pending = recorder.publish(sample()); recorder.stop(); await pending;
  await recorder.publish(sample());
  if (recorder.path) await assert.rejects(lstat(recorder.path), { code: "ENOENT" });
  const other = createBuildProcessRecord({ runtimeDir: root, startTicks: "789" });
  t.after(() => other.stop());
  await assert.rejects(other.publish(sample({ padding: "x".repeat(512 * 1024) })), /size bound/);
  await assert.rejects(lstat(other.path), { code: "ENOENT" });
});

async function fixture(t, patchSources = streamingPatchSources) {
  const root = await rootFor(t), runtimeDir = join(root, "xdg"), runtime = join(root, "runtime");
  await mkdir(runtimeDir, { mode: 0o700 });
  const scope = join(runtime, "node_modules", "@earendil-works"), packageRoot = join(scope, "pi-coding-agent");
  const cli = join(packageRoot, "dist", "cli.js"), extension = join(root, "extension.mjs"), profile = join(root, "profile.mjs"), launcher = join(root, "pi");
  const sdkFiles = new Map([
    ["pi-coding-agent/dist/cli.js", "export const version = '0.84.2';\n"],
    ["pi-coding-agent/dist/core/session.js", "export const active = true;\n"],
    ["pi-tui/dist/index.js", "export class Terminal {}\n"],
  ]);
  for (const [path, text] of sdkFiles) await write(join(scope, path), text);
  await write(join(packageRoot, "package.json"), JSON.stringify({ name: "@earendil-works/pi-coding-agent", version: "0.84.2", type: "module" }));
  await write(extension, "export default function extension() {}\n"); await write(profile, "export const profile = 'normal';\n"); await symlink(cli, launcher);
  const patches = [];
  for (const name of patchSources) {
    const path = join(root, "scripts", name), text = `# fixture-only patch identity ${name}\n`;
    await write(path, text); patches.push({ path: relative(packageRoot, path), sha256: hash(text) });
  }
  const manifestPath = join(packageRoot, "coherence-build.json");
  async function manifest() {
    const entries = [];
    for (const path of [...sdkFiles.keys()].sort()) entries.push({ path, sha256: hash(await readFile(join(scope, path))) });
    // Contract oracle: SDK-relative UTF-8 path, NUL, lowercase SHA-256, newline.
    const sdk_fingerprint = hash(entries.map(({ path, sha256 }) => `${path}\0${sha256}\n`).join(""));
    const value = { schema: "coherence.pi.installed-build.v1", package: "@earendil-works/pi-coding-agent", version: "0.84.2", entry_point: "dist/cli.js", sdk_fingerprint, sdk_files: entries, patches, patches_verified: true };
    await write(manifestPath, JSON.stringify(value)); return value;
  }
  const installed = await manifest();
  const monitor = (options = {}) => {
    const value = createRunningBuildMonitor({ entrypoint: cli, launcher, extensionRoots: [extension], profileFiles: [profile], recordBuild: true, recordOptions: { runtimeDir }, pollMs: 100000, watchFactory: () => ({ on() {}, close() {} }), ...options });
    t.after(() => value.stop()); return value;
  };
  return { root, runtimeDir, runtime, scope, packageRoot, cli, extension, profile, launcher, patches, installed, manifestPath, manifest, monitor };
}

test("monitor publishes current installed identity with independently computed SDK fingerprint", async (t) => {
  const f = await fixture(t), monitor = f.monitor();
  await monitor.start();
  const value = await json(monitor.recordPath);
  assert.equal(value.entry_point, f.cli);
  assert.equal(value.package_manifest, join(f.packageRoot, "package.json"));
  assert.equal(value.version, "0.84.2");
  assert.equal(value.running.sdk_fingerprint, f.installed.sdk_fingerprint);
  assert.deepEqual(value.running.extensions, [f.extension]);
  assert.match(value.running.extensions_fingerprint, /^[a-f0-9]{64}$/);
  assert.equal(value.latest.sdk_fingerprint, f.installed.sdk_fingerprint);
  assert.equal(value.latest.patches_status, "verified");
  assert.equal(value.latest.status, "current");
  assert.equal(value.latest.manifest_path, f.manifestPath);
  assert.ok(value.latest.checked_at_unix_ms >= value.captured_at_unix_ms);
});

test("monitor accepts a complete legacy installed build manifest", async (t) => {
  const f = await fixture(t, legacyPatchSources), monitor = f.monitor();
  await monitor.start();
  const value = await json(monitor.recordPath);
  assert.equal(value.latest.patches_status, "verified");
  assert.equal(value.latest.status, "current");
});

test("SDK drift first requires installation update; renewed installation then requires restart", async (t) => {
  const f = await fixture(t), monitor = f.monitor(); await monitor.start();
  const original = await json(monitor.recordPath);
  await write(f.cli, "export const version = '0.84.2'; export const repaired = true;\n");
  await monitor.check();
  let value = await json(monitor.recordPath);
  assert.equal(value.latest.status, "installation_update_required");
  assert.equal(value.latest.patches_status, "outdated");
  assert.notEqual(value.latest.sdk_fingerprint, original.running.sdk_fingerprint);
  assert.deepEqual(value.running, original.running);
  await f.manifest(); await monitor.check();
  value = await json(monitor.recordPath);
  assert.equal(value.latest.status, "restart_required");
  assert.equal(value.latest.patches_status, "verified");
  assert.deepEqual(value.running, original.running);
  assert.equal(value.captured_at_unix_ms, original.captured_at_unix_ms);
});

test("extension and launch-profile changes require restart without alleging an SDK patch failure", async (t) => {
  const f = await fixture(t), monitor = f.monitor(); await monitor.start();
  const original = await json(monitor.recordPath), source = await readFile(f.extension, "utf8");
  await write(f.extension, "export default function changedExtension() {}\n"); await monitor.check();
  let value = await json(monitor.recordPath);
  assert.equal(value.latest.status, "restart_required"); assert.equal(value.latest.patches_status, "verified");
  assert.notEqual(value.latest.extensions_fingerprint, original.running.extensions_fingerprint);
  assert.equal(value.latest.sdk_fingerprint, original.running.sdk_fingerprint);
  await write(f.extension, source); await monitor.check();
  assert.equal((await json(monitor.recordPath)).latest.status, "current");
  await write(f.profile, "export const profile = 'changed';\n"); await monitor.check();
  value = await json(monitor.recordPath);
  assert.equal(value.latest.status, "restart_required");
  assert.deepEqual(value.running, original.running);
});

test("patch-source drift requires installing updates even when loaded SDK bytes match", async (t) => {
  const f = await fixture(t), monitor = f.monitor(); await monitor.start();
  const helper = f.patches.find(({ path }) => path.endsWith("/pi-streaming-markdown.mjs"));
  await write(join(f.packageRoot, helper.path), "# updated patch source, not executed\n"); await monitor.check();
  const value = await json(monitor.recordPath);
  assert.equal(value.latest.status, "installation_update_required");
  assert.equal(value.latest.patches_status, "outdated");
  assert.equal(value.latest.sdk_fingerprint, value.running.sdk_fingerprint);
  assert.equal(monitor.outdated, false, "title remains a code comparison, not an installation-verification claim");
});

test("missing manifest or incomplete candidate publishes unknown rather than stale current proof", async (t) => {
  const f = await fixture(t), monitor = f.monitor(); await monitor.start();
  const original = await json(monitor.recordPath);
  await rm(f.manifestPath); await monitor.check();
  assert.equal((await json(monitor.recordPath)).latest.status, "unknown");
  await f.manifest(); await monitor.check();
  assert.equal((await json(monitor.recordPath)).latest.status, "current");
  await write(f.cli, "export const unfinished = ;\n"); await monitor.check();
  const value = await json(monitor.recordPath);
  assert.equal(value.latest.status, "unknown");
  assert.deepEqual(value.running, original.running);
});

test("unverified, malformed or incomplete installed manifests cannot claim current", async (t) => {
  const f = await fixture(t), monitor = f.monitor(); await monitor.start();
  const original = await json(monitor.recordPath);
  const invalid = [
    { ...f.installed, patches_verified: false },
    { ...f.installed, schema: "unsupported" },
    { ...f.installed, package: "another-package" },
    { ...f.installed, sdk_fingerprint: "not-a-digest" },
    { ...f.installed, patches: f.installed.patches.slice(1) },
    { ...f.installed, patches: [f.installed.patches[0], ...f.installed.patches.slice(0, 5)] },
  ];
  for (const manifest of invalid) {
    await write(f.manifestPath, JSON.stringify(manifest)); await monitor.check();
    const value = await json(monitor.recordPath);
    assert.equal(value.latest.status, "unknown");
    assert.equal(value.latest.patches_status, "unknown");
    assert.deepEqual(value.running, original.running);
  }
  await f.manifest(); await monitor.check();
  assert.equal((await json(monitor.recordPath)).latest.status, "current");
  await rm(join(f.packageRoot, f.patches[2].path)); await monitor.check();
  assert.equal((await json(monitor.recordPath)).latest.status, "unknown");
});

test("changed extension selection publishes restart while preserving the loaded extension list", async (t) => {
  const f = await fixture(t), extra = join(f.root, "new-extension.mjs");
  await write(extra, "export default function extra() {}\n");
  let selected = [];
  const monitor = f.monitor({ extensionRootProvider: () => selected }); await monitor.start();
  const original = await json(monitor.recordPath);
  selected = [extra]; await monitor.check();
  let value = await json(monitor.recordPath);
  assert.equal(value.latest.status, "restart_required");
  assert.deepEqual(value.running.extensions, [f.extension]);
  assert.deepEqual(value.running, original.running);
  selected = []; await monitor.check();
  value = await json(monitor.recordPath);
  assert.equal(value.latest.status, "current");
});

test("retargeting the installed selector preserves running path and startup fingerprints", async (t) => {
  const f = await fixture(t), monitor = f.monitor(); await monitor.start();
  const original = await json(monitor.recordPath);
  const other = await fixture(t);
  await write(other.cli, "export const version = '0.84.2'; export const newer = true;\n"); await other.manifest();
  await rm(f.launcher); await symlink(other.cli, f.launcher); await monitor.check();
  const value = await json(monitor.recordPath);
  assert.equal(value.latest.status, "restart_required");
  assert.equal(value.latest.entry_point, other.cli);
  assert.equal(value.entry_point, f.cli);
  assert.deepEqual(value.running, original.running);
  assert.equal(value.package_manifest, original.package_manifest);
});

test("recording is opt-in; missing runtime directory never prevents normal monitoring", async (t) => {
  const f = await fixture(t), disabled = f.monitor({ recordBuild: false }); await disabled.start();
  assert.equal(disabled.recordPath, undefined);
  assert.deepEqual(await readdir(f.runtimeDir), []);
  const absent = f.monitor({ recordOptions: { runtimeDir: "" } }); await absent.start();
  assert.equal(absent.recordPath, undefined);
  await write(f.extension, "export default function changed() {}\n");
  assert.equal(await absent.check(), true);
});
