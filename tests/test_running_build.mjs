import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, rm, symlink, rename, utimes } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { createRunningBuildMonitor, strikeAppTitle } from "../integrations/pi/qwen-running-build.mjs";

const source = (value) => `export const version = "0.84.2"; export const value = ${JSON.stringify(value)};\n`;
async function write(path, text) { await mkdir(dirname(path), { recursive: true }); await writeFile(path, text); }
async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "qwen-running-build-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const cli = join(root, "installed", "cli.mjs");
  const launcher = join(root, "bin", "pi");
  await write(cli, source("original"));
  await mkdir(dirname(launcher));
  await symlink(cli, launcher);
  function monitor(options = {}) {
    const value = createRunningBuildMonitor({ entrypoint: cli, launcher, debounceMs: 15, pollMs: 10000, ...options });
    t.after(() => value.stop());
    return value;
  }
  return { root, cli, launcher, monitor };
}
async function until(predicate, timeout = 2500) {
  const deadline = Date.now() + timeout;
  while (!predicate()) {
    assert.ok(Date.now() < deadline, "expected monitor state before deadline");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

test("title strikes the app only, preserves custom text and disabled titles, removes controls", () => {
  assert.equal(strikeAppTitle("π - project Pi - Generating", true), "π̶ - project Pi - Generating");
  assert.equal(strikeAppTitle("Pi: project", true), "P̶i̶: project");
  assert.equal(strikeAppTitle("pi - project", true), "p̶i̶ - project");
  assert.equal(strikeAppTitle("Piano project Pi", true), "Piano project Pi - π̶");
  assert.equal(strikeAppTitle("project π name", true), "project π name - π̶");
  assert.equal(strikeAppTitle("π - Generating", false), "π - Generating");
  assert.equal(strikeAppTitle("\x1b\x07\u202ePi - project", true), "P̶i̶ - project");
  assert.equal(strikeAppTitle("", true), "");
  assert.equal(strikeAppTitle(undefined, true), "");
});

test("same-version replacement is stale; metadata or byte-identical replacement is current", async (t) => {
  const f = await fixture(t);
  const changes = [];
  const m = f.monitor({ onChange: (value) => changes.push(value) });
  const ready = m.start();
  assert.equal(m.ready, ready, "readiness promise identity is stable");
  assert.equal(m.start(), ready, "start is idempotent");
  await ready;
  await utimes(f.cli, new Date(), new Date());
  assert.equal(await m.check(), false);
  const replacement = join(f.root, "next.mjs");
  await write(replacement, source("changed"));
  await rename(replacement, f.cli);
  assert.equal(await m.check(), true);
  await write(replacement, source("original"));
  await rename(replacement, f.cli);
  assert.equal(await m.check(), false);
  assert.deepEqual(changes, [true, false]);
});

test("retargeting and restoring launcher changes identity, not the displayed version", async (t) => {
  const f = await fixture(t);
  const m = f.monitor();
  await m.start();
  const other = join(f.root, "new", "cli.mjs");
  await write(other, source("new"));
  await rm(f.launcher); await symlink(other, f.launcher);
  assert.equal(await m.check(), true);
  await write(other, source("original"));
  assert.equal(await m.check(), false, "same contents in a different directory are equivalent");
});

test("in-place modification and restoration are detected without recapturing baseline", async (t) => {
  const f = await fixture(t);
  const m = f.monitor();
  await m.start();
  await write(f.cli, source("new"));
  assert.equal(await m.check(), true);
  await m.start();
  assert.equal(await m.check(), true);
  await write(f.cli, source("original"));
  assert.equal(await m.check(), false);
});

test("missing or empty candidate preserves both previously current and previously stale state", async (t) => {
  const f = await fixture(t);
  const m = f.monitor();
  await m.start();
  await writeFile(f.cli, "");
  assert.equal(await m.check(), false);
  await write(f.cli, source("new"));
  assert.equal(await m.check(), true);
  await rm(f.cli);
  assert.equal(await m.check(), true);
  await writeFile(f.cli, "");
  assert.equal(await m.check(), true);
  await write(f.cli, source("original"));
  assert.equal(await m.check(), false);
});

test("malformed partial CLI is unknown; candidate code is syntax checked without executing", async (t) => {
  const f = await fixture(t);
  const m = f.monitor();
  await m.start();
  await write(f.cli, "export const incomplete =");
  assert.equal(await m.check(), false);
  await write(f.cli, 'throw new Error("candidate must never run");\n');
  assert.equal(await m.check(), true, "syntax checking must not execute candidate code");
  await write(f.cli, "export const incomplete =");
  assert.equal(await m.check(), true);
  await write(f.cli, source("original"));
  assert.equal(await m.check(), false);
});

test("SDK sibling compiled modules participate; test, artifact and native target trees do not", async (t) => {
  const f = await fixture(t);
  const scope = join(f.root, "sdk", "node_modules", "@earendil-works");
  const cli = join(scope, "pi-coding-agent", "dist", "cli.js");
  const tui = join(scope, "pi-tui", "dist", "terminal.js");
  await write(cli, source("cli"));
  await write(tui, source("tui"));
  await rm(f.launcher); await symlink(cli, f.launcher);
  const m = f.monitor({ entrypoint: cli });
  await m.start();
  await write(join(scope, "pi-tui", "tests", "private.json"), "not code");
  await write(join(scope, "pi-tui", "dist", "target", "generated.js"), source("ignored"));
  assert.equal(await m.check(), false);
  await write(tui, source("updated"));
  assert.equal(await m.check(), true);
  await write(tui, source("tui"));
  assert.equal(await m.check(), false);
});

test("extension roots include local import/export dependencies and avoid unrelated workspace files", async (t) => {
  const f = await fixture(t);
  const extension = join(f.root, "extension", "index.mjs");
  const helper = join(f.root, "extension", "helper.mjs");
  const nested = join(f.root, "extension", "nested.mjs");
  await write(extension, 'import { x } from "./helper.mjs"; export { y } from "./nested.mjs";\n');
  await write(helper, "export const x = 1;\n");
  await write(nested, "export const y = 2;\n");
  const m = f.monitor({ extensionRoots: [extension] });
  await m.start();
  await write(join(f.root, "private-session.jsonl"), "not read by monitor");
  assert.equal(await m.check(), false);
  await write(nested, "export const y = 3;\n");
  assert.equal(await m.check(), true);
  await write(nested, "export const y = 2;\n");
  assert.equal(await m.check(), false);
  await rm(helper);
  assert.equal(await m.check(), false, "missing required dependency is unknown");
});

test("symlinked extension modules resolve local dependencies from their real module directory", async (t) => {
  const f = await fixture(t);
  const original = join(f.root, "real-extension", "index.mjs");
  const helper = join(f.root, "real-extension", "helper.mjs");
  const alias = join(f.root, "extension-link.mjs");
  await write(original, 'export { x } from "./helper.mjs";\n');
  await write(helper, "export const x = 1;\n");
  await symlink(original, alias);
  const m = f.monitor({ extensionRoots: [alias] });
  await m.start();
  await write(helper, "export const x = 2;\n");
  assert.equal(await m.check(), true, "link-location lookup must not disable the baseline");
  await write(helper, "export const x = 1;\n");
  assert.equal(await m.check(), false);
});

test("supported extension directories resolve index modules; unrelated directories do not disable SDK identity", async (t) => {
  const f = await fixture(t);
  const extension = join(f.root, "extension-package");
  const unrelated = join(f.root, "not-an-extension");
  await write(join(extension, "index.mjs"), "export const value = 1;\n");
  await write(join(unrelated, "private-session.jsonl"), "must not scan this directory");
  const m = f.monitor({ extensionRoots: [extension, unrelated] });
  await m.start();
  await write(join(extension, "index.mjs"), "export const value = 2;\n");
  assert.equal(await m.check(), true);
  await write(join(extension, "index.mjs"), "export const value = 1;\n");
  assert.equal(await m.check(), false);
  await write(f.cli, source("changed SDK"));
  assert.equal(await m.check(), true, "unsupported directory must not hide a runtime update");
});

test("real filesystem watcher updates idle sessions on loaded extension edits and restores titles", async (t) => {
  const f = await fixture(t);
  const extension = join(f.root, "extension.mjs");
  await write(extension, source("original"));
  const updates = [];
  const m = f.monitor({ extensionRoots: [extension], onChange: (value) => updates.push(value) });
  await m.start();
  await write(extension, source("watched"));
  await until(() => m.outdated);
  await write(extension, source("original"));
  await until(() => !m.outdated);
  assert.deepEqual(updates, [true, false]);
});

test("periodic fallback detects replacement when filesystem watching is unavailable", async (t) => {
  const f = await fixture(t);
  const m = f.monitor({ watchFactory: () => { throw new Error("no inotify"); }, pollMs: 25 });
  await m.start();
  await write(f.cli, source("poll"));
  await until(() => m.outdated);
});

test("concurrent manual checks share work and bounded inventory failures retain prior state", async (t) => {
  const f = await fixture(t);
  const updates = [];
  const m = f.monitor({ onChange: (value) => updates.push(value) });
  await m.start();
  await write(f.cli, source("concurrent"));
  assert.deepEqual(await Promise.all(Array.from({ length: 12 }, () => m.check())), Array(12).fill(true));
  assert.deepEqual(updates, [true], "one state transition for concurrent checks");
  const extension = join(f.root, "extension", "index.mjs");
  await write(extension, 'export const x = 1;\n');
  const bounded = f.monitor({ extensionRoots: [extension], maxFiles: 1 });
  await bounded.start();
  await write(f.cli, source("different"));
  assert.equal(await bounded.check(), false, "an incomplete baseline cannot invent stale evidence");
});

test("dispose prevents further monitoring and callbacks, including repeated stop/start", async (t) => {
  const f = await fixture(t);
  const updates = [];
  const m = f.monitor({ onChange: (value) => updates.push(value), pollMs: 20 });
  await m.start();
  m.stop(); m.stop();
  await write(f.cli, source("after dispose"));
  await m.start();
  assert.equal(await m.check(), false);
  assert.deepEqual(updates, []);
});

test("immutable release launchers report retargeting and ignore incomplete release candidates", async (t) => {
  const f = await fixture(t);
  async function release(name, text) {
    const root = join(f.root, name);
    const launcher = join(root, "scripts", "pi-radiance.py");
    const cli = join(root, "runtime", "node_modules", "@earendil-works", "pi-coding-agent", "dist", "cli.js");
    await write(launcher, "#!/usr/bin/env python3\n# immutable launch\n");
    if (text !== null) await write(cli, source(text));
    return { root, launcher, cli };
  }
  const a = await release("hash-a", "same runtime");
  const b = await release("hash-b", "same runtime");
  const partial = await release("hash-partial", null);
  await rm(f.launcher); await symlink(a.launcher, f.launcher);
  const m = f.monitor({ entrypoint: a.cli, releaseRoot: a.root });
  await m.start();
  await rm(f.launcher); await symlink(partial.launcher, f.launcher);
  assert.equal(await m.check(), false);
  await rm(f.launcher); await symlink(b.launcher, f.launcher);
  assert.equal(await m.check(), true, "release identity changes even with identical runtime JS");
  await rm(b.cli);
  assert.equal(await m.check(), true);
  await rm(f.launcher); await symlink(a.launcher, f.launcher);
  assert.equal(await m.check(), false);
});
