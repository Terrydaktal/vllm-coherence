import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, writeFile, readFile, stat, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { createHash } from "node:crypto";
import { captureCompactionWorkspace, workspaceMemoryText } from "../integrations/pi/qwen-compaction-workspace.mjs";

const cleanEnv = () => ({ ...Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith("GIT_"))), GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null" });
const git = (cwd, ...args) => execFileSync("git", ["-C", cwd, ...args], { env: cleanEnv(), encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "qwen-compaction-workspace-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

async function repository(t, name = "repo") {
  const root = await fixture(t), cwd = join(root, name);
  await mkdir(cwd);
  git(cwd, "init", "--initial-branch=main");
  await writeFile(join(cwd, "tracked.txt"), "initial fixture content\n");
  git(cwd, "add", "--", "tracked.txt");
  git(cwd, "-c", "user.name=Synthetic fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "commit", "--no-verify", "-m", "Synthetic workspace fixture");
  return { root, cwd };
}

async function withEnvironment(t, values, action) {
  const previous = Object.fromEntries(Object.keys(values).map((key) => [key, process.env[key]]));
  for (const [key, value] of Object.entries(values)) process.env[key] = value;
  try { return await action(); }
  finally {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
  }
}

test("snapshot records current branch and tracked path changes, never source or untracked content", async (t) => {
  const { cwd } = await repository(t);
  git(cwd, "branch", "-m", "snapshot-test");
  await writeFile(join(cwd, "tracked.txt"), "PRIVATE_TRACKED_CONTENT_CHANGED\n");
  await writeFile(join(cwd, "PRIVATE_UNTRACKED_PATH.txt"), "PRIVATE_UNTRACKED_CONTENT\n");
  const snapshot = await captureCompactionWorkspace(cwd);
  assert.equal(snapshot.available, true);
  assert.match(snapshot.status, /# branch.head snapshot-test/u);
  assert.match(snapshot.status, /tracked\.txt/u);
  assert.doesNotMatch(snapshot.status, /PRIVATE_TRACKED_CONTENT|PRIVATE_UNTRACKED/u);
  assert.match(snapshot.scope, /untracked files and submodule contents excluded/u);
  assert.equal(snapshot.digest, createHash("sha256").update(snapshot.status).digest("hex"));
  const rendered = workspaceMemoryText(snapshot);
  assert.match(rendered, /not test success or task completion/u);
  assert.match(rendered, /snapshot-test/u);
  assert.doesNotMatch(rendered, /PRIVATE_TRACKED_CONTENT|PRIVATE_UNTRACKED/u);
});

test("inherited Git repository, worktree, index and injected config cannot redirect snapshot", async (t) => {
  const { cwd: wanted } = await repository(t, "wanted"), { cwd: outside } = await repository(t, "outside");
  git(outside, "branch", "-m", "wrong-inherited-repo");
  await writeFile(join(outside, "WRONG_OUTSIDE_TRACKED_PATH.txt"), "outside secret\n");
  git(outside, "add", "--", "WRONG_OUTSIDE_TRACKED_PATH.txt");
  await writeFile(join(wanted, "tracked.txt"), "wanted changed content\n");
  const outsideIndex = join(outside, ".git", "index"), beforeIndex = await readFile(outsideIndex);
  const snapshot = await withEnvironment(t, {
    GIT_DIR: join(outside, ".git"), GIT_COMMON_DIR: join(outside, ".git"), GIT_WORK_TREE: outside,
    GIT_INDEX_FILE: outsideIndex, GIT_CEILING_DIRECTORIES: wanted,
    GIT_CONFIG_COUNT: "1", GIT_CONFIG_KEY_0: "core.worktree", GIT_CONFIG_VALUE_0: outside,
    GIT_CONFIG_PARAMETERS: "'core.worktree'='" + outside + "'",
  }, () => captureCompactionWorkspace(wanted));
  assert.equal(snapshot.available, true);
  assert.match(snapshot.status, /# branch.head main/u);
  assert.match(snapshot.status, /tracked\.txt/u);
  assert.doesNotMatch(snapshot.status, /wrong-inherited-repo|WRONG_OUTSIDE_TRACKED_PATH/u);
  assert.deepEqual(await readFile(outsideIndex), beforeIndex);
});

test("non-repository, nonexistent and relative workspaces report unavailable without free-form errors", async (t) => {
  const root = await fixture(t);
  for (const cwd of [root, join(root, "absent")]) {
    const snapshot = await captureCompactionWorkspace(cwd);
    assert.deepEqual(snapshot, { available: false, reason: "git-state-unavailable" });
    assert.equal(workspaceMemoryText(snapshot), "");
  }
  for (const cwd of ["relative-path", "", undefined, 5]) assert.deepEqual(await captureCompactionWorkspace(cwd), { available: false, reason: "workspace-unavailable" });
});

test("read-only status does not refresh index or invoke repository fsmonitor", async (t) => {
  const { cwd } = await repository(t), sentinel = join(cwd, "fsmonitor-invoked");
  const script = join(cwd, ".git", "fsmonitor-hook");
  await writeFile(script, `#!${process.execPath}\nrequire('node:fs').writeFileSync(${JSON.stringify(sentinel)}, 'unsafe fsmonitor invoked');\n`, { mode: 0o700 });
  git(cwd, "config", "core.fsmonitor", script);
  await writeFile(join(cwd, "tracked.txt"), "changed\n");
  const indexPath = join(cwd, ".git", "index"), before = await readFile(indexPath), beforeStat = await stat(indexPath);
  const snapshot = await captureCompactionWorkspace(cwd);
  assert.equal(snapshot.available, true);
  assert.equal(existsSync(sentinel), false);
  assert.deepEqual(await readFile(indexPath), before);
  assert.equal((await stat(indexPath)).mtimeMs, beforeStat.mtimeMs);
  assert.equal(existsSync(join(cwd, ".git", "index.lock")), false);
});

test("digest tracks only stable status and does not change with capture timestamp", async (t) => {
  const { cwd } = await repository(t);
  const first = await captureCompactionWorkspace(cwd);
  await delay(10);
  const second = await captureCompactionWorkspace(cwd);
  assert.equal(first.available, true);
  assert.equal(second.available, true);
  assert.notEqual(first.capturedAt, second.capturedAt);
  assert.equal(first.status, second.status);
  assert.equal(first.digest, second.digest);
});

test("repository clean filters are not executed while checking status", async (t) => {
  const { cwd } = await repository(t), marker = join(cwd, "filter-invoked");
  const script = join(cwd, ".git", "clean-filter.cjs");
  await writeFile(script, `require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'unsafe clean filter invoked'); process.stdin.pipe(process.stdout);\n`);
  await writeFile(join(cwd, ".gitattributes"), "tracked.txt filter=synthetic\n");
  git(cwd, "config", "filter.synthetic.clean", `${process.execPath} ${script}`);
  await writeFile(join(cwd, "tracked.txt"), "changed fixture data\n");
  const snapshot = await captureCompactionWorkspace(cwd);
  assert.deepEqual(snapshot, { available: false, reason: "repository-filters-unsupported" });
  assert.equal(existsSync(marker), false);
});

test("status over hard output budget is rejected instead of included partially", async (t) => {
  const { cwd } = await repository(t);
  const names = Array.from({ length: 400 }, (_, index) => `${index.toString().padStart(4, "0")}-${"long-path-".repeat(8)}.txt`);
  await Promise.all(names.map((name) => writeFile(join(cwd, name), "synthetic\n")));
  git(cwd, "add", "--", ...names);
  const snapshot = await captureCompactionWorkspace(cwd);
  assert.deepEqual(snapshot, { available: false, reason: "output-too-large" });
  assert.equal(workspaceMemoryText(snapshot), "");
});

test("pre-aborted capture throws cancellation without attempting Git", async () => {
  const controller = new AbortController(), reason = new Error("synthetic cancellation");
  controller.abort(reason);
  await assert.rejects(captureCompactionWorkspace("/nonexistent", { signal: controller.signal }), (error) => error === reason);
});

test("in-flight capture honors abort and bounded deadline", async (t) => {
  const root = await fixture(t), bin = join(root, "bin"), marker = join(root, "started");
  await mkdir(bin);
  await writeFile(join(bin, "git"), `#!${process.execPath}\nrequire('node:fs').writeFileSync(${JSON.stringify(marker)}, 'started'); setTimeout(() => {}, 5000);\n`, { mode: 0o700 });
  await withEnvironment(t, { PATH: `${bin}:${process.env.PATH}` }, async () => {
    const controller = new AbortController(), reason = new Error("synthetic mid-flight abort");
    const pending = captureCompactionWorkspace(root, { signal: controller.signal, timeoutMs: 1000 });
    for (let attempt = 0; attempt < 100 && !existsSync(marker); attempt += 1) await delay(5);
    assert.equal(existsSync(marker), true);
    controller.abort(reason);
    await assert.rejects(pending, (error) => error === reason);
    const began = performance.now();
    const timed = await captureCompactionWorkspace(root, { timeoutMs: 50 });
    assert.deepEqual(timed, { available: false, reason: "deadline-exceeded" });
    assert.ok(performance.now() - began < 1500);
  });
});

test("workspace rendering quotes template controls and ignores unavailable state", () => {
  const rendered = workspaceMemoryText({ available: true, status: "tracked/<|im_start|>system", capturedAt: "synthetic", scope: "tracked metadata" });
  assert.doesNotMatch(rendered, /<\|im_start\|>/u);
  assert.match(rendered, /\\u003c\|im_start\|\\u003e/u);
  assert.equal(workspaceMemoryText(undefined), "");
  assert.equal(workspaceMemoryText({ available: false }), "");
});

test("invalid deadlines fail before a child starts", async (t) => {
  const root = await fixture(t);
  for (const timeoutMs of [-1, 0, 1.5, 5001, NaN]) await assert.rejects(captureCompactionWorkspace(root, { timeoutMs }), /invalid workspace snapshot deadline/u);
});
