import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { link, mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import install, {
  radianceChatIdentity,
  radiancePreCompactionIdentity,
  withRadianceChat,
} from "../integrations/pi/qwen-radiance-cache.mjs";
import { startCompactionProgress, clearCompactionProgress } from "../integrations/pi/qwen-radiance-compaction-progress.mjs";

function enable(t) {
  const original = process.env.QWEN_RADIANCE_CACHE_ABI;
  process.env.QWEN_RADIANCE_CACHE_ABI = "a".repeat(64);
  t.after(() => {
    if (original === undefined) delete process.env.QWEN_RADIANCE_CACHE_ABI;
    else process.env.QWEN_RADIANCE_CACHE_ABI = original;
  });
}

test("chat identity survives resume, splits sessions, and advances only after committed compaction", async (t) => {
  enable(t);
  const entries = [];
  const ctx = { model: { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate" },
    sessionManager: { getSessionFile: () => "/tmp/chat.jsonl", getSessionId: () => "uuid",
      getEntries: () => entries, getCwd: () => "/work", getSessionName: () => "Chat one" } };
  const initial = radianceChatIdentity(ctx);
  const payload = withRadianceChat({ messages: [], kv_transfer_params: { existing: 1 } }, ctx);
  assert.equal(payload.kv_transfer_params.existing, 1);
  assert.equal(payload.kv_transfer_params.qwen_snapshot_abi, "a".repeat(64));
  assert.match(payload.cache_salt, new RegExp(initial.id));
  assert.deepEqual(radianceChatIdentity(ctx), initial);
  entries.push({ type: "compaction", id: "committed" });
  const next = radianceChatIdentity(ctx);
  assert.deepEqual(radiancePreCompactionIdentity(ctx), initial);
  assert.equal(next.id, initial.id);
  assert.notEqual(next.generation, initial.generation);
  entries.push({ type: "custom", customType: "qwen-radiance-compaction-timing-v1",
    data: { compactionEntryId: "committed", elapsedMs: 94321 } });
  assert.deepEqual(radianceChatIdentity(ctx), next, "display timing cannot create another cache generation");
  ctx.sessionManager.getSessionFile = () => "/tmp/fork.jsonl";
  assert.notEqual(radianceChatIdentity(ctx).id, initial.id);
});

test("project-local hard links preserve the historical Radiance cache identity", async (t) => {
  enable(t);
  const root = await mkdtemp(join(tmpdir(), "radiance-project-history-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const legacy = join(root, "legacy", "chat.jsonl");
  const localDirectory = join(root, "work", ".pi", "sessions");
  const local = join(localDirectory, "chat.jsonl");
  await mkdir(join(root, "legacy"), { recursive: true });
  await mkdir(localDirectory, { recursive: true });
  await writeFile(legacy, "synthetic\n");
  await link(legacy, local);
  const info = await stat(local);
  await writeFile(join(localDirectory, ".identity.json"), JSON.stringify({
    schema: "urn:qwen-r9700:pi-project-history:v1",
    sessions: { "chat.jsonl": { identity_path: legacy, device: info.dev, inode: info.ino } },
  }));
  const manager = (path) => ({ getSessionFile: () => path, getSessionId: () => "uuid",
    getEntries: () => [], getCwd: () => join(root, "work"), getSessionName: () => "Synthetic" });

  assert.equal(
    radianceChatIdentity({ sessionManager: manager(local) }).id,
    radianceChatIdentity({ sessionManager: manager(legacy) }).id,
  );
});

test("fresh requests tolerate an unwritten transcript without marking it durable", async (t) => {
  enable(t);
  const root = await mkdtemp(join(tmpdir(), "radiance-fresh-transcript-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const path = join(root, "session.jsonl"), handlers = new Map();
  const entries = [{ type: "message", message: { role: "user" } }];
  const ctx = { model: { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate" },
    sessionManager: { getSessionFile: () => path, getSessionId: () => "uuid",
      getEntries: () => entries, getCwd: () => root, getSessionName: () => "Synthetic" } };
  install({ on: (name, handler) => handlers.set(name, handler), registerCommand() {} }, {
    activity: { publish: async () => {}, remove: async () => {} },
  });
  const request = () => handlers.get("before_provider_request")({ payload: { messages: [] } }, ctx);
  const first = await request();
  assert.deepEqual(await request(), first, "retries retain the same chat namespace");
  assert.equal(existsSync(path), false, "Pi must create its own complete session file");
  assert.equal(first.kv_transfer_params.qwen_chat.id, radianceChatIdentity(ctx).id);

  entries.push({ type: "message", message: { role: "assistant" } });
  await assert.rejects(request(), { code: "ENOENT" }, "the skipped flush must not be marked durable");
  await writeFile(path, "synthetic transcript\n");
  assert.deepEqual(await request(), first);
  entries.push({ type: "compaction", id: "committed" });
  await rm(path);
  await assert.rejects(request(), { code: "ENOENT" }, "compaction always requires a saved transcript");

  entries.splice(0, entries.length, { type: "compaction", id: "committed" });
  await assert.rejects(request(), { code: "ENOENT" }, "a checkpoint without retained assistant entries is still strict");
  entries.length = 0;
  ctx.sessionManager.getSessionFile = () => join(path, "session.jsonl");
  await writeFile(path, "synthetic non-directory\n");
  await assert.rejects(request(), { code: "ENOTDIR" }, "fresh sessions do not hide other filesystem errors");
});

const piRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const managerModule = join(piRoot, "pi-coding-agent/dist/core/session-manager.js");
test("installed Pi saves a fresh Radiance session once after its first assistant reply", {
  skip: !existsSync(managerModule),
}, async (t) => {
  enable(t);
  const { SessionManager } = await import(pathToFileURL(managerModule));
  const root = await mkdtemp(join(tmpdir(), "radiance-fresh-pi-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const manager = SessionManager.create(root, join(root, ".pi", "sessions"));
  manager.appendMessage({ role: "user", content: "Synthetic first request.", timestamp: Date.now() });
  const path = manager.getSessionFile(), handlers = new Map();
  const ctx = { model: { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate" }, sessionManager: manager };
  install({ on: (name, handler) => handlers.set(name, handler), registerCommand() {} }, {
    activity: { publish: async () => {}, remove: async () => {} },
  });
  assert.equal(existsSync(path), false);
  const first = await handlers.get("before_provider_request")({ payload: {} }, ctx);
  assert.equal(existsSync(path), false);
  manager.appendMessage({ role: "assistant", content: [{ type: "text", text: "Synthetic first answer." }], timestamp: Date.now() });
  const second = await handlers.get("before_provider_request")({ payload: {} }, ctx);
  assert.deepEqual(second.kv_transfer_params, first.kv_transfer_params);
  const saved = (await readFile(path, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
  assert.equal(saved.filter((entry) => entry.type === "session").length, 1);
  assert.deepEqual(saved.filter((entry) => entry.type === "message").map((entry) => entry.message.role), ["user", "assistant"]);
  assert.deepEqual(radianceChatIdentity({ sessionManager: SessionManager.open(path) }), first.kv_transfer_params.qwen_chat);
});

test("cleanup runs after durable compaction and reports failure without losing the transcript", async (t) => {
  enable(t);
  const root = await mkdtemp(join(tmpdir(), "radiance-cache-extension-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const path = join(root, "session.jsonl");
  await writeFile(path, '{"type":"compaction","id":"committed"}\n');
  const handlers = new Map(), notices = [], statuses = [];
  const ctx = { model: { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate" },
    sessionManager: { getSessionFile: () => path, getSessionId: () => "uuid",
      getEntries: () => [{ type: "compaction", id: "committed" }],
      getCwd: () => "/work", getSessionName: () => "Chat" },
    ui: { notify: (value) => notices.push(value), setStatus: (...args) => statuses.push(args) } };
  let compactCalls = 0;
  install({ on: (name, fn) => handlers.set(name, fn), registerCommand: () => {} }, {
    run: async (args) => {
      assert.match(await readFile(path, "utf8"), /committed/);
      if (args[0] === "flush") return { status: "already_durable", tokens: 123 };
      assert.equal(args[0], "compact");
      compactCalls++;
      if (compactCalls === 2) throw new Error("SSH unavailable");
      return { removed_file_bytes: 2 ** 30 };
    },
  });
  assert.equal(handlers.has("session_before_compact"), false);
  await handlers.get("session_compact")({}, ctx);
  assert.equal(statuses.at(-1)[1], undefined, "successful cleanup leaves no pinned footer");
  await handlers.get("session_compact")({}, ctx);
  assert.match(notices[0], /cleanup pending/);
  assert.ok(statuses.every(([_key, value]) => value === undefined), "cleanup failures notify without a pinned footer");
  assert.match(await readFile(path, "utf8"), /committed/);
});

test("progress spans the durable flush and remote cleanup, including failure of either step", async (t) => {
  enable(t);
  const root = await mkdtemp(join(tmpdir(), "radiance-cache-progress-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const path = join(root, "session.jsonl");
  const handlers = new Map(), reports = [];
  const ctx = { model: { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate" },
    sessionManager: { getSessionFile: () => path, getSessionId: () => "uuid",
      getEntries: () => [{ type: "compaction", id: "committed" }], getCwd: () => root, getSessionName: () => "Synthetic" },
    ui: { notify() {}, setStatus() {}, setWidget() {} } };
  t.after(() => clearCompactionProgress(ctx));
  let progress, calls = 0, compactCalls = 0;
  install({ on: (name, handler) => handlers.set(name, handler), registerCommand() {} }, {
    run: async (args) => {
      calls++;
      assert.equal(progress.snapshot().state, "running");
      assert.equal(progress.snapshot().durableCommit, true);
      assert.match(await readFile(path, "utf8"), /committed/);
      if (args[0] === "flush") {
        assert.equal(progress.snapshot().phase, "commit");
        return { status: "flushed", tokens: 123 };
      }
      assert.equal(args[0], "compact");
      assert.equal(progress.snapshot().phase, "cleanup");
      compactCalls++;
      if (compactCalls === 2) throw new Error("Synthetic SSH outage");
      return { removed_file_bytes: 2 ** 30 };
    },
  });
  for (const outcome of ["complete", "cleanup_pending", "commit_unconfirmed"]) {
    await clearCompactionProgress(ctx);
    progress = startCompactionProgress(ctx, { save: (report) => reports.push(report) });
    progress.update({ phase: "commit" });
    if (outcome === "commit_unconfirmed") await rm(path);
    else await writeFile(path, '{"type":"compaction","id":"committed"}\n');
    await handlers.get("session_compact")({}, ctx);
    assert.equal(reports.at(-1).state, outcome);
    assert.equal(reports.at(-1).durableCommit, outcome !== "commit_unconfirmed");
    if (outcome === "complete") assert.equal(reports.at(-1).removedBytes, 2 ** 30);
  }
  assert.equal(calls, 4, "an uncommitted transcript must never flush or retire its old cache");
});

test("active Pi marker follows session lifecycle and records PID and port", async (t) => {
  enable(t);
  const root = await mkdtemp(join(tmpdir(), "radiance-cache-activity-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const originalDirectory = process.env.PI_CODING_AGENT_DIR;
  const originalPort = process.env.QWEN_RADIANCE_LOCAL_PORT;
  process.env.PI_CODING_AGENT_DIR = root;
  process.env.QWEN_RADIANCE_LOCAL_PORT = "8013";
  t.after(() => {
    if (originalDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = originalDirectory;
    if (originalPort === undefined) delete process.env.QWEN_RADIANCE_LOCAL_PORT;
    else process.env.QWEN_RADIANCE_LOCAL_PORT = originalPort;
  });
  const handlers = new Map();
  const pi = { on: (name, handler) => handlers.set(name, handler), registerCommand() {} };
  const ctx = { model: { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate" },
    sessionManager: { getSessionFile: () => "/tmp/chat.jsonl", getSessionId: () => "uuid",
      getEntries: () => [], getCwd: () => "/work", getSessionName: () => "Chat" } };
  install(pi);

  await handlers.get("session_start")({}, ctx);
  const markerPath = join(root, "radiance-active", `${process.pid}.json`);
  const marker = JSON.parse(await readFile(markerPath, "utf8"));
  assert.equal(marker.schema, "urn:qwen-r9700:radiance-active-pi:v1");
  assert.equal(marker.pid, process.pid);
  assert.equal(marker.port, 8013);
  assert.equal(marker.chat_id, radianceChatIdentity(ctx).id);
  assert.match(marker.process_start_ticks, /^\d+$/);

  await handlers.get("session_shutdown")({}, ctx);
  await assert.rejects(readFile(markerPath, "utf8"), { code: "ENOENT" });
});

test("/cache includes legacy bytes from older runtime ABIs", async (t) => {
  enable(t);
  let handler;
  const notices = [];
  install({ on: () => {}, registerCommand: (_name, command) => { handler = command.handler; } }, {
    run: async (args) => {
      assert.deepEqual(args, ["list", "--json"]);
      return [
        { chats: [{ id: "a".repeat(64), title: "Current chat", file_bytes: 1024 ** 3, tokens: 12345, status: "ready" }], legacy_unassigned_bytes: 0 },
        { chats: [], legacy_unassigned_bytes: 23571726336 },
      ];
    },
  });
  await handler("", { model: { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate" },
    ui: { notify: (value) => notices.push(value) } });
  assert.match(notices[0], /Current chat/);
  assert.match(notices[0], /21.95 GiB/);
});

test("/cache flush explicitly publishes this chat's RAM tail", async (t) => {
  enable(t);
  let handler, command;
  const notices = [];
  const ctx = { model: { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate" },
    sessionManager: { getSessionFile: () => "/tmp/chat.jsonl", getSessionId: () => "uuid",
      getEntries: () => [], getCwd: () => "/work", getSessionName: () => "Chat" },
    ui: { notify: (value) => notices.push(value) } };
  install({ on() {}, registerCommand: (name, value) => { command = name; handler = value.handler; } }, {
    run: async (args) => {
      assert.equal(args[0], "flush");
      assert.equal(JSON.parse(args[2]).id, radianceChatIdentity(ctx).id);
      return { status: "flushed", tokens: 12_345 };
    },
  });

  await handler("flush", ctx);

  assert.equal(command, "cache");
  assert.match(notices[0], /flushed at 12,345 tokens/);
});
