import assert from "node:assert/strict";
import test from "node:test";
import install, { BACKEND_SCHEMA, backendCommand, backendStatusText, validateBackendReport } from "../integrations/pi/qwen-radiance-backend.mjs";

const status = (changes = {}) => ({ schema: BACKEND_SCHEMA, state: "stopped", ready: false,
  running: false, pinned: true, busy: false, operation: null, ...changes });
const operation = (changes = {}) => ({ id: "a".repeat(32), action: "start", status: "pending", stage: "Loading model", ...changes });

function fixture(run) {
  const commands = new Map(), notices = [], sleeps = [];
  install({ registerCommand: (key, value) => commands.set(key, value) }, {
    run, sleep: async (ms) => { sleeps.push(ms); }, now: () => 0,
  });
  return { command: (arg = "") => commands.get("backend").handler(arg, { ui: {
    notify: (text, kind) => notices.push({ text, kind }),
  } }), notices, sleeps, commands };
}

test("bare /backend is status, works when stopped, and never emits a model request", async () => {
  const calls = [];
  const f = fixture(async (action) => { calls.push(action); return status(); });
  await f.command();
  assert.deepEqual(calls, ["status"]);
  assert.equal(f.notices[0].text, "Backend: stopped");
  assert.deepEqual(f.sleeps, []);
  assert.deepEqual(f.commands.get("backend").getArgumentCompletions("st").map((v) => v.value), ["status", "start", "stop"]);
});

test("start waits for actual readiness, reporting changed host stages", async () => {
  const calls = [], reports = [
    status({ state: "starting", busy: true, operation: operation() }),
    status({ state: "starting", busy: true, running: true, operation: operation({ stage: "Waiting for backend readiness" }) }),
    status({ state: "idle", ready: true, running: true, operation: operation({ status: "complete", stage: "Backend ready" }) }),
  ];
  const f = fixture(async (action) => { calls.push(action); return reports.shift(); });
  await f.command("start");
  assert.deepEqual(calls, ["start", "status", "status"]);
  assert.equal(f.sleeps.length, 2);
  assert.match(f.notices.at(-1).text, /idle.*API ready/);
});

test("stop waits for snapshot flush and clean shutdown", async () => {
  const reports = [status({ state: "stopping", busy: true, running: true,
    operation: operation({ action: "stop", stage: "Flushing snapshot tails (1/2)" }) }),
  status({ operation: operation({ action: "stop", status: "complete", stage: "Backend stopped" }) })];
  const f = fixture(async () => reports.shift());
  await f.command("stop");
  assert.match(f.notices[0].text, /Flushing snapshot tails/);
  assert.equal(f.notices.at(-1).text, "Backend: stopped");
});

test("failed flush is an error, never claimed as successful shutdown", async () => {
  const reports = [status({ state: "stopping", busy: true, running: true,
    operation: operation({ action: "stop" }) }),
  status({ state: "idle", ready: true, running: true,
    operation: operation({ action: "stop", status: "failed", error: "disk full" }) })];
  const f = fixture(async () => reports.shift());
  await f.command("stop");
  assert.equal(f.notices.at(-1).kind, "error");
  assert.match(f.notices.at(-1).text, /disk full/);
});

test("reject arbitrary arguments before transport and unknown responses", async () => {
  const f = fixture(async () => { assert.fail("must not execute"); });
  for (const arg of ["restart", "start extra", "stop; echo unsafe"]) await f.command(arg);
  assert.equal(f.notices.length, 3);
  assert.throws(() => validateBackendReport({ ...status(), schema: "other" }));
  assert.throws(() => validateBackendReport(status({ operation: { action: "start" } })));
  assert.throws(() => validateBackendReport(status({ active_requests: -1 })));
});

test("unreachable host remains an error, not a stopped status", async () => {
  const f = fixture(async () => { throw new Error("SSH unavailable"); });
  await f.command();
  assert.match(f.notices.at(-1).text, /SSH unavailable/);
  assert.equal(f.notices.at(-1).kind, "error");
});

test("VM uses only the fixed bridge operation, without a Python helper or SSH", async (t) => {
  const original = process.env.QWEN_RADIANCE_BRIDGE_URL;
  const fetcher = globalThis.fetch;
  process.env.QWEN_RADIANCE_BRIDGE_URL = "http://127.0.0.1:18080/qwen-radiance/control";
  t.after(() => {
    if (original === undefined) delete process.env.QWEN_RADIANCE_BRIDGE_URL;
    else process.env.QWEN_RADIANCE_BRIDGE_URL = original;
    globalThis.fetch = fetcher;
  });
  const sent = [];
  globalThis.fetch = async (url, options) => {
    sent.push({ url, value: JSON.parse(options.body) });
    return new Response(JSON.stringify(status()), { status: 200 });
  };
  assert.equal((await backendCommand("status")).state, "stopped");
  assert.deepEqual(sent[0].value, { operation: "backend", action: "status" });
});

test("status exposes last failure and respects a foreign container", () => {
  assert.match(backendStatusText(status({ operation: operation({ status: "failed", error: "source mismatch" }) })), /Last start failed: source mismatch/);
  assert.match(backendStatusText(status({ state: "unavailable", pinned: false })), /differs from the pinned release/);
});
