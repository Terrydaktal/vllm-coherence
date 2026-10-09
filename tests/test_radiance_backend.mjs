import assert from "node:assert/strict";
import test from "node:test";
import install, { BACKEND_PROGRESS_KEY, BACKEND_SCHEMA, backendCommand, backendStatusText, validateBackendReport } from "../integrations/pi/qwen-radiance-backend.mjs";

const status = (changes = {}) => ({ schema: BACKEND_SCHEMA, state: "stopped", ready: false,
  running: false, pinned: true, busy: false, operation: null, ...changes });
const operation = (changes = {}) => ({ id: "a".repeat(32), action: "start", status: "pending", stage: "Loading model", ...changes });

function fixture(run, options = {}) {
  const commands = new Map(), notices = [], sleeps = [], widgets = [], events = new Map(), timers = new Map();
  let timerId = 0;
  const ctx = { ui: {
    notify: (text, kind) => notices.push({ text, kind }),
    setWidget: (key, value) => widgets.push({ key, value }),
  }, ...options.ctx };
  install({ registerCommand: (key, value) => commands.set(key, value), on: (key, handler) => events.set(key, handler) }, {
    run, sleep: async (ms) => { sleeps.push(ms); await options.onSleep?.(ms); }, now: options.now || (() => 0),
    setIntervalFn: (callback, ms) => { const id = ++timerId; timers.set(id, { callback, ms }); return id; },
    clearIntervalFn: (id) => timers.delete(id),
  });
  return { command: (arg = "") => commands.get("backend").handler(arg, ctx), notices, sleeps, commands,
    widgets, timers, event: (name) => events.get(name)?.({}, ctx), tick: () => [...timers.values()].forEach(({ callback }) => callback()) };
}

test("bare /backend is status, works when stopped, and never emits a model request", async () => {
  const calls = [];
  const f = fixture(async (action) => { calls.push(action); return status(); });
  await f.command();
  assert.deepEqual(calls, ["status"]);
  assert.equal(f.notices[0].text, "Backend: stopped");
  assert.deepEqual(f.sleeps, []);
  assert.match(f.widgets[0].value[0], /Query backend status/);
  assert.ok(f.widgets.filter(({ value }) => value).every(({ value }) => value.length === 1));
  assert.equal(f.widgets.at(-1).value, undefined);
  assert.equal(f.timers.size, 0);
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
  const panels = f.widgets.filter(({ value }) => value).map(({ value }) => value.join("\n"));
  assert.ok(panels.some((value) => value.includes("Loading model")));
  assert.ok(panels.some((value) => value.includes("Waiting for backend readiness")));
  assert.ok(panels.some((value) => value.includes("Backend ready")));
  assert.ok(panels.every((value) => !value.includes("Verify pinned release")));
  assert.ok(f.widgets.filter(({ value }) => value).every(({ value }) => value.length === 1));
  assert.equal(f.notices.length, 1);
  assert.equal(f.widgets.at(-1).value, undefined);
  assert.equal(f.timers.size, 0);
});

test("stop waits for snapshot flush and clean shutdown", async () => {
  const reports = [status({ state: "stopping", busy: true, running: true,
    operation: operation({ action: "stop", stage: "Flushing snapshot tails (1/2)" }) }),
  status({ operation: operation({ action: "stop", status: "complete", stage: "Backend stopped" }) })];
  const f = fixture(async () => reports.shift());
  await f.command("stop");
  assert.equal(f.notices.at(-1).text, "Backend: stopped");
  const panels = f.widgets.filter(({ value }) => value).map(({ value }) => value.join("\n"));
  assert.ok(panels.some((value) => value.includes("Flushing snapshot tails (1/2)")));
  assert.ok(panels.every((value) => !value.includes("Wait for active requests")));
  assert.equal(f.notices.length, 1);
  assert.equal(f.timers.size, 0);
});

test("failed flush is an error, never claimed as successful shutdown", async () => {
  const reports = [status({ state: "stopping", busy: true, running: true,
    operation: operation({ action: "stop", stage: "Flushing snapshot tails (1/2)" }) }),
  status({ state: "idle", ready: true, running: true,
    operation: operation({ action: "stop", status: "failed", error: "disk full" }) })];
  const f = fixture(async () => reports.shift());
  await f.command("stop");
  assert.equal(f.notices.at(-1).kind, "error");
  assert.match(f.notices.at(-1).text, /disk full/);
  assert.ok(f.widgets.filter(({ value }) => value).every(({ value }) => !value[0].includes("Backend stopped")));
  assert.ok(f.widgets.some(({ value }) => value?.[0].includes("Flushing snapshot tails (1/2)")));
  assert.equal(f.widgets.at(-1).value, undefined);
  assert.equal(f.timers.size, 0);
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
  assert.equal(f.widgets.at(-1).value, undefined);
  assert.equal(f.timers.size, 0);
});

test("elapsed redraws every 100 ms without another controller query", async () => {
  let resolve, clock = 0;
  const calls = [];
  const f = fixture((action) => { calls.push(action); return new Promise((value) => { resolve = value; }); }, { now: () => clock });
  const command = f.command("start");
  assert.deepEqual([...f.timers.values()].map(({ ms }) => ms), [100]);
  clock = 300;
  f.tick();
  assert.match(f.widgets.at(-1).value[0], /0\.3s total/);
  assert.deepEqual(calls, ["start"]);
  resolve(status({ ready: true, running: true, state: "idle" }));
  await command;
  assert.equal(f.timers.size, 0);
  assert.ok(f.widgets.every(({ key }) => key === BACKEND_PROGRESS_KEY));
});

test("one progress line follows observed start stages and resets only the stage timer", async () => {
  const reports = [
    status({ busy: true, state: "starting", operation: operation({ stage: "Verifying pinned release" }) }),
    status({ busy: true, state: "starting", operation: operation({ stage: "Loading model and preparing GPU" }) }),
    status({ busy: true, running: true, state: "starting", operation: operation({ stage: "Waiting for backend readiness" }) }),
    status({ running: true, ready: true, state: "idle", operation: operation({ status: "complete", stage: "Backend ready" }) }),
  ];
  let clock = 0;
  const f = fixture(async () => reports.shift(), { now: () => clock, onSleep: () => { clock += 1000; } });
  await f.command("start");
  const panels = f.widgets.filter(({ value }) => value);
  assert.ok(panels.every(({ value }) => value.length === 1));
  const lines = panels.map(({ value }) => value[0]);
  assert.ok(lines.some((value) => value.includes("Verifying pinned release · 0.0s in stage · 0.0s total")));
  assert.ok(lines.some((value) => value.includes("Loading model and preparing GPU · 0.0s in stage · 1.0s total")));
  assert.ok(lines.some((value) => value.includes("Waiting for backend readiness · 0.0s in stage · 2.0s total")));
  assert.ok(lines.every((value) => !value.includes("compile") && !value.includes("warmup")));
  assert.equal(f.notices.length, 1);
  assert.deepEqual(f.sleeps, [1000, 1000, 1000]);
});

test("repeated reports keep the current stage timer and normalize host line breaks", async () => {
  const stage = "Loading model\nand preparing GPU";
  const reports = [
    status({ busy: true, operation: operation({ stage }) }),
    status({ busy: true, operation: operation({ stage }) }),
    status({ running: true, ready: true, state: "idle", operation: operation({ status: "complete", stage: "Backend ready" }) }),
  ];
  let clock = 0;
  const f = fixture(async () => reports.shift(), { now: () => clock, onSleep: () => { clock += 1000; } });
  await f.command("start");
  const lines = f.widgets.filter(({ value }) => value).map(({ value }) => value[0]);
  assert.ok(lines.every((line) => !line.includes("\n")));
  assert.ok(lines.some((line) => line.includes("Loading model and preparing GPU · 1.0s in stage · 1.0s total")));
});

test("a changed operation clears the panel and does not observe a replacement's progress", async () => {
  const reports = [status({ busy: true, state: "starting", operation: operation() }),
    status({ busy: true, state: "starting", operation: operation({ id: "b".repeat(32), stage: "Replacement" }) })];
  const f = fixture(async () => reports.shift());
  await f.command("start");
  assert.match(f.notices.at(-1).text, /operation changed/);
  assert.ok(f.widgets.filter(({ value }) => value).every(({ value }) => !value.join("\n").includes("Replacement")));
  assert.equal(f.widgets.at(-1).value, undefined);
  assert.equal(f.timers.size, 0);
});

test("monitor timeout clears its local timer but reports that the host operation may continue", async () => {
  let clock = 0;
  const calls = [];
  const f = fixture(async (action) => { calls.push(action); return status({ busy: true, state: "starting", operation: operation() }); }, {
    now: () => clock, onSleep: () => { clock += 16 * 60_000; },
  });
  await f.command("start");
  assert.match(f.notices.at(-1).text, /still pending on the host/);
  assert.deepEqual(calls, ["start", "status"]);
  assert.equal(f.widgets.at(-1).value, undefined);
  assert.equal(f.timers.size, 0);
});

for (const event of ["session_switch", "session_tree", "session_shutdown"]) {
  test(`${event} clears waiting progress and suppresses stale replies`, async () => {
    let resolve;
    const f = fixture(() => new Promise((value) => { resolve = value; }));
    const command = f.command("start");
    f.event(event);
    assert.equal(f.widgets.at(-1).value, undefined);
    assert.equal(f.timers.size, 0);
    const length = f.widgets.length;
    resolve(status({ busy: true, operation: operation() }));
    await command;
    f.tick();
    assert.equal(f.widgets.length, length);
    assert.equal(f.notices.length, 0);
    assert.deepEqual(f.sleeps, []);
  });
}

test("an aborted monitor clears progress without cancelling or repeating the host operation", async () => {
  const signal = new AbortController();
  let resolve;
  const calls = [];
  const f = fixture((action) => { calls.push(action); return new Promise((value) => { resolve = value; }); }, { ctx: { signal: signal.signal } });
  const command = f.command("stop");
  signal.abort();
  assert.equal(f.widgets.at(-1).value, undefined);
  assert.equal(f.timers.size, 0);
  resolve(status({ busy: true, operation: operation({ action: "stop" }) }));
  await command;
  assert.deepEqual(calls, ["stop"]);
  assert.deepEqual(f.sleeps, []);
  assert.deepEqual(f.notices, []);
});

test("session switch during the poll delay prevents the next controller request", async () => {
  const calls = [];
  let f;
  f = fixture(async (action) => { calls.push(action); return status({ busy: true, operation: operation() }); }, {
    onSleep: () => f.event("session_switch"),
  });
  await f.command("start");
  assert.deepEqual(calls, ["start"]);
  assert.equal(f.widgets.at(-1).value, undefined);
  assert.equal(f.timers.size, 0);
});

test("headless command contexts do not allocate a redraw timer", async () => {
  const notices = [];
  const f = fixture(async () => status(), { ctx: { ui: { notify: (text) => notices.push(text) } } });
  await f.command("status");
  assert.equal(f.widgets.length, 0);
  assert.equal(f.timers.size, 0);
  assert.deepEqual(notices, ["Backend: stopped"]);
});

test("status during a start keeps the start panel and mutation lock", async () => {
  let resolveStart;
  const f = fixture((action) => action === "start" ? new Promise((value) => { resolveStart = value; }) : status());
  const command = f.command("start");
  const length = f.widgets.length;
  await f.command("status");
  assert.equal(f.widgets.length, length);
  await f.command("stop");
  assert.match(f.notices.at(-1).text, /already pending/);
  resolveStart(status({ running: true, ready: true, state: "idle" }));
  await command;
  assert.equal(f.timers.size, 0);
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
