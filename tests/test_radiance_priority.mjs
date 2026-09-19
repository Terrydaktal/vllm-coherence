import assert from "node:assert/strict";
import test from "node:test";
import install, { PRIORITY_ENTRY, savedPriority, sendPriority } from "../integrations/pi/qwen-radiance-priority.mjs";

function fixture(t, overrides = {}) {
  const env = process.env.QWEN_RADIANCE_CACHE_ABI;
  process.env.QWEN_RADIANCE_CACHE_ABI = "a".repeat(64);
  t.after(() => { if (env === undefined) delete process.env.QWEN_RADIANCE_CACHE_ABI; else process.env.QWEN_RADIANCE_CACHE_ABI = env; });
  const entries = [], commands = new Map(), events = new Map(), sent = [], notices = [], timers = new Map();
  let timerId = 0;
  const ctx = { idle: true, chat: { id: "b".repeat(64), generation: "c".repeat(64) },
    model: { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", baseUrl: "http://127.0.0.1:8012/v1" },
    sessionManager: { getEntries: () => entries }, isIdle: () => ctx.idle,
    ui: { notify: (text, kind) => notices.push({ text, kind }) } };
  const pi = { registerCommand: (key, value) => commands.set(key, value),
    on: (key, value) => events.set(key, [...events.get(key) ?? [], value]),
    appendEntry: (customType, data) => entries.push({ type: "custom", customType, data }) };
  install(pi, { identity: (context) => ({ ...context.chat }),
    send: async (_ctx, chat, update) => { sent.push({ chat, ...update }); return { applied: true, priority: update.priority }; },
    every: (cb, ms) => { assert.equal(ms, 10_000); timers.set(++timerId, cb); return timerId; },
    clear: (id) => timers.delete(id), ...overrides });
  const event = async (key) => { for (const fn of events.get(key) ?? []) await fn({}, ctx); };
  const command = (value = "") => commands.get("priority").handler(value, ctx);
  t.after(() => event("session_shutdown"));
  return { ctx, entries, sent, notices, timers, event, command };
}

test("default zero has no control requests, heartbeats or prompt/history changes", async (t) => {
  const f = fixture(t);
  await f.event("session_start");
  await f.command();
  assert.match(f.notices.at(-1).text, /Chat priority: 0/);
  await f.event("agent_start");
  await f.event("agent_settled");
  assert.deepEqual(f.sent, []);
  assert.equal(f.timers.size, 0);
  assert.deepEqual(f.entries, []);
});

test("priority is saved per chat and owns the whole answer including tools", async (t) => {
  const f = fixture(t);
  await f.command("1");
  assert.equal(f.sent[0].active, false, "idle priority has no GPU reservation");
  assert.deepEqual(f.entries, [{ type: "custom", customType: PRIORITY_ENTRY, data: { priority: 1, chat_id: f.ctx.chat.id } }]);
  await f.event("agent_start");
  const answer = f.sent.at(-1).answer;
  assert.equal(f.sent.at(-1).active, true);
  await f.event("turn_end");
  await f.event("tool_execution_start");
  await f.event("tool_execution_end");
  await f.event("agent_end");
  assert.equal(f.sent.length, 2, "per-tool and recovery boundaries cannot release ownership");
  for (const tick of f.timers.values()) tick();
  await new Promise(setImmediate);
  assert.equal(f.sent.at(-1).answer, answer);
  assert.equal(f.sent.at(-1).active, true);
  await f.event("agent_settled");
  assert.equal(f.sent.at(-1).active, false);
  assert.equal(f.timers.size, 0);
  assert.equal(savedPriority(f.ctx), 1);
  await f.event("agent_start");
  assert.notEqual(f.sent.at(-1).answer, answer);
  assert.equal(f.sent.at(-1).priority, 1);
});

test("priority two command applies while generating and can be removed immediately", async (t) => {
  const f = fixture(t);
  f.ctx.idle = false;
  await f.event("agent_start");
  await f.command("2");
  assert.equal(f.sent.at(-1).active, true);
  assert.equal(f.sent.at(-1).priority, 2);
  assert.equal(f.timers.size, 1);
  await f.command("0");
  assert.equal(f.sent.at(-1).priority, 0);
  assert.equal(f.timers.size, 0);
  assert.equal(savedPriority(f.ctx), 0);
});

test("switching chats releases old ownership and compaction does not change answer identity", async (t) => {
  const f = fixture(t);
  await f.command("1");
  await f.event("agent_start");
  const before = f.sent.at(-1);
  f.ctx.chat.generation = "e".repeat(64);
  await f.command("2");
  assert.equal(f.sent.at(-1).answer, before.answer);
  assert.equal(f.sent.at(-1).chat.id, before.chat.id);
  f.ctx.chat = { id: "f".repeat(64), generation: "e".repeat(64) };
  f.entries.length = 0;
  await f.event("session_switch");
  assert.equal(f.sent.at(-1).chat.id, before.chat.id);
  assert.equal(f.sent.at(-1).active, false);
  await f.command();
  assert.match(f.notices.at(-1).text, /Chat priority: 0/);
});

test("unsupported backend or lost control request is visible and does not persist success", async (t) => {
  const f = fixture(t, { send: async () => { throw new Error("backend unavailable"); } });
  await f.command("2");
  assert.match(f.notices.at(-1).text, /Priority change was not confirmed: backend unavailable/);
  assert.equal(savedPriority(f.ctx), 0);
  assert.equal(f.entries.length, 0);
  for (const invalid of ["-1", "3", "2 extra", "1.0", "true"]) await f.command(invalid);
  assert.match(f.notices.at(-1).text, /Usage:/);
});

test("resume retains the setting, while a fork with copied entries starts at zero", async (t) => {
  const f = fixture(t);
  await f.command("2");
  await f.event("session_start");
  await f.command();
  assert.match(f.notices.at(-1).text, /Chat priority: 2/);
  f.ctx.chat.id = "f".repeat(64);
  await f.event("session_switch");
  await f.command();
  assert.match(f.notices.at(-1).text, /Chat priority: 0/);
});

test("an ambiguous timeout is followed by an ordered correction to the prior setting", async (t) => {
  const sent = [];
  const f = fixture(t, { send: async (_ctx, _chat, update) => {
    sent.push(update);
    if (update.sequence === 1) throw new Error("response timed out after submission");
    return { applied: true, priority: update.priority };
  } });
  await f.command("2");
  await new Promise(setImmediate);
  assert.deepEqual(sent.map((v) => [v.sequence, v.priority]), [[1, 2], [2, 0]]);
  assert.equal(savedPriority(f.ctx), 0);
  assert.match(f.notices.at(-1).text, /not confirmed/);
});

test("overlapping commands persist the setting each request actually confirmed", async (t) => {
  const f = fixture(t);
  await Promise.all([f.command("1"), f.command("2"), f.command("0")]);
  assert.deepEqual(f.sent.map((v) => v.priority), [1, 2, 0]);
  assert.deepEqual(f.entries.map((v) => v.data.priority), [1, 2, 0]);
});

test("a slow heartbeat cannot run after the ordered answer release", async (t) => {
  const sent = [];
  let unblock;
  const f = fixture(t, { send: async (_ctx, _chat, update) => {
    if (update.sequence === 3) await new Promise((resolve) => { unblock = resolve; });
    sent.push(update);
    return { applied: true, priority: update.priority };
  } });
  await f.command("1");
  await f.event("agent_start");
  for (const tick of f.timers.values()) tick();
  await new Promise(setImmediate);
  const release = f.event("agent_settled");
  await new Promise(setImmediate);
  assert.equal(sent.length, 2);
  unblock(); await release;
  assert.deepEqual(sent.map((v) => [v.sequence, v.active]), [[1, false], [2, true], [3, true], [4, false]]);
});

test("HTTP control uses the selected Pi port and carries no inference contents", async (t) => {
  const f = fixture(t);
  const old = process.env.QWEN_RADIANCE_BRIDGE_URL;
  delete process.env.QWEN_RADIANCE_BRIDGE_URL;
  t.after(() => { if (old === undefined) delete process.env.QWEN_RADIANCE_BRIDGE_URL; else process.env.QWEN_RADIANCE_BRIDGE_URL = old; });
  const update = { client: "d".repeat(32), answer: "e".repeat(32), sequence: 1, priority: 2, active: true };
  let seen;
  await sendPriority(f.ctx, f.ctx.chat, update, async (url, options) => {
    seen = { url: String(url), body: JSON.parse(options.body) };
    return { ok: true, json: async () => ({ applied: true, priority: 2 }) };
  });
  assert.equal(seen.url, "http://127.0.0.1:8012/qwen-radiance/priority");
  assert.deepEqual(seen.body, { ...update, chat_id: f.ctx.chat.id, abi: "a".repeat(64) });
  process.env.QWEN_RADIANCE_BRIDGE_URL = "http://127.0.0.1:18080/qwen-radiance/control";
  await sendPriority(f.ctx, f.ctx.chat, update, async (url, options) => {
    seen = { url: String(url), body: JSON.parse(options.body) };
    return new Response(JSON.stringify({ applied: true, priority: 2 }));
  });
  assert.equal(seen.url, process.env.QWEN_RADIANCE_BRIDGE_URL);
  assert.deepEqual(seen.body, { operation: "priority", chat: f.ctx.chat, update });
});
