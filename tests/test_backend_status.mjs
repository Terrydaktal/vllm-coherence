import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync, symlinkSync, rmSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { backendStatusDetails, readBackendLocalStatus, readBackendTelemetry, statusLabel } from "../integrations/pi/qwen-backend-status.mjs";
import install, { BACKEND_SCHEMA, validateBackendReport } from "../integrations/pi/qwen-radiance-backend.mjs";
import { radianceChatIdentity } from "../integrations/pi/qwen-radiance-cache.mjs";

const now = 1_000_000;
const abi = "a".repeat(64);
const model = { id: "expected-model", baseUrl: "http://127.0.0.1:18015/v1" };
const context = () => ({ model, getContextUsage: () => ({ tokens: 100 }), sessionManager: {
  getSessionFile: () => "/synthetic/project-session.jsonl", getSessionId: () => "synthetic", getEntries: () => [],
  getSessionName: () => "synthetic", getCwd: () => "/synthetic", getBranch: () => [],
} });
const report = (changes = {}) => ({ schema: BACKEND_SCHEMA, state: "idle", ready: true, running: true,
  pinned: true, busy: false, operation: null, ...changes });
const local = (changes = {}) => ({ host: "ai", api: { status: "verified" }, connection: { state: "ready", stage: "Connected",
  last_ready_at_unix_ms: now - 500 }, ...changes });

function recordingTheme() {
  const calls = [];
  const codes = { text: 37, success: 32, warning: 33, error: 31, accent: 36, dim: 90 };
  return {
    calls,
    fg(role, value) {
      assert.ok(Object.hasOwn(codes, role), `unsupported theme role: ${role}`);
      calls.push({ role, value });
      return `\x1b[${codes[role]}m${value}\x1b[39m`;
    },
    bold(value) { return `\x1b[1m${value}\x1b[22m`; },
  };
}
const plainStatus = (value) => value.replace(/\x1b\[[\d;]*m/g, "");

function fixture(t) {
  const directory = mkdtempSync(join(tmpdir(), "coherence-backend-status-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const env = { QWEN_RADIANCE_GPU_TEMPERATURE_STATE: directory, QWEN_RADIANCE_CACHE_ABI: abi };
  return { directory, env, write(name, value) { writeFileSync(join(directory, name), JSON.stringify(value), { mode: 0o600 }); } };
}

function feeds(ctx) {
  const chat = radianceChatIdentity(ctx);
  const own = { chat_id: chat.id, generation: chat.generation, state: "queued", input_tokens: 100, computed_tokens: 90 };
  const scheduler = { schema: "urn:qwen-r9700:scheduler-telemetry:v2", observed_at_ms: now - 100,
    backend: { scheduler: { pid: 1234, updated_at: now / 1000 - .100123, quantum_seconds: 0, switches: 0, cached_chats: 2,
      max_cached_chats: 2, requests: [own, { ...own, chat_id: "b".repeat(64), state: "running" }] }, worker: null } };
  const cache = { schema: "urn:qwen-r9700:cache-residency:v2", observed_at_ms: now - 300, abi, live: true, complete: true,
    chats: [{ chat_id: chat.id, generation: chat.generation, gpu_tokens: 20, ram_tokens: 60, disk_tokens: 90, disk_saved_tokens: 85, input_tokens: 100 },
      { chat_id: "c".repeat(64), generation: chat.generation, gpu_tokens: 999, ram_tokens: 999, disk_tokens: 999, disk_saved_tokens: 999, input_tokens: 999 }] };
  return { scheduler, cache };
}

test("status separates a healthy local model connection from host control reachability", () => {
  const value = backendStatusDetails({ local: local(), error: new Error("SSH unavailable"), now });
  assert.match(value, /^Backend: unknown/);
  assert.match(value, /control unavailable · tunnel authenticated · API ready; expected model verified/);
  assert.doesNotMatch(value, /confirmed stopped|interrupted request/);
  assert.match(value, /Request incidents: none recorded/);
});

test("confirmed stopped is reported only with positive host evidence", () => {
  const value = backendStatusDetails({ report: report({ state: "stopped", running: false, ready: false }), local: local({ api: { status: "offline" } }), now });
  assert.match(value, /stopped \(confirmed stopped\)/);
  assert.match(value, /Use \/backend start, then retry/);
  const unknown = report({ state: "unavailable", running: null, ready: null, pinned: null });
  assert.equal(validateBackendReport(unknown), unknown);
  const missing = backendStatusDetails({ report: unknown, local: local(), now });
  assert.match(missing, /^Backend: unknown \(host reachable; container inspection unavailable\)/);
  assert.doesNotMatch(missing, /confirmed stopped/);
});

test("host report exposes release, resources, remote-provided uptime and queue age without comparing clocks", () => {
  const value = backendStatusDetails({ report: report({ observed_at: 999_999_999, uptime_seconds: 12.3,
    release: { expected: { model: "expected-model", runtime_abi: abi, data_abi: abi }, observed: { model: "loaded-model", image: "image:v1", runtime_abi: abi, data_abi: abi } },
    resources: { gpu: { junction_millicelsius: 80000, edge_millicelsius: 50000, utilization_percent: 85, fan_percent: 60, vram_used_bytes: 2 ** 30, vram_total_bytes: 2 ** 35 },
      ram: { total_bytes: 2 ** 36, available_bytes: 2 ** 35, used_bytes: 2 ** 35 } },
    telemetry: { scheduler: { status: "current", active_requests: 1, queued_requests: 2, generating_requests: 1, age_seconds: .2 } } }), local: local(), now });
  for (const expected of [/Uptime: 12.3s/, /Model: loaded-model · expected expected-model/, /matches pinned container configuration/,
    /GPU junction 80.0°C, edge 50.0°C/, /VRAM 1.0 GiB \/ 32.0 GiB/, /RAM 32.0 GiB \/ 64.0 GiB/, /1 active · 2 queued · 1 generating · sample 0.2s old/]) assert.match(value, expected);
  assert.doesNotMatch(value, /999999999|negative|clock/);
});

test("last known host report is aged and never substitutes for current evidence", () => {
  const value = backendStatusDetails({ local: local({ api: { status: "offline" } }), lastKnown: { report: report(), observed_at_ms: now - 20_000 }, now });
  assert.match(value, /^Backend: unknown/);
  assert.match(value, /Last known host snapshot: 20.0s old; not current/);
  assert.match(value, /Last known backend: idle/);
  assert.match(value, /Last known model:/);
});

test("reconnect reports retries without displaying historical connection failures", () => {
  const connection = { state: "ready", stage: "Connected", connection_attempt: 3,
    last_ready_at_unix_ms: now - 100, failure_kind: "host_unreachable", last_failure_at_unix_ms: now - 5000 };
  const value = backendStatusDetails({ report: report(), local: local({ connection }), now });
  assert.match(value, /tunnel authenticated/);
  assert.match(value, /attempt 3 · last verified 0.1s old/);
  assert.doesNotMatch(value, /Previous connection failure|host unreachable/);
  assert.equal(connection.failure_kind, "host_unreachable");
  assert.equal(connection.last_failure_at_unix_ms, now - 5000);
  const waiting = backendStatusDetails({ local: local({ connection: { state: "offline", stage: "Waiting before reconnect", next_retry_at_unix_ms: now + 2500,
    error: "Backend is stopped" } }), now });
  assert.match(waiting, /retry in 2.5s/);
  assert.match(waiting, /Connection issue: Backend is stopped/);
});

test("themed status preserves the complete plain report in healthy, stopped and stale states", () => {
  const variants = [
    { report: report({ uptime_seconds: 12.3, resources: { gpu: { vram_used_bytes: 2 ** 30, vram_total_bytes: 2 ** 35 } } }), local: local() },
    { report: report({ state: "stopped", running: false, ready: false }), local: local({ api: { status: "offline" },
      connection: { state: "offline", stage: "Backend offline", failure_kind: "backend_stopped", last_failure_at_unix_ms: now - 5000 } }) },
    { local: local({ observed_at_ms: now - 20_000, api: { status: "offline" } }),
      lastKnown: { report: report(), observed_at_ms: now - 20_000 }, error: new Error("SSH unavailable") },
  ];
  for (const options of variants) {
    const theme = recordingTheme();
    const value = backendStatusDetails({ ...options, now, theme });
    assert.match(value, /\x1b\[/, "status should contain theme colours");
    assert.equal(plainStatus(value), backendStatusDetails({ ...options, now }));
    assert.doesNotMatch(plainStatus(value), /Previous connection failure/);
    assert.ok(theme.calls.some(({ role, value }) => role === "text" && value.includes("Backend")), "field labels use the text colour");
    assert.match(value, /\x1b\[1m/, "field labels are bold");
    assert.ok(theme.calls.some(({ role }) => role === "dim"), "separators and secondary details are dimmed");
  }
});

test("themed status distinguishes healthy, unavailable and failed evidence with semantic colours", () => {
  const healthy = recordingTheme();
  backendStatusDetails({ report: report({ uptime_seconds: 12.3, release: { observed: { model: model.id, runtime_abi: abi } } }),
    local: local({ endpoint: model.baseUrl }), now, theme: healthy });
  assert.ok(healthy.calls.some(({ role, value }) => role === "success" && /idle|ready|authenticated|verified|matches/.test(value)));
  assert.ok(healthy.calls.some(({ role }) => role === "accent"), "identifiers and numeric values use accent colour");

  const offline = recordingTheme();
  backendStatusDetails({ report: report({ state: "stopped", running: false, ready: false }),
    local: local({ api: { status: "offline" }, connection: { state: "offline", stage: "Backend offline" } }), now, theme: offline });
  assert.ok(offline.calls.some(({ role, value }) => role === "warning" && /stopped|offline|unknown/.test(value)));

  const failed = recordingTheme();
  const failure = { action: "start", status: "failed", error: "Release authentication failed" };
  backendStatusDetails({ report: report({ pinned: false, operation: failure }),
    local: local({ api: { status: "mismatch" } }), now, theme: failed });
  assert.ok(failed.calls.some(({ role, value }) => role === "error" && /failed|mismatch|does not match/.test(value)), "genuine failures use error colour");

  const stale = recordingTheme();
  backendStatusDetails({ local: local({ observed_at_ms: now - 20_000 }),
    lastKnown: { report: report(), observed_at_ms: now - 20_000 }, now, theme: stale });
  assert.ok(stale.calls.some(({ role, value }) => role === "warning" && /not current|unknown|old/.test(value)), "stale evidence is a warning");
});

test("registered /backend applies the active Pi theme on successful and failed status reads", async () => {
  const theme = recordingTheme(), notices = [], commands = new Map(), ctx = context();
  let fail = false;
  ctx.ui = { theme, notify: (text) => notices.push(text) };
  install({ registerCommand: (name, handler) => commands.set(name, handler), on() {} }, {
    run: async () => { if (fail) throw new Error("SSH unavailable"); return report(); },
    readLocal: async () => local(), wall: () => now,
  });
  await commands.get("backend").handler("status", ctx);
  assert.equal(plainStatus(notices[0]), backendStatusDetails({ report: report(), local: local(), now }));
  assert.match(notices[0], /\x1b\[/);
  const firstCalls = theme.calls.length;
  fail = true;
  await commands.get("backend").handler("status", ctx);
  assert.match(plainStatus(notices[1]), /^Backend: unknown/);
  assert.match(notices[1], /\x1b\[/);
  assert.ok(theme.calls.length > firstCalls, "transport failures still use the active theme");
});

test("control, release, stage and failure strings cannot inject terminal commands or new display lines", () => {
  const malicious = "bad\x1b]2;injected\x07\n\u202evalue";
  const value = backendStatusDetails({ report: report({ release: { observed: { model: malicious, image: malicious, runtime_abi: malicious } },
    operation: { status: "failed", action: malicious, error: malicious } }), local: local({ connection: { state: "offline", stage: malicious, error: malicious } }), error: new Error(malicious), now });
  assert.doesNotMatch(value, /\x1b|\x07|\u202e|injected/);
  assert.ok(value.split("\n").every((line) => line.length <= 700));
  assert.equal(statusLabel("x".repeat(500)), "x".repeat(180));
  assert.throws(() => validateBackendReport(report({ release: { observed: { runtime_abi: 123 } } })));
});

test("normal and Opsec standalone telemetry selects only this exact chat generation", (t) => {
  const f = fixture(t), ctx = context(), values = feeds(ctx);
  f.write("scheduler-v2.json", values.scheduler); f.write("cache-residency-v2.json", values.cache);
  const telemetry = readBackendTelemetry(ctx, { env: f.env, now });
  assert.equal(telemetry.scheduler.own.state, "queued");
  assert.equal(telemetry.scheduler.active, 1);
  assert.equal(telemetry.scheduler.observed_at_ms, 999899);
  assert.deepEqual(telemetry.cache, { gpu: 20, ram: 40, disk: 30, cold: 10, diskSaved: 85, observed_at_ms: now - 300 });
  const value = backendStatusDetails({ report: report(), local: local({ telemetry }), now });
  assert.match(value, /observed queued request \(submitted\); exact queue position unavailable/);
  assert.match(value, /VRAM 20 · RAM 40 · disk backup 85 · cold 10 tokens/);
  assert.doesNotMatch(value, /999|position 1|position 2/);
});

test("combined telemetry and standalone feed produce the same chat cache evidence", (t) => {
  const f = fixture(t), ctx = context(), values = feeds(ctx);
  f.write("telemetry-v1.json", { schema: "urn:qwen-r9700:telemetry:v1", observed_at_ms: now - 100,
    scheduler: values.scheduler.backend.scheduler, worker: null, phases: null, cache: values.cache, temperature: null });
  const telemetry = readBackendTelemetry(ctx, { env: f.env, now });
  assert.equal(telemetry.scheduler.own.state, "queued"); assert.equal(telemetry.cache.diskSaved, 85);
});

test("stale, wrong-generation, wrong-ABI, world-writable and symlinked local files are not current cache evidence", (t) => {
  const f = fixture(t), ctx = context(), values = feeds(ctx);
  f.write("scheduler-v2.json", { ...values.scheduler, observed_at_ms: now - 6000 });
  f.write("cache-residency-v2.json", { ...values.cache, abi: "b".repeat(64) });
  assert.deepEqual(readBackendTelemetry(ctx, { env: f.env, now }), {});
  f.write("cache-residency-v2.json", { ...values.cache, chats: values.cache.chats.map((row) => ({ ...row, generation: "d".repeat(64) })) });
  assert.equal(readBackendTelemetry(ctx, { env: f.env, now }).cache.gpu, 0);
  f.write("scheduler-v2.json", values.scheduler); chmodSync(join(f.directory, "scheduler-v2.json"), 0o666);
  assert.equal(readBackendTelemetry(ctx, { env: f.env, now }).scheduler, undefined);
  const target = join(f.directory, "outside.json"); f.write("outside.json", values.scheduler);
  rmSync(join(f.directory, "scheduler-v2.json")); symlinkSync(target, join(f.directory, "scheduler-v2.json"));
  assert.equal(readBackendTelemetry(ctx, { env: f.env, now }).scheduler, undefined);
});

test("connection status checks freshness after asynchronous reads rather than freezing the entry clock", async () => {
  let clock = now, sampledTelemetry;
  const value = await readBackendLocalStatus(context(), { env: {}, now: () => clock,
    readConnection: async ({ now: receiptClock }) => {
      assert.equal(typeof receiptClock, "function");
      clock += 1500;
      assert.equal(receiptClock(), clock);
      return undefined;
    },
    telemetry: (_ctx, options) => { sampledTelemetry = options.now; return {}; },
    fetcher: async (url) => new Response(url.endsWith("/health") ? "" : JSON.stringify({ data: [{ id: model.id }] })),
  });
  assert.equal(value.observed_at_ms, now + 1500);
  assert.equal(sampledTelemetry, now + 1500);
  assert.equal(value.api.status, "verified");
});

test("connection status probes only local health/model metadata and no generation content", async () => {
  const calls = [];
  const value = await readBackendLocalStatus(context(), { env: {}, now, telemetry: () => ({}), readConnection: async () => undefined,
    fetcher: async (url, options) => { calls.push({ url, options }); return new Response(url.endsWith("/health") ? "" : JSON.stringify({ data: [{ id: model.id }] })); } });
  assert.equal(value.api.status, "verified");
  assert.deepEqual(calls.map(({ url }) => url), ["http://127.0.0.1:18015/health", "http://127.0.0.1:18015/v1/models"]);
  assert.ok(calls.every(({ options }) => options.method === "GET" && !options.body));
});

test("offline receipt and invalid receipt are visible without contacting the model endpoint", async () => {
  const options = { env: {}, now, telemetry: () => ({}), fetcher: async () => assert.fail("offline must not probe"),
    readConnection: async () => ({ state: "offline", stage: "Backend not running" }) };
  assert.equal((await readBackendLocalStatus(context(), options)).api.status, "offline");
  const invalid = await readBackendLocalStatus(context(), { ...options, readConnection: async () => { throw new Error("bad receipt"); } });
  assert.equal(invalid.connectionInvalid, true);
});

test("only selected-branch diagnostic metadata is shown, without reading assistant/user text", async () => {
  const ctx = context();
  ctx.sessionManager.getEntries = () => assert.fail("must not scan transcript content");
  ctx.sessionManager.getBranch = () => [{ type: "message", get message() { assert.fail("must not read message"); } },
    { type: "custom", customType: "qwen-radiance-transport-error-v1", data: { timestamp: now - 10, causes: [{ code: "ECONNREFUSED" }] } }];
  const value = await readBackendLocalStatus(ctx, { env: {}, now, telemetry: () => ({}), readConnection: async () => ({ state: "offline" }) });
  assert.equal(value.incident.summary, "ECONNREFUSED");
  const display = backendStatusDetails({ local: value, now });
  assert.match(display, /Latest recorded request incident: ECONNREFUSED/);
  assert.match(display, /\/backend-error for details/);
});

test("cancelled read does not misclassify cancellation as backend offline", async () => {
  const controller = new AbortController(); controller.abort();
  await assert.rejects(readBackendLocalStatus({ ...context(), signal: controller.signal }, { env: {}, now, telemetry: () => ({}),
    readConnection: async () => undefined, fetcher: async (_url, options) => { options.signal.throwIfAborted(); } }), { name: "AbortError" });
});

test("/backend retains a host snapshot with its age after transport failure and adds no conversation messages", async () => {
  const notices = [], commands = new Map(), ctx = context(); let clock = now, calls = 0;
  ctx.ui = { notify: (text) => notices.push(text) };
  install({ registerCommand: (name, handler) => commands.set(name, handler), on() {}, sendMessage() { assert.fail("must not add model context"); }, sendUserMessage() { assert.fail("must not send inference"); } }, {
    run: async (action) => { assert.equal(action, "status"); if (calls++) throw new Error("SSH unavailable"); return report(); },
    readLocal: async () => local(), wall: () => clock,
  });
  await commands.get("backend").handler("", ctx); clock += 12_000; await commands.get("backend").handler("status", ctx);
  assert.match(notices[1], /^Backend: unknown/);
  assert.match(notices[1], /Last known host snapshot: 12.0s old; not current/);
});

 test("confirmed backend stop demotes lingering GPU/RAM records to historical evidence", () => {
  const value = backendStatusDetails({ report: report({ state: "stopped", running: false, ready: false }),
    local: local({ telemetry: { scheduler: { observed_at_ms: now, own: { state: "running" } },
      cache: { observed_at_ms: now, gpu: 100, ram: 0, diskSaved: 90, cold: 0 } } }), now });
  assert.match(value, /This chat: no active request \(backend confirmed stopped\)/);
  assert.match(value, /Last observed cache \(backend stopped; not live\): VRAM 100/);
  assert.doesNotMatch(value, /observed running request/);
});

 test("slow host control cannot turn an aged local ready receipt into fresh connection evidence", () => {
  const value = backendStatusDetails({ report: report(), local: local({ observed_at_ms: now - 20_000,
    connection: { state: "ready", stage: "Connected", updated_at_unix_ms: now - 20_000 } }), now });
  assert.match(value, /tunnel last authenticated \(20.0s old\)/);
  assert.match(value, /API last probe ready; expected model verified \(20.0s old; not current\)/);
});

for (const entry of ["qwen-radiance-identity", "qwen-radiance-cache", "qwen-radiance-scheduler-telemetry", "qwen-cache-residency",
  "qwen-gpu-temperature", "qwen-context", "qwen-radiance-backend", "qwen-backend-status"]) {
  test(`clean-process import of ${entry} cannot depend on module initialization order`, () => {
    const moduleUrl = new URL(`../integrations/pi/${entry}.mjs`, import.meta.url).href;
    const result = spawnSync(process.execPath, ["--input-type=module", "-e", "await import(process.argv[1])", moduleUrl],
      { encoding: "utf8", timeout: 5000, maxBuffer: 65_536 });
    assert.equal(result.status, 0, `${result.error?.message ?? ""} ${result.stderr}`);
    assert.equal(result.stdout, "");
  });
}

test("identity module extraction preserves cache identity, generation digests and public cache exports", async () => {
  const pure = await import("../integrations/pi/qwen-radiance-identity.mjs");
  const cache = await import("../integrations/pi/qwen-radiance-cache.mjs");
  for (const name of ["radianceIdentityPath", "radianceChatIdentity", "radiancePreCompactionIdentity"]) assert.equal(cache[name], pure[name]);
  assert.deepEqual(pure.radianceChatIdentity(context()), {
    id: "d2aa64b63c3a67b5f61efc0a18a076c3caac3f6cbe2607a6049999fb9b6c4333",
    generation: "4f1cf8d86a66caf554db9ac662dfe98a531cc614c2c4309a5e04c5f74912d3fd",
    title: "synthetic", cwd: "/synthetic", session_file: "/synthetic/project-session.jsonl",
  });
});
