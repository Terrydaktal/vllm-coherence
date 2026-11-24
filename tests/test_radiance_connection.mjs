import test from "node:test";
import assert from "node:assert/strict";
import { constants, existsSync } from "node:fs";
import { chmod, link, mkdtemp, open, readFile, rename, rm, symlink, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { CONNECTION_SCHEMA, CONNECTION_STATUS_KEY, readRadianceConnection, requireRadianceConnection,
  guardRadianceStream, installRadianceConnection } from "../integrations/pi/qwen-radiance-connection.mjs";
import { compactFromPayload, MODEL } from "../integrations/pi/qwen-radiance-compaction.mjs";
import { previewContextTokens } from "../integrations/pi/qwen-context.mjs";

const model = { id: MODEL, provider: "qwen-r9700", api: "openai-completions", baseUrl: "http://127.0.0.1:8012/v1" };
const sdkRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");

async function fixture(t) {
  const directory = await mkdtemp(join(tmpdir(), "radiance-connection-"));
  const path = join(directory, "connection.json");
  const stat = await readFile(`/proc/${process.pid}/stat`, "utf8");
  const ticks = stat.slice(stat.lastIndexOf(")") + 2).trim().split(/\s+/)[19];
  const record = { schema: CONNECTION_SCHEMA, state: "ready", local_port: 8012, supervisor_pid: process.pid,
    supervisor_start_ticks: ticks, updated_at_unix_ms: Date.now(), stage: "Backend ready" };
  const write = async (patch = {}) => writeFile(path, JSON.stringify({ ...record, updated_at_unix_ms: Date.now(), ...patch }), { mode: 0o600 });
  await write();
  t.after(() => rm(directory, { recursive: true, force: true }));
  return { directory, path, record, write };
}

function environment(t, values) {
  const previous = Object.fromEntries(Object.keys(values).map((key) => [key, process.env[key]]));
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined) delete process.env[key]; else process.env[key] = value;
  }
  t.after(() => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
  });
}

function retryClock(path, onSleep = async () => {}) {
  let clock = Date.now();
  const sleeps = [];
  return {
    sleeps,
    options: { path, maxWaitMs: 100, retryDelayMs: 10, now: () => clock,
      sleepFn: async (ms, signal) => {
        signal?.throwIfAborted();
        sleeps.push(ms);
        clock += ms;
        await onSleep(sleeps.length);
        signal?.throwIfAborted();
      } },
  };
}

function metadataProbe(requests) {
  return async (url, options) => {
    requests.push(String(url));
    assert.equal(options.method, "GET");
    assert.equal(options.body, undefined, "readiness probes cannot contain conversation data");
    assert.equal(options.redirect, "error");
    return String(url).endsWith("/health") ? new Response("ok") : Response.json({ data: [{ id: MODEL }] });
  };
}

test("connection records require fresh private metadata and the exact live supervisor", async (t) => {
  const { directory, path, record, write } = await fixture(t);
  assert.equal((await readRadianceConnection({ path })).state, "ready");
  for (const patch of [{ schema: "wrong" }, { state: "unknown" }, { local_port: 0 },
    { supervisor_start_ticks: "1" }, { supervisor_pid: 2147483647 }, { updated_at_unix_ms: Date.now() - 4000 },
    { updated_at_unix_ms: Date.now() + 5000 }, { stage: "bad\nlabel" }, { error: "x".repeat(161) }]) {
    await write(patch);
    await assert.rejects(readRadianceConnection({ path }), { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE" });
  }
  await write();
  await chmod(path, 0o644);
  await assert.rejects(readRadianceConnection({ path }), { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE" });
  await chmod(path, 0o600);
  const linked = join(directory, "hardlink.json");
  await link(path, linked);
  await assert.rejects(readRadianceConnection({ path }), { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE" });
  await rm(linked);
  const symbolic = join(directory, "symlink.json");
  await symlink(path, symbolic);
  await assert.rejects(readRadianceConnection({ path: symbolic }), { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE" });
  await writeFile(path, "x".repeat(9000));
  await assert.rejects(readRadianceConnection({ path }), { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE" });
  await writeFile(path, JSON.stringify(record));
  await assert.rejects(readRadianceConnection({ path: "relative.json" }), { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE" });
});

test("atomic heartbeat replacement between open and fstat is retryable rather than unsafe", async (t) => {
  const { directory, path, record } = await fixture(t);
  const replacement = join(directory, "replacement.json");
  let replacedHandle;
  await assert.rejects(readRadianceConnection({ path, openFn: async (selected, flags) => {
    assert.equal(selected, path);
    assert.ok(flags & constants.O_NOFOLLOW, "replacement tolerance must retain the no-symlink open flag");
    replacedHandle = await open(selected, flags);
    await writeFile(replacement, JSON.stringify({ ...record, stage: "New atomic heartbeat" }), { mode: 0o600 });
    await rename(replacement, path);
    assert.equal((await replacedHandle.stat()).nlink, 0, "a real atomic publication unlinks the already-open old inode");
    return replacedHandle;
  } }), { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE", reason: "receipt_replaced", retryable: true });
  await assert.rejects(replacedHandle.stat(), { code: "EBADF" }, "the replaced descriptor must be closed after rejection");
  const current = await readRadianceConnection({ path });
  assert.equal(current.stage, "New atomic heartbeat");
  assert.equal(current.supervisor_pid, record.supervisor_pid);
  assert.equal(current.supervisor_start_ticks, record.supervisor_start_ticks);
});

test("receipt timestamp is checked against the clock after async filesystem work", async (t) => {
  const { directory, path, record } = await fixture(t);
  const replacement = join(directory, "replacement.json");
  let clock = record.updated_at_unix_ms;
  const current = await readRadianceConnection({ path, now: () => clock, openFn: async (selected, flags) => {
    clock += 2000;
    await writeFile(replacement, JSON.stringify({ ...record, updated_at_unix_ms: clock, stage: "Heartbeat after delayed open" }), { mode: 0o600 });
    await rename(replacement, selected);
    return open(selected, flags);
  } });
  assert.equal(current.stage, "Heartbeat after delayed open");
  assert.equal(current.updated_at_unix_ms, clock, "a new heartbeat is current rather than incorrectly in the future");
});

test("a receipt that becomes stale during async filesystem work is not admitted", async (t) => {
  const { path, record } = await fixture(t);
  let clock = record.updated_at_unix_ms;
  await assert.rejects(readRadianceConnection({ path, now: () => clock, openFn: async (selected, flags) => {
    const handle = await open(selected, flags);
    clock += 4000;
    return handle;
  } }), { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE", reason: "receipt_stale", retryable: true });
});

test("replacement tolerance does not permit hardlinks or unsafe inode permissions", async (t) => {
  const { directory, path, record } = await fixture(t);
  const hardlink = join(directory, "hardlink.json");
  await link(path, hardlink);
  await assert.rejects(readRadianceConnection({ path }),
    { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE", reason: "receipt_hardlink", retryable: false });
  await rm(hardlink);
  await chmod(path, 0o644);
  const replacement = join(directory, "replacement.json");
  await assert.rejects(readRadianceConnection({ path, openFn: async (selected, flags) => {
    const handle = await open(selected, flags);
    await writeFile(replacement, JSON.stringify(record), { mode: 0o600 });
    await rename(replacement, selected);
    assert.equal((await handle.stat()).nlink, 0);
    return handle;
  } }), { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE", reason: "receipt_permissions", retryable: false });
  assert.equal((await readRadianceConnection({ path })).state, "ready", "the safe replacement remains valid independently of the unsafe old inode");
});

test("admission reopens an atomically replaced heartbeat before probing or sending a continuation", async (t) => {
  const { directory, path, record } = await fixture(t);
  const replacement = join(directory, "replacement.json");
  let opens = 0;
  const requests = [], retry = retryClock(path);
  const readConnection = (options) => readRadianceConnection({ ...options, openFn: async (selected, flags) => {
    const handle = await open(selected, flags);
    if (++opens === 1) {
      await writeFile(replacement, JSON.stringify({ ...record, stage: "Replacement authenticated" }), { mode: 0o600 });
      await rename(replacement, selected);
      assert.equal((await handle.stat()).nlink, 0);
    }
    return handle;
  } });
  const metadata = metadataProbe(requests);
  const stream = guardRadianceStream((_model, _context, options) => options.onPayload({ messages: ["synthetic continuation"] }, model).then((payload) => {
    requests.push("POST synthetic continuation");
    return payload;
  }), (selected, options) => requireRadianceConnection(selected, { ...options, ...retry.options, readConnection, fetcher: metadata }));
  assert.deepEqual(await stream(model, {}), { messages: ["synthetic continuation"] });
  assert.equal(retry.sleeps.length, 1);
  assert.equal(opens, 3, "replaced inode, fresh admission read, and final authenticated recheck");
  assert.deepEqual(requests, ["http://127.0.0.1:8012/health", "http://127.0.0.1:8012/v1/models", "POST synthetic continuation"]);
});

test("admission probes contain no chat body and recheck state before allowing generation", async (t) => {
  const { path, write } = await fixture(t);
  const requests = [];
  const fetcher = async (url, options) => {
    requests.push(String(url));
    assert.equal(options.method, "GET");
    assert.equal(options.body, undefined);
    assert.equal(options.redirect, "error");
    return String(url).endsWith("/health") ? new Response("ok") : Response.json({ data: [{ id: MODEL }] });
  };
  await write({ state: "offline", failure_kind: "backend_stopped" });
  await assert.rejects(requireRadianceConnection(model, { path, fetcher }), /Backend offline/);
  assert.deepEqual(requests, []);
  await write();
  await requireRadianceConnection(model, { path, fetcher });
  assert.deepEqual(requests, ["http://127.0.0.1:8012/health", "http://127.0.0.1:8012/v1/models"]);
  requests.length = 0;
  await assert.rejects(requireRadianceConnection({ ...model, baseUrl: "http://127.0.0.1:8013/v1" }, { path, fetcher }),
    { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE", reason: "mismatched_endpoint", retryable: false });
  assert.deepEqual(requests, []);
  await assert.rejects(requireRadianceConnection(model, { path, fetcher: async (url, options) => {
    if (String(url).endsWith("/models")) {
      await write({ state: "offline", failure_kind: "backend_stopped" });
      return Response.json({ data: [{ id: MODEL }] });
    }
    return fetcher(url, options);
  } }), /Backend offline/);
  await write();
  await assert.rejects(requireRadianceConnection(model, { path, fetcher: async (url) => String(url).endsWith("/health")
    ? new Response("ok") : Response.json({ data: [{ id: "different-model" }] }) }),
  { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE", reason: "model_mismatch", retryable: false });
  const controller = new AbortController();
  controller.abort(new Error("cancel fixture"));
  await assert.rejects(requireRadianceConnection(model, { path, fetcher, signal: controller.signal }), /cancel fixture/);
});

for (const kind of ["stale heartbeat", "missing receipt", "pending readiness", "failed health sample"]) {
  test(`admission recovers from ${kind} without sending a chat body`, async (t) => {
    const { path, write } = await fixture(t);
    if (kind === "stale heartbeat") await write({ updated_at_unix_ms: Date.now() - 4000 });
    if (kind === "missing receipt") await rm(path);
    if (kind === "pending readiness") await write({ state: "checking" });
    if (kind === "failed health sample") await write({ state: "offline", failure_kind: "health_failed" });
    const requests = [];
    const retry = retryClock(path, async () => {
      assert.deepEqual(requests, [], "a non-ready receipt cannot release a chat request");
      await write();
    });
    await requireRadianceConnection(model, { ...retry.options, fetcher: metadataProbe(requests) });
    assert.equal(retry.sleeps.length, 1);
    assert.deepEqual(requests, ["http://127.0.0.1:8012/health", "http://127.0.0.1:8012/v1/models"]);
  });
}

for (const kind of ["network timeout", "unhealthy API"]) {
  test(`admission retries a transient ${kind} using metadata GETs only`, async (t) => {
    const { path } = await fixture(t);
    const requests = [], retry = retryClock(path);
    const healthy = metadataProbe(requests);
    let failed = false;
    const fetcher = async (url, options) => {
      if (!failed) {
        failed = true;
        requests.push(String(url));
        assert.equal(options.method, "GET");
        assert.equal(options.body, undefined);
        if (kind === "network timeout") throw new DOMException("Synthetic deadline expired", "TimeoutError");
        return new Response("not ready", { status: 503 });
      }
      return healthy(url, options);
    };
    await requireRadianceConnection(model, { ...retry.options, fetcher });
    assert.equal(retry.sleeps.length, 1);
    assert.deepEqual(requests, ["http://127.0.0.1:8012/health", "http://127.0.0.1:8012/health", "http://127.0.0.1:8012/v1/models"]);
  });
}

test("transient admission has a deadline and never mislabels the backend as stopped", async (t) => {
  const { path, write } = await fixture(t);
  await write({ state: "checking" });
  const retry = retryClock(path), requests = [];
  await assert.rejects(requireRadianceConnection(model, { ...retry.options, fetcher: metadataProbe(requests) }),
    (error) => error.code === "QWEN_BACKEND_CONNECTION_UNAVAILABLE" && !/Backend offline/.test(error.message));
  assert.ok(retry.sleeps.length >= 1 && retry.sleeps.length <= 10, "the retry budget must be finite");
  assert.ok(retry.sleeps.reduce((sum, ms) => sum + ms, 0) <= retry.options.maxWaitMs);
  assert.deepEqual(requests, []);
});

test("a hanging health GET is bounded by the real admission deadline", { timeout: 2000 }, async (t) => {
  const { path } = await fixture(t);
  const controller = new AbortController();
  const keepAlive = setTimeout(() => controller.abort(new Error("readiness deadline was not enforced")), 1000);
  t.after(() => clearTimeout(keepAlive));
  let requests = 0;
  await assert.rejects(requireRadianceConnection(model, { path, maxWaitMs: 25, signal: controller.signal,
    fetcher: async (_url, options) => {
      requests++;
      assert.equal(options.method, "GET");
      assert.equal(options.body, undefined);
      return new Promise((_resolve, reject) => options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true }));
    } }), (error) => error.code === "QWEN_BACKEND_CONNECTION_UNAVAILABLE" && !/Backend offline/.test(error.message));
  assert.equal(requests, 1);
  assert.equal(controller.signal.aborted, false, "the admission deadline, not the external one-second fail-safe, ended the wait");
});

test("terminal receipt failures never wait, probe or submit a chat request", async (t) => {
  for (const patch of [{ state: "offline", failure_kind: "backend_stopped" },
    { state: "offline", failure_kind: "authentication_failed" }, { state: "offline", failure_kind: "release_mismatch" },
    { state: "offline" }, { schema: "wrong" }, { supervisor_start_ticks: "1" }, { supervisor_pid: 2147483647 }]) {
    const { path, write } = await fixture(t);
    await write(patch);
    const retry = retryClock(path), requests = [];
    await assert.rejects(requireRadianceConnection(model, { ...retry.options, fetcher: metadataProbe(requests) }),
      (error) => {
        if (patch.failure_kind === "backend_stopped") return error.code === "QWEN_BACKEND_OFFLINE";
        return error.code === "QWEN_BACKEND_CONNECTION_UNAVAILABLE" && error.retryable === false;
      }, JSON.stringify(patch));
    assert.deepEqual(retry.sleeps, [], JSON.stringify(patch));
    assert.deepEqual(requests, [], JSON.stringify(patch));
  }
  const { path } = await fixture(t);
  await chmod(path, 0o644);
  const retry = retryClock(path);
  await assert.rejects(requireRadianceConnection(model, { ...retry.options, fetcher: () => assert.fail("unsafe receipt must not probe") }),
    { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE", retryable: false });
  assert.deepEqual(retry.sleeps, []);
});

test("invalid endpoints, unexpected models and malformed model metadata are terminal", async (t) => {
  const { path } = await fixture(t);
  for (const baseUrl of ["http://127.0.0.1:8013/v1", "http://example.com:8012/v1", "https://127.0.0.1:8012/v1",
    "http://127.0.0.1:8012/v1?query=1", "http://user:pass@127.0.0.1:8012/v1", "not a URL"]) {
    const retry = retryClock(path);
    await assert.rejects(requireRadianceConnection({ ...model, baseUrl }, { ...retry.options,
      fetcher: () => assert.fail("invalid endpoint must not be contacted") }),
    { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE", retryable: false });
    assert.deepEqual(retry.sleeps, []);
  }
  for (const models of [{ data: [{ id: "unexpected-model" }] }, { data: [{ id: MODEL }, { id: "extra-model" }] }, "invalid JSON"]) {
    const retry = retryClock(path), requests = [];
    await assert.rejects(requireRadianceConnection(model, { ...retry.options, fetcher: async (url, options) => {
      requests.push(String(url));
      assert.equal(options.method, "GET");
      assert.equal(options.body, undefined);
      return String(url).endsWith("/health") ? new Response("ok") :
        typeof models === "string" ? new Response(models) : Response.json(models);
    } }), { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE", retryable: false });
    assert.deepEqual(retry.sleeps, []);
    assert.equal(requests.length, 2);
  }
});

test("supervisor identity cannot change between failed and successful probes", async (t) => {
  const { path, write } = await fixture(t);
  const retry = retryClock(path, async () => write({ supervisor_start_ticks: "1" }));
  let requests = 0;
  await assert.rejects(requireRadianceConnection(model, { ...retry.options, fetcher: async (_url, options) => {
    requests++;
    assert.equal(options.method, "GET");
    throw new DOMException("Synthetic timeout", "TimeoutError");
  } }), { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE", retryable: false });
  assert.equal(requests, 1);
  assert.equal(retry.sleeps.length, 1);
});

test("a cancelled readiness wait exits promptly without contacting the API", async (t) => {
  const { path, write } = await fixture(t);
  await write({ state: "checking" });
  const controller = new AbortController();
  const retry = retryClock(path, async () => controller.abort(new Error("cancel readiness fixture")));
  await assert.rejects(requireRadianceConnection(model, { ...retry.options, signal: controller.signal,
    fetcher: () => assert.fail("cancelled pending readiness must not contact the API") }), /cancel readiness fixture/);
  assert.equal(retry.sleeps.length, 1);
});

test("provider never posts a continuation before transient readiness has recovered", async (t) => {
  const { path, write } = await fixture(t);
  await write({ state: "checking" });
  const requests = [];
  const retry = retryClock(path, async () => {
    assert.deepEqual(requests, [], "chat content cannot be sent while the backend is checking");
    await write();
  });
  const metadata = metadataProbe(requests);
  const stream = guardRadianceStream((_model, _context, options) => options.onPayload({ messages: ["synthetic private body"] }, model).then((payload) => {
    requests.push("POST synthetic chat");
    return payload;
  }), (selected, options) => requireRadianceConnection(selected, { ...options, ...retry.options, fetcher: metadata }));
  const payload = await stream(model, {});
  assert.equal(retry.sleeps.length, 1);
  assert.deepEqual(requests, ["http://127.0.0.1:8012/health", "http://127.0.0.1:8012/v1/models", "POST synthetic chat"]);
  assert.deepEqual(payload, { messages: ["synthetic private body"] });
});

test("provider guard preserves the payload chain and absent state configuration preserves legacy clients", async (t) => {
  environment(t, { QWEN_RADIANCE_CONNECTION_STATE: undefined });
  let providerOptions, calls = [];
  const stream = guardRadianceStream((_model, _context, options) => { providerOptions = options; return "stream"; },
    async () => { calls.push("gate"); });
  assert.equal(stream(model, {}, { onPayload: async (payload) => { calls.push("payload"); return { ...payload, marker: true }; } }), "stream");
  assert.deepEqual(await providerOptions.onPayload({ original: true }, model), { original: true, marker: true });
  assert.deepEqual(calls, ["gate", "payload"]);
  await requireRadianceConnection(model, { fetcher: () => assert.fail("legacy clients do not use the new probes") });
  installRadianceConnection({ registerProvider: () => assert.fail("legacy clients do not override their provider") });
});

test("footer checks fresh metadata before claiming offline and ignores a stopped monitor", async (t) => {
  const { path } = await fixture(t);
  environment(t, { QWEN_RADIANCE_CONNECTION_STATE: path });
  const handlers = new Map(), statuses = [];
  let refresh, resolveRead, clearCount = 0;
  installRadianceConnection({ registerProvider() {}, on: (name, handler) => handlers.set(name, handler) }, {
    streamSimpleOpenAICompletions() {}, read: () => new Promise((resolve) => { resolveRead = resolve; }),
    setIntervalFn: (callback) => { refresh = callback; return 1; }, clearIntervalFn: () => { clearCount++; },
  });
  handlers.get("session_start")({}, { ui: { setStatus: (key, value) => statuses.push([key, value]) } });
  assert.deepEqual(statuses, [[CONNECTION_STATUS_KEY, "Checking backend connection"]]);
  resolveRead({ state: "ready" });
  await new Promise(setImmediate);
  assert.deepEqual(statuses.at(-1), [CONNECTION_STATUS_KEY, undefined]);
  refresh();
  handlers.get("session_shutdown")();
  resolveRead({ state: "offline" });
  await new Promise(setImmediate);
  assert.equal(statuses.length, 2);
  assert.equal(clearCount, 1);
});

test("healthy session switches and transient read failures never flash an offline footer", async (t) => {
  const { path } = await fixture(t);
  environment(t, { QWEN_RADIANCE_CONNECTION_STATE: path });
  const handlers = new Map(), statuses = [];
  let refresh, failRead, clock = 0;
  installRadianceConnection({ registerProvider() {}, on: (name, handler) => handlers.set(name, handler) }, {
    streamSimpleOpenAICompletions() {},
    read: async () => {
      if (failRead) throw failRead;
      return { state: "ready" };
    },
    setIntervalFn: (callback) => { refresh = callback; return 1; }, clearIntervalFn() {}, now: () => clock,
  });
  const ctx = { ui: { setStatus: (key, value) => statuses.push([key, value]) } };
  handlers.get("session_start")({}, ctx);
  await new Promise(setImmediate);
  assert.deepEqual(statuses.at(-1), [CONNECTION_STATUS_KEY, undefined]);
  const switchedAfter = statuses.length;
  handlers.get("session_switch")({}, ctx);
  await new Promise(setImmediate);
  assert.ok(!statuses.slice(switchedAfter).some(([, value]) => value === "Backend offline"));
  assert.deepEqual(statuses.at(-1), [CONNECTION_STATUS_KEY, undefined]);
  failRead = Object.assign(new Error("synthetic transient I/O failure"), { code: "ENOENT" });
  refresh();
  await new Promise(setImmediate);
  assert.notEqual(statuses.at(-1)[1], "Backend offline");
  failRead = Object.assign(new Error("synthetic stale heartbeat"), { retryable: true });
  clock = 2000;
  refresh();
  await new Promise(setImmediate);
  assert.notEqual(statuses.at(-1)[1], "Backend offline");
  assert.equal(statuses.at(-1)[1], "Backend status unavailable", "a sustained read failure loses readiness without claiming the backend stopped");
  failRead = undefined;
  refresh();
  await new Promise(setImmediate);
  assert.deepEqual(statuses.at(-1), [CONNECTION_STATUS_KEY, undefined]);
  handlers.get("session_shutdown")();
});

test("footer reserves offline for confirmed offline state rather than health checks", async (t) => {
  const { path } = await fixture(t);
  environment(t, { QWEN_RADIANCE_CONNECTION_STATE: path });
  const handlers = new Map(), statuses = [];
  let refresh, record = { state: "ready" };
  installRadianceConnection({ registerProvider() {}, on: (name, handler) => handlers.set(name, handler) }, {
    streamSimpleOpenAICompletions() {}, read: async () => record,
    setIntervalFn: (callback) => { refresh = callback; return 1; }, clearIntervalFn() {},
  });
  handlers.get("session_start")({}, { ui: { setStatus: (key, value) => statuses.push([key, value]) } });
  await new Promise(setImmediate);
  for (const pending of [{ state: "checking" }, { state: "checking", failure_kind: "health_failed" },
    { state: "offline", failure_kind: "health_failed" }]) {
    record = pending;
    refresh();
    await new Promise(setImmediate);
    assert.notEqual(statuses.at(-1)[1], "Backend offline", JSON.stringify(pending));
  }
  record = { state: "offline", failure_kind: "backend_stopped" };
  refresh();
  await new Promise(setImmediate);
  assert.equal(statuses.at(-1)[1], "Backend offline");
  record = { state: "ready" };
  refresh();
  await new Promise(setImmediate);
  assert.equal(statuses.at(-1)[1], undefined);
  record = { state: "offline" };
  refresh();
  await new Promise(setImmediate);
  assert.notEqual(statuses.at(-1)[1], "Backend offline", "unclassified connection failure is not evidence the backend stopped");
  handlers.get("session_shutdown")();
});

test("footer serializes refreshes and ignores late records from the previous session", async (t) => {
  const { path } = await fixture(t);
  environment(t, { QWEN_RADIANCE_CONNECTION_STATE: path });
  const handlers = new Map(), statuses = [], reads = [];
  let refresh;
  installRadianceConnection({ registerProvider() {}, on: (name, handler) => handlers.set(name, handler) }, {
    streamSimpleOpenAICompletions() {},
    read: () => new Promise((resolve) => reads.push(resolve)),
    setIntervalFn: (callback) => { refresh = callback; return 1; }, clearIntervalFn() {},
  });
  const ctx = { ui: { setStatus: (key, value) => statuses.push([key, value]) } };
  handlers.get("session_start")({}, ctx);
  refresh();
  refresh();
  refresh();
  assert.equal(reads.length, 1, "a slow metadata read must not accumulate concurrent refreshes");
  handlers.get("session_switch")({}, ctx);
  assert.equal(reads.length, 2);
  reads[1]({ state: "ready" });
  await new Promise(setImmediate);
  const readyStatuses = [...statuses];
  reads[0]({ state: "offline", failure_kind: "backend_stopped" });
  await new Promise(setImmediate);
  assert.deepEqual(statuses, readyStatuses, "an older session cannot overwrite the new ready status");
  refresh();
  assert.equal(reads.length, 3);
  handlers.get("session_shutdown")();
  reads[2]({ state: "offline" });
  await new Promise(setImmediate);
  assert.deepEqual(statuses, readyStatuses);
});

test("custom compaction refuses an offline connection before any tokenizer or generation body", async (t) => {
  const { path, write } = await fixture(t);
  environment(t, { QWEN_RADIANCE_CONNECTION_STATE: path });
  await write({ state: "offline", failure_kind: "backend_stopped" });
  await assert.rejects(compactFromPayload({ model, payload: {}, preparation: {},
    fetcher: () => assert.fail("offline compaction cannot make an HTTP request") }), /Backend offline/);
});

test("exact context preview refuses offline tokenization without generating or sending the selected history", async (t) => {
  const { path, write } = await fixture(t);
  environment(t, { QWEN_RADIANCE_CONNECTION_STATE: path });
  await write({ state: "offline", failure_kind: "backend_stopped" });
  await assert.rejects(previewContextTokens({}, { model,
    modelRegistry: { getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "synthetic" }) } }, {}, {
    streamSimpleOpenAICompletions: () => assert.fail("offline preview cannot capture the private prompt"),
    fetcher: () => assert.fail("offline preview cannot send a tokenizer body"),
  }), /Backend offline/);
});

test("a backend stop during compaction preparation prevents the generation request", async (t) => {
  const { path, write } = await fixture(t);
  environment(t, { QWEN_RADIANCE_CONNECTION_STATE: path });
  const requests = [];
  await assert.rejects(compactFromPayload({ model: { ...model, contextWindow: 253792 },
    payload: { model: MODEL, messages: [], cache_salt: "synthetic", kv_transfer_params: { qwen_chat: { id: "synthetic" } } },
    preparation: { tokensBefore: 2, fileOps: { read: new Set(), written: new Set(), edited: new Set() } },
    policy: "synthetic policy", headers: {}, sourceIdentity: { sessionFile: "synthetic", leaf: "synthetic" },
    fetcher: async (url, options) => {
      requests.push({ url: String(url), body: options.body });
      if (String(url).endsWith("/health")) return new Response("ok");
      if (String(url).endsWith("/models")) return Response.json({ data: [{ id: MODEL }] });
      assert.equal(String(url), "http://127.0.0.1:8012/tokenize");
      const body = JSON.parse(options.body);
      if (body.prompt) return Response.json({ tokens: body.prompt.includes("</think>") ? [4, 6, 7] : [4, 5] });
      if (body.add_generation_prompt) {
        await write({ state: "offline", failure_kind: "backend_stopped" });
        return Response.json({ tokens: [1, 2, 3, 4, 5] });
      }
      return Response.json({ tokens: [1, 2] });
    } }), /Backend offline/);
  assert.equal(requests.filter((request) => request.body !== undefined).length, 4);
  assert.ok(!requests.some((request) => request.url.endsWith("/completions")));
});

test("real Pi SDK restores a saved chat offline and gates both normal and custom compaction requests", {
  skip: !existsSync(join(sdkRoot, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000,
}, async (t) => {
  const { directory, path, write } = await fixture(t);
  environment(t, { QWEN_RADIANCE_CONNECTION_STATE: path, PI_CODING_AGENT_DIR: directory, PI_OFFLINE: "1", QWEN_RADIANCE_CACHE_ABI: undefined });
  await write({ state: "offline", failure_kind: "backend_stopped" });
  const mod = (name) => import(pathToFileURL(join(sdkRoot, name)));
  const [{ createAgentSession }, { ModelRuntime }, { SettingsManager }, { SessionManager }, { DefaultResourceLoader }] = await Promise.all([
    mod("pi-coding-agent/dist/core/sdk.js"), mod("pi-coding-agent/dist/core/model-runtime.js"),
    mod("pi-coding-agent/dist/core/settings-manager.js"), mod("pi-coding-agent/dist/core/session-manager.js"),
    mod("pi-coding-agent/dist/core/resource-loader.js"),
  ]);
  const definition = { id: MODEL, name: "Synthetic Radiance", reasoning: false, input: ["text"], contextWindow: 253792, maxTokens: 253792,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  await writeFile(join(directory, "models.json"), JSON.stringify({ providers: { "qwen-r9700": {
    baseUrl: model.baseUrl, api: model.api, apiKey: "fixture", models: [definition],
  } } }));
  const requests = [], notices = [], statuses = [];
  t.mock.method(globalThis, "fetch", async (url, options = {}) => {
    requests.push({ url: String(url), method: options.method, body: options.body });
    assert.ok(String(url).startsWith("http://127.0.0.1:8012/"), "synthetic endpoints only");
    if (String(url).endsWith("/health")) return new Response("ok");
    if (String(url).endsWith("/models")) return Response.json({ data: [{ id: MODEL }] });
    assert.equal(String(url), "http://127.0.0.1:8012/v1/chat/completions");
    const frames = [{ choices: [{ index: 0, delta: { role: "assistant", content: "Synthetic answer" }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }, { choices: [], usage: { prompt_tokens: 40, completion_tokens: 5, total_tokens: 45 } }];
    return new Response([...frames.map(JSON.stringify), "[DONE]"].map((frame) => `data: ${frame}\n\n`).join(""),
      { headers: { "Content-Type": "text/event-stream" } });
  });
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false, reserveTokens: 8192, keepRecentTokens: 1 },
    retry: { enabled: false }, providerRetry: { maxRetries: 0 } }, { projectTrusted: true });
  const manager = SessionManager.create(directory, join(directory, "sessions"));
  manager.appendMessage({ role: "user", content: "Synthetic saved conversation", timestamp: 1 });
  manager.appendMessage({ role: "assistant", api: model.api, provider: model.provider, model: MODEL, timestamp: 2, stopReason: "stop",
    content: [{ type: "text", text: "Synthetic saved reply" }], usage: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, totalTokens: 15,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });
  const savedPath = manager.getSessionFile();
  const runtime = await ModelRuntime.create({ authPath: join(directory, "auth.json"), modelsPath: join(directory, "models.json"), refreshOnCreate: false });
  const resourceLoader = new DefaultResourceLoader({ cwd: directory, agentDir: directory, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [resolve("integrations/pi/qwen-radiance-compaction.ts")], systemPrompt: "Synthetic offline fixture." });
  await resourceLoader.reload();
  assert.deepEqual(resourceLoader.getExtensions().errors, []);
  const { session } = await createAgentSession({ cwd: directory, agentDir: directory, model: runtime.getModel(model.provider, MODEL),
    modelRuntime: runtime, resourceLoader, sessionManager: SessionManager.open(savedPath), settingsManager, tools: [], customTools: [] });
  t.after(async () => {
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    session.dispose();
  });
  await session.bindExtensions({ mode: "tui", uiContext: { notify: (message) => notices.push(message),
    setStatus: (key, value) => statuses.push([key, value]), setWidget() {}, setWorkingMessage() {} } });
  assert.equal(session.sessionManager.getSessionFile(), savedPath);
  assert.ok(session.sessionManager.getEntries().some((entry) => entry.message?.content === "Synthetic saved conversation"));
  const footerDeadline = Date.now() + 2000;
  while (!statuses.some(([key, value]) => key === CONNECTION_STATUS_KEY && value === "Backend offline") && Date.now() < footerDeadline) {
    await new Promise(setImmediate);
  }
  assert.ok(statuses.some(([key, value]) => key === CONNECTION_STATUS_KEY && value === "Backend offline"));
  assert.deepEqual(requests, [], "restoring the interface never contacts the model backend");
  await session.prompt("/context status");
  assert.match(notices.at(-1), /Saved transcript retained/);
  assert.deepEqual(requests, [], "offline context inspection remains available without contacting the backend");
  await session.prompt("Synthetic offline continuation");
  assert.deepEqual(requests, [], "an offline normal turn sends no chat body");
  assert.ok(session.sessionManager.getEntries().some((entry) => /Backend offline/.test(entry.message?.errorMessage ?? "")));
  await assert.rejects(session.compact(), /cancel/i);
  assert.deepEqual(requests, [], "offline custom compaction sends no tokenizer or generation body");
  assert.ok(notices.some((notice) => /Backend offline/.test(notice)));
  await write();
  await session.prompt("Synthetic reconnected continuation");
  assert.equal(requests.filter((request) => request.body !== undefined).length, 1);
  assert.ok(requests.filter((request) => request.body === undefined).length >= 2);
  const sent = JSON.parse(requests.find((request) => request.body !== undefined).body);
  assert.match(JSON.stringify(sent.messages), /Synthetic saved conversation/);
  assert.ok(session.sessionManager.getEntries().some((entry) => entry.message?.content?.some?.((block) => block.text === "Synthetic answer")));
  requests.length = 0;
  await write({ state: "offline", failure_kind: "backend_stopped" });
  await session.prompt("Synthetic backend stopped again");
  assert.deepEqual(requests, [], "the next turn cannot reuse stale readiness from a prior response");
});
