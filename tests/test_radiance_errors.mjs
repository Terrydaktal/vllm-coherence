import test from "node:test";
import assert from "node:assert/strict";
import { BACKEND_ERROR_ENTRY, TRANSPORT_ERROR_ENTRY, backendErrorMessage, diagnosticLines, installRadianceErrors,
  reportBackendFailure, connectionDiagnosticLines, lookupFailure, probeLocalConnection } from "../integrations/pi/qwen-radiance-errors.mjs";
import { createServer } from "node:net";
import { createTransportDiagnostics, TRANSPORT_ERROR } from "../integrations/pi/qwen-transport-diagnostics.mjs";

const failure = "EngineCore encountered an issue. See stack trace (above) for the root cause.";
const report = { schema: "urn:qwen-r9700:backend-error:v1", status: "found", backend: { ready: false, running: false },
  incident: { id: "incident-a", timestamp: 10000, container_id: "a".repeat(64), summary: "AssertionError: synthetic failure",
    traceback: 'Traceback (most recent call last):\n  File "/opt/vllm/scheduler.py", line 42, in schedule\nAssertionError: synthetic failure' } };

function fixture(probe = async () => report) {
  const handlers = new Map(), entries = [], renderers = new Map(), commands = new Map(), probes = [];
  let session = "synthetic-session-a";
  const pi = { on: (name, fn) => handlers.set(name, fn), appendEntry: (customType, data) => entries.push({ customType, data }),
    registerEntryRenderer: (name, fn) => renderers.set(name, fn), registerCommand: (name, command) => commands.set(name, command) };
  const ctx = { model: { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate" },
    sessionManager: { getSessionFile: () => session } };
  class Text { constructor(text) { this.text = text; } }
  installRadianceErrors(pi, { Text, now: () => 20000, connectionProbe: async () => ({ listening: true }),
    probe: (args) => { probes.push(args); return probe(args); } });
  return { pi, ctx, entries, probes, renderers, commands, switch: () => { session = "synthetic-session-b"; },
    emit: (name, event = {}) => handlers.get(name)?.(event, ctx) };
}
const message = () => ({ role: "assistant", stopReason: "error", errorMessage: failure, timestamp: 5000,
  content: [{ type: "text", text: "Preserved synthetic partial response" }] });

test("errors retain their partial response and get an expandable diagnostic after the message", async () => {
  const f = fixture(), original = message();
  const result = f.emit("message_end", { message: original });
  assert.equal(result.message.errorMessage, "EngineCore encountered an issue.");
  assert.equal(result.message.content, original.content);
  assert.equal(f.entries.length, 0);
  await f.emit("turn_end");
  assert.deepEqual(f.probes, [{ since: 5000, until: 20000, latest: false }]);
  assert.equal(f.entries.length, 1);
  assert.equal(f.entries[0].customType, BACKEND_ERROR_ENTRY);
  const renderer = f.renderers.get(BACKEND_ERROR_ENTRY), theme = { fg: (_color, value) => value };
  assert.match(renderer(f.entries[0], { expanded: false }, theme).text, /ctrl\+o to expand/);
  assert.doesNotMatch(renderer(f.entries[0], { expanded: false }, theme).text, /scheduler.py/);
  assert.match(renderer(f.entries[0], { expanded: true }, theme).text, /scheduler.py/);
  await f.emit("agent_settled");
  assert.equal(f.entries.length, 1);
});

test("repeated failures share one diagnostic entry but an explicit command can show it again", async () => {
  const f = fixture();
  for (let i = 0; i < 2; i++) {
    f.emit("message_end", { message: message() });
    await f.emit("turn_end");
  }
  assert.equal(f.entries.length, 1);
  await f.commands.get("backend-error").handler("", f.ctx);
  assert.equal(f.entries.length, 2);
  assert.equal(f.probes.at(-1).latest, true);
});

test("other providers and successful turns perform no diagnostic lookup", async () => {
  const f = fixture();
  f.ctx.model = { id: "another-provider" };
  assert.equal(f.emit("message_end", { message: { ...message(), errorMessage: "synthetic network error" } }), undefined);
  assert.equal(f.emit("message_end", { message: { ...message(), stopReason: "stop" } }), undefined);
  await f.emit("turn_end");
  assert.equal(f.probes.length, 0);
  assert.equal(f.entries.length, 0);
  assert.equal(backendErrorMessage("synthetic network error"), "synthetic network error");
});

test("a failed SSH lookup leaves a visible, retryable diagnostic instead of losing the error", async () => {
  const f = fixture(async () => { throw new Error("synthetic SSH failure"); });
  f.emit("message_end", { message: message() });
  await f.emit("turn_end");
  assert.equal(f.entries[0].data.status, "lookup_failed");
  assert.match(diagnosticLines(f.entries[0].data, true).join("\n"), /backend-error to retry/);
});

test("an old or missing traceback is described honestly and current backend health is separate", () => {
  const text = diagnosticLines({ status: "unavailable", backend: { ready: true } }, true).join("\n");
  assert.match(text, /No backend traceback was recorded/);
  assert.match(text, /currently ready/);
  assert.doesNotMatch(text, /Traceback \(most recent/);
});

test("manual healthy backend lookup reports readiness without failure advice or error coloring", async () => {
  const healthy = { schema: report.schema, status: "unavailable", incident: null, latest: true,
    backend: { ready: true, running: true }, captured_at: 20000,
    diagnosis: { kind: "backend_ready", summary: "Backend is ready. No recorded backend error was found.", recovery: "" } };
  const f = fixture(async () => healthy);
  await f.commands.get("backend-error").handler("", f.ctx);
  assert.equal(f.entries.length, 1);
  const colors = [], theme = { fg: (color, value) => { colors.push(color); return value; } };
  const render = f.renderers.get(BACKEND_ERROR_ENTRY);
  for (const expanded of [false, true]) {
    const text = render(f.entries[0], { expanded }, theme).text;
    assert.match(text, /Backend is ready\. No recorded backend error was found\./);
    assert.doesNotMatch(text, /interrupted|failed request|retry|restart|relaunch|clean stop|forced kill/i);
  }
  assert.deepEqual(colors, ["success", "success"]);
  assert.equal(f.probes[0].latest, true);
});

test("healthy manual lookup corrects an older generic interrupted-request diagnosis", async () => {
  const legacy = { schema: report.schema, status: "unavailable", incident: null, latest: true,
    backend: { ready: true }, captured_at: 20000,
    diagnosis: { kind: "cause_unknown", summary: "Backend request was interrupted; cause unknown.",
      recovery: "Retry your message or restart the backend." } };
  const f = fixture(async () => legacy);
  await f.commands.get("backend-error").handler("", f.ctx);
  for (const expanded of [false, true]) {
    const text = diagnosticLines(f.entries[0].data, expanded).join("\n");
    assert.match(text, /Backend is ready\. No recorded backend error was found\./);
    assert.doesNotMatch(text, /interrupted|failed request|retry|restart|relaunch|clean stop|forced kill/i);
  }
});

test("automatic failures retain their diagnosis even when the backend is ready now", async () => {
  const failed = { schema: report.schema, status: "unavailable", incident: null, latest: false,
    backend: { ready: true }, captured_at: 20000,
    diagnosis: { kind: "cause_unknown", summary: "Backend request was interrupted; cause unknown.",
      recovery: "Retry your message." } };
  const f = fixture(async () => failed);
  f.emit("message_end", { message: message() });
  await f.emit("turn_end");
  const colors = [], theme = { fg: (color, value) => { colors.push(color); return value; } };
  const text = f.renderers.get(BACKEND_ERROR_ENTRY)(f.entries[0], { expanded: true }, theme).text;
  assert.match(text, /Backend request was interrupted/);
  assert.match(text, /Retry your message/);
  assert.doesNotMatch(text, /No recorded backend error was found/);
  assert.deepEqual(colors, ["error"]);
});

test("manual lookup still shows historical errors after backend recovery", async () => {
  const historical = { ...report, latest: true, backend: { ready: true, running: true } };
  const f = fixture(async () => historical);
  await f.commands.get("backend-error").handler("", f.ctx);
  const colors = [], theme = { fg: (color, value) => { colors.push(color); return value; } };
  const text = f.renderers.get(BACKEND_ERROR_ENTRY)(f.entries[0], { expanded: true }, theme).text;
  assert.match(text, /Latest recorded backend failure/);
  assert.match(text, /scheduler\.py/);
  assert.match(text, /currently ready/);
  assert.doesNotMatch(text, /No recorded backend error was found/);
  assert.deepEqual(colors, ["error"]);
});

test("ready backend with failed manual journal lookup does not claim no errors", () => {
  for (const status of ["unavailable", "lookup_failed"]) {
    const unavailable = { schema: report.schema, status, incident: null, latest: true,
      backend: { ready: true }, lookup_issue: "lookup_timeout" };
    const text = diagnosticLines(unavailable, true).join("\n");
    assert.match(text, /lookup failed or timed out/);
    assert.match(text, /diagnostic lookup timed out/);
    assert.doesNotMatch(text, /No recorded backend error was found|No backend traceback was found/);
  }
});

test("late diagnostics cannot be appended to a switched or closed chat", async () => {
  for (const event of ["session_switch", "session_shutdown"]) {
    let resolve;
    const f = fixture(() => new Promise((done) => { resolve = done; }));
    f.emit("message_end", { message: message() });
    await Promise.resolve();
    const flushing = f.emit("turn_end");
    f.emit(event);
    f.switch();
    resolve(report);
    await flushing;
    assert.equal(f.entries.length, 0);
  }
});

test("compaction can report the same backend failure without generating a chat message", async () => {
  const f = fixture();
  assert.equal(await reportBackendFailure(f.pi, f.ctx, failure, 10000), true);
  assert.equal(f.entries.length, 1);
  assert.equal(await reportBackendFailure(f.pi, f.ctx, "ordinary compaction validation failure"), false);
  assert.equal(f.entries.length, 1);
});

test("terminal escape sequences cannot be replayed by a traceback", () => {
  const malicious = { ...report, incident: { ...report.incident, traceback: "before\x1b]52;c;bad\x07after\x00" } };
  const text = diagnosticLines(malicious, true).join("\n");
  assert.match(text, /beforeafter/);
  assert.doesNotMatch(text, /\x1b|\x07|\x00/);
});

test("a diagnostic persistence failure cannot bypass compaction cancellation", async () => {
  const f = fixture();
  f.pi.appendEntry = () => { throw new Error("synthetic persistence failure"); };
  assert.equal(await reportBackendFailure(f.pi, f.ctx, failure, 10000), false);
});

test("connection diagnostics automatically fetch backend evidence after the failed message without model text", async () => {
  const f = fixture();
  const capture = createTransportDiagnostics({ fetch: async () => {
    throw new TypeError("fetch failed", { cause: Object.assign(new Error(), { code: "UND_ERR_CONNECT_TIMEOUT" }) });
  } });
  await assert.rejects(capture.fetch("http://127.0.0.1:18080/v1/chat/completions", {}));
  const failure = { ...message(), errorMessage: "Request timed out.", [TRANSPORT_ERROR]: capture.failure(new Error("Request timed out.")) };
  f.emit("message_end", { message: failure });
  assert.equal(f.entries.length, 0);
  assert.doesNotMatch(JSON.stringify(failure), /UND_ERR_CONNECT_TIMEOUT/);
  await f.emit("turn_end");
  await f.emit("agent_settled");
  assert.equal(f.probes.length, 1);
  assert.equal(f.entries.length, 1);
  assert.equal(f.entries[0].customType, TRANSPORT_ERROR_ENTRY);
  assert.equal(f.entries[0].data.backend_report.incident.id, "incident-a");
  const render = f.renderers.get(TRANSPORT_ERROR_ENTRY), theme = { fg: (_color, value) => value };
  assert.match(render(f.entries[0], { expanded: false }, theme).text, /UND_ERR_CONNECT_TIMEOUT.*ctrl\+o/);
  assert.match(render(f.entries[0], { expanded: true }, theme).text, /Endpoint: http:\/\/127.0.0.1:18080/);
  f.emit("message_end", { message: failure });
  await f.emit("turn_end");
  assert.equal(f.entries.length, 1);
});

test("collapsed connection failures explain a confirmed host restart and how to recover", async () => {
  const f = fixture(async () => ({ schema: report.schema, status: "unavailable", incident: null,
    backend: { ready: true, running: true },
    diagnosis: { kind: "host_restarted", summary: "AI host restarted; fatal CPU watchdog error recorded on CPU 9.",
      recovery: "Check /backend status, then restart Pi to recreate its connection." } }));
  const transport = { schema: "urn:qwen-r9700:transport-error:v1", id: "reboot", timestamp: 10000,
    elapsed_ms: 60000, request_elapsed_ms: 60000, endpoint: "http://127.0.0.1:8013/v1/chat/completions",
    phase: "stream", attempt: 1, headers_ms: 300, http_status: 200, sdk_error: "TypeError",
    causes: [{ parent: null, name: "SocketError", code: "UND_ERR_SOCKET" }] };
  f.emit("message_end", { message: { ...message(), [TRANSPORT_ERROR]: transport } });
  await f.emit("turn_end");
  const renderer = f.renderers.get(TRANSPORT_ERROR_ENTRY), theme = { fg: (_color, value) => value };
  const text = renderer(f.entries[0], { expanded: false }, theme).text;
  assert.match(text, /AI host restarted/);
  assert.match(text, /restart Pi/);
});

test("every failed Qwen request is diagnosed, including HTTP and incomplete-stream errors", async () => {
  for (const errorMessage of ["400 invalid or unsupported inference request", "Stream ended without finish reason"]) {
    const f = fixture();
    f.emit("message_end", { message: { ...message(), errorMessage } });
    await f.emit("turn_end");
    assert.equal(f.probes.length, 1);
    assert.equal(f.entries.length, 1);
  }
});

test("a dead local tunnel is distinguished from a stopped backend", () => {
  const transport = { elapsed_ms: 0, request_elapsed_ms: 0, sdk_error: "Error", phase: "connect", causes: [{ code: "ECONNREFUSED" }],
    local_connection: { listening: false, code: "ECONNREFUSED" },
    backend_report: { status: "unavailable", backend: { ready: true },
      diagnosis: { kind: "cause_unknown", summary: "Backend is ready now." } } };
  const text = connectionDiagnosticLines(transport, false).join("\n");
  assert.match(text, /local tunnel or relay is not listening/);
  assert.match(text, /relaunch Pi or pi-opsec/);
  assert.doesNotMatch(text, /backend start/);
});

test("an unreachable diagnostic host is reported without pretending to know it rebooted", () => {
  const backend_report = lookupFailure({ since: 10000, until: 20000 }, "host_unreachable", "ssh", { code: 255 });
  const text = connectionDiagnosticLines({ elapsed_ms: 0, phase: "connect", sdk_error: "Error",
    causes: [{ code: "ECONNREFUSED" }], backend_report }, false).join("\n");
  assert.match(text, /could not reach the AI host/);
  assert.doesNotMatch(text, /host restarted|CPU/);
});

test("local listener probe sends no HTTP/model request and identifies a closed port", async (t) => {
  let dataReceived = false;
  const server = createServer((socket) => { socket.on("data", () => { dataReceived = true; }); });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => { server.close(); });
  const endpoint = `http://127.0.0.1:${server.address().port}/v1/chat/completions`;
  assert.deepEqual(await probeLocalConnection(endpoint), { listening: true });
  await new Promise((resolve) => server.close(resolve));
  assert.deepEqual(await probeLocalConnection(endpoint), { listening: false, code: "ECONNREFUSED" });
  assert.equal(dataReceived, false);
  assert.equal(await probeLocalConnection("https://external.invalid/v1/chat/completions"), undefined);
});

test("compaction HTTP and transport failures get evidence; validation failures do not", async () => {
  const f = fixture();
  assert.equal(await reportBackendFailure(f.pi, f.ctx, "compaction endpoint returned HTTP 503", 10000), true);
  assert.equal(f.probes.length, 1);
  assert.equal(await reportBackendFailure(f.pi, f.ctx, "checkpoint missing required Key Decisions section"), false);
});
