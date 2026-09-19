import test from "node:test";
import assert from "node:assert/strict";
import { BACKEND_ERROR_ENTRY, backendErrorMessage, diagnosticLines, installRadianceErrors,
  reportBackendFailure } from "../integrations/pi/qwen-radiance-errors.mjs";

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
  installRadianceErrors(pi, { Text, now: () => 20000, probe: (args) => { probes.push(args); return probe(args); } });
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

test("ordinary errors and successful turns perform no diagnostic lookup", async () => {
  const f = fixture();
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
  assert.match(text, /No EngineCore traceback matched this request/);
  assert.match(text, /currently ready/);
  assert.doesNotMatch(text, /Traceback \(most recent/);
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
