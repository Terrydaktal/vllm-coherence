import test from "node:test";
import assert from "node:assert/strict";
import { createTransportDiagnostics, transportCauses, transportDiagnosticLines } from "../integrations/pi/qwen-transport-diagnostics.mjs";

const timeout = () => Object.assign(new Error("private URL, authorization or request content"),
  { name: "ConnectTimeoutError", code: "UND_ERR_CONNECT_TIMEOUT" });
const sdkTimeout = () => Object.assign(new Error("Request timed out."), { name: "APIConnectionTimeoutError" });

test("retains the cause the SDK discards, timing the fetch rather than prompt preparation", async () => {
  let clock = 0;
  const original = new TypeError("fetch failed", { cause: timeout() });
  const capture = createTransportDiagnostics({ now: () => clock, wall: () => 1234,
    fetch: async () => { clock += 10559; throw original; } });
  clock = 400;
  await assert.rejects(capture.fetch("http://user:secret@127.0.0.1:18080/v1/chat/completions?key=secret", {}), (e) => e === original);
  const report = capture.failure(sdkTimeout());
  assert.equal(report.phase, "connect");
  assert.equal(report.elapsed_ms, 10559);
  assert.equal(report.request_elapsed_ms, 10959);
  assert.equal(report.timestamp, 1234);
  assert.equal(report.headers_ms, null);
  assert.equal(report.endpoint, "http://127.0.0.1:18080/v1/chat/completions");
  assert.deepEqual(report.causes.map((cause) => cause.code), [undefined, "UND_ERR_CONNECT_TIMEOUT"]);
  const expanded = transportDiagnosticLines(report, true).join("\n");
  assert.match(expanded, /10.56s.*connecting/);
  assert.match(expanded, /No HTTP response headers received/);
  assert.doesNotMatch(JSON.stringify(report) + expanded, /private|authorization|secret|key=/);
});

test("raw exception fields are bounded and allowlisted, including aggregate causes and cycles", () => {
  const error = Object.assign(new Error("SECRET"), { code: "SECRET", name: "SECRET", syscall: "SECRET",
    address: "SECRET", port: "SECRET", errno: "SECRET", headers: { auth: "SECRET" }, stack: "SECRET" });
  error.cause = error;
  const root = new AggregateError([error, Object.assign(new Error("SECRET"),
    { code: "ECONNREFUSED", syscall: "connect", address: "127.0.0.1", port: 18080, errno: -111 })], "SECRET");
  const causes = transportCauses(root);
  assert.equal(causes.length, 3);
  assert.equal(causes[2].parent, 0);
  assert.equal(causes[2].code, "ECONNREFUSED");
  assert.doesNotMatch(JSON.stringify(causes), /SECRET/);
  assert.equal(transportCauses(new AggregateError(Array.from({ length: 100 }, timeout))).length, 12);
});

test("retains streaming failure timings without wrapping or reading the response body", async () => {
  let clock = 0;
  const response = new Response("synthetic stream");
  const capture = createTransportDiagnostics({ now: () => clock, fetch: async () => { clock = 125; return response; } });
  assert.equal(await capture.fetch("http://127.0.0.1/v1/chat/completions", {}), response);
  assert.equal(response.bodyUsed, false);
  clock = 4000;
  const report = capture.failure(new TypeError("terminated", { cause: Object.assign(new Error(), { code: "UND_ERR_BODY_TIMEOUT" }) }));
  assert.equal(report.phase, "stream");
  assert.equal(report.headers_ms, 125);
  assert.equal(report.elapsed_ms, 4000);
  assert.equal(report.http_status, 200);
});

test("an SDK abort before headers is not misreported as a proved connect timeout", async () => {
  const capture = createTransportDiagnostics({ fetch: async () => { throw new DOMException("aborted", "AbortError"); } });
  await assert.rejects(capture.fetch("http://127.0.0.1/v1/chat/completions", {}));
  assert.equal(capture.failure(sdkTimeout()).phase, "before_headers");
});

test("cancellation and nontransport HTTP/parser errors do not become connection incidents", async () => {
  const controller = new AbortController();
  const capture = createTransportDiagnostics({ signal: controller.signal, fetch: async () => { throw timeout(); } });
  await assert.rejects(capture.fetch("http://127.0.0.1", {}));
  controller.abort();
  assert.equal(capture.failure(sdkTimeout()), undefined);
  const success = createTransportDiagnostics({ fetch: async () => new Response("{}", { status: 400 }) });
  assert.equal(success.failure(new Error("prompt capture intentionally stopped")), undefined);
  await success.fetch("http://127.0.0.1", {});
  assert.equal(success.failure(new Error("HTTP 400")), undefined);
  assert.equal(success.failure(new Error("Stream ended without finish_reason")), undefined);
});

test("retry and concurrent request records stay separate", async () => {
  let attempts = 0;
  const a = createTransportDiagnostics({ fetch: async () => { if (++attempts < 3) throw timeout(); return new Response(""); } });
  const b = createTransportDiagnostics({ fetch: async () => { throw Object.assign(new Error(), { code: "ECONNRESET" }); } });
  await Promise.all([assert.rejects(a.fetch("http://127.0.0.1:8012", {})), assert.rejects(b.fetch("http://127.0.0.1:8013", {}))]);
  await assert.rejects(a.fetch("http://127.0.0.1:8012", {}));
  const report = a.failure(sdkTimeout());
  assert.equal(report.attempt, 2);
  assert.equal(report.previous_attempts.length, 1);
  assert.equal(b.failure(sdkTimeout()).causes[0].code, "ECONNRESET");
  assert.notEqual(report.id, b.failure(sdkTimeout()).id);
  await a.fetch("http://127.0.0.1:8012", {});
  assert.equal(a.failure(new Error("HTTP/format error after recovery")), undefined);
});

const local = "http://127.0.0.1:18080/v1/chat/completions";

test("one failed local handshake retries the same JSON before submission", async () => {
  const attempts = [];
  const response = new Response("synthetic stream");
  const capture = createTransportDiagnostics({ fetch: async (input, init) => {
    attempts.push([input, init]);
    if (attempts.length === 1) throw new TypeError("fetch failed", { cause: timeout() });
    return response;
  } });
  const init = { method: "POST", body: '{"fixture":true}' };
  assert.equal(await capture.fetch(local, init), response);
  assert.deepEqual(attempts, [[local, { ...init, redirect: "error" }], [local, { ...init, redirect: "error" }]]);
  assert.equal(response.bodyUsed, false);
  assert.equal(capture.failure(new Error("later format failure")), undefined);
});

test("repeated connect timeout stops after two attempts and keeps both diagnostics", async () => {
  let attempts = 0;
  const capture = createTransportDiagnostics({ fetch: async () => { attempts++; throw timeout(); } });
  await assert.rejects(capture.fetch(local, { method: "POST", body: "{}" }));
  assert.equal(attempts, 2);
  const report = capture.failure(sdkTimeout());
  assert.equal(report.attempt, 2);
  assert.equal(report.previous_attempts.length, 1);
  assert.equal(report.phase, "connect");
});

test("failures after possible submission are never replayed", async () => {
  for (const code of ["ECONNRESET", "EPIPE", "ETIMEDOUT", "UND_ERR_SOCKET", "UND_ERR_HEADERS_TIMEOUT", "UND_ERR_BODY_TIMEOUT"]) {
    let attempts = 0;
    const capture = createTransportDiagnostics({ fetch: async () => {
      attempts++; throw Object.assign(new Error(), { code });
    } });
    await assert.rejects(capture.fetch(local, { method: "POST", body: "{}" }));
    assert.equal(attempts, 1, code);
  }
});

test("remote endpoints, credentials and consumed or streaming bodies do not get retried", async () => {
  for (const [input, init] of [
    ["https://remote.invalid/v1/chat/completions", { body: "{}" }],
    ["http://secret@127.0.0.1/v1/chat/completions", { body: "{}" }],
    [new Request(local, { method: "POST", body: "{}" }), undefined],
    [local, { body: new ReadableStream() }],
  ]) {
    let attempts = 0;
    const capture = createTransportDiagnostics({ fetch: async () => { attempts++; throw timeout(); } });
    await assert.rejects(capture.fetch(input, init));
    assert.equal(attempts, 1);
  }
});

test("cancellation during reconnect backoff prevents another connection", async () => {
  const controller = new AbortController();
  let attempts = 0;
  const capture = createTransportDiagnostics({ signal: controller.signal, fetch: async () => {
    attempts++;
    setTimeout(() => controller.abort(), 5);
    throw timeout();
  } });
  await assert.rejects(capture.fetch(local, { body: "{}", signal: controller.signal }), { name: "AbortError" });
  assert.equal(attempts, 1);
  assert.equal(capture.failure(sdkTimeout()), undefined);
});
