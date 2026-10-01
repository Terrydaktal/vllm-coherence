import { randomUUID } from "node:crypto";
import { isIP } from "node:net";
import { setTimeout as delay } from "node:timers/promises";

// Shared by the provider adapter and the error renderer. Symbols never enter
// assistant JSON/model context; the renderer persists a separate custom entry.
export const TRANSPORT_ERROR = Symbol.for("qwen-r9700:transport-error:v1");
export const TRANSPORT_SCHEMA = "urn:qwen-r9700:transport-error:v1";
const CODES = new Set([
  "UND_ERR_CONNECT_TIMEOUT", "UND_ERR_HEADERS_TIMEOUT", "UND_ERR_BODY_TIMEOUT",
  "UND_ERR_SOCKET", "UND_ERR_ABORTED", "UND_ERR_DESTROYED", "UND_ERR_CLOSED",
  "ECONNREFUSED", "ECONNRESET", "ECONNABORTED", "ETIMEDOUT", "EPIPE",
  "ENETUNREACH", "EHOSTUNREACH", "ENOTFOUND", "EAI_AGAIN", "ABORT_ERR",
  "ERR_TLS_CERT_ALTNAME_INVALID", "CERT_HAS_EXPIRED", "DEPTH_ZERO_SELF_SIGNED_CERT",
  "UNABLE_TO_VERIFY_LEAF_SIGNATURE", "ERR_SSL_WRONG_VERSION_NUMBER",
]);
const NAMES = new Set([
  "Error", "TypeError", "AggregateError", "AbortError", "TimeoutError",
  "ConnectTimeoutError", "HeadersTimeoutError", "BodyTimeoutError", "SocketError",
  "RequestAbortedError", "APIConnectionTimeoutError", "APIConnectionError", "APIUserAbortError",
]);
const SYSCALLS = new Set(["connect", "read", "write", "getaddrinfo", "getnameinfo"]);
const PHASES = {
  dns: "resolving the server address", connect: "connecting to the server",
  headers: "waiting for response headers", stream: "reading the response stream",
  before_headers: "before response headers arrived",
};
const milliseconds = (value) => Math.round(Math.max(0, value) * 10) / 10;

export function transportCauses(error) {
  const pending = [{ error, parent: null }], seen = new Set(), result = [];
  while (pending.length && result.length < 12) {
    const { error: current, parent } = pending.shift();
    if (!current || typeof current !== "object" || seen.has(current)) continue;
    seen.add(current);
    const node = { parent, name: NAMES.has(current.name) ? current.name : "Error" };
    if (CODES.has(current.code)) node.code = current.code;
    if (Number.isSafeInteger(current.errno)) node.errno = current.errno;
    if (SYSCALLS.has(current.syscall)) node.syscall = current.syscall;
    if (typeof current.address === "string" && isIP(current.address)) node.address = current.address;
    if (Number.isInteger(current.port) && current.port > 0 && current.port <= 65535) node.port = current.port;
    const index = result.push(node) - 1;
    pending.push({ error: current.cause, parent: index });
    if (Array.isArray(current.errors)) {
      for (const child of current.errors.slice(0, 12)) pending.push({ error: child, parent: index });
    }
  }
  // Deliberately exclude free-form messages, stacks, bodies and headers. Fetch
  // errors can embed the full URL (including credentials) or application data.
  return result;
}

function endpoint(input) {
  try {
    const url = new URL(typeof input === "string" || input instanceof URL ? input : input.url);
    if (!["http:", "https:"].includes(url.protocol)) return "unavailable";
    const path = ["/v1/chat/completions", "/chat/completions", "/v1/completions"].includes(url.pathname)
      ? url.pathname : "/[path omitted]";
    return url.origin + path;
  } catch { return "unavailable"; }
}

function replayableLocalRequest(input, init) {
  // Stream bodies and Request objects may have been consumed. The SDK supplies
  // a URL and a serialized JSON string, which remains usable after a failed SYN.
  if (!(typeof input === "string" || input instanceof URL) ||
      (init?.body !== undefined && init.body !== null && typeof init.body !== "string")) return false;
  try {
    const url = new URL(input);
    return url.protocol === "http:" && url.hostname === "127.0.0.1" && !url.username && !url.password &&
      ["/v1/chat/completions", "/v1/completions"].includes(url.pathname);
  } catch { return false; }
}

function failedBeforeSend(causes) {
  return causes.some((cause) => cause.code === "UND_ERR_CONNECT_TIMEOUT" ||
    (cause.code === "ECONNREFUSED" && cause.syscall === "connect"));
}

export function createTransportDiagnostics({ fetch: customFetch, signal,
  now = () => performance.now(), wall = () => Date.now() } = {}) {
  const started = now();
  let attempt, attemptCount = 0, retriedConnect = false;
  const failures = [];
  return {
    async fetch(input, init) {
      // One recorder per provider request, so concurrent chats and SDK retries
      // cannot borrow another request's exception or last-success timings.
      const reconnectable = replayableLocalRequest(input, init);
      // A timeout on a followed redirect would not prove that the original
      // POST was unsent. The fixed local inference endpoint never redirects.
      const request = reconnectable ? { ...init, redirect: "error" } : init;
      for (;;) {
        signal?.throwIfAborted();
        init?.signal?.throwIfAborted();
        const current = attempt = { number: ++attemptCount, started: now(), timestamp: wall(), endpoint: endpoint(input) };
        try {
          const response = await (customFetch ?? globalThis.fetch)(input, request);
          current.headers_ms = milliseconds(now() - current.started);
          current.status = response.status;
          return response; // No stream wrapping, body copies, or per-token hooks.
        } catch (error) {
          current.failed_ms = milliseconds(now() - current.started);
          current.causes = transportCauses(error);
          failures.push({ attempt: current.number, elapsed_ms: current.failed_ms, causes: current.causes });
          if (failures.length > 4) failures.shift();
          if (retriedConnect || signal?.aborted || init?.signal?.aborted ||
              !reconnectable || !failedBeforeSend(current.causes)) throw error;
          // One reconnect, before any HTTP request was sent. Header/body
          // timeouts, resets and interrupted generations are never replayed.
          retriedConnect = true;
          await delay(50, undefined, { signal: init?.signal ?? signal });
        }
      }
    },
    failure(error) {
      if (!attempt || signal?.aborted) return undefined;
      const causes = attempt.causes ?? transportCauses(error);
      const codes = causes.map((cause) => cause.code);
      // HTTP/provider/format errors aren't transport failures. A successful
      // retry also must not inherit an earlier failed connection's diagnosis.
      if (!attempt.causes && !codes.some((code) => CODES.has(code)) &&
          !["APIConnectionTimeoutError", "APIConnectionError", "AbortError"].includes(error?.name)) return undefined;
      const phase = codes.some((code) => ["ENOTFOUND", "EAI_AGAIN"].includes(code)) ? "dns" :
        codes.includes("UND_ERR_CONNECT_TIMEOUT") || causes.some((cause) => cause.syscall === "connect") ? "connect" :
        codes.includes("UND_ERR_HEADERS_TIMEOUT") ? "headers" :
        attempt.headers_ms !== undefined ? "stream" : "before_headers";
      return {
        schema: TRANSPORT_SCHEMA, id: randomUUID(), timestamp: attempt.timestamp,
        endpoint: attempt.endpoint, phase, attempt: attempt.number,
        elapsed_ms: attempt.failed_ms ?? milliseconds(now() - attempt.started),
        request_elapsed_ms: milliseconds(now() - started),
        headers_ms: attempt.headers_ms ?? null, http_status: attempt.status ?? null,
        sdk_error: NAMES.has(error?.name) ? error.name : "Error", causes,
        previous_attempts: failures.filter((value) => value.attempt < attempt.number),
      };
    },
  };
}

export function transportDiagnosticLines(report, expanded) {
  const code = report.causes?.map((cause) => cause.code).filter(Boolean).join(" → ") || report.sdk_error;
  const summary = `Connection failure: ${code} · ${(report.elapsed_ms / 1000).toFixed(2)}s · ${PHASES[report.phase] ?? "transport stage unknown"}`;
  if (!expanded) return [`${summary} (ctrl+o to expand)`];
  const lines = [summary, `Endpoint: ${report.endpoint}`, `Recorded ${new Date(report.timestamp).toISOString()} · incident ${report.id}`,
    `Attempt ${report.attempt} · ${(report.request_elapsed_ms / 1000).toFixed(2)}s since provider request began`,
    report.headers_ms === null ? "No HTTP response headers received; backend admission is unconfirmed." :
      `HTTP ${report.http_status} headers after ${(report.headers_ms / 1000).toFixed(2)}s; failure occurred in the response stream.`,
    `SDK error: ${report.sdk_error}`, "Underlying cause chain:"];
  for (const [index, cause] of (report.causes ?? []).entries()) {
    lines.push(`  ${index}: ${cause.name}${cause.code ? ` [${cause.code}]` : ""}` +
      `${cause.parent === null ? "" : ` · cause of ${cause.parent}`}` +
      `${cause.syscall ? ` · ${cause.syscall}` : ""}${cause.address ? ` · ${cause.address}` : ""}` +
      `${cause.port ? `:${cause.port}` : ""}${cause.errno !== undefined ? ` · errno ${cause.errno}` : ""}`);
  }
  for (const previous of report.previous_attempts ?? []) {
    lines.push(`Earlier attempt ${previous.attempt}: ${(previous.elapsed_ms / 1000).toFixed(2)}s · ` +
      previous.causes.map((cause) => cause.code ?? cause.name).join(" → "));
  }
  lines.push("Request/response contents, headers, URL credentials and free-form exception text are not recorded.");
  return lines;
}
