import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { createConnection } from "node:net";
import { radianceBridgeRequest, radianceBridgeUrl } from "./qwen-radiance-bridge.mjs";
import { TRANSPORT_ERROR, TRANSPORT_SCHEMA, transportDiagnosticLines } from "./qwen-transport-diagnostics.mjs";

export const BACKEND_ERROR_ENTRY = "qwen-radiance-backend-error-v1";
export const TRANSPORT_ERROR_ENTRY = "qwen-radiance-transport-error-v1";
const SCHEMA = "urn:qwen-r9700:backend-error:v1";
const MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate";
const reporters = globalThis[Symbol.for("qwen.radiance.backend.errors")] ??= new WeakMap();

export const isBackendFailure = (message) => typeof message === "string" &&
  /EngineCore encountered an issue|EngineDeadError|EngineCore (?:encountered a fatal error|failed to start)/.test(message);
export const isBackendRequestFailure = (message) => isBackendFailure(message) || typeof message === "string" &&
  /compaction endpoint returned HTTP|stream ended|Provider streaming error|Connection error|Request timed out|fetch failed|UND_ERR_|ECONN|ETIMEDOUT|^terminated$/i.test(message);
export const backendErrorMessage = (message) => isBackendFailure(message)
  ? message.replace(/\s*See stack trace \(above\) for the root cause\.?/g, "").trim() : message;
const sessionKey = (ctx) => ctx.sessionManager.getSessionFile() ?? ctx.sessionManager.getSessionId();
const safeText = (text) => text.replace(/\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))/g, "")
  .replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f]/g, "");

const LOOKUP_ISSUES = {
  host_unreachable: "The diagnostic SSH connection could not reach the AI host.",
  authentication_failed: "SSH authentication failed; backend diagnostics could not be read.",
  lookup_timeout: "The backend diagnostic lookup timed out.",
  bridge_unavailable: "The VM's diagnostic relay could not return backend evidence.",
  invalid_report: "The diagnostic probe returned an invalid report.",
  host_not_configured: "The backend host is not configured for diagnostics.",
  probe_failed: "The backend diagnostic probe failed.",
};

const isHealthyManualLookup = (report) => report.latest === true && report.status === "unavailable" &&
  !report.incident && report.backend?.ready === true && !report.lookup_issue &&
  (!report.diagnosis || ["backend_ready", "cause_unknown"].includes(report.diagnosis.kind));

export function lookupFailure({ since, until, latest = false }, reason, stage = "probe", error) {
  return { schema: SCHEMA, status: "lookup_failed", incident: null, since, until, latest,
    lookup_issue: reason, lookup_failure: { stage, reason,
      ...(typeof error?.code === "number" ? { exit_code: error.code } : {}),
      ...(error?.killed ? { timed_out: true } : {}) } };
}

function validReport(report) {
  if (report?.schema !== SCHEMA || !["found", "unavailable"].includes(report.status) ||
      (report.status === "found" && (typeof report.incident?.traceback !== "string" ||
        report.incident.traceback.length > 66000 || typeof report.incident.summary !== "string" ||
        !Number.isSafeInteger(report.incident.timestamp)))) throw new Error("invalid backend diagnostic");
  if (report.diagnosis !== undefined && (typeof report.diagnosis?.summary !== "string" ||
      report.diagnosis.summary.length > 2048 || typeof report.diagnosis.recovery !== "string" ||
      report.diagnosis.recovery.length > 1024)) throw new Error("invalid backend diagnosis");
  return report;
}

export async function probeLocalConnection(endpoint, { connect = createConnection } = {}) {
  let url;
  try { url = new URL(endpoint); } catch { return undefined; }
  if (!["http:", "https:"].includes(url.protocol) || !["127.0.0.1", "[::1]"].includes(url.hostname)) return undefined;
  return await new Promise((resolve) => {
    let socket, settled = false;
    const done = (value) => {
      if (settled) return;
      settled = true;
      socket?.destroy();
      resolve(value);
    };
    try {
      socket = connect({ host: url.hostname.replace(/^\[|\]$/g, ""), port: Number(url.port) || (url.protocol === "https:" ? 443 : 80) });
      socket.setTimeout(300);
      socket.once("connect", () => done({ listening: true }));
      socket.once("timeout", () => done({ listening: null, code: "ETIMEDOUT" }));
      socket.once("error", (error) => done({ listening: error.code === "ECONNREFUSED" ? false : null,
        code: ["ECONNREFUSED", "ECONNRESET", "EHOSTUNREACH", "ENETUNREACH"].includes(error.code) ? error.code : "UNKNOWN" }));
    } catch { done({ listening: null, code: "PROBE_FAILED" }); }
  });
}

export async function fetchBackendError({ since, until, latest = false, host = process.env.QWEN_RADIANCE_CACHE_HOST }) {
  const options = { since, until, latest };
  if (radianceBridgeUrl()) {
    try {
      return validReport(await radianceBridgeRequest({ operation: "error", since: Math.floor(since), until: Math.floor(until), latest }, { timeout: 12000 }));
    } catch (error) { return lookupFailure(options, "bridge_unavailable", "bridge", error); }
  }
  if (!host || !/^[a-zA-Z0-9][a-zA-Z0-9_.@-]*$/.test(host)) return lookupFailure(options, "host_not_configured");
  // An already-open legacy Pi has no container variable when extensions reload.
  const container = process.env.QWEN_RADIANCE_CONTAINER || "qwen38-27b-uncensored-mxfp4-public-snapshot-candidate";
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(container)) throw new Error("invalid backend container");
  const source = await readFile(new URL("../../src/qwen_r9700_lab/radiance_error_report.py", import.meta.url), "utf8");
  const probeArgs = ["-", String(Math.floor(since)), String(Math.floor(until)),
    "--container", container, ...(latest ? ["--latest"] : [])];
  const args = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", host, "/usr/bin/python3", "-",
    ...probeArgs.slice(1)];
  return await new Promise((resolve) => {
    const local = host === "local";
    const child = execFile(local ? "python3" : "ssh", local ? probeArgs : args, { timeout: 10000, maxBuffer: 256 * 1024, encoding: "utf8" }, (error, stdout, stderr) => {
      if (error) {
        const reason = /Permission denied|Host key verification failed/i.test(stderr ?? "") ? "authentication_failed" :
          /Connection refused|Connection timed out|No route to host|Network is unreachable|Could not resolve hostname/i.test(stderr ?? "") ? "host_unreachable" :
          error.killed ? "lookup_timeout" : "probe_failed";
        return resolve(lookupFailure(options, reason, local ? "probe" : "ssh", error));
      }
      try {
        resolve(validReport(JSON.parse(stdout)));
      } catch { resolve(lookupFailure(options, "invalid_report")); }
    });
    child.stdin.on("error", () => {});
    child.stdin.end(source);
  });
}

export function diagnosticLines(report, expanded) {
  const incident = report.incident;
  const healthy = isHealthyManualLookup(report);
  const lookupProblem = report.status === "lookup_failed" || Boolean(report.lookup_issue);
  const summary = healthy ? "Backend is ready. No recorded backend error was found." :
    report.diagnosis?.summary ?? (incident ? `Backend traceback: ${incident.summary}` :
      lookupProblem ? "Backend diagnostics could not be fetched" : "Backend diagnostics: no recorded traceback");
  const recovery = healthy ? "" : report.diagnosis?.recovery;
  if (!expanded) return [safeText(`${summary} (ctrl+o to expand)`), ...(recovery ? [safeText(recovery)] : [])];
  const lines = [summary];
  if (incident) {
    lines.push(`Recorded ${new Date(incident.timestamp).toISOString()} · backend ${incident.container_id.slice(0, 12)}`);
    if (report.latest) lines.push("Latest recorded backend failure; it may predate the current request.");
    lines.push("", incident.traceback);
  } else {
    lines.push(lookupProblem ? "The backend diagnostic lookup failed or timed out. Use /backend-error to retry." :
      healthy ? "No backend traceback was found in the retained journal from the last 24 hours." :
      (report.latest ? "No backend traceback was found in the backend journal from the last 24 hours." :
        "No backend traceback was recorded in this request's time window in the retained backend journal.") +
      " A clean stop or forced kill may leave no Python traceback.");
    if (report.lookup_issue) lines.push(LOOKUP_ISSUES[report.lookup_issue] ?? `Journal lookup: ${report.lookup_issue.replaceAll("_", " ")}.`);
  }
  if (report.backend) lines.push("", report.backend.ready ? "Backend is currently ready." :
    report.backend.running ? "Backend process exists but is not ready." :
      report.backend.running === false ? "Backend container is no longer running." : "Backend is not ready; container state could not be confirmed.");
  if (report.backend?.running === false && report.backend.exit_code !== undefined && report.backend.exit_code !== null) lines.push(`Container exit code: ${report.backend.exit_code}${report.backend.oom_killed ? " · OOM killed" : ""}`);
  if (report.captured_at) lines.push(`Backend checked: ${new Date(report.captured_at).toISOString()} (${report.latest ? "current status" : "current status, separate from the failed request"}).`);
  if (report.host?.boot_started_at) lines.push(`Host boot: ${new Date(report.host.boot_started_at).toISOString()} · boot ${report.host.boot_id ?? "unknown"}`);
  for (const fs of report.host?.filesystems ?? []) if (Number.isFinite(fs.available_bytes)) {
    lines.push(`${fs.name === "cache" ? "Cache" : "System"} filesystem: ${(fs.available_bytes / 1024 ** 3).toFixed(2)} GiB available · ${fs.available_inodes} free inodes`);
  }
  if (recovery) lines.push("", recovery);
  return lines.map(safeText);
}

export function connectionDiagnosticLines(report, expanded) {
  const backend = report.backend_report;
  let diagnosis = backend?.diagnosis;
  if ((!diagnosis || diagnosis.kind === "cause_unknown") && backend?.backend?.ready && report.local_connection?.listening === false) {
    diagnosis = { summary: "Pi's local tunnel or relay is not listening; the AI backend is ready.",
      recovery: "Quit and relaunch Pi or pi-opsec to recreate the connection, then retry your message." };
  }
  if (!diagnosis && backend?.lookup_issue) diagnosis = {
    summary: LOOKUP_ISSUES[backend.lookup_issue] ?? "Backend evidence could not be retrieved; the underlying cause is unconfirmed.",
    recovery: "Check /backend status. If the local endpoint is refused, relaunch Pi or pi-opsec to recreate its connection.",
  };
  const lines = diagnosis ? [diagnosis.summary, diagnosis.recovery] : [];
  lines.push(...transportDiagnosticLines(report, expanded));
  if (expanded && backend) lines.push("", ...diagnosticLines(backend, true));
  if (expanded && report.local_connection) lines.push(`Local endpoint: ${report.local_connection.listening === true ? "TCP listener reachable" :
    report.local_connection.listening === false ? "no TCP listener" : "listener check inconclusive"}${report.local_connection.code ? ` (${report.local_connection.code})` : ""}`);
  return lines.filter(Boolean).map(safeText);
}

export async function reportBackendFailure(pi, ctx, message, since = Date.now()) {
  if (!isBackendRequestFailure(message)) return false;
  try { return await reporters.get(pi)?.report(ctx, { since }) ?? false; }
  catch { return false; } // Diagnostics must never cancel the compactor's error handling.
}

export function installRadianceErrors(pi, { Text, probe = fetchBackendError, connectionProbe = probeLocalConnection, now = () => Date.now() }) {
  let active = true, epoch = 0;
  const pending = [], shown = new Set();
  pi.registerEntryRenderer(BACKEND_ERROR_ENTRY, (entry, { expanded, outputPad = 1 }, theme) =>
    new Text(theme.fg(isHealthyManualLookup(entry.data) ? "success" : "error", diagnosticLines(entry.data, expanded).join("\n")), outputPad, 0));
  pi.registerEntryRenderer(TRANSPORT_ERROR_ENTRY, (entry, { expanded, outputPad = 1 }, theme) =>
    new Text(theme.fg("error", connectionDiagnosticLines(entry.data, expanded).join("\n")), outputPad, 0));

  function start(ctx, { since, latest = false, transport }) {
    const key = sessionKey(ctx), generation = epoch, until = now();
    since = Number.isFinite(since) ? Math.max(until - 86390000, Math.min(since, until)) : until - 300000;
    const options = { since, until, latest };
    const result = Promise.resolve().then(() => probe(options)).catch(() => lookupFailure(options, "probe_failed"));
    const connection = transport ? Promise.resolve().then(() => connectionProbe(transport.endpoint)).catch(() => undefined) : undefined;
    return async () => {
      const [report, local] = await Promise.all([result, connection]);
      if (!active || generation !== epoch || sessionKey(ctx) !== key) return false;
      const id = transport?.id ?? `${key}:${report.incident?.id ?? until}`;
      if (!latest && shown.has(id)) return false;
      shown.add(id);
      if (shown.size > 32) shown.delete(shown.values().next().value);
      pi.appendEntry(transport ? TRANSPORT_ERROR_ENTRY : BACKEND_ERROR_ENTRY,
        transport ? { ...transport, backend_report: report, local_connection: local } : report);
      return true;
    };
  }
  async function flush() {
    for (const finish of pending.splice(0)) await finish();
  }
  reporters.set(pi, { report: (ctx, options) => start(ctx, options)() });
  pi.on("message_end", (event, ctx) => {
    const message = event.message;
    if (ctx.model?.id !== MODEL || message?.role !== "assistant" || message.stopReason !== "error") return;
    const transport = message[TRANSPORT_ERROR];
    if (transport?.schema === TRANSPORT_SCHEMA) {
      if (shown.has(transport.id)) return;
      pending.push(start(ctx, { since: transport.timestamp, transport }));
      return;
    }
    pending.push(start(ctx, { since: message.timestamp }));
    return { message: { ...message, errorMessage: backendErrorMessage(message.errorMessage) } };
  });
  // The error message is rendered and persisted before its diagnostic entry.
  pi.on("turn_end", flush);
  pi.on("agent_settled", flush);
  for (const event of ["session_switch", "session_tree"]) pi.on(event, () => { epoch++; pending.length = 0; });
  pi.on("session_shutdown", () => { active = false; epoch++; pending.length = 0; });
  pi.registerCommand("backend-error", {
    description: "Show backend and host failure evidence, current health and recorded tracebacks",
    handler: async (_args, ctx) => { await start(ctx, { since: now() - 3600000, latest: true })(); },
  });
}
