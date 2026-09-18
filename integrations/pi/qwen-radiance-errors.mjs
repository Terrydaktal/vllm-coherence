import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { radianceBridgeRequest, radianceBridgeUrl } from "./qwen-radiance-bridge.mjs";

export const BACKEND_ERROR_ENTRY = "qwen-radiance-backend-error-v1";
const SCHEMA = "urn:qwen-r9700:backend-error:v1";
const MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate";
const reporters = globalThis[Symbol.for("qwen.radiance.backend.errors")] ??= new WeakMap();

export const isBackendFailure = (message) => typeof message === "string" &&
  /EngineCore encountered an issue|EngineDeadError|EngineCore (?:encountered a fatal error|failed to start)/.test(message);
export const backendErrorMessage = (message) => isBackendFailure(message)
  ? message.replace(/\s*See stack trace \(above\) for the root cause\.?/g, "").trim() : message;
const sessionKey = (ctx) => ctx.sessionManager.getSessionFile() ?? ctx.sessionManager.getSessionId();
const safeText = (text) => text.replace(/\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))/g, "")
  .replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f]/g, "");

export async function fetchBackendError({ since, until, latest = false, host = process.env.QWEN_RADIANCE_CACHE_HOST }) {
  if (radianceBridgeUrl()) {
    const report = await radianceBridgeRequest({ operation: "error", since: Math.floor(since), until: Math.floor(until), latest }, { timeout: 12000 });
    if (report?.schema !== SCHEMA || !["found", "unavailable"].includes(report.status) ||
        (report.status === "found" && (typeof report.incident?.traceback !== "string" ||
          report.incident.traceback.length > 66000 || typeof report.incident.summary !== "string" ||
          !Number.isSafeInteger(report.incident.timestamp)))) throw new Error("invalid backend diagnostic");
    return report;
  }
  if (!host || !/^[a-zA-Z0-9][a-zA-Z0-9_.@-]*$/.test(host)) throw new Error("backend host unavailable");
  const source = await readFile(new URL("../../src/qwen_r9700_lab/radiance_error_report.py", import.meta.url), "utf8");
  const args = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", host, "/usr/bin/python3", "-",
    String(Math.floor(since)), String(Math.floor(until)), ...(latest ? ["--latest"] : [])];
  return await new Promise((resolve, reject) => {
    const local = host === "local";
    const child = execFile(local ? "python3" : "ssh", local ? ["-", String(Math.floor(since)), String(Math.floor(until)), ...(latest ? ["--latest"] : [])] : args, { timeout: 8000, maxBuffer: 256 * 1024, encoding: "utf8" }, (error, stdout) => {
      if (error) return reject(new Error("backend diagnostic lookup failed"));
      try {
        const report = JSON.parse(stdout);
        if (report.schema !== SCHEMA || !["found", "unavailable"].includes(report.status) ||
            (report.status === "found" && (typeof report.incident?.traceback !== "string" ||
              report.incident.traceback.length > 66000 || typeof report.incident.summary !== "string" ||
              !Number.isSafeInteger(report.incident.timestamp)))) {
          throw new Error("invalid backend diagnostic");
        }
        resolve(report);
      } catch (error) { reject(error); }
    });
    child.stdin.on("error", () => {});
    child.stdin.end(source);
  });
}

export function diagnosticLines(report, expanded) {
  const incident = report.incident;
  const summary = incident ? `Backend traceback: ${incident.summary}` :
    report.status === "lookup_failed" ? "Backend diagnostics could not be fetched" : "Backend diagnostics: no recorded traceback";
  if (!expanded) return [safeText(`${summary} (ctrl+o to expand)`)];
  const lines = [summary];
  if (incident) {
    lines.push(`Recorded ${new Date(incident.timestamp).toISOString()} · backend ${incident.container_id.slice(0, 12)}`);
    if (report.latest) lines.push("Latest recorded backend failure; it may predate the current request.");
    lines.push("", incident.traceback);
  } else {
    lines.push(report.status === "lookup_failed" ? "The backend diagnostic lookup failed or timed out. Use /backend-error to retry." :
      (report.latest ? "No EngineCore traceback was found in the backend journal from the last 24 hours." :
        "No EngineCore traceback matched this request in the retained backend journal.") +
      " A clean stop or forced kill may leave no Python traceback.");
    if (report.lookup_issue) lines.push(`Journal lookup: ${report.lookup_issue.replaceAll("_", " ")}.`);
  }
  if (report.backend) lines.push("", report.backend.ready ? "Backend is currently ready." :
    report.backend.running ? "Backend process exists but is not ready." :
      report.backend.running === false ? "Backend container is no longer running." : "Backend is not ready; container state could not be confirmed.");
  return lines.map(safeText);
}

export async function reportBackendFailure(pi, ctx, message, since = Date.now()) {
  if (!isBackendFailure(message)) return false;
  try { return await reporters.get(pi)?.report(ctx, { since }) ?? false; }
  catch { return false; } // Diagnostics must never cancel the compactor's error handling.
}

export function installRadianceErrors(pi, { Text, probe = fetchBackendError, now = () => Date.now() }) {
  let active = true, epoch = 0;
  const pending = [], shown = new Set();
  pi.registerEntryRenderer(BACKEND_ERROR_ENTRY, (entry, { expanded, outputPad = 1 }, theme) =>
    new Text(theme.fg("error", diagnosticLines(entry.data, expanded).join("\n")), outputPad, 0));

  function start(ctx, { since, latest = false }) {
    const key = sessionKey(ctx), generation = epoch, until = now();
    since = Number.isFinite(since) ? Math.max(until - 86390000, Math.min(since, until)) : until - 300000;
    const result = Promise.resolve().then(() => probe({ since, until, latest })).catch(() => ({
      schema: SCHEMA, status: "lookup_failed", incident: null, since, until, latest,
    }));
    return async () => {
      const report = await result;
      if (!active || generation !== epoch || sessionKey(ctx) !== key) return false;
      const id = `${key}:${report.incident?.id ?? until}`;
      if (!latest && shown.has(id)) return false;
      shown.add(id);
      if (shown.size > 32) shown.delete(shown.values().next().value);
      pi.appendEntry(BACKEND_ERROR_ENTRY, report);
      return true;
    };
  }
  async function flush() {
    for (const finish of pending.splice(0)) await finish();
  }
  reporters.set(pi, { report: (ctx, options) => start(ctx, options)() });
  pi.on("message_end", (event, ctx) => {
    const message = event.message;
    if (ctx.model?.id !== MODEL || message?.role !== "assistant" || message.stopReason !== "error" ||
        !isBackendFailure(message.errorMessage)) return;
    pending.push(start(ctx, { since: message.timestamp }));
    return { message: { ...message, errorMessage: backendErrorMessage(message.errorMessage) } };
  });
  // The error message is rendered and persisted before its diagnostic entry.
  pi.on("turn_end", flush);
  pi.on("agent_settled", flush);
  for (const event of ["session_switch", "session_tree"]) pi.on(event, () => { epoch++; pending.length = 0; });
  pi.on("session_shutdown", () => { active = false; epoch++; pending.length = 0; });
  pi.registerCommand("backend-error", {
    description: "Show the latest Radiance backend traceback; Ctrl+O expands it",
    handler: async (_args, ctx) => { await start(ctx, { since: now() - 3600000, latest: true })(); },
  });
}
