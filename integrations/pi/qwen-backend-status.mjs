import { constants, closeSync, openSync, fstatSync, readSync } from "node:fs";
import { join } from "node:path";
import { readRadianceConnection } from "./qwen-radiance-connection.mjs";
import { radianceChatIdentity } from "./qwen-radiance-identity.mjs";
import { parseCombinedTelemetry, parseSchedulerSample, schedulerStatusForChat } from "./qwen-radiance-scheduler-telemetry.mjs";
import { parseResidencySample, cacheBreakdown } from "./qwen-cache-residency.mjs";
import { parseTemperatureSample } from "./qwen-gpu-temperature.mjs";

const integer = (value) => Number.isSafeInteger(value) && value >= 0;
const finite = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
const count = (value) => integer(value) ? value.toLocaleString("en-GB") : "unknown";
export const statusLabel = (value, limit = 180) => typeof value === "string" ? value
  .replace(/\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))/g, "")
  .replace(/[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060-\u206f]/g, " ")
  .replace(/\s+/g, " ").trim().slice(0, limit) : "unknown";
const seconds = (value) => finite(value) ? `${value.toFixed(1)}s` : "unknown";
const bytes = (value) => integer(value) ? `${(value / 1024 ** 3).toFixed(1)} GiB` : "unknown";
const timestamp = (value) => integer(value) && value <= 8.64e15 ? new Date(value).toISOString() : "unknown";
const age = (at, now) => integer(at) && at <= now + 1000 ? `${seconds(Math.max(0, now - at) / 1000)} old` : "age unknown";

// A command-scoped read reuses the existing recorder's files. It never starts a
// second telemetry monitor, reads message bodies, or changes the active context.
function localJson(path, maxBytes = 2_097_152) {
  let fd;
  try {
    fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    const info = fstatSync(fd);
    if (!info.isFile() || info.nlink !== 1 || info.uid !== process.getuid() || (info.mode & 0o022) || info.size > maxBytes) return;
    const buffer = Buffer.alloc(Math.min(maxBytes + 1, info.size + 1));
    const size = readSync(fd, buffer, 0, buffer.length, 0);
    if (size !== info.size || size > maxBytes) return;
    return JSON.parse(buffer.subarray(0, size).toString("utf8"));
  } catch { return; } finally { if (fd !== undefined) closeSync(fd); }
}

export function readBackendTelemetry(ctx, { env = process.env, now = Date.now(), read = localJson } = {}) {
  const directory = env.QWEN_RADIANCE_GPU_TEMPERATURE_STATE;
  if (!directory?.startsWith("/")) return {};
  const result = {};
  let chat, combined;
  try { chat = radianceChatIdentity(ctx); } catch { /* Unsaved sessions have no cache identity. */ }
  try { combined = parseCombinedTelemetry(JSON.stringify(read(join(directory, "telemetry-v1.json"))), now); } catch { /* Opsec uses the standalone feeds. */ }
  const schedulerCandidates = [combined && { schema: "urn:qwen-r9700:scheduler-telemetry:v2", observed_at_ms: combined.observed_at_ms,
    backend: { scheduler: combined.scheduler, worker: combined.worker } }, read(join(directory, "scheduler-v2.json")), read(join(directory, "scheduler.json"))];
  for (const candidate of schedulerCandidates) {
    try {
      const scheduler = parseSchedulerSample(JSON.stringify(candidate), now);
      result.scheduler = { observed_at_ms: Math.min(candidate.observed_at_ms, Math.floor(scheduler.updated_at * 1000)),
        active: scheduler.requests.filter((row) => row.state === "running").length,
        queued: scheduler.requests.filter((row) => row.state === "queued").length,
        paused: scheduler.requests.filter((row) => row.state === "paused").length,
        ...(chat ? { own: schedulerStatusForChat(scheduler, chat).request ?? null } : {}) };
      break;
    } catch { /* Another authenticated, fresh feed may still be available. */ }
  }
  if (chat) for (const candidate of [combined?.cache, read(join(directory, "cache-residency-v2.json")), read(join(directory, "cache-residency.json"))]) {
    try {
      const sample = parseResidencySample(JSON.stringify(candidate), env.QWEN_RADIANCE_CACHE_ABI, now);
      result.cache = { ...cacheBreakdown(sample, chat, ctx.getContextUsage?.()?.tokens), observed_at_ms: sample.observed_at_ms };
      break;
    } catch { /* Do not promote stale or incompatible cache data. */ }
  }
  for (const candidate of [combined?.temperature, read(join(directory, "sample.json"))]) {
    try { result.temperature = parseTemperatureSample(JSON.stringify(candidate), now); break; } catch { /* Missing temperature does not hide cache/scheduler status. */ }
  }
  return result;
}

async function probeApi(model, fetcher, signal) {
  let url;
  try {
    url = new URL(model?.baseUrl);
    if (!['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) || !['http:', 'https:'].includes(url.protocol) ||
      url.username || url.password || url.search || url.hash || !/^\/v1\/?$/.test(url.pathname)) return { status: "not_probed" };
    const options = { method: "GET", signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(1500)]) : AbortSignal.timeout(1500), redirect: "error" };
    const health = await fetcher(`${url.origin}/health`, options);
    await health.body?.cancel();
    if (!health.ok) return { status: "unhealthy", http_status: health.status };
    const response = await fetcher(`${url.origin}/v1/models`, options);
    if (!response.ok || !response.body) return { status: "models_unavailable" };
    const reader = response.body.getReader(), chunks = [];
    let size = 0;
    try {
      while (true) {
        const item = await reader.read();
        if (item.done) break;
        size += item.value.byteLength;
        if (size > 65_536) { await reader.cancel(); return { status: "models_unavailable" }; }
        chunks.push(Buffer.from(item.value));
      }
    } finally { reader.releaseLock(); }
    const models = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    return { status: Array.isArray(models?.data) && models.data.length === 1 && models.data[0]?.id === model?.id ? "verified" : "model_mismatch" };
  } catch { signal?.throwIfAborted(); return { status: "unreachable" }; }
}

function localIncident(ctx) {
  try {
    // Only selected-branch custom diagnostic records are considered. No message
    // text or traceback is read or copied into a status display.
    const entries = ctx.sessionManager?.getBranch?.();
    if (!Array.isArray(entries)) return;
    for (const entry of [...entries].reverse()) {
      if (entry.type !== "custom" || !["qwen-radiance-backend-error-v1", "qwen-radiance-transport-error-v1"].includes(entry.customType)) continue;
      const data = entry.data;
      const incident = data?.incident ?? data?.backend_report?.incident;
      if (integer(incident?.timestamp)) return { timestamp: incident.timestamp, summary: statusLabel(incident.summary, 220), source: "backend diagnostic" };
      if (entry.customType === "qwen-radiance-transport-error-v1" && integer(data?.timestamp)) return { timestamp: data.failed_at ?? data.timestamp,
        summary: data.causes?.map((item) => statusLabel(item.code, 40)).filter((item) => item !== "unknown").join(" → ") || "Transport request failed",
        source: "this chat's connection" };
    }
  } catch { /* Missing diagnostic metadata is not itself a failed request. */ }
}

export async function readBackendLocalStatus(ctx, { env = process.env, now = Date.now, readConnection = readRadianceConnection,
  telemetry = readBackendTelemetry, fetcher = fetch } = {}) {
  const clock = typeof now === "function" ? now : () => now;
  let connection, connectionInvalid = false;
  try { connection = await readConnection({ path: env.QWEN_RADIANCE_CONNECTION_STATE, now: clock }); } catch { connectionInvalid = true; }
  const api = connection && connection.state !== "ready" || connectionInvalid ? { status: "offline" } : await probeApi(ctx.model, fetcher, ctx.signal);
  const observed = clock();
  return { observed_at_ms: observed, host: env.QWEN_RADIANCE_CACHE_HOST || "ai", connection, connectionInvalid, api,
    endpoint: ctx.model?.baseUrl, expected_model: ctx.model?.id, telemetry: telemetry(ctx, { env, now: observed }), incident: localIncident(ctx) };
}

function resourcesText(resources) {
  if (!resources) return "Resources: unavailable";
  const gpu = resources.gpu, ram = resources.ram, parts = [];
  if (gpu) {
    const degrees = (value) => integer(value) ? `${(value / 1000).toFixed(1)}°C` : "unknown";
    parts.push(`GPU junction ${degrees(gpu.junction_millicelsius)}, edge ${degrees(gpu.edge_millicelsius)}`);
    if (finite(gpu.utilization_percent)) parts.push(`load ${gpu.utilization_percent}%`);
    if (finite(gpu.fan_percent)) parts.push(`fan ${gpu.fan_percent}%`);
    parts.push(`VRAM ${bytes(gpu.vram_used_bytes)} / ${bytes(gpu.vram_total_bytes)}`);
  }
  if (ram) parts.push(`RAM ${bytes(ram.used_bytes)} / ${bytes(ram.total_bytes)} (${bytes(ram.available_bytes)} available)`);
  return `Resources: ${parts.join(" · ") || "unavailable"}`;
}

function reportLines(report) {
  const lines = [];
  if (finite(report.uptime_seconds)) lines.push(`Uptime: ${seconds(report.uptime_seconds)}`);
  if (report.operation) lines.push(`Lifecycle: ${statusLabel(report.operation.action)} ${statusLabel(report.operation.status)} · ${statusLabel(report.operation.stage, 240)}` +
    (finite(report.operation_elapsed_seconds) ? ` · ${seconds(report.operation_elapsed_seconds)} elapsed` : "") +
    (finite(report.operation_age_seconds) ? ` · updated ${seconds(report.operation_age_seconds)} ago` : ""));
  if (Array.isArray(report.issues) && report.issues.length) lines.push(`Host issues: ${report.issues.slice(0, 6).map((issue) => statusLabel(issue, 80).replaceAll("_", " ")).join(" · ")}`);
  const release = report.release, expected = release?.expected, observed = release?.observed;
  lines.push(`Model: ${observed?.model ? statusLabel(observed.model) : report.ready ? "API ready; model identity unavailable" : "not observed"}` +
    (expected?.model ? ` · expected ${statusLabel(expected.model)}` : ""));
  const identity = (value) => [value?.image && `image ${statusLabel(value.image, 100)}`,
    (value?.runtime_abi || value?.runtime_abi_prefix) && `runtime ${statusLabel(value.runtime_abi || value.runtime_abi_prefix, 12)}`,
    value?.data_abi && `cache ${statusLabel(value.data_abi, 12)}`].filter(Boolean).join(" · ");
  lines.push(`Release: ${report.pinned === true ? "matches pinned container configuration" : report.pinned === false ? "configuration mismatch" : "verification unavailable"}` +
    (observed ? ` · ${identity(observed) || "identity unavailable"}` : ""));
  if (expected) lines.push(`Expected release: ${identity(expected) || "identity unavailable"}`);
  lines.push(resourcesText(report.resources));
  const scheduler = report.telemetry?.scheduler;
  if (scheduler?.status === "current") lines.push(`Requests: ${count(scheduler.active_requests)} active · ${count(scheduler.queued_requests)} queued` +
    (integer(scheduler.generating_requests) ? ` · ${count(scheduler.generating_requests)} generating` : "") +
    (finite(scheduler.age_seconds) ? ` · sample ${seconds(scheduler.age_seconds)} old` : ""));
  else if (scheduler?.status) lines.push(`Requests: ${statusLabel(scheduler.status)}${finite(scheduler.age_seconds) ? ` · sample ${seconds(scheduler.age_seconds)} old` : ""}`);
  else if (integer(report.active_requests)) lines.push(`Requests: ${count(report.active_requests)} active · ${count(report.queued_requests)} queued`);
  else lines.push("Requests: unavailable");
  return lines;
}

function colourStatusLines(lines, theme) {
  if (typeof theme?.fg !== "function") return lines.join("\n");
  const colour = (role, value) => theme.fg(role, value);
  const severity = (value, label) => {
    if (/^Last (?:known|observed)\b/.test(label) || /^Request incidents$/.test(label)) return "dim";
    if (/^(?:Expected (?:model|release)|Local endpoint)$/.test(label) || /^(?:host|image|runtime|cache|expected) /.test(value)) return "accent";
    if (/^(?:Latest .*failure|Latest recorded request incident|Control issue|Host issues)$/.test(label) || /\b(?:failed|failure|mismatch|error)\b/i.test(value)) return "error";
    if (/\b(?:offline|stopped|unknown|unavailable|unreachable|stale|pending|queued|paused|starting|stopping|waiting)\b|\bnot (?:ready|current|observed|managed)\b/i.test(value)) return "warning";
    if (/\b(?:idle|generating|running|ready|reachable|authenticated|verified|complete|matches)\b/i.test(value)) return "success";
    if (/\b(?:old|ago|elapsed|attempt|sample|no successful connection recorded)\b/i.test(value)) return "dim";
    return "accent";
  };
  return lines.map((line) => {
    const colon = line.indexOf(":");
    if (colon === -1) return colour("accent", line);
    const label = line.slice(0, colon), value = line.slice(colon + 1);
    const heading = colour("text", `${label}:`);
    const body = value.split(/( · )/).map((part) => part === " · " ? colour("dim", part) : colour(severity(part.trim(), label), part)).join("");
    return (typeof theme.bold === "function" ? theme.bold(heading) : heading) + body;
  }).join("\n");
}

export function backendStatusDetails({ report, local = {}, lastKnown, error, now = Date.now(), theme } = {}) {
  const lines = [], host = statusLabel(local.host || "ai"), connection = local.connection;
  const current = report && report.host_control?.reachable !== false;
  const stopped = current && report.state === "stopped" && report.running === false;
  const backendUnknown = current && report.running === null;
  lines.push(backendUnknown ? "Backend: unknown (host reachable; container inspection unavailable)" :
    current ? `Backend: ${statusLabel(report.state)}${stopped ? " (confirmed stopped)" : ""}` :
      "Backend: unknown (host status unavailable; a connection failure does not confirm a stopped backend)");
  const localStale = integer(local.observed_at_ms) && now - local.observed_at_ms > 3000;
  const connectionStale = integer(connection?.updated_at_unix_ms) && now - connection.updated_at_unix_ms > 3000;
  const tunnel = connection ? connection.state === "ready" ? `${connectionStale || stopped ? "last authenticated" : "authenticated"}${connectionStale ? ` (${age(connection.updated_at_unix_ms, now)})` : ""}` :
    statusLabel(connection.state) : local.connectionInvalid ? "receipt missing or stale" : "not managed by this launcher";
  const api = { verified: "ready; expected model verified", offline: "offline", unreachable: "unreachable", unhealthy: "not ready",
    model_mismatch: "model mismatch", models_unavailable: "model identity unavailable", not_probed: "not probed" }[local.api?.status] || "unknown";
  lines.push(`Connection: host ${host} · control ${current ? "reachable" : "unavailable"} · tunnel ${tunnel} · ` +
    `API ${localStale || stopped ? `last probe ${api} (${age(local.observed_at_ms, now)}; not current)` : api}`);
  if (local.endpoint) {
    try { const endpoint = new URL(local.endpoint); lines.push(`Local endpoint: ${statusLabel(endpoint.origin)}`); } catch { /* Ignore invalid endpoint metadata. */ }
  }
  if (current) lines.push(...reportLines(report));
  else {
    if (local.expected_model) lines.push(`Expected model: ${statusLabel(local.expected_model)}`);
    if (lastKnown?.report) lines.push(`Last known host snapshot: ${age(lastKnown.observed_at_ms, now)}; not current`,
      `Last known backend: ${statusLabel(lastKnown.report.state)}`, ...reportLines(lastKnown.report).map((line) => `Last known ${line[0].toLowerCase()}${line.slice(1)}`));
    else lines.push("Host snapshot: none recorded in this Pi process");
    if (report?.resources) lines.push(resourcesText(report.resources));
  }
  const telemetry = local.telemetry || {}, scheduler = telemetry.scheduler;
  if (scheduler) {
    if (!current) lines.push(`Local scheduler: ${count(scheduler.active)} active · ${count(scheduler.queued)} queued · ${count(scheduler.paused)} paused · ${age(scheduler.observed_at_ms, now)}`);
    const fresh = !stopped && integer(scheduler.observed_at_ms) && now - scheduler.observed_at_ms <= 5000 && scheduler.observed_at_ms <= now + 1000;
    lines.push(`This chat: ${stopped ? "no active request (backend confirmed stopped); last scheduler record" : scheduler.own ? `${fresh ? "observed" : "last observed"} ${statusLabel(scheduler.own.state)} request (submitted)` : "no active request observed; prior submission unknown"}` +
      `${scheduler.own?.state === "queued" ? "; exact queue position unavailable" : ""} · scheduler ${age(scheduler.observed_at_ms, now)}`);
  } else lines.push("This chat: request admission unknown (scheduler telemetry unavailable)");
  const cache = telemetry.cache;
  lines.push(cache ? `${stopped ? "Last observed cache (backend stopped; not live)" : "This chat's cache"}: VRAM ${count(cache.gpu)} · RAM ${count(cache.ram)} · disk backup ${count(cache.diskSaved)} · cold ${count(cache.cold)} tokens · ${age(cache.observed_at_ms, now)}` : "This chat's cache: unavailable");
  if (!current && telemetry.temperature) lines.push(`Local hardware sample: ${resourcesText({ gpu: telemetry.temperature }).replace(/^Resources: /, "")} · ${age(telemetry.temperature.observed_at_ms, now)}`);
  if (connection) {
    lines.push(`Reconnect: ${statusLabel(connection.stage)}${integer(connection.connection_attempt) ? ` · attempt ${connection.connection_attempt}` : ""}` +
      (integer(connection.next_retry_at_unix_ms) ? ` · retry in ${seconds(Math.max(0, connection.next_retry_at_unix_ms - now) / 1000)}` : "") +
      (integer(connection.last_ready_at_unix_ms) ? ` · last verified ${age(connection.last_ready_at_unix_ms, now)}` : " · no successful connection recorded"));
    if (connection.error) lines.push(`Connection issue: ${statusLabel(connection.error, 240)}`);
  } else lines.push("Reconnect: no managed connection worker");
  if (error) lines.push(`Control issue: ${statusLabel(error.message ?? error, 240)}`);
  const incident = local.incident;
  const failure = report?.operation?.status === "failed" ? report.operation : report?.last_failure;
  if (failure) lines.push(`Latest lifecycle failure: ${statusLabel(failure.action)} · ${statusLabel(failure.error || failure.summary, 240)}` +
    (finite(failure.finished_at ?? failure.started_at ?? failure.updated_at) ? ` · ${timestamp(Math.floor((failure.finished_at ?? failure.started_at ?? failure.updated_at) * 1000))}` : ""));
  if (incident) lines.push(`Latest recorded request incident: ${statusLabel(incident.summary, 240)} · ${timestamp(incident.timestamp)} · ${statusLabel(incident.source)} · /backend-error for details`);
  else lines.push("Request incidents: none recorded on this session branch");
  if (stopped) lines.push("Use /backend start, then retry your request when ready.");
  else if (!current || local.api?.status !== "verified") lines.push("Use /backend status to recheck host control and connection.");
  // Sanitize external fields before applying Pi's trusted theme sequences.
  return colourStatusLines(lines.map((line) => statusLabel(line, 700)), theme);
}
