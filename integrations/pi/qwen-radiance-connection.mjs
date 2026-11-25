import { constants } from "node:fs";
import { open, readFile } from "node:fs/promises";
import { isAbsolute } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

export const CONNECTION_SCHEMA = "coherence.pi.backend-connection.v1";
export const CONNECTION_STATUS_KEY = "qwen-backend-connection";
const MAX_STATE_BYTES = 8192;
const MAX_STATE_AGE_MS = 3000;
const OFFLINE = "Backend offline. Check /backend status or use /backend start, then retry. Your saved chat remains available.";
const statePath = () => process.env.QWEN_RADIANCE_CONNECTION_STATE;
const unavailable = (reason, detail, retryable = false) => Object.assign(
  new Error(`Backend connection unavailable: ${detail}. Check /backend status, then retry. Your saved chat remains available.`),
  { code: "QWEN_BACKEND_CONNECTION_UNAVAILABLE", reason, retryable },
);
const offline = () => Object.assign(new Error(OFFLINE), { code: "QWEN_BACKEND_OFFLINE", reason: "backend_stopped", retryable: false });
const invalid = (reason = "unsafe_receipt", detail = "supervisor status could not be authenticated") => unavailable(reason, detail);
const label = (value) => typeof value === "string" && value.length <= 160 && !/[\u0000-\u001f\u007f]/.test(value);

// The launcher publishes only connection metadata. Never read a transcript or
// accept a state left behind by a different process after a supervisor exits.
export async function readRadianceConnection({ path = statePath(), now = Date.now, openFn = open } = {}) {
  if (!path) return undefined;
  if (!isAbsolute(path)) throw invalid();
  let file;
  try {
    file = await openFn(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    const info = await file.stat();
    if (!info.isFile()) throw invalid("receipt_type", "supervisor status is not a regular file");
    if (info.uid !== process.getuid()) throw invalid("receipt_owner", "supervisor status has the wrong owner");
    if ((info.mode & 0o7777) !== 0o600) throw invalid("receipt_permissions", "supervisor status permissions are not private");
    if (info.nlink === 0) {
      // The publisher atomically renames a fresh receipt over this pathname.
      // A descriptor opened just before that rename still holds the valid old
      // inode, now unlinked. Reopen the current receipt rather than trusting the
      // retired inode or treating normal publication as an authentication fault.
      throw unavailable("receipt_replaced", "supervisor status was replaced while being read", true);
    }
    if (info.nlink !== 1) throw invalid("receipt_hardlink", "supervisor status has unexpected hard links");
    if (info.size < 1 || info.size > MAX_STATE_BYTES) throw invalid("receipt_size", "supervisor status has an invalid size");
    const bytes = Buffer.alloc(MAX_STATE_BYTES + 1);
    const { bytesRead } = await file.read(bytes, 0, bytes.length, 0);
    if (bytesRead !== info.size || bytesRead > MAX_STATE_BYTES) throw invalid("receipt_changed", "supervisor status changed while being read");
    const record = JSON.parse(bytes.subarray(0, bytesRead).toString("utf8"));
    if (record?.schema !== CONNECTION_SCHEMA || !["offline", "checking", "ready"].includes(record.state) ||
      !Number.isInteger(record.local_port) || record.local_port < 1 || record.local_port > 65535 ||
      !Number.isSafeInteger(record.supervisor_pid) || record.supervisor_pid < 2 ||
      !/^[1-9][0-9]*$/.test(record.supervisor_start_ticks ?? "") ||
      !Number.isSafeInteger(record.updated_at_unix_ms) || !label(record.stage) ||
      (record.error !== undefined && !label(record.error))) throw invalid("receipt_schema", "supervisor status fields are invalid");
    const processStat = await readFile(`/proc/${record.supervisor_pid}/stat`, "utf8");
    const fields = processStat.slice(processStat.lastIndexOf(")") + 2).trim().split(/\s+/);
    if (["Z", "X"].includes(fields[0]) || fields[19] !== record.supervisor_start_ticks) {
      throw invalid("supervisor_identity", "supervisor process identity does not match its status");
    }
    // File I/O may wait in Node's worker pool. Validate against the clock after
    // the asynchronous reads, rather than a timestamp captured before them.
    const observedNow = typeof now === "function" ? now() : now;
    if (record.updated_at_unix_ms - observedNow > 1000) throw invalid("receipt_future", "supervisor heartbeat is dated in the future");
    if (observedNow - record.updated_at_unix_ms > MAX_STATE_AGE_MS) {
      throw unavailable("receipt_stale", "supervisor heartbeat is stale", true);
    }
    return record;
  } catch (error) {
    if (typeof error?.code === "string" && error.code.startsWith("QWEN_BACKEND_")) throw error;
    if (!file && ["ENOENT", "EAGAIN", "EINTR", "EBUSY"].includes(error?.code)) {
      throw unavailable("receipt_missing", "supervisor status is temporarily unavailable", true);
    }
    if (error instanceof SyntaxError) throw invalid("receipt_json", "supervisor status is not valid JSON");
    if (file && error?.code === "ENOENT") throw invalid("supervisor_identity", "supervisor process no longer exists");
    throw invalid();
  } finally {
    await file?.close();
  }
}

async function boundedModels(response) {
  if (!response.ok || !response.body) throw unavailable("models_unavailable", "model identity probe failed", response.status >= 500 || response.status === 429);
  const reader = response.body.getReader();
  let length = 0;
  const chunks = [];
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > 65536) {
        await reader.cancel();
        throw unavailable("model_mismatch", "model identity response exceeded its size limit");
      }
      chunks.push(Buffer.from(value));
    }
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } finally {
    reader.releaseLock();
  }
}

const sleep = (ms, signal) => delay(ms, undefined, { signal });
const sameSupervisor = (left, right) => left.local_port === right.local_port && left.supervisor_pid === right.supervisor_pid &&
  left.supervisor_start_ticks === right.supervisor_start_ticks;

function requireReady(record) {
  if (record.state === "ready") return;
  if (record.state === "offline" && record.failure_kind === "backend_stopped") throw offline();
  if (record.state === "checking" || record.failure_kind === "health_failed") {
    throw unavailable("readiness_pending", "waiting for the connection worker to verify readiness", true);
  }
  throw unavailable("connection_not_ready", "the connection worker has not authenticated a ready backend");
}

export async function requireRadianceConnection(model, { signal, fetcher = fetch, path = statePath(), now = Date.now,
  maxWaitMs = 6000, retryDelayMs = 150, sleepFn = sleep, readConnection = readRadianceConnection } = {}) {
  signal?.throwIfAborted();
  if (!path) return;
  const budget = AbortSignal.timeout(Math.max(1, Math.ceil(maxWaitMs)));
  const waitSignal = signal ? AbortSignal.any([signal, budget]) : budget;
  const deadline = now() + maxWaitMs;
  let identity, lastError;
  while (true) {
    signal?.throwIfAborted();
    if (budget.aborted) throw lastError ?? unavailable("probe_failed", "readiness check timed out");
    let stage = "receipt";
    try {
      const record = await readConnection({ path, now });
      if (identity && !sameSupervisor(identity, record)) throw unavailable("supervisor_changed", "connection ownership changed during readiness checks");
      identity ??= record;
      const url = new URL(model.baseUrl);
      if (url.protocol !== "http:" || url.hostname !== "127.0.0.1" || Number(url.port) !== record.local_port ||
        !/^\/v1\/?$/.test(url.pathname) || url.username || url.password || url.search || url.hash || !model.id) {
        throw unavailable("mismatched_endpoint", "model endpoint does not match the authenticated local tunnel");
      }
      requireReady(record);
      const requestSignal = AbortSignal.any([waitSignal, AbortSignal.timeout(2000)]);
      const options = { method: "GET", signal: requestSignal, redirect: "error" };
      stage = "health";
      const response = await fetcher(`${url.origin}/health`, options);
      await response.body?.cancel();
      if (!response.ok) throw unavailable("api_unhealthy", `health probe returned HTTP ${response.status}`, response.status >= 500 || response.status === 429);
      stage = "models";
      const models = await boundedModels(await fetcher(`${url.origin}/v1/models`, options));
      if (!Array.isArray(models?.data) || models.data.length !== 1 || models.data[0]?.id !== model.id) {
        throw unavailable("model_mismatch", "the expected backend model was not returned");
      }
      const observedWindow = models.data[0].max_model_len;
      if (observedWindow != null && Number.isSafeInteger(model.contextWindow) && observedWindow !== model.contextWindow) {
        throw unavailable("model_context_mismatch", "the backend context window changed; reopen Pi to load the pinned model profile");
      }
      // A stop or supervisor replacement during probes invalidates admission.
      stage = "receipt";
      const current = await readConnection({ path, now });
      if (!sameSupervisor(record, current)) throw unavailable("supervisor_changed", "connection ownership changed during readiness checks");
      requireReady(current);
      signal?.throwIfAborted();
      return;
    } catch (error) {
      signal?.throwIfAborted();
      const failure = typeof error?.code === "string" && error.code.startsWith("QWEN_BACKEND_") ? error :
        unavailable(stage === "models" && error instanceof SyntaxError ? "model_mismatch" : "probe_failed",
          `${stage} check ${error?.name === "TimeoutError" ? "timed out" : "failed"}`,
          stage !== "receipt" && !(error instanceof SyntaxError));
      lastError = failure;
      const remaining = deadline - now();
      if (!failure.retryable || budget.aborted || remaining <= 0) throw failure;
      try { await sleepFn(Math.min(retryDelayMs, remaining), waitSignal); }
      catch { signal?.throwIfAborted(); throw failure; }
    }
  }
}

export function guardRadianceStream(streamSimple, gate = requireRadianceConnection) {
  return (model, context, options = {}) => streamSimple(model, context, {
    ...options,
    onPayload: async (payload, selected) => {
      await gate(model, { signal: options.signal, fetcher: options.fetch ?? fetch });
      return options.onPayload ? options.onPayload(payload, selected) : payload;
    },
  });
}

export function installRadianceConnection(pi, { streamSimpleOpenAICompletions,
  read = readRadianceConnection, setIntervalFn = setInterval, clearIntervalFn = clearInterval, now = Date.now } = {}) {
  if (!statePath()) return;
  pi.registerProvider("qwen-r9700", { api: "openai-completions", streamSimple: guardRadianceStream(streamSimpleOpenAICompletions) });
  let timer, revision = 0, lastReadyAt, lastStatus = Symbol("unobserved");
  const show = (ctx, value) => {
    if (value !== lastStatus) ctx.ui?.setStatus?.(CONNECTION_STATUS_KEY, value);
    lastStatus = value;
  };
  const stop = () => { revision++; if (timer !== undefined) clearIntervalFn(timer); timer = undefined; };
  const start = (_event, ctx) => {
    stop();
    const active = revision;
    if (typeof lastStatus === "symbol") show(ctx, "Checking backend connection");
    let inFlight = false;
    const refresh = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        let record;
        try { record = await read(); } catch { /* Unknown status is not evidence of a stopped backend. */ }
        if (active !== revision) return;
        if (record?.state === "ready") { lastReadyAt = now(); show(ctx, undefined); }
        else if (record?.state === "offline" && record.failure_kind === "backend_stopped") show(ctx, "Backend offline");
        else if (lastReadyAt !== undefined && now() - lastReadyAt < 1500) { /* Ignore one short observation gap. */ }
        else show(ctx, record?.state === "checking" || record?.failure_kind === "health_failed" ? "Checking backend connection" : "Backend status unavailable");
      } finally { inFlight = false; }
    };
    void refresh();
    timer = setIntervalFn(() => { void refresh(); }, 500);
    timer?.unref?.();
  };
  pi.on("session_start", start);
  pi.on("session_switch", start);
  pi.on("session_shutdown", stop);
}
