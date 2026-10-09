// Transcript metadata only: no filesystem, process enumeration, shell commands,
// clock or model calls. The caller supplies the selected, allowed branch.
export const COMPACTION_TASKS_CONTRACT = "qwen-compaction-task-reminders-v1";
const scopes = new Set(["host", "vm", "unknown"]);
const statuses = new Set(["running", "started", "in_progress", "pending", "completed", "complete", "exited", "finished", "cancelled", "canceled", "terminated", "aborted", "failed", "error"]);
const validId = (value) => typeof value === "string" && value.length > 0 && value.length <= 512 && !/[\u0000-\u001f\u007f]/u.test(value);
const quoted = (value) => JSON.stringify(value).replace(/[<>\u2028\u2029]/gu, (character) => `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`);

function limit(value, name, maximum) {
  if (!Number.isSafeInteger(value) || value < 0 || value > maximum) throw new Error(`Invalid ${name}`);
  return value;
}

function handleFrom(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const ids = [value.session_id, value.sessionId].filter((id) => id !== undefined);
  const valid = (id) => validId(id) || Number.isSafeInteger(id) && id >= 0;
  if (ids.some((id) => !valid(id)) || ids.length > 1 && String(ids[0]) !== String(ids[1])) return undefined;
  const processIds = [value.pid, value.process_id, value.processId].filter((pid) => pid !== undefined);
  if (processIds.some((pid) => !Number.isSafeInteger(pid) || pid < 1) ||
      processIds.length > 1 && processIds.some((pid) => pid !== processIds[0])) return undefined;
  if (!ids.length && !processIds.length) return undefined;
  return { ...(ids.length ? { sessionId: ids[0] } : {}), ...(processIds.length ? { pid: processIds[0] } : {}) };
}

function statusFrom(details, isError) {
  if (isError === true) return "result-error";
  const values = [details?.status, details?.process_status, details?.state].filter((value) => value !== undefined);
  if (!values.length || values.some((value) => typeof value !== "string") ||
      values.some((value) => value !== values[0]) || !statuses.has(values[0])) return "unknown";
  return values[0];
}

function scopeFrom(details, fallback) {
  return details?.scope === undefined ? fallback : scopes.has(details.scope) ? details.scope : "unknown";
}

function sourceEntries(entries) {
  if (!Array.isArray(entries)) throw new Error("Allowed task context entries must be an array");
  const seen = new Set();
  for (const entry of entries) {
    if (!entry || !validId(entry.id) || seen.has(entry.id)) throw new Error("Ambiguous allowed task context entry identity");
    seen.add(entry.id);
    if (entry.type === "message" && (!entry.message || typeof entry.message.role !== "string")) throw new Error("Invalid allowed task context message");
  }
}

function clip(text, maxChars) {
  if (text.length <= maxChars) return text;
  let end = maxChars;
  if (end && /[\uD800-\uDBFF]/u.test(text[end - 1]) && /[\uDC00-\uDFFF]/u.test(text[end] ?? "")) end--;
  return text.slice(0, end);
}

/** Render quoted identifiers, never command contents or free-form tool prose. */
export function renderCompactionTasks(report, { maxChars = report?.maxChars ?? 4000 } = {}) {
  limit(maxChars, "task reminder character budget", 20000);
  if (!report || report.contract !== COMPACTION_TASKS_CONTRACT || !Array.isArray(report.records)) throw new Error("Invalid task reminder report");
  if (!maxChars) return "";
  const header = "### Execution and pending-tool reminders\n" +
    "Historical metadata from selected context; no process was checked. Handles are unverified, including after resume. " +
    "An unmatched call proves only that its result was not observed, never that an OS process is running. " +
    "Scopes host/vm/unknown are explicit; do not reuse handles across them.\n";
  const footer = `\nLimits: ${report.limitations.join("; ")}.`;
  if (header.length + footer.length > maxChars) return clip(header + footer, maxChars);
  let text = header;
  let rendered = 0;
  for (const record of report.records) {
    const row = `- ${quoted(record.kind)}; tool ${quoted(record.tool)}; scope ${record.scope}; ` +
      `status ${quoted(record.status)}; availability unverified; ` +
      `${record.namespace ? `namespace ${quoted(record.namespace)}; ` : ""}` +
      `${record.handle ? `handle ${quoted(record.handle)}; ` : ""}` +
      `sources ${quoted(record.sourceEntryIds)}; call ${quoted(record.toolCallId ?? null)}` +
      `${record.supersedesEntryId ? `; supersedes source ${quoted(record.supersedesEntryId)}` : ""}.\n`;
    const omission = `\n[${report.omitted + report.records.length - rendered} reminder(s) omitted by bounds; retrieve exact source entries if needed.]`;
    if (text.length + row.length + footer.length + omission.length > maxChars) break;
    text += row;
    rendered++;
  }
  const omitted = report.omitted + report.records.length - rendered;
  if (omitted) {
    const notice = `\n[${omitted} reminder(s) omitted by bounds; retrieve exact source entries if needed.]`;
    if (text.length + notice.length + footer.length <= maxChars) text += notice;
  }
  if (!report.records.length && !report.omitted) {
    const empty = "No structured execution handles or unmatched calls in the selected context. This is not proof that no background processes exist.\n";
    if (text.length + empty.length + footer.length <= maxChars) text += empty;
  }
  return text + footer;
}

/**
 * Known execution tools must be supplied by the harness, never inferred from
 * assistant prose. Pi 0.84.2's builtin bash does not expose process handles in
 * completed results; adapters may add explicit session_id/pid and status data.
 */
export function captureCompactionTasks({ entries, maxChars = 4000, maxRecords = 32,
  executionTools = ["bash"], executionNamespaces = {}, scope = "unknown", resumed = false } = {}) {
  sourceEntries(entries);
  limit(maxChars, "task reminder character budget", 20000);
  limit(maxRecords, "task reminder record budget", 256);
  if (!scopes.has(scope) || typeof resumed !== "boolean") throw new Error("Invalid task execution scope or resume flag");
  if (!Array.isArray(executionTools) || executionTools.length > 32 || executionTools.some((name) => !validId(name))) throw new Error("Invalid known execution tool names");
  const known = new Set(executionTools);
  if (!executionNamespaces || typeof executionNamespaces !== "object" || Array.isArray(executionNamespaces) ||
      Object.entries(executionNamespaces).some(([name, namespace]) => !known.has(name) || !validId(namespace))) throw new Error("Invalid execution handle namespaces");
  const pending = new Map(), handles = new Map(), seenCalls = new Set();
  let ignoredAbortedCalls = 0, unsupportedHandles = 0, orphanResults = 0, sequence = 0;
  for (const entry of entries) {
    if (entry.type !== "message") continue;
    const message = entry.message;
    if (message.role === "assistant") {
      const calls = (Array.isArray(message.content) ? message.content : []).filter((block) => block?.type === "toolCall");
      if (["aborted", "error"].includes(message.stopReason)) { ignoredAbortedCalls += calls.length; continue; }
      for (const call of calls) {
        if (!validId(call.id) || !validId(call.name) || seenCalls.has(call.id)) throw new Error("Ambiguous tool-call identity in allowed task context");
        seenCalls.add(call.id);
        pending.set(call.id, { call, entryId: entry.id, order: sequence++ });
      }
    } else if (message.role === "toolResult") {
      const call = pending.get(message.toolCallId);
      // Do not let another tool's result clear this call's pending status.
      if (call && message.toolName && message.toolName !== call.call.name) { unsupportedHandles++; continue; }
      if (call) pending.delete(message.toolCallId);
      else orphanResults++;
      const tool = call?.call.name ?? message.toolName;
      if (!known.has(tool)) continue;
      const details = message.details;
      const resultHandle = handleFrom(details);
      const resultHasHandle = details && ["session_id", "sessionId", "pid", "process_id", "processId"].some((key) => details[key] !== undefined);
      // Present but invalid result metadata is not permission to invent a
      // successful association with the requested argument handle instead.
      if (resultHasHandle && !resultHandle) { unsupportedHandles++; continue; }
      const argumentHandle = handleFrom(call?.call.arguments);
      const handle = resultHandle ?? argumentHandle;
      if (!handle) {
        continue;
      }
      const recordScope = scopeFrom(details, scope);
      const namespace = Object.hasOwn(executionNamespaces, tool) ? executionNamespaces[tool] : tool;
      const key = quoted([recordScope, handle.sessionId !== undefined ? ["session", namespace, String(handle.sessionId)] : ["pid", handle.pid]]);
      const prior = handles.get(key);
      handles.set(key, {
        kind: "execution-handle", tool, namespace, scope: recordScope, status: statusFrom(details, message.isError),
        handle, handleSource: resultHandle ? "tool-result-details" : "tool-call-arguments",
        availability: "unverified", verifiedRunning: false, historical: true, resumed,
        toolCallId: validId(message.toolCallId) ? message.toolCallId : undefined,
        sourceEntryIds: [...new Set([call?.entryId, entry.id].filter(Boolean))],
        ...(prior ? { supersedesEntryId: prior.sourceEntryIds.at(-1) } : {}), order: sequence++,
      });
    }
  }
  const all = [...handles.values(), ...[...pending.values()].map(({ call, entryId, order }) => ({
    kind: "unobserved-tool-result", tool: call.name, scope, status: "result-unobserved",
    availability: "unverified", verifiedRunning: false, historical: true, resumed,
    toolCallId: call.id, sourceEntryIds: [entryId], order,
  }))].sort((left, right) => right.order - left.order);
  const records = all.slice(0, maxRecords).map(({ order, ...record }) => record);
  const limitations = ["selected branch only", "no OS process or handle availability check", "tool prose and command contents not parsed",
    "builtin Pi bash results normally expose no process/session handle", "result-error is a tool failure, not a known OS process exit",
    "superseded handle observations use the last supplied chronological result"];
  if (resumed) limitations.push("resumed handles may be stale");
  if (ignoredAbortedCalls) limitations.push(`${ignoredAbortedCalls} aborted/error assistant call(s) ignored`);
  if (unsupportedHandles) limitations.push(`${unsupportedHandles} malformed/conflicting handle observation(s) ignored`);
  if (orphanResults) limitations.push(`${orphanResults} result(s) lacked a selected matching call`);
  const report = { contract: COMPACTION_TASKS_CONTRACT, scope, resumed, records,
    omitted: all.length - records.length, maxChars, limitations };
  return { ...report, text: renderCompactionTasks(report) };
}
