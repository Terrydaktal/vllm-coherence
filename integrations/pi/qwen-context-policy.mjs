import { createHash } from "node:crypto";

export const CONTEXT_POLICY_ENTRY = "qwen-context-selection-v1";
export const THINKING_PURGE_ENTRY = "qwen-radiance-thinking-purge-v1";
const RADIANCE_MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate";
const identified = new WeakMap();
const failures = new Map();
const sessionKey = (ctx) => ctx.sessionManager.getSessionId?.() ?? ctx.sessionManager.getSessionFile?.() ?? ctx.sessionManager;
const validId = (id) => typeof id === "string" && id.length > 0;
const policyFailure = (message) => Object.assign(new Error(message), { code: "QWEN_CONTEXT_FILTER_FAILURE" });
export const hasThinking = (message) => message?.role === "assistant" && Array.isArray(message.content) &&
  message.content.some((block) => block?.type === "thinking");
// The provider drops failed attempts and strips tool calls from retained Qwen
// interruptions. These attempts cannot require an execution result or own
// another attempt's result in the selected provider context.
const discardedAssistant = (message) => message?.role === "assistant" &&
  ["aborted", "error"].includes(message.stopReason);

export function contextPolicy(ctx) {
  const excluded = new Set(), thinking = new Set();
  let active = false, preserveFutureThinking = false;
  for (const entry of ctx.sessionManager.getBranch()) {
    if (entry.type !== "custom") continue;
    if (entry.customType === THINKING_PURGE_ENTRY && ctx.model?.id === RADIANCE_MODEL) {
      const data = entry.data;
      if (data?.version !== 1 || data.preserveFutureThinking !== true || !Array.isArray(data.entryIds) ||
          !data.entryIds.every(validId)) throw new Error("Invalid saved thinking purge; refusing to restore excluded thinking");
      active = preserveFutureThinking = true;
      for (const id of data.entryIds) thinking.add(id);
    } else if (entry.customType === CONTEXT_POLICY_ENTRY) {
      const data = entry.data;
      if (data?.version !== 1 || !Array.isArray(data.changes) || typeof data.preserveFutureThinking !== "boolean" ||
          (data.undoOf !== undefined && !validId(data.undoOf)) ||
          !data.changes.every((change) => validId(change?.entryId) &&
            ["message", "thinking"].includes(change.part) && typeof change.excluded === "boolean")) {
        throw new Error("Invalid saved context selection; request blocked to protect excluded context");
      }
      active = true;
      preserveFutureThinking ||= data.preserveFutureThinking;
      for (const change of data.changes) {
        const target = change.part === "message" ? excluded : thinking;
        if (change.excluded) target.add(change.entryId); else target.delete(change.entryId);
      }
    }
  }
  if (active && typeof ctx.sessionManager.buildContextEntries !== "function") {
    throw new Error("Context selection needs the patched Pi session-context API");
  }
  return { active, excluded, thinking, preserveFutureThinking };
}

export function clonePolicy(policy) {
  return { ...policy, excluded: new Set(policy.excluded), thinking: new Set(policy.thinking) };
}

// A decision always names a session entry, never a timestamp or matching text.
// Pi clones context messages; occurrence queues recover the entry IDs. Omitting
// thinking from the key also recognizes an already-purged assistant message.
function messageKey(message) {
  return createHash("sha256").update(JSON.stringify([
    message.role, message.timestamp, message.api, message.provider, message.model,
    message.stopReason, message.toolCallId, message.toolName,
    message.command, message.output, message.exitCode, message.excludeFromContext,
    Array.isArray(message.content) ? message.content.filter((block) => block?.type !== "thinking") : message.content,
  ])).digest("hex");
}
const messageIdentity = (message) => JSON.stringify([message.role, message.timestamp, message.api,
  message.provider, message.model, message.stopReason, message.toolCallId, message.toolName]);

export function contextEntries(ctx) {
  if (typeof ctx.sessionManager.buildContextEntries !== "function") {
    throw new Error("Context selection needs the patched Pi session-context API");
  }
  const entries = ctx.sessionManager.buildContextEntries().filter((entry) => entry.type === "message");
  if (entries.some((entry) => !entry.message || typeof entry.message.role !== "string")) {
    throw new Error("Invalid message metadata; context retained");
  }
  return entries;
}

// All calls in one assistant message and their results form one exclusion unit.
// Scope reused call IDs to their most recent preceding assistant, rather than
// grouping unrelated turns by tool name or globally by call ID.
export function toolDependencies(entries) {
  const groups = new Map(), owners = new Map(), orphans = new Set();
  for (const entry of entries) {
    if (!validId(entry.id)) throw new Error("Cannot identify context messages; context retained");
    const message = entry.message;
    if (discardedAssistant(message)) continue;
    if (message?.role === "assistant") {
      const calls = Array.isArray(message.content) ? message.content.filter((b) => b?.type === "toolCall") : [];
      if (!calls.length) continue;
      const group = new Set([entry.id]);
      const callIds = new Set();
      for (const call of calls) {
        if (!validId(call.id) || callIds.has(call.id)) throw new Error("Ambiguous tool-call identity; context retained");
        callIds.add(call.id);
        owners.set(call.id, group);
      }
      groups.set(entry.id, group);
    } else if (message?.role === "toolResult") {
      const group = owners.get(message.toolCallId);
      if (!group) { orphans.add(entry.id); continue; }
      group.add(entry.id);
      groups.set(entry.id, group);
    }
  }
  return { groups, orphans };
}

export function effectiveExclusions(entries, policy) {
  const result = new Set(policy.excluded);
  for (const group of new Set(toolDependencies(entries).groups.values())) {
    if ([...group].some((id) => result.has(id))) for (const id of group) result.add(id);
  }
  return result;
}

export function changeSelection(entries, policy, ids, part, excluded) {
  const next = clonePolicy(policy), available = new Map(entries.map((entry) => [entry.id, entry]));
  const dependencies = toolDependencies(entries);
  for (const id of ids) {
    const entry = available.get(id);
    if (!entry) throw new Error("The selected message is no longer in the active context");
    if (part === "thinking") {
      if (!hasThinking(entry.message)) continue;
      if (excluded) next.thinking.add(id); else next.thinking.delete(id);
      next.preserveFutureThinking = true;
    } else if (part === "message") {
      for (const member of dependencies.groups.get(id) ?? [id]) {
        if (excluded) next.excluded.add(member); else next.excluded.delete(member);
      }
    } else throw new Error("Unknown context selection action");
  }
  next.active = true;
  return next;
}

export function selectionChanges(before, after) {
  const changes = [];
  for (const [part, key] of [["message", "excluded"], ["thinking", "thinking"]]) {
    for (const id of new Set([...before[key], ...after[key]])) {
      if (before[key].has(id) !== after[key].has(id)) {
        changes.push({ entryId: id, part, excluded: after[key].has(id) });
      }
    }
  }
  return changes;
}

export function filterContext(messages, ctx, override) {
  const policy = override ?? contextPolicy(ctx);
  if (!policy.active) return messages;
  const entries = contextEntries(ctx), excluded = effectiveExclusions(entries, policy), queues = new Map(), selectedIdentities = new Set();
  const available = new Set(entries.map((entry) => entry.id));
  const roles = new Set(entries.map((entry) => entry.message.role));
  for (const entry of entries) {
    const key = messageKey(entry.message);
    if (!queues.has(key)) queues.set(key, []);
    queues.get(key).push(entry.id);
    if (excluded.has(entry.id) || policy.thinking.has(entry.id)) selectedIdentities.add(messageIdentity(entry.message));
  }
  const offsets = new Map(), result = [];
  for (const message of messages) {
    if (!roles.has(message.role) && message.role !== "toolResult") { result.push(message); continue; }
    const key = messageKey(message), offset = offsets.get(key) ?? 0;
    offsets.set(key, offset + 1);
    const known = identified.get(message);
    const id = available.has(known) ? known : queues.get(key)?.[offset];
    if (!id) {
      if (message.role === "toolResult") {
        throw new Error("Unrecorded tool result in selected context; request blocked instead of inventing a result");
      }
      if (selectedIdentities.has(messageIdentity(message))) {
        throw new Error("Cannot safely associate a changed selected message with its entry; request blocked");
      }
      result.push(message); continue; // New, not-yet-saved input is never implicitly excluded.
    }
    if (excluded.has(id)) continue;
    const output = policy.thinking.has(id) && hasThinking(message)
      ? { ...message, content: message.content.filter((block) => block?.type !== "thinking") } : message;
    identified.set(output, id);
    result.push(output);
  }
  assertToolPairs(result);
  return result;
}

function assertToolPairs(messages) {
  const pending = new Set();
  for (const message of messages) {
    if (discardedAssistant(message)) continue;
    if (message.role === "assistant") {
      for (const block of Array.isArray(message.content) ? message.content : []) {
        if (block?.type === "toolCall") {
          if (pending.has(block.id)) throw new Error("Ambiguous unfinished tool calls in selected context; request blocked");
          pending.add(block.id);
        }
      }
    } else if (message.role === "toolResult") {
      if (!pending.has(message.toolCallId)) throw new Error("Context selection would leave an orphan tool result; request blocked");
      pending.delete(message.toolCallId);
    }
  }
  if (pending.size) throw new Error("Context selection would retain a tool call with missing results; exclude the whole message instead");
}

// Pi catches context-hook errors and otherwise continues with the old messages.
// Remember a failure and reject in before_provider_request, where errors propagate.
export function filteredRequestContext(messages, ctx) {
  const key = sessionKey(ctx);
  try {
    const result = filterContext(messages, ctx);
    failures.delete(key);
    return result;
  } catch (error) {
    failures.set(key, error.message);
    throw policyFailure(error.message);
  }
}

export function assertContextReady(ctx) {
  try { contextPolicy(ctx); } // Validate saved metadata even if the context hook did not run.
  catch (error) { throw policyFailure(error.message); }
  if (failures.has(sessionKey(ctx))) throw policyFailure(failures.get(sessionKey(ctx)));
}

export function clearContextFailure(ctx) { failures.delete(sessionKey(ctx)); }

export function contextRows(ctx) {
  let turn = 0;
  return contextEntries(ctx).map((entry, index) => {
    if (entry.message?.role === "user") turn++;
    return { ...entry, index, turn };
  });
}

export function estimateMessageTokens(message) {
  if (message.role === "bashExecution") return Math.ceil(((message.command ?? "").length + (message.output ?? "").length) / 4) + 8;
  if (typeof message.summary === "string") return Math.ceil(message.summary.length / 4) + 8;
  if (typeof message.content === "string") return Math.ceil(message.content.length / 4) + 8;
  let chars = 0;
  for (const block of message.content ?? []) {
    if (block?.type === "text") chars += (block.text ?? "").length;
    else if (block?.type === "thinking") chars += (block.thinking ?? "").length;
    else if (block?.type === "toolCall") chars += (block.name ?? "").length + JSON.stringify(block.arguments ?? {}).length;
    else if (block?.type === "image") chars += 4800;
  }
  return Math.ceil(chars / 4) + 8;
}
