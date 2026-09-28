import { createHash } from "node:crypto";
import { getCompactionProgress } from "./qwen-radiance-compaction-progress.mjs";

export const THINKING_PURGE_ENTRY = "qwen-radiance-thinking-purge-v1";
const MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate";
const applies = (ctx) => ctx.model?.id === MODEL;
const thinkingBlocks = (message) => message?.role === "assistant" && Array.isArray(message.content)
  ? message.content.filter((block) => block?.type === "thinking") : [];

export function thinkingPurgePolicy(ctx) {
  const entryIds = new Set();
  let active = false;
  if (!applies(ctx)) return { active, entryIds };
  for (const entry of ctx.sessionManager.getBranch()) {
    if (entry.type !== "custom" || entry.customType !== THINKING_PURGE_ENTRY) continue;
    const data = entry.data;
    if (data?.version !== 1 || data.preserveFutureThinking !== true || !Array.isArray(data.entryIds) ||
        !data.entryIds.every((id) => typeof id === "string" && id.length > 0)) {
      throw new Error("Invalid saved thinking purge; refusing to restore excluded thinking");
    }
    active = true;
    for (const id of data.entryIds) entryIds.add(id);
  }
  if (active && typeof ctx.sessionManager.buildContextEntries !== "function") {
    throw new Error("Thinking purge needs the patched Pi session-context API");
  }
  return { active, entryIds };
}

// Pi clones messages before the context hook, so object identity is unavailable.
// Match against the compaction-aware entry list, in occurrence order. Excluding
// thinking from the key makes filtering idempotent. Queues distinguish identical
// assistant messages (even with identical timestamps) on opposite sides of a
// purge. The persisted decision uses entry IDs, never a wall-clock cutoff.
function messageKey(message) {
  return createHash("sha256").update(JSON.stringify([
    message.timestamp, message.api, message.provider, message.model, message.stopReason,
    Array.isArray(message.content) ? message.content.filter((block) => block?.type !== "thinking") : message.content,
  ])).digest("hex");
}

export function purgeContextThinking(messages, ctx) {
  const policy = thinkingPurgePolicy(ctx);
  if (!policy.active || policy.entryIds.size === 0) return messages;
  const queues = new Map();
  for (const entry of ctx.sessionManager.buildContextEntries()) {
    if (entry.type !== "message" || entry.message?.role !== "assistant") continue;
    const key = messageKey(entry.message);
    if (!queues.has(key)) queues.set(key, { ids: [], offset: 0 });
    queues.get(key).ids.push(entry.id);
  }
  return messages.map((message) => {
    if (message.role !== "assistant") return message;
    const queue = queues.get(messageKey(message));
    const entryId = queue?.ids[queue.offset++];
    if (!policy.entryIds.has(entryId) || thinkingBlocks(message).length === 0) return message;
    return { ...message, content: message.content.filter((block) => block?.type !== "thinking") };
  });
}

export function preserveFutureThinking(payload, ctx) {
  if (!thinkingPurgePolicy(ctx).active) return payload;
  return { ...payload, chat_template_kwargs: {
    ...payload.chat_template_kwargs, preserve_thinking: true,
    // Qwen's alias takes precedence if it was explicitly configured.
    ...(Object.hasOwn(payload.chat_template_kwargs ?? {}, "preserve_reasoning") ? { preserve_reasoning: true } : {}),
  } };
}

export function installThinkingPurge(pi, { Text } = {}) {
  if (Text) pi.registerEntryRenderer(THINKING_PURGE_ENTRY, (entry, _options, theme) =>
    new Text(theme.fg("dim", `Thinking purged from ${entry.data?.entryIds?.length ?? 0} earlier messages; future thinking retained.`), 0, 0));
  pi.on("context", (event, ctx) => ({ messages: purgeContextThinking(event.messages, ctx) }));
  pi.on("before_provider_request", (event, ctx) => preserveFutureThinking(event.payload, ctx));
  pi.registerCommand("purge-thinking", {
    description: "Exclude existing thinking from this chat's context; retain future thinking. /purge-thinking status shows the saved policy.",
    handler: async (args, ctx) => {
      if (!applies(ctx)) {
        ctx.ui.notify("/purge-thinking is available on the Radiance backend.", "warning");
        return;
      }
      if (!["", "status"].includes(args.trim())) {
        ctx.ui.notify("Usage: /purge-thinking [status]", "warning");
        return;
      }
      const policy = thinkingPurgePolicy(ctx);
      if (args.trim() === "status") {
        ctx.ui.notify(policy.active
          ? `Thinking purge is active for this branch: ${policy.entryIds.size} earlier messages excluded; future thinking retained.`
          : "No thinking purge is saved for this branch; the provider's preserve_thinking setting applies.", "info");
        return;
      }
      if (!ctx.isIdle() || getCompactionProgress(ctx)?.snapshot().state === "running") {
        ctx.ui.notify("Stop generation or compaction with Escape before /purge-thinking.", "warning");
        return;
      }
      if (typeof ctx.sessionManager.buildContextEntries !== "function") {
        throw new Error("Thinking purge needs the patched Pi session-context API");
      }
      const entries = ctx.sessionManager.buildContextEntries().filter((entry) =>
        entry.type === "message" && !policy.entryIds.has(entry.id) && thinkingBlocks(entry.message).length > 0);
      if (!entries.every((entry) => typeof entry.id === "string" && entry.id.length > 0)) {
        throw new Error("Cannot identify messages to purge; context retained");
      }
      if (entries.length || !policy.active) {
        // This metadata entry is persisted by Pi but excluded from model context.
        // Original message bodies remain untouched in the transcript.
        pi.appendEntry(THINKING_PURGE_ENTRY, { version: 1,
          entryIds: entries.map((entry) => entry.id), preserveFutureThinking: true });
      }
      ctx.ui.notify(entries.length
        ? `Excluded thinking from ${entries.length} messages. Future thinking will be retained. Prose and tools are unchanged; the next request may need to rebuild the changed cache prefix.`
        : "No additional thinking to purge. Future thinking will be retained.", "info");
    },
  });
}
