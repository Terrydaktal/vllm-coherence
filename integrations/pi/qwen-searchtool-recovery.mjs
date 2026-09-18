import { createHash } from "node:crypto";
import { realpathSync } from "node:fs";

export const SESSION_ID = "01a02132-9518-735b-805a-7b66a18c023d";
export const PROJECT = "/home/lewis/tasks/searchtool";
export const CONTROL = "qwen-searchtool-recovery-v1";
export const MAX_RECOVERIES = 2;
export const RADIANCE_MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate";

export const AUTHORIZED_CHATS = new Map([
  ["01a02132-9518-735b-805a-7b66a18c023d", {
    name: "searchtool",
    label: "Searchtool",
    project: "/home/lewis/tasks/searchtool",
    statusKey: "searchtool-recovery",
  }],
  ["01a067d9-69cf-761f-8503-6554ccfd7703", {
    name: "money",
    label: "Money",
    project: "/home/lewis/tasks/money",
    statusKey: "money-recovery",
  }],
  ["01a064da-27e6-7108-992a-7aa6ef3195d1", {
    name: "money",
    label: "Money",
    project: "/home/lewis/tasks/money",
    statusKey: "money-recovery",
  }],
  ["01a07934-a41f-7a80-8fb3-3b0e0b5d3ab5", {
    name: "moneychat",
    label: "Moneychat",
    project: "/home/lewis/tasks/moneychat",
    statusKey: "moneychat-recovery",
  }],
]);

export function instructionFor(chat) {
  const name = typeof chat === "string" ? chat : chat?.name ?? "task";
  const task = name === "task" ? "task" : `${name} task`;
  return `Continue the user's already-authorized ${task} from its current state. ` +
    "The previous response stopped before delivering a complete tool action or answer. " +
    "Use the existing tool results; do not repeat completed actions. If a tool is needed, " +
    "emit its complete structured call with all required arguments, without another announcement. " +
    "Do not guess missing commands or arguments from a partial call. If the task is complete, " +
    "give the final answer. Do not perform anything awaiting user approval.";
}

export const INSTRUCTION = instructionFor("searchtool");
const PROTOCOL_ERROR = /(?:Qwen|Provider)[^\n]{0,100}(?:tool[- ]call|tool protocol|structured[- ]outcome|JSON[- ]outcome)[^\n]{0,150}(?:incomplete|invalid|violation|rejected|integrity)|Stream ended without finish_reason/i;
const ACTION = "(?:read(?:ing)?|inspect(?:ing)?|check(?:ing)?|search(?:ing)?|look(?:ing)? (?:up|at|through)|" +
  "fetch(?:ing)?|extract(?:ing)?|run(?:ning)?|execut(?:e|ing)|test(?:ing)?|edit(?:ing)?|writ(?:e|ing)|" +
  "updat(?:e|ing)|implement(?:ing)?|appl(?:y|ying)|verif(?:y|ying)|validat(?:e|ing)|open(?:ing)?|" +
  "build(?:ing)?|fix(?:ing)?|confirm(?:ing)?|quer(?:y|ying)|investigat(?:e|ing)|construct(?:ing)?|" +
  "patch(?:ing)?|add(?:ing)?|remov(?:e|ing)|replac(?:e|ing)|compil(?:e|ing)|install(?:ing)?|" +
  "measur(?:e|ing)|debug(?:ging)?|diagnos(?:e|ing))";
const ANNOUNCEMENT = new RegExp("(?:^|[.!?]\\s+|\\n)\\s*(?:(?:now|next)[,:]?\\s+)?" +
  "(?:i(?:['’]ll| will|(?: am|['’]m)(?: going to)?)|let me|let(?:['’]s| us))\\s+" +
  "(?:(?:now|first|then|also|just)\\s+){0,2}" + ACTION + "\\b[^\\n]{0,450}[:.]$", "i");
const SHORT_ANNOUNCEMENT = /(?:^|[.!?]\s+|\n)\s*(?:(?:now|next)[,:]?\s+)?(?:reading|inspecting|checking|searching|fetching|running|executing|testing|editing|updating|implementing|applying|verifying|validating|opening|building|fixing|querying|patching|compiling)\s+[^\n]{1,450}:$/i;

export function getAuthorizedChat(ctx) {
  if (ctx?.model?.provider !== "qwen-r9700" || ctx.model.api !== "openai-completions") return undefined;
  const sessionId = ctx?.sessionManager?.getSessionId?.();
  if (!sessionId) return undefined;
  const chat = AUTHORIZED_CHATS.get(sessionId);
  if (!chat) return undefined;
  try {
    const cwd = realpathSync(ctx.cwd);
    const sessionCwd = realpathSync(ctx.sessionManager.getCwd());
    if (cwd === chat.project && sessionCwd === chat.project) {
      return chat;
    }
  } catch {
    return undefined;
  }
  return undefined;
}

function inScope(ctx) {
  return Boolean(getRecoveryChat(ctx));
}

export function getRecoveryChat(ctx) {
  const legacy = getAuthorizedChat(ctx);
  if (legacy) return legacy;
  if (ctx?.model?.provider !== "qwen-r9700" || ctx.model.api !== "openai-completions" ||
      ctx.model.id !== RADIANCE_MODEL || !ctx.sessionManager?.getSessionId?.()) return undefined;
  // Every Radiance session gets announcement recovery, including a new session
  // in an existing project. The legacy protocol-repair allowlist stays separate.
  return { name: "task", label: "Qwen", statusKey: "qwen-recovery", announcementsOnly: true };
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  }
  return value;
}
function fingerprint(name, input) {
  return createHash("sha256").update(JSON.stringify([name, canonical(input)])).digest("hex");
}

export function recoveryKind(message) {
  if (message?.role !== "assistant" || !Array.isArray(message.content)) return undefined;
  if (message.stopReason === "error" &&
      (/_incomplete_output$/.test(message.rawStopReason ?? "") || PROTOCOL_ERROR.test(message.errorMessage ?? ""))) {
    return "protocol";
  }
  if (message.stopReason !== "stop" || message.content.some(c => c.type === "toolCall")) return undefined;
  const text = message.content.filter(c => c.type === "text").map(c => c.text).join("\n").trimEnd();
  // A trailing colon is sufficient under the user's continuation policy. The
  // run-end, cancellation and retry-limit checks still apply before sending it.
  if (text.endsWith(":")) return "announcement";
  if (message.rawStopReason === "qwen_json_outcome_final") return undefined;
  if (!text.trim()) return message.content.some(c => c.type === "thinking" && c.thinking?.trim()) ? "missing-outcome" : undefined;
  // Without a trailing colon, retain the conservative action-word heuristic:
  // only inspect a short, unquoted final paragraph without approval conditions.
  const lines = text.split("\n");
  let fence;
  for (const line of lines) {
    const marker = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
    if (marker && !fence) fence = marker[1];
    else if (marker && marker[1][0] === fence?.[0] && marker[1].length >= fence.length && !marker[2].trim()) fence = undefined;
  }
  if (fence) return undefined;
  const tail = text.split(/\n\s*\n/).at(-1).trimEnd();
  if (tail.length > 1_000 || /(?:^|\n)(?: {0,3}(?:>|#{1,6}\s|`{3,}|~{3,})| {4}|\t)/.test(tail) ||
      /[?]|\b(?:if|unless|after you|once you|approve|approval|permission|would|could|should|awaiting|waiting for|when you)\b/i.test(tail)) return undefined;
  // Commands/file names in balanced inline code are normal action announcements.
  // Mask code and quotations so an example cannot supply the action verb.
  const prose = tail.replace(/(`+)[^`\n]*\1/g, "[code]")
    .replace(/"[^"\n]*"|“[^”\n]*”|(?<!\w)'[^'\n]*'(?!\w)/g, "[quote]")
    .replace(/\*\*|__/g, "");
  if (/[`"“”]/.test(prose)) return undefined;
  return ANNOUNCEMENT.test(prose) || SHORT_ANNOUNCEMENT.test(prose) ? "announcement" : undefined;
}

export default function installSearchtoolRecovery(pi) {
  let attempts = 0;
  let recovering = false;
  let pending;
  let userQueued = false;
  const completed = new Set();
  const blockedIds = new Set();

  function reset() {
    attempts = 0; recovering = false; pending = undefined; userQueued = false;
    completed.clear(); blockedIds.clear();
  }
  function restore(ctx) {
    reset();
    const chat = getRecoveryChat(ctx);
    if (!chat) return;
    const calls = new Map();
    for (const entry of ctx.sessionManager.getBranch()) {
      const message = entry.type === "message" ? entry.message : undefined;
      if (message?.role === "user") { reset(); calls.clear(); }
      if (entry.type === "custom_message" && entry.customType === CONTROL) attempts += 1;
      if (message?.role === "assistant") {
        for (const call of message.content ?? []) {
          if (call.type === "toolCall") calls.set(call.id, fingerprint(call.name, call.arguments));
        }
      }
      if (message?.role === "toolResult" && !message.isError && calls.has(message.toolCallId)) {
        completed.add(calls.get(message.toolCallId));
      }
    }
    // Recovery remains enabled, but readiness is not a persistent footer event.
    // Clearing the key also removes the notice left by an older extension load.
    ctx.ui?.setStatus?.(chat.statusKey, undefined);
  }
  pi.on("session_start", (_event, ctx) => restore(ctx));
  pi.on("session_switch", (_event, ctx) => restore(ctx));
  pi.on("session_tree", (_event, ctx) => restore(ctx));
  pi.on("input", (event, ctx) => {
    if (inScope(ctx) && event.source !== "extension" && !ctx.isIdle()) userQueued = true;
  });
  pi.on("message_start", (event, ctx) => {
    if (inScope(ctx) && event.message?.role === "user") reset();
    if (event.message?.role === "assistant") pending = undefined;
  });
  pi.on("message_end", (event, ctx) => {
    const message = event.message;
    const chat = getRecoveryChat(ctx);
    if (!chat || message?.role !== "assistant" || message.provider !== ctx.model.provider ||
        message.model !== ctx.model.id || message.api !== ctx.model.api) return;
    pending = recoveryKind(message);
    if (chat.announcementsOnly && pending !== "announcement") pending = undefined;
    if (recovering) {
      for (const call of message.content ?? []) {
        if (call.type === "toolCall" && completed.has(fingerprint(call.name, call.arguments))) blockedIds.add(call.id);
      }
    }
    if (pending === "protocol") {
      // Avoid Pi's destructive provider-retry path. agent_end queues a NEW
      // continuation instead; the original displayed text remains in history.
      return { message: { ...message,
        content: message.content.filter(c => c.type !== "toolCall"),
        rawStopReason: `${chat.name}_incomplete_output`,
        errorMessage: `${chat.label} structured-output failure; earlier output preserved.`,
      } };
    }
  });
  pi.on("tool_call", (event, ctx) => {
    if (inScope(ctx) && blockedIds.has(event.toolCallId)) {
      return { block: true, reason: "This exact tool call already completed before automatic recovery. Use its existing result; it was not executed again." };
    }
  });
  pi.on("tool_result", (event, ctx) => {
    if (inScope(ctx) && !event.isError) {
      completed.add(fingerprint(event.toolName, event.input));
      recovering = false;
    }
  });
  pi.on("agent_end", (event, ctx) => {
    const reason = pending;
    pending = undefined;
    const chat = getRecoveryChat(ctx);
    if (!chat || !reason || userQueued || ctx.signal?.aborted || ctx.hasPendingMessages()) return;
    // Inspect the settled run's final response, never an intermediate streaming
    // colon or a response superseded by another tool/model step.
    if (Array.isArray(event.messages)) {
      const last = event.messages.findLast(message => message.role === "assistant");
      if (!last || last.model !== ctx.model.id || last.provider !== ctx.model.provider ||
          last.api !== ctx.model.api || recoveryKind(last) !== reason) return;
    }
    if (attempts >= MAX_RECOVERIES) {
      recovering = false;
      ctx.ui?.notify?.(`${chat.label} automatic recovery failed twice. Output retained; stopped to avoid a retry loop.`, "warning");
      return;
    }
    attempts += 1;
    recovering = true;
    ctx.ui?.setWorkingMessage?.(`${chat.label} ${reason === "announcement" ? "continuing omitted tool action" : "recovering incomplete output"} (${attempts}/${MAX_RECOVERIES})`);
    pi.sendMessage({ customType: CONTROL, content: instructionFor(chat), display: false,
      details: { attempt: attempts, reason, sessionId: ctx.sessionManager.getSessionId() } },
    { deliverAs: "followUp", triggerTurn: true });
  });
  pi.on("agent_settled", () => { pending = undefined; recovering = false; });
}
