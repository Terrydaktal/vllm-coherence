import { randomBytes } from "node:crypto";
import { radianceBridgeRequest, radianceBridgeUrl } from "./qwen-radiance-bridge.mjs";

export const PRIORITY_ENTRY = "qwen-radiance-priority-v1";
const MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate";
const descriptions = [
  "normal scheduling",
  "take over at a response boundary; keep the GPU through tools until this answer finishes",
  "interrupt a lower-priority chat at the next safe GPU step; keep the GPU until this answer finishes",
];
const nonce = () => randomBytes(16).toString("hex");

export function savedPriority(ctx, chatId) {
  const entry = ctx.sessionManager.getEntries().filter((e) => e.type === "custom" && e.customType === PRIORITY_ENTRY).at(-1);
  if (chatId !== undefined && entry?.data?.chat_id !== chatId) return 0;
  return [0, 1, 2].includes(entry?.data?.priority) ? entry.data.priority : 0;
}

export async function sendPriority(ctx, chat, update, fetcher = fetch) {
  let result;
  if (radianceBridgeUrl()) {
    result = await radianceBridgeRequest({ operation: "priority", chat, update }, { timeout: 6500, fetcher });
  } else {
    const base = new URL(ctx.model.baseUrl);
    if (base.protocol !== "http:" || base.hostname !== "127.0.0.1" || !/^\/v1\/?$/.test(base.pathname))
      throw new Error("Priority needs the Radiance local API connection");
    const response = await fetcher(new URL("/qwen-radiance/priority", base), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...update, chat_id: chat.id, abi: process.env.QWEN_RADIANCE_CACHE_ABI }),
      redirect: "error", signal: AbortSignal.timeout(6500),
    });
    result = await response.json();
    if (!response.ok) throw new Error(typeof result.error === "string" ? result.error : "Backend does not support priority yet");
  }
  if (result?.applied !== true || result.priority !== update.priority)
    throw new Error("Backend did not confirm the priority update");
  return result;
}

// Installed by the existing cache extension so host and Whonix load the same
// command, with no prompt additions, fake user turns or separate tool calls.
export default function installPriority(pi, { identity, send = sendPriority,
  every = setInterval, clear = clearInterval } = {}) {
  let state, commands = Promise.resolve();
  const inScope = (ctx) => ctx.model?.id === MODEL && !!process.env.QWEN_RADIANCE_CACHE_ABI;
  const notify = (ctx, text, kind = "info") => ctx.ui?.notify(text, kind);
  function make(ctx) {
    const chat = identity(ctx);
    return { ctx, chat, priority: savedPriority(ctx, chat.id), client: nonce(), answer: nonce(),
      sequence: 0, active: false, timer: undefined, queue: Promise.resolve(), pending: 0, warned: false, touched: false };
  }
  function sync(s) {
    s.touched = true;
    const update = { client: s.client, answer: s.answer, sequence: ++s.sequence, priority: s.priority, active: s.active };
    // Preserve start/change/heartbeat/release order even when a control request
    // takes longer than a tool. The backend also rejects stale sequence numbers.
    s.pending++;
    const operation = s.queue.catch(() => {}).then(() => send(s.ctx, s.chat, update));
    s.queue = operation.finally(() => { s.pending--; });
    s.queue.catch(() => {});
    return operation;
  }
  function heartbeat(s) {
    if (s.timer !== undefined || !s.active || !s.priority) return;
    s.timer = every(() => {
      if (!s.active || !s.priority || s.pending) return;
      sync(s).then(() => { s.warned = false; }).catch(() => {
        if (s.active && !s.warned) {
          s.warned = true;
          notify(s.ctx, "Priority heartbeat failed; the GPU reservation expires if the connection stays unavailable.", "warning");
        }
      });
    }, 10_000);
    s.timer?.unref?.();
  }
  async function release(s) {
    if (!s) return;
    clear(s.timer); s.timer = undefined;
    const wasActive = s.active;
    s.active = false;
    if (wasActive && s.touched) {
      try { await sync(s); }
      catch { notify(s.ctx, "Priority release was not confirmed; its reservation expires within 60 seconds.", "warning"); }
    }
  }
  async function current(ctx) {
    if (!inScope(ctx)) throw new Error("/priority is available on the Radiance backend");
    const chat = identity(ctx);
    if (!state || state.chat.id !== chat.id) {
      await release(state);
      state = make(ctx);
    }
    state.ctx = ctx;
    state.chat = chat; // Compaction advances generation, never answer ownership.
    return state;
  }
  async function start(_event, ctx) {
    if (!inScope(ctx)) return;
    const s = await current(ctx);
    if (s.active) return;
    s.active = true; s.answer = nonce();
    if (s.priority) {
      try { await sync(s); }
      catch (error) { notify(ctx, `Priority is not confirmed: ${error.message}`, "warning"); }
      heartbeat(s);
    }
  }
  pi.registerCommand("priority", {
    description: "Show chat priority, or set /priority 0, 1 or 2 (saved for this chat)",
    handler: (args, ctx) => {
      const operation = commands.catch(() => {}).then(async () => {
        const value = args.trim();
        if (value && !/^[012]$/.test(value)) return notify(ctx, "Usage: /priority [0|1|2]", "warning");
        try {
          const s = await current(ctx);
          if (!value) return notify(ctx, `Chat priority: ${s.priority} — ${descriptions[s.priority]}. Equal priorities use normal scheduling.`);
          const previous = s.priority;
          s.priority = Number(value);
          // Commands execute immediately even while Pi is streaming or in a tool.
          if (!s.active && ctx.isIdle?.() === false) { s.active = true; s.answer = nonce(); }
          try { await sync(s); }
          catch (error) {
            s.priority = previous;
            // A timed-out utility may still reach the engine. Order a correction
            // after it; sequence validation prevents late traffic reversing it.
            sync(s).catch(() => {});
            throw error;
          }
          pi.appendEntry(PRIORITY_ENTRY, { priority: s.priority, chat_id: s.chat.id });
          clear(s.timer); s.timer = undefined; heartbeat(s);
          notify(ctx, `Chat priority: ${s.priority} — ${descriptions[s.priority]}.`);
        } catch (error) { notify(ctx, `Priority change was not confirmed: ${error.message}`, "error"); }
      });
      commands = operation;
      return operation;
    },
  });
  pi.on("agent_start", start);
  // agent_end can be followed by automatic recovery/compaction. agent_settled
  // is the patched Pi boundary after the entire answer and all its tools.
  pi.on("agent_settled", () => release(state));
  pi.on("session_shutdown", () => release(state));
  for (const event of ["session_start", "session_switch"]) pi.on(event, async (_event, ctx) => {
    await release(state); state = undefined;
    if (inScope(ctx)) {
      await current(ctx);
      if (ctx.isIdle?.() === false) await start(_event, ctx);
    }
  });
  pi.on("model_select", async (_event, ctx) => {
    if (!inScope(ctx)) { await release(state); state = undefined; }
  });
}
