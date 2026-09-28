import test from "node:test";
import assert from "node:assert/strict";
import { THINKING_PURGE_ENTRY, installThinkingPurge, preserveFutureThinking,
  purgeContextThinking, thinkingPurgePolicy } from "../integrations/pi/qwen-radiance-thinking.mjs";

const MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate";
const assistant = (id, thought, content = [], timestamp = 10) => ({ id, type: "message", message: {
  role: "assistant", api: "openai-completions", provider: "qwen-r9700", model: MODEL, timestamp,
  content: [{ type: "thinking", thinking: thought, thinkingSignature: "reasoning" }, ...content], stopReason: "stop",
} });
const marker = (id, entryIds) => ({ id, type: "custom", customType: THINKING_PURGE_ENTRY,
  data: { version: 1, entryIds, preserveFutureThinking: true } });
const text = (value) => ({ type: "text", text: value });

function fixture(entries) {
  let branch = entries, context = entries, idle = true;
  const notices = [], commands = new Map(), handlers = new Map();
  const ctx = { model: { id: MODEL }, isIdle: () => idle, ui: { notify: (...v) => notices.push(v) },
    sessionManager: { getBranch: () => branch, buildContextEntries: () => context,
      getSessionFile: () => undefined, getSessionId: () => "synthetic-thinking-purge" } };
  const pi = { on: (name, fn) => handlers.set(name, fn), registerCommand: (name, value) => commands.set(name, value),
    appendEntry: (customType, data) => {
      const entry = { id: `marker-${branch.length}`, type: "custom", customType, data };
      branch.push(entry);
      if (context !== branch) context.push(entry);
    } };
  installThinkingPurge(pi);
  return { ctx, notices, handlers, command: (arg = "") => commands.get("purge-thinking").handler(arg, ctx),
    branch: () => branch, select: (nextBranch, nextContext = nextBranch) => { branch = nextBranch; context = nextContext; },
    idle: (value) => { idle = value; }, messages: () => context.filter((e) => e.type === "message").map((e) => structuredClone(e.message)) };
}

test("purge removes only earlier thinking, preserves tools/prose, and never rewrites saved messages", async () => {
  const old = assistant("old", "old reasoning", [text("Prose including literal <think> tags."),
    { type: "toolCall", id: "call-a", name: "edit", arguments: { oldText: "old", newText: "new" } }]);
  const tool = { type: "message", id: "tool", message: { role: "toolResult", toolCallId: "call-a", toolName: "edit",
    content: [text("Tool result")], timestamp: 11 } };
  const entries = [{ type: "message", id: "user", message: { role: "user", content: "Request", timestamp: 9 } }, old, tool];
  const originals = structuredClone(entries);
  const f = fixture(entries);
  await f.command();
  entries.push(assistant("future", "new reasoning", [text("New answer")], 5)); // Clock went backwards.
  const result = purgeContextThinking(f.messages(), f.ctx);
  assert.deepEqual(result[0], originals[0].message);
  assert.deepEqual(result[1].content, originals[1].message.content.slice(1));
  assert.deepEqual(result[2], originals[2].message);
  assert.equal(result[3].content[0].thinking, "new reasoning");
  assert.deepEqual(entries.slice(0, 3), originals);
  assert.deepEqual(entries[3].data.entryIds, ["old"]);
  assert.doesNotMatch(JSON.stringify(entries[3].data), /old reasoning|Prose|Tool result/);
});

test("identical timestamp/content on either side of cutoff are distinguished and filtering is idempotent", () => {
  const old = assistant("old", "repeated reasoning"), future = assistant("future", "repeated reasoning");
  const f = fixture([old, marker("purge", ["old"]), future]);
  const first = purgeContextThinking(f.messages(), f.ctx);
  assert.deepEqual(first[0].content, []);
  assert.equal(first[1].content[0].thinking, "repeated reasoning");
  assert.deepEqual(purgeContextThinking(first, f.ctx), first);
});

test("compaction uses only retained entries for matching and keeps a purge outside the retained tail", () => {
  const old = assistant("old", "same"), kept = assistant("kept", "same"), fresh = assistant("fresh", "future");
  const purge = marker("purge", ["old", "kept"]);
  const compaction = { type: "compaction", id: "compact", summary: "Existing checkpoint", firstKeptEntryId: "kept" };
  const f = fixture([old, purge, kept, compaction, fresh]);
  f.select(f.branch(), [compaction, kept, fresh]);
  const result = purgeContextThinking(f.messages(), f.ctx);
  assert.deepEqual(result[0].content, []);
  assert.equal(result[1].content[0].thinking, "future");
  // If only the future duplicate survives compaction, it must not match the
  // earlier, summarized entry just because content and timestamps coincide.
  f.select(f.branch(), [compaction, fresh]);
  assert.equal(purgeContextThinking(f.messages(), f.ctx)[0].content[0].thinking, "future");
});

test("purges accumulate only when requested and remain scoped to the selected branch", async () => {
  const old = assistant("old", "first");
  const f = fixture([old]);
  const original = f.messages();
  assert.deepEqual(purgeContextThinking(original, f.ctx), original);
  await f.command();
  f.branch().push(assistant("future", "second"));
  assert.equal(purgeContextThinking(f.messages(), f.ctx)[1].content[0].thinking, "second");
  await f.command();
  assert.ok(purgeContextThinking(f.messages(), f.ctx).every((m) => m.content.length === 0));
  const count = f.branch().length;
  await f.command();
  await f.command("status");
  assert.equal(f.branch().length, count);
  assert.match(f.notices.at(-1)[0], /2 earlier messages excluded; future thinking retained/);
  f.select([old]);
  assert.deepEqual(purgeContextThinking(original, f.ctx), original);
  assert.equal(thinkingPurgePolicy(f.ctx).active, false);
  f.select([]);
  assert.equal(thinkingPurgePolicy(f.ctx).active, false);
});

test("per-chat template override retains future thinking even when globally disabled, without enabling fresh thinking", async () => {
  const f = fixture([]);
  const payload = { chat_template_kwargs: { enable_thinking: false, preserve_thinking: false, preserve_reasoning: false }, messages: [] };
  assert.strictEqual(preserveFutureThinking(payload, f.ctx), payload);
  await f.command(); // A fresh chat still gets the future-retention policy.
  const result = preserveFutureThinking(payload, f.ctx);
  assert.deepEqual(result.chat_template_kwargs, { enable_thinking: false, preserve_thinking: true, preserve_reasoning: true });
  assert.equal(payload.chat_template_kwargs.preserve_thinking, false);
  assert.equal(payload.chat_template_kwargs.preserve_reasoning, false);
  f.ctx.model.id = "different-model";
  assert.strictEqual(preserveFutureThinking(payload, f.ctx), payload);
});

test("busy commands and invalid arguments do not change the session or input", async () => {
  const f = fixture([assistant("old", "reasoning")]);
  f.idle(false);
  await f.command();
  assert.equal(f.branch().length, 1);
  assert.match(f.notices.at(-1)[0], /Escape/);
  f.idle(true);
  await f.command("everything");
  assert.equal(f.branch().length, 1);
  assert.match(f.notices.at(-1)[0], /Usage/);
});

test("saved malformed state and unsupported runtimes reject the provider request instead of restoring old thinking", () => {
  const f = fixture([{ ...marker("bad", ["old"]), data: { version: 1, entryIds: null } }]);
  assert.throws(() => f.handlers.get("before_provider_request")({ payload: {} }, f.ctx), /Invalid saved thinking purge/);
  f.select([marker("valid", ["old"])]);
  delete f.ctx.sessionManager.buildContextEntries;
  assert.throws(() => f.handlers.get("before_provider_request")({ payload: {} }, f.ctx), /patched Pi/);
});
