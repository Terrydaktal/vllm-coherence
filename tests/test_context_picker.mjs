import test from "node:test";
import assert from "node:assert/strict";
import { CONTEXT_POLICY_ENTRY, THINKING_PURGE_ENTRY, contextPolicy, clonePolicy, contextRows,
  changeSelection, effectiveExclusions, selectionChanges, filterContext, filteredRequestContext,
  assertContextReady, clearContextFailure } from "../integrations/pi/qwen-context-policy.mjs";
import { ContextPicker, installContextPicker, previewContextTokens } from "../integrations/pi/qwen-context.mjs";
import { preserveFutureThinking } from "../integrations/pi/qwen-radiance-thinking.mjs";

const text = (value) => ({ type: "text", text: value });
const user = (id, value = "Synthetic request") => ({ id, type: "message", message: { role: "user", content: value, timestamp: 1 } });
const assistant = (id, content = [text("Synthetic answer")]) => ({ id, type: "message", message: { role: "assistant", content,
  timestamp: 2, api: "openai-completions", model: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", provider: "qwen-r9700", stopReason: "stop" } });
const call = (id, toolId = "call") => assistant(id, [{ type: "thinking", thinking: "Synthetic thinking" }, text("Call a tool"),
  { type: "toolCall", id: toolId, name: "lookup", arguments: {} }]);
const result = (id, toolId = "call") => ({ id, type: "message", message: { role: "toolResult", toolCallId: toolId,
  toolName: "lookup", content: [text("Synthetic result")], timestamp: 3 } });
const marker = (id, changes, extra = {}) => ({ id, type: "custom", customType: CONTEXT_POLICY_ENTRY,
  data: { version: 1, changes, preserveFutureThinking: false, ...extra } });
const edit = (entryId, part, excluded) => ({ entryId, part, excluded });

function fixture(entries) {
  let branch = entries, active = entries, idle = true, editor = "unfinished input";
  const notices = [], commands = new Map();
  const ctx = { mode: "tui", contextFilterErrorsPropagate: true, model: { id: assistant("x").message.model }, isIdle: () => idle,
    ui: { notify: (...v) => notices.push(v), getEditorText: () => editor, setEditorText: () => { throw new Error("editor must not be cleared"); } },
    sessionManager: { getBranch: () => branch, buildContextEntries: () => active, getSessionId: () => "synthetic-context-picker",
      getSessionFile: () => undefined, getLeafId: () => branch.at(-1)?.id,
      buildSessionContext: () => ({ messages: active.filter((entry) => entry.type === "message").map((entry) => structuredClone(entry.message)) }) } };
  const pi = { registerCommand: (name, spec) => commands.set(name, spec),
    appendEntry: (customType, data) => { branch.push({ id: `meta-${branch.length}`, type: "custom", customType, data }); } };
  installContextPicker(pi, {});
  return { ctx, pi, notices, command: (args = "") => commands.get("context").handler(args, ctx),
    select: (b, a = b) => { branch = b; active = a; }, busy: () => { idle = false; },
    messages: () => ctx.sessionManager.buildSessionContext().messages, editor: () => editor };
}

test("exclusion/restoration affects only selected entries and never alters saved bodies", () => {
  const entries = [user("u"), assistant("a"), user("v"), assistant("b")], original = structuredClone(entries), f = fixture(entries);
  let policy = changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["u", "a"], "message", true);
  assert.deepEqual(filterContext(f.messages(), f.ctx, policy).map((m) => m.role), ["user", "assistant"]);
  assert.equal(filterContext(f.messages(), f.ctx, policy)[0].content, entries[2].message.content);
  policy = changeSelection(contextRows(f.ctx), policy, ["a"], "message", false);
  assert.deepEqual(filterContext(f.messages(), f.ctx, policy).map((m) => m.role), ["assistant", "user", "assistant"]);
  assert.deepEqual(entries, original);
  assert.deepEqual(selectionChanges(contextPolicy(f.ctx), policy), [edit("u", "message", true)]);
});

test("excluding one result removes its assistant and all results; restoring any member restores all", () => {
  const owner = call("a");
  owner.message.content.push({ type: "toolCall", id: "other-call", name: "edit", arguments: {} });
  const entries = [user("u"), owner, result("r"), result("s", "other-call"), assistant("answer")], f = fixture(entries);
  const policy = changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["r"], "message", true);
  assert.deepEqual([...policy.excluded].sort(), ["a", "r", "s"]);
  assert.deepEqual(filterContext(f.messages(), f.ctx, policy), [entries[0].message, entries[4].message]);
  assert.equal(changeSelection(contextRows(f.ctx), policy, ["s"], "message", false).excluded.size, 0);
});

test("reused call IDs in later turns and identical messages do not leak exclusions across entries", () => {
  const entries = [call("old"), result("old-result"), call("new"), result("new-result")], f = fixture(entries);
  const policy = changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["old"], "message", true);
  const first = filterContext(f.messages(), f.ctx, policy);
  assert.deepEqual(first, entries.slice(2).map((entry) => entry.message));
  assert.deepEqual(filterContext(first, f.ctx, policy), first, "repeated filtering is idempotent");
});

test("a later-arriving result inherits the excluded call owner, without excluding a different call", () => {
  const entries = [call("old"), marker("policy", [edit("old", "message", true)]), result("late"), call("new", "next"), result("next-result", "next")];
  const f = fixture(entries), policy = contextPolicy(f.ctx);
  assert.deepEqual([...effectiveExclusions(contextRows(f.ctx), policy)].sort(), ["late", "old"]);
  assert.deepEqual(filterContext(f.messages(), f.ctx), entries.slice(3).map((entry) => entry.message));
});

test("restoring an interrupted call cannot synthesize a missing tool result", () => {
  const entries = [call("unfinished"), marker("policy", [edit("unfinished", "message", true)])], f = fixture(entries);
  assert.deepEqual(filterContext(f.messages(), f.ctx), []);
  const restored = changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["unfinished"], "message", false);
  assert.throws(() => filterContext(f.messages(), f.ctx, restored), /missing results/);
  entries.push(result("finished"));
  assert.deepEqual(filterContext(f.messages(), f.ctx, restored), [entries[0].message, entries[2].message]);
});

test("aborted and errored streamed calls cannot block a selected context or thinking purge", () => {
  for (const stopReason of ["aborted", "error"]) {
    const interrupted = call("interrupted");
    interrupted.message.stopReason = stopReason;
    for (const policy of [marker("selection", []), { id: "purge", type: "custom", customType: THINKING_PURGE_ENTRY,
      data: { version: 1, entryIds: ["interrupted"], preserveFutureThinking: true } }]) {
      const entries = [user("old"), interrupted, policy, user("continue")], original = structuredClone(entries), f = fixture(entries);
      const filtered = filteredRequestContext(f.messages(), f.ctx);
      assert.doesNotThrow(() => assertContextReady(f.ctx));
      assert.equal(filtered.at(-1).role, "user");
      assert.equal(filtered[1].stopReason, stopReason, "retain the attempt for Pi's normal provider transform to omit");
      if (policy.customType === THINKING_PURGE_ENTRY) assert.ok(!filtered[1].content.some((b) => b.type === "thinking"));
      assert.equal(filtered.filter((m) => m.role === "toolResult").length, 0, "never fabricate execution results");
      assert.deepEqual(entries, original, "saved messages and selection metadata remain unchanged");
    }
  }
});

test("failed tool attempts have no tool dependency ownership or tool-identity requirements", () => {
  const interrupted = call("interrupted");
  interrupted.message.stopReason = "aborted";
  interrupted.message.content.push({ type: "toolCall", id: "call", name: "lookup", arguments: {} });
  const entries = [interrupted, call("complete"), result("executed"), marker("selection", [])], f = fixture(entries);
  const excluded = changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["interrupted"], "message", true);
  assert.deepEqual([...excluded.excluded], ["interrupted"]);
  assert.deepEqual(filterContext(f.messages(), f.ctx, excluded), entries.slice(1, 3).map((e) => e.message));
  const completed = changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["executed"], "message", true);
  assert.deepEqual([...completed.excluded].sort(), ["complete", "executed"]);
  assert.deepEqual(filterContext(f.messages(), f.ctx, completed), [interrupted.message]);
});

test("an aborted attempt cannot authorize an otherwise orphaned result", () => {
  const interrupted = call("interrupted");
  interrupted.message.stopReason = "aborted";
  const f = fixture([interrupted, result("orphan"), marker("selection", [])]);
  assert.throws(() => filterContext(f.messages(), f.ctx), /orphan tool result/);
});

test("a synthetic tool result injected by another hook cannot hide an incomplete saved call", () => {
  const entries = [call("a"), marker("policy", [])], f = fixture(entries);
  assert.throws(() => filteredRequestContext([...f.messages(), result("not-saved").message], f.ctx), /Unrecorded tool result/);
  clearContextFailure(f.ctx);
});

test("thinking selections restore a legacy purge and a later purge excludes restored thinking again", () => {
  const original = call("a"), legacy = { type: "custom", id: "purge", customType: THINKING_PURGE_ENTRY,
    data: { version: 1, entryIds: ["a"], preserveFutureThinking: true } };
  const entries = [original, result("r"), legacy], f = fixture(entries);
  assert.ok(!filterContext(f.messages(), f.ctx)[0].content.some((b) => b.type === "thinking"));
  entries.push(marker("restore", [edit("a", "thinking", false)], { preserveFutureThinking: true }));
  assert.equal(filterContext(f.messages(), f.ctx)[0].content[0].thinking, "Synthetic thinking");
  assert.equal(preserveFutureThinking({ chat_template_kwargs: { preserve_thinking: false } }, f.ctx).chat_template_kwargs.preserve_thinking, true);
  entries.push({ ...legacy, id: "purge-again" });
  assert.ok(!filterContext(f.messages(), f.ctx)[0].content.some((b) => b.type === "thinking"));
  assert.deepEqual(original.message.content[0], { type: "thinking", thinking: "Synthetic thinking" });
});

test("policy survives compaction while matching only its kept tail, and branches before it restore their own policy", () => {
  const old = assistant("old"), kept = assistant("kept"), fresh = assistant("fresh");
  const entries = [old, kept, marker("policy", [edit("kept", "message", true)]), { type: "compaction", id: "compact", summary: "Synthetic checkpoint" }, fresh];
  const f = fixture(entries);
  f.select(entries, [entries[3], kept, fresh]);
  assert.deepEqual(filterContext(f.messages(), f.ctx), [fresh.message]);
  f.select([old, kept]);
  assert.deepEqual(filterContext(f.messages(), f.ctx), [old.message, kept.message]);
});

test("new messages stay included, regardless of clock skew or repeated text", () => {
  const entries = [assistant("old"), marker("policy", [edit("old", "message", true)]), assistant("future")], f = fixture(entries);
  assert.deepEqual(filterContext(f.messages(), f.ctx), [entries[2].message]);
  const unsaved = { role: "user", timestamp: -5, content: "Unsaved current request" };
  assert.deepEqual(filterContext([...f.messages(), unsaved], f.ctx).at(-1), unsaved);
});

test("bang-shell execution entries can be excluded and restored along with the selected turn", () => {
  const shell = { id: "shell", type: "message", message: { role: "bashExecution", command: "fixture", output: "Synthetic shell output", timestamp: 4 } };
  const entries = [user("u"), assistant("a"), shell], f = fixture(entries);
  const after = changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["shell"], "message", true);
  assert.deepEqual(filterContext(f.messages(), f.ctx, after), entries.slice(0, 2).map((entry) => entry.message));
  assert.deepEqual(filterContext(f.messages(), f.ctx, changeSelection(contextRows(f.ctx), after, ["shell"], "message", false)), entries.map((entry) => entry.message));
});

test("malformed metadata and tool identity errors block requests rather than silently restoring excluded context", () => {
  const f = fixture([user("u"), marker("bad", [edit("u", "message", "yes")])]);
  assert.throws(() => assertContextReady(f.ctx), /Invalid saved context/);
  f.select([result("orphan"), marker("valid", [edit("unused", "message", true)])]);
  assert.throws(() => filteredRequestContext(f.messages(), f.ctx), /orphan/);
  f.select([user("u")]);
  assert.throws(() => assertContextReady(f.ctx), /orphan/, "a swallowed context error must still stop the provider");
  filteredRequestContext(f.messages(), f.ctx);
  assert.doesNotThrow(() => assertContextReady(f.ctx));
  const bad = call("bad"); bad.message.content.push(bad.message.content.at(-1));
  f.select([bad]);
  assert.throws(() => changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["bad"], "message", true), /Ambiguous/);
  clearContextFailure(f.ctx);
});

test("context metadata works with another model without adding Qwen template options", () => {
  const f = fixture([assistant("a"), marker("selection", [edit("a", "thinking", true)], { preserveFutureThinking: true })]);
  f.ctx.model.id = "another-model";
  const payload = { messages: [] };
  assert.strictEqual(preserveFutureThinking(payload, f.ctx), payload);
});

test("another hook changing a selected body cannot silently leak it through entry matching", () => {
  const entries = [assistant("a"), marker("selection", [edit("a", "message", true)])], f = fixture(entries);
  const messages = f.messages();
  messages[0].content[0].text = "A transformed private body";
  assert.throws(() => filteredRequestContext(messages, f.ctx), /Cannot safely associate/);
  assert.throws(() => assertContextReady(f.ctx), /request blocked/);
  clearContextFailure(f.ctx);
});

test("a running old runtime cannot apply selections until it loads the safety patch", async () => {
  const entries = [user("u")], f = fixture(entries);
  delete f.ctx.contextFilterErrorsPropagate;
  await f.command();
  assert.equal(entries.length, 1);
  assert.match(f.notices.at(-1)[0], /Restart Pi/);
});

function pickerFixture(entries) {
  const f = fixture(entries), saved = contextPolicy(f.ctx), done = [], renders = [];
  const picker = new ContextPicker({ rows: contextRows(f.ctx), policy: saved, messages: f.messages(), ctx: f.ctx,
    tui: { terminal: { rows: 30 }, requestRender: () => renders.push(true) }, theme: { fg: (_name, value) => value },
    done: (value) => done.push(value), matchesKey: (data, key) => data === key,
    truncateToWidth: (value, width) => value.slice(0, width), preview: async () => ({ before: 100, after: 20, capacity: 250000 }) });
  picker.render(100);
  return { ...f, picker, done, renders };
}

test("picker selects whole turns and older ranges, applies paired exclusions and supports local undo/cancel", () => {
  const f = pickerFixture([user("u"), call("a"), result("r"), user("v"), assistant("answer")]);
  const p = f.picker;
  p.handleInput("home"); p.handleInput("t"); p.handleInput("e");
  assert.deepEqual([...p.policy.excluded].sort(), ["a", "r", "u"]);
  p.handleInput("u"); assert.equal(p.policy.excluded.size, 0);
  p.handleInput("c"); p.handleInput("r"); p.handleInput("down"); p.handleInput("r");
  assert.deepEqual([...p.selected], ["u", "a"]);
  p.handleInput("c"); p.handleInput("o"); assert.deepEqual([...p.selected], ["u", "a"]);
  p.handleInput("escape"); assert.deepEqual(f.done, [undefined]);
  assert.equal(f.editor(), "unfinished input");
});

test("picker handles thinking-only changes, terminal controls and narrow/short terminals", async () => {
  const entry = call("a"); entry.message.content[1].text = "\x1b[31mBAD\x1b[0m\x1b]2;title\x07";
  const f = pickerFixture([entry, result("r")]), p = f.picker;
  p.tui.terminal.rows = 12;
  p.handleInput("home"); p.handleInput("h");
  assert.ok(p.policy.thinking.has("a")); assert.equal(p.policy.excluded.size, 0);
  p.handleInput("shift+h"); assert.equal(p.policy.thinking.size, 0);
  const lines = p.render(25);
  assert.ok(lines.every((line) => line.length <= 25 && !line.includes("\x1b")));
  p.handleInput("p"); await new Promise(setImmediate);
  assert.match(p.render(120).join("\n"), /Exact prompt: 100 → 20/);
  p.dispose();
});

test("an inherited orphan result can be inspected and excluded without crashing the picker", () => {
  const f = pickerFixture([result("orphan"), marker("policy", [edit("old", "message", true)])]);
  f.picker.handleInput("home"); f.picker.handleInput("e");
  assert.deepEqual(filterContext(f.messages(), f.ctx, f.picker.policy), []);
});

test("a saved repair clears a prior filter failure before a manual compaction request", async () => {
  const entries = [result("orphan"), marker("policy", [edit("old", "message", true)])], f = fixture(entries);
  assert.throws(() => filteredRequestContext(f.messages(), f.ctx), /orphan/);
  f.ctx.ui.custom = async () => changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["orphan"], "message", true);
  await f.command();
  assert.doesNotThrow(() => assertContextReady(f.ctx));
  assert.deepEqual(filterContext(f.messages(), f.ctx), []);
});

test("save persists only IDs and reversible decisions; /context undo reverses repeated changes and leaves draft input alone", async () => {
  const entries = [user("u"), assistant("a")], f = fixture(entries), original = structuredClone(entries);
  f.ctx.ui.custom = async () => changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["u"], "message", true);
  await f.command();
  assert.deepEqual(entries.at(-1).data.changes, [edit("u", "message", true)]);
  assert.doesNotMatch(JSON.stringify(entries.at(-1)), /Synthetic request|Synthetic answer/);
  f.ctx.ui.custom = async () => changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["a"], "message", true);
  await f.command(); await f.command("undo");
  assert.deepEqual([...contextPolicy(f.ctx).excluded], ["u"]);
  await f.command("undo"); assert.equal(contextPolicy(f.ctx).excluded.size, 0);
  await f.command("undo"); assert.match(f.notices.at(-1)[0], /No context selection/);
  assert.deepEqual(entries.slice(0, 2), original);
  assert.equal(f.editor(), "unfinished input");
});

test("cancel, busy commands, non-TUI picker and a changed leaf never save context edits", async () => {
  const entries = [user("u")], f = fixture(entries);
  f.ctx.ui.custom = async () => undefined;
  await f.command(); assert.equal(entries.length, 1);
  f.ctx.ui.custom = async () => {
    const after = changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["u"], "message", true);
    entries.push(assistant("new")); return after;
  };
  await f.command(); assert.match(f.notices.at(-1)[0], /chat changed/);
  assert.equal(entries.length, 2);
  f.ctx.mode = "rpc"; await f.command(); assert.match(f.notices.at(-1)[0], /interactive terminal/);
  f.busy(); await f.command(); assert.match(f.notices.at(-1)[0], /Escape/);
  assert.equal(entries.length, 2);
});

test("exact preview tokenizes complete before/after prompts and cannot contact the generation endpoint", async () => {
  const entries = [user("old", "Long obsolete request"), assistant("a")], f = fixture(entries), bodies = [];
  f.ctx.model = { ...f.ctx.model, api: "openai-completions", provider: "qwen-r9700", baseUrl: "http://fixture.invalid/v1", contextWindow: 253792 };
  f.ctx.getSystemPrompt = () => "Fixture system prompt";
  f.ctx.modelRegistry = { getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "synthetic-key" }) };
  Object.assign(f.pi, { getAllTools: () => [], getActiveTools: () => [], getThinkingLevel: () => "off" });
  const policy = changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ["old"], "message", true);
  const dependencies = { convertToLlm: (messages) => messages,
    streamSimpleOpenAICompletions: (_model, context, options) => ({ result: async () => {
      try { options.onPayload({ model: "fixture", messages: [{ role: "system", content: context.systemPrompt }, ...context.messages], tools: [] }); }
      catch (error) { return { errorMessage: error.message }; }
      throw new Error("preview tried to generate");
    } }),
    fetcher: async (url, options) => { assert.equal(url, "http://fixture.invalid/tokenize"); bodies.push(JSON.parse(options.body));
      return Response.json({ tokens: new Array(bodies.length === 1 ? 500 : 200).fill(1) }); } };
  assert.deepEqual(await previewContextTokens(f.pi, f.ctx, policy, dependencies, AbortSignal.timeout(1000)), { before: 500, after: 200, capacity: 253792 });
  assert.match(JSON.stringify(bodies[0].messages), /Long obsolete request/);
  assert.doesNotMatch(JSON.stringify(bodies[1].messages), /Long obsolete request/);
  assert.match(JSON.stringify(bodies[1].messages), /Fixture system prompt/);
  assert.ok(bodies.every((body) => body.add_generation_prompt && !body.add_special_tokens));
  assert.equal(entries.length, 2, "preview cannot persist decisions");
});

test("bounded combinations of message selections always preserve valid tool/result pairs", () => {
  const entries = [user("u"), call("a"), result("r"), user("v"), call("b", "other"), result("s", "other")], f = fixture(entries);
  for (let mask = 0; mask < (1 << entries.length); mask++) {
    const ids = entries.filter((_entry, i) => mask & (1 << i)).map((entry) => entry.id);
    const selected = changeSelection(contextRows(f.ctx), contextPolicy(f.ctx), ids, "message", true);
    const messages = filterContext(f.messages(), f.ctx, selected);
    const calls = new Set(messages.flatMap((m) => (Array.isArray(m.content) ? m.content : []).filter((b) => b.type === "toolCall").map((b) => b.id)));
    for (const message of messages) if (message.role === "toolResult") assert.ok(calls.has(message.toolCallId));
    const restored = changeSelection(contextRows(f.ctx), selected, ids, "message", false);
    assert.deepEqual(filterContext(f.messages(), f.ctx, restored), entries.map((entry) => entry.message));
  }
});
