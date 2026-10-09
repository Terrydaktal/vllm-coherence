// Exercise the production extension through the pinned Pi SDK using synthetic
// history and an offline provider. Never load a user's transcript or model.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { TASK_PLAN_ENTRY, replayTaskPlan } from "../integrations/pi/qwen-task-plan.mjs";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const available = existsSync(join(root, "pi-coding-agent/dist/core/sdk.js"));
const headings = ["Goal", "Current Authoritative State", "Constraints & Invariants", "Progress", "Measurements & Evidence", "Key Decisions",
  "Rejected / Failed Approaches", "Unresolved Questions & Hypotheses", "Next Steps", "Critical Context"];
const checkpoint = headings.map((heading) => `### ${heading}\nSynthetic continuity fixture.`).join("\n\n");

async function fixture(t, { forced = false, promptTokens = 6, stableUsageTokens = 200 } = {}) {
  const mod = (path) => import(pathToFileURL(join(root, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const agentDir = await mkdtemp(join(tmpdir(), "compaction-continuity-sdk-"));
  const previous = Object.fromEntries(["PI_CODING_AGENT_DIR", "QWEN_RADIANCE_CACHE_ABI", "PI_OFFLINE"].map((name) => [name, process.env[name]]));
  process.env.PI_CODING_AGENT_DIR = agentDir;
  process.env.PI_OFFLINE = "1";
  delete process.env.QWEN_RADIANCE_CACHE_ABI;
  t.after(async () => {
    for (const [name, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[name]; else process.env[name] = value;
    }
    await rm(agentDir, { recursive: true, force: true });
  });
  const model = { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", name: "Synthetic Continuity", provider: "qwen-r9700", api: "openai-completions",
    baseUrl: "http://fixture.invalid/v1", reasoning: true, input: ["text"], contextWindow: 32768, maxTokens: 8192,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: true, reserveTokens: 8192, keepRecentTokens: 256 }, retry: { enabled: false } }, { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd: agentDir, agentDir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [resolve("integrations/pi/qwen-radiance-compaction.ts"), resolve("integrations/pi/qwen-task-plan.ts")], systemPrompt: "Synthetic continuity integration fixture." });
  await resourceLoader.reload();
  const loaded = resourceLoader.getExtensions();
  assert.deepEqual(loaded.errors, []);
  const finishShortcut = loaded.extensions.flatMap((extension) => [...extension.shortcuts.values()]).find((shortcut) => shortcut.shortcut === "alt+c");
  const requests = [], tokenizationBodies = [], notices = [], compactEvents = [];
  const forcedText = "### Goal\nKeep this exact unfinished state: αβ\n\n### Progress\n- not finished ";
  let streamCancelled = false, finishRequested = false;
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.ok(String(url).startsWith("http://fixture.invalid/"), "real network access is forbidden");
    const body = JSON.parse(options.body);
    requests.push(String(url));
    if (String(url).endsWith("/tokenize")) {
      tokenizationBodies.push(body);
      return Response.json({ tokens: body.prompt ? (body.prompt.includes("</think>") ? [4, 6, 7] : [4, 5]) :
        body.add_generation_prompt ? [1, 2, ...Array(promptTokens - 5).fill(3), 4, 5] : [1, 2] });
    }
    assert.equal(String(url), "http://fixture.invalid/v1/completions");
    if (forced) {
      return new Response(new ReadableStream({
        start(controller) { controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({ choices: [{ index: 0, text: forcedText, finish_reason: null }] })}\n\n`)); },
        cancel() { streamCancelled = true; },
      }), { headers: { "Content-Type": "text/event-stream" } });
    }
    const frames = [
      { choices: [{ index: 0, text: checkpoint + "\nCOMPACTION_SUMMARY_COMPLETE", finish_reason: "stop" }] },
      { usage: { prompt_tokens: 6, completion_tokens: 300, prompt_tokens_details: { cached_tokens: 2 } } },
      "[DONE]",
    ];
    return new Response(frames.map((frame) => `data: ${typeof frame === "string" ? frame : JSON.stringify(frame)}\n\n`).join(""),
      { headers: { "Content-Type": "text/event-stream" } });
  });
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: () => { throw new Error("compaction must not call the acting model or an auxiliary model"); } };
  const sessionManager = SessionManager.create(agentDir, join(agentDir, "sessions"));
  const usage = { input: stableUsageTokens - 100, output: 100, cacheRead: 0, cacheWrite: 0, totalTokens: stableUsageTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const user = (text) => sessionManager.appendMessage({ role: "user", content: text, timestamp: Date.now() });
  const assistant = (content, stopReason = "stop") => sessionManager.appendMessage({ role: "assistant", api: model.api, provider: model.provider,
    model: model.id, timestamp: Date.now(), content, stopReason, usage });
  user("USER REQUIREMENT: Preserve exact FP8 arithmetic. Do not repeat rejected approximation experiments.");
  assistant([{ type: "text", text: "Old synthetic work already completed. ".repeat(1200) }]);
  const { session } = await createAgentSession({ cwd: agentDir, agentDir, model, modelRuntime, resourceLoader, sessionManager, settingsManager, tools: [] });
  const unsubscribe = session.subscribe((event) => { if (event.type === "compaction_end") compactEvents.push(event); });
  t.after(async () => {
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    unsubscribe(); session.dispose();
  });
  await session.bindExtensions({ uiContext: { setWidget: () => {}, setStatus: () => {}, notify: (text) => notices.push(text), setWorkingMessage: (message) => {
    if (forced && !finishRequested && message?.includes("Alt+C finish now")) {
      finishRequested = true;
      assert.ok(finishShortcut, "the original Alt+C cutoff remains registered");
      finishShortcut.handler(session.extensionRunner.createContext());
    }
  } } });
  return { session, sessionManager, SessionManager, agentDir, user, assistant, requests, tokenizationBodies, notices, compactEvents, forcedText,
    streamCancelled: () => streamCancelled, finishRequested: () => finishRequested };
}

test("pinned SDK commits source-linked continuity memory and an intact original tool tail without extra model calls", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t);
  f.user("Latest objective: correctness is done; now measure the final release performance. Keep 320 / 320 as the acceptance condition.");
  const call = f.assistant([{ type: "text", text: "Running the accepted check." }, { type: "toolCall", id: "continuity-probe", name: "probe", arguments: { check: "qualified" } }], "toolUse");
  const result = f.sessionManager.appendMessage({ role: "toolResult", toolCallId: "continuity-probe", toolName: "probe", content: [{ type: "text", text: "MEASURED: 320 / 320 exact; performance pending." }], isError: false, timestamp: Date.now() });
  const before = f.sessionManager.getEntries().filter((entry) => entry.type === "message");
  const compacted = await f.session.compact();
  assert.deepEqual(f.notices, []);
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, 1);
  assert.equal(f.requests.length, 5, "the continuity pass is deterministic CPU work, not another inference request");
  assert.equal(compacted.details.summaryRequests, 1);
  const memory = compacted.details.continuity;
  assert.equal(memory.version, 1);
  assert.ok(memory.packet.text.length > 0);
  assert.match(memory.packet.digest, /^[a-f0-9]{64}$/);
  assert.ok(compacted.summary.includes(memory.packet.text), "normal compaction supplies deterministic continuity memory to the resumed model");
  assert.ok(memory.packet.sourceIds.every((id) => before.some((entry) => entry.id === id)));
  assert.equal(memory.tail.firstKeptEntryId, compacted.firstKeptEntryId);
  assert.ok(f.tokenizationBodies.some((body) => body.messages?.some((message) => {
    const text = typeof message.content === "string" ? message.content : (message.content ?? []).map((block) => block.text ?? "").join("\n");
    return text.includes(memory.packet.text);
  })), "the summary request receives source-linked evidence before generation");
  const activeIds = new Set(f.sessionManager.buildContextEntries().map((entry) => entry.id));
  assert.ok(activeIds.has(call) && activeIds.has(result), "a retained result keeps its owning call");
  assert.deepEqual(f.sessionManager.getEntries().filter((entry) => entry.type === "message"), before, "original transcript entries are immutable");
  const reopened = f.SessionManager.open(f.sessionManager.getSessionFile(), join(f.agentDir, "sessions"));
  const saved = reopened.getEntries().find((entry) => entry.type === "compaction");
  assert.deepEqual(saved.details.continuity, JSON.parse(JSON.stringify(memory)), "all serializable continuity evidence survives reopening");
  assert.equal(saved.summary, compacted.summary);
  assert.equal(reopened.buildSessionContext().messages[0].summary, compacted.summary);
});

test("pinned SDK compaction restores persistent pending work, constraints and fresh relevant file contents", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t);
  const goal = f.user("Implement the final repair, preserve exact arithmetic, then run the pending performance check.");
  const path = join(f.agentDir, "repair.mjs");
  await writeFile(path, "export const finalArithmetic = 42;\n");
  f.assistant([{ type: "toolCall", id: "file-write", name: "write", arguments: { path: "repair.mjs", content: "export const oldArithmetic = 41;" } }], "toolUse");
  const result = f.sessionManager.appendMessage({ role: "toolResult", toolName: "write", toolCallId: "file-write",
    content: [{ type: "text", text: "Write completed." }], isError: false, timestamp: Date.now() });
  const state = { version: 1, mode: "execute", plan: {
    goal: "Final arithmetic repair and performance check", goalSourceEntryIds: [goal],
    steps: [
      { id: "repair", title: "Repair arithmetic", status: "completed", sourceEntryIds: [goal], relevantFiles: ["repair.mjs"], evidence: [{ entryId: result, note: "File written; numerical qualification is separate." }] },
      { id: "measure", title: "Run the pending performance check", status: "pending", sourceEntryIds: [goal], relevantFiles: ["repair.mjs"], evidence: [] },
    ], nextStepId: 3, relevantFiles: ["repair.mjs"], relevantFileSources: { "repair.mjs": [goal] },
    constraints: [{ text: "Preserve exact arithmetic", sourceEntryIds: [goal] }], notes: [],
  } };
  f.sessionManager.appendCustomEntry(TASK_PLAN_ENTRY, state);
  const sourceLeaf = f.sessionManager.getLeafId();
  const first = await f.session.compact().catch((error) => { throw new Error(f.notices.join("\n"), { cause: error }); });
  assert.deepEqual(f.notices, []);
  const restoration = first.details.continuity.restoration;
  assert.ok(restoration.charCount <= restoration.maxChars);
  assert.ok(first.summary.includes("Run the pending performance check"));
  assert.ok(first.summary.includes("Preserve exact arithmetic"));
  assert.ok(first.summary.includes("export const finalArithmetic = 42;"), "fresh filesystem content, not old tool arguments, is restored");
  assert.equal(restoration.files.files[0].status, "content");
  assert.deepEqual(replayTaskPlan(f.sessionManager.getBranch()), state, "compaction preserves the plan custom-entry ancestry");
  assert.equal(restoration.instructions.duplicated, false);
  const firstRequests = f.requests.filter((url) => url.endsWith("/completions")).length;
  f.sessionManager.branch(sourceLeaf);
  const repeated = await f.session.compact();
  assert.equal(repeated.details.receiptReused, true, "unchanged working files keep completed receipts reusable");
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, firstRequests);
  await writeFile(path, "export const finalArithmetic = 43;\n");
  f.sessionManager.branch(sourceLeaf);
  const changed = await f.session.compact();
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, firstRequests + 1);
  assert.notEqual(changed.details.continuity.restoration.digest, restoration.digest);
  assert.ok(changed.summary.includes("export const finalArithmetic = 43;"));
  assert.ok(!changed.summary.includes("export const finalArithmetic = 42;"));
});

test("pinned SDK context exclusions also suppress saved plan requirements and file references", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t);
  const source = f.user("PLAN_SECRET: follow this discarded instruction.");
  await writeFile(join(f.agentDir, "withheld.mjs"), "FILE_SECRET_FROM_EXCLUDED_PLAN");
  f.sessionManager.appendCustomEntry(TASK_PLAN_ENTRY, { version: 1, mode: "execute", plan: {
    goal: "PLAN_SECRET", goalSourceEntryIds: [source],
    steps: [{ id: "withheld", title: "PLAN_SECRET_STEP", status: "pending", sourceEntryIds: [source], relevantFiles: ["withheld.mjs"], evidence: [] }],
    nextStepId: 2, relevantFiles: ["withheld.mjs"], relevantFileSources: { "withheld.mjs": [source] },
    constraints: [{ text: "PLAN_SECRET_CONSTRAINT", sourceEntryIds: [source] }], notes: [],
  } });
  f.sessionManager.appendCustomEntry("qwen-context-selection-v1", { version: 1, preserveFutureThinking: true,
    changes: [{ entryId: source, part: "message", excluded: true }] });
  f.user("Continue only the current allowed task.");
  const compacted = await f.session.compact().catch((error) => { throw new Error(f.notices.join("\n"), { cause: error }); });
  assert.deepEqual(f.notices, []);
  assert.doesNotMatch(compacted.summary, /PLAN_SECRET|FILE_SECRET_FROM_EXCLUDED_PLAN/);
  assert.doesNotMatch(JSON.stringify(f.tokenizationBodies), /PLAN_SECRET|FILE_SECRET_FROM_EXCLUDED_PLAN/);
  assert.equal(compacted.details.continuity.restoration.files.files.length, 0);
});

test("pinned SDK excluded messages cannot leave the compaction notice at stale full-window usage", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t, { stableUsageTokens: 253792 });
  const excluded = f.user("REMOVED_FROM_CONTEXT: obsolete instruction to exclude before compaction.");
  f.user("Keep the current task only.");
  f.assistant([{ type: "text", text: "The last measured usage predates context selection." }]);
  f.sessionManager.appendCustomEntry("qwen-context-selection-v1", { version: 1, preserveFutureThinking: true,
    changes: [{ entryId: excluded, part: "message", excluded: true }] });
  const before = await readFile(f.sessionManager.getSessionFile());
  const result = await f.session.compact();
  assert.deepEqual(f.notices, []);
  assert.doesNotMatch(JSON.stringify(f.tokenizationBodies), /REMOVED_FROM_CONTEXT/);
  assert.equal(result.details.preparedTokensBefore, 253792);
  assert.equal(result.tokensBefore, 2, "actual selected history excludes checkpoint-specific prompt tokens");
  assert.equal(result.usage.input + result.usage.cacheRead, 6);
  const event = f.compactEvents.find((item) => item.result);
  assert.equal(event.result.tokensBefore, 2);
  const bytes = await readFile(f.sessionManager.getSessionFile());
  assert.ok(bytes.subarray(0, before.length).equals(before), "original transcript bytes remain untouched");
  const saved = bytes.toString("utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line))
    .find((entry) => entry.type === "compaction");
  assert.equal(saved.tokensBefore, 2);
  const reopened = f.SessionManager.open(f.sessionManager.getSessionFile(), join(f.agentDir, "sessions"));
  assert.equal(reopened.getEntries().find((entry) => entry.type === "compaction").tokensBefore, 2);
  assert.equal(reopened.buildSessionContext().messages[0].tokensBefore, 2);
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, 1);
  assert.equal(f.requests.filter((url) => url.endsWith("/tokenize")).length, 4);
});

test("pinned SDK continuity memory respects selected message and thinking exclusions", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t);
  const excluded = f.user("EXCLUDED_POISONED_USER_SECRET: This discarded request must not reappear in memory.");
  const thinking = f.assistant([{ type: "thinking", thinking: "EXCLUDED_POISONED_THINKING_SECRET", thinkingSignature: "synthetic" },
    { type: "text", text: "Allowed final conclusion: the operator check passed." }]);
  f.sessionManager.appendCustomEntry("qwen-context-selection-v1", { version: 1, preserveFutureThinking: true,
    changes: [{ entryId: excluded, part: "message", excluded: true }, { entryId: thinking, part: "thinking", excluded: true }] });
  f.user("Continue with the allowed final performance check.");
  f.assistant([{ type: "text", text: "Recent allowed activity. ".repeat(40) }]);
  const compacted = await f.session.compact();
  assert.deepEqual(f.notices, []);
  const memory = compacted.details.continuity;
  assert.ok(!memory.packet.sourceIds.includes(excluded));
  assert.ok(!memory.packet.text.includes("EXCLUDED_POISONED_USER_SECRET"));
  assert.ok(!memory.packet.text.includes("EXCLUDED_POISONED_THINKING_SECRET"));
  const requests = JSON.stringify(f.tokenizationBodies);
  assert.ok(!requests.includes("EXCLUDED_POISONED_USER_SECRET"));
  assert.ok(!requests.includes("EXCLUDED_POISONED_THINKING_SECRET"));
  assert.ok(JSON.stringify(f.sessionManager.getEntries()).includes("EXCLUDED_POISONED_THINKING_SECRET"), "purging active context never deletes original evidence");
});

test("pinned SDK binds identical duplicate messages to their retained entry IDs", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t);
  const content = "Identical user requirement: preserve the qualified arithmetic and continue the pending performance check.";
  const timestamp = Date.now();
  const excluded = f.sessionManager.appendMessage({ role: "user", content, timestamp });
  const retained = f.sessionManager.appendMessage({ role: "user", content, timestamp });
  f.sessionManager.appendCustomEntry("qwen-context-selection-v1", { version: 1, preserveFutureThinking: true,
    changes: [{ entryId: excluded, part: "message", excluded: true }] });
  f.assistant([{ type: "text", text: "Recent allowed synthetic activity. ".repeat(80) }]);
  const compacted = await f.session.compact();
  assert.deepEqual(f.notices, []);
  assert.ok(!compacted.details.continuity.packet.sourceIds.includes(excluded));
  assert.ok(compacted.details.continuity.packet.sourceIds.includes(retained), "the equal body belongs to the retained occurrence, not its excluded twin");
  const history = f.tokenizationBodies.find((body) => body.messages && body.add_generation_prompt === false);
  assert.ok(history);
  assert.equal(history.messages.filter((message) => message.content === content).length, 1);
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, 1);
});

test("pinned SDK Alt+C keeps the exact partial checkpoint while persisting continuity metadata", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t, { forced: true });
  f.user("Current objective: retain the existing Alt+C cutoff behavior.");
  f.assistant([{ type: "text", text: "Recent continuity fixture work. ".repeat(80) }]);
  const compacted = await f.session.compact();
  assert.deepEqual(f.notices, []);
  assert.equal(f.finishRequested(), true);
  assert.equal(f.streamCancelled(), true);
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, 1, "Alt+C never requests a replacement summary");
  assert.equal(compacted.summary, f.forcedText, "continuity memory must not rewrite the user's explicitly accepted partial text");
  assert.equal(compacted.details.forcedCheckpoint, true);
  assert.equal(compacted.details.continuity.version, 1);
  assert.ok(compacted.details.continuity.packet.text.length > 0);
  const saved = f.sessionManager.getEntries().find((entry) => entry.type === "compaction");
  assert.equal(saved.summary, f.forcedText);
  assert.deepEqual(saved.details.continuity, compacted.details.continuity);
});

test("pinned SDK repeated compaction refreshes continuity sources without importing a sibling branch", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t);
  const base = f.sessionManager.getLeafId();
  const siblingUser = f.user("SIBLING_BRANCH_SECRET: obsolete alternative, never selected for this continuation.");
  const siblingAssistant = f.assistant([{ type: "text", text: "Sibling experiment. ".repeat(400) }]);
  f.sessionManager.branch(base);
  f.user("Current correction: keep the accepted arithmetic; only the final performance check is pending.");
  f.assistant([{ type: "text", text: "Selected branch activity. ".repeat(80) }]);
  const first = await f.session.compact();
  const latestUser = f.user("Latest correction: performance has been measured; next publish the exact qualified result.");
  f.assistant([{ type: "text", text: "Latest selected branch activity. ".repeat(80) }]);
  const second = await f.session.compact();
  assert.deepEqual(f.notices, []);
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, 2);
  assert.equal(f.requests.length, 10, "each compaction has exactly one summary request");
  const all = f.sessionManager.getEntries().filter((entry) => entry.type === "compaction");
  assert.equal(all.length, 2);
  assert.notEqual(first.summary, second.summary, "fresh source-linked memory distinguishes otherwise identical generated summaries");
  assert.ok(second.details.continuity.packet.sourceIds.includes(latestUser));
  for (const result of [first, second]) {
    assert.ok(!result.details.continuity.packet.sourceIds.includes(siblingUser));
    assert.ok(!result.details.continuity.packet.sourceIds.includes(siblingAssistant));
    assert.ok(!result.summary.includes("SIBLING_BRANCH_SECRET"));
  }
  assert.ok(!JSON.stringify(f.tokenizationBodies).includes("SIBLING_BRANCH_SECRET"));
  assert.equal(f.sessionManager.buildContextEntries()[0].id, all[1].id);
  assert.deepEqual(all[1].details.continuity, second.details.continuity);
});

test("pinned SDK refuses a continuity request with inadequate exact tokenizer headroom", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t, { promptTokens: 32768 - 1024 });
  f.user("Compact only if the exact request leaves enough room for a valid checkpoint.");
  f.assistant([{ type: "text", text: "Recent synthetic activity. ".repeat(80) }]);
  const before = f.sessionManager.getEntries().filter((entry) => entry.type === "message");
  await assert.rejects(f.session.compact(), /Compaction cancelled/);
  assert.ok(f.notices.some((notice) => /at least 2048 required/.test(notice)));
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, 0, "insufficient space cannot trigger expensive replacement or fallback inference");
  assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, 0);
  assert.deepEqual(f.sessionManager.getEntries().filter((entry) => entry.type === "message"), before);
});

test("pinned SDK reports an oversized newest tool group and preserves the prepared boundary without truncating it", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t);
  f.user("Keep the tool call and its original large result together.");
  const body = "Exact unmodified synthetic action context. ".repeat(800);
  const call = f.assistant([{ type: "text", text: body }, { type: "toolCall", id: "oversized-probe", name: "probe", arguments: {} }], "toolUse");
  const result = f.sessionManager.appendMessage({ role: "toolResult", toolCallId: "oversized-probe", toolName: "probe", content: [{ type: "text", text: "Exact small synthetic result." }], isError: false, timestamp: Date.now() });
  const compacted = await f.session.compact();
  assert.deepEqual(f.notices, []);
  const tail = compacted.details.continuity.tail;
  assert.equal(tail.fallbackToPreparedBoundary, true);
  assert.ok(tail.oversizedNewestGroup.estimatedTokens > tail.tokenBudget);
  const activeIds = new Set(f.sessionManager.buildContextEntries().map((entry) => entry.id));
  assert.ok(activeIds.has(call) && activeIds.has(result));
  assert.equal(f.sessionManager.getEntries().find((entry) => entry.id === call).message.content[0].text, body);
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, 1);
});

test("pinned SDK reuses the same bound continuity receipt and rejects corrupted persisted memory", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t);
  f.user("Current objective: publish the qualified result without repeating completed work.");
  f.assistant([{ type: "text", text: "Recent completed synthetic work. ".repeat(80) }]);
  const sourceLeaf = f.sessionManager.getLeafId();
  const first = await f.session.compact();
  f.sessionManager.branch(sourceLeaf);
  const reused = await f.session.compact();
  assert.equal(reused.details.receiptReused, true);
  assert.equal(reused.summary, first.summary);
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, 1, "an unchanged source and memory packet recover without inference");
  const directory = join(f.agentDir, "radiance-compaction-receipts");
  const name = (await readdir(directory)).find((name) => /^[a-f0-9]{64}\.json$/.test(name));
  assert.ok(name);
  const path = join(directory, name);
  const receipt = JSON.parse(await readFile(path, "utf8"));
  receipt.result.details.continuity.packet.text += "\nCORRUPTED_PERSISTED_MEMORY";
  await writeFile(path, JSON.stringify(receipt));
  f.sessionManager.branch(sourceLeaf);
  await assert.rejects(f.session.compact(), /Compaction cancelled/);
  assert.ok(f.notices.some((notice) => /invalid compaction receipt/.test(notice)));
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, 1, "receipt corruption fails closed instead of silently producing another summary");
  assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, 2);
});

test("documented pinned SDK oversized-last-result preparation failure retains all original context", { skip: !available, timeout: 20000 }, async (t) => {
  const f = await fixture(t);
  f.user("Retain a large final tool result even if the upstream preparation gate cannot compact it.");
  f.assistant([{ type: "toolCall", id: "last-large-result", name: "probe", arguments: {} }], "toolUse");
  f.sessionManager.appendMessage({ role: "toolResult", toolCallId: "last-large-result", toolName: "probe",
    content: [{ type: "text", text: "Large newest synthetic result. ".repeat(800) }], isError: false, timestamp: Date.now() });
  const before = f.sessionManager.getEntries().filter((entry) => entry.type === "message");
  const contextBefore = f.sessionManager.buildSessionContext().messages;
  // The pinned SDK computes its preparation before emitting our extension hook.
  // This assertion records the unsupported edge instead of blessing truncation.
  await assert.rejects(f.session.compact(), /Nothing to compact \(session too small\)/);
  assert.equal(f.requests.length, 0);
  assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, 0);
  assert.deepEqual(f.sessionManager.getEntries().filter((entry) => entry.type === "message"), before);
  assert.deepEqual(f.sessionManager.buildSessionContext().messages, contextBefore);
});
