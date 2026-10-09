// Synthetic compaction streams and editor input only; no model or live terminal.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import * as compaction from "../integrations/pi/qwen-radiance-compaction.mjs";
import { getCompactionProgress } from "../integrations/pi/qwen-radiance-compaction-progress.mjs";

const SDK_ROOT = process.env.QWEN_TEST_PI_ROOT ??
  join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const installed = existsSync(join(SDK_ROOT, "pi-coding-agent/dist/modes/interactive/interactive-mode.js"));
const headings = ["Goal", "Current Authoritative State", "Constraints & Invariants", "Progress", "Measurements & Evidence",
  "Key Decisions", "Rejected / Failed Approaches", "Unresolved Questions & Hypotheses", "Next Steps", "Critical Context"];
const summary = headings.map((heading) => `### ${heading}\nSynthetic verified state.`).join("\n\n");
const partial = "### Goal\nSYNTHETIC_UNFINISHED_DRAFT\nThis checkpoint has not reached its required sections.";
const marker = "COMPACTION_SUMMARY_COMPLETE";
const hash = (value) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
const choice = (text, finish_reason = null) => ({ choices: [{ index: 0, text, finish_reason }] });
const encode = (value) => new TextEncoder().encode(`data: ${typeof value === "string" ? value : JSON.stringify(value)}\n\n`);

function completedResponse(promptTokens, { text = `${summary}\n${marker}`, finishReason = "stop", badUsage = false, trailingData = false } = {}) {
  const frames = [choice(text, finishReason), { usage: { prompt_tokens: promptTokens + Number(badUsage),
    completion_tokens: 300, prompt_tokens_details: { cached_tokens: 2 } } }, "[DONE]"];
  if (trailingData) frames.push(choice("Forbidden synthetic data after DONE."));
  return new Response(frames.map((value) => new TextDecoder().decode(encode(value))).join(""),
    { headers: { "Content-Type": "text/event-stream" } });
}

async function receipts(t) {
  const directory = await mkdtemp(join(tmpdir(), "radiance-finish-test-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

function fixture({ originalComplete = false, originalResponse = {}, streamResponse,
  onOriginalRequest = () => {}, contextWindow = 10000, initialText = partial, initialUsage = true,
  signal = new AbortController().signal, finishControl = compaction.createCompactionFinishControl() } = {}) {
  const payload = { model: "synthetic", messages: [{ role: "user", content: "Synthetic fact A=742." }],
    tools: [{ type: "function", function: { name: "lookup", parameters: { type: "object", properties: {} } } }],
    chat_template_kwargs: { enable_thinking: true, preserve_thinking: true, reasoning_effort: "xhigh" },
    cache_salt: "synthetic-salt", kv_transfer_params: { qwen_chat: { id: "a".repeat(64), generation: "b".repeat(64) } } };
  const preparation = { firstKeptEntryId: "retained", tokensBefore: 60000, isSplitTurn: true,
    settings: { reserveTokens: 16384 }, fileOps: { read: new Set(), written: new Set(), edited: new Set() } };
  const requests = [], instructions = [];
  let controller;
  const fetcher = async (url, options) => {
    const body = JSON.parse(options.body);
    requests.push({ url, body, signal: options.signal });
    if (url.endsWith("/tokenize")) {
      if (body.prompt) return Response.json({ tokens: body.prompt.includes("</think>") ? [4, 6, 7] : [4, 5] });
      if (!body.add_generation_prompt) return Response.json({ tokens: [1, 2] });
      assert.deepEqual(body.messages.slice(0, -1), payload.messages, "canonical history remains unchanged");
      assert.deepEqual(body.tools, payload.tools);
      assert.deepEqual(body.chat_template_kwargs, payload.chat_template_kwargs);
      instructions.push(body.messages.at(-1).content);
      return Response.json({ tokens: [1, 2, 3, 4, 5] });
    }
    assert.equal(url, "http://fixture.invalid/v1/completions");
    assert.deepEqual(body.prompt.slice(0, 2), [1, 2], "canonical history remains an exact prefix");
    assert.equal(body.cache_salt, payload.cache_salt);
    assert.deepEqual(body.kv_transfer_params.qwen_chat, payload.kv_transfer_params.qwen_chat);
    assert.equal(body.kv_transfer_params.qwen_snapshot_force_flush, true);
    assert.equal(requests.filter((r) => r.url.endsWith("/completions")).length, 1,
      "Alt+C must never start a replacement generation request");
    onOriginalRequest(options.signal, body);
    if (streamResponse) return streamResponse(body, options.signal);
    if (originalComplete) return completedResponse(body.prompt.length, originalResponse);
    return new Response(new ReadableStream({ start(value) {
      controller = value;
      if (initialText !== undefined) controller.enqueue(encode({ ...choice(initialText),
        ...(initialUsage ? { usage: typeof initialUsage === "object" ? initialUsage : {
          prompt_tokens: body.prompt.length, completion_tokens: 100, prompt_tokens_details: { cached_tokens: 2 } } } : {}) }));
      options.signal.addEventListener("abort", () => controller.error(options.signal.reason), { once: true });
    } }), { headers: { "Content-Type": "text/event-stream" } });
  };
  return { input: { payload, preparation, model: { baseUrl: "http://fixture.invalid/v1", maxTokens: 32768, contextWindow },
    policy: "Synthetic checkpoint policy.", customInstructions: "Preserve synthetic fact A=742.", headers: {},
    sourceIdentity: { sessionFile: "synthetic-session", leaf: "synthetic-leaf" }, signal, finishControl, fetcher },
  requests, instructions, emit: (value) => controller.enqueue(encode(value)),
  generations: () => requests.filter((request) => request.url.endsWith("/completions")) };
}

function cutOnText(f, options = {}) {
  return compaction.compactFromPayload({ ...f.input, ...options, progress: (update) => {
    if (update.phase === "generate" && update.characters > 0 && !f.input.finishControl.requested) {
      assert.equal(f.input.finishControl.requestFinish(), true);
    }
    options.progress?.(update);
  } });
}

test("finish control requires checkpoint text, accepts one cutoff and becomes inactive when closed", () => {
  const finish = compaction.createCompactionFinishControl();
  assert.equal(finish.requested, false);
  assert.equal(finish.available, false);
  assert.equal(finish.requestFinish(), false, "Alt+C cannot finish an empty checkpoint");
  assert.equal(finish.signal.aborted, false);
  finish.setAvailable(true);
  assert.equal(finish.available, true);
  assert.equal(finish.requestFinish(), true);
  assert.equal(finish.requested, true);
  assert.equal(finish.signal.aborted, true);
  assert.equal(finish.requestFinish(), false);
  finish.close(); finish.close();
  assert.equal(finish.requestFinish(), false);
  const unused = compaction.createCompactionFinishControl();
  unused.setAvailable(true); unused.close();
  assert.equal(unused.available, false);
  assert.equal(unused.requestFinish(), false, "a stale shortcut cannot finish a later operation");
  assert.equal(unused.signal.aborted, false);
});

test("Alt+C cuts the current checkpoint exactly and issues no replacement request", { timeout: 10000 }, async (t) => {
  const directory = await receipts(t), f = fixture();
  const before = structuredClone(f.input.payload), updates = [];
  const result = await cutOnText(f, { receiptDirectory: directory, progress: (update) => updates.push(update) });
  assert.equal(f.generations().length, 1);
  assert.equal(f.requests.length, 5, "four original tokenizer calls and one original completion only");
  assert.equal(f.instructions.length, 1);
  assert.equal(f.generations()[0].signal.aborted, true);
  assert.equal(f.input.signal.aborted, false, "Alt+C must not cancel the compaction transaction");
  assert.equal(result.summary, partial, "no rewriting, marker synthesis or replacement summary");
  assert.equal(result.firstKeptEntryId, "retained");
  assert.equal(result.usage.input + result.usage.cacheRead, f.generations()[0].body.prompt.length);
  assert.equal(result.usage.output, 100);
  assert.equal(result.details.forcedCheckpoint, true);
  assert.equal(result.details.finishRequested, true);
  assert.equal(result.details.checkpointComplete, false);
  assert.equal(result.details.summaryRequests, 1);
  assert.equal(result.details.streamCompleted, false);
  assert.equal(result.details.streamFinishReason, undefined, "never manufacture a normal stop reason");
  assert.equal(result.details.tokenAccounting, "observed_at_cutoff");
  assert.equal(result.details.promptSha256, hash(f.generations()[0].body.prompt));
  assert.deepEqual(f.input.payload, before);
  assert.equal(updates.find((u) => u.phase === "finalize").forcedCheckpoint, true);
  const saved = await compaction.readReceipt(directory, compaction.compactionReceiptKey(f.input));
  assert.equal(saved.summary, partial);
  assert.equal((await readdir(directory)).filter((name) => name.endsWith(".json")).length, 1,
    "the selected cutoff is the receipt; no abandoned replacement or interrupted draft artifact");
  const resumed = fixture();
  const recovered = await compaction.compactFromPayload({ ...resumed.input, receiptDirectory: directory,
    fetcher: () => assert.fail("the selected cutoff is reused without inference or tokenization") });
  assert.equal(recovered.summary, partial);
  assert.equal(recovered.details.receiptReused, true);
});

for (const text of ["### Goal\nunfinished sente", "```python\ndef unfinished(", "  literal leading and trailing whitespace  \n"]) {
  test(`Alt+C preserves incomplete checkpoint text exactly: ${JSON.stringify(text)}`, async () => {
    const f = fixture({ initialText: text });
    assert.equal((await cutOnText(f)).summary, text);
    assert.equal(f.generations().length, 1);
  });
}

test("Alt+C before generation and during whitespace is a no-op, then becomes available for real text", async () => {
  const f = fixture({ initialText: "   \n" });
  assert.equal(f.input.finishControl.requestFinish(), false);
  let sawWhitespace = false;
  const result = await compaction.compactFromPayload({ ...f.input, progress: (update) => {
    if (update.phase === "submit" || update.phase === "wait") assert.equal(f.input.finishControl.requestFinish(), false);
    if (update.phase === "generate" && !sawWhitespace) {
      sawWhitespace = true;
      assert.equal(f.input.finishControl.available, false);
      assert.equal(f.input.finishControl.requestFinish(), false);
      f.emit(choice(partial));
    } else if (update.phase === "generate") assert.equal(f.input.finishControl.requestFinish(), true);
  } });
  assert.equal(sawWhitespace, true);
  assert.equal(result.summary, "   \n" + partial);
  assert.equal(f.generations().length, 1);
});

test("Alt+C does not consume later deltas buffered in the same network chunk", async () => {
  const f = fixture({ streamResponse: (body) => new Response([
    { ...choice(partial), usage: { prompt_tokens: body.prompt.length, completion_tokens: 100 } },
    choice("MUST NOT APPEAR"), choice("", "stop"), "[DONE]",
  ].map((frame) => new TextDecoder().decode(encode(frame))).join(""),
  { headers: { "Content-Type": "text/event-stream" } }) });
  const result = await cutOnText(f);
  assert.equal(result.summary, partial);
  assert.equal(result.details.streamCompleted, false);
  assert.equal(result.details.streamFinishReason, undefined);
});

test("Alt+C works before the provider has sent usage and explicitly records unknown token accounting", async (t) => {
  const directory = await receipts(t), f = fixture({ initialUsage: false });
  const result = await cutOnText(f, { receiptDirectory: directory });
  assert.equal(result.summary, partial);
  assert.equal(result.usage, undefined);
  assert.equal(result.details.outputTokenCountKnown, false);
  assert.equal(result.details.usageAuthenticated, false);
  assert.equal(result.details.tokenAccounting, "unavailable");
  assert.equal(result.details.inputTokens, f.generations()[0].body.prompt.length);
  const retry = fixture();
  const recovered = await compaction.compactFromPayload({ ...retry.input, receiptDirectory: directory,
    fetcher: () => assert.fail("no replacement request for a forced cutoff without usage") });
  assert.equal(recovered.summary, partial);
  assert.equal(recovered.usage, undefined);
});

test("Alt+C while awaiting a stream read does not parse an unfinished network frame", async () => {
  const f = fixture({ streamResponse: () => new Response(new ReadableStream({ start(controller) {
    controller.enqueue(new TextEncoder().encode(new TextDecoder().decode(encode(choice(partial))) +
      'data: {"choices":['));
  } }), { headers: { "Content-Type": "text/event-stream" } }) });
  const pending = compaction.compactFromPayload({ ...f.input, progress: (update) => {
    if (update.phase === "generate") setImmediate(() => f.input.finishControl.requestFinish());
  } });
  const result = await pending;
  assert.equal(result.summary, partial);
  assert.equal(result.details.forcedCheckpoint, true);
  assert.equal(f.generations().length, 1);
});

test("Alt+C during stream cleanup still selects the current text without a replacement", async () => {
  const closing = Promise.withResolvers(), closed = Promise.withResolvers();
  const f = fixture({ streamResponse: (body) => {
    const frames = [encode({ ...choice(partial, "length"),
      usage: { prompt_tokens: body.prompt.length, completion_tokens: 100 } }), encode("[DONE]")];
    let index = 0;
    return { headers: new Headers({ "Content-Type": "text/event-stream" }), body: { getReader: () => ({
      read: async () => index < frames.length ? { done: false, value: frames[index++] } : { done: true },
      cancel: async () => { closing.resolve(); await closed.promise; },
      releaseLock() {},
    }) }, ok: true };
  } });
  const pending = compaction.compactFromPayload(f.input);
  await closing.promise;
  assert.equal(f.input.finishControl.requestFinish(), true);
  closed.resolve();
  const result = await pending;
  assert.equal(result.summary, partial);
  assert.equal(result.details.forcedCheckpoint, true);
  assert.equal(result.details.streamFinishReason, "length");
  assert.equal(f.generations().length, 1);
});

for (const [label, initialUsage] of [
  ["wrong prompt length", { prompt_tokens: 7, completion_tokens: 100 }],
  ["fractional output count", { prompt_tokens: 6, completion_tokens: 1.5 }],
  ["negative cache count", { prompt_tokens: 6, completion_tokens: 100, prompt_tokens_details: { cached_tokens: -1 } }],
]) {
  test(`forced cutoff does not accept malformed accounting: ${label}`, async (t) => {
    const directory = await receipts(t), f = fixture({ initialUsage });
    await assert.rejects(cutOnText(f, { receiptDirectory: directory }), /usage|accounting|prompt length/);
    assert.equal(f.generations().length, 1);
    assert.equal(await compaction.readReceipt(directory, compaction.compactionReceiptKey(f.input)), undefined);
  });
}

for (const cancelPhase of ["generate", "finalize", "validate"]) {
  test(`Escape cancels a forced cutoff at ${cancelPhase}, retaining the original conversation`, async (t) => {
    const cancel = new AbortController(), f = fixture({ signal: cancel.signal }), directory = await receipts(t);
    await assert.rejects(cutOnText(f, { receiptDirectory: directory, progress: (update) => {
      if (update.phase === cancelPhase) cancel.abort();
    } }), /abort|cancel/i);
    assert.equal(f.generations().length, 1);
    assert.equal(await compaction.readReceipt(directory, compaction.compactionReceiptKey(f.input)), undefined);
  });
}

test("Escape before checkpoint generation makes no inference request", async () => {
  const cancel = new AbortController(), f = fixture({ signal: cancel.signal });
  cancel.abort();
  await assert.rejects(compaction.compactFromPayload(f.input), /abort/i);
  assert.equal(f.requests.length, 0);
});

for (const [label, originalResponse, expected] of [
  ["missing completion marker", { text: summary }, /marker/],
  ["length stop", { finishReason: "length" }, /incomplete/],
  ["unauthenticated prompt usage", { badUsage: true }, /prompt length|usage/],
  ["data after DONE", { trailingData: true }, /data after DONE/],
]) {
  test(`without Alt+C normal compaction still rejects ${label}`, async (t) => {
    const directory = await receipts(t), f = fixture({ originalComplete: true, originalResponse });
    await assert.rejects(compaction.compactFromPayload({ ...f.input, receiptDirectory: directory }), expected);
    assert.equal(f.generations().length, 1);
    assert.equal(await compaction.readReceipt(directory, compaction.compactionReceiptKey(f.input)), undefined);
  });
}

test("without Alt+C a disconnected stream with partial text cannot commit", async () => {
  const f = fixture({ streamResponse: () => new Response(new TextDecoder().decode(encode(choice(partial))),
    { headers: { "Content-Type": "text/event-stream" } }) });
  await assert.rejects(compaction.compactFromPayload(f.input), /finish_reason\/DONE/);
  assert.equal(f.generations().length, 1);
});

test("a length finish reason observed at the explicit cutoff remains length, never synthetic stop", async () => {
  const f = fixture({ streamResponse: (body) => completedResponse(body.prompt.length, { text: partial, finishReason: "length" }) });
  const result = await cutOnText(f);
  assert.equal(result.summary, partial);
  assert.equal(result.details.streamFinishReason, "length");
  assert.equal(result.details.checkpointComplete, false);
});

test("forced receipt requires both explicit flags and a valid result hash", async (t) => {
  const directory = await receipts(t), f = fixture();
  await cutOnText(f, { receiptDirectory: directory });
  const key = compaction.compactionReceiptKey(f.input), path = join(directory, `${key}.json`);
  const receipt = JSON.parse(await readFile(path, "utf8"));
  for (const mutate of [
    (r) => { delete r.userForced; },
    (r) => { delete r.result.details.forcedCheckpoint; },
    (r) => { r.result.details.checkpointComplete = true; },
  ]) {
    const altered = structuredClone(receipt); mutate(altered); altered.resultHash = hash(altered.result);
    await compaction.writeReceipt(directory, key, altered);
    await assert.rejects(compaction.readReceipt(directory, key), /invalid user-forced/);
  }
  const altered = structuredClone(receipt); altered.result.summary += "changed";
  await compaction.writeReceipt(directory, key, altered);
  await assert.rejects(compaction.readReceipt(directory, key), /invalid compaction receipt/);
});

test("the extension Alt+C shortcut finishes only its active session and leaf and releases the control afterward", {
  timeout: 10000,
}, async (t) => {
  const directory = await receipts(t), previousDir = process.env.PI_CODING_AGENT_DIR;
  const previousAbi = process.env.QWEN_RADIANCE_CACHE_ABI;
  process.env.PI_CODING_AGENT_DIR = directory;
  delete process.env.QWEN_RADIANCE_CACHE_ABI;
  t.after(() => {
    if (previousDir === undefined) delete process.env.PI_CODING_AGENT_DIR; else process.env.PI_CODING_AGENT_DIR = previousDir;
    if (previousAbi === undefined) delete process.env.QWEN_RADIANCE_CACHE_ABI; else process.env.QWEN_RADIANCE_CACHE_ABI = previousAbi;
  });
  const started = Promise.withResolvers(), f = fixture({ onOriginalRequest: () => started.resolve() });
  t.mock.method(globalThis, "fetch", f.input.fetcher);
  const handlers = new Map(), shortcuts = new Map(), notices = [], messages = [];
  const pi = { on: (name, handler) => handlers.set(name, handler), registerShortcut: (key, options) => shortcuts.set(key, options),
    getActiveTools: () => [], getAllTools: () => [], getThinkingLevel: () => "xhigh" };
  const sessionManager = { getSessionFile: () => join(directory, "synthetic-session.jsonl"), getLeafId: () => "synthetic-leaf",
    getSessionId: () => "synthetic-ui", getCwd: () => "/synthetic", getEntries: () => [], getBranch: () => [],
    buildSessionContext: () => ({ messages: f.input.payload.messages }) };
  const ctx = { mode: "tui", model: { ...f.input.model, id: compaction.MODEL, api: "openai-completions", provider: "qwen-r9700" },
    sessionManager, modelRegistry: { getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "fixture" }) },
    getSystemPrompt: () => "Synthetic system prompt.", ui: { notify: (message) => notices.push(message),
      setWidget() {}, setStatus() {}, setWorkingMessage: (message) => messages.push(message) } };
  compaction.installRadianceCompaction(pi, { convertToLlm: (value) => value,
    flushSnapshotTail: () => assert.fail("this synthetic test has no disk snapshot backend"),
    streamSimpleOpenAICompletions: (_model, _context, options) => ({ result: async () => {
      try { options.onPayload(f.input.payload); }
      catch (error) { return { errorMessage: error.message }; }
      assert.fail("canonical payload capture must stop before any SDK fetch");
    } }),
  });
  const shortcut = shortcuts.get("alt+c");
  assert.ok(shortcut, "the extension registers Alt+C through Pi's supported shortcut API");
  assert.equal(shortcuts.has("f8"), false, "finish no longer requires or registers a function key");
  assert.match(shortcut.description, /finish|checkpoint|compact/i);
  await shortcut.handler(ctx);
  assert.equal(f.generations().length, 0, "Alt+C with no active compaction is a no-op");
  const cancellation = new AbortController();
  const running = handlers.get("session_before_compact")({ preparation: f.input.preparation, signal: cancellation.signal }, ctx);
  try {
    await started.promise;
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(getCompactionProgress(ctx).snapshot().phase, "generate");
    await shortcut.handler({ ...ctx, sessionManager: { ...sessionManager, getSessionFile: () => join(directory, "another-session.jsonl") } });
    await shortcut.handler({ ...ctx, sessionManager: { ...sessionManager, getLeafId: () => "another-leaf" } });
    assert.equal(f.generations()[0].signal.aborted, false, "another session or branch cannot finish the current checkpoint");
    const firstPress = shortcut.handler(ctx), repeatedPress = shortcut.handler(ctx);
    await Promise.all([firstPress, repeatedPress]);
    const response = await running;
    assert.ok(response.compaction, JSON.stringify(notices));
    assert.equal(response.compaction.details.finishRequested, true);
    assert.equal(response.compaction.details.forcedCheckpoint, true);
    assert.equal(response.compaction.summary, partial);
    assert.equal(f.generations().length, 1);
    assert.equal(cancellation.signal.aborted, false);
    assert.ok(messages.some((message) => /Alt\+C.*finish|finish.*requested/i.test(message ?? "")), "the working hint explains finish behavior");
    await shortcut.handler(ctx);
    assert.equal(f.generations().length, 1, "the finally block releases the completed operation's shortcut control");
  } finally {
    cancellation.abort();
    await running;
    await handlers.get("session_shutdown")({}, ctx);
  }
});

test("native Pi delivers Alt+C during compaction without changing the draft or Escape cancellation", {
  skip: !installed,
}, async () => {
  const load = (path) => import(pathToFileURL(join(SDK_ROOT, path)));
  const { InteractiveMode } = await load("pi-coding-agent/dist/modes/interactive/interactive-mode.js");
  const { CustomEditor } = await load("pi-coding-agent/dist/modes/interactive/components/custom-editor.js");
  const { KeybindingsManager } = await load("pi-coding-agent/dist/core/keybindings.js");
  const { IdleStatus } = await load("pi-coding-agent/dist/modes/interactive/components/status-indicator.js");
  const { initTheme, getEditorTheme } = await load("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { Container, TuiMainScreen } = await load("pi-tui/dist/index.js");
  initTheme("dark", false);
  const terminal = { columns: 100, rows: 24, write() {}, hideCursor() {}, showCursor() {}, stop() {} };
  const ui = new TuiMainScreen(terminal, false), keybindings = new KeybindingsManager();
  const editor = new CustomEditor(ui, getEditorTheme(), keybindings);
  let finishRequests = 0, cancellations = 0;
  const sessionManager = { getCwd: () => "/synthetic", buildContextEntries: () => [] };
  const session = { isCompacting: true, isStreaming: false, isIdle: false, model: {}, sessionManager,
    agent: { signal: new AbortController().signal },
    settingsManager: { getShowTerminalProgress: () => false, isProjectTrusted: () => true },
    abortCompaction: () => { cancellations++; } };
  const mode = Object.assign(Object.create(InteractiveMode.prototype), {
    isInitialized: true, options: { tuiMode: "regular" }, ui, runtimeHost: { session }, keybindings, footer: { invalidate() {} },
    editor, defaultEditor: editor, editorContainer: new Container(), statusContainer: new Container(),
    chatContainer: new Container(), pendingTools: new Map(), idleStatus: new IdleStatus(), compactionQueuedMessages: [],
    createExtensionUIContext: () => ({}), showError: (message) => assert.fail(message),
  });
  const shortcut = { handler: (ctx) => {
    assert.equal(ctx.sessionManager, sessionManager);
    assert.equal(ctx.isIdle(), false);
    finishRequests++;
  } };
  mode.setupExtensionShortcuts({ getModelRegistry: () => ({}), getShortcuts: () => new Map([["alt+c", shortcut]]) });
  mode.editorContainer.addChild(editor);
  ui.addChild(mode.chatContainer); ui.addChild(mode.statusContainer); ui.addChild(mode.editorContainer); ui.setFocus(editor);
  try {
    editor.setText("Synthetic unsent draft.\nKeep the cursor and draft intact.");
    const before = { text: editor.getText(), state: structuredClone(editor.state) };
    await mode.handleEvent({ type: "compaction_start", reason: "manual" });
    // Termux ALT + c uses the ordinary ESC-prefixed letter, without CSI-u or
    // any function-key sequence. Drive the actual editor shortcut routing.
    editor.handleInput("\x1bc");
    await Promise.resolve();
    assert.equal(finishRequests, 1);
    assert.equal(cancellations, 0);
    assert.deepEqual({ text: editor.getText(), state: structuredClone(editor.state) }, before);
    editor.handleInput("\x1b");
    assert.equal(cancellations, 1, "Escape still invokes the native compaction abort handler");
    assert.equal(finishRequests, 1);
    assert.deepEqual({ text: editor.getText(), state: structuredClone(editor.state) }, before);
  } finally {
    mode.clearStatusIndicator();
    ui.stop({ preserveScreen: true });
  }
});
