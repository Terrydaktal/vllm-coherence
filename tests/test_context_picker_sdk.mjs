// Real installed Pi SDK and provider conversion, with synthetic sessions/SSE.
// No user transcript is opened and no request reaches a GPU or real endpoint.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const selectionType = "qwen-context-selection-v1";
test("installed picker persists context exclusions through resume/compaction, supports undo and protects tool pairs", {
  skip: !existsSync(join(root, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000,
}, async (t) => {
  const mod = (name) => import(pathToFileURL(join(root, name)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { streamSimple } = await mod("pi-ai/dist/api/openai-completions.js");
  const { matchesKey } = await mod("pi-tui/dist/index.js");
  const temporary = await mkdtemp(join(tmpdir(), "context-picker-sdk-"));
  const previous = Object.fromEntries(["PI_CODING_AGENT_DIR", "QWEN_RADIANCE_CACHE_ABI", "PI_OFFLINE"].map((key) => [key, process.env[key]]));
  process.env.PI_CODING_AGENT_DIR = temporary;
  process.env.PI_OFFLINE = "1";
  delete process.env.QWEN_RADIANCE_CACHE_ABI;
  t.after(async () => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
    await rm(temporary, { recursive: true, force: true });
  });
  const model = { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", name: "Synthetic Radiance",
    provider: "qwen-r9700", api: "openai-completions", baseUrl: "http://fixture.invalid/v1", reasoning: true,
    input: ["text"], contextWindow: 253792, maxTokens: 253792,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    compat: { thinkingFormat: "chat-template", chatTemplateKwargs: { enable_thinking: true, preserve_thinking: true } } };
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false, reserveTokens: 8192, keepRecentTokens: 10 }, retry: { enabled: false } }, { projectTrusted: true });
  const headings = ["Goal", "Current Authoritative State", "Constraints & Invariants", "Progress", "Measurements & Evidence", "Key Decisions",
    "Rejected / Failed Approaches", "Unresolved Questions & Hypotheses", "Next Steps", "Critical Context"];
  const checkpoint = headings.map((heading) => `### ${heading}\nSynthetic checkpoint.`).join("\n\n");
  const requests = [], tokenizations = [], notices = [];
  const sse = (frames) => new Response(frames.concat("[DONE]").map((frame) =>
    `data: ${typeof frame === "string" ? frame : JSON.stringify(frame)}\n\n`).join(""), { headers: { "Content-Type": "text/event-stream" } });
  const fetcher = async (url, options) => {
    assert.ok(String(url).startsWith("http://fixture.invalid/"), "no real network destination is allowed");
    const body = JSON.parse(options.body);
    if (String(url).endsWith("/tokenize")) {
      tokenizations.push(body);
      return Response.json({ tokens: body.prompt ? (body.prompt.includes("</think>") ? [4, 6, 7] : [4, 5]) :
        body.add_generation_prompt ? [1, 2, 3, 4, 5] : [1, 2] });
    }
    if (String(url).endsWith("/v1/completions")) return sse([
      { choices: [{ index: 0, text: checkpoint + "\nCOMPACTION_SUMMARY_COMPLETE", finish_reason: "stop" }] }, { usage: { prompt_tokens: 6, completion_tokens: 300 } },
    ]);
    assert.equal(String(url), "http://fixture.invalid/v1/chat/completions");
    requests.push(body);
    return sse([{ choices: [{ index: 0, delta: { role: "assistant", reasoning_content: "FUTURE_THINKING", content: "New answer" }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }, { choices: [], usage: { prompt_tokens: 50, completion_tokens: 20, total_tokens: 70 } }]);
  };
  t.mock.method(globalThis, "fetch", fetcher);
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: (selected, context, options) => streamSimple(selected, context, { ...options, apiKey: "fixture", fetch: fetcher }) };
  const usage = { input: 50, output: 20, cacheRead: 0, cacheWrite: 0, totalTokens: 70,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const original = SessionManager.create(temporary, join(temporary, "sessions"));
  original.appendMessage({ role: "user", content: "EXCLUDED_USER", timestamp: 1 });
  const callId = original.appendMessage({ role: "assistant", api: model.api, provider: model.provider, model: model.id,
    timestamp: 2, stopReason: "toolUse", usage, content: [{ type: "thinking", thinking: "EXCLUDED_THINKING", thinkingSignature: "reasoning_content" },
      { type: "text", text: "EXCLUDED_PROSE" }, { type: "toolCall", id: "old-call", name: "lookup", arguments: { value: "EXCLUDED_ARGUMENT" } }] });
  const resultId = original.appendMessage({ role: "toolResult", toolCallId: "old-call", toolName: "lookup",
    content: [{ type: "text", text: "EXCLUDED_RESULT" }], timestamp: 3, isError: false });
  original.appendMessage({ role: "user", content: "KEPT_USER", timestamp: 4 });
  const keptId = original.appendMessage({ role: "assistant", api: model.api, provider: model.provider, model: model.id,
    timestamp: 5, stopReason: "stop", usage, content: [{ type: "thinking", thinking: "HIDDEN_THINKING_ONLY", thinkingSignature: "reasoning_content" },
      { type: "text", text: "KEPT_PROSE" }] });
  const create = async (manager) => {
    const resourceLoader = new DefaultResourceLoader({ cwd: temporary, agentDir: temporary, settingsManager,
      noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
      additionalExtensionPaths: [resolve("integrations/pi/qwen-radiance-compaction.ts")], systemPrompt: "Synthetic context picker fixture." });
    await resourceLoader.reload();
    assert.deepEqual(resourceLoader.getExtensions().errors, []);
    const { session } = await createAgentSession({ cwd: temporary, agentDir: temporary, model, modelRuntime, resourceLoader,
      sessionManager: manager, settingsManager, tools: [], customTools: [] });
    await session.bindExtensions({ mode: "tui", uiContext: { notify: (value) => notices.push(value), setWidget: () => {}, setStatus: () => {}, setWorkingMessage: () => {},
      custom: async (factory) => {
        let outcome;
        const component = await factory({ terminal: { rows: 40 }, requestRender: () => {} }, { fg: (_name, value) => value }, {}, (value) => { outcome = value; });
        assert.ok(component.render(120).join("\n").includes("/context"));
        component.handleInput("\x1b[H"); // Home
        component.handleInput("t"); component.handleInput("e"); component.handleInput("c");
        component.handleInput("\x1b[F"); // End
        component.handleInput("h");
        component.handleInput("p");
        const deadline = Date.now() + 2000;
        while (component.previewAbort && Date.now() < deadline) await new Promise(setImmediate);
        assert.ok(component.previewResult, component.notice);
        assert.ok(component.render(120).join("\n").includes("Exact prompt:"));
        component.handleInput("\r"); component.dispose(); return outcome;
      } } });
    assert.ok(matchesKey("\x1b[H", "home") && matchesKey("\x1b[F", "end"));
    return session;
  };
  let session = await create(original), manager = original;
  const close = async () => { await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" }); session.dispose(); };
  try {
    await session.prompt("/context");
    assert.equal(requests.length, 0, "opening/saving the picker cannot generate a model request");
    const selection = original.getEntries().find((entry) => entry.customType === selectionType);
    assert.ok(selection.data.changes.some((c) => c.entryId === callId && c.part === "message" && c.excluded));
    assert.ok(selection.data.changes.some((c) => c.entryId === resultId && c.part === "message" && c.excluded));
    assert.ok(selection.data.changes.some((c) => c.entryId === keptId && c.part === "thinking" && c.excluded));
    assert.doesNotMatch(JSON.stringify(selection.data), /EXCLUDED_|HIDDEN_THINKING/);
    await session.prompt("Synthetic continuation.");
    assert.equal(requests.length, 1, notices.join("\n"));
    assert.doesNotMatch(JSON.stringify(requests[0].messages), /EXCLUDED_|HIDDEN_THINKING_ONLY|qwen-context-selection/);
    assert.match(JSON.stringify(requests[0].messages), /KEPT_USER|KEPT_PROSE/);
    const path = original.getSessionFile();
    const originalLines = (await readFile(path, "utf8")).trim().split("\n");
    const callLine = originalLines.find((line) => JSON.parse(line).id === callId);
    assert.match(callLine, /EXCLUDED_THINKING/);
    await close(); manager = SessionManager.open(path); session = await create(manager);
    await session.prompt("After resume.");
    assert.doesNotMatch(JSON.stringify(requests[1].messages), /EXCLUDED_|HIDDEN_THINKING_ONLY/);
    assert.match(JSON.stringify(requests[1].messages), /FUTURE_THINKING/);
    await session.prompt("/context undo");
    await session.prompt("After undo.");
    assert.match(JSON.stringify(requests[2].messages), /EXCLUDED_USER|EXCLUDED_RESULT|HIDDEN_THINKING_ONLY/);
    await session.prompt("/context");
    const prior = tokenizations.length;
    await session.compact();
    const checkpointRequests = tokenizations.slice(prior).filter((body) => body.messages);
    assert.ok(checkpointRequests.length, notices.join("\n"));
    for (const body of checkpointRequests) assert.doesNotMatch(JSON.stringify(body.messages), /EXCLUDED_/);
    assert.ok(manager.getEntries().some((entry) => entry.type === "compaction"));
    assert.equal((await readFile(path, "utf8")).split("\n").find((line) => line && JSON.parse(line).id === callId), callLine);
    await session.prompt("/context status"); assert.match(notices.at(-1), /Saved transcript retained/);
    // Resume a real saved session after Escape interrupted a streamed call.
    // The provider retains interrupted prose, omits failures, and never replays
    // unexecuted tools; selection/purge validation must not invent results.
    for (const stopReason of ["aborted", "error"]) {
      await close();
      manager.appendMessage({ role: "assistant", api: model.api, provider: model.provider, model: model.id,
        timestamp: Date.now(), stopReason, usage,
        content: [{ type: "text", text: `${stopReason}_partial_output` },
          { type: "toolCall", id: `unfinished-${stopReason}`, name: "lookup", arguments: {} }] });
      session = await create(manager);
      if (stopReason === "error") {
        const prior = tokenizations.length;
        await session.compact();
        const checkpointRequests = tokenizations.slice(prior).filter((body) => body.messages);
        assert.ok(checkpointRequests.length, notices.join("\n"));
        for (const body of checkpointRequests) {
          assert.doesNotMatch(JSON.stringify(body.messages), /error_partial_output|unfinished-|No result provided/);
        }
      }
      const before = requests.length;
      await session.prompt("After interrupted synthetic tool emission.");
      assert.equal(requests.length, before + 1, notices.join("\n"));
      assert.equal(session.messages.at(-1).stopReason, "stop");
      const payload = JSON.stringify(requests.at(-1).messages);
      assert.doesNotMatch(payload, /error_partial_output|unfinished-|No result provided/);
      if (stopReason === "aborted") assert.match(payload, /aborted_partial_output/);
      assert.ok(manager.getEntries().some((e) => e.type === "message" && e.message.stopReason === stopReason));
    }
    // A corrupt saved policy must not fall through the SDK's swallowed hook
    // exception and make an HTTP inference request.
    const count = requests.length;
    manager.appendCustomEntry(selectionType, { version: 1, changes: null, preserveFutureThinking: false });
    await session.prompt("Blocked synthetic request.");
    assert.equal(requests.length, count);
    assert.equal(session.messages.at(-1).stopReason, "error");
  } finally { await close(); }
});

test("real Pi overlay restores keyboard focus and leaves a multiline input draft intact on save/cancel", {
  skip: !existsSync(join(root, "pi-coding-agent/dist/modes/interactive/interactive-mode.js")),
}, async () => {
  const mod = (name) => import(pathToFileURL(join(root, name)));
  const { InteractiveMode } = await mod("pi-coding-agent/dist/modes/interactive/interactive-mode.js");
  const { CustomEditor } = await mod("pi-coding-agent/dist/modes/interactive/components/custom-editor.js");
  const { KeybindingsManager } = await mod("pi-coding-agent/dist/core/keybindings.js");
  const { initTheme, getEditorTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { Container, TuiMainScreen, matchesKey, truncateToWidth } = await mod("pi-tui/dist/index.js");
  const { ContextPicker } = await import("../integrations/pi/qwen-context.mjs");
  initTheme("dark", false);
  const writes = [], terminal = { columns: 90, rows: 28, write: (value) => writes.push(value), hideCursor() {}, showCursor() {}, stop() {} };
  const ui = new TuiMainScreen(terminal, false), editor = new CustomEditor(ui, getEditorTheme(), new KeybindingsManager());
  const mode = Object.assign(Object.create(InteractiveMode.prototype), { ui, editor, editorContainer: new Container() });
  mode.editorContainer.addChild(editor); ui.addChild(mode.editorContainer); ui.setFocus(editor);
  const entry = { id: "assistant", type: "message", index: 0, turn: 1,
    message: { role: "assistant", timestamp: 1, content: [{ type: "thinking", thinking: "Synthetic thought" }, { type: "text", text: "Synthetic prose" }] } };
  const ctx = { sessionManager: { buildContextEntries: () => [entry], getBranch: () => [] } };
  const policy = { active: false, excluded: new Set(), thinking: new Set(), preserveFutureThinking: false };
  try {
    for (const save of [false, true]) {
      editor.setText("Unsent multiline draft\nSecond line");
      const before = { state: structuredClone(editor.state), undo: structuredClone(editor.undoStack), text: editor.getText() };
      const outcome = mode.showExtensionCustom((tui, theme, _keybindings, done) => new ContextPicker({ rows: [entry], policy,
        messages: [entry.message], ctx, tui, theme, done, matchesKey, truncateToWidth, preview: async () => { throw new Error("not requested"); } }),
      { overlay: true, overlayOptions: { width: "95%", maxHeight: "95%", anchor: "center" } });
      await new Promise(setImmediate);
      ui.renderNow(); ui.handleTerminalInput("h"); ui.renderNow();
      ui.handleTerminalInput(save ? "\r" : "\x1b");
      const result = await outcome;
      assert.equal(Boolean(result?.thinking.has("assistant")), save);
      assert.equal(editor.getText(), before.text);
      assert.deepEqual(editor.state, before.state);
      assert.deepEqual(structuredClone(editor.undoStack), before.undo);
      ui.handleTerminalInput(" more");
      assert.match(editor.getText(), / more/, "the overlay must restore keyboard focus to the original editor");
    }
  } finally { ui.stop({ preserveScreen: true }); }
});
