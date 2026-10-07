// Real pinned Pi runtime, synthetic session/stream only; no backend requests.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const available = existsSync(join(root, "pi-coding-agent/dist/core/agent-session.js"));
const mod = path => import(pathToFileURL(join(root, path)));
const model = { id: "synthetic-context", name: "Synthetic context", api: "openai-completions", provider: "qwen-r9700",
  baseUrl: "http://fixture.invalid/v1", reasoning: true, input: ["text"], contextWindow: 253_792, maxTokens: 100,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
const usage = (input, output, cacheRead = 0, cacheWrite = 0) => ({ input, output, cacheRead, cacheWrite,
  totalTokens: input + output + cacheRead + cacheWrite, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } });
const assistant = (values) => ({ role: "assistant", model: model.id, provider: model.provider, api: model.api,
  content: [{ type: "text", text: "Synthetic answer." }], timestamp: Date.now(), stopReason: "stop", usage: values });

test("post-compaction context and footer use measured streaming usage before persistence", { skip: !available }, async () => {
  const { AgentSession } = await mod("pi-coding-agent/dist/core/agent-session.js");
  const { FooterComponent } = await mod("pi-coding-agent/dist/modes/interactive/components/footer.js");
  const { initTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { stripTerminalSequences } = await mod("pi-tui/dist/index.js");
  initTheme("dark", false);
  const old = assistant(usage(200_000, 200));
  const branch = [{ type: "message", message: old }, { type: "compaction", timestamp: new Date().toISOString() }];
  const state = { isStreaming: true, streamingMessage: assistant(usage(0, 0)), model, thinkingLevel: "xhigh" };
  const session = { model, state, agent: { state }, messages: [old], modelRuntime: { isUsingSubscription: () => false },
    sessionManager: { getBranch: () => branch, getEntries: () => branch, getCwd: () => "/synthetic", getSessionName: () => "Synthetic" } };
  session.getContextUsage = () => AgentSession.prototype.getContextUsage.call(session);
  const footer = new FooterComponent(session, { getGitBranch: () => "main", getExtensionStatuses: () => new Map(), getAvailableProviderCount: () => 1 });
  footer.setAutoCompactEnabled(true);
  const rendered = () => footer.render(1000).map(stripTerminalSequences).join(" ");
  assert.equal(session.getContextUsage().tokens, null, "old usage must remain invalidated");
  assert.match(rendered(), /\? \/ 253,792 \(\?%\)/);
  for (const output of [0, 160, 6866]) {
    state.streamingMessage = assistant(usage(38_000, output, 1500, 500));
    assert.equal(session.getContextUsage().tokens, 40_000 + output);
    assert.equal(session.getContextUsage().percent, (40_000 + output) / 253_792 * 100);
    assert.ok(rendered().includes(`${(40_000 + output).toLocaleString("en-GB")} / 253,792`));
    assert.equal(branch.length, 2, "answer has not been persisted yet");
  }
});

test("stale, incomplete and invalid partials cannot revive pre-compaction usage", { skip: !available }, async () => {
  const { AgentSession } = await mod("pi-coding-agent/dist/core/agent-session.js");
  const old = assistant(usage(200_000, 200));
  const state = { isStreaming: true };
  const session = { model, agent: { state }, messages: [old], sessionManager: {
    getBranch: () => [{ type: "message", message: old }, { type: "compaction", timestamp: new Date().toISOString() }],
  } };
  const cases = [
    ["no partial", undefined],
    ["zero usage", assistant(usage(0, 0))],
    ["output only", assistant(usage(0, 500))],
    ["wrong model", { ...assistant(usage(40_000, 100)), model: "another-model" }],
    ["wrong provider", { ...assistant(usage(40_000, 100)), provider: "another-provider" }],
    ["error", { ...assistant(usage(40_000, 100)), stopReason: "error" }],
    ["abort", { ...assistant(usage(40_000, 100)), stopReason: "aborted" }],
    ["non-finite", assistant({ ...usage(40_000, 100), input: NaN })],
    ["negative", assistant(usage(40_000, -1))],
    ["overflow", assistant(usage(Number.MAX_SAFE_INTEGER, 1))],
    ["contradictory total", assistant({ ...usage(40_000, 100), totalTokens: 200_000 })],
  ];
  for (const [name, partial] of cases) {
    state.streamingMessage = partial;
    assert.equal(AgentSession.prototype.getContextUsage.call(session).tokens, null, name);
  }
  state.streamingMessage = assistant(usage(40_000, 100));
  state.isStreaming = false;
  assert.equal(AgentSession.prototype.getContextUsage.call(session).tokens, null, "settled stale stream");
});

test("real SDK delayed post-compaction stream updates context before message_end and resets across sessions", {
  skip: !available, timeout: 15000,
}, async (t) => {
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { AssistantMessageEventStream } = await mod("pi-ai/dist/utils/event-stream.js");
  const directory = await mkdtemp(join(tmpdir(), "pi-streaming-context-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }, { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd: directory, agentDir: directory, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true, systemPrompt: "Synthetic fixture." });
  await resourceLoader.reload();
  const sessionManager = SessionManager.create(directory, join(directory, "sessions"));
  sessionManager.appendMessage({ role: "user", content: "Earlier synthetic request.", timestamp: Date.now() - 2 });
  sessionManager.appendMessage(assistant(usage(200_000, 200)));
  sessionManager.appendCompaction("Synthetic compacted summary.", sessionManager.getEntries()[0].id, 200_200);
  const started = Promise.withResolvers();
  let calls = 0;
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }), streamSimple: () => {
      assert.equal(++calls, 1);
      const stream = new AssistantMessageEventStream();
      queueMicrotask(() => {
        stream.push({ type: "start", partial: assistant(usage(0, 0)) });
        started.resolve(stream);
      });
      return stream;
    } };
  const { session } = await createAgentSession({ cwd: directory, agentDir: directory, model, modelRuntime,
    resourceLoader, sessionManager, settingsManager, tools: [] });
  t.after(() => session.dispose());
  const updated = Promise.withResolvers();
  const events = [];
  const unsubscribe = session.subscribe(event => {
    events.push({ type: event.type, role: event.message?.role });
    if (event.type === "message_update") updated.resolve(session.getContextUsage());
  });
  t.after(unsubscribe);
  assert.equal(session.getContextUsage().tokens, null);
  const running = session.prompt("Synthetic request after compaction.");
  t.after(() => session.abort());
  const stream = await started.promise;
  stream.push({ type: "usage_update", partial: assistant(usage(39_000, 100, 1000)) });
  try {
    const measured = await updated.promise;
    assert.equal(measured.tokens, 40_100);
    assert.ok(!events.some(event => event.type === "message_end" && event.role === "assistant"), "live measurement precedes completion");
    assert.equal(sessionManager.getBranch().filter(entry => entry.type === "message" && entry.message.role === "assistant").length, 1);
  } finally {
    stream.push({ type: "done", reason: "stop", message: assistant(usage(39_000, 200, 1000)) });
    stream.end();
  }
  await running;
  assert.equal(session.getContextUsage().tokens, 40_200);
  assert.equal(session.state.streamingMessage, undefined);
  sessionManager.appendCompaction("New synthetic summary.", sessionManager.getEntries()[0].id, 40_200);
  session.agent.state.messages = sessionManager.buildSessionContext().messages;
  assert.equal(session.getContextUsage().tokens, null, "new compaction invalidates previous response usage");
  session.agent.reset();
  assert.equal(session.getContextUsage().tokens, null, "reset cannot retain old live usage");
  const nextManager = SessionManager.create(directory, join(directory, "other-sessions"));
  nextManager.appendMessage({ role: "user", content: "Another synthetic session.", timestamp: Date.now() });
  nextManager.appendMessage(assistant(usage(150_000, 100)));
  nextManager.appendCompaction("Another synthetic checkpoint.", nextManager.getEntries()[0].id, 150_100);
  const { session: nextSession } = await createAgentSession({ cwd: directory, agentDir: directory, model, modelRuntime,
    resourceLoader, sessionManager: nextManager, settingsManager, tools: [] });
  t.after(() => nextSession.dispose());
  assert.equal(nextSession.getContextUsage().tokens, null, "another session cannot inherit the first stream's usage");
});
