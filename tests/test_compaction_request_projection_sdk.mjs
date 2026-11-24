// The installed provider adapter must receive the same historical messages and
// tool schemas for an ordinary request and a checkpoint request. Synthetic only.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { compactionRequestMessages } from "../integrations/pi/qwen-radiance-compaction.mjs";
import { BUILTIN_TOOL_NAMES } from "../integrations/pi/qwen-tool-names.mjs";
import { CONTEXT_POLICY_ENTRY } from "../integrations/pi/qwen-context-policy.mjs";
import { TASK_PLAN_CONTEXT } from "../integrations/pi/qwen-task-plan.mjs";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
test("installed SDK ordinary and checkpoint captures preserve identical system, tool order, interrupted output and exclusions", {
  skip: !existsSync(join(root, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000,
}, async (t) => {
  const mod = (path) => import(pathToFileURL(join(root, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { convertToLlm } = await mod("pi-coding-agent/dist/core/messages.js");
  const { streamSimpleOpenAICompletions: stream } = await mod("pi-ai/dist/compat.js");
  const directory = await mkdtemp(join(tmpdir(), "compaction-projection-sdk-"));
  const previous = Object.fromEntries(["PI_CODING_AGENT_DIR", "PI_OFFLINE", "QWEN_RADIANCE_CACHE_ABI"].map((key) => [key, process.env[key]]));
  process.env.PI_CODING_AGENT_DIR = directory; process.env.PI_OFFLINE = "1";
  delete process.env.QWEN_RADIANCE_CACHE_ABI;
  t.after(async () => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
    await rm(directory, { recursive: true, force: true });
  });
  t.mock.method(globalThis, "fetch", () => assert.fail("network is forbidden in the projection fixture"));
  const model = { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", name: "Synthetic projection", provider: "qwen-r9700", api: "openai-completions",
    baseUrl: "http://synthetic.invalid/v1", reasoning: true, input: ["text"], contextWindow: 253792, maxTokens: 32768,
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: false, thinkingFormat: "qwen-chat-template" },
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }, { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd: directory, agentDir: directory, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: ["qwen-radiance-compaction.ts", "qwen-task-plan.ts", "qwen-tool-output-condense.mjs", "qwen-tool-names.ts"].map((name) => resolve("integrations/pi", name)),
    systemPrompt: "Synthetic ordinary/checkpoint projection fixture." });
  await resourceLoader.reload();
  assert.deepEqual(resourceLoader.getExtensions().errors, []);
  let ordinary;
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "synthetic" }, env: {} }),
    streamSimple: (selected, context, options) => stream(selected, context, { ...options, apiKey: "synthetic",
      onPayload: async (payload) => {
        ordinary = await options.onPayload?.(payload) ?? payload;
        throw new Error("SYNTHETIC_CAPTURE");
      }, fetch: () => assert.fail("capture must precede inference") }) };
  const manager = SessionManager.create(directory, join(directory, "sessions"));
  const usage = { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const assistant = (content, stopReason, timestamp) => ({ role: "assistant", provider: model.provider, api: model.api, model: model.id, content, stopReason, timestamp, usage });
  const removed = manager.appendMessage({ role: "user", content: "EXCLUDED_SYNTHETIC_SOURCE", timestamp: 1 });
  manager.appendMessage({ role: "user", content: "Synthetic retained task", timestamp: 2 });
  manager.appendMessage(assistant([{ type: "toolCall", id: "legacy-read", name: "read", arguments: { path: "synthetic.mjs" } }], "toolUse", 3));
  manager.appendMessage({ role: "toolResult", toolCallId: "legacy-read", toolName: "read", content: [{ type: "text", text: "Synthetic tool output" }], isError: false, timestamp: 4 });
  manager.appendCustomMessageEntry(TASK_PLAN_CONTEXT, "EXCLUDED_SYNTHETIC_PLAN", false, { sourceEntryIds: [removed] });
  manager.appendMessage(assistant([{ type: "thinking", thinking: "Interrupted synthetic reasoning", thinkingSignature: "reasoning_content" },
    { type: "text", text: "Interrupted synthetic prose" }, { type: "toolCall", id: "never-executed", name: "write", arguments: {} }], "aborted", 5));
  manager.appendCustomEntry(CONTEXT_POLICY_ENTRY, { version: 1, preserveFutureThinking: false,
    changes: [{ entryId: removed, part: "message", excluded: true }] });
  const { session } = await createAgentSession({ cwd: directory, agentDir: directory, model, modelRuntime, resourceLoader,
    sessionManager: manager, settingsManager, excludeTools: BUILTIN_TOOL_NAMES });
  t.after(async () => { await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" }); session.dispose(); });
  await session.bindExtensions({ uiContext: { notify() {}, setStatus() {}, setWidget() {}, setWorkingMessage() {} } });
  await session.setThinkingLevel("xhigh");
  await session.prompt("Continue the synthetic fixture.");
  assert.ok(ordinary, "the ordinary agent path reaches the real provider conversion");
  const raw = manager.buildSessionContext().messages, before = await readFile(manager.getSessionFile());
  const ctx = session.extensionRunner.createContext();
  const tools = new Map(session.getAllTools().map((tool) => [tool.name, tool]));
  let checkpoint;
  await stream(model, { systemPrompt: ctx.getSystemPrompt(), messages: convertToLlm(compactionRequestMessages(raw, ctx)),
    tools: session.getActiveToolNames().map((name) => tools.get(name)) }, { apiKey: "synthetic", reasoning: session.thinkingLevel, maxTokens: 1,
    onPayload: (payload) => { checkpoint = payload; throw new Error("SYNTHETIC_CAPTURE"); }, fetch: () => assert.fail("no inference") }).result();
  const historical = (payload) => ({ model: payload.model, messages: payload.messages, tools: payload.tools, chat_template_kwargs: payload.chat_template_kwargs });
  assert.deepEqual(historical(checkpoint), historical(ordinary));
  assert.deepEqual(checkpoint.tools.map((tool) => tool.function.name), session.getActiveToolNames());
  assert.match(checkpoint.messages[0].content, /Qwen bounded tool-output discipline/);
  assert.ok(checkpoint.messages.some((message) => message.tool_calls?.some((call) => call.function.name === "read_file")));
  assert.ok(checkpoint.messages.some((message) => message.reasoning_content === "Interrupted synthetic reasoning" && message.content === "Interrupted synthetic prose"));
  assert.ok(!JSON.stringify(checkpoint).includes("EXCLUDED_SYNTHETIC"));
  assert.ok(!JSON.stringify(checkpoint).includes("never-executed"));
  assert.deepEqual(await readFile(manager.getSessionFile()), before, "capture cannot rewrite the transcript");
});
