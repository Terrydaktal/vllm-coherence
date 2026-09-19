// Exercises actual installed Pi scheduling/extension/client interfaces with a
// synthetic in-memory provider and number tool. No network or live transcript.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, existsSync, readFileSync, rmSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { SESSION_ID, PROJECT, CONTROL, RADIANCE_MODEL } from "../integrations/pi/qwen-searchtool-recovery.mjs";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
for (const failureCase of ["protocol", "announcement", "missing-outcome", "radiance-command", "radiance-shorthand", "radiance-colon", "radiance-exhaustion"]) {
test(`installed Pi handles ${failureCase} with hidden, bounded recovery`, {
  skip: !existsSync(join(root, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000,
}, async () => {
  const mod = path => import(pathToFileURL(join(root, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { AssistantMessageEventStream } = await mod("pi-ai/dist/utils/event-stream.js");
  const agentDir = mkdtempSync(join(tmpdir(), "qwen-searchtool-recovery-sdk-"));
  const globalRecovery = failureCase.startsWith("radiance-");
  const exhausted = failureCase === "radiance-exhaustion";
  const cwd = globalRecovery ? agentDir : PROJECT;
  const model = { id: globalRecovery ? RADIANCE_MODEL : "fixture", name: "In-memory number lookup", provider: "qwen-r9700", api: "openai-completions",
    baseUrl: "http://fixture.invalid/v1", reasoning: true, input: ["text"], contextWindow: 253792, maxTokens: 1024,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: true, maxRetries: 3, baseDelayMs: 1 } }, { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd, agentDir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [resolve("integrations/pi/qwen-searchtool-recovery.mjs")],
    systemPrompt: "Harmless in-memory number lookup fixture." });
  await resourceLoader.reload();
  assert.equal(resourceLoader.getExtensions().errors.length, 0);
  const executions = [], contexts = [], events = [];
  const call = (id, key) => ({ type: "toolCall", id, name: "lookup_number", arguments: { key } });
  const failureText = {
    protocol: "Keep this earlier text.", announcement: "Let me look up beta:", "missing-outcome": "",
    "radiance-command": "I'll query `lookup_number` for beta:",
    "radiance-shorthand": "Checking the second number:",
    "radiance-colon": "Results: \n\t",
    "radiance-exhaustion": "Results:",
  }[failureCase];
  const responses = [
    { content: [call("first", "alpha")], stopReason: "toolUse" },
    { content: [{ type: "thinking", thinking: "Keep this reasoning." }, { type: "text", text: failureText },
      ...(failureCase === "protocol" ? [{ type: "toolCall", id: "incomplete", name: "lookup_number", arguments: {} }] : [])],
      stopReason: failureCase === "protocol" ? "error" : "stop",
      ...(failureCase === "protocol" ? { errorMessage: "Stream ended without finish_reason" } : {}) },
    ...(exhausted ? [
      { content: [{ type: "text", text: failureText }], stopReason: "stop" },
      { content: [{ type: "text", text: failureText }], stopReason: "stop" },
    ] : [
      { content: [call("duplicate", "alpha"), call("second", "beta")], stopReason: "toolUse" },
      { content: [{ type: "text", text: "The total is 24." }], stopReason: "stop" },
    ]),
  ];
  const modelRuntime = {
    getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: (_model, context) => {
      contexts.push({ messages: structuredClone(context.messages) });
      assert.ok(contexts.length <= responses.length, "unexpected retry loop");
      const message = { role: "assistant", api: model.api, provider: model.provider, model: model.id,
        timestamp: Date.now(), usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, ...responses[contexts.length - 1] };
      const stream = new AssistantMessageEventStream();
      queueMicrotask(() => {
        stream.push({ type: "start", partial: { ...message, content: [] } });
        stream.push(message.stopReason === "error"
          ? { type: "error", reason: "error", error: message }
          : { type: "done", reason: message.stopReason, message });
        stream.end();
      });
      return stream;
    },
  };
  const sessionManager = globalRecovery
    ? SessionManager.create(cwd, join(agentDir, "sessions"))
    : SessionManager.inMemory(PROJECT, { id: SESSION_ID });
  const { session } = await createAgentSession({ cwd, agentDir, model, modelRuntime,
    resourceLoader, sessionManager, settingsManager, tools: ["lookup_number"],
    customTools: [{ name: "lookup_number", label: "Lookup", description: "Return the fixture number.",
      parameters: { type: "object", properties: { key: { type: "string" } }, required: ["key"], additionalProperties: false },
      execute: async (_id, args) => {
        executions.push(args.key);
        return { content: [{ type: "text", text: "12" }], details: {} };
      } }],
  });
  const unsubscribe = session.subscribe(event => events.push(event));
  try {
    await session.bindExtensions({});
    await session.prompt("Add alpha and beta using the fixture tool.");
    assert.equal(contexts.length, 4, JSON.stringify(sessionManager.buildSessionContext().messages));
    assert.deepEqual(executions, exhausted ? ["alpha"] : ["alpha", "beta"]);
    const messages = sessionManager.buildSessionContext().messages;
    const controls = messages.filter(m => m.role === "custom" && m.customType === CONTROL);
    assert.equal(controls.length, exhausted ? 2 : 1);
    assert.ok(controls.every(control => control.display === false));
    assert.equal(messages.filter(m => m.role === "user").length, 1, "no synthetic user go-ahead message");
    assert.equal(events.filter(e => e.type === "message_end" && e.message.role === "user").length, 1);
    assert.equal(messages.at(-1).content[0].text, exhausted ? failureText : "The total is 24.");
    const failed = messages.find(m => m.role === "assistant" && m.content.some(c => c.thinking === "Keep this reasoning."));
    assert.deepEqual(failed.content.map(c => c.thinking ?? c.text), ["Keep this reasoning.", failureText]);
    assert.equal(failed.content.some(c => c.type === "toolCall"), false);
    assert.equal(events.filter(e => e.type === "auto_retry_start").length, 0);
    assert.equal(events.filter(e => e.type === "agent_settled").length, 1, "the run settles after all continuations finish");
    if (!exhausted) assert.equal(messages.find(m => m.role === "toolResult" && m.toolCallId === "duplicate").isError, true);
    if (globalRecovery) {
      // Read only this test's newly created synthetic transcript. The hidden
      // control is tagged as an extension entry on disk, never as user input.
      const entries = readFileSync(sessionManager.getSessionFile(), "utf8").trim().split("\n").map(line => JSON.parse(line));
      const savedControls = entries.filter(e => e.type === "custom_message" && e.customType === CONTROL);
      assert.equal(savedControls.length, exhausted ? 2 : 1);
      assert.equal(savedControls[0].display, false);
      assert.equal(savedControls[0].details.reason, "announcement");
      assert.equal(entries.filter(e => e.type === "message" && e.message.role === "user").length, 1);
      assert.deepEqual(contexts[2].messages.slice(0, contexts[1].messages.length), contexts[1].messages,
        "recovery appends to the existing provider prefix instead of rewriting/replaying it");
      assert.equal(SessionManager.open(sessionManager.getSessionFile()).buildSessionContext().messages.filter(m => m.role === "user").length, 1);
    }
  } finally {
    unsubscribe(); session.dispose();
    rmSync(agentDir, { recursive: true, force: true });
  }
});
}
