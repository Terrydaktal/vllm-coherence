// Exercise the production extension through the pinned Pi SDK with synthetic
// JSONL, the real search worker, and an in-memory model. No model/network calls.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readdir, rm, stat } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const piRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const installed = { skip: !existsSync(join(piRoot, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000 };
const textOf = (message) => message.content.filter((part) => part.type === "text").map((part) => part.text).join("\n");

test("installed Pi searches compacted original JSONL, expands entries, and validates its active tool schema", installed, async (t) => {
  const mod = (path) => import(pathToFileURL(join(piRoot, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { AssistantMessageEventStream } = await mod("pi-ai/dist/utils/event-stream.js");
  const { validateToolArguments } = await mod("pi-ai/dist/utils/validation.js");
  const directory = await mkdtemp(join(tmpdir(), "session-search-sdk-"));
  // Mirror the guest's inherited mask before its launcher makes new files
  // private. Existing transcripts can retain this mode inside a private root.
  const previousUmask = process.umask(0o007);
  const previous = Object.fromEntries(["PI_CODING_AGENT_DIR", "QWEN_SESSION_SEARCH_DIR", "PI_OFFLINE"].map((name) => [name, process.env[name]]));
  process.env.PI_CODING_AGENT_DIR = directory;
  process.env.QWEN_SESSION_SEARCH_DIR = join(directory, "search-index");
  process.env.PI_OFFLINE = "1";
  t.after(async () => {
    process.umask(previousUmask);
    for (const [name, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[name]; else process.env[name] = value;
    }
    await rm(directory, { recursive: true, force: true });
  });
  t.mock.method(globalThis, "fetch", async () => { throw new Error("session search must remain offline"); });
  const model = { id: "synthetic-session-search", name: "Synthetic session search", provider: "synthetic", api: "openai-completions",
    baseUrl: "http://fixture.invalid/v1", reasoning: false, input: ["text"], contextWindow: 32768, maxTokens: 1024,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const usage = { input: 10, output: 10, cacheRead: 0, cacheWrite: 0, totalTokens: 20,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const assistant = (text) => ({ role: "assistant", api: model.api, provider: model.provider, model: model.id,
    timestamp: Date.now(), content: [{ type: "text", text }], stopReason: "stop", usage });
  const manager = SessionManager.create(directory);
  manager.appendMessage({ role: "user", content: "What code names the archived bridge?", timestamp: Date.now() });
  const fact = "The archived bridge key is amber-pegasus-731.";
  const factId = manager.appendMessage(assistant(fact));
  manager.appendMessage({ role: "user", content: "The adjacent archive note is copper-ridge-842.", timestamp: Date.now() });
  manager.appendMessage(assistant("Recorded the adjacent archive note."));
  const recentId = manager.appendMessage({ role: "user", content: "Continue with the current deployment.", timestamp: Date.now() });
  manager.appendMessage(assistant("Only the current deployment remains active."));
  manager.appendCompaction("Continue the current deployment. Earlier archive details were omitted.", recentId, 1000, {}, true);
  assert.ok(existsSync(manager.getSessionFile()), "the SDK must persist the original JSONL fixture");
  assert.equal((await stat(manager.getSessionDir())).mode & 0o777, 0o700, "the default SDK session directory is private");
  assert.equal((await stat(manager.getSessionFile())).mode & 0o777, 0o660, "the fixture covers existing guest transcripts created with umask 0007");
  assert.ok(!JSON.stringify(manager.buildSessionContext().messages).includes("amber-pegasus-731"), "compaction removes the exact fact from active context");
  const settingsManager = SettingsManager.inMemory({ defaultTools: ["fixture"], compaction: { enabled: false }, retry: { enabled: false } },
    { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd: directory, agentDir: directory, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [resolve("integrations/pi/qwen-session-search.mjs")], systemPrompt: "Synthetic archived-fact retrieval fixture." });
  await resourceLoader.reload();
  assert.deepEqual(resourceLoader.getExtensions().errors, []);
  let requests = 0;
  const returned = [];
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: (_model, context) => {
      const turn = requests++;
      assert.ok(turn < 4, "the synthetic retrieval must terminate");
      assert.ok(context.tools.some((tool) => tool.name === "pi_session_search"), "search is exposed to the actual model request");
      assert.ok(!context.tools.some((tool) => tool.name === "session_search"), "the original alias stays unadvertised");
      let params;
      if (turn === 0) {
        assert.ok(!JSON.stringify(context.messages).includes("amber-pegasus-731"), "the original fact is not silently injected before retrieval");
        params = { query: "amber-pegasus-731", roles: ["assistant"], limit: 1 };
      } else {
        const result = context.messages.filter((message) => message.role === "toolResult").at(-1);
        assert.ok(result, "the SDK forwards the production tool result into the continuation");
        returned.push(result);
        if (turn === 1) {
          assert.equal(result.isError, false);
          assert.match(textOf(result), /UNTRUSTED HISTORICAL TRANSCRIPT DATA/);
          assert.ok(textOf(result).includes(fact));
          assert.ok(textOf(result).includes(factId));
          params = { around_entry_id: factId, window: 1 };
        } else if (turn === 2) {
          assert.equal(result.isError, false);
          assert.match(textOf(result), /What code names the archived bridge\?/);
          assert.match(textOf(result), /copper-ridge-842/);
          params = { query: "amber", window: 4 };
        } else {
          assert.equal(result.isError, true);
          assert.match(textOf(result), /Validation failed for tool "pi_session_search"/);
          assert.match(textOf(result), /window/);
        }
      }
      const message = params ? { ...assistant(""), content: [{ type: "toolCall", id: `search-${turn}`, name: "pi_session_search", arguments: params }],
        stopReason: "toolUse" } : assistant("The original archived fact and its neighboring context were retrieved.");
      const stream = new AssistantMessageEventStream();
      queueMicrotask(() => {
        stream.push({ type: "start", partial: { ...message, content: [] } });
        stream.push({ type: "done", reason: message.stopReason, message });
        stream.end();
      });
      return stream;
    } };
  const { session } = await createAgentSession({ cwd: directory, agentDir: directory, model, modelRuntime, resourceLoader,
    sessionManager: manager, settingsManager, customTools: [{ name: "fixture", label: "Fixture", description: "Preconfigured synthetic tool",
      parameters: { type: "object", properties: {}, additionalProperties: false }, execute: async () => { throw new Error("fixture tool must not run"); } }] });
  try {
    await session.bindExtensions({ uiContext: { notify() {}, setWidget() {}, setStatus() {}, setWorkingMessage() {} } });
    assert.deepEqual(session.getActiveToolNames().sort(), ["fixture", "pi_session_search"], "activation adds search while retaining explicit default tools");
    assert.ok(!session.getAllTools().some((tool) => tool.name === "session_search"), "historical names are not registered tools");
    const tool = session.agent.state.tools.find((candidate) => candidate.name === "pi_session_search");
    assert.ok(tool, "the SDK registers and wraps the production extension tool");
    assert.deepEqual(validateToolArguments(tool, { name: tool.name, arguments: { query: "amber", limit: 1 } }), { query: "amber", limit: 1 });
    assert.throws(() => validateToolArguments(tool, { name: tool.name, arguments: { query: "amber", limit: 11 } }), /Validation failed/);
    assert.throws(() => validateToolArguments(tool, { name: tool.name, arguments: { query: "amber", unexpected: true } }), /Validation failed/);
    await session.prompt("Recover the omitted bridge key, inspect its adjacent note, then finish.");
    assert.equal(requests, 4);
    assert.equal(returned.length, 3);
    assert.equal(returned[0].details.matches, 1);
    assert.ok(returned[1].details.matches >= 3);
    assert.ok((await readdir(join(directory, "search-index"))).some((name) => name.endsWith(".sqlite")), "the production worker creates the local SQLite index");
    const reopened = SessionManager.open(manager.getSessionFile());
    assert.equal(textOf(reopened.getEntries().find((entry) => entry.id === factId).message), fact, "search preserves the authoritative original message");
  } finally {
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    session.dispose();
  }
});
