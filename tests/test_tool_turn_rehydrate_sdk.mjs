// Public SDK tool calls with synthetic archives and an in-memory provider.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { mkdir, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const piRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const installed = { skip: !existsSync(join(piRoot, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000 };
const digest = (data) => createHash("sha256").update(data).digest("hex");
const textOf = (message) => message.content.filter((part) => part.type === "text").map((part) => part.text).join("\n");

test("installed Pi condenses a result and rehydrates live and historical archives through its public tool path", installed, async (t) => {
  const mod = (path) => import(pathToFileURL(join(piRoot, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { AssistantMessageEventStream } = await mod("pi-ai/dist/utils/event-stream.js");
  const { validateToolArguments } = await mod("pi-ai/dist/utils/validation.js");
  const directory = await mkdtemp(join(tmpdir(), "tool-turn-rehydrate-sdk-"));
  const envNames = ["PI_CODING_AGENT_DIR", "PI_OFFLINE", "QWEN_PI_TOOL_RESULT_DIR", "QWEN_PI_TOOL_RESULT_MAX_BYTES", "QWEN_PI_TOOL_TURN_ARCHIVE_ROOT", "QWEN_PI_FIXED_SLOT_RESUME"];
  const previous = Object.fromEntries(envNames.map((name) => [name, process.env[name]]));
  process.env.PI_CODING_AGENT_DIR = directory;
  process.env.PI_OFFLINE = "1";
  process.env.QWEN_PI_TOOL_RESULT_DIR = join(directory, "live-archive");
  process.env.QWEN_PI_TOOL_RESULT_MAX_BYTES = "2048";
  process.env.QWEN_PI_TOOL_TURN_ARCHIVE_ROOT = join(directory, "historical-archive");
  delete process.env.QWEN_PI_FIXED_SLOT_RESUME;
  t.after(async () => {
    for (const [name, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[name]; else process.env[name] = value;
    }
    await rm(directory, { recursive: true, force: true });
  });
  t.mock.method(globalThis, "fetch", async () => { throw new Error("archive rehydration must remain offline"); });
  const model = { id: "synthetic-rehydrate", name: "Synthetic rehydrate", provider: "qwen-r9700", api: "openai-completions",
    baseUrl: "http://fixture.invalid/v1", reasoning: false, input: ["text"], contextWindow: 32768, maxTokens: 1024,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const usage = { input: 10, output: 10, cacheRead: 0, cacheWrite: 0, totalTokens: 20,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const assistant = (text) => ({ role: "assistant", api: model.api, provider: model.provider, model: model.id,
    timestamp: Date.now(), content: [{ type: "text", text }], stopReason: "stop", usage });
  const lines = Array.from({ length: 240 }, (_, index) => `row-${String(index + 1).padStart(3, "0")} ${"x".repeat(90)}`);
  lines[116] = "archived-sdk-fact=sea-otter-625";
  const original = `${lines.join("\n")}\n`;
  const originalDigest = digest(original);
  const fixtureResult = { content: [{ type: "text", text: original }], details: { fixtureSource: "synthetic original" } };
  const historicalFact = "historical-sdk-fact=violet-badger-943";
  const historical = Buffer.from(JSON.stringify({ archive_schema: "qwen-pi-archived-tool-turn-v1", tool_name: "fixture", tool_call_id: "earlier-fixture",
    tool_result_entry: { type: "message", message: { role: "toolResult", content: [{ type: "text", text: `before\n${historicalFact}\nafter\n` }] } } }));
  const historicalDigest = digest(historical);
  const historicalPrefix = join(process.env.QWEN_PI_TOOL_TURN_ARCHIVE_ROOT, "sha256", historicalDigest.slice(0, 2));
  await mkdir(historicalPrefix, { recursive: true, mode: 0o700 });
  const historicalPath = join(historicalPrefix, `${historicalDigest}.json`);
  await writeFile(historicalPath, historical, { mode: 0o400 });
  const settingsManager = SettingsManager.inMemory({ defaultTools: ["fixture"], compaction: { enabled: false }, retry: { enabled: false } },
    { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd: directory, agentDir: directory, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [resolve("integrations/pi/qwen-tool-output-condense.mjs"), resolve("integrations/pi/qwen-tool-turn-rehydrate.mjs")],
    systemPrompt: "Synthetic tool archive roundtrip fixture." });
  await resourceLoader.reload();
  assert.deepEqual(resourceLoader.getExtensions().errors, []);
  const returned = [];
  let requests = 0;
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: (_model, context) => {
      const turn = requests++;
      assert.ok(turn < 7, "synthetic archive retrieval must terminate");
      assert.ok(context.tools.some((tool) => tool.name === "rehydrate_tool_result"), "the production tool is present in the model request");
      assert.ok(!context.tools.some((tool) => tool.name === "qwen_rehydrate_tool_turn"), "the original alias stays unadvertised");
      assert.match(context.systemPrompt, /Qwen bounded tool-output discipline/);
      let name = "rehydrate_tool_result", params;
      if (turn === 0) { name = "fixture"; params = {}; }
      else {
        const result = context.messages.filter((message) => message.role === "toolResult").at(-1);
        assert.ok(result, "SDK forwards the production result to the continuation");
        returned.push(result);
        const text = textOf(result);
        if (turn === 1) {
          assert.equal(result.isError, false);
          assert.ok(Buffer.byteLength(text) <= 2048);
          assert.match(text, /output condensed for model context/);
          assert.ok(!text.includes(lines[116]), "the middle detail is omitted from the bounded view");
          assert.equal(result.details.fixtureSource, "synthetic original");
          assert.equal(result.details.qwenToolResultArchive.sha256, originalDigest);
          assert.equal(readFileSync(result.details.fullOutputPath, "utf8"), original);
          params = { sha256: originalDigest };
        } else if (turn === 2) {
          assert.equal(result.isError, false);
          assert.match(text, /UNTRUSTED LIVE TOOL DATA/);
          assert.match(text, /digest-only preview/);
          assert.ok(Buffer.byteLength(text) <= 1536);
          assert.deepEqual(result.details.selectedLines, Array.from({ length: 12 }, (_, index) => index + 1));
          params = { sha256: originalDigest, pattern: "archived-sdk-fact", context: 0 };
        } else if (turn === 3) {
          assert.equal(result.isError, false);
          assert.ok(text.includes(`117: ${lines[116]}`));
          assert.deepEqual(result.details.selectedLines, [117]);
          assert.equal(result.details.archiveSha256, originalDigest);
          params = { sha256: originalDigest, start_line: 100, end_line: 135, max_lines: 36 };
        } else if (turn === 4) {
          assert.equal(result.isError, false);
          assert.ok(Buffer.byteLength(text) > 2048, "targeted retrieval may exceed the condensation budget");
          assert.match(text, /UNTRUSTED LIVE TOOL DATA/);
          assert.ok(!text.includes("output condensed for model context"), "rehydration must not be condensed recursively");
          assert.equal(result.details.qwenToolResultArchive, undefined, "rehydration retains the original digest rather than creating another archive");
          assert.deepEqual(result.details.selectedLines, Array.from({ length: 36 }, (_, index) => index + 100));
          params = { sha256: historicalDigest, pattern: "historical-sdk-fact", context: 0 };
        } else if (turn === 5) {
          assert.equal(result.isError, false);
          assert.match(text, /UNTRUSTED HISTORICAL TOOL DATA/);
          assert.ok(text.includes(`2: ${historicalFact}`));
          assert.equal(result.details.archiveSchema, "qwen-pi-archived-tool-turn-v1");
          params = { pattern: "missing required digest" };
        } else {
          assert.equal(result.isError, true);
          assert.match(text, /Validation failed for tool "rehydrate_tool_result"/);
          assert.match(text, /sha256/);
        }
      }
      const message = params ? { ...assistant(""), content: [{ type: "toolCall", id: `archive-${turn}`, name, arguments: params }], stopReason: "toolUse" }
        : assistant("Retrieved the exact original live and historical details.");
      const stream = new AssistantMessageEventStream();
      queueMicrotask(() => {
        stream.push({ type: "start", partial: { ...message, content: [] } });
        stream.push({ type: "done", reason: message.stopReason, message });
        stream.end();
      });
      return stream;
    } };
  const manager = SessionManager.create(directory, join(directory, "sessions"));
  const { session } = await createAgentSession({ cwd: directory, agentDir: directory, model, modelRuntime, resourceLoader,
    sessionManager: manager, settingsManager, customTools: [{ name: "fixture", label: "Fixture", description: "Synthetic oversized output",
      parameters: { type: "object", properties: {}, additionalProperties: false }, execute: async () => fixtureResult }] });
  try {
    await session.bindExtensions({ uiContext: { notify() {}, setWidget() {}, setStatus() {}, setWorkingMessage() {} } });
    assert.ok(session.getAllTools().some((tool) => tool.name === "rehydrate_tool_result"), "real SDK registers the extension");
    for (const name of ["qwen_rehydrate_tool_turn", "read_archived_tool_result"]) {
      assert.ok(!session.getAllTools().some((tool) => tool.name === name), "historical names are not registered tools");
    }
    assert.deepEqual(session.getActiveToolNames().sort(), ["fixture", "rehydrate_tool_result"], "explicit defaultTools retains the fixture and activates the extension tool");
    const tool = session.agent.state.tools.find((candidate) => candidate.name === "rehydrate_tool_result");
    assert.deepEqual(validateToolArguments(tool, { name: tool.name, arguments: { sha256: originalDigest, max_lines: 2 } }), { sha256: originalDigest, max_lines: 2 });
    for (const params of [{}, { sha256: "invalid" }, { sha256: originalDigest, max_lines: 201 }, { sha256: originalDigest, unexpected: true }]) {
      assert.throws(() => validateToolArguments(tool, { name: tool.name, arguments: params }), /Validation failed/);
    }
    await session.prompt("Read the fixture output, recover its exact archived detail, inspect a wider range, then read the historical detail.");
    assert.equal(requests, 7);
    assert.equal(returned.length, 6);
    assert.equal(fixtureResult.content[0].text, original, "the condensation hook leaves the original tool result object intact");
    const livePath = returned[0].details.fullOutputPath;
    assert.equal(readFileSync(livePath, "utf8"), original);
    assert.equal(digest(readFileSync(livePath)), originalDigest);
    assert.equal(statSync(livePath).mode & 0o777, 0o400);
    assert.equal(statSync(livePath).nlink, 1);
    assert.deepEqual(await readdir(join(process.env.QWEN_PI_TOOL_RESULT_DIR, "sha256", originalDigest.slice(0, 2))), [`${originalDigest}.txt`]);
    assert.deepEqual(readFileSync(historicalPath), historical);
    const reopened = SessionManager.open(manager.getSessionFile(), join(directory, "sessions"));
    const persisted = reopened.getEntries().find((entry) => entry.type === "message" && entry.message.role === "toolResult" && entry.message.toolName === "fixture").message;
    assert.equal(persisted.details.qwenToolResultArchive.sha256, originalDigest, "the SDK persists the authoritative archive locator");
  } finally {
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    session.dispose();
  }
});
