// Offline integration against the installed Pi SDK, using disposable synthetic data.
import assert from "node:assert/strict";
import test from "node:test";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { TOOL_NAME_ALIASES, BUILTIN_TOOL_NAMES } from "../integrations/pi/qwen-tool-names.mjs";
import { CONTEXT_POLICY_ENTRY } from "../integrations/pi/qwen-context-policy.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const sdkRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const searchRoot = process.env.QWEN_TEST_SEARCHTOOL_ROOT ?? "/home/lewis/tasks/searchtool";
const textOf = (result) => result.content.filter((block) => block.type === "text").map((block) => block.text).join("\n");

test("installed Pi exposes canonical tools and preserves native execution and historical context", {
  skip: !existsSync(join(sdkRoot, "pi-coding-agent/dist/core/sdk.js")) || !existsSync(join(searchRoot, "index.ts")),
  timeout: 20000,
}, async (t) => {
  const mod = (path) => import(pathToFileURL(join(sdkRoot, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const nativeTools = await mod("pi-coding-agent/dist/core/tools/index.js");
  const { createWriteToolDefinition, createEditToolDefinition } = nativeTools;
  const { validateToolArguments } = await mod("pi-ai/dist/utils/validation.js");
  const directory = await mkdtemp(join(tmpdir(), "qwen-tool-names-sdk-"));
  const cwd = join(directory, "workspace"), agentDir = join(directory, "agent"), outside = join(directory, "process-cwd");
  await Promise.all([mkdir(cwd), mkdir(agentDir), mkdir(outside)]);
  const originalCwd = process.cwd();
  const environment = ["PI_CODING_AGENT_DIR", "PI_OFFLINE", "QWEN_RADIANCE_CACHE_ABI"];
  const previous = Object.fromEntries(environment.map((key) => [key, process.env[key]]));
  process.env.PI_CODING_AGENT_DIR = agentDir; process.env.PI_OFFLINE = "1";
  delete process.env.QWEN_RADIANCE_CACHE_ABI;
  process.chdir(outside);
  t.after(async () => {
    process.chdir(originalCwd);
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
    await rm(directory, { recursive: true, force: true });
  });
  let networkRequests = 0, inferenceRequests = 0;
  t.mock.method(globalThis, "fetch", () => { networkRequests++; throw new Error("Networking forbidden in canonical-tool fixture"); });
  await writeFile(join(agentDir, "settings.json"), JSON.stringify({ defaultTools: Object.keys(TOOL_NAME_ALIASES),
    shellCommandPrefix: "export SDK_PREFIX_MARKER=from-persisted-settings", shellPath: "/usr/bin/bash",
    compaction: { enabled: false }, retry: { enabled: false } }));
  const settingsManager = SettingsManager.create(cwd, agentDir, { projectTrusted: true });
  const model = { id: "synthetic-canonical-tools", name: "Synthetic canonical tools", provider: "fixture", api: "openai-completions",
    baseUrl: "http://fixture.invalid/v1", reasoning: false, input: ["text"], contextWindow: 32768, maxTokens: 8192,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: () => { inferenceRequests++; throw new Error("Inference forbidden in canonical-tool fixture"); } };
  const namingPath = join(root, "integrations/pi/qwen-tool-names.ts");
  const resourceLoader = new DefaultResourceLoader({ cwd, agentDir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [join(root, "integrations/pi/qwen-radiance-compaction.ts"),
      join(root, "integrations/pi/qwen-task-plan.ts"), join(root, "integrations/pi/qwen-tool-output-condense.mjs"),
      join(root, "integrations/pi/qwen-tool-turn-rehydrate.mjs"), join(root, "integrations/pi/qwen-session-search.mjs"),
      join(searchRoot, "index.ts"), namingPath] });
  await resourceLoader.reload();
  assert.deepEqual(resourceLoader.getExtensions().errors, []);
  assert.equal(resourceLoader.getExtensions().extensions.at(-1).path, namingPath, "normalization loads after context selection");
  const manager = SessionManager.create(cwd, join(directory, "sessions"));
  manager.appendMessage({ role: "user", content: "Synthetic local tool fixture", timestamp: 1 });
  const { session } = await createAgentSession({ cwd, agentDir, model, modelRuntime, settingsManager, resourceLoader,
    excludeTools: [...BUILTIN_TOOL_NAMES], sessionManager: manager });
  const extensionErrors = [];
  session.extensionRunner.onError((error) => extensionErrors.push(error));
  t.after(async () => { await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" }); session.dispose(); });
  await session.bindExtensions({ uiContext: { notify() {}, setWidget() {}, setStatus() {}, setWorkingMessage() {} } });
  const tool = (name) => session.agent.state.tools.find((candidate) => candidate.name === name);
  const invoke = (name, params, signal = new AbortController().signal) => {
    const current = tool(name);
    assert.ok(current, `${name} is active in the installed SDK`);
    validateToolArguments(current, { name, arguments: params });
    return current.execute(`synthetic-${name}`, params, signal);
  };

  await t.test("all 13 canonical tools are active with original schemas, and native tools use session cwd/settings", async () => {
    assert.deepEqual(session.getActiveToolNames().sort(), Object.values(TOOL_NAME_ALIASES).sort());
    const registry = new Map(session.getAllTools().map((info) => [info.name, info]));
    assert.match(registry.get("search_file_contents").description, /literal: true.*context: 0.*limit: 50/);
    assert.match(registry.get("run_shell_command").description, /Prefer search_file_contents, find_files and list_directory/);
    const prompt = session.agent.state.systemPrompt;
    assert.match(prompt, /Run builds, tests, version-control commands/);
    assert.match(prompt, /downstream grep -v filters results after the files were read/);
    assert.match(prompt, /explicit timeout \(20 seconds initially\)/);
    assert.ok(!prompt.includes("Execute run_shell_command commands"), "native tool names are never advertised as executable shell commands");
    assert.deepEqual([...registry.keys()].sort(), Object.values(TOOL_NAME_ALIASES).sort(), "registry contains exactly the 13 current names");
    assert.ok(!registry.has("read_archived_tool_result"), "previous archive-reader name is not registered");
    for (const [original, canonical] of Object.entries(TOOL_NAME_ALIASES)) {
      assert.ok(!registry.has(original), `${original} is absent from the registry`);
    }
    const factories = [nativeTools.createReadToolDefinition, nativeTools.createBashToolDefinition,
      createEditToolDefinition, createWriteToolDefinition, nativeTools.createGrepToolDefinition,
      nativeTools.createFindToolDefinition, nativeTools.createLsToolDefinition];
    for (const factory of factories) {
      const original = factory(cwd);
      assert.deepEqual(registry.get(TOOL_NAME_ALIASES[original.name]).parameters, original.parameters,
        `${original.name} delegate preserves the SDK parameter schema`);
    }
    assert.notEqual(process.cwd(), cwd);
    await invoke("write_file", { path: "fixtures/input.txt", content: "alpha\nbeta\n" });
    await invoke("edit_file", { path: "fixtures/input.txt", edits: [{ oldText: "beta", newText: "gamma" }] });
    assert.match(textOf(await invoke("read_file", { path: "fixtures/input.txt" })), /alpha\ngamma/);
    assert.match(textOf(await invoke("search_file_contents", { path: "fixtures", pattern: "gamma" })), /input\.txt.*gamma/);
    assert.match(textOf(await invoke("find_files", { path: "fixtures", pattern: "*.txt" })), /input\.txt/);
    assert.match(textOf(await invoke("list_directory", { path: "fixtures" })), /input\.txt/);
    const command = 'printf "%s\\n" "$SDK_PREFIX_MARKER"; pwd; cat fixtures/input.txt';
    const shell = textOf(await invoke("run_shell_command", { command }));
    assert.match(shell, /from-persisted-settings/);
    assert.ok(shell.includes(cwd), "shell execution uses session cwd independently of process.cwd()");
    assert.match(shell, /alpha\ngamma/);
    assert.equal(existsSync(join(outside, "fixtures")), false, "relative writes never target the process directory");
    settingsManager.setShellCommandPrefix("export SDK_PREFIX_MARKER=updated-persisted-settings");
    await settingsManager.flush();
    assert.equal(JSON.parse(await readFile(join(agentDir, "settings.json"), "utf8")).shellCommandPrefix,
      "export SDK_PREFIX_MARKER=updated-persisted-settings");
    assert.match(textOf(await invoke("run_shell_command", { command: 'printf "%s" "$SDK_PREFIX_MARKER"' })), /updated-persisted-settings/);
  });

  await t.test("canonical write/edit delegates share native SDK serialization and queued cancellation", async () => {
    const path = join(cwd, "shared-queue.txt");
    await writeFile(path, "before\n");
    let entered, release;
    const started = new Promise((resolve) => { entered = resolve; });
    const held = new Promise((resolve) => { release = resolve; });
    const nativeWrite = createWriteToolDefinition(cwd, { operations: {
      mkdir: (directory) => mkdir(directory, { recursive: true }),
      async writeFile(target, content) { entered(); await held; await writeFile(target, content); },
    } });
    const first = nativeWrite.execute("native-write", { path: "shared-queue.txt", content: "native base\n" });
    await started;
    const controller = new AbortController(); controller.abort();
    const tasks = [first,
      invoke("edit_file", { path: "shared-queue.txt", edits: [{ oldText: "native base", newText: "canonical edit" }] }),
      invoke("write_file", { path: "shared-queue.txt", content: "canonical base\n" }),
      assert.rejects(invoke("write_file", { path: "shared-queue.txt", content: "cancelled overwrite\n" }, controller.signal), /aborted/i),
      createEditToolDefinition(cwd).execute("native-edit", { path: "shared-queue.txt", edits: [{ oldText: "canonical base", newText: "native final" }] }),
    ];
    const completed = Promise.all(tasks);
    release();
    await completed;
    assert.equal(await readFile(path, "utf8"), "native final\n", "native and renamed calls preserve one ordered file-mutation queue");
    await assert.rejects(invoke("edit_file", { path: "shared-queue.txt", edits: [{ oldText: "native final", newText: "cancelled edit" }] }, controller.signal), /aborted/i);
    assert.equal(await readFile(path, "utf8"), "native final\n", "aborted delegates leave the file intact");
  });

  await t.test("context selection runs before historical names normalize, leaving stored JSONL unchanged", async () => {
    const assistant = (content, timestamp) => ({ role: "assistant", api: "openai-completions", provider: "fixture", model: model.id,
      content, timestamp, stopReason: "toolUse", usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });
    const originalNames = Object.keys(TOOL_NAME_ALIASES);
    manager.appendMessage(assistant(originalNames.map((name) => ({ type: "toolCall", id: `historical-${name}`, name,
      arguments: { quotedHistory: `The original ${name} name stays in argument data.` } })), 2));
    for (const name of originalNames) manager.appendMessage({ role: "toolResult", toolCallId: `historical-${name}`, toolName: name,
      content: [{ type: "text", text: `Synthetic archived output quoted ${name}.` }], isError: false, timestamp: 3 });
    const excludedId = manager.appendMessage(assistant([{ type: "toolCall", id: "excluded-read", name: "read", arguments: { path: "excluded.txt" } }], 4));
    manager.appendMessage({ role: "toolResult", toolCallId: "excluded-read", toolName: "read", timestamp: 5, isError: false,
      content: [{ type: "text", text: "Synthetic excluded group" }] });
    manager.appendCustomEntry(CONTEXT_POLICY_ENTRY, { version: 1, preserveFutureThinking: false,
      changes: [{ entryId: excludedId, part: "message", excluded: true }] });
    const originalEntries = structuredClone(manager.getEntries());
    const sessionFile = manager.getSessionFile(), originalBytes = await readFile(sessionFile);
    const input = manager.buildSessionContext().messages, originalInput = structuredClone(input);
    assert.ok(input.some((message) => message.role === "toolResult" && message.toolCallId === "excluded-read"));
    const selected = await session.extensionRunner.emitContext(input);
    assert.ok(!selected.some((message) => message.role === "toolResult" && message.toolCallId === "excluded-read"));
    const calls = selected.filter((message) => message.role === "assistant").flatMap((message) => message.content)
      .filter((block) => block.type === "toolCall");
    assert.deepEqual(calls.map((call) => call.name), Object.values(TOOL_NAME_ALIASES));
    for (const [original, canonical] of Object.entries(TOOL_NAME_ALIASES)) {
      const call = calls.find((candidate) => candidate.id === `historical-${original}`);
      assert.equal(call.name, canonical);
      assert.equal(call.arguments.quotedHistory, `The original ${original} name stays in argument data.`);
      const result = selected.find((message) => message.role === "toolResult" && message.toolCallId === call.id);
      assert.equal(result.toolName, canonical);
      assert.equal(result.content[0].text, `Synthetic archived output quoted ${original}.`);
    }
    assert.deepEqual(input, originalInput);
    assert.deepEqual(manager.getEntries(), originalEntries);
    assert.deepEqual(await readFile(sessionFile), originalBytes, "only outgoing identity metadata changes; JSONL bytes remain authoritative");
  });
  assert.deepEqual(extensionErrors, []);
  assert.equal(networkRequests, 0);
  assert.equal(inferenceRequests, 0);
});
