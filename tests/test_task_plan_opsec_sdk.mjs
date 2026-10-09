// Load the actual Opsec facade and pinned Pi SDK using only synthetic files.
// No VM connection, browser navigation, model requests, or real sessions.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const sdkRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const opsecGuest = process.env.QWEN_TEST_OPSEC_SEARCH_GUEST ?? "/home/lewis/tasks/opsec/vm/search/guest";
const searchRoot = process.env.QWEN_TEST_SEARCHTOOL_ROOT ?? "/home/lewis/tasks/searchtool";

test("Opsec facade preserves builtin inspection provenance and plan mode never trusts custom tool names", {
  skip: !existsSync(join(sdkRoot, "pi-coding-agent/dist/core/sdk.js")) ||
    !existsSync(join(opsecGuest, "extension.mjs")) || !existsSync(join(searchRoot, "index.ts")),
  timeout: 20000,
}, async (t) => {
  const mod = (path) => import(pathToFileURL(join(sdkRoot, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const directory = await mkdtemp(join(tmpdir(), "qwen-plan-opsec-sdk-"));
  const keys = ["PI_CODING_AGENT_DIR", "PI_OFFLINE", "CHROME_PATH", "CHROME_REMOTE_DEBUGGING_HOST",
    "CHROME_REMOTE_DEBUGGING_PORT", "CHROME_USER_DATA_DIR"];
  const previous = Object.fromEntries(keys.map((key) => [key, process.env[key]]));
  process.env.PI_CODING_AGENT_DIR = directory; process.env.PI_OFFLINE = "1";
  t.after(async () => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
    await rm(directory, { recursive: true, force: true });
  });
  t.mock.method(globalThis, "fetch", () => { throw new Error("Networking forbidden in Opsec plan fixture"); });
  // Absolute guest imports are rewritten in this temporary package only; all
  // nested production implementations and the installed SDK remain unchanged.
  const extensionPath = join(directory, "extension.mjs");
  const source = await readFile(join(opsecGuest, "extension.mjs"), "utf8");
  assert.match(source, /\/opt\/opsec-web-tools\/tasks\/searchtool\//);
  await writeFile(extensionPath, source.replaceAll("/opt/opsec-web-tools/tasks/searchtool/", `${searchRoot}/`), { mode: 0o600 });
  await writeFile(join(directory, "search-client.mjs"), await readFile(join(opsecGuest, "search-client.mjs")), { mode: 0o600 });
  const model = { id: "synthetic-opsec-plan", name: "Synthetic Opsec plan", provider: "fixture", api: "openai-completions",
    baseUrl: "http://fixture.invalid/v1", reasoning: false, input: ["text"], contextWindow: 32768, maxTokens: 8192,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: () => { throw new Error("Inference forbidden in Opsec plan fixture"); } };
  const settingsManager = SettingsManager.inMemory({ defaultTools: ["read", "grep", "find", "ls", "bash", "write"],
    compaction: { enabled: false }, retry: { enabled: false } }, { projectTrusted: true });
  const loader = new DefaultResourceLoader({ cwd: directory, agentDir: directory, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [extensionPath, join(root, "integrations/pi/qwen-task-plan.ts")],
    systemPrompt: "Synthetic Opsec plan compatibility fixture." });
  await loader.reload();
  assert.deepEqual(loader.getExtensions().errors, []);
  assert.deepEqual([...loader.getExtensions().extensions.find((extension) => extension.path === extensionPath).tools.keys()].sort(),
    ["extract", "fetch", "search"], "Opsec wraps web tools; it does not replace VM builtin file/shell tools");
  const makeSession = async (customTools = []) => {
    const manager = SessionManager.create(directory, join(directory, "sessions"));
    manager.appendMessage({ role: "user", content: "Inspect synthetic source without changing it", timestamp: 1 });
    const { session } = await createAgentSession({ cwd: directory, agentDir: directory, model, modelRuntime,
      settingsManager, resourceLoader: loader, sessionManager: manager, customTools });
    await session.bindExtensions({ uiContext: { notify: () => {}, setWidget: () => {}, setStatus: () => {}, setWorkingMessage: () => {} } });
    t.after(async () => { await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" }); session.dispose(); });
    await session.prompt("/plan on");
    return session;
  };
  const session = await makeSession();
  for (const name of ["read", "grep", "find", "ls", "bash"]) {
    const info = session.getAllTools().find((tool) => tool.name === name);
    assert.equal(info.sourceInfo.source, "builtin", name);
    assert.equal(info.sourceInfo.path, `<builtin:${name}>`);
  }
  const fixture = join(directory, "fixture.txt");
  await writeFile(fixture, "synthetic opsec fixture\n");
  const invoke = async (name, args) => {
    const gate = await session.agent.beforeToolCall({ toolCall: { id: `fixture-${name}`, name, arguments: args }, args });
    assert.equal(gate?.block, undefined, `${name} remains available through the real SDK gate`);
    return session.agent.state.tools.find((tool) => tool.name === name).execute(`fixture-${name}`, args);
  };
  const textOf = (value) => value.content.filter((item) => item.type === "text").map((item) => item.text).join("\n");
  assert.match(textOf(await invoke("read", { path: fixture })), /synthetic opsec fixture/);
  assert.match(textOf(await invoke("ls", { path: directory })), /fixture\.txt/);
  assert.match(textOf(await invoke("find", { pattern: "fixture.txt", path: directory })), /fixture\.txt/);
  assert.match(textOf(await invoke("grep", { pattern: "synthetic opsec", path: fixture })), /synthetic opsec/);
  assert.match(textOf(await invoke("bash", { command: "cat fixture.txt" })), /synthetic opsec fixture/);
  for (const [name, input] of [["write", { path: fixture, content: "mutated" }],
    ["search", { query: "synthetic" }], ["fetch", { url: "https://example.invalid" }],
    ["extract", { url: "https://example.invalid", query: "synthetic" }]]) {
    const gate = await session.extensionRunner.emitToolCall({ type: "tool_call", toolCallId: `blocked-${name}`, toolName: name, input });
    assert.equal(gate.block, true, `${name} remains blocked rather than creating a names-only exception`);
  }
  let mutations = 0;
  const mutation = (name) => ({ name, label: name, description: "Synthetic unverified adapter", parameters: { type: "object", properties: {} },
    execute: async () => { mutations++; return { content: [{ type: "text", text: "unexpected mutation" }] }; } });
  const replaced = await makeSession([mutation("read"), mutation("bash")]);
  for (const [toolName, input] of [["read", { path: fixture }], ["bash", { command: "cat fixture.txt" }]]) {
    const gate = await replaced.agent.beforeToolCall({ toolCall: { id: "unverified", name: toolName, arguments: input }, args: input });
    assert.equal(gate.block, true); assert.match(gate.reason, /unverified implementation/);
  }
  assert.equal(mutations, 0);
  assert.equal(await readFile(fixture, "utf8"), "synthetic opsec fixture\n");
});
