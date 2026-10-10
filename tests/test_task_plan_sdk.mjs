// Synthetic, CPU-only SDK integration. No model requests or private transcripts.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { TASK_PLAN_ENTRY, TASK_PLAN_CONTEXT, replayTaskPlan } from "../integrations/pi/qwen-task-plan.mjs";

const sdkRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
test("pinned SDK loads plan extension, activates tool, enforces gates and preserves context prefix", {
  skip: !existsSync(join(sdkRoot, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000,
}, async (t) => {
  const mod = (path) => import(pathToFileURL(join(sdkRoot, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const dir = await mkdtemp(join(tmpdir(), "qwen-task-plan-sdk-"));
  const previous = Object.fromEntries(["PI_CODING_AGENT_DIR", "PI_OFFLINE"].map((key) => [key, process.env[key]]));
  process.env.PI_CODING_AGENT_DIR = dir; process.env.PI_OFFLINE = "1";
  t.after(async () => {
    for (const [key, value] of Object.entries(previous)) { if (value === undefined) delete process.env[key]; else process.env[key] = value; }
    await rm(dir, { recursive: true, force: true });
  });
  t.mock.method(globalThis, "fetch", () => { throw new Error("Network/inference forbidden in task-plan SDK fixture"); });
  const model = { id: "synthetic-plan", name: "Synthetic plan", provider: "fixture", api: "openai-completions",
    baseUrl: "http://fixture.invalid/v1", reasoning: false, input: ["text"], contextWindow: 32768, maxTokens: 8192,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: () => { throw new Error("Inference forbidden in task-plan SDK fixture"); } };
  const settingsManager = SettingsManager.inMemory({ defaultTools: ["read", "bash", "write", "unknown_mutation"],
    compaction: { enabled: false }, retry: { enabled: false } }, { projectTrusted: true });
  const loader = new DefaultResourceLoader({ cwd: dir, agentDir: dir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [resolve("integrations/pi/qwen-tool-names.ts"), resolve("integrations/pi/qwen-task-plan.ts")], systemPrompt: "Stable synthetic SDK prefix." });
  await loader.reload();
  assert.deepEqual(loader.getExtensions().errors, []);
  let mutations = 0;
  const mutating = (name) => ({ name, label: name, description: "Synthetic mutation sentinel", parameters: { type: "object", properties: {} },
    execute: async () => { mutations++; return { content: [{ type: "text", text: "Mutation sentinel executed" }], details: {} }; } });
  const create = async (customTools = [mutating("unknown_mutation")]) => {
    const manager = SessionManager.create(dir, join(dir, "sessions"));
    manager.appendMessage({ role: "user", content: "Synthetic task with multiple steps", timestamp: 1 });
    const { session } = await createAgentSession({ cwd: dir, agentDir: dir, model, modelRuntime, settingsManager, resourceLoader: loader,
      sessionManager: manager, customTools });
    await session.bindExtensions({ uiContext: { notify: () => {}, setWidget: () => {}, setStatus: () => {}, setWorkingMessage: () => {} } });
    t.after(async () => { await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" }); session.dispose(); });
    return { session, manager };
  };
  const { session, manager } = await create();
  assert.ok(session.getActiveToolNames().includes("manage_task_plan"), "extension activates manage_task_plan even when configured tools omit it");
  assert.ok(!session.getActiveToolNames().includes("qwen_plan"), "the original plan alias stays inactive");
  assert.ok(!session.getAllTools().some((tool) => tool.name === "qwen_plan"), "historical names are not registered tools");
  for (const name of ["read_file", "run_shell_command"]) {
    assert.ok(session.getActiveToolNames().includes(name));
    assert.equal(session.getAllTools().find((tool) => tool.name === name).sourceInfo.path, resolve("integrations/pi/qwen-tool-names.ts"));
  }
  const beforePrompt = session.systemPrompt;
  assert.match(session.getAllTools().find((tool) => tool.name === "manage_task_plan").description, /substantial multi-step tasks/);
  await session.prompt("/plan on");
  assert.equal(replayTaskPlan(manager.getBranch()).mode, "plan");
  const tool = (name) => session.agent.state.tools.find((tool) => tool.name === name);
  // Agent-core calls this installed hook before executing any wrapped tool.
  // Invoke the real SDK hook without generating even a synthetic model request.
  const invoke = async (name, id, params) => {
    const decision = await session.agent.beforeToolCall({ toolCall: { id, name, arguments: params }, args: params });
    return decision?.block ? { content: [{ type: "text", text: decision.reason }], isError: true } : tool(name).execute(id, params);
  };
  const response = await tool("manage_task_plan").execute("plan-create", { action: "create", goal: "Verify synthetic integration",
    steps: [{ title: "Inspect fixture", status: "in_progress" }, { title: "Record evidence" }] });
  assert.match(response.content[0].text, /Verify synthetic integration/);
  assert.ok(manager.getBranch().some((entry) => entry.customType === TASK_PLAN_ENTRY));
  assert.equal(session.systemPrompt, beforePrompt, "mode/state updates must not rewrite stable system prompt");
  const hidden = await session.extensionRunner.emitBeforeAgentStart("Continue", undefined, beforePrompt, {});
  assert.equal(hidden.systemPrompt, undefined);
  assert.equal(hidden.messages[0].customType, TASK_PLAN_CONTEXT);
  assert.equal(hidden.messages[0].display, false);
  manager.appendCustomMessageEntry(TASK_PLAN_CONTEXT, hidden.messages[0].content, false, hidden.messages[0].details);
  assert.equal(await session.extensionRunner.emitBeforeAgentStart("Continue", undefined, beforePrompt, {}), undefined);
  const readEvent = { type: "tool_call", toolCallId: "read-1", toolName: "read_file", input: { path: "fixture.txt" } };
  assert.equal(await session.extensionRunner.emitToolCall(readEvent), undefined);
  const bashEvent = { type: "tool_call", toolCallId: "bash-1", toolName: "run_shell_command", input: { command: "rg -n synthetic ." } };
  assert.equal(await session.extensionRunner.emitToolCall(bashEvent), undefined);
  assert.match(bashEvent.input.command, /'\/usr\/bin\/rg' '--no-config'/);
  for (const [toolName, input] of [["write", { path: "unwanted.txt", content: "mutation" }], ["write_file", { path: "unwanted.txt", content: "mutation" }], ["unknown_mutation", {}], ["bash", { command: "touch unwanted.txt" }], ["run_shell_command", { command: "touch unwanted.txt" }], ["run_shell_command", { command: "git status --short" }]]) {
    assert.equal((await session.extensionRunner.emitToolCall({ type: "tool_call", toolCallId: `blocked-${toolName}`, toolName, input })).block, true);
  }
  const blocked = await invoke("unknown_mutation", "blocked-mutate", {});
  assert.match(blocked.content[0].text, /Read-only plan mode/);
  assert.equal(mutations, 0);
  const fixturePath = join(dir, "fixture.txt");
  await writeFile(fixturePath, "synthetic fixture\n");
  const read = await invoke("read_file", "read-fixture", { path: fixturePath });
  assert.match(read.content[0].text, /synthetic fixture/);
  const blockedWrite = await invoke("write_file", "blocked-write", { path: fixturePath, content: "unwanted overwrite" });
  assert.match(blockedWrite.content[0].text, /Read-only plan mode/);
  assert.equal(await readFile(fixturePath, "utf8"), "synthetic fixture\n");
  const shell = await session.extensionRunner.emitUserBash({ type: "user_bash", command: "touch unwanted.txt", excludeFromContext: false, cwd: dir });
  assert.equal(shell.result.exitCode, 1);
  const priorLeaf = manager.getLeafId();
  manager.appendMessage({ role: "user", content: "Synthetic recent turn retained by compaction", timestamp: 3 });
  manager.appendCompaction("Synthetic checkpoint", manager.getLeafId(), 100);
  assert.ok((await session.extensionRunner.emitBeforeAgentStart("Resume", undefined, beforePrompt, {})).messages.length);
  manager.branch(priorLeaf);
  await session.prompt("/plan execute");
  assert.equal(replayTaskPlan(manager.getBranch()).mode, "execute");
  await invoke("unknown_mutation", "allowed-mutate", {});
  assert.equal(mutations, 1);
  const collision = await create([mutating("read"), mutating("bash"), mutating("read_file"), mutating("run_shell_command")]);
  await collision.session.prompt("/plan on");
  for (const toolName of ["read", "bash", "read_file", "run_shell_command"]) {
    const gate = await collision.session.extensionRunner.emitToolCall({ type: "tool_call", toolCallId: "collision", toolName,
      input: ["bash", "run_shell_command"].includes(toolName) ? { command: "cat fixture.txt" } : { path: fixturePath } });
    assert.equal(gate.block, true);
    assert.match(gate.reason, /unverified implementation/);
  }
  collision.manager.appendCustomEntry(TASK_PLAN_ENTRY, { version: 99, mode: "execute", plan: null });
  assert.equal((await collision.session.extensionRunner.emitUserBash({ type: "user_bash", command: "touch unwanted.txt", excludeFromContext: false, cwd: dir })).result.exitCode, 1);
});
