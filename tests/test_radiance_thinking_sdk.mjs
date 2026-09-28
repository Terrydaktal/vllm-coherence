// Exercise the installed Pi runtime and real OpenAI request conversion against
// synthetic SSE responses. No GPU requests or private session data are used.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { THINKING_PURGE_ENTRY } from "../integrations/pi/qwen-radiance-thinking.mjs";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
test("installed Pi purges only old thinking across tools, resume and compaction without rewriting the transcript", {
  skip: !existsSync(join(root, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000,
}, async (t) => {
  const mod = (path) => import(pathToFileURL(join(root, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { streamSimple } = await mod("pi-ai/dist/api/openai-completions.js");
  const agentDir = await mkdtemp(join(tmpdir(), "radiance-thinking-sdk-"));
  const previous = Object.fromEntries(["PI_CODING_AGENT_DIR", "QWEN_RADIANCE_CACHE_ABI", "PI_OFFLINE"].map((key) => [key, process.env[key]]));
  process.env.PI_CODING_AGENT_DIR = agentDir;
  process.env.PI_OFFLINE = "1";
  delete process.env.QWEN_RADIANCE_CACHE_ABI;
  t.after(async () => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
    await rm(agentDir, { recursive: true, force: true });
  });
  const model = { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", name: "Synthetic Radiance",
    provider: "qwen-r9700", api: "openai-completions", baseUrl: "http://fixture.invalid/v1",
    reasoning: true, input: ["text"], contextWindow: 253792, maxTokens: 253792,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    compat: { thinkingFormat: "chat-template", chatTemplateKwargs: { enable_thinking: true, preserve_thinking: false } } };
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false, reserveTokens: 8192, keepRecentTokens: 10 },
    retry: { enabled: false } }, { projectTrusted: true });
  const headings = ["Goal", "Current Authoritative State", "Constraints & Invariants", "Progress", "Measurements & Evidence", "Key Decisions",
    "Rejected / Failed Approaches", "Unresolved Questions & Hypotheses", "Next Steps", "Critical Context"];
  const checkpoint = headings.map((heading) => `### ${heading}\nSynthetic checkpoint.`).join("\n\n");
  const requests = [], tokenizeRequests = [], notices = [];
  const sse = (frames) => new Response(frames.concat("[DONE]").map((frame) =>
    `data: ${typeof frame === "string" ? frame : JSON.stringify(frame)}\n\n`).join(""), { headers: { "Content-Type": "text/event-stream" } });
  const fetcher = async (url, options) => {
    assert.ok(String(url).startsWith("http://fixture.invalid/"), "unexpected network destination");
    const body = JSON.parse(options.body);
    if (String(url).endsWith("/tokenize")) {
      tokenizeRequests.push(body);
      return Response.json({ tokens: body.prompt ? (body.prompt.includes("</think>") ? [4, 6, 7] : [4, 5]) :
        body.add_generation_prompt ? [1, 2, 3, 4, 5] : [1, 2] });
    }
    if (String(url).endsWith("/v1/completions")) {
      return sse([{ choices: [{ index: 0, text: checkpoint + "\nCOMPACTION_SUMMARY_COMPLETE", finish_reason: "stop" }] },
        { usage: { prompt_tokens: 6, completion_tokens: 300 } }]);
    }
    assert.equal(String(url), "http://fixture.invalid/v1/chat/completions");
    requests.push(body);
    const first = requests.length === 1;
    const delta = first
      ? { role: "assistant", reasoning_content: "NEW_THINKING_TOOL", tool_calls: [{ index: 0, id: "new-call", type: "function",
        function: { name: "lookup", arguments: "{}" } }] }
      : { role: "assistant", reasoning_content: `NEW_THINKING_${requests.length}`, content: "New synthetic answer." };
    return sse([{ choices: [{ index: 0, delta, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: first ? "tool_calls" : "stop" }] },
      { choices: [], usage: { prompt_tokens: 50, completion_tokens: 20, total_tokens: 70 } }]);
  };
  t.mock.method(globalThis, "fetch", fetcher);
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: (selected, context, options) => streamSimple(selected, context, { ...options, apiKey: "fixture", fetch: fetcher }) };
  const usage = { input: 50, output: 20, cacheRead: 0, cacheWrite: 0, totalTokens: 70,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const sessionManager = SessionManager.create(agentDir, join(agentDir, "sessions"));
  sessionManager.appendMessage({ role: "user", content: "Old synthetic request.", timestamp: 1 });
  const oldId = sessionManager.appendMessage({ role: "assistant", api: model.api, provider: model.provider, model: model.id,
    timestamp: 2, stopReason: "toolUse", usage, content: [
      { type: "thinking", thinking: "OLD_THINKING_SECRET", thinkingSignature: "reasoning_content" },
      { type: "text", text: "Retained prose." },
      { type: "toolCall", id: "old-call", name: "lookup", arguments: { value: "retained-argument" } },
    ] });
  sessionManager.appendMessage({ role: "toolResult", toolCallId: "old-call", toolName: "lookup",
    content: [{ type: "text", text: "Retained tool result." }], timestamp: 3, isError: false });
  const create = async (manager) => {
    const resourceLoader = new DefaultResourceLoader({ cwd: agentDir, agentDir, settingsManager,
      noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
      additionalExtensionPaths: [resolve("integrations/pi/qwen-radiance-compaction.ts")], systemPrompt: "Synthetic thinking purge fixture." });
    await resourceLoader.reload();
    assert.deepEqual(resourceLoader.getExtensions().errors, []);
    const { session } = await createAgentSession({ cwd: agentDir, agentDir, model, modelRuntime, resourceLoader,
      sessionManager: manager, settingsManager, tools: ["lookup"], customTools: [{ name: "lookup", label: "Lookup",
        description: "Synthetic lookup", parameters: { type: "object", properties: {} },
        execute: async () => ({ content: [{ type: "text", text: "New tool result." }], details: {} }) }] });
    await session.bindExtensions({ uiContext: { notify: (value) => notices.push(value), setWidget: () => {},
      setStatus: () => {}, setWorkingMessage: () => {} } });
    return session;
  };
  let session = await create(sessionManager);
  try {
    await session.prompt("/purge-thinking");
    assert.equal(requests.length, 0, "the command must not send an invisible user prompt");
    const purge = sessionManager.getEntries().find((entry) => entry.customType === THINKING_PURGE_ENTRY);
    assert.deepEqual(purge.data.entryIds, [oldId]);
    await session.prompt("Continue the synthetic task.");
    assert.equal(requests.length, 2, JSON.stringify(notices));
    for (const body of requests) {
      assert.equal(body.chat_template_kwargs.preserve_thinking, true);
      assert.doesNotMatch(JSON.stringify(body.messages), /OLD_THINKING_SECRET|qwen-radiance-thinking-purge|\/purge-thinking/);
      assert.match(JSON.stringify(body.messages), /Retained prose\./);
      assert.match(JSON.stringify(body.messages), /retained-argument/);
      assert.match(JSON.stringify(body.messages), /Retained tool result\./);
    }
    assert.match(JSON.stringify(requests[1].messages), /NEW_THINKING_TOOL/);
    assert.equal(model.compat.chatTemplateKwargs.preserve_thinking, false, "the global provider setting remains unchanged");
    const path = sessionManager.getSessionFile();
    const originalLines = (await readFile(path, "utf8")).trim().split("\n");
    const oldLine = originalLines.find((line) => JSON.parse(line).id === oldId);
    assert.match(oldLine, /OLD_THINKING_SECRET/);
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    session.dispose();
    const resumed = SessionManager.open(path);
    session = await create(resumed);
    await session.prompt("Continue after resume.");
    assert.equal(requests.length, 3);
    assert.doesNotMatch(JSON.stringify(requests[2].messages), /OLD_THINKING_SECRET/);
    assert.match(JSON.stringify(requests[2].messages), /NEW_THINKING_TOOL/);
    assert.match(JSON.stringify(requests[2].messages), /NEW_THINKING_2/);
    await session.compact();
    const contextRequests = tokenizeRequests.filter((body) => body.messages);
    assert.ok(contextRequests.length > 0, JSON.stringify(notices));
    for (const body of contextRequests) {
      assert.equal(body.chat_template_kwargs.preserve_thinking, true);
      assert.doesNotMatch(JSON.stringify(body.messages), /OLD_THINKING_SECRET/);
      assert.match(JSON.stringify(body.messages), /NEW_THINKING_TOOL/);
      assert.match(JSON.stringify(body.messages), /NEW_THINKING_3/);
    }
    assert.ok(resumed.getEntries().some((entry) => entry.type === "compaction"));
    assert.ok(resumed.getBranch().some((entry) => entry.customType === THINKING_PURGE_ENTRY));
    assert.equal((await readFile(path, "utf8")).split("\n").find((line) => line && JSON.parse(line).id === oldId), oldLine);
    await session.prompt("/purge-thinking status");
    assert.match(notices.at(-1), /future thinking retained/);
    assert.equal(requests.length, 3, "status must not generate a model request");
  } finally {
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    session.dispose();
  }
});
