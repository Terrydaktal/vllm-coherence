// Run the installed Pi tool-continuation scheduler and production TS extension
// against synthetic messages and an in-memory provider. No real model requests.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir, homedir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
test("installed Pi auto-compacts a tool continuation with a bullet marker and visible stage updates", {
  skip: !existsSync(join(root, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000,
}, async (t) => {
  const mod = (path) => import(pathToFileURL(join(root, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { AssistantMessageEventStream } = await mod("pi-ai/dist/utils/event-stream.js");
  const agentDir = await mkdtemp(join(tmpdir(), "radiance-compaction-sdk-"));
  const previous = Object.fromEntries(["PI_CODING_AGENT_DIR", "QWEN_RADIANCE_CACHE_ABI", "PI_OFFLINE"].map((name) => [name, process.env[name]]));
  process.env.PI_CODING_AGENT_DIR = agentDir;
  process.env.PI_OFFLINE = "1";
  delete process.env.QWEN_RADIANCE_CACHE_ABI; // Cache hook is exercised with injected cleanup in its own test.
  t.after(async () => {
    for (const [name, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[name]; else process.env[name] = value;
    }
    await rm(agentDir, { recursive: true, force: true });
  });
  const model = { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", name: "Synthetic Radiance", provider: "qwen-r9700", api: "openai-completions",
    baseUrl: "http://fixture.invalid/v1", reasoning: true, input: ["text"], contextWindow: 32768, maxTokens: 8192,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: true, reserveTokens: 8192, keepRecentTokens: 1500 },
    retry: { enabled: false } }, { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd: agentDir, agentDir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [resolve("integrations/pi/qwen-radiance-compaction.ts"), resolve("integrations/pi/qwen-radiance-cache.mjs")],
    systemPrompt: "Synthetic compaction fixture. Use the fixture tool once." });
  await resourceLoader.reload();
  assert.deepEqual(resourceLoader.getExtensions().errors, []);
  const headings = ["Goal", "Current Authoritative State", "Constraints & Invariants", "Progress", "Measurements & Evidence", "Key Decisions",
    "Rejected / Failed Approaches", "Unresolved Questions & Hypotheses", "Next Steps", "Critical Context"];
  const checkpoint = headings.map((heading) => `### ${heading}\nSynthetic fixture.`).join("\n\n");
  let summaries = 0, ordinaryRequests = 0;
  const events = [], messages = [], notices = [], requests = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.ok(String(url).startsWith("http://fixture.invalid/"), "unexpected network destination");
    const body = JSON.parse(options.body);
    requests.push(String(url));
    if (String(url).endsWith("/tokenize")) {
      return Response.json({ tokens: body.prompt ? (body.prompt.includes("</think>") ? [4, 6, 7] : [4, 5]) :
        body.add_generation_prompt ? [1, 2, 3, 4, 5] : [1, 2] });
    }
    assert.equal(String(url), "http://fixture.invalid/v1/completions");
    assert.deepEqual(body.prompt, [1, 2, 3, 4, 6, 7]);
    assert.equal(body.max_tokens, model.contextWindow - body.prompt.length);
    assert.ok(body.max_tokens > model.maxTokens, "ordinary response limits must not cap the checkpoint");
    summaries++;
    const frames = [
      { choices: [{ index: 0, text: checkpoint + "\n- COMPACTION_SUMMARY_COMPLETE", finish_reason: "stop" }] },
      { usage: { prompt_tokens: 6, completion_tokens: 350, prompt_tokens_details: { cached_tokens: 2 } } },
      "[DONE]",
    ];
    return new Response(frames.map((frame) => `data: ${typeof frame === "string" ? frame : JSON.stringify(frame)}\n\n`).join(""),
      { headers: { "Content-Type": "text/event-stream" } });
  });
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: () => {
      ordinaryRequests++;
      assert.ok(ordinaryRequests <= 2, "unexpected continuation loop");
      const first = ordinaryRequests === 1;
      const message = { role: "assistant", api: model.api, provider: model.provider, model: model.id, timestamp: Date.now(),
        content: first ? [{ type: "toolCall", id: "fixture-call", name: "lookup", arguments: {} }] : [{ type: "text", text: "Synthetic task complete." }],
        stopReason: first ? "toolUse" : "stop",
        usage: { input: first ? 24500 : 40, output: 100, cacheRead: 0, cacheWrite: 0, totalTokens: first ? 24600 : 140,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
      const stream = new AssistantMessageEventStream();
      queueMicrotask(() => {
        stream.push({ type: "start", partial: { ...message, content: [] } });
        stream.push({ type: "done", reason: message.stopReason, message });
        stream.end();
      });
      return stream;
    } };
  const sessionManager = SessionManager.create(agentDir, join(agentDir, "sessions"));
  sessionManager.appendMessage({ role: "user", content: "Earlier synthetic request.", timestamp: Date.now() - 2 });
  sessionManager.appendMessage({ role: "assistant", api: model.api, provider: model.provider, model: model.id, timestamp: Date.now() - 1,
    content: [{ type: "text", text: "Earlier synthetic answer. ".repeat(100) }], stopReason: "stop",
    usage: { input: 100, output: 100, cacheRead: 0, cacheWrite: 0, totalTokens: 200,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });
  const { session } = await createAgentSession({ cwd: agentDir, agentDir, model, modelRuntime,
    resourceLoader, sessionManager, settingsManager, tools: ["lookup"], customTools: [{ name: "lookup", label: "Lookup", description: "Synthetic lookup",
      parameters: { type: "object", properties: {}, additionalProperties: false }, execute: async () => ({ content: [{ type: "text", text: "synthetic value ".repeat(300) }], details: {} }) }] });
  const unsubscribe = session.subscribe((event) => events.push(event));
  try {
    await session.bindExtensions({ uiContext: { setWidget: (_key, lines) => {
      assert.equal(lines, undefined, "compaction must not create a pinned widget");
    }, setStatus: (_key, value) => assert.equal(value, undefined, "compaction must not create a pinned footer"),
    notify: (value) => notices.push(value),
    setWorkingMessage: (message) => messages.push({ message, whileCompacting: events.at(-1)?.type === "compaction_start" }) } });
    await session.prompt("Use the synthetic lookup tool, then finish.");
    assert.equal(ordinaryRequests, 2, JSON.stringify({ notices,
      events: events.map((event) => ({ type: event.type, reason: event.reason, aborted: event.aborted, errorMessage: event.errorMessage })),
      lastError: sessionManager.buildSessionContext().messages.at(-1)?.errorMessage }));
    assert.equal(summaries, 1);
    assert.equal(requests.length, 5);
    const entries = sessionManager.getEntries().filter((entry) => entry.type === "compaction");
    assert.equal(entries.length, 1);
    assert.equal(entries[0].summary, checkpoint);
    assert.equal(entries[0].fromHook, true);
    assert.equal(sessionManager.buildSessionContext().messages.at(-1).content[0].text, "Synthetic task complete.");
    assert.equal(events.filter((event) => event.type === "compaction_start" && event.reason === "threshold").length, 1);
    assert.equal(events.filter((event) => event.type === "compaction_end" && !event.aborted).length, 1);
    assert.ok(messages.some((update) => update.whileCompacting && update.message?.includes("Generate checkpoint")));
    assert.ok(messages.some((update) => update.message?.includes("Compacted")));
    assert.equal(messages.at(-1).message, undefined, "the next model request must not inherit compaction progress");
    assert.deepEqual(notices, []);
    const reportFile = (await readdir(join(agentDir, "radiance-compaction-receipts"))).find((name) => name.endsWith(".progress.json"));
    const report = JSON.parse(await readFile(join(agentDir, "radiance-compaction-receipts", reportFile), "utf8"));
    assert.equal(report.state, "complete");
    assert.equal(report.transcriptAppended, true);
    assert.equal(report.tokens.cacheRead, 2);
  } finally {
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    unsubscribe(); session.dispose();
  }
});
