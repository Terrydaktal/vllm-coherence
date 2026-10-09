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
    assert.ok(entries[0].summary.startsWith(`${checkpoint}\n\n`), "the validated checkpoint is preserved before deterministic continuity evidence");
    assert.ok(entries[0].summary.includes(entries[0].details.continuity.packet.text));
    assert.equal(entries[0].details.continuity.appendedToSummary, true);
    assert.ok(entries[0].details.continuity.packet.sourceIds.every((id) => sessionManager.getBranch().some((entry) => entry.id === id)),
      "continuity provenance names entries on the selected branch");
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

async function manualCutoffFixture(t, { usage, cancel = false }) {
  const mod = (path) => import(pathToFileURL(join(root, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const agentDir = await mkdtemp(join(tmpdir(), "radiance-cutoff-sdk-"));
  const previous = Object.fromEntries(["PI_CODING_AGENT_DIR", "QWEN_RADIANCE_CACHE_ABI", "PI_OFFLINE"].map((name) => [name, process.env[name]]));
  process.env.PI_CODING_AGENT_DIR = agentDir;
  process.env.PI_OFFLINE = "1";
  delete process.env.QWEN_RADIANCE_CACHE_ABI;
  t.after(async () => {
    for (const [name, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[name]; else process.env[name] = value;
    }
    await rm(agentDir, { recursive: true, force: true });
  });
  const model = { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", name: "Synthetic Radiance", provider: "qwen-r9700", api: "openai-completions",
    baseUrl: "http://fixture.invalid/v1", reasoning: true, input: ["text"], contextWindow: 32768, maxTokens: 8192,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: true, reserveTokens: 8192, keepRecentTokens: 128 },
    retry: { enabled: false } }, { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd: agentDir, agentDir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [resolve("integrations/pi/qwen-radiance-compaction.ts")], systemPrompt: "Synthetic manual cutoff fixture." });
  await resourceLoader.reload();
  const loaded = resourceLoader.getExtensions();
  assert.deepEqual(loaded.errors, []);
  const finishShortcut = loaded.extensions.flatMap((extension) => [...extension.shortcuts.values()]).find((shortcut) => shortcut.shortcut === "alt+c");
  assert.ok(finishShortcut, "production compactor registers Alt+C");
  const checkpoint = "### Goal\nKeep this exact unfinished checkpoint: αβ\n\n### Progress\n- unfinished ";
  const requests = [], events = [], messages = [], notices = [];
  let streamCancelled = false, requested = false;
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.ok(String(url).startsWith("http://fixture.invalid/"), "unexpected network destination");
    requests.push(String(url));
    const body = JSON.parse(options.body);
    if (String(url).endsWith("/tokenize")) {
      return Response.json({ tokens: body.prompt ? (body.prompt.includes("</think>") ? [4, 6, 7] : [4, 5]) :
        body.add_generation_prompt ? [1, 2, 3, 4, 5] : [1, 2] });
    }
    assert.equal(String(url), "http://fixture.invalid/v1/completions");
    assert.deepEqual(body.prompt, [1, 2, 3, 4, 6, 7]);
    const frame = { choices: [{ index: 0, text: checkpoint, finish_reason: null }], ...(usage ? { usage } : {}) };
    return new Response(new ReadableStream({
      start(controller) { controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(frame)}\n\n`)); },
      cancel() { streamCancelled = true; },
    }), { headers: { "Content-Type": "text/event-stream" } });
  });
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: () => { throw new Error("cutoff must not call the ordinary model"); } };
  const sessionManager = SessionManager.create(agentDir, join(agentDir, "sessions"));
  const originalUsage = { input: 100, output: 100, cacheRead: 0, cacheWrite: 0, totalTokens: 200,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  for (const [index, text] of ["Earlier synthetic answer. ".repeat(400), "Recent synthetic answer. ".repeat(50)].entries()) {
    sessionManager.appendMessage({ role: "user", content: `Synthetic request ${index}.`, timestamp: Date.now() });
    sessionManager.appendMessage({ role: "assistant", api: model.api, provider: model.provider, model: model.id, timestamp: Date.now(),
      content: [{ type: "text", text }], stopReason: "stop", usage: originalUsage });
  }
  const originalMessages = JSON.stringify(sessionManager.getEntries().filter((entry) => entry.type === "message"));
  const { session } = await createAgentSession({ cwd: agentDir, agentDir, model, modelRuntime,
    resourceLoader, sessionManager, settingsManager, tools: [] });
  const unsubscribe = session.subscribe((event) => events.push(event));
  t.after(async () => {
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    unsubscribe(); session.dispose();
  });
  await session.bindExtensions({ uiContext: {
    setWidget: (_key, lines) => assert.equal(lines, undefined), setStatus: (_key, value) => assert.equal(value, undefined),
    notify: (value) => notices.push(value),
    setWorkingMessage(message) {
      messages.push(message);
      if (!requested && message?.includes("Alt+C finish now")) {
        requested = true;
        finishShortcut.handler(session.extensionRunner.createContext());
        if (cancel) session.abortCompaction();
      }
    },
  } });
  return { session, sessionManager, SessionManager, agentDir, checkpoint, requests, events, messages, notices,
    originalMessages, requested: () => requested, streamCancelled: () => streamCancelled };
}

for (const usage of [undefined, { prompt_tokens: 6, completion_tokens: 37, prompt_tokens_details: { cached_tokens: 2 } }]) {
  test(`installed Pi appends the exact incomplete Alt+C checkpoint with ${usage ? "observed" : "undefined"} usage`, {
    skip: !existsSync(join(root, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000,
  }, async (t) => {
    const f = await manualCutoffFixture(t, { usage });
    const result = await f.session.compact();
    assert.equal(f.requested(), true);
    assert.equal(f.streamCancelled(), true);
    assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, 1, "Alt+C cannot launch a replacement request");
    assert.equal(f.requests.length, 5);
    assert.equal(result.summary, f.checkpoint, "no marker, missing heading or unfinished sentence is repaired");
    assert.equal(result.details.forcedCheckpoint, true);
    assert.equal(result.details.checkpointComplete, false);
    assert.equal(result.details.outputTokenCountKnown, Boolean(usage));
    assert.equal(result.details.summaryRequests, 1);
    assert.equal(result.usage?.output, usage?.completion_tokens);
    const entries = f.sessionManager.getEntries().filter((entry) => entry.type === "compaction");
    assert.equal(entries.length, 1);
    assert.equal(entries[0].summary, f.checkpoint);
    assert.deepEqual(entries[0].details, result.details);
    assert.deepEqual(entries[0].usage, result.usage);
    assert.equal(entries[0].fromHook, true);
    const reopened = f.SessionManager.open(f.sessionManager.getSessionFile(), join(f.agentDir, "sessions"));
    const saved = reopened.getEntries().find((entry) => entry.type === "compaction");
    assert.equal(saved.summary, f.checkpoint);
    const { elapsedMs: savedElapsedMs, ...savedDetails } = saved.details;
    const { elapsedMs: finalElapsedMs, ...resultDetails } = result.details;
    assert.deepEqual(savedDetails, resultDetails, "cutoff metadata persists exactly; final timing has its own linked entry");
    assert.ok(savedElapsedMs <= finalElapsedMs);
    assert.deepEqual(saved.usage, result.usage);
    assert.equal(f.events.filter((event) => event.type === "compaction_end" && !event.aborted).length, 1);
    assert.ok(f.messages.some((message) => message?.includes("User-selected cutoff; checkpoint may be incomplete.")));
    assert.equal(f.messages.at(-1), undefined);
    assert.deepEqual(f.notices, []);
  });
}

test("installed Pi Escape wins over Alt+C and prevents appending the incomplete checkpoint", {
  skip: !existsSync(join(root, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000,
}, async (t) => {
  const f = await manualCutoffFixture(t, { cancel: true });
  await assert.rejects(f.session.compact(), /Compaction cancelled/);
  assert.equal(f.requested(), true);
  assert.equal(f.streamCancelled(), true);
  assert.equal(f.requests.filter((url) => url.endsWith("/completions")).length, 1);
  assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, 0);
  assert.equal(JSON.stringify(f.sessionManager.getEntries().filter((entry) => entry.type === "message")), f.originalMessages,
    "Escape retains the original messages while allowing display-only failure diagnostics");
  const reopened = f.SessionManager.open(f.sessionManager.getSessionFile(), join(f.agentDir, "sessions"));
  assert.equal(reopened.getEntries().filter((entry) => entry.type === "compaction").length, 0);
  assert.equal(f.events.filter((event) => event.type === "compaction_end" && event.aborted).length, 1);
  assert.equal(f.messages.at(-1), undefined);
});
