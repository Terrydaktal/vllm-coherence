// Real pinned Pi session/JSONL + production compactor, with an offline provider.
// Only synthetic messages and temporary paths are used; no model or GPU access.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const available = existsSync(join(root, "pi-coding-agent/dist/core/sdk.js"));
const options = { skip: !available, timeout: 15000 };
const OBJECTIVE = "Original synthetic task: finish the arithmetic repair and report its measured result.";
const PROSE = "Partial repair explanation: αβ\nThe measured result is ";
const THINKING = "Keep the accepted arithmetic; investigate the remaining measurement. ";
const DRAFT_ID = "unexecuted-stream-draft";
const headings = ["Goal", "Current Authoritative State", "Constraints & Invariants", "Progress", "Measurements & Evidence", "Key Decisions",
  "Rejected / Failed Approaches", "Unresolved Questions & Hypotheses", "Next Steps", "Critical Context"];
const checkpoint = headings.map((heading) => `### ${heading}\nSynthetic checkpoint: continue the original arithmetic repair.`).join("\n\n");
const mod = (path) => import(pathToFileURL(join(root, path)));
const usage = (input, output = 0, cacheRead = 0, cacheWrite = 0) => ({ input, output, cacheRead, cacheWrite,
  totalTokens: input + output + cacheRead + cacheWrite, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } });
const turn = () => new Promise((resolve) => setImmediate(resolve));
async function until(predicate, description, timeout = 2000) {
  const deadline = performance.now() + timeout;
  while (!predicate()) {
    assert.ok(performance.now() < deadline, `Timed out waiting for ${description}`);
    await turn();
  }
}
const snapshot = (value) => JSON.parse(JSON.stringify(value));
const partialEntries = (entries) => entries.filter((entry) => entry.type === "message" && entry.message.role === "assistant" &&
  entry.message.content.some((block) => block.type === "text" && block.text.startsWith(PROSE)));

async function fixture(t, { enabled = true, provider = "qwen-r9700", contextWindow = 253792, reserveTokens = 8192,
  compactor = "success", historicalUsage = 200, priorCheckpoint = false, honorAbort = true,
  includeDraft = true, expectedStop = "aborted", partialText = PROSE, partialThinking = THINKING, beforeCheckpoint = async () => {}, realProvider = false } = {}) {
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { AssistantMessageEventStream } = await mod("pi-ai/dist/utils/event-stream.js");
  const { transformMessages } = await mod("pi-ai/dist/api/transform-messages.js");
  const { streamSimpleOpenAICompletions } = await mod("pi-ai/dist/compat.js");
  const directory = await mkdtemp(join(tmpdir(), "pi-stream-compaction-"));
  const previous = Object.fromEntries(["PI_CODING_AGENT_DIR", "QWEN_RADIANCE_CACHE_ABI", "PI_OFFLINE"].map((name) => [name, process.env[name]]));
  process.env.PI_CODING_AGENT_DIR = directory;
  process.env.PI_OFFLINE = "1";
  delete process.env.QWEN_RADIANCE_CACHE_ABI;
  t.after(async () => {
    for (const [name, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[name]; else process.env[name] = value;
    }
    await rm(directory, { recursive: true, force: true });
  });
  const model = { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", name: "Synthetic streaming compaction", provider,
    api: "openai-completions", baseUrl: "http://fixture.invalid/v1", reasoning: true, input: ["text"], contextWindow, maxTokens: 8192,
    compat: { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false, supportsUsageInStreaming: true,
      maxTokensField: "max_completion_tokens", thinkingFormat: "chat-template", chatTemplateKwargs: { enable_thinking: true, preserve_thinking: true } },
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled, reserveTokens, keepRecentTokens: 1500 }, retry: { enabled: false } }, { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd: directory, agentDir: directory, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [resolve("integrations/pi/qwen-radiance-compaction.ts"), ...(realProvider ? [resolve("integrations/pi/qwen-progress.mjs")] : [])],
    systemPrompt: "Synthetic streaming-compaction fixture." });
  await resourceLoader.reload();
  assert.deepEqual(resourceLoader.getExtensions().errors, []);
  const sessionManager = SessionManager.create(directory, join(directory, "sessions"));
  const assistant = (content, tokens = usage(100, 100), stopReason = "stop", extras = {}) => ({ role: "assistant", api: model.api,
    provider: model.provider, model: model.id, timestamp: Date.now(), content, usage: tokens, stopReason, ...extras });
  const earlierUser = sessionManager.appendMessage({ role: "user", content: "Earlier synthetic completed work.", timestamp: Date.now() });
  sessionManager.appendMessage(assistant([{ type: "text", text: "Earlier completed context. ".repeat(400) }], usage(historicalUsage - 100, 100)));
  const completedCall = sessionManager.appendMessage(assistant([{ type: "toolCall", id: "completed-call", name: "probe", arguments: {} }], usage(100, 100), "toolUse"));
  const completedResult = sessionManager.appendMessage({ role: "toolResult", toolCallId: "completed-call", toolName: "probe",
    content: [{ type: "text", text: "Earlier accepted result: 320 / 320 exact." }], isError: false, timestamp: Date.now() });
  if (priorCheckpoint) sessionManager.appendCompaction("Synthetic previous checkpoint", earlierUser, historicalUsage);
  const events = [], requests = [], tokenizationBodies = [], notices = [], working = [], controls = [], order = [];
  let session, summaries = 0, toolExecutions = 0, compactionObserved = false;
  let running;
  t.mock.method(globalThis, "fetch", async (url, fetchOptions) => {
    assert.ok(String(url).startsWith("http://fixture.invalid/"), "real network requests are forbidden");
    const body = JSON.parse(fetchOptions.body);
    requests.push(String(url));
    if (!compactionObserved) {
      compactionObserved = true;
      const persisted = (await readFile(sessionManager.getSessionFile(), "utf8")).trim().split("\n").map(JSON.parse);
      const interrupted = partialEntries(persisted);
      assert.equal(interrupted.length, 1, "the original partial must already be saved exactly once before compactor I/O");
      assert.equal(interrupted[0].message.stopReason, expectedStop);
      assert.deepEqual(interrupted[0].message.content.map((block) => block.type), includeDraft ? ["thinking", "text", "toolCall"] : ["thinking", "text"]);
      assert.equal(interrupted[0].message.content[0].thinking, partialThinking);
      assert.equal(interrupted[0].message.content[1].text, partialText);
      if (includeDraft) assert.equal(interrupted[0].message.content[2].id, DRAFT_ID);
      order.push("partial-on-disk-before-compactor");
    }
    if (String(url).endsWith("/tokenize")) {
      tokenizationBodies.push(body);
      return Response.json({ tokens: body.prompt ? (body.prompt.includes("</think>") ? [4, 6, 7] : [4, 5]) :
        body.add_generation_prompt ? [1, 2, 3, 4, 5] : [1, 2] });
    }
    assert.equal(String(url), "http://fixture.invalid/v1/completions");
    summaries++;
    await beforeCheckpoint({ session, sessionManager, model });
    if (compactor === "fail") return Response.json({ error: { message: "Synthetic checkpoint failure" } }, { status: 503 });
    if (compactor === "cancel") {
      session.abortCompaction();
      throw fetchOptions.signal.reason ?? new Error("Synthetic checkpoint cancellation");
    }
    const frames = [{ choices: [{ index: 0, text: checkpoint + "\nCOMPACTION_SUMMARY_COMPLETE", finish_reason: "stop" }] },
      { usage: { prompt_tokens: 6, completion_tokens: 350, prompt_tokens_details: { cached_tokens: 2 } } }, "[DONE]"];
    return new Response(frames.map((frame) => `data: ${typeof frame === "string" ? frame : JSON.stringify(frame)}\n\n`).join(""),
      { headers: { "Content-Type": "text/event-stream" } });
  });
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, checkAuth: async () => true, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: (_model, context, streamOptions) => {
      assert.ok(controls.length < 3, "a stale usage counter must not cause an automatic continuation loop");
      const stream = new AssistantMessageEventStream();
      const control = { stream, signal: streamOptions.signal, context: snapshot(context), requestMessages: snapshot(transformMessages(context.messages, model)),
        partial: assistant([], usage(0)), ended: false, wireBody: undefined, controller: undefined,
        frame(delta, tokens) {
          assert.ok(this.controller, "the actual provider must have opened its synthetic SSE connection");
          this.controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: null }],
            ...(tokens ? { usage: tokens } : {}) })}\n\n`));
        },
        update(type, message, { preserveTimestamp = true } = {}) {
          this.partial = { ...message, timestamp: preserveTimestamp ? this.partial.timestamp : message.timestamp };
          stream.push({ type, contentIndex: 0, delta: "", partial: this.partial });
        },
        finish(reason = "stop", message = this.partial) {
          if (this.ended) return;
          this.ended = true;
          const final = { ...message, timestamp: this.partial.timestamp, stopReason: reason };
          if (realProvider) {
            const content = final.content.filter((block) => block.type === "text").map((block) => block.text).join("");
            const raw = { prompt_tokens: final.usage.input + final.usage.cacheRead + final.usage.cacheWrite,
              completion_tokens: final.usage.output, prompt_tokens_details: { cached_tokens: final.usage.cacheRead } };
            this.controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content }, finish_reason: reason }], usage: raw })}\n\ndata: [DONE]\n\n`));
            this.controller.close();
            return;
          }
          stream.push(reason === "aborted" || reason === "error" ? { type: "error", reason, error: final } : { type: "done", reason, message: final });
          stream.end();
        } };
      controls.push(control);
      order.push(`acting-request-${controls.length}`);
      streamOptions.signal.addEventListener("abort", () => {
        order.push(`provider-abort-${controls.indexOf(control) + 1}`);
        if (honorAbort && !realProvider) control.finish("aborted", { ...control.partial, errorMessage: "Synthetic provider interrupted" });
      }, { once: true });
      if (realProvider) return streamSimpleOpenAICompletions(model, context, { ...streamOptions, apiKey: "fixture", maxRetries: 0, fetch: async (url, fetchOptions) => {
        assert.equal(String(url), "http://fixture.invalid/v1/chat/completions", "the acting adapter must use only synthetic SSE");
        control.wireBody = JSON.parse(fetchOptions.body);
        return new Response(new ReadableStream({ start(controller) {
          control.controller = controller;
          fetchOptions.signal.addEventListener("abort", () => controller.error(new DOMException("Synthetic provider interrupted", "AbortError")), { once: true });
        } }), { headers: { "Content-Type": "text/event-stream" } });
      } });
      queueMicrotask(() => stream.push({ type: "start", partial: control.partial }));
      return stream;
    } };
  ({ session } = await createAgentSession({ cwd: directory, agentDir: directory, model, modelRuntime, resourceLoader,
    sessionManager, settingsManager, tools: ["probe"], customTools: [{ name: "probe", label: "Probe", description: "Synthetic probe",
      parameters: { type: "object", properties: {}, additionalProperties: false }, execute: async () => {
        toolExecutions++;
        return { content: [{ type: "text", text: "Unexpected draft execution" }], details: {} };
      } }] }));
  const unsubscribe = session.subscribe((event) => { events.push(snapshot(event)); order.push(event.type); });
  t.after(async () => {
    await session.abort();
    await running?.catch(() => {});
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    unsubscribe(); session.dispose();
  });
  await session.bindExtensions({ uiContext: { setWidget: () => {}, setStatus: () => {}, notify: (message) => notices.push(message),
    setWorkingMessage: (message) => working.push(message) } });
  const f = { session, sessionManager, SessionManager, model, directory, assistant, controls, events, notices, working, requests, tokenizationBodies, order,
    earlierUser, completedCall, completedResult, summaries: () => summaries, toolExecutions: () => toolExecutions,
    start() { running = session.prompt(OBJECTIVE); return running; }, running: () => running,
    partial(tokens = usage(238000, 2000), extras = {}) { return assistant([{ type: "thinking", thinking: partialThinking }, { type: "text", text: partialText },
      ...(includeDraft ? [{ type: "toolCall", id: DRAFT_ID, name: "probe", arguments: { incomplete: "draft" } }] : [])], tokens, "stop", extras); },
    async first() {
      await until(() => controls.length >= 1 && (!realProvider || controls[0].controller) && events.some((event) => event.type === "message_start"), "first provider stream")
        .catch((error) => { throw new Error(JSON.stringify({ controls: controls.length, events, notices }), { cause: error }); });
      return controls[0];
    },
    async settle() { await until(() => !session.isStreaming && !session.isCompacting && events.some((event) => event.type === "agent_settled"), "settled session"); await running; } };
  return f;
}

test("streaming 240K usage saves the exact interrupted attempt before one checkpoint and automatically resumes the original task", options, async (t) => {
  const f = await fixture(t);
  f.start();
  const first = await f.first();
  first.update("thinking_delta", f.partial(usage(230000, 100, 8000)));
  await until(() => f.events.some((event) => event.type === "message_update"), "below-threshold partial");
  assert.equal(first.signal.aborted, false, "fresh 238100 usage remains below 240K");
  first.update("usage_update", f.partial(usage(230000, 2000, 8000)));
  await until(() => first.signal.aborted, "240K stream interruption");
  await until(() => f.controls.length === 2, "automatic continuation after checkpoint commit");
  const resumed = f.controls[1];
  assert.equal(f.summaries(), 1);
  const compactions = f.sessionManager.getEntries().filter((entry) => entry.type === "compaction");
  assert.equal(compactions.length, 1);
  assert.ok(f.order.indexOf("partial-on-disk-before-compactor") < f.order.indexOf("acting-request-2"));
  assert.match(JSON.stringify(resumed.context), /Original synthetic task/);
  assert.ok(resumed.context.messages.some((message) => message.role === "user" && JSON.stringify(message.content).includes(OBJECTIVE)), "the original objective remains available");
  assert.ok(!resumed.context.messages.some((message) => message.role === "toolResult" && message.toolCallId === DRAFT_ID));
  const requestDrafts = resumed.requestMessages.flatMap((message) => Array.isArray(message.content) ? message.content : [])
    .filter((block) => block.type === "toolCall" && block.id === DRAFT_ID);
  assert.deepEqual(requestDrafts, [], "unexecuted draft is filtered from the provider request");
  assert.ok(JSON.stringify(f.tokenizationBodies).includes(JSON.stringify(PROSE).slice(1, -1)), "compaction receives original partial prose");
  assert.ok(JSON.stringify(f.tokenizationBodies).includes(THINKING), "compaction receives original partial reasoning");
  assert.equal(f.toolExecutions(), 0);
  resumed.update("usage_update", f.assistant([{ type: "text", text: "Finishing original task" }], usage(40000, 100)));
  resumed.finish("stop", f.assistant([{ type: "text", text: "Original task complete: 320 / 320 exact." }], usage(40000, 200)));
  await f.settle();
  assert.equal(f.controls.length, 2);
  assert.equal(f.events.filter((event) => event.type === "compaction_start").length, 1);
  assert.equal(f.events.filter((event) => event.type === "compaction_end" && !event.aborted && event.result).length, 1);
  const entries = f.sessionManager.getEntries();
  assert.equal(partialEntries(entries).length, 1);
  assert.equal(entries.filter((entry) => entry.type === "message" && entry.message.role === "user" && JSON.stringify(entry.message.content).includes(OBJECTIVE)).length, 1);
  assert.ok(entries.some((entry) => entry.id === f.completedCall));
  assert.ok(entries.some((entry) => entry.id === f.completedResult));
  assert.equal(entries.filter((entry) => entry.type === "message" && entry.message.role === "toolResult" && entry.message.toolCallId === DRAFT_ID).length, 0);
  const reopened = f.SessionManager.open(f.sessionManager.getSessionFile(), join(f.directory, "sessions"));
  assert.deepEqual(partialEntries(reopened.getEntries()), JSON.parse(JSON.stringify(partialEntries(entries))), "saved interrupted transcript survives reopening exactly");
  assert.equal(reopened.getEntries().filter((entry) => entry.type === "compaction").length, 1);
  const interrupted = partialEntries(reopened.getEntries())[0].message;
  assert.equal(interrupted.stopReason, "aborted");
  assert.match(interrupted.errorMessage, /interrupt|compact/i, "transcript status distinguishes an interrupted attempt from completed output");
});

test("the reserve guard interrupts below 240K using prompt, cache-read, cache-write and output accounting", options, async (t) => {
  const f = await fixture(t, { contextWindow: 32768, reserveTokens: 8192 });
  f.start();
  const first = await f.first();
  first.update("usage_update", f.partial(usage(16000, 2575, 4000, 2000)));
  await until(() => f.events.some((event) => event.type === "message_update"), "reserve-minus-one update");
  await turn();
  assert.equal(first.signal.aborted, false, "24575 tokens remain below the 24576-token reserve boundary");
  first.update("usage_update", f.partial(usage(16000, 2576, 4000, 2000)));
  await until(() => first.signal.aborted, "reserve-boundary interruption");
  await until(() => f.controls.length === 2, "reserve-triggered continuation");
  f.controls[1].finish("stop", f.assistant([{ type: "text", text: "Reserve-guard task complete." }], usage(100, 100)));
  await f.settle();
  assert.equal(f.summaries(), 1);
  assert.equal(f.toolExecutions(), 0);
});

for (const [name, settings, tokens, extras] of [
  ["disabled compaction", { enabled: false }, usage(238000, 2000), {}],
  ["another provider", { provider: "synthetic-other-provider" }, usage(238000, 2000), {}],
  ["another model's streamed usage", {}, usage(238000, 2000), { model: "another-model" }],
  ["zero usage", {}, usage(0), {}],
  ["output-only usage", {}, usage(0, 240000), {}],
  ["missing usage", {}, undefined, { usage: undefined }],
  ["inconsistent totals", {}, { ...usage(238000, 2000), totalTokens: 250000 }, {}],
  ["old usage before a checkpoint", { historicalUsage: 245000, priorCheckpoint: true }, usage(0), {}],
]) {
  test(`${name} cannot interrupt the current acting stream`, options, async (t) => {
    const f = await fixture(t, settings);
    f.start();
    const first = await f.first();
    first.update("usage_update", f.partial(tokens, extras));
    await until(() => f.events.some((event) => event.type === "message_update"), "negative-control usage update");
    await turn();
    await turn();
    assert.equal(first.signal.aborted, false);
    assert.equal(f.summaries(), 0);
    first.finish("stop", f.assistant([{ type: "text", text: "Ordinary stream completed." }], usage(100, 100)));
    await f.settle();
    assert.equal(f.controls.length, 1);
    assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, settings.priorCheckpoint ? 1 : 0);
    assert.equal(f.toolExecutions(), 0);
  });
}

for (const compactor of ["fail", "cancel"]) {
  test(`a ${compactor === "fail" ? "failed" : "cancelled"} streaming checkpoint preserves the attempt and never restarts the task`, options, async (t) => {
    const f = await fixture(t, { compactor });
    f.start();
    const first = await f.first();
    first.update("usage_update", f.partial());
    await until(() => first.signal.aborted, "stream interruption");
    await f.settle();
    assert.equal(f.controls.length, 1, "failure/cancellation requires a user's later decision to continue");
    assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, 0);
    assert.equal(partialEntries(f.sessionManager.getEntries()).length, 1);
    assert.equal(f.toolExecutions(), 0);
    const ended = f.events.filter((event) => event.type === "compaction_end");
    assert.equal(ended.length, 1);
    assert.ok(ended[0].aborted || ended[0].errorMessage || !ended[0].result);
    const reopened = f.SessionManager.open(f.sessionManager.getSessionFile(), join(f.directory, "sessions"));
    assert.equal(partialEntries(reopened.getEntries()).length, 1);
  });
}

test("a user cancellation racing with threshold settlement suppresses checkpoint and automatic restart", options, async (t) => {
  const f = await fixture(t);
  f.start();
  const first = await f.first();
  const unsubscribe = f.session.subscribe((event) => {
    if (event.type === "message_end" && event.message.role === "assistant" && event.message.stopReason === "aborted") void f.session.abort();
  });
  t.after(unsubscribe);
  first.update("usage_update", f.partial());
  await until(() => first.signal.aborted, "threshold/user-cancel race");
  await f.settle();
  assert.equal(f.controls.length, 1);
  assert.equal(f.summaries(), 0);
  assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, 0);
  assert.equal(partialEntries(f.sessionManager.getEntries()).length, 1);
});

test("batched repeated threshold frames commit once and post-checkpoint zero usage cannot retrigger stale usage", options, async (t) => {
  const f = await fixture(t);
  f.start();
  const first = await f.first();
  for (const output of [2000, 2001, 2002]) first.update("usage_update", f.partial(usage(238000, output)));
  await until(() => f.controls.length === 2, "one continuation after batched cutoff");
  const resumed = f.controls[1];
  resumed.update("text_delta", f.assistant([{ type: "text", text: "Fresh resumed prose without usage yet." }], usage(0)));
  await until(() => f.events.some((event) => event.type === "message_update" && event.message.content?.[0]?.text === "Fresh resumed prose without usage yet."), "resumed zero-usage event");
  await turn();
  assert.equal(resumed.signal.aborted, false, "pre-checkpoint counts cannot trigger the new request");
  assert.equal(f.controls.length, 2);
  resumed.finish("stop", f.assistant([{ type: "text", text: "Batched interruption task complete." }], usage(100, 100)));
  await f.settle();
  assert.equal(f.summaries(), 1);
  assert.equal(partialEntries(f.sessionManager.getEntries()).length, 1);
  assert.equal(f.events.filter((event) => event.type === "compaction_start").length, 1);
  assert.equal(f.toolExecutions(), 0);
});

test("a natural stop racing with threshold interruption compacts once and keeps the completed response complete", options, async (t) => {
  const f = await fixture(t, { honorAbort: false, includeDraft: false, expectedStop: "stop" });
  f.start();
  const first = await f.first();
  first.update("usage_update", f.partial());
  first.finish("stop", f.partial());
  await f.settle();
  assert.equal(f.summaries(), 1);
  assert.equal(f.controls.length, 1, "naturally completed answers do not need an automatic task continuation");
  assert.equal(partialEntries(f.sessionManager.getEntries()).length, 1);
  assert.equal(partialEntries(f.sessionManager.getEntries())[0].message.stopReason, "stop");
  assert.equal(f.events.filter((event) => event.type === "compaction_start").length, 1);
});

for (const terminal of ["toolUse", "stop"]) test(`a terminal ${terminal} with a tool draft racing with threshold abort never executes or invents a result`, options, async (t) => {
  const f = await fixture(t, { honorAbort: false });
  f.start();
  const first = await f.first();
  first.update("usage_update", f.partial());
  first.finish(terminal, f.partial());
  await until(() => f.controls.length === 2, "continuation after terminal-tool cutoff race");
  f.controls[1].finish("stop", f.assistant([{ type: "text", text: "Tool draft safely re-evaluated." }], usage(100, 100)));
  await f.settle();
  assert.equal(f.summaries(), 1);
  assert.equal(f.toolExecutions(), 0);
  assert.equal(partialEntries(f.sessionManager.getEntries())[0].message.stopReason, "aborted");
  assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "message" && entry.message.role === "toolResult" && entry.message.toolCallId === DRAFT_ID).length, 0);
});

for (const at of ["compaction_start", "compaction_end"]) {
  test(`Escape at ${at} prevents automatic restart, including after a checkpoint was committed`, options, async (t) => {
    const f = await fixture(t);
    f.start();
    const first = await f.first();
    const unsubscribe = f.session.subscribe((event) => { if (event.type === at) f.session.abortCompaction(); });
    t.after(unsubscribe);
    first.update("usage_update", f.partial());
    await f.settle();
    assert.equal(f.controls.length, 1);
    assert.equal(partialEntries(f.sessionManager.getEntries()).length, 1);
    const compactions = f.sessionManager.getEntries().filter((entry) => entry.type === "compaction");
    assert.equal(compactions.length, at === "compaction_end" ? 1 : 0);
    if (at === "compaction_start") assert.equal(f.summaries(), 0, "Escape must cancel before checkpoint generation starts");
  });
}

test("switching away from and back to the original model during a checkpoint invalidates continuation intent", options, async (t) => {
  const f = await fixture(t, { beforeCheckpoint: async ({ session, model }) => {
    await session.setModel({ ...model, id: "synthetic-other-model" });
    await session.setModel(model);
  } });
  f.start();
  const first = await f.first();
  first.update("usage_update", f.partial());
  await f.settle();
  assert.equal(f.session.model.id, f.model.id);
  assert.equal(f.controls.length, 1, "returning to the original identity cannot revive a cancelled intention");
  assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, 0);
  assert.equal(partialEntries(f.sessionManager.getEntries()).length, 1);
});

test("branch changes during checkpoint generation cannot attach the old interrupted task to the newly selected branch", options, async (t) => {
  const f = await fixture(t, { beforeCheckpoint: async ({ sessionManager }) => {
    sessionManager.branch(sessionManager.getEntries().find((entry) => entry.type === "message" && entry.message.role === "user").id);
  } });
  f.start();
  const first = await f.first();
  first.update("usage_update", f.partial());
  await f.settle();
  assert.equal(f.controls.length, 1);
  assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, 0, "checkpoint commits only to its original ancestry");
  assert.equal(f.sessionManager.getLeafId(), f.earlierUser);
  assert.equal(partialEntries(f.sessionManager.getEntries()).length, 1, "old branch transcript remains available");
});

test("an interrupted assistant larger than the retained-tail budget stays exact in JSONL while the checkpoint context stays bounded", options, async (t) => {
  const partialText = PROSE + "Synthetic unfinished prose fragment. ".repeat(4000);
  const partialThinking = THINKING + "Synthetic unfinished reasoning fragment. ".repeat(4000);
  const f = await fixture(t, { partialText, partialThinking });
  f.start();
  const first = await f.first();
  first.update("usage_update", f.partial());
  await until(() => f.controls.length === 2, "continuation after oversized partial checkpoint");
  const entries = f.sessionManager.getEntries();
  const saved = partialEntries(entries);
  assert.equal(saved.length, 1);
  assert.equal(saved[0].message.content[0].thinking, partialThinking);
  assert.equal(saved[0].message.content[1].text, partialText);
  assert.ok(JSON.stringify(f.controls[1].context.messages).length < 50000, "active checkpoint context excludes the oversized atomic tail");
  assert.ok(JSON.stringify(f.sessionManager.buildSessionContext().messages).length < 50000);
  const compaction = entries.find((entry) => entry.type === "compaction");
  assert.ok(compaction.summary.includes("continue the original arithmetic repair"));
  f.controls[1].update("usage_update", f.assistant([{ type: "text", text: "Resume after oversized partial." }], usage(40000, 100)));
  f.controls[1].finish("stop", f.assistant([{ type: "text", text: "Oversized partial task completed." }], usage(40000, 200)));
  await f.settle();
  assert.equal(f.controls.length, 2);
  assert.equal(f.summaries(), 1);
  assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, 1);
  const reopened = f.SessionManager.open(f.sessionManager.getSessionFile(), join(f.directory, "sessions"));
  assert.deepEqual(partialEntries(reopened.getEntries()), JSON.parse(JSON.stringify(saved)));
  assert.ok(JSON.stringify(reopened.buildSessionContext().messages).length < 50000);
  assert.equal(f.toolExecutions(), 0);
});

test("a user abort queued after checkpoint commit but before continuation wins the microtask race", options, async (t) => {
  const f = await fixture(t);
  const unsubscribe = f.session.subscribe((event) => {
    if (event.type === "compaction_end" && event.result) queueMicrotask(() => queueMicrotask(() => void f.session.abort()));
  });
  t.after(unsubscribe);
  f.start();
  const first = await f.first();
  first.update("usage_update", f.partial());
  await f.settle();
  assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, 1);
  assert.equal(f.controls.length, 1, "a pending continuation must observe the user's final cancellation");
});

for (const failureAt of [1, 2]) {
  test(`${failureAt === 1 ? "interrupted-transcript" : "committed-checkpoint"} durability failure prevents automatic continuation`, options, async (t) => {
    const f = await fixture(t);
    const sync = f.session._syncStreamCompactionTranscript.bind(f.session);
    let syncCalls = 0;
    t.mock.method(f.session, "_syncStreamCompactionTranscript", (...args) => {
      if (++syncCalls === failureAt) throw new Error("Synthetic fsync failure");
      return sync(...args);
    });
    f.start();
    const first = await f.first();
    first.update("usage_update", f.partial());
    await f.settle();
    assert.equal(f.controls.length, 1);
    assert.equal(f.sessionManager.getEntries().filter((entry) => entry.type === "compaction").length, failureAt === 2 ? 1 : 0);
    assert.equal(f.summaries(), failureAt === 2 ? 1 : 0);
    assert.equal(partialEntries(f.sessionManager.getEntries()).length, 1);
    assert.equal(f.toolExecutions(), 0);
  });
}

for (const type of ["thinking_delta", "text_delta", "toolcall_delta"]) {
  test(`fresh usage carried by ${type} interrupts immediately without waiting for a usage-only frame`, options, async (t) => {
    const f = await fixture(t);
    f.start();
    const first = await f.first();
    first.update(type, f.partial());
    await until(() => first.signal.aborted, `${type} threshold interruption`);
    await until(() => f.controls.length === 2, "continuation after content-frame threshold");
    f.controls[1].finish("stop", f.assistant([{ type: "text", text: "Content-frame interruption task complete." }], usage(100, 100)));
    await f.settle();
    assert.equal(f.summaries(), 1);
    assert.equal(f.toolExecutions(), 0);
    assert.equal(partialEntries(f.sessionManager.getEntries()).length, 1);
  });
}

test("an unrelated ordinary agent.continue cannot use the automatic interruption capability", options, async (t) => {
  const f = await fixture(t, { compactor: "fail" });
  f.start();
  const first = await f.first();
  first.update("usage_update", f.partial());
  await f.settle();
  await assert.rejects(f.session.agent.continue(), /assistant|Cannot continue/);
  assert.equal(f.controls.length, 1);
});

test("the real OpenAI SSE adapter interrupts content-bearing continuous usage and resumes with a clean wire payload", options, async (t) => {
  const f = await fixture(t, { realProvider: true });
  f.start();
  const first = await f.first();
  assert.equal(first.wireBody.stream_options.continuous_usage_stats, true, "continuous accounting is requested from the actual provider adapter");
  first.frame({ reasoning_content: THINKING }, { prompt_tokens: 238000, completion_tokens: 100, prompt_tokens_details: { cached_tokens: 8000 } });
  first.frame({ content: PROSE.slice(0, -1) });
  first.frame({ tool_calls: [{ index: 0, id: DRAFT_ID, type: "function", function: { name: "probe", arguments: '{"incomplete":"draft' } }] });
  await until(() => f.events.some((event) => event.type === "message_update" && event.assistantMessageEvent.type === "toolcall_delta"), "actual SSE partial tool frame");
  assert.equal(first.signal.aborted, false);
  first.frame({ content: PROSE.slice(-1) }, { prompt_tokens: 238000, completion_tokens: 2000, prompt_tokens_details: { cached_tokens: 8000 } });
  await until(() => first.signal.aborted, "content-bearing SSE threshold abort");
  await until(() => f.controls.length === 2 && f.controls[1].controller, "actual SSE automatic continuation");
  assert.equal(f.events.some((event) => event.type === "message_update" && event.assistantMessageEvent.type === "text_delta" &&
    event.message.usage.totalTokens === 240000), true, "240K arrives on the prose delta through the production adapter");
  const body = f.controls[1].wireBody;
  assert.ok(body.messages.some((message) => message.role === "user" && JSON.stringify(message.content).includes(OBJECTIVE)));
  assert.ok(body.messages.some((message) => message.role === "assistant" && message.content === PROSE && message.reasoning_content === THINKING));
  assert.equal(body.messages.some((message) => message.tool_calls?.some((call) => call.id === DRAFT_ID) || message.tool_call_id === DRAFT_ID), false);
  assert.doesNotMatch(JSON.stringify(body.messages), /No result provided|Operation aborted/);
  f.controls[1].finish("stop", f.assistant([{ type: "text", text: "Actual SSE original task complete." }], usage(40000, 100)));
  await f.settle();
  assert.equal(f.summaries(), 1);
  assert.equal(f.controls.length, 2);
  assert.equal(f.toolExecutions(), 0);
  const reopened = f.SessionManager.open(f.sessionManager.getSessionFile(), join(f.directory, "sessions"));
  assert.equal(partialEntries(reopened.getEntries()).length, 1);
  assert.equal(partialEntries(reopened.getEntries())[0].message.content[0].thinking, THINKING);
  assert.equal(partialEntries(reopened.getEntries())[0].message.content[1].text, PROSE);
  assert.equal(partialEntries(reopened.getEntries())[0].message.stopReason, "aborted");
  assert.equal(reopened.getEntries().filter((entry) => entry.type === "compaction").length, 1);
});
