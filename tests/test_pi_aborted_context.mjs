// Installed Pi SDK and an in-memory SSE provider; all messages are synthetic.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import {
  CONTEXT_POLICY_ENTRY, THINKING_PURGE_ENTRY, filteredRequestContext,
} from "../integrations/pi/qwen-context-policy.mjs";

const SDK_ROOT = process.env.QWEN_TEST_PI_ROOT ?? process.env.SDK_ROOT ??
  join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const installed = { skip: !existsSync(join(SDK_ROOT, "pi-ai/dist/api/transform-messages.js")), timeout: 10000 };
const load = (path) => import(pathToFileURL(join(SDK_ROOT, path)));
const templates = ["models-radiance-public-clean-snapshot.json", "models-radiance-uncensored.json"];
const usage = { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, totalTokens: 15,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
const user = (content, timestamp) => ({ role: "user", content, timestamp });
const thinking = (value = "Synthetic partial reasoning.") => ({ type: "thinking", thinking: value,
  thinkingSignature: "reasoning_content" });
const partialTool = { type: "toolCall", id: "unexecuted-call", name: "lookup", arguments: { value: "partial" } };
const tools = [{ name: "lookup", description: "Synthetic lookup", parameters: {
  type: "object", properties: { value: { type: "string" } }, required: ["value"],
} }];

function modelFrom(template = templates[0]) {
  const config = JSON.parse(readFileSync(new URL(`../integrations/pi/${template}`, import.meta.url), "utf8"));
  const provider = config.providers["qwen-r9700"];
  return { ...provider.models[0], provider: "qwen-r9700", api: provider.api,
    compat: structuredClone(provider.compat), baseUrl: "http://fixture.invalid/v1" };
}

function aborted(model, content = [thinking(), { type: "text", text: "Synthetic partial answer." }, partialTool]) {
  return { role: "assistant", api: model.api, provider: model.provider, model: model.id,
    content: structuredClone(content), stopReason: "aborted", errorMessage: "Synthetic cancellation.", usage, timestamp: 2 };
}

function freeze(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) freeze(child);
  }
  return value;
}

const frame = (delta, finish_reason = null) => ({ choices: [{ index: 0, delta, finish_reason }] });
const encode = (value) => new TextEncoder().encode(`data: ${typeof value === "string" ? value : JSON.stringify(value)}\n\n`);

function transport({ interrupt = false } = {}) {
  const requests = [];
  return { requests, fetch: async (url, options) => {
    assert.equal(String(url), "http://fixture.invalid/v1/chat/completions", "only the synthetic provider may be used");
    requests.push(JSON.parse(options.body));
    if (interrupt) {
      return new Response(new ReadableStream({ start(controller) {
        controller.enqueue(encode(frame({ reasoning_content: "Synthetic partial reasoning." })));
        controller.enqueue(encode(frame({ content: "Synthetic partial answer." })));
        controller.enqueue(encode(frame({ tool_calls: [{ index: 0, id: partialTool.id, type: "function",
          function: { name: partialTool.name, arguments: '{"value":"partial' } }] })));
        options.signal.addEventListener("abort", () => controller.error(new DOMException("Synthetic cancellation.", "AbortError")), { once: true });
      } }), { headers: { "Content-Type": "text/event-stream" } });
    }
    return new Response([frame({ content: "Synthetic next answer." }, "stop"), "[DONE]"]
      .map((value) => new TextDecoder().decode(encode(value))).join(""),
    { headers: { "Content-Type": "text/event-stream" } });
  } };
}

async function request(model, messages, options = {}) {
  const { streamSimpleOpenAICompletions } = await load("pi-ai/dist/compat.js");
  const wire = transport();
  const result = await streamSimpleOpenAICompletions(model, { systemPrompt: "Synthetic fixture.", messages, tools },
    { apiKey: "fixture", reasoning: "xhigh", fetch: wire.fetch, maxRetries: 0, ...options }).result();
  assert.equal(result.stopReason, "stop", result.errorMessage);
  assert.equal(wire.requests.length, 1);
  return wire.requests[0];
}

for (const template of templates) {
  test(`${template}: interrupted reasoning and prose reach the next request without partial tools`, installed, async () => {
    const { streamSimpleOpenAICompletions } = await load("pi-ai/dist/compat.js");
    const model = modelFrom(template), signal = new AbortController(), wire = transport({ interrupt: true });
    const initial = user("Begin the synthetic task.", 1);
    const stream = streamSimpleOpenAICompletions(model, { messages: [initial], tools }, {
      apiKey: "fixture", reasoning: "xhigh", signal: signal.signal, fetch: wire.fetch, maxRetries: 0,
    });
    for await (const event of stream) {
      if (event.type === "toolcall_delta") signal.abort();
    }
    const saved = await stream.result();
    assert.equal(saved.stopReason, "aborted", saved.errorMessage);
    assert.ok(saved.content.some((block) => block.type === "toolCall"), "exercise a genuinely unfinished tool call");
    const history = freeze([initial, saved, user("Continue the synthetic task.", 3)]);
    const before = structuredClone(history);
    const body = await request(model, history);
    assert.deepEqual(body.messages.filter((message) => message.role === "assistant"), [{
      role: "assistant", content: "Synthetic partial answer.", reasoning_content: "Synthetic partial reasoning.",
    }]);
    assert.equal(body.messages.some((message) => message.role === "tool" || message.tool_calls), false);
    assert.doesNotMatch(JSON.stringify(body.messages), /unexecuted-call|No result provided/);
    assert.equal(body.chat_template_kwargs.preserve_thinking, model.compat.chatTemplateKwargs.preserve_thinking);
    assert.deepEqual(history, before, "the saved interrupted message and its aborted status stay unchanged");
  });
}

test("the installed transformer retains only nonempty text and unredacted thinking from an aborted reply", installed, async () => {
  const { transformMessages } = await load("pi-ai/dist/api/transform-messages.js");
  const model = modelFrom();
  const retained = [thinking(), { type: "text", text: "Synthetic partial answer." }];
  const saved = aborted(model, [...retained, partialTool,
    { type: "image", data: "synthetic", mimeType: "image/png" },
    { type: "audio", data: "synthetic" },
    { ...thinking("Opaque synthetic reasoning."), redacted: true },
    thinking(""), thinking(" \n "), { type: "text", text: "" }, { type: "text", text: " \n " },
  ]);
  const history = freeze([user("Synthetic first request.", 1), saved, user("Synthetic continuation.", 3)]);
  const before = structuredClone(history), result = transformMessages(history, model);
  assert.equal(result.length, 3);
  assert.notEqual(result[1], saved, "replay uses a request-only message clone");
  assert.notEqual(result[1].content, saved.content);
  assert.equal(result[1].stopReason, "aborted");
  assert.deepEqual(result[1].content, retained);
  assert.equal(result.some((message) => message.role === "toolResult"), false, "never fabricate results for a partial tool call");
  assert.deepEqual(history, before);
});

test("empty, tool-only, redacted-only and whitespace-only aborted replies remain omitted", installed, async () => {
  const { transformMessages } = await load("pi-ai/dist/api/transform-messages.js");
  const model = modelFrom();
  for (const content of [[], [partialTool], [thinking("")], [thinking(" \n ")],
    [{ type: "text", text: " \n " }], [{ ...thinking("Opaque synthetic reasoning."), redacted: true }]]) {
    const messages = freeze([user("Synthetic request.", 1), aborted(model, content), user("Continue.", 3)]);
    const transformed = transformMessages(messages, model);
    assert.deepEqual(transformed, [messages[0], messages[2]], JSON.stringify(content));
    const body = await request(model, messages);
    assert.equal(body.messages.some((message) => message.role === "assistant" || message.role === "tool"), false);
  }
});

test("thinking-only and text-only interruptions are useful context without requiring a tool result", installed, async () => {
  const model = modelFrom();
  for (const [content, expected] of [
    [[thinking()], { role: "assistant", content: null, reasoning_content: "Synthetic partial reasoning." }],
    [[{ type: "text", text: "Synthetic partial answer." }], { role: "assistant", content: "Synthetic partial answer." }],
  ]) {
    const body = await request(model, freeze([user("Synthetic request.", 1), aborted(model, content), user("Continue.", 3)]));
    assert.deepEqual(body.messages.filter((message) => message.role === "assistant"), [expected]);
    assert.equal(body.messages.some((message) => message.role === "tool"), false);
  }
});

test("errors and interruptions outside the exact Radiance model, provider and API remain omitted", installed, async () => {
  const { transformMessages } = await load("pi-ai/dist/api/transform-messages.js");
  const model = modelFrom();
  const cases = [
    ["error", model, { stopReason: "error" }],
    ["source provider", model, { provider: "another-provider" }],
    ["source API", model, { api: "openai-responses" }],
    ["source model", model, { model: modelFrom(templates[1]).id }],
    ["target provider", { ...model, provider: "another-provider" }, {}],
    ["target API", { ...model, api: "openai-responses" }, {}],
    ["target model", { ...model, id: "another-model" }, {}],
    ["unrelated qwen model", { ...model, id: "qwen3.8-27b-frozenlock" }, {}],
  ];
  for (const [label, selected, changes] of cases) {
    const saved = { ...aborted(selected), ...changes };
    const history = freeze([user("Synthetic request.", 1), saved, user("Continue.", 3)]);
    assert.deepEqual(transformMessages(history, selected), [history[0], history[2]], label);
    if (selected.api === "openai-completions") {
      const body = await request(selected, history);
      assert.equal(body.messages.some((message) => message.role === "assistant" || message.role === "tool"), false, label);
    }
  }
  const completed = { ...aborted(model, [thinking(), { type: "text", text: "Completed synthetic answer." }]), stopReason: "stop" };
  assert.equal(transformMessages([completed], { ...model, id: "another-model" }).length, 1,
    "the existing treatment of completed cross-model messages is unchanged");
});

for (const part of ["message", "thinking", "purge-thinking"]) {
  test(`${part} exclusions apply before replaying aborted context`, installed, async () => {
    const model = modelFrom(), old = aborted(model), future = { ...aborted(model, [thinking("Synthetic future reasoning.")]), timestamp: 4 };
    const entries = freeze([
      { type: "message", id: "user", message: user("Synthetic request.", 1) },
      { type: "message", id: "old", message: old },
      { type: "message", id: "future", message: future },
    ]);
    const policy = part === "purge-thinking"
      ? { type: "custom", customType: THINKING_PURGE_ENTRY,
        data: { version: 1, entryIds: ["old"], preserveFutureThinking: true } }
      : { type: "custom", customType: CONTEXT_POLICY_ENTRY,
        data: { version: 1, changes: [{ entryId: "old", part, excluded: true }], preserveFutureThinking: true } };
    const ctx = { model, sessionManager: {
      getBranch: () => [...entries, policy], buildContextEntries: () => entries,
    } };
    const messages = freeze([...entries.map((entry) => structuredClone(entry.message)), user("Synthetic next request.", 5)]);
    const before = structuredClone(messages);
    const body = await request(model, filteredRequestContext(messages, ctx));
    const replies = body.messages.filter((message) => message.role === "assistant");
    assert.equal(replies.some((message) => message.reasoning_content === "Synthetic partial reasoning."), false);
    assert.equal(replies.some((message) => message.content === "Synthetic partial answer."), part !== "message");
    assert.ok(replies.some((message) => message.reasoning_content === "Synthetic future reasoning."), "future thinking remains available");
    assert.equal(body.messages.some((message) => message.role === "tool" || message.tool_calls), false);
    assert.deepEqual(messages, before, "saved thinking, tool calls and stop reasons are preserved");
  });
}

test("aborted replay leaves preserve_thinking at its configured value", installed, async () => {
  for (const preserve of [false, true]) {
    const model = modelFrom();
    model.compat.chatTemplateKwargs.preserve_thinking = preserve;
    freeze(model);
    const body = await request(model, freeze([user("Synthetic request.", 1), aborted(model), user("Continue.", 3)]));
    assert.equal(body.chat_template_kwargs.preserve_thinking, preserve);
    assert.equal(model.compat.chatTemplateKwargs.preserve_thinking, preserve);
    assert.equal(body.messages.find((message) => message.role === "assistant")?.reasoning_content, "Synthetic partial reasoning.");
  }
});
