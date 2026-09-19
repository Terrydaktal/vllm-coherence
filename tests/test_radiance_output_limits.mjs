// Exercise the installed provider and native summarizers with synthetic messages.
// Mock fetch is the final wire boundary: no chat files or model server are used.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const installed = existsSync(join(root, "pi-ai/dist/api/openai-completions.js"));
const load = (path) => import(pathToFileURL(join(root, path)));
const context = { systemPrompt: "Synthetic output budget check.",
  messages: [{ role: "user", content: "Complete this synthetic request.", timestamp: 1 }],
  tools: [{ name: "lookup", description: "Synthetic lookup",
    parameters: { type: "object", properties: {}, additionalProperties: false } }] };

function modelFrom(name = "models-radiance-public-clean-snapshot.json") {
  const config = JSON.parse(readFileSync(new URL(`../integrations/pi/${name}`, import.meta.url), "utf8"));
  const provider = config.providers["qwen-r9700"];
  return { ...provider.models[0], provider: "qwen-r9700", api: provider.api,
    compat: provider.compat, baseUrl: "http://fixture.invalid/v1" };
}

function transport() {
  const requests = [];
  return { requests, fetch: async (url, options) => {
    assert.equal(String(url), "http://fixture.invalid/v1/chat/completions");
    requests.push(JSON.parse(options.body));
    const frames = [
      { choices: [{ index: 0, delta: { reasoning: "Synthetic thinking." }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { content: "Synthetic complete." }, finish_reason: "stop" }] },
      { choices: [], usage: { prompt_tokens: 50, completion_tokens: 40001, total_tokens: 40051,
        completion_tokens_details: { reasoning_tokens: 40000 } } },
    ];
    return new Response(frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("") + "data: [DONE]\n\n",
      { headers: { "Content-Type": "text/event-stream" } });
  } };
}

function assertUncapped(body) {
  for (const key of ["max_tokens", "max_completion_tokens", "thinking_token_budget"]) {
    assert.equal(Object.hasOwn(body, key), false, `${key} must be absent from the wire request`);
  }
}

for (const template of ["models-radiance-public-clean-snapshot.json", "models-radiance-uncensored.json"]) {
  for (const reasoning of ["off", "low", "medium", "xhigh"]) {
    test(`${template}: ${reasoning} has no fixed output or thinking cap`, { skip: !installed }, async () => {
      const { streamSimple } = await load("pi-ai/dist/api/openai-completions.js");
      const model = modelFrom(template);
      assert.equal(model.maxTokens, model.contextWindow);
      const wire = transport();
      const result = await streamSimple(model, context, { apiKey: "fixture", reasoning,
        fetch: wire.fetch, maxRetries: 0 }).result();
      assert.equal(result.stopReason, "stop", result.errorMessage);
      assert.equal(result.usage.output, 40001);
      assert.equal(wire.requests.length, 1, "no extra tokenization or generation request");
      const body = wire.requests[0];
      assertUncapped(body);
      assert.equal(body.chat_template_kwargs.enable_thinking, reasoning !== "off");
      assert.equal(body.chat_template_kwargs.reasoning_effort, reasoning === "off" ? undefined : reasoning);
      assert.equal(body.tools[0].function.name, "lookup");
      assert.equal(body.temperature, model.samplingParams.temperature);
      assert.equal(body.top_p, model.samplingParams.top_p);
    });
  }
}

test("stale model caps, context estimates and payload hooks cannot truncate Radiance output", { skip: !installed }, async () => {
  const { streamSimple } = await load("pi-ai/dist/api/openai-completions.js");
  const model = { ...modelFrom(), maxTokens: 32768, contextWindow: 8192 };
  const wire = transport();
  let hookPayload;
  const result = await streamSimple(model, { ...context,
    messages: [{ role: "user", content: "x".repeat(20000), timestamp: 1 }] }, {
    apiKey: "fixture", reasoning: "xhigh", fetch: wire.fetch, maxRetries: 0,
    onPayload: (body) => {
      assert.equal(body.max_completion_tokens, 1, "exercise Pi's near-full-context estimate clamp");
      hookPayload = Object.freeze({ ...body, max_tokens: 2048, max_completion_tokens: 32768,
        thinking_token_budget: 8192, cache_salt: "synthetic-salt",
        kv_transfer_params: { qwen_chat: { id: "a", generation: "b" } } });
      return hookPayload;
    },
  }).result();
  assert.equal(result.stopReason, "stop", result.errorMessage);
  assert.equal(wire.requests.length, 1);
  assertUncapped(wire.requests[0]);
  assert.deepEqual(wire.requests[0].messages, hookPayload.messages);
  assert.deepEqual(wire.requests[0].tools, hookPayload.tools);
  assert.equal(wire.requests[0].cache_salt, hookPayload.cache_salt);
  assert.deepEqual(wire.requests[0].kv_transfer_params, hookPayload.kv_transfer_params);
  assert.equal(hookPayload.max_tokens, 2048, "do not mutate another extension's payload");
});

test("tool-result continuations have no independent output cap", { skip: !installed }, async () => {
  const { streamSimple } = await load("pi-ai/dist/api/openai-completions.js");
  const model = modelFrom(), wire = transport();
  const result = await streamSimple(model, { ...context, messages: [...context.messages,
    { role: "assistant", api: model.api, provider: model.provider, model: model.id,
      timestamp: 2, content: [{ type: "toolCall", id: "synthetic-call", name: "lookup", arguments: {} }],
      stopReason: "toolUse", usage: { input: 20, output: 10, cacheRead: 0, cacheWrite: 0, totalTokens: 30 } },
    { role: "toolResult", toolCallId: "synthetic-call", toolName: "lookup", timestamp: 3,
      content: [{ type: "text", text: "Synthetic result." }], isError: false },
  ] }, { apiKey: "fixture", fetch: wire.fetch, maxRetries: 0 }).result();
  assert.equal(result.stopReason, "stop", result.errorMessage);
  assertUncapped(wire.requests[0]);
  assert.ok(wire.requests[0].messages.some((message) => message.role === "tool"));
});

test("native branch and fallback summaries cannot impose their smaller output budgets", { skip: !installed }, async () => {
  const { streamSimple } = await load("pi-ai/dist/api/openai-completions.js");
  const { generateBranchSummary } = await load("pi-coding-agent/dist/core/compaction/branch-summarization.js");
  const { generateSummaryWithUsage } = await load("pi-coding-agent/dist/core/compaction/compaction.js");
  const model = modelFrom(), wire = transport(), requestedBudgets = [];
  const streamFn = (selectedModel, prompt, options) => {
    requestedBudgets.push(options.maxTokens);
    return streamSimple(selectedModel, prompt, { ...options, fetch: wire.fetch, maxRetries: 0 });
  };
  const branch = await generateBranchSummary([
    { type: "message", id: "synthetic-entry", parentId: null, message: context.messages[0] },
  ], { model, apiKey: "fixture", streamFn, retry: { enabled: false } });
  assert.equal(branch.error, undefined);
  assert.ok(branch.summary.includes("Synthetic complete."));
  const summary = await generateSummaryWithUsage(context.messages, model, 16384, "fixture",
    undefined, undefined, undefined, undefined, "xhigh", streamFn);
  assert.equal(summary.text, "Synthetic complete.");
  assert.deepEqual(requestedBudgets, [2048, 13107], "exercise the actual native summary ceilings");
  assert.equal(wire.requests.length, 2);
  for (const body of wire.requests) assertUncapped(body);
});

test("unrelated providers and models retain explicit output caps", { skip: !installed }, async () => {
  const { streamSimple } = await load("pi-ai/dist/api/openai-completions.js");
  for (const change of [{ provider: "other" }, { id: "other-model" }]) {
    const model = { ...modelFrom(), ...change }, wire = transport();
    const result = await streamSimple(model, context, { apiKey: "fixture", maxTokens: 17,
      fetch: wire.fetch, maxRetries: 0 }).result();
    assert.equal(result.stopReason, "stop", result.errorMessage);
    assert.equal(wire.requests[0].max_completion_tokens, 17);
  }
});

test("an aborted payload capture still cannot dispatch a request", { skip: !installed }, async () => {
  const { streamSimple } = await load("pi-ai/dist/api/openai-completions.js");
  const wire = transport();
  const result = await streamSimple(modelFrom(), context, { apiKey: "fixture", maxTokens: 1,
    fetch: wire.fetch, maxRetries: 0, onPayload: () => { throw new Error("synthetic capture only"); },
  }).result();
  assert.equal(result.stopReason, "error");
  assert.equal(result.errorMessage, "synthetic capture only");
  assert.equal(wire.requests.length, 0);
});
