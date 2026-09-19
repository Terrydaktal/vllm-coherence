// Exercise the actual installed OpenAI adapter, agent loop and progress extension
// with an in-memory SSE provider and synthetic tool. No network or file edits.
import assert from "node:assert/strict";
import test from "node:test";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import progress from "../integrations/pi/qwen-progress.mjs";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
test("buffered SSE usage reaches Pi before a complete edit can execute", {
  skip: !existsSync(join(root, "pi-ai/dist/api/openai-completions.js")), timeout: 10000,
}, async (t) => {
  const { stream } = await import(pathToFileURL(join(root, "pi-ai/dist/api/openai-completions.js")));
  const { Agent } = await import(pathToFileURL(join(root, "pi-agent-core/dist/agent.js")));
  const handlers = new Map(), rows = [], received = [];
  progress({ on: (name, handler) => handlers.set(name, handler) });
  const model = { id: "synthetic", name: "Synthetic", api: "openai-completions", provider: "qwen-r9700",
    baseUrl: "http://fixture.invalid/v1", reasoning: true, input: ["text"], contextWindow: 32768, maxTokens: 8192,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const ctx = { mode: "tui", model, getContextUsage: () => ({ tokens: 100 }),
    ui: { setWorkingMessage: (value) => rows.push(value), setStatus() {} } };
  const emit = (name, event) => handlers.get(name)?.(event, ctx);
  t.after(() => emit("session_shutdown", {}));
  let enqueue, executions = 0, requests = 0;
  const buffered = Promise.withResolvers();
  const args = { path: "synthetic.txt", edits: [{ oldText: "before", newText: "after" }] };
  const tool = { name: "edit", label: "edit", description: "Synthetic edit",
    parameters: { type: "object", properties: { path: { type: "string" }, edits: { type: "array", items: { type: "object" } } }, required: ["path", "edits"] },
    execute: async (_id, input) => {
      assert.deepEqual(input, args); executions++;
      assert.ok(rows.includes("Qwen applying edit 0s • model output ended"));
      return { content: [{ type: "text", text: "synthetic success" }], details: {} };
    } };
  const usage = (output) => ({ prompt_tokens: 100, completion_tokens: output, total_tokens: 100 + output });
  const choice = (delta, output, finish_reason = null) => ({ choices: [{ index: 0, delta, finish_reason }], usage: usage(output) });
  const fetcher = async (url) => {
    assert.equal(String(url), "http://fixture.invalid/v1/chat/completions");
    assert.equal(++requests, 1);
    const body = new ReadableStream({ start(controller) {
      enqueue = (frame) => {
        controller.enqueue(new TextEncoder().encode(`data: ${typeof frame === "string" ? frame : JSON.stringify(frame)}\n\n`));
        if (frame === "[DONE]") controller.close();
      };
      enqueue(choice({ reasoning_content: "synthetic thinking" }, 100));
      enqueue(choice({ tool_calls: [{ index: 0, id: "edit-one", type: "function", function: { name: "edit", arguments: '{"path":"synthetic.txt","edits":' } }] }, 111));
      enqueue({ choices: [], usage: usage(150) });
      enqueue(choice({}, 200));
    } });
    return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
  };
  const agent = new Agent({ initialState: { model, systemPrompt: "Synthetic fixture", tools: [tool] },
    streamFn: (m, context, options) => stream(m, context, { ...options, fetch: fetcher }),
    getApiKey: () => "synthetic",
    onPayload: (payload) => emit("before_provider_request", { payload }),
    shouldStopAfterTurn: () => true });
  agent.subscribe((event) => {
    emit(event.type, event);
    if (event.type === "agent_end" && !received.includes(200)) {
      buffered.reject(new Error("agent ended before buffered progress: " + JSON.stringify(
        agent.state.messages.map((message) => ({ role: message.role, error: message.errorMessage })))));
    }
    if (event.type === "message_update" && event.assistantMessageEvent.type === "usage_update") {
      const count = event.assistantMessageEvent.partial.usage.output;
      received.push(count);
      if (count === 200) buffered.resolve();
    }
  });
  const running = agent.prompt("Perform the synthetic edit.");
  t.after(() => agent.abort());
  await buffered.promise;
  assert.deepEqual(received, [150, 200]);
  assert.equal(executions, 0, "usage is progress, not permission to execute partial JSON");
	assert.match(rows.at(-1), /generating edit arguments: 200 tok/);
	assert.doesNotMatch(rows.at(-1), /reasoning count unavailable|\([\d,]+ reasoning\)/);
  enqueue(choice({ tool_calls: [{ index: 0, function: { arguments: JSON.stringify(args.edits) + "}" } }] }, 220, "tool_calls"));
  enqueue({ choices: [], usage: usage(220) });
  enqueue("[DONE]");
  await running;
  assert.equal(executions, 1);
	assert.ok(rows.every((row) => !row?.includes("reasoning count unavailable")));
	assert.ok(rows.every((row) => !row?.includes("(0 reasoning)")));
  assert.equal(agent.state.messages.find((message) => message.role === "assistant").usage.output, 220);
});
