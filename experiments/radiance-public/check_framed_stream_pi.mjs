// Isolated fixture: exercise the installed Pi client with an in-memory SSE
// response. This does not connect to an endpoint or execute any tool.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

const root = process.argv[2];
if (!root) throw new Error("Pass the installed pi-ai/dist directory");
const { stream } = await import(pathToFileURL(resolve(root, "api/openai-completions.js")));
const { isRetryableAssistantError } = await import(pathToFileURL(resolve(root, "utils/retry.js")));
const fixture = JSON.parse(readFileSync(0, "utf8"));
const model = {
  id: "number-lookup-fixture", name: "Harmless in-memory fixture", api: "openai-completions",
  provider: "fixture", baseUrl: "http://fixture.invalid/v1", reasoning: true,
  input: ["text"], contextWindow: 8192, maxTokens: 512,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
let requests = 0;
const events = stream(model, {
  messages: [{ role: "user", content: "Synthetic number lookup.", timestamp: 0 }],
}, {
  apiKey: "in-memory-test-only",
  fetch: async () => {
    requests += 1;
    return new Response(fixture.sse, { headers: { "content-type": "text/event-stream" } });
  },
});
const message = await events.result();
assert.equal(requests, 1, "provider must not retry the fixture");
assert.equal(message.stopReason, fixture.stopReason);
assert.equal(message.content.filter(c => c.type === "thinking").map(c => c.thinking).join(""), fixture.reasoning);
assert.equal(message.content.filter(c => c.type === "text").map(c => c.text).join(""), fixture.content);
assert.equal(message.content.filter(c => c.type === "toolCall").length, fixture.calls);
for (const call of message.content.filter(c => c.type === "toolCall")) {
  assert.equal(call.name, "lookup_number");
  assert.deepEqual(call.arguments, { key: "beta" });
}
assert.equal(isRetryableAssistantError(message), false, "agent retry classifier must reject protocol failures");
if (fixture.stopReason === "error") assert.match(message.errorMessage, /Earlier text retained/);
process.stdout.write(JSON.stringify({ passed: true, requests, stopReason: message.stopReason, retryable: false }) + "\n");
