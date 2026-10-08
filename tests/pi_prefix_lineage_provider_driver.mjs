import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SourceTextModule, SyntheticModule } from "node:vm";
import { createPrefixLineage } from "../integrations/pi/qwen-prefix-lineage.mjs";

const records = [];
const wireBodies = [];
const input = { messages: [{ role: "user", content: "private-original" }] };
const contextBefore = JSON.stringify(input);
class EventStream { constructor() { this.events = []; } push(value) { this.events.push(value); } end() { this.ended = true; } }
class OpenAI {
  constructor(options) {
    this.chat = { completions: { create(params) {
      return { async withResponse() {
        await options.fetch("http://127.0.0.1/v1/chat/completions", { method: "POST", body: JSON.stringify(params) });
        return { data: (async function* () {
          yield { id: "chatcmpl-random-fixture", choices: [{ delta: { content: "private-response" }, finish_reason: null }] };
          yield { id: "chatcmpl-random-fixture", choices: [{ delta: {}, finish_reason: "stop" }] };
        })(), response: { status: 200, headers: new Headers() } };
      } };
    } } };
  }
}
const stubs = {
  default: OpenAI,
  createPrefixLineage: (options) => createPrefixLineage({ ...options, emit: (record) => records.push(record) }),
  createTransportDiagnostics: ({ fetch }) => ({ fetch, failure: () => undefined }),
  TRANSPORT_ERROR: Symbol("transport"),
  calculateCost: () => {}, clampThinkingLevel: (value) => value,
  formatProviderError: (value) => String(value), normalizeProviderError: (value) => value,
  AssistantMessageEventStream: EventStream, shortHash: () => "fixture",
  headersToRecord: () => ({}), parseStreamingJson: JSON.parse,
  getProviderEnvValue: () => undefined, retryProviderRequest: async (callback) => callback(),
  sanitizeSurrogates: (value) => value,
  appendGrammarToolInputJsonDelta: () => undefined, createGrammarToolInputProperties: () => new Map(),
  getGrammarToolInput: () => "", getJsonSchemaToolParameters: () => ({}),
  resolveGrammarConstrainedSampling: () => undefined, resolveJsonSchemaStrictSampling: () => undefined,
  buildCopilotDynamicHeaders: () => ({}), hasCopilotVisionInput: () => false,
  clampOpenAIPromptCacheKey: (value) => value, buildBaseOptions: () => ({}),
  clampReasoning: (value) => value, MIN_ANSWER_TOKENS: 1,
  transformMessages: (messages) => messages,
};
const provider = new SourceTextModule(readFileSync(process.argv[2], "utf8"));
await provider.link((specifier, parent) => {
  const exports = new Set();
  for (const match of parent.identifier === provider.identifier
    ? readFileSync(process.argv[2], "utf8").matchAll(/import\s+(.*?)\s+from\s+"([^"]+)";/g) : []) {
    if (match[2] !== specifier) continue;
    if (match[1].includes("{")) {
      for (const item of match[1].split("{")[1].split("}")[0].split(",")) if (item.trim()) exports.add(item.trim());
    } else exports.add("default");
  }
  return new SyntheticModule([...exports], function () {
    for (const name of exports) { assert.ok(name in stubs, `missing dependency stub ${name}`); this.setExport(name, stubs[name]); }
  });
});
await provider.evaluate();
const stream = provider.namespace.stream({ id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate",
  provider: "qwen-r9700", api: "openai-completions", baseUrl: "http://127.0.0.1/v1", reasoning: true,
  compat: { supportsFinishReason: true, thinkingFormat: "chat-template", chatTemplateKwargs: { preserve_thinking: true } } }, input, {
  apiKey: "synthetic-unused", sessionId: "fixture-session",
  fetch: async (url, options) => { wireBodies.push(JSON.parse(options.body)); return new Response(""); },
  onPayload: (params) => ({ ...params, messages: [{ role: "user", content: "private-after-hook" }],
    kv_transfer_params: { qwen_chat: { id: "a".repeat(64), generation: "b".repeat(64) } } }),
});
for (let i = 0; !stream.ended && i < 100; i++) await new Promise((resolve) => setTimeout(resolve, 1));
assert.equal(stream.ended, true);
assert.equal(stream.events.at(-1).type, "done", JSON.stringify(stream.events.at(-1)));
assert.equal(wireBodies[0].messages[0].content, "private-after-hook");
assert.equal(JSON.stringify(input), contextBefore);
assert.deepEqual(records.map((record) => record.stage), ["context_before_conversion", "provider_converted",
  "provider_sdk_input", "provider_wire", "provider_response_identity", "provider_response_assembled"]);
assert.equal(records[2].equal, false);
assert.equal(records[2].first_changed_message, 0);
assert.equal(records[3].equal, true);
assert.equal(records.at(-1).coverage_complete, true);
assert.equal(records.at(-1).text_bytes, Buffer.byteLength("private-response"));
assert.equal(JSON.stringify(records).includes("private-"), false);
console.log("provider boundaries qualified");
