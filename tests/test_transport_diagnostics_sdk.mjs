// Exercise the installed adapter and real session persistence with synthetic
// requests only. These tests never contact the model or read existing sessions.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, rm, writeFile, readFile } from "node:fs/promises";
import { tmpdir, homedir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createServer } from "node:http";
import { TRANSPORT_ERROR } from "../integrations/pi/qwen-transport-diagnostics.mjs";
import { TRANSPORT_ERROR_ENTRY } from "../integrations/pi/qwen-radiance-errors.mjs";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const mod = (path) => import(pathToFileURL(join(root, path)));
const installed = { skip: !existsSync(join(root, "pi-ai/dist/api/openai-completions.js")), timeout: 20000 };
const model = { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", name: "Synthetic Radiance", provider: "qwen-r9700",
  api: "openai-completions", baseUrl: "http://fixture.invalid/v1", reasoning: true, input: ["text"],
  contextWindow: 253792, maxTokens: 32768, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
const context = { messages: [{ role: "user", content: "SYNTHETIC_PRIVATE_REQUEST", timestamp: 1 }], systemPrompt: "Synthetic fixture." };
const fetchTimeout = async () => { throw new TypeError("fetch failed", { cause: Object.assign(new Error("SYNTHETIC_PRIVATE_ERROR"),
  { name: "ConnectTimeoutError", code: "UND_ERR_CONNECT_TIMEOUT" }) }); };

test("installed SDK timeout loses its cause but the provider preserves our safe record", installed, async () => {
  const { stream } = await mod("pi-ai/dist/api/openai-completions.js");
  const message = await stream(model, context, { apiKey: "SYNTHETIC_PRIVATE_KEY", fetch: fetchTimeout, maxRetries: 0 }).result();
  assert.equal(message.stopReason, "error");
  assert.match(message.errorMessage, /Request timed out/);
  assert.equal(message[TRANSPORT_ERROR].phase, "connect");
  assert.equal(message[TRANSPORT_ERROR].causes[1].code, "UND_ERR_CONNECT_TIMEOUT");
  assert.doesNotMatch(JSON.stringify(message[TRANSPORT_ERROR]), /SYNTHETIC_PRIVATE/);
  assert.doesNotMatch(JSON.stringify(message), /UND_ERR_CONNECT_TIMEOUT/);
});

test("real loopback refusal records the Undici socket cause without touching inference", installed, async () => {
  const { stream } = await mod("pi-ai/dist/api/openai-completions.js");
  const server = createServer();
  await new Promise((done) => server.listen(0, "127.0.0.1", done));
  const port = server.address().port;
  await new Promise((done) => server.close(done));
  const message = await stream({ ...model, baseUrl: `http://127.0.0.1:${port}/v1` }, context,
    { apiKey: "synthetic", maxRetries: 0, timeoutMs: 1000 }).result();
  assert.equal(message.stopReason, "error");
  assert.ok(message[TRANSPORT_ERROR].causes.some((cause) => cause.code === "ECONNREFUSED"));
  assert.equal(message[TRANSPORT_ERROR].phase, "connect");
  assert.equal(message[TRANSPORT_ERROR].headers_ms, null);
});

test("response-stream socket failure preserves partial content and headers timing", installed, async () => {
  const { stream } = await mod("pi-ai/dist/api/openai-completions.js");
  const fetch = async () => new Response(new ReadableStream({ start(controller) {
    controller.enqueue(new TextEncoder().encode('data: {"choices":[{"index":0,"delta":{"content":"Synthetic partial"},"finish_reason":null}]}\n\n'));
    setTimeout(() => controller.error(new TypeError("terminated", { cause: Object.assign(new Error(), { code: "UND_ERR_SOCKET" }) })), 30);
  } }), { headers: { "Content-Type": "text/event-stream" } });
  const message = await stream(model, context, { apiKey: "synthetic", fetch, maxRetries: 0 }).result();
  assert.equal(message.stopReason, "error");
  assert.equal(message.content[0].text, "Synthetic partial");
  assert.equal(message[TRANSPORT_ERROR].phase, "stream");
  assert.equal(message[TRANSPORT_ERROR].http_status, 200);
  assert.ok(message[TRANSPORT_ERROR].headers_ms >= 0);
});

test("installed provider reconnects before submission without duplicating the model request", installed, async (t) => {
  const { stream } = await mod("pi-ai/dist/api/openai-completions.js");
  let received = 0, attempts = 0;
  const server = createServer((request, response) => {
    received++;
    request.resume();
    request.on("end", () => {
      response.writeHead(200, { "Content-Type": "text/event-stream" });
      response.end('data: {"choices":[{"index":0,"delta":{"content":"Recovered once"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n');
    });
  });
  await new Promise((done) => server.listen(0, "127.0.0.1", done));
  t.after(() => { server.closeAllConnections(); server.close(); });
  const fetch = async (...args) => {
    if (++attempts === 1) return fetchTimeout();
    return globalThis.fetch(...args);
  };
  const result = await stream({ ...model, baseUrl: `http://127.0.0.1:${server.address().port}/v1` }, context,
    { apiKey: "synthetic", fetch, maxRetries: 0 }).result();
  assert.equal(result.stopReason, "stop");
  assert.equal(result.content[0].text, "Recovered once");
  assert.equal(received, 1);
  assert.equal(attempts, 2);
  assert.equal(result[TRANSPORT_ERROR], undefined);
});

test("transport report survives session persistence and expands outside model context", installed, async (t) => {
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { CustomEntryComponent } = await mod("pi-coding-agent/dist/modes/interactive/components/custom-entry.js");
  const { initTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { stream } = await mod("pi-ai/dist/api/openai-completions.js");
  initTheme("dark", false);
  const directory = await mkdtemp(join(tmpdir(), "transport-errors-sdk-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const extension = join(directory, "errors.ts");
  await writeFile(extension, `import { Text } from "@earendil-works/pi-tui";
import { installRadianceErrors } from ${JSON.stringify(resolve("integrations/pi/qwen-radiance-errors.mjs"))};
export default function (pi) { installRadianceErrors(pi, { Text, probe: async () => { throw new Error("unexpected backend lookup"); } }); }
`);
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }, { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd: directory, agentDir: directory, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [extension], systemPrompt: "Synthetic connection failure fixture." });
  await resourceLoader.reload();
  assert.deepEqual(resourceLoader.getExtensions().errors, []);
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "synthetic" }, env: {} }),
    streamSimple: (m, context, options) => stream(m, context, { ...options, apiKey: "synthetic", fetch: fetchTimeout, maxRetries: 0 }) };
  const sessionManager = SessionManager.create(directory, join(directory, "sessions"));
  const { session } = await createAgentSession({ cwd: directory, agentDir: directory, model, modelRuntime,
    resourceLoader, sessionManager, settingsManager, tools: [] });
  t.after(() => session.dispose());
  const events = [];
  session.subscribe((event) => events.push(event));
  await session.bindExtensions({ uiContext: { notify: () => {} } });
  await session.prompt("Synthetic request.");
  const saved = (await readFile(sessionManager.getSessionFile(), "utf8")).trim().split("\n").map(JSON.parse);
  const entries = saved.filter((entry) => entry.type === "custom" && entry.customType === TRANSPORT_ERROR_ENTRY);
  assert.equal(entries.length, 1);
  assert.equal(entries[0].data.causes[1].code, "UND_ERR_CONNECT_TIMEOUT");
  assert.doesNotMatch(JSON.stringify(saved), /SYNTHETIC_PRIVATE/);
  const reopened = SessionManager.open(sessionManager.getSessionFile());
  assert.doesNotMatch(JSON.stringify(reopened.buildSessionContext().messages), /UND_ERR_CONNECT_TIMEOUT|transport-error|fixture.invalid/);
  const errorIndex = events.findIndex((event) => event.type === "message_end" && event.message.role === "assistant");
  assert.ok(events.findIndex((event) => event.type === "entry_appended") > errorIndex);
  const renderer = resourceLoader.getExtensions().extensions.find((ext) => ext.entryRenderers?.has(TRANSPORT_ERROR_ENTRY)).entryRenderers.get(TRANSPORT_ERROR_ENTRY);
  const component = new CustomEntryComponent(entries[0], renderer);
  assert.match(component.render(100).join("\n"), /ctrl\+o to expand/);
  component.setExpanded(true);
  assert.match(component.render(100).join("\n"), /Underlying cause chain/);
});
