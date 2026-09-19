import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir, homedir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { BACKEND_ERROR_ENTRY } from "../integrations/pi/qwen-radiance-errors.mjs";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
test("installed Pi renders and expands backend traces without putting them in model context", {
  skip: !existsSync(join(root, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000,
}, async (t) => {
  const mod = (path) => import(pathToFileURL(join(root, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { AssistantMessageEventStream } = await mod("pi-ai/dist/utils/event-stream.js");
  const { CustomEntryComponent } = await mod("pi-coding-agent/dist/modes/interactive/components/custom-entry.js");
  const { initTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  initTheme("dark", false);
  const directory = await mkdtemp(join(tmpdir(), "radiance-errors-sdk-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const fixture = { schema: "urn:qwen-r9700:backend-error:v1", status: "found", backend: { ready: false, running: false },
    incident: { id: "synthetic-error", timestamp: 10000, container_id: "a".repeat(64), summary: "RuntimeError: synthetic engine failure",
      traceback: 'Traceback (most recent call last):\n  File "/opt/vllm/synthetic_backend.py", line 42, in run\nRuntimeError: synthetic engine failure' } };
  const extension = join(directory, "errors.ts");
  await writeFile(extension, `import { Text } from "@earendil-works/pi-tui";
import { installRadianceErrors } from ${JSON.stringify(resolve("integrations/pi/qwen-radiance-errors.mjs"))};
export default function (pi) { installRadianceErrors(pi, { Text, probe: async () => (${JSON.stringify(fixture)}) }); }
`);
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }, { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd: directory, agentDir: directory, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [extension], systemPrompt: "Synthetic backend error fixture." });
  await resourceLoader.reload();
  assert.deepEqual(resourceLoader.getExtensions().errors, []);
  const model = { id: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", name: "Synthetic Radiance", provider: "qwen-r9700",
    api: "openai-completions", baseUrl: "http://fixture.invalid/v1", reasoning: true, input: ["text"],
    contextWindow: 253792, maxTokens: 32768, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: () => {
      const message = { role: "assistant", api: model.api, provider: model.provider, model: model.id, timestamp: Date.now(),
        content: [], stopReason: "error", errorMessage: "EngineCore encountered an issue. See stack trace (above) for the root cause.",
        usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
      const stream = new AssistantMessageEventStream();
      queueMicrotask(() => { stream.push({ type: "error", reason: "error", error: message }); stream.end(); });
      return stream;
    } };
  const sessionManager = SessionManager.create(directory, join(directory, "sessions"));
  const { session } = await createAgentSession({ cwd: directory, agentDir: directory, model, modelRuntime, resourceLoader,
    sessionManager, settingsManager, tools: [] });
  const events = [];
  const unsubscribe = session.subscribe((event) => events.push(event));
  try {
    await session.bindExtensions({ uiContext: { notify: () => {} } });
    await session.prompt("Synthetic request that returns a backend failure.");
    const entries = sessionManager.getEntries().filter((entry) => entry.type === "custom" && entry.customType === BACKEND_ERROR_ENTRY);
    assert.equal(entries.length, 1);
    assert.equal(events.filter((event) => event.type === "entry_appended").length, 1);
    const errorIndex = events.findIndex((event) => event.type === "message_end" && event.message.role === "assistant");
    assert.ok(events.findIndex((event) => event.type === "entry_appended") > errorIndex);
    const messages = sessionManager.buildSessionContext().messages;
    assert.doesNotMatch(JSON.stringify(messages), /synthetic_backend.py|See stack trace \(above\)/);
    const renderer = resourceLoader.getExtensions().extensions.find((ext) => ext.entryRenderers?.has(BACKEND_ERROR_ENTRY))
      .entryRenderers.get(BACKEND_ERROR_ENTRY);
    const component = new CustomEntryComponent(entries[0], renderer);
    const collapsed = component.render(80).join("\n");
    assert.match(collapsed, /ctrl\+o to expand/);
    assert.doesNotMatch(collapsed, /synthetic_backend.py/);
    component.setExpanded(true);
    assert.match(component.render(80).join("\n"), /synthetic_backend.py/);
    component.setExpanded(false);
    assert.doesNotMatch(component.render(80).join("\n"), /synthetic_backend.py/);
  } finally {
    unsubscribe();
    session.dispose();
  }
});
