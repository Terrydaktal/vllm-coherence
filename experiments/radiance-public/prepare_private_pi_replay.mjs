// Rebuild selected private requests without sending them or printing their contents.
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { createHash } from "node:crypto";

process.umask(0o077);
process.env.PI_OFFLINE = "1";
globalThis.fetch = async () => { throw new Error("Replay preparation forbids network access"); };

const root = process.argv[2];
if (!root?.startsWith(`/run/user/${process.getuid()}/qwen-private-replay-`)) {
  throw new Error("Private replay directory must be in the user's temporary memory filesystem");
}
const source = "/home/lewis/tasks/qwen";
const cwd = "/home/lewis/tasks/legitmoney";
const installed = "/home/lewis/.local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works/pi-coding-agent/dist";
const sdk = await import(pathToFileURL(path.join(installed, "index.js")));
const { ModelRuntime } = await import(pathToFileURL(path.join(installed, "core/model-runtime.js")));
const selection = JSON.parse(await fs.readFile(path.join(root, "selection.json"), "utf8"));
const agentDir = path.join(root, "agent");
await fs.mkdir(agentDir, { mode: 0o700 });
for (const [from, to] of [
  ["integrations/pi/settings-radiance.json", "settings.json"],
  ["integrations/pi/models-radiance-public-clean-snapshot.json", "models.json"],
]) {
  await fs.copyFile(path.join(source, from), path.join(agentDir, to));
}
const modelRuntime = await ModelRuntime.create({ modelsPath: path.join(agentDir, "models.json"), authPath: path.join(agentDir, "auth.json") });
const model = modelRuntime.getModel("qwen-r9700", "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate");
if (!model) throw new Error("Configured Radiance model was not found");
const settings = sdk.SettingsManager.create(cwd, agentDir);
const loader = new sdk.DefaultResourceLoader({
  cwd, agentDir, settingsManager: settings, noContextFiles: true, noExtensions: true,
  noThemes: true, noPromptTemplates: true,
  appendSystemPrompt: [path.join(source, "integrations/pi/qwen-radiance-operating-prompt.md")],
  additionalExtensionPaths: [
    path.join(source, "integrations/pi/qwen-tool-output-condense.mjs"),
    path.join(source, "integrations/pi/qwen-tool-turn-rehydrate.mjs"),
    "/home/lewis/tasks/searchtool/index.ts",
  ],
});
await loader.reload();
if (loader.getExtensions().errors.length) throw new Error("Replay tool-definition loading failed");
const report = [];
for (const [nominal, selected] of Object.entries(selection.selected)) {
  const copy = path.join(root, `session-${nominal}.jsonl`);
  await fs.copyFile(selection.session_copy, copy);
  const manager = sdk.SessionManager.open(copy, root, cwd);
  manager.branch(selected.parent_id);
  const { session } = await sdk.createAgentSession({
    cwd, agentDir, modelRuntime, model, thinkingLevel: "xhigh", settingsManager: settings,
    sessionManager: manager, resourceLoader: loader,
  });
  let systemPrompt = session.agent.state.systemPrompt;
  const runner = session.extensionRunner;
  const before = await runner.emitBeforeAgentStart("", undefined, systemPrompt, session._baseSystemPromptOptions);
  systemPrompt = before?.systemPrompt ?? systemPrompt;
  const messages = sdk.convertToLlm(await runner.emitContext(session.agent.state.messages));
  let captured = false;
  let attemptedFetch = false;
  const response = modelRuntime.streamSimple(model, { systemPrompt, messages, tools: session.agent.state.tools }, {
    reasoning: "xhigh", maxTokens: 1024, maxRetries: 0,
    fetch: async () => { attemptedFetch = true; throw new Error("Replay network guard"); },
    onPayload: async (payload) => {
      const body = JSON.stringify(payload);
      await fs.writeFile(path.join(root, `payload-${nominal}.json`), body, { mode: 0o600 });
      report.push({
        nominal_context: Number(nominal), recorded_input_tokens: selected.recorded_input_tokens,
        recorded_output_tokens: selected.recorded_output_tokens,
        message_count: payload.messages.length, tool_count: payload.tools?.length ?? 0,
        payload_sha256: createHash("sha256").update(body).digest("hex"),
      });
      captured = true;
      throw new Error("REPLAY_PAYLOAD_CAPTURED");
    },
  });
  await response.result();
  if (!captured || attemptedFetch) throw new Error("Private replay capture failed its no-request guard");
  session.dispose();
}
await fs.writeFile(path.join(root, "capture-report.json"), JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ captured: report, model_requests_sent: 0, tools_executed: 0 }));
