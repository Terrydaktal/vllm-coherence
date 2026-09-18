// Reconstruct opaque failing requests inside the VM; never execute their tools.
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { createHash } from "node:crypto";

process.umask(0o077);
process.env.PI_OFFLINE = "1";
globalThis.fetch = async () => { throw new Error("Replay network access prohibited"); };
let stage = "validate";
try {
  const root = process.argv[2];
  if (!root?.startsWith(`/run/user/${process.getuid()}/qwen-private-replay-loops-`)) {
    throw new Error("Invalid private replay directory");
  }
  const selection = JSON.parse(await fs.readFile(path.join(root, "selection.json"), "utf8"));
  const source = selection.release_root;
  const cwd = selection.cwd;
  const installed = path.join(source, "runtime/node_modules/@earendil-works/pi-coding-agent/dist");
  const integrations = path.join(source, "integrations/pi");
  const sdk = await import(pathToFileURL(path.join(installed, "index.js")));
  const { ModelRuntime } = await import(pathToFileURL(path.join(installed, "core/model-runtime.js")));
  const agentDir = path.join(root, "agent");
  await fs.mkdir(agentDir, { mode: 0o700, recursive: true });
  for (const name of ["settings.json", "models.json"]) {
    await fs.copyFile(`/home/qwen/.pi/agent/${name}`, path.join(agentDir, name));
  }
  const modelRuntime = await ModelRuntime.create({
    modelsPath: path.join(agentDir, "models.json"), authPath: path.join(agentDir, "auth.json"),
  });
  const model = modelRuntime.getModel("qwen-r9700", "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate");
  if (!model) throw new Error("Model unavailable");
  const settings = sdk.SettingsManager.create(cwd, agentDir);
  const loader = new sdk.DefaultResourceLoader({
    cwd, agentDir, settingsManager: settings, noContextFiles: true, noExtensions: true,
    noThemes: true, noPromptTemplates: true,
    appendSystemPrompt: [path.join(integrations, "qwen-radiance-operating-prompt.md")],
    additionalExtensionPaths: [
      path.join(integrations, "qwen-tool-output-condense.mjs"),
      path.join(integrations, "qwen-tool-turn-rehydrate.mjs"),
      "/opt/opsec-web-tools/service/extension.mjs",
    ],
  });
  stage = "load tool definitions";
  await loader.reload();
  if (loader.getExtensions().errors.length) throw new Error("Tool definitions unavailable");
  const report = [];
  for (const [label, selected] of Object.entries(selection.selected)) {
    if (!/^[a-z_]+$/.test(label)) throw new Error("Invalid fixture label");
    stage = `reconstruct ${label}`;
    const manager = sdk.SessionManager.open(selection.session_copy, root, cwd);
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
    const response = modelRuntime.streamSimple(model, {
      systemPrompt, messages, tools: session.agent.state.tools,
    }, {
      reasoning: "xhigh", maxTokens: 16384, maxRetries: 0,
      fetch: async () => { attemptedFetch = true; throw new Error("Replay network guard"); },
      onPayload: async (payload) => {
        const body = JSON.stringify(payload);
        await fs.writeFile(path.join(root, `payload-${label}.json`), body, { mode: 0o600 });
        report.push({ label, ...selected, message_count: payload.messages.length,
          tool_count: payload.tools?.length ?? 0, bytes: Buffer.byteLength(body),
          payload_sha256: createHash("sha256").update(body).digest("hex") });
        captured = true;
        throw new Error("REPLAY_PAYLOAD_CAPTURED");
      },
    });
    await response.result();
    if (!captured || attemptedFetch) throw new Error("Replay capture invariant failed");
    session.dispose();
  }
  await fs.writeFile(path.join(root, "capture-report.json"), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify({ captured: report, model_requests_sent: 0, tools_executed: 0 }));
} catch (error) {
  // Error messages from SDK/provider code can include private payload fragments.
  console.error(JSON.stringify({ stage, error_type: error?.name ?? "Error" }));
  process.exitCode = 1;
}
