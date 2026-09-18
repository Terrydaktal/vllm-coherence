#!/usr/bin/env node

import { createHash } from "node:crypto";
import { chmodSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const CAPTURE_SENTINEL = "QWEN_PROVIDER_PAYLOAD_CAPTURED_BEFORE_NETWORK";

function fail(message) {
  throw new Error(`qwen-pi-provider-payload-capture: ${message}`);
}

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const name = argv[index];
    if (!name.startsWith("--")) fail(`unexpected argument: ${name}`);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) fail(`missing value for ${name}`);
    options[name.slice(2)] = value;
    index += 1;
  }
  for (const required of [
    "pi-root",
    "agent-dir",
    "source-session",
    "source-leaf",
    "output-dir",
    "provider",
    "model",
  ]) {
    if (!options[required]) fail(`--${required} is required`);
  }
  return options;
}

function sha256Bytes(value) {
  return createHash("sha256").update(value).digest("hex");
}

function sha256File(path) {
  return sha256Bytes(readFileSync(path));
}

const options = parseArgs(process.argv.slice(2));
const piRoot = resolve(options["pi-root"]);
const agentDir = resolve(options["agent-dir"]);
const sourceSession = resolve(options["source-session"]);
const outputDir = resolve(options["output-dir"]);
const cwd = options.cwd ? resolve(options.cwd) : "/home/lewis/tasks/money";
const sourceHashBefore = sha256File(sourceSession);

const {
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  SettingsManager,
  createAgentSession,
} = await import(
  pathToFileURL(`${piRoot}/node_modules/@earendil-works/pi-coding-agent/dist/index.js`)
);

mkdirSync(outputDir, { recursive: true, mode: 0o700 });
chmodSync(outputDir, 0o700);
const sourceManager = SessionManager.open(sourceSession, outputDir, cwd);
const clonedSession = sourceManager.createBranchedSession(options["source-leaf"]);
if (!clonedSession) fail("SessionManager did not persist the capture branch");

const settingsManager = SettingsManager.inMemory({
  compaction: { enabled: false },
  retry: { enabled: false, maxRetries: 0, baseDelayMs: 0 },
});
const resourceLoader = new DefaultResourceLoader({
  cwd,
  agentDir,
  settingsManager,
  noExtensions: true,
  noSkills: true,
  noPromptTemplates: true,
  noThemes: true,
});
await resourceLoader.reload();
const modelRuntime = await ModelRuntime.create({
  authPath: `${agentDir}/auth.json`,
  modelsPath: `${agentDir}/models.json`,
});
const model = modelRuntime.getModel(options.provider, options.model);
if (!model) fail(`model is absent: ${options.provider}/${options.model}`);

const sessionManager = SessionManager.open(clonedSession, outputDir, cwd);
const { session } = await createAgentSession({
  cwd,
  agentDir,
  modelRuntime,
  model,
  thinkingLevel: options.thinking ?? "xhigh",
  resourceLoader,
  sessionManager,
  settingsManager,
  tools: ["read", "bash", "edit", "write"],
});

let capturedPayload;
let responseObserved = false;
session.agent.onPayload = async (payload) => {
  if (capturedPayload !== undefined) fail("more than one provider payload was attempted");
  capturedPayload = payload;
  throw new Error(CAPTURE_SENTINEL);
};
session.agent.onResponse = async () => {
  responseObserved = true;
};

await session.agent.continue();
session.dispose();

if (capturedPayload === undefined) fail("provider payload hook was never reached");
if (responseObserved) fail("a provider response was observed after the capture sentinel");
const sourceHashAfter = sha256File(sourceSession);
if (sourceHashAfter !== sourceHashBefore) fail("source transcript changed during capture");

const branchEntries = sessionManager.getBranch();
const sourceLeafIndex = branchEntries.findIndex((entry) => entry.id === options["source-leaf"]);
if (sourceLeafIndex < 0) fail("source leaf disappeared from the capture branch");
const newEntries = branchEntries.slice(sourceLeafIndex + 1);
const captureFailure = newEntries.find(
  (entry) =>
    entry.type === "message" &&
    entry.message?.role === "assistant" &&
    entry.message?.stopReason === "error",
);
if (captureFailure?.message?.errorMessage !== CAPTURE_SENTINEL) {
  fail("capture branch did not stop at the pre-network sentinel");
}

const payloadBytes = Buffer.from(`${JSON.stringify(capturedPayload)}\n`);
const payloadPath = resolve(outputDir, "provider-payload.json");
writeFileSync(payloadPath, payloadBytes, { mode: 0o600 });
chmodSync(payloadPath, 0o600);

const evidence = {
  schema: "qwen-pi-provider-payload-capture-v1",
  passed: true,
  sourceSession,
  sourceLeaf: options["source-leaf"],
  sourceSha256: sourceHashBefore,
  clonedSession,
  model: `${options.provider}/${options.model}`,
  payloadPath,
  payloadSha256: sha256Bytes(payloadBytes),
  messages: Array.isArray(capturedPayload.messages)
    ? capturedPayload.messages.length
    : null,
  tools: Array.isArray(capturedPayload.tools) ? capturedPayload.tools.length : null,
  maxCompletionTokens:
    capturedPayload.max_completion_tokens ?? capturedPayload.max_tokens ?? null,
  temperature: capturedPayload.temperature ?? null,
  topP: capturedPayload.top_p ?? null,
  topK: capturedPayload.top_k ?? null,
  responseObserved,
  sourceSessionUnchanged: sourceHashAfter === sourceHashBefore,
};
const evidencePath = resolve(outputDir, "capture.json");
writeFileSync(evidencePath, `${JSON.stringify(evidence, null, 2)}\n`, { mode: 0o600 });
chmodSync(evidencePath, 0o600);
process.stdout.write(`${JSON.stringify(evidence, null, 2)}\n`);
