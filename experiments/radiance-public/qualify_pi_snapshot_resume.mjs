#!/usr/bin/env node

import { createHash } from "node:crypto";
import { chmodSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

function fail(message) {
  throw new Error(`qwen-pi-snapshot-resume-qualification: ${message}`);
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

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

const options = parseArgs(process.argv.slice(2));
const piRoot = resolve(options["pi-root"]);
const agentDir = resolve(options["agent-dir"]);
const sourceSession = resolve(options["source-session"]);
const outputDir = resolve(options["output-dir"]);
const cwd = options.cwd ? resolve(options.cwd) : "/home/lewis/tasks/money";
const minimumCacheFraction = Number(options["minimum-cache-fraction"] ?? 0.8);
const sourceHashBefore = sha256(sourceSession);

const {
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  SettingsManager,
  createAgentSession,
} = await import(
  pathToFileURL(`${piRoot}/node_modules/@earendil-works/pi-coding-agent/dist/index.js`)
);

mkdirSync(outputDir, { recursive: true });
const sourceManager = SessionManager.open(sourceSession, outputDir, cwd);
const clonedSession = sourceManager.createBranchedSession(options["source-leaf"]);
if (!clonedSession) fail("SessionManager did not persist the resume branch");

const settingsManager = SettingsManager.inMemory({
  compaction: { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 },
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

let completedTurns = 0;
session.agent.beforeToolCall = async () => ({
  block: true,
  reason: "Qualification fixture: tool execution intentionally blocked after structural validation",
});
session.agent.shouldStopAfterTurn = async () => completedTurns >= 1;
session.subscribe((event) => {
  if (event.type === "turn_end") completedTurns += 1;
});

let runError;
try {
  await session.agent.continue();
} catch (error) {
  runError = error instanceof Error ? error.stack ?? error.message : String(error);
} finally {
  session.dispose();
}

const sourceHashAfter = sha256(sourceSession);
const branchEntries = sessionManager.getBranch();
const sourceLeafIndex = branchEntries.findIndex((entry) => entry.id === options["source-leaf"]);
if (sourceLeafIndex < 0) fail("source leaf disappeared from the resume branch");
const newAssistantEntries = branchEntries
  .slice(sourceLeafIndex + 1)
  .filter((entry) => entry.type === "message" && entry.message?.role === "assistant");
const assistant = newAssistantEntries[0]?.message;
const usage = assistant?.usage;
const promptTokens = (usage?.input ?? 0) + (usage?.cacheRead ?? 0);
const cacheFraction = promptTokens > 0 ? (usage?.cacheRead ?? 0) / promptTokens : 0;
const checks = {
  sourceSessionUnchanged: sourceHashAfter === sourceHashBefore,
  runCompletedWithoutThrow: !runError,
  exactlyOneAssistantTurn: completedTurns === 1 && newAssistantEntries.length === 1,
  responseCompletedNormally:
    assistant !== undefined && !["length", "error", "aborted"].includes(assistant.stopReason),
  durableSnapshotWasReused: cacheFraction >= minimumCacheFraction,
};
const passed = Object.values(checks).every(Boolean);
const evidence = {
  schema: "qwen-pi-snapshot-resume-qualification-v1",
  passed,
  sourceSession,
  sourceLeaf: options["source-leaf"],
  sourceSha256: sourceHashBefore,
  clonedSession,
  model: `${options.provider}/${options.model}`,
  minimumCacheFraction,
  cacheFraction,
  usage,
  stopReason: assistant?.stopReason,
  checks,
  runError,
};
const evidencePath = resolve(outputDir, "qualification.json");
writeFileSync(evidencePath, `${JSON.stringify(evidence, null, 2)}\n`, { mode: 0o600 });
chmodSync(evidencePath, 0o600);
process.stdout.write(`${JSON.stringify(evidence, null, 2)}\n`);
if (!passed) process.exitCode = 1;
