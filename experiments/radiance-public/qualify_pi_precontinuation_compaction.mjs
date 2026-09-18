#!/usr/bin/env node

import { createHash } from "node:crypto";
import { chmodSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

function fail(message) {
  throw new Error(`qwen-pi-precontinuation-compaction-qualification: ${message}`);
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

function assistantStopReason(entry) {
  if (entry.type !== "message" || entry.message?.role !== "assistant") return undefined;
  return entry.message.stopReason;
}

const options = parseArgs(process.argv.slice(2));
const piRoot = resolve(options["pi-root"]);
const agentDir = resolve(options["agent-dir"]);
const sourceSession = resolve(options["source-session"]);
const outputDir = resolve(options["output-dir"]);
const cwd = options.cwd ? resolve(options.cwd) : "/home/lewis/tasks/money";
const sourceHashBefore = sha256(sourceSession);

const codingAgent = await import(
  pathToFileURL(`${piRoot}/node_modules/@earendil-works/pi-coding-agent/dist/index.js`)
);
const {
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  SettingsManager,
  createAgentSession,
} = codingAgent;

mkdirSync(outputDir, { recursive: true });
const sourceManager = SessionManager.open(sourceSession, outputDir, cwd);
const clonedSession = sourceManager.createBranchedSession(options["source-leaf"]);
if (!clonedSession) fail("SessionManager did not persist the qualification branch");

const settingsManager = SettingsManager.inMemory({
  compaction: {
    enabled: true,
    reserveTokens: Number(options["reserve-tokens"] ?? 16384),
    keepRecentTokens: Number(options["keep-recent-tokens"] ?? 20000),
  },
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
// The preserved incident can legitimately contain historical failed attempts
// after the selected source leaf.  SessionManager's branch representation is
// not an invocation journal, so slicing at sourceLeaf would misattribute those
// old entries to this qualification.  Snapshot the exact pre-run IDs and judge
// only entries appended by the run below.
const preRunEntryIds = new Set(sessionManager.getBranch().map((entry) => entry.id));
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

let compactionStarted = 0;
let compactionCommitted = 0;
let compactionFailed = false;
let postCompactionTurns = 0;
const eventLog = [];

session.subscribe((event) => {
  if (event.type === "compaction_start") {
    compactionStarted += 1;
    eventLog.push({ type: event.type, reason: event.reason });
  } else if (event.type === "compaction_end") {
    if (event.aborted || !event.result || event.errorMessage) compactionFailed = true;
    else compactionCommitted += 1;
    eventLog.push({
      type: event.type,
      reason: event.reason,
      aborted: event.aborted,
      committed: Boolean(event.result),
      errorMessage: event.errorMessage,
    });
  } else if (event.type === "turn_end") {
    const usage = event.message?.usage;
    eventLog.push({
      type: event.type,
      stopReason: event.message?.stopReason,
      totalTokens: usage?.totalTokens,
      toolResults: event.toolResults?.length ?? 0,
      afterCompaction: compactionCommitted > 0,
    });
    if (compactionCommitted > 0) postCompactionTurns += 1;
  }
});

// This qualification exercises model generation and Pi's real tool-loop state,
// but it must never execute the model's arbitrary shell or filesystem actions.
session.agent.beforeToolCall = async () => ({
  block: true,
  reason: "Qualification fixture: tool execution intentionally blocked after structural validation",
});
session.agent.shouldStopAfterTurn = async () =>
  compactionCommitted > 0 && postCompactionTurns >= 1;

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
if (sourceLeafIndex < 0) fail("source leaf disappeared from the qualification branch");
const qualificationEntries = branchEntries.filter((entry) => !preRunEntryIds.has(entry.id));
const compactionEntries = qualificationEntries.filter((entry) => entry.type === "compaction");
const firstCompactionIndex = compactionEntries[0]
  ? branchEntries.findIndex((entry) => entry.id === compactionEntries[0].id)
  : -1;
const assistantEntriesAfterCompaction = branchEntries
  .slice(firstCompactionIndex + 1)
  .filter((entry) => entry.type === "message" && entry.message?.role === "assistant");
const badAssistantStops = qualificationEntries
  .filter((entry) => ["length", "error", "aborted"].includes(assistantStopReason(entry)))
  .map((entry) => ({ id: entry.id, stopReason: assistantStopReason(entry) }));

const checks = {
  sourceSessionUnchanged: sourceHashAfter === sourceHashBefore,
  runCompletedWithoutThrow: !runError,
  oneCompactionStarted: compactionStarted === 1,
  oneCompactionCommitted: compactionCommitted === 1 && compactionEntries.length === 1,
  compactionDidNotFail: !compactionFailed,
  postCompactionTurnCompleted: postCompactionTurns >= 1 && assistantEntriesAfterCompaction.length >= 1,
  noLengthErrorOrAbort: badAssistantStops.length === 0,
};
const passed = Object.values(checks).every(Boolean);
const evidence = {
  schema: "qwen-pi-precontinuation-compaction-qualification-v1",
  passed,
  sourceSession,
  sourceLeaf: options["source-leaf"],
  sourceSha256: sourceHashBefore,
  clonedSession,
  model: `${options.provider}/${options.model}`,
  contextWindow: model.contextWindow,
  reserveTokens: settingsManager.getCompactionReserveTokens(),
  keepRecentTokens: settingsManager.getCompactionKeepRecentTokens(),
  checks,
  runError,
  compactionEntries: compactionEntries.map((entry) => ({
    id: entry.id,
    parentId: entry.parentId,
    tokensBefore: entry.tokensBefore,
  })),
  assistantEntriesAfterCompaction: assistantEntriesAfterCompaction.map((entry) => ({
    id: entry.id,
    parentId: entry.parentId,
    stopReason: entry.message.stopReason,
    usage: entry.message.usage,
  })),
  badAssistantStops,
  eventLog,
};
const evidencePath = resolve(outputDir, "qualification.json");
writeFileSync(evidencePath, `${JSON.stringify(evidence, null, 2)}\n`, { mode: 0o600 });
chmodSync(evidencePath, 0o600);
process.stdout.write(`${JSON.stringify(evidence, null, 2)}\n`);
if (!passed) process.exitCode = 1;
