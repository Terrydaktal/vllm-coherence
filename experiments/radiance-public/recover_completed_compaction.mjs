#!/usr/bin/env node
// Repair a completed heading-rejected summary, not the chat transcript. This
// only publishes hash-bound receipts after reconstructing Pi's original request.
// The endpoint allowlist prohibits GPU generation; /tokenize is CPU-only.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, lstat } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { compactFromPayload, compactionReceiptKey, compactionRetryIdentities, validateSummary,
  writeReceipt, CONTRACT, MODEL, fileOperations } from "../../integrations/pi/qwen-radiance-compaction.mjs";
import condense from "../../integrations/pi/qwen-tool-output-condense.mjs";

const args = Object.fromEntries(process.argv.slice(2).reduce((out, word, i, words) => i % 2 ? out : [...out, [word.slice(2), words[i + 1]]], []));
assert.ok(args.artifact, "--artifact is required");
const artifactPath = resolve(args.artifact);
assert.match(basename(artifactPath), /^[a-f0-9]{64}\.failed\.json$/);
const receiptDirectory = dirname(artifactPath);
const agentDir = dirname(receiptDirectory);
assert.equal(basename(receiptDirectory), "radiance-compaction-receipts");
const metadata = await lstat(artifactPath);
assert.ok(metadata.isFile() && !metadata.isSymbolicLink() && metadata.uid === process.getuid() && metadata.nlink === 1 && !(metadata.mode & 0o077));
const artifactBytes = await readFile(artifactPath);
const artifact = JSON.parse(artifactBytes);
assert.equal(artifact.contract, CONTRACT);
validateSummary(artifact.completion.text, artifact.completion.finishReason, artifact.marker);
const sourceFile = artifact.sourceIdentity.sessionFile;
const sourceBytes = await readFile(sourceFile);
const entries = sourceBytes.toString("utf8").trim().split("\n").map((line) => JSON.parse(line));
const cwd = entries.find((entry) => entry.type === "session").cwd;
const byId = new Map(entries.filter((entry) => entry.id).map((entry) => [entry.id, entry]));
function branchAt(leaf) {
  const branch = [], seen = new Set();
  while (leaf) {
    assert.ok(!seen.has(leaf), "cycle in session branch"); seen.add(leaf);
    const entry = byId.get(leaf); assert.ok(entry, "missing session ancestor");
    branch.unshift(entry); leaf = entry.parentId;
  }
  return branch;
}
const branch = branchAt(artifact.sourceIdentity.leaf);
const latest = entries.findLast((entry) => entry.id && entry.type !== "session");
const currentIdentity = { sessionFile: sourceFile, leaf: latest.id };
const currentBranch = branchAt(currentIdentity.leaf);
assert.ok(currentIdentity.leaf === artifact.sourceIdentity.leaf ||
  compactionRetryIdentities(currentBranch, currentIdentity).some((identity) => identity.leaf === artifact.sourceIdentity.leaf),
  "chat advanced beyond error-only retries; explicit requalification is required");

const piRoot = args["pi-root"] ?? "/home/lewis/.local/share/qwen-r9700/pi/0.84.2";
const coding = await import(pathToFileURL(`${piRoot}/node_modules/@earendil-works/pi-coding-agent/dist/index.js`));
const { prepareCompaction } = await import(pathToFileURL(`${piRoot}/node_modules/@earendil-works/pi-coding-agent/dist/core/compaction/compaction.js`));
const { streamSimpleOpenAICompletions } = await import(pathToFileURL(`${piRoot}/node_modules/@earendil-works/pi-ai/dist/compat.js`));
const repository = resolve(import.meta.dirname, "../..");
const settings = coding.SettingsManager.inMemory(JSON.parse(await readFile(join(agentDir, "settings.json"), "utf8")));
const loader = new coding.DefaultResourceLoader({ cwd, agentDir, settingsManager: settings, noExtensions: true,
  additionalExtensionPaths: ["qwen-progress.mjs", "qwen-tool-output-condense.mjs", "qwen-radiance-compaction.ts"].map((name) => join(repository, "integrations/pi", name))
    .concat("/home/lewis/tasks/searchtool/index.ts") });
await loader.reload();
assert.equal(loader.getExtensions().errors.length, 0, "extension loading failed");
const runtime = await coding.ModelRuntime.create({ modelsPath: join(agentDir, "models.json"), authPath: join(agentDir, "auth.json") });
const configuredModel = runtime.getModel("qwen-r9700", MODEL);
assert.ok(configuredModel);
const { session } = await coding.createAgentSession({ cwd, agentDir, modelRuntime: runtime, model: configuredModel, thinkingLevel: "xhigh",
  resourceLoader: loader, settingsManager: settings, sessionManager: coding.SessionManager.inMemory(cwd) });
await session.bindExtensions({});
const auth = await new coding.ModelRegistry(runtime).getApiKeyAndHeaders(configuredModel);
assert.ok(auth.ok);
const model = { ...configuredModel, baseUrl: auth.baseUrl ?? configuredModel.baseUrl };
const toolsByName = new Map(session.getAllTools().map((tool) => [tool.name, tool]));
const tools = session.getActiveToolNames().map((name) => toolsByName.get(name));
const startupSystem = session.systemPrompt;
let guidanceHook;
condense({ on: (name, handler) => { if (name === "before_agent_start") guidanceHook = handler; } });
const guidedSystem = guidanceHook({ systemPrompt: startupSystem }, { model })?.systemPrompt ?? startupSystem;
const policy = await readFile(join(repository, "integrations/pi/qwen-loss-sensitive-compact-prompt.md"), "utf8");
const preparation = prepareCompaction(branch, settings.getCompactionSettings());
assert.ok(preparation);
async function capture(systemPrompt, sourceBranch) {
  let payload;
  const result = await streamSimpleOpenAICompletions(model, { systemPrompt, tools,
    messages: coding.convertToLlm(coding.buildSessionContext(sourceBranch).messages) }, { ...auth, reasoning: "xhigh", maxTokens: 1,
    onPayload: (value) => { payload = value; throw new Error("capture-only"); },
    fetch: () => { throw new Error("network prohibited during payload capture"); },
  }).result();
  assert.equal(result.errorMessage, "capture-only"); assert.ok(payload);
  return payload;
}
const expectedKey = basename(artifactPath).slice(0, 64);
let payload;
for (const system of new Set([startupSystem, guidedSystem])) {
  const candidate = await capture(system, branch);
  if (compactionReceiptKey({ payload: candidate, preparation, policy, model, sourceIdentity: artifact.sourceIdentity }) === expectedKey) {
    payload = candidate; break;
  }
}
assert.ok(payload, "could not authenticate the exact preserved request; no receipt published");
const sha = (value) => createHash("sha256").update(value).digest("hex");
assert.equal(sha(await readFile(sourceFile)), sha(sourceBytes), "source changed during reconstruction");
const result = await compactFromPayload({ payload, preparation, policy, model, sourceIdentity: artifact.sourceIdentity,
  receiptDirectory, headers: { "Content-Type": "application/json", Authorization: `Bearer ${auth.apiKey}` },
  signal: new AbortController().signal, fetcher: (url, options) => {
    assert.equal(url, `${model.baseUrl.slice(0, -3)}/tokenize`, "generation is prohibited during recovery");
    return fetch(url, options);
  } });
assert.ok(result.details.recoveredCompletedSummary || result.details.receiptReused);

// Also bind a ready receipt to Pi's fresh-start system prompt, which has not yet
// received the launcher's per-turn output guidance. Both variants are produced
// by the installed launcher extensions; history/tools/model/policy must agree.
const currentPreparation = prepareCompaction(currentBranch, settings.getCompactionSettings());
assert.deepEqual(fileOperations(currentPreparation.fileOps), fileOperations(preparation.fileOps));
assert.equal(currentPreparation.firstKeptEntryId, preparation.firstKeptEntryId);
const withoutSystem = (value) => value.messages.filter((message) => !["system", "developer"].includes(message.role));
const published = new Set([expectedKey]);
let verifiedRestartVariants = 0;
for (const system of new Set([startupSystem, guidedSystem])) {
  const currentPayload = await capture(system, currentBranch);
  assert.deepEqual(withoutSystem(currentPayload), withoutSystem(payload), "provider-visible conversation changed");
  const key = compactionReceiptKey({ payload: currentPayload, preparation: currentPreparation, policy, model, sourceIdentity: currentIdentity });
  if (!published.has(key)) {
    assert.equal(sha(await readFile(sourceFile)), sha(sourceBytes), "source changed before receipt publication");
    const recovered = { ...result, tokensBefore: currentPreparation.tokensBefore, details: { ...result.details,
      sourceIdentity: currentIdentity, recoveredSourceIdentity: artifact.sourceIdentity, recoveredArtifactSha256: sha(artifactBytes),
      recoveryValidation: "exact original request key; identical provider history; known launcher system-prompt variant" } };
    await writeReceipt(receiptDirectory, key, { contract: CONTRACT, key, marker: artifact.marker, result: recovered,
      resultHash: sha(JSON.stringify(recovered)) });
    published.add(key);
  }
  const retry = await compactFromPayload({ payload: currentPayload, preparation: currentPreparation, policy, model,
    sourceIdentity: currentIdentity, receiptDirectory, signal: new AbortController().signal,
    fetcher: () => assert.fail("ready restart receipt must be reusable without any network request") });
  assert.equal(retry.details.receiptReused, true);
  assert.equal(retry.summary, result.summary);
  verifiedRestartVariants++;
}
session.dispose();
assert.equal(sha(await readFile(sourceFile)), sha(sourceBytes));
assert.equal(sha(await readFile(artifactPath)), sha(artifactBytes));
console.log(JSON.stringify({ passed: true, originalRequestAuthenticated: true, generationRequests: 0, sourceUnchanged: true,
  artifactUnchanged: true, recoveredOutputTokens: result.usage.output, verifiedRestartVariants, readyReceipts: [...published] }));
