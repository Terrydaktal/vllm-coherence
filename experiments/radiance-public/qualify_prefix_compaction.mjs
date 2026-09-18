#!/usr/bin/env node
// Isolated Pi sessions only. No tools execute and no existing transcript/backend
// is modified. Warm a real provider prompt, compact it, reopen it, and verify a
// normal continuation can recover facts that existed only in the removed prefix.
import { lstat, mkdtemp, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";
import { MODEL, CONTRACT, readCompletion, postJson } from "../../integrations/pi/qwen-radiance-compaction.mjs";

const repository = resolve(import.meta.dirname, "../..");
const args = Object.fromEntries(process.argv.slice(2).reduce((out, word, i, words) => i % 2 ? out : [...out, [word.slice(2), words[i + 1]]], []));
const baseUrl = args["base-url"] ?? "http://127.0.0.1:18124/v1";
const requested = Number(args["context-tokens"] ?? 6000);
const piRoot = args["pi-root"] ?? "/home/lewis/.local/share/qwen-r9700/pi/0.84.2";
const { ModelRuntime, SessionManager, SettingsManager, DefaultResourceLoader, createAgentSession, convertToLlm } =
  await import(pathToFileURL(`${piRoot}/node_modules/@earendil-works/pi-coding-agent/dist/index.js`));
const stateRoot = "/home/lewis/.local/state/qwen-r9700/radiance-compaction-qualification";
await mkdir(stateRoot, { recursive: true, mode: 0o700 });
const outputDir = args["resume-run"] ? resolve(args["resume-run"]) : await mkdtemp(join(stateRoot, "run-"));
assert.equal(dirname(outputDir), stateRoot, "only isolated qualification directories can be reopened");
assert.ok(!(await lstat(outputDir)).isSymbolicLink());
const agentDir = join(outputDir, "agent");
await mkdir(agentDir, { mode: 0o700, recursive: true });
const config = JSON.parse(await readFile(join(repository, "integrations/pi/models-radiance-public-clean-snapshot.json"), "utf8"));
config.providers["qwen-r9700"].baseUrl = baseUrl;
if (!args["resume-run"]) await writeFile(join(agentDir, "models.json"), JSON.stringify(config), { mode: 0o600 });
process.env.PI_CODING_AGENT_DIR = agentDir;
process.env.PI_OFFLINE = "1";
const modelRuntime = await ModelRuntime.create({ modelsPath: join(agentDir, "models.json"), authPath: join(agentDir, "auth.json") });
const model = modelRuntime.getModel("qwen-r9700", MODEL);
assert.ok(model);
const timestamp = Date.now();
const zeroUsage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
const assistant = (content, stopReason = "stop") => ({ role: "assistant", content, stopReason, timestamp,
  api: model.api, provider: model.provider, model: model.id, usage: zeroUsage });
const user = (text) => ({ role: "user", content: [{ type: "text", text }], timestamp });
const factual = "QUALIFICATION FIXTURE, NOT A REAL TASK. Preserve these exact critical facts: RELEASE_CODE=ORCHID-742; " +
  "BUDGET_LIMIT=9137; rejected approach=cached-GEMM selector, reason=changed reference tokens. " +
  "Do not run tools or implement anything. Incidental measurement rows below can be collapsed into a count. ";
const row = (i) => `Measurement ${i}: path=/fixture/module-${i % 43}.ts; test=${i % 23}; seed=${(i * 7829) % 99173}; result=pass; latency=${i % 97}.25us.\n`;
// Approximate size is measured exactly before warming; use --context-tokens to
// request long-context qualification. This filler must not be mistaken for facts.
let rows = Array.from({ length: Math.max(10, Math.floor((requested - 2300) / 43)) }, (_, i) => row(i)).join("");
const sessionFiles = (await readdir(outputDir)).filter((file) => file.endsWith(".jsonl"));
if (args["resume-run"]) assert.equal(sessionFiles.length, 1);
const sessionManager = args["resume-run"] ? SessionManager.open(join(outputDir, sessionFiles[0]), outputDir, outputDir) : SessionManager.create(outputDir, outputDir);
if (!args["resume-run"]) {
sessionManager.appendMessage(user(factual + "\n" + rows));
sessionManager.appendMessage(assistant([{ type: "text", text: "Fixture facts recorded; measurement rows are incidental." }]));
sessionManager.appendMessage(user("Current task: preserve the previous three facts and NEXT_ACTION=verify-snapshot-restart. This is a partial-turn fixture; do not execute tools."));
sessionManager.appendMessage(assistant([{ type: "toolCall", id: "fixture_read", name: "read", arguments: { path: "/fixture/notes.txt" } }], "toolUse"));
sessionManager.appendMessage({ role: "toolResult", toolCallId: "fixture_read", toolName: "read", isError: false, timestamp,
  content: [{ type: "text", text: "Synthetic read result (no tool executed).\n" + Array.from({ length: 80 }, (_, i) => row(i + 50000)).join("") }] });
sessionManager.appendMessage(assistant([{ type: "toolCall", id: "fixture_tail", name: "read", arguments: { path: "/fixture/recent.txt" } }], "toolUse"));
sessionManager.appendMessage({ role: "toolResult", toolCallId: "fixture_tail", toolName: "read", isError: false, timestamp,
  content: [{ type: "text", text: "Synthetic recent tool result: ready for checkpoint qualification." }] });
sessionManager.appendMessage(assistant([{ type: "text", text: "The fixture is ready. No real work has been performed." }]));
}

const notifications = [];
async function openSession(manager) {
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false, reserveTokens: 16384, keepRecentTokens: 600 }, retry: { enabled: false } });
  const loader = new DefaultResourceLoader({ cwd: outputDir, agentDir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true,
    additionalExtensionPaths: [
      ...(args.extensions === "launcher" ? [join(repository, "integrations/pi/qwen-progress.mjs"),
        join(repository, "integrations/pi/qwen-tool-output-condense.mjs"), "/home/lewis/tasks/searchtool/index.ts"] : []),
      join(repository, "integrations/pi/qwen-radiance-compaction.ts"),
    ] });
  await loader.reload();
  const errors = loader.getExtensions().errors;
  assert.equal(errors.length, 0, JSON.stringify(errors));
  const { session } = await createAgentSession({ cwd: outputDir, agentDir, modelRuntime, model, thinkingLevel: "xhigh",
    resourceLoader: loader, settingsManager, sessionManager: manager, tools: ["read", "bash", "edit", "write"] });
  let lastProgress = 0;
  await session.bindExtensions({ onError: (error) => { notifications.push(error); console.error(JSON.stringify(error)); },
    uiContext: { notify: (message, type) => { notifications.push({ message, type }); console.error(message); },
      setWorkingMessage: (message) => {
        if (message && Date.now() - lastProgress >= 15000) { console.log(message); lastProgress = Date.now(); }
      }, setStatus: () => {}, setWidget: () => {} } });
  session.agent.beforeToolCall = async () => ({ block: true, reason: "qualification: real tools prohibited" });
  session.agent.shouldStopAfterTurn = async () => true;
  return session;
}

let session = await openSession(sessionManager);
let payload;
await modelRuntime.streamSimple(model, { systemPrompt: session.agent.state.systemPrompt,
  messages: convertToLlm(session.agent.state.messages), tools: session.agent.state.tools }, {
  reasoning: "xhigh", maxTokens: 1, onPayload: (value) => { payload = value; throw new Error("capture"); },
}).result();
assert.ok(payload);
const headers = { "Content-Type": "application/json", Authorization: "Bearer local" };
const tokenized = await (await postJson(`${baseUrl.slice(0, -3)}/tokenize`, { model: MODEL, messages: payload.messages,
  tools: payload.tools, chat_template_kwargs: payload.chat_template_kwargs, add_generation_prompt: true }, headers)).json();
console.log(JSON.stringify({ phase: args["resume-run"] ? "reusing isolated cached fixture" : "warming isolated fixture", outputDir, promptTokens: tokenized.count }));
const warmStart = performance.now();
let warm;
if (args["resume-run"]) {
  const measured = sessionManager.getBranch().findLast((entry) => entry.message?.role === "assistant" && entry.message.usage?.input > 0)?.message.usage;
  assert.ok(measured, "resumed fixture must have a persisted measured warm boundary");
  warm = { usage: { prompt_tokens: measured.input, completion_tokens: measured.output, total_tokens: measured.totalTokens } };
} else warm = await readCompletion(await postJson(`${baseUrl}/completions`, { model: MODEL, prompt: tokenized.tokens,
  max_tokens: 1, temperature: 0, stream: true, stream_options: { include_usage: true }, add_special_tokens: false }, headers));
console.log(JSON.stringify({ phase: "warm complete", elapsedMs: Math.round(performance.now() - warmStart), usage: warm.usage }));
const compactStart = performance.now();
let result;
if (args.mode === "auto") {
  // Seed the isolated fixture's final assistant with the independently measured
  // warm request usage. This exercises Pi's public prompt-time threshold path,
  // not a direct call to its private _runAutoCompaction implementation.
  sessionManager.appendMessage({ ...assistant([{ type: "text", text: "Fixture cache warming completed; ready for checkpoint." }]),
    timestamp: Date.now(), usage: { ...zeroUsage, input: warm.usage.prompt_tokens, output: 1, totalTokens: warm.usage.total_tokens } });
  session.agent.state.messages = sessionManager.buildSessionContext().messages;
  const reserveTokens = Math.max(16384, model.contextWindow - warm.usage.prompt_tokens + 1);
  session.settingsManager.applyOverrides({ compaction: { enabled: true, reserveTokens, keepRecentTokens: 600 } });
  session.subscribe((event) => {
    if (event.type === "compaction_start") console.log(JSON.stringify({ phase: "automatic compaction", reason: event.reason, reserveTokens }));
    if (event.type === "compaction_end") {
      assert.equal(event.reason, "threshold");
      assert.ok(event.result, event.errorMessage ?? "automatic compaction did not commit");
      assert.equal(result, undefined, "compaction repeated unnecessarily");
      result = event.result;
    }
  });
  await session.prompt("Reply only READY. Do not run tools or do any work.");
  assert.ok(result, "automatic compaction did not trigger");
} else result = await session.compact();
assert.equal(result.details?.contract, CONTRACT);
assert.equal(result.details.summaryRequests, 1);
for (const fact of ["ORCHID-742", "9137", "cached-GEMM", "verify-snapshot-restart"]) assert.ok(result.summary.includes(fact), `lost fact ${fact}`);
// Hybrid checkpoints are aligned to 1648-token blocks; a small fixture can
// legitimately replay up to two trailing blocks. Long fixtures still need >90%.
assert.ok(result.usage.cacheRead >= tokenized.count - 3296, `cache reuse too low: ${result.usage.cacheRead}/${tokenized.count}`);
assert.equal(result.usage.reasoning, 0);
console.log(JSON.stringify({ phase: "compacted", elapsedMs: Math.round(performance.now() - compactStart), usage: result.usage, details: result.details }));
session.dispose();
const reopened = SessionManager.open(sessionManager.getSessionFile(), outputDir, outputDir);
assert.ok(reopened.getBranch().some((entry) => entry.type === "compaction" && entry.fromHook));
session = await openSession(reopened);
await session.prompt("Return only a one-line JSON object with release_code, budget_limit, rejected_approach and next_action from the fixture's earlier critical facts. No tools, no analysis, do not run the fixture task.");
const answer = session.agent.state.messages.at(-1);
assert.equal(answer.role, "assistant");
assert.equal(answer.stopReason, "stop", answer.errorMessage);
const visible = answer.content.filter((block) => block.type === "text").map((block) => block.text).join("");
for (const fact of ["ORCHID-742", "9137", "cached-GEMM", "verify-snapshot-restart"]) assert.ok(visible.includes(fact), `continuation lost ${fact}: ${visible}`);
session.dispose();
const evidence = { passed: true, outputDir, contextTokens: tokenized.count, warm: warm.usage, result, answer, notifications };
await writeFile(join(outputDir, "evidence.json"), JSON.stringify(evidence, null, 2), { mode: 0o600 });
console.log(JSON.stringify({ phase: "PASS: compaction + reopen + continuation", outputDir, visible, usage: answer.usage }));
