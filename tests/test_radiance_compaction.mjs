import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { closeSummaryThinking, checkpointOutputBudget, summaryInstruction, validateSummary, readCompletion, compactFromPayload,
  fileOperations, readReceipt, CONTRACT, MODEL, installRadianceCompaction, writeReceipt,
  compactionReceiptKey, compactionRetryIdentities } from "../integrations/pi/qwen-radiance-compaction.mjs";
import { getCompactionProgress } from "../integrations/pi/qwen-radiance-compaction-progress.mjs";

const headings = ["Goal", "Current Authoritative State", "Constraints & Invariants", "Progress", "Measurements & Evidence",
  "Key Decisions", "Rejected / Failed Approaches", "Unresolved Questions & Hypotheses", "Next Steps", "Critical Context"];
const summary = headings.map((name) => `### ${name}\nVerified state.`).join("\n\n");
const signal = () => new AbortController().signal;
function sse(frames, width = 7) {
  const bytes = new TextEncoder().encode(frames.map((data) => `data: ${typeof data === "string" ? data : JSON.stringify(data)}\r\n\r\n`).join(""));
  return new Response(new ReadableStream({ start(controller) {
    for (let i = 0; i < bytes.length; i += width) controller.enqueue(bytes.slice(i, i + width));
    controller.close();
  } }), { headers: { "Content-Type": "text/event-stream" } });
}
const choice = (text = "", finish_reason = null) => ({ choices: [{ index: 0, text, finish_reason }] });

test("session resume backfills the latest exact compaction total from safe timing metadata", async (t) => {
  const previous = process.env.PI_CODING_AGENT_DIR;
  const agentDir = await mkdtemp(join(tmpdir(), "radiance-compaction-resume-"));
  process.env.PI_CODING_AGENT_DIR = agentDir;
  t.after(async () => {
    if (previous === undefined) delete process.env.PI_CODING_AGENT_DIR; else process.env.PI_CODING_AGENT_DIR = previous;
    await rm(agentDir, { recursive: true, force: true });
  });
  const sessionFile = join(agentDir, "session.jsonl");
  await writeFile(sessionFile, "");
  const attemptId = "completed-attempt";
  const compactionEntry = { type: "compaction", id: "compacted", details: { diagnostics: { attemptId } } };
  const entries = [compactionEntry];
  const handlers = new Map(), notices = [];
  const pi = { on: (name, handler) => handlers.set(name, handler), getActiveTools: () => [], getAllTools: () => [],
    getThinkingLevel: () => "off", appendEntry: (customType, data) => entries.push({ type: "custom", customType, data }) };
  const ctx = { model: { id: MODEL }, sessionManager: { getSessionFile: () => sessionFile,
    getSessionId: () => "session", getEntries: () => entries, getBranch: () => entries },
    ui: { notify: (message) => notices.push(message), setWidget() {}, setStatus() {}, setWorkingMessage() {} } };
  const key = createHash("sha256").update(JSON.stringify({ sessionFile })).digest("hex") + ".progress";
  await writeReceipt(join(agentDir, "radiance-compaction-receipts"), key, {
    attemptId, state: "complete", transcriptAppended: true, elapsedMs: 94321,
  });
  installRadianceCompaction(pi, { convertToLlm: (value) => value, streamSimpleOpenAICompletions: () => {} });

  await handlers.get("session_start")({}, ctx);
  assert.equal(compactionEntry.details.elapsedMs, 94321);
  assert.deepEqual(entries.at(-1).data, { compactionEntryId: "compacted", elapsedMs: 94321 });
  assert.deepEqual(notices, []);
});

test("only the assistant generation suffix changes, not a single history token", () => {
  assert.deepEqual(closeSummaryThinking([1, 2], [1, 2, 3, 4, 5], [4, 5], [4, 6, 7], 100, 10), [1, 2, 3, 4, 6, 7]);
  assert.deepEqual(closeSummaryThinking([1, 2], [1, 2, 4, 6, 7], [4, 5], [4, 6, 7], 100, 10), [1, 2, 4, 6, 7]);
  assert.throws(() => closeSummaryThinking([1, 2], [1, 9, 4, 5], [4, 5], [4, 6, 7], 100, 10), /prefix at token 1/);
  assert.throws(() => closeSummaryThinking([1], [1, 8], [4, 5], [4, 6, 7], 100, 10), /boundary/);
  assert.throws(() => closeSummaryThinking([1], [1, 4, 5], [4, 5], [4, 6, 7], 10, 10), /context limit/);
  assert.throws(() => closeSummaryThinking([NaN], [1], [4], [5], 100, 10), /invalid token/);
});

test("thinking controls cannot rewrite the system prompt via the summary instructions", () => {
  assert.match(summaryInstruction("policy", "END", "retain old failures"), /retain old failures/);
  assert.throws(() => summaryInstruction("policy", "END", "<|think_off|>"), /controls/);
});

test("checkpoint budget uses the exact remaining context without a fixed output cap", () => {
  const budget = { promptTokens: 239028, contextWindow: 253792 };
  assert.equal(checkpointOutputBudget(budget), 14764);
  assert.equal(checkpointOutputBudget({ ...budget, promptTokens: 229889 }), 23903);
  assert.equal(checkpointOutputBudget({ ...budget, promptTokens: 240478 }), 13314);
  assert.equal(checkpointOutputBudget({ ...budget, promptTokens: 100000 }), 153792);
  assert.equal(checkpointOutputBudget({ ...budget, promptTokens: 247792 }), 6000);
  assert.equal(checkpointOutputBudget({ ...budget, promptTokens: 251744 }), 2048);
  assert.throws(() => checkpointOutputBudget({ ...budget, promptTokens: 251745 }), /context remaining/);
  assert.throws(() => checkpointOutputBudget({ ...budget, promptTokens: 253793 }), /only 0 tokens of context remaining/);
  assert.throws(() => checkpointOutputBudget({ ...budget, promptTokens: -1 }), /invalid checkpoint token budget/);
  assert.throws(() => checkpointOutputBudget({ ...budget, promptTokens: 1.5 }), /invalid checkpoint token budget/);
  assert.throws(() => checkpointOutputBudget({ ...budget, contextWindow: NaN }), /invalid checkpoint token budget/);
});

test("truncated, unfinished, protocol and section-incomplete summaries never commit", () => {
  assert.equal(validateSummary(`${summary}\nEND`, "stop", "END"), summary);
  assert.equal(validateSummary(`${summary}\nEarlier instruction mentioned END\nEND`, "stop", "END"), `${summary}\nEarlier instruction mentioned END`);
  for (const reason of [undefined, "length", "error", "tool_calls"]) {
    assert.throws(() => validateSummary(`${summary}\nEND`, reason, "END"), /incomplete/);
  }
  assert.throws(() => validateSummary(summary, "stop", "END"), /marker/);
  assert.throws(() => validateSummary(`END\n${summary}\nEND`, "stop", "END"), /duplicated/);
  assert.throws(() => validateSummary("short\nEND", "stop", "END"), /section/);
  assert.throws(() => validateSummary(`${summary}\n<tool_call>bad</tool_call>\nEND`, "stop", "END"), /protocol/);
});

test("terminal bullet marker is cosmetic, but missing, duplicate, inline and fenced markers still fail", () => {
  for (const marker of ["END", "- END", "* END", "+ END", "  - END  "]) {
    assert.equal(validateSummary(`${summary}\n${marker}`, "stop", "END"), summary);
  }
  for (const tail of ["END\n- END", "- END\nEND", "- END\n- END", "All done END", "- END more", "**END**", "- END\ntrailing text"]) {
    assert.throws(() => validateSummary(`${summary}\n${tail}`, "stop", "END"), /marker/);
  }
  assert.throws(() => validateSummary(`${summary}\n- END`, "length", "END"), /incomplete/);
  assert.throws(() => validateSummary(`${summary}\n\x60\x60\x60text\n- END`, "stop", "END"), /code fence/);
  assert.throws(() => validateSummary(`${summary.replace("### Key Decisions", "Decisions")}\n- END`, "stop", "END"), /missing required section: Key Decisions/);
});

test("SSE handles arbitrary byte boundaries and Unicode; requires finish and DONE", async () => {
  const result = await readCompletion(sse([choice("λ—é"), choice("", "stop"), { usage: { completion_tokens: 3 } }, "[DONE]"], 1), signal());
  assert.equal(result.text, "λ—é");
  assert.equal(result.usage.completion_tokens, 3);
  await assert.rejects(readCompletion(sse([choice("lost"), "[DONE]"]), signal()), /finish_reason/);
  await assert.rejects(readCompletion(sse([choice("lost", "stop")]), signal()), /DONE/);
  await assert.rejects(readCompletion(sse([{ error: { message: "failed" } }, "[DONE]"]), signal()), /stream error/);
  await assert.rejects(readCompletion(sse([choice("", "stop"), choice("late"), "[DONE]"]), signal()), /followed/);
  const abort = new AbortController(); abort.abort();
  await assert.rejects(readCompletion(sse([choice("hi", "stop"), "[DONE]"]), abort.signal), /abort/i);
});

test("actual Hyptheses typo and harmless heading formatting normalize without changing body text", () => {
  for (const variant of ["### Unresolved Questions & Hyptheses", "## **unresolved questions and hypotheses**:",
    "### Unresolved   Questions & Hypotheses ###", "### Unresolved Questions & Hypothesess"]) {
    const input = summary.replace("### Unresolved Questions & Hypotheses", variant);
    assert.equal(validateSummary(`${input}\nEND`, "stop", "END"), summary);
  }
  const body = `${summary}\nBody spelling stays Hyptheses, config=HYPTHESIS_V1 and code stay byte-for-byte.\nEND`;
  assert.equal(validateSummary(body, "stop", "END"), body.slice(0, -4));
  assert.throws(() => validateSummary(`${summary.replace("### Goal", "### Coal")}\nEND`, "stop", "END"), /missing required section: Goal/);
  assert.throws(() => validateSummary(`${summary.replace("### Unresolved Questions & Hypotheses", "### Resolved Questions & Answers")}\nEND`, "stop", "END"), /missing required section/);
  assert.throws(() => validateSummary(`${summary}\n## goal\nEND`, "stop", "END"), /duplicate/);
  const fenced = summary.replace("### Unresolved Questions & Hypotheses", "```md\n### Unresolved Questions & Hypotheses\n```");
  assert.throws(() => validateSummary(`${fenced}\nEND`, "stop", "END"), /missing required section/);
  const falseFenceEnd = fenced.replace("```md\n", "```md\n```not-a-closing-fence\n");
  assert.throws(() => validateSummary(`${falseFenceEnd}\nEND`, "stop", "END"), /missing required section/);
});

function fixture({ expectedLimit = 253786, completionTokens = 350 } = {}) {
  const preparation = { firstKeptEntryId: "kept", tokensBefore: 60000, isSplitTurn: true,
    settings: { reserveTokens: 16384 }, fileOps: { read: new Set(["a", "b"]), written: new Set(["a"]), edited: new Set(["c"]) } };
  const payload = { model: "fixture", messages: [{ role: "user", content: "preserve exact fact A=742" }],
    tools: [{ type: "function", function: { name: "read", parameters: {} } }],
    chat_template_kwargs: { enable_thinking: true, preserve_thinking: true, reasoning_effort: "xhigh" },
    cache_salt: "fixture-salt", kv_transfer_params: { qwen_chat: { id: "a", generation: "b" } } };
  const model = { baseUrl: "http://fixture/v1", maxTokens: 32768, contextWindow: 253792 };
  const requests = [];
  let marker, badFinish = false;
  async function fetcher(url, options) {
    const body = JSON.parse(options.body); requests.push({ url, body });
    if (url.endsWith("/tokenize")) {
      let tokens;
      if (body.prompt) tokens = body.prompt.includes("</think>") ? [4, 6, 7] : [4, 5];
      else if (body.add_generation_prompt) {
        marker = body.messages.at(-1).content.match(/COMPACTION_SUMMARY_COMPLETE/)[0];
        assert.deepEqual(body.tools, payload.tools);
        assert.deepEqual(body.chat_template_kwargs, payload.chat_template_kwargs);
        assert.deepEqual(body.messages.slice(0, -1), payload.messages);
        tokens = [1, 2, 3, 4, 5];
      } else tokens = [1, 2];
      return Response.json({ tokens });
    }
    assert.equal(url, "http://fixture/v1/completions");
    assert.deepEqual(body.prompt, [1, 2, 3, 4, 6, 7]);
    assert.equal(body.max_tokens, expectedLimit);
    assert.equal(body.return_token_ids, undefined);
    assert.equal(body.stop, undefined);
    assert.equal(body.kv_transfer_params.qwen_snapshot_force_flush, true);
    assert.ok(body.kv_transfer_params.qwen_chat);
    return sse([choice(`${summary}\n${marker}`, badFinish ? "length" : "stop"),
      { usage: { prompt_tokens: 6, completion_tokens: completionTokens, prompt_tokens_details: { cached_tokens: 2 } } }, "[DONE]"]);
  }
  return { input: { payload, preparation, model, policy: "policy", signal: signal(), headers: {}, fetcher,
    sourceIdentity: { sessionFile: "session-a", leaf: "leaf-a" } }, requests, truncate: () => { badFinish = true; } };
}

test("one summary covers split turns, retains tools and exact file tracking", async () => {
  const { input, requests } = fixture();
  const before = JSON.stringify(input.payload);
  const phases = [];
  const result = await compactFromPayload({ ...input, progress: (value) => phases.push(value.phase) });
  assert.equal(JSON.stringify(input.payload), before);
  assert.equal(requests.filter((request) => request.url.endsWith("/completions")).length, 1);
  assert.equal(result.firstKeptEntryId, "kept");
  assert.equal(result.details.splitTurn, true);
  assert.equal(result.details.contract, CONTRACT);
  assert.equal(result.usage.cacheRead, 2);
  assert.equal(result.usage.reasoning, 0);
  assert.deepEqual(result.details.readFiles, ["b"]);
  assert.deepEqual(result.details.modifiedFiles, ["a", "c"]);
  assert.match(result.summary, /<modified-files>\na\nc\n<\/modified-files>/);
  assert.equal(fileOperations(input.preparation.fileOps).suffix.length > 0, true);
  assert.deepEqual(phases.filter((phase, index) => phase !== phases[index - 1]),
    ["tokenize", "submit", "wait", "generate", "validate"]);
});

test("a checkpoint can exceed former fixed, ordinary-response and reserve limits", async () => {
  const { input } = fixture({ completionTokens: 35000 });
  const updates = [];
  const result = await compactFromPayload({ ...input, progress: (value) => updates.push(value) });
  assert.equal(result.usage.output, 35000);
  assert.equal(updates.find((value) => value.phase === "submit").outputTokenLimit, 253786);
  assert.equal(updates.at(-1).finishReason, "stop");
});

test("a nearly full context clamps the HTTP output allowance instead of rejecting a usable prompt", async () => {
  const { input, requests } = fixture({ expectedLimit: 6000 });
  input.model.contextWindow = 6006;
  const updates = [];
  const result = await compactFromPayload({ ...input, progress: (value) => updates.push(value) });
  assert.ok(result.summary);
  const request = requests.find((value) => value.url.endsWith("/completions")).body;
  assert.equal(request.prompt.length + request.max_tokens, input.model.contextWindow);
  assert.equal(updates.find((value) => value.phase === "submit").outputTokenLimit, 6000);
});

test("insufficient exact context fails before a checkpoint generation request", async () => {
  const { input, requests } = fixture();
  input.model.contextWindow = 2053;
  await assert.rejects(compactFromPayload(input), /only 2047 tokens of context remaining/);
  assert.equal(requests.filter((value) => value.url.endsWith("/completions")).length, 0);
});

test("completed receipt survives a new invocation; another session/leaf cannot reuse it", async () => {
  const receiptDirectory = await mkdtemp(join(tmpdir(), "radiance-compaction-test-"));
  const first = fixture();
  const original = await compactFromPayload({ ...first.input, receiptDirectory });
  const second = fixture();
  const resumed = await compactFromPayload({ ...second.input, receiptDirectory });
  assert.equal(resumed.summary, original.summary);
  assert.equal(resumed.details.receiptReused, true);
  assert.equal(second.requests.length, 0);
  const third = fixture();
  await compactFromPayload({ ...third.input, receiptDirectory, sourceIdentity: { sessionFile: "session-b", leaf: "leaf-a" } });
  assert.equal(third.requests.length, 5);
});

test("failed completion and HTTP failures do not write usable receipts", async () => {
  const receiptDirectory = await mkdtemp(join(tmpdir(), "radiance-compaction-test-"));
  const broken = fixture(); broken.truncate();
  await assert.rejects(compactFromPayload({ ...broken.input, receiptDirectory }), /incomplete/);
  const retry = fixture();
  await compactFromPayload({ ...retry.input, receiptDirectory });
  assert.equal(retry.requests.length, 5);
  const down = fixture();
  await assert.rejects(compactFromPayload({ ...down.input, fetcher: async () => new Response("down", { status: 503 }) }), /HTTP 503/);
  const corrupted = join(receiptDirectory, "corrupted.json");
  await writeFile(corrupted, JSON.stringify({ contract: CONTRACT, key: "corrupted", result: {}, resultHash: "wrong" }));
  await assert.rejects(readReceipt(receiptDirectory, "corrupted"), /invalid/);
  assert.ok(await readFile(corrupted));
});

async function saveLegacyFailure(input, receiptDirectory, overrides = {}) {
  const key = compactionReceiptKey(input);
  const marker = "COMPACTION_SUMMARY_COMPLETE";
  const artifact = { contract: CONTRACT, sourceIdentity: input.sourceIdentity, marker,
    error: "checkpoint is missing required section: Unresolved Questions & Hypotheses",
    completion: { text: `${summary.replace("Hypotheses", "Hyptheses")}\n${marker}`, finishReason: "stop",
      usage: { prompt_tokens: 6, completion_tokens: 350, prompt_tokens_details: { cached_tokens: 2 } } }, ...overrides };
  await writeReceipt(receiptDirectory, `${key}.failed`, artifact);
  return { key, artifact, bytes: await readFile(join(receiptDirectory, `${key}.failed.json`)) };
}

test("legacy completed failure is reused with zero generation and the original artifact is unchanged", async () => {
  const receiptDirectory = await mkdtemp(join(tmpdir(), "radiance-completed-recovery-"));
  const example = fixture();
  const saved = await saveLegacyFailure(example.input, receiptDirectory);
  const result = await compactFromPayload({ ...example.input, receiptDirectory, fetcher: (url, options) => {
    assert.ok(url.endsWith("/tokenize"), "a saved completed summary must never regenerate");
    return example.input.fetcher(url, options);
  } });
  assert.equal(result.details.recoveredCompletedSummary, true);
  assert.equal(result.details.summaryRequests, 0);
  assert.equal(result.details.recoveredReceiptKey, saved.key);
  assert.ok(result.summary.startsWith(summary));
  assert.equal(example.requests.length, 4);
  assert.deepEqual(await readFile(join(receiptDirectory, `${saved.key}.failed.json`)), saved.bytes);
  const retry = fixture();
  const prior = await compactFromPayload({ ...retry.input, receiptDirectory, fetcher: () => assert.fail("published recovery must be network-free") });
  assert.equal(prior.details.receiptReused, true);
  assert.ok((await readdir(receiptDirectory)).includes(`${saved.key}.json`));
});

test("the incident's bullet-marker failure recovers its completed checkpoint without another generation", async () => {
  const receiptDirectory = await mkdtemp(join(tmpdir(), "radiance-bullet-recovery-"));
  const example = fixture();
  const saved = await saveLegacyFailure(example.input, receiptDirectory, {
    error: "checkpoint completion marker missing or duplicated",
    completion: { text: `${summary}\n- COMPACTION_SUMMARY_COMPLETE`, finishReason: "stop",
      usage: { prompt_tokens: 6, completion_tokens: 350, prompt_tokens_details: { cached_tokens: 2 } } },
  });
  const phases = [];
  const result = await compactFromPayload({ ...example.input, receiptDirectory, progress: (value) => phases.push(value), fetcher: (url, options) => {
    assert.ok(url.endsWith("/tokenize"), "recovering a completed checkpoint cannot generate again");
    return example.input.fetcher(url, options);
  } });
  assert.equal(result.details.summaryRequests, 0);
  assert.ok(result.summary.startsWith(`${summary}\n\n<read-files>`), "the bullet must not remain in the checkpoint body");
  assert.equal(phases.some((value) => ["submit", "wait", "generate"].includes(value.phase)), false);
  assert.equal(phases.at(-1).phase, "validate");
  assert.deepEqual(await readFile(join(receiptDirectory, `${saved.key}.failed.json`)), saved.bytes);
});

test("retry identity traversal stops at real activity; error-only descendants can reuse the same provider prompt", async () => {
  const branch = [
    { id: "leaf-a", type: "message", message: { role: "toolResult" } },
    { id: "error1", parentId: "leaf-a", type: "message", message: { role: "assistant", stopReason: "error" } },
    { id: "error2", parentId: "error1", type: "message", message: { role: "assistant", stopReason: "aborted" } },
  ];
  const receiptDirectory = await mkdtemp(join(tmpdir(), "radiance-retry-recovery-"));
  const example = fixture();
  await saveLegacyFailure(example.input, receiptDirectory);
  const sourceIdentity = { sessionFile: "session-a", leaf: "error2" };
  const retryIdentities = compactionRetryIdentities(branch, sourceIdentity);
  assert.deepEqual(retryIdentities.map((identity) => identity.leaf), ["error1", "leaf-a"]);
  const result = await compactFromPayload({ ...example.input, receiptDirectory, sourceIdentity, retryIdentities });
  assert.equal(result.details.summaryRequests, 0);
  const diagnosticBranch = [...branch, { id: "diagnostic", parentId: "error2", type: "custom", customType: "qwen-radiance-backend-error-v1" }];
  const diagnosticIdentity = { ...sourceIdentity, leaf: "diagnostic" };
  const diagnosticRetries = compactionRetryIdentities(diagnosticBranch, diagnosticIdentity);
  assert.deepEqual(diagnosticRetries.map((identity) => identity.leaf), ["error2", "error1", "leaf-a"]);
  const retry = await compactFromPayload({ ...fixture().input, receiptDirectory, sourceIdentity: diagnosticIdentity,
    retryIdentities: diagnosticRetries });
  assert.equal(retry.details.summaryRequests, 0);
  branch.push({ id: "new-user", parentId: "error2", type: "message", message: { role: "user" } });
  assert.deepEqual(compactionRetryIdentities(branch, { ...sourceIdentity, leaf: "new-user" }), []);
  assert.deepEqual(compactionRetryIdentities(branch, { ...sourceIdentity, leaf: "unknown" }), []);
});

test("completed failure recovery refuses mismatched usage, token hashes and other sessions", async () => {
  for (const change of ["usage", "token hash", "source"]) {
    const receiptDirectory = await mkdtemp(join(tmpdir(), "radiance-recovery-binding-"));
    const example = fixture();
    const { key, artifact } = await saveLegacyFailure(example.input, receiptDirectory);
    if (change === "usage") artifact.completion.usage.prompt_tokens++;
    if (change === "token hash") artifact.promptSha256 = "wrong";
    if (change === "source") artifact.sourceIdentity = { sessionFile: "other", leaf: "leaf-a" };
    await writeReceipt(receiptDirectory, `${key}.failed`, artifact);
    await assert.rejects(compactFromPayload({ ...example.input, receiptDirectory }), /binding|prefix differs|prompt length/);
    assert.equal(example.requests.filter((request) => request.url.endsWith("/completions")).length, 0);
    assert.ok(!(await readdir(receiptDirectory)).includes(`${key}.json`));
  }
});

test("Pi hook commits through Pi only; capture cannot hit network and errors cancel rather than run stock", async (t) => {
  const originalDir = process.env.PI_CODING_AGENT_DIR;
  const originalAbi = process.env.QWEN_RADIANCE_CACHE_ABI;
  process.env.PI_CODING_AGENT_DIR = await mkdtemp(join(tmpdir(), "radiance-compaction-hook-"));
  process.env.QWEN_RADIANCE_CACHE_ABI = "a".repeat(64);
  t.after(() => {
    if (originalDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = originalDir;
    if (originalAbi === undefined) delete process.env.QWEN_RADIANCE_CACHE_ABI;
    else process.env.QWEN_RADIANCE_CACHE_ABI = originalAbi;
  });
  const good = fixture();
  t.mock.method(globalThis, "fetch", good.input.fetcher);
  const handlers = new Map(), notices = [], messages = [], active = ["read"];
  const pi = { on: (name, handler) => handlers.set(name, handler), getActiveTools: () => active,
    getAllTools: () => [{ name: "read", parameters: {}, description: "read" }], getThinkingLevel: () => "xhigh" };
  const ctx = { model: { ...good.input.model, id: MODEL },
    modelRegistry: { getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "local" }) },
    getSystemPrompt: () => "exact system",
    sessionManager: { getLeafId: () => "leaf", getSessionFile: () => "one-session",
      getSessionId: () => "session-id", getEntries: () => [], getCwd: () => "/work",
      getSessionName: () => "Synthetic",
      buildSessionContext: () => ({ messages: good.input.payload.messages }) },
    ui: { notify: (message) => notices.push(message), setWidget: (_key, lines) => assert.equal(lines, undefined),
      setWorkingMessage: (message) => messages.push(message), setStatus: () => {} } };
  const flushed = [];
  installRadianceCompaction(pi, { convertToLlm: (value) => value, flushSnapshotTail: async (chat) => flushed.push(chat),
    streamSimpleOpenAICompletions: (_model, context, options) => ({
    result: async () => {
      assert.equal(context.systemPrompt, "exact system");
      assert.equal(options.reasoning, "xhigh");
      assert.deepEqual(context.messages, good.input.payload.messages);
      try { options.onPayload(good.input.payload); } catch (error) { return { errorMessage: error.message }; }
      assert.fail("payload capture did not halt before network");
    },
  }) });
  const hook = handlers.get("session_before_compact");
  const event = { preparation: good.input.preparation, signal: signal() };
  const completed = (await hook(event, ctx)).compaction;
  assert.ok(completed);
  assert.equal(completed.details.snapshotTailFlushed, true);
  assert.equal(flushed.length, 1);
  assert.match(flushed[0].id, /^[0-9a-f]{64}$/);
  assert.match(flushed[0].generation, /^[0-9a-f]{64}$/);
  assert.equal(getCompactionProgress(ctx).snapshot().state, "running");
  assert.equal(getCompactionProgress(ctx).snapshot().phase, "commit");
  assert.match(messages.at(-1), /Commit conversation/);
  assert.doesNotMatch(messages.at(-1), /Compacted/);
  assert.equal(notices.length, 0);
  assert.equal(await hook(event, { ...ctx, model: { id: "vanilla" } }), undefined);
  const abort = new AbortController(); abort.abort();
  assert.deepEqual(await hook({ ...event, signal: abort.signal }, ctx), { cancel: true });
  assert.match(notices.at(-1), /Transcript retained/);
  ctx.modelRegistry.getApiKeyAndHeaders = async () => ({ ok: false, error: "missing auth" });
  assert.deepEqual(await hook(event, ctx), { cancel: true });
  assert.match(notices.at(-1), /missing auth/);
  const reportFiles = (await readdir(process.env.PI_CODING_AGENT_DIR + "/radiance-compaction-receipts")).filter((name) => name.endsWith(".progress.json"));
  assert.equal(reportFiles.length, 1, "each attempt replaces this chat's previous timing report");
  const report = JSON.parse(await readFile(join(process.env.PI_CODING_AGENT_DIR, "radiance-compaction-receipts", reportFiles[0]), "utf8"));
  assert.equal(report.state, "failed");
  assert.equal(report.phase, "prepare");
  assert.doesNotMatch(JSON.stringify(report), /exact system|preserve exact fact|one-session/);
  await handlers.get("session_shutdown")({}, ctx);
  await handlers.get("session_start")({}, ctx);
  assert.equal(getCompactionProgress(ctx), undefined);
  assert.equal(messages.at(-1), undefined);
});
