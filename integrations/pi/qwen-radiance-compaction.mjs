import { createHash, randomUUID } from "node:crypto";
import { constants, readFileSync } from "node:fs";
import { lstat, mkdir, open, rename } from "node:fs/promises";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import { fileURLToPath } from "node:url";
import {
  flushRadianceSnapshotTail,
  radianceIdentityPath,
  withRadianceChat,
} from "./qwen-radiance-cache.mjs";
import { appendCompactionTiming, startCompactionProgress, getCompactionProgress, clearCompactionProgress } from "./qwen-radiance-compaction-progress.mjs";
import { BACKEND_ERROR_ENTRY, backendErrorMessage, reportBackendFailure } from "./qwen-radiance-errors.mjs";

// This is a Pi-only adapter. It does not change the serving/snapshot ABI, model
// sampling in ordinary turns, or the old fixed-slot compaction implementation.
export const CONTRACT = "radiance-prefix-compaction-v1";
export const MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate";
const CAPTURE = "RADIANCE_COMPACTION_CAPTURE_BEFORE_NETWORK";
const OPEN = "<|im_start|>assistant\n<think>\n";
const CLOSED = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
const HEADINGS = ["Goal", "Current Authoritative State", "Constraints & Invariants", "Progress",
  "Measurements & Evidence", "Key Decisions", "Rejected / Failed Approaches",
  "Unresolved Questions & Hypotheses", "Next Steps", "Critical Context"];
const hash = (value) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
const receiptDirectory = () => join(process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent"), "radiance-compaction-receipts");
const progressReceiptKey = (sessionFile) => `${hash({ sessionFile: radianceIdentityPath(sessionFile) })}.progress`;

function headingKey(text) {
  return text.replace(/\s+#+\s*$/, "").replace(/[*_`]/g, "").replace(/:\s*$/, "")
    .toLowerCase().replace(/&/g, " and ").replace(/\s+/g, " ").trim();
}

function oneEditApart(left, right) {
  if (Math.abs(left.length - right.length) > 1) return false;
  let i = 0, j = 0, edits = 0;
  while (i < left.length && j < right.length) {
    if (left[i] === right[j]) { i++; j++; continue; }
    if (++edits > 1) return false;
    if (left.length >= right.length) i++;
    if (right.length >= left.length) j++;
  }
  return edits + (left.length - i) + (right.length - j) <= 1;
}

// Only normalize actual Markdown headings, never body text or code examples.
// A single spelling edit in a long, unambiguous title is cosmetic; absent
// sections, duplicate sections, premature stops and missing markers still fail.
export function normalizeSummaryHeadings(summary) {
  const seen = new Set();
  let fence;
  const normalized = summary.split("\n").map((line) => {
    const fenceMatch = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (fenceMatch) {
      if (!fence) fence = fenceMatch[1];
      else if (fenceMatch[1][0] === fence[0] && fenceMatch[1].length >= fence.length &&
        !line.slice(fenceMatch[0].length).trim()) fence = undefined;
      return line;
    }
    if (fence) return line;
    const match = line.match(/^\s{0,3}#{1,6}\s+(.+?)\s*$/);
    if (!match) return line;
    const key = headingKey(match[1]);
    const exact = HEADINGS.filter((heading) => headingKey(heading) === key);
    const candidates = exact.length ? exact : HEADINGS.filter((heading) => {
      const expected = headingKey(heading);
      return expected.length >= 12 && oneEditApart(key, expected);
    });
    if (!candidates.length) return line;
    if (candidates.length !== 1 || seen.has(candidates[0])) throw new Error("checkpoint has ambiguous or duplicate sections");
    seen.add(candidates[0]);
    return `### ${candidates[0]}`;
  }).join("\n");
  for (const heading of HEADINGS) {
    if (!seen.has(heading)) throw new Error(`checkpoint is missing required section: ${heading}`);
  }
  if (fence) throw new Error("checkpoint ends inside a code fence");
  return normalized;
}

export function summaryInstruction(policy, marker, customInstructions = "") {
  // The Qwen template scans user text for these controls and rewrites the SYSTEM
  // prompt. Reject them rather than quietly throwing away the existing prefix.
  if (/<\|(?:think|im_)/.test(customInstructions)) throw new Error("compaction instructions contain chat-template controls");
  return `${policy}\n\nThis is a checkpoint request, not a request to continue the task. ` +
    `Summarize the whole preceding conversation, including its previous checkpoint and the current partial tool turn, in ONE checkpoint. ` +
    `Recent messages will also be retained verbatim by Pi. This final checkpoint instruction is scaffolding: do NOT summarize it or copy its formatting rules into the checkpoint. ` +
    `Do not call tools, answer the previous user request, or produce a reasoning preamble. ` +
    `Aim for 2500–3000 tokens; be shorter when the state is simple. Preserve consequential facts before incidental detail. ` +
    `Include every required ### heading (use None when empty). Finish with this exact line: ${marker}` +
    (customInstructions ? `\n\nAdditional user compaction focus:\n${customInstructions}` : "");
}

export function assertTokens(tokens) {
  if (!Array.isArray(tokens) || !tokens.length || !tokens.every((t) => Number.isSafeInteger(t) && t >= 0)) {
    throw new Error("tokenizer returned an invalid token vector");
  }
  return tokens;
}

export function checkpointOutputBudget({ promptTokens, contextWindow }) {
  for (const [name, value] of Object.entries({ promptTokens, contextWindow })) {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error(`invalid checkpoint token budget: ${name}`);
  }
  const available = contextWindow - promptTokens;
  if (available < 2048) {
    throw new Error(`checkpoint has only ${Math.max(0, available)} tokens of context remaining; at least 2048 required; original transcript retained`);
  }
  // The exact prompt already includes the checkpoint instructions. Neither Pi's
  // compaction reserve nor stale model metadata should cap the checkpoint.
  // Like ordinary Radiance replies, it can use all remaining context.
  return available;
}

export function closeSummaryThinking(history, continuation, openHeader, closedHeader, capacity, maxTokens) {
  for (const tokens of [history, continuation, openHeader, closedHeader]) assertTokens(tokens);
  const mismatch = history.findIndex((token, i) => token !== continuation[i]);
  if (mismatch !== -1) throw new Error(`summary changed the conversation prefix at token ${mismatch}`);
  const endsWith = (suffix) => suffix.every((token, i) => token === continuation[continuation.length - suffix.length + i]);
  let prompt;
  if (endsWith(closedHeader)) prompt = continuation;
  else if (endsWith(openHeader)) prompt = [...continuation.slice(0, -openHeader.length), ...closedHeader];
  else throw new Error("unrecognized Qwen generation boundary; refusing to change the historical prompt");
  if (prompt.length + maxTokens > capacity) {
    throw new Error(`checkpoint needs ${prompt.length + maxTokens} tokens, context limit is ${capacity}; original transcript retained`);
  }
  return prompt;
}

export function validateSummary(text, finishReason, marker) {
  if (finishReason !== "stop") throw new Error(`incomplete checkpoint (finish_reason=${finishReason ?? "missing"})`);
  const lines = text.trim().split("\n");
  // Qwen sometimes makes the final marker a Markdown bullet. Remove only that
  // wrapper, never invent a marker or accept inline prose as a terminator.
  const isMarker = (line) => line.trim().replace(/^[-*+]\s+/, "") === marker;
  if (!isMarker(lines.at(-1)) || lines.filter(isMarker).length !== 1) {
    throw new Error("checkpoint completion marker missing or duplicated");
  }
  const summary = lines.slice(0, -1).join("\n").trim();
  if (/<\/?(?:think|tool_call|tool_response)>|<\|im_/.test(summary)) throw new Error("checkpoint contains reasoning/tool protocol output");
  return normalizeSummaryHeadings(summary);
}

export function fileOperations(fileOps) {
  const modifiedFiles = [...new Set([...fileOps.edited, ...fileOps.written])].sort();
  const readFiles = [...fileOps.read].filter((path) => !modifiedFiles.includes(path)).sort();
  const suffix = [["read-files", readFiles], ["modified-files", modifiedFiles]]
    .filter(([, paths]) => paths.length).map(([tag, paths]) => `\n\n<${tag}>\n${paths.join("\n")}\n</${tag}>`).join("");
  return { readFiles, modifiedFiles, suffix };
}

export async function postJson(url, body, headers, signal, fetcher = fetch) {
  signal?.throwIfAborted();
  const response = await fetcher(url, { method: "POST", headers, body: JSON.stringify(body), signal });
  if (!response.ok) {
    await response.body?.cancel();
    throw new Error(`compaction endpoint returned HTTP ${response.status}`);
  }
  return response;
}

// Require both the provider finish_reason and the final SSE marker. Never accept
// a disconnected stream as a completed checkpoint, even if its text looks good.
export async function readCompletion(response, signal, progress = () => {}) {
  if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new Error("compaction endpoint did not return SSE");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "", text = "", finishReason, usage, done = false, tokens = 0;
  function consume(frame) {
    const data = frame.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
    if (!data) return;
    if (done) throw new Error("compaction stream has data after DONE");
    if (data === "[DONE]") { done = true; return; }
    const chunk = JSON.parse(data);
    if (chunk.error) throw new Error("compaction provider returned a stream error");
    if (chunk.usage) usage = chunk.usage;
    for (const choice of chunk.choices ?? []) {
      if (choice.index !== 0) throw new Error("unexpected multiple compaction choices");
      if (finishReason && choice.text) throw new Error("compaction text followed finish_reason");
      text += choice.text ?? "";
      tokens += choice.token_ids?.length ?? 0;
      if (choice.finish_reason) finishReason = choice.finish_reason;
    }
    if (text.length > 150000) throw new Error("checkpoint output exceeded safety bound");
    progress(usage?.completion_tokens ?? tokens, text.length, usage);
  }
  try {
    while (true) {
      signal?.throwIfAborted();
      const next = await reader.read();
      pending += decoder.decode(next.value, { stream: !next.done });
      pending = pending.replace(/\r\n/g, "\n");
      let boundary;
      while ((boundary = pending.indexOf("\n\n")) >= 0) {
        consume(pending.slice(0, boundary)); pending = pending.slice(boundary + 2);
      }
      if (next.done) break;
      if (pending.length > 200000) throw new Error("oversized compaction SSE frame");
    }
    if (pending.trim()) consume(pending);
    signal?.throwIfAborted();
    if (!done || !finishReason) throw new Error("compaction stream ended before finish_reason/DONE");
    return { text, finishReason, usage };
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

// A completed receipt can be reused after Pi exits between summary completion
// and appendCompaction. Partial summaries never become authoritative. The source
// transcript is not rewritten by this adapter; Pi owns the single commit.
export async function writeReceipt(directory, key, receipt) {
  await mkdir(directory, { recursive: true, mode: 0o700 });
  const metadata = await lstat(directory);
  if (!metadata.isDirectory() || metadata.isSymbolicLink() || metadata.uid !== process.getuid() || (metadata.mode & 0o077)) {
    throw new Error("compaction receipt directory must be private and owned by this user");
  }
  const path = join(directory, `${key}.json`);
  const temporary = join(directory, `${key}.${randomUUID()}.tmp`);
  const handle = await open(temporary, "wx", 0o600);
  try { await handle.writeFile(JSON.stringify(receipt)); await handle.sync(); }
  finally { await handle.close(); }
  await rename(temporary, path);
  const dir = await open(directory, "r");
  try { await dir.sync(); } finally { await dir.close(); }
}

async function readPrivateJson(directory, key) {
  let handle;
  try {
    handle = await open(join(directory, `${key}.json`), constants.O_RDONLY | constants.O_NOFOLLOW);
    const stat = await handle.stat();
    if (!stat.isFile() || stat.uid !== process.getuid() || stat.nlink !== 1 || stat.size > 200000) {
      throw new Error("unsafe compaction receipt");
    }
    return JSON.parse(await handle.readFile("utf8"));
  } catch (error) {
    if (error.code === "ENOENT") return undefined;
    throw error;
  } finally { await handle?.close(); }
}

export async function readReceipt(directory, key) {
  const receipt = await readPrivateJson(directory, key);
  if (!receipt) return undefined;
  if (receipt.contract !== CONTRACT || receipt.key !== key || hash(receipt.result) !== receipt.resultHash) {
    throw new Error("invalid compaction receipt");
  }
  const summary = validateSummary(`${receipt.result.summary}\n${receipt.marker}`, "stop", receipt.marker);
  return { ...receipt.result, summary };
}

export function compactionReceiptKey({ payload, policy, customInstructions, model, preparation, sourceIdentity }) {
  // Preserve v1's field order so the already-saved incident can be recovered.
  return hash({ contract: CONTRACT, payload, policy, customInstructions, model,
    firstKeptEntryId: preparation.firstKeptEntryId, sourceIdentity, fileOps: fileOperations(preparation.fileOps) });
}

// Walk back across discarded error/abort messages and display-only diagnostics. A real user,
// assistant or tool result is a new context and must never reuse an old summary.
export function compactionRetryIdentities(branchEntries, sourceIdentity) {
  const result = [];
  let index = branchEntries.findIndex((entry) => entry.id === sourceIdentity.leaf);
  while (index > 0 && result.length < 16) {
    const entry = branchEntries[index];
    const discarded = entry.type === "message" && entry.message?.role === "assistant" &&
      ["error", "aborted"].includes(entry.message.stopReason);
    const diagnostic = entry.type === "custom" && entry.customType === BACKEND_ERROR_ENTRY;
    if (!discarded && !diagnostic) break;
    if (entry.parentId !== branchEntries[index - 1].id) break;
    index--;
    result.push({ ...sourceIdentity, leaf: branchEntries[index].id });
  }
  return result;
}

export async function compactFromPayload({ payload, preparation, policy, customInstructions, model, headers,
  signal, receiptDirectory, sourceIdentity, retryIdentities = [], progress = () => {}, diagnostics = () => undefined, fetcher = fetch }) {
  const started = performance.now();
  signal?.throwIfAborted();
  const keyFor = (identity) => compactionReceiptKey({ payload, preparation, policy, customInstructions, model, sourceIdentity: identity });
  const key = keyFor(sourceIdentity);
  // A fixed protocol terminator is easier to reproduce than an arbitrary hash.
  // Session/prefix authentication belongs in the receipt key, not model prose.
  const marker = "COMPACTION_SUMMARY_COMPLETE";
  let recovered;
  if (receiptDirectory) {
    for (const identity of [sourceIdentity, ...retryIdentities]) {
      if (identity.sessionFile !== sourceIdentity.sessionFile) throw new Error("cross-session compaction recovery refused");
      const candidateKey = keyFor(identity);
      const prior = await readReceipt(receiptDirectory, candidateKey);
      if (prior) {
        progress({ phase: "validate", reusedCheckpoint: true, inputTokens: prior.usage.input + prior.usage.cacheRead,
          outputTokens: prior.usage.output, cacheRead: prior.usage.cacheRead });
        return { ...prior, details: { ...prior.details, receiptReused: true } };
      }
      const failed = await readPrivateJson(receiptDirectory, `${candidateKey}.failed`);
      if (!failed) continue;
      if (failed.contract !== CONTRACT || failed.marker !== marker || hash(failed.sourceIdentity) !== hash(identity)) {
        throw new Error("invalid failed-compaction source binding");
      }
      try { validateSummary(failed.completion?.text, failed.completion?.finishReason, marker); }
      catch { continue; } // Truly incomplete results are not recoverable.
      recovered = { ...failed, key: candidateKey };
      break;
    }
  }
  const baseUrl = model.baseUrl.replace(/\/+$/, "");
  if (!baseUrl.endsWith("/v1")) throw new Error("Radiance base URL must end in /v1");
  const tokenizerUrl = `${baseUrl.slice(0, -3)}/tokenize`;
  const chat = { model: payload.model, messages: payload.messages, tools: payload.tools,
    chat_template_kwargs: payload.chat_template_kwargs, add_special_tokens: false };
  const instruction = summaryInstruction(policy, marker, customInstructions);
  const tokenize = async (body) => assertTokens((await (await postJson(tokenizerUrl, body, headers, signal, fetcher)).json()).tokens);
  progress({ phase: "tokenize" });
  const [history, continuation, openHeader, closedHeader] = await Promise.all([
    tokenize({ ...chat, add_generation_prompt: false }),
    tokenize({ ...chat, messages: [...chat.messages, { role: "user", content: instruction }], add_generation_prompt: true }),
    tokenize({ model: payload.model, prompt: OPEN, add_special_tokens: false }),
    tokenize({ model: payload.model, prompt: CLOSED, add_special_tokens: false }),
  ]);
  const prompt = closeSummaryThinking(history, continuation, openHeader, closedHeader, model.contextWindow, 0);
  const maxTokens = checkpointOutputBudget({ promptTokens: prompt.length, contextWindow: model.contextWindow });
  progress({ phase: recovered ? "validate" : "submit", inputTokens: prompt.length, reusedCheckpoint: Boolean(recovered),
    ...(!recovered ? { outputTokenLimit: maxTokens } : {}) });
  let completion;
  if (recovered) {
    if (recovered.promptSha256 && recovered.promptSha256 !== hash(prompt)) throw new Error("saved checkpoint token prefix differs");
    completion = recovered.completion;
  } else {
    const responsePending = postJson(`${baseUrl}/completions`, {
      model: payload.model, prompt, max_tokens: maxTokens, temperature: 0.3, top_p: 0.95, top_k: 20,
      add_special_tokens: false, stream: true, stream_options: { include_usage: true, continuous_usage_stats: true },
      // Do not make the terminator a server stop-string: the model may quote the
      // instruction while summarizing constraints, before reaching its last line.
      // Continuous usage gives progress without echoing all 250K prompt token IDs
      // into the first SSE frame (vLLM's return_token_ids also includes the prompt).
      skip_special_tokens: false,
      ...(payload.cache_salt ? { cache_salt: payload.cache_salt,
        kv_transfer_params: { ...payload.kv_transfer_params, qwen_snapshot_force_flush: true } } : {}),
    }, headers, signal, fetcher);
    // fetch() has synchronously started the request. Begin scheduler observation
    // before awaiting response headers so queueing and cache work are not charged
    // to the HTTP-submission stage.
    progress({ phase: "wait" });
    const response = await responsePending;
    completion = await readCompletion(response, signal, (outputTokens, characters, usage) => progress({
      phase: outputTokens || characters ? "generate" : "wait",
      outputTokens, characters, cacheRead: usage?.prompt_tokens_details?.cached_tokens,
    }));
  }
  progress({ phase: "validate", outputTokens: completion.usage?.completion_tokens,
    cacheRead: completion.usage?.prompt_tokens_details?.cached_tokens, finishReason: completion.finishReason });
  let checkpoint;
  try {
    checkpoint = validateSummary(completion.text, completion.finishReason, marker);
  } catch (error) {
    if (receiptDirectory) await writeReceipt(receiptDirectory, `${key}.failed`, {
      contract: CONTRACT, sourceIdentity, marker, error: error.message, completion, promptSha256: hash(prompt),
      diagnostics: diagnostics(),
    });
    throw error;
  }
  const ops = fileOperations(preparation.fileOps);
  const raw = completion.usage;
  if (!raw || raw.prompt_tokens !== prompt.length || !Number.isSafeInteger(raw.completion_tokens)) {
    throw new Error("compaction usage did not authenticate the rendered prompt length");
  }
  const cacheRead = raw.prompt_tokens_details?.cached_tokens ?? 0;
  if (!Number.isSafeInteger(cacheRead) || cacheRead < 0 || cacheRead > raw.prompt_tokens || raw.completion_tokens < 1) {
    throw new Error("invalid compaction token accounting");
  }
  const usage = { input: raw.prompt_tokens - cacheRead, output: raw.completion_tokens, cacheRead, cacheWrite: 0,
    totalTokens: raw.prompt_tokens + raw.completion_tokens, reasoning: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const result = { summary: checkpoint + ops.suffix, firstKeptEntryId: preparation.firstKeptEntryId,
    tokensBefore: preparation.tokensBefore, usage,
    details: { readFiles: ops.readFiles, modifiedFiles: ops.modifiedFiles, contract: CONTRACT,
      sourceIdentity, promptSha256: hash(prompt), historicalTokens: history.length, prefixVerified: true,
      summaryRequests: recovered ? 0 : 1, splitTurn: preparation.isSplitTurn, elapsedMs: Math.round(performance.now() - started),
      diagnostics: diagnostics(),
      ...(recovered ? { recoveredCompletedSummary: true, recoveredReceiptKey: recovered.key } : {}) } };
  signal?.throwIfAborted();
  if (receiptDirectory) await writeReceipt(receiptDirectory, key, { contract: CONTRACT, key, marker, result, resultHash: hash(result) });
  return result;
}

export function installRadianceCompaction(pi, {
  convertToLlm,
  streamSimpleOpenAICompletions,
  flushSnapshotTail = flushRadianceSnapshotTail,
}) {
  const policy = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "qwen-loss-sensitive-compact-prompt.md"), "utf8");
  pi.on("session_start", async (_event, ctx) => {
    await clearCompactionProgress(ctx);
    if (ctx.model?.id !== MODEL) return;
    try {
      const sessionFile = ctx.sessionManager.getSessionFile();
      if (!sessionFile) return;
      const report = await readPrivateJson(receiptDirectory(), progressReceiptKey(sessionFile));
      if (!report?.transcriptAppended || !["complete", "cleanup_pending", "commit_unconfirmed"].includes(report.state)) return;
      const compactionEntry = ctx.sessionManager.getBranch().filter((entry) => entry.type === "compaction").at(-1);
      if (!compactionEntry || compactionEntry.details?.diagnostics?.attemptId !== report.attemptId) return;
      appendCompactionTiming(pi, { compactionEntry }, ctx, report);
    } catch (error) {
      ctx.ui.notify(`Could not restore completed compaction duration: ${error.message}`, "warning");
    }
  });
  pi.on("session_shutdown", async (_event, ctx) => clearCompactionProgress(ctx));
  pi.on("session_switch", async (_event, ctx) => clearCompactionProgress(ctx));
  pi.on("agent_settled", async (_event, ctx) => {
    const progress = getCompactionProgress(ctx);
    if (progress?.snapshot().state === "running") await progress.finish("commit_unconfirmed");
  });
  pi.on("session_compact", async (event, ctx) => {
    const progress = getCompactionProgress(ctx);
    if (!progress || progress.snapshot().state !== "running") return;
    progress.markAppended();
    // With disk snapshots enabled, the cache extension owns the durable flush
    // and cleanup stages. Otherwise Pi's append completes this operation.
    if (!process.env.QWEN_RADIANCE_CACHE_ABI) {
      await progress.finish("complete");
      try { appendCompactionTiming(pi, event, ctx, progress); }
      catch (error) { ctx.ui.notify(`Could not save completed compaction duration: ${error.message}`, "warning"); }
    }
  });
  pi.on("session_before_compact", async (event, ctx) => {
    if (ctx.model?.id !== MODEL) return;
    const attemptStarted = Date.now();
    let progress;
    try {
      const leaf = ctx.sessionManager.getLeafId();
      const sessionFile = ctx.sessionManager.getSessionFile();
      const receiptSessionFile = radianceIdentityPath(sessionFile);
      const directory = receiptDirectory();
      await clearCompactionProgress(ctx);
      progress = startCompactionProgress(ctx, { signal: event.signal,
        // Replace the previous timing report for this session; never store its
        // messages, summary, tool output, title or path in the diagnostic report.
        save: (report) => writeReceipt(directory, progressReceiptKey(sessionFile), report) });
      event.signal.throwIfAborted();
      const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
      if (!auth.ok) throw new Error(auth.error);
      const model = { ...ctx.model, baseUrl: auth.baseUrl ?? ctx.model.baseUrl };
      const allTools = new Map(pi.getAllTools().map((tool) => [tool.name, tool]));
      const tools = pi.getActiveTools().map((name) => allTools.get(name));
      if (tools.some((tool) => !tool)) throw new Error("active tool schema missing during compaction");
      let payload;
      const captured = await streamSimpleOpenAICompletions(model, {
        systemPrompt: ctx.getSystemPrompt(), messages: convertToLlm(ctx.sessionManager.buildSessionContext().messages), tools,
      }, { ...auth, reasoning: pi.getThinkingLevel(), maxTokens: 1, signal: event.signal,
        onPayload: (value) => { payload = value; throw new Error(CAPTURE); },
        fetch: () => { throw new Error("unexpected provider network call during compaction payload capture"); },
      }).result();
      if (!payload || captured.errorMessage !== CAPTURE) throw new Error("could not capture Pi's canonical provider prompt");
      payload = withRadianceChat(payload, ctx);
      const headers = { "Content-Type": "application/json", ...(auth.apiKey ? { Authorization: `Bearer ${auth.apiKey}` } : {}),
        ...Object.fromEntries(Object.entries(auth.headers ?? {}).filter(([, value]) => value != null)) };
      const compaction = await compactFromPayload({ payload, preparation: event.preparation, policy,
        customInstructions: event.customInstructions, model, headers, signal: event.signal,
        receiptDirectory: directory, sourceIdentity: { sessionFile: receiptSessionFile, leaf },
        retryIdentities: compactionRetryIdentities(event.branchEntries ?? [], { sessionFile: receiptSessionFile, leaf }),
        progress: progress.update, diagnostics: progress.snapshot });
      if (process.env.QWEN_RADIANCE_CACHE_ABI) {
        progress.update({ phase: "flush" });
        await flushSnapshotTail(payload.kv_transfer_params.qwen_chat);
        compaction.details.snapshotTailFlushed = true;
      }
      event.signal.throwIfAborted();
      if (ctx.sessionManager.getLeafId() !== leaf || ctx.sessionManager.getSessionFile() !== sessionFile) {
        throw new Error("session changed during compaction; original branch retained");
      }
      // Returning a validated checkpoint is not a committed conversation. Keep
      // the timer alive until Pi appends it and the cache hook finishes cleanup.
      progress.update({ phase: "commit" });
      return { compaction };
    } catch (error) {
      // Pi swallows exceptions from extension handlers. Return cancel explicitly
      // so an error never silently falls through to its expensive stock compactor.
      ctx.ui.notify(`Radiance compaction not committed: ${backendErrorMessage(error.message)}. Transcript retained; retry /compact.`, "error");
      await progress?.finish(event.signal.aborted ? "cancelled" : "failed");
      await reportBackendFailure(pi, ctx, error.message, attemptStarted);
      return { cancel: true };
    }
  });
}
