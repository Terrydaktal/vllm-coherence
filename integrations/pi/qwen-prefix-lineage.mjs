import { createHash, createHmac, randomBytes } from "node:crypto";
import { constants, readFileSync } from "node:fs";
import { mkdir, open, rename } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// Equality fingerprints never leave this process. In particular, the recorder
// must not persist hashes of short messages that an observer could guess.
const KEY = randomBytes(32);
const TRACE = randomBytes(16).toString("hex");
const PREVIOUS = new Map();
const MAX_CHATS = 32;
const MAX_MESSAGES = 1024;
const MAX_PENDING = 128;
const MAX_BYTES = 2 * 1024 * 1024;
const ROLES = new Set(["system", "developer", "user", "assistant", "tool", "toolResult"]);
const SCHEMA = "urn:coherence:pi-prefix-lineage:v1";
const SOURCE = (() => {
  try { return createHash("sha256").update(readFileSync(fileURLToPath(import.meta.url))).digest("hex"); }
  catch { return null; }
})();
const HEALTH = { enqueued: 0, written: 0, dropped: 0, errors: 0 };
const pending = [];
let writing = false;
let ordinal = 0;

function stable(value) {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort()
    .filter((key) => value[key] !== undefined).map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}`;
  return JSON.stringify(value ?? null);
}
function fingerprint(value) { return createHmac("sha256", KEY).update(stable(value)).digest("hex"); }
function bytes(value) { return typeof value === "string" ? Buffer.byteLength(value) : 0; }
function identity(value) { return typeof value === "string" && /^[0-9a-f]{64}$/.test(value) ? value : null; }

function semantic(message, source) {
  const role = ROLES.has(message?.role) ? message.role : "unknown";
  let content = message?.content;
  let text = typeof content === "string" ? content : "";
  let reasoning = "";
  let thinkingBlocks = 0;
  let whitespaceBlocks = 0;
  let unsupported = role === "unknown";
  let tools = [];
  if (Array.isArray(content)) {
    const textParts = [], thinkingParts = [];
    for (const block of content) {
      if (block?.type === "text") { textParts.push(block.text ?? ""); if (!block.text?.trim()) whitespaceBlocks++; }
      else if (block?.type === "thinking") { thinkingParts.push(block.thinking ?? ""); thinkingBlocks++; if (!block.thinking?.trim()) whitespaceBlocks++; }
      else if (block?.type === "toolCall") tools.push({ id: block.id, name: block.name, arguments: block.arguments });
      else unsupported = true;
    }
    text = textParts.join("");
    // Do not reproduce the provider's inserted separators here: detecting that
    // transformation is the point of comparing the two boundaries.
    reasoning = thinkingParts.join("");
  }
  if (!source) {
    for (const name of ["reasoning_content", "reasoning", "reasoning_text", "thinking"]) {
      if (typeof message?.[name] === "string") { reasoning = message[name]; break; }
    }
    if (Array.isArray(message?.tool_calls)) tools = message.tool_calls.map((call) => {
      const fn = call.function ?? call.custom ?? {};
      let args = fn.arguments ?? fn.input;
      if (typeof args === "string") { try { args = JSON.parse(args); } catch { unsupported = true; } }
      return { id: call.id, name: fn.name, arguments: args };
    });
  }
  const projection = { role: role === "toolResult" ? "tool" : role,
    text, reasoning, tools, tool_call_id: message?.toolCallId ?? message?.tool_call_id ?? null };
  return { digest: fingerprint(projection), textDigest: fingerprint(text), reasoningDigest: fingerprint(reasoning),
    toolsDigest: fingerprint(tools), role: projection.role, textBytes: bytes(text), reasoningBytes: bytes(reasoning),
    thinkingBlocks, whitespaceBlocks, unsupported };
}

function snapshot(messages, source) {
  if (!Array.isArray(messages)) return { rows: [], total: 0, truncated: false, unsupported: true };
  const rows = messages.slice(0, MAX_MESSAGES).map((message) => semantic(message, source));
  return { rows, total: messages.length, truncated: messages.length > MAX_MESSAGES,
    unsupported: rows.some((row) => row.unsupported) };
}
function sourceSnapshot(context) {
  const messages = Array.isArray(context?.messages) ? context.messages : [];
  return snapshot(typeof context?.systemPrompt === "string" && context.systemPrompt
    ? [{ role: "system", content: context.systemPrompt }, ...messages] : messages, true);
}
function compare(before, after) {
  const limit = Math.min(before.rows.length, after.rows.length);
  let common = 0;
  while (common < limit && before.rows[common].digest === after.rows[common].digest) common++;
  const equal = common === before.total && common === after.total && !before.truncated && !after.truncated;
  const left = before.rows[common], right = after.rows[common];
  const changed = common < limit || (!before.truncated && !after.truncated && before.total !== after.total);
  return { equal, common_messages: common, first_changed_message: changed ? common : null,
    before_messages: before.total, after_messages: after.total,
    first_change_text_equal: left && right ? left.textDigest === right.textDigest : null,
    first_change_reasoning_equal: left && right ? left.reasoningDigest === right.reasoningDigest : null,
    first_change_toolcalls_equal: left && right ? left.toolsDigest === right.toolsDigest : null,
    first_change_role_equal: left && right ? left.role === right.role : null,
    coverage_complete: !before.truncated && !after.truncated && !before.unsupported && !after.unsupported };
}
function config(payload) {
  return fingerprint({ model: payload?.model, template: payload?.chat_template_kwargs,
    tools: payload?.tools, tool_choice: payload?.tool_choice, parallel_tool_calls: payload?.parallel_tool_calls });
}

async function drain() {
  if (writing) return;
  writing = true;
  let directoryHandle;
  try {
    const directory = join(homedir(), ".local/state/qwen-r9700/diagnostics");
    await mkdir(directory, { recursive: true, mode: 0o700 });
    directoryHandle = await open(directory, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW);
    const metadata = await directoryHandle.stat();
    if (!metadata.isDirectory() || metadata.uid !== process.getuid()) throw new Error("unsafe diagnostic directory");
    // Older diagnostic recorders created this owned directory with mode 0755.
    // Privatize the validated inode, never a path that might become a symlink.
    if (metadata.mode & 0o077) await directoryHandle.chmod(0o700);
    // Keep all writes/rotation anchored to that validated directory even if its
    // pathname is replaced while the asynchronous recorder is draining.
    const path = `/proc/self/fd/${directoryHandle.fd}/pi-prefix-lineage-${process.pid}-${TRACE}.jsonl`;
    while (pending.length) {
      let file = await open(path, constants.O_WRONLY | constants.O_APPEND | constants.O_CREAT | constants.O_NOFOLLOW | constants.O_NONBLOCK, 0o600);
      try {
        const details = await file.stat();
        if (!details.isFile() || details.nlink !== 1 || details.uid !== process.getuid() || details.mode & 0o077) throw new Error("unsafe diagnostic file");
        if (details.size >= MAX_BYTES) {
          await file.close(); file = null;
          await rename(path, `${path}.1`);
          continue;
        }
        const batch = pending.splice(0, 16);
        await file.writeFile(batch.map((record) => `${JSON.stringify(record)}\n`).join(""));
        HEALTH.written += batch.length;
      } finally { await file?.close(); }
    }
  } catch { HEALTH.errors++; HEALTH.dropped += pending.length; pending.length = 0; }
  finally {
    try { await directoryHandle?.close(); } catch { HEALTH.errors++; }
    writing = false;
  }
}
function emitDefault(record) {
  if (pending.length >= MAX_PENDING) { HEALTH.dropped++; return; }
  HEALTH.enqueued++; pending.push(record); void drain();
}

export function prefixLineageHealth() { return { ...HEALTH, pending: pending.length, chats: PREVIOUS.size }; }

export function createPrefixLineage({ context, sessionId, fetch: customFetch, emit = emitDefault, now = () => Date.now() } = {}) {
  const requestOrdinal = ++ordinal;
  let source;
  try { source = sourceSnapshot(context); }
  catch { source = { rows: [], total: 0, truncated: false, unsupported: true }; }
  let key = typeof sessionId === "string" && sessionId ? fingerprint(sessionId) : null;
  let chat = null, generation = null, converted, wire, configIdentity, wireConfigIdentity, responseId = null, rawResponseId = null;
  let mask = 1, completed = false;
  let incomplete = source.truncated || source.unsupported;
  const write = (stage, values = {}) => {
    try { emit({ schema: SCHEMA, stage, timestamp_ms: now(), pid: process.pid, trace_id: TRACE,
      ordinal: requestOrdinal, source_id: SOURCE, chat_id: chat, generation,
      request_id_sha256: responseId, hook_mask: mask, recorder: prefixLineageHealth(), ...values }); }
    catch { HEALTH.errors++; }
  };
  write("context_before_conversion", { messages: source.total, truncated: source.truncated,
    unsupported: source.unsupported, whitespace_only_blocks: source.rows.reduce((sum, row) => sum + row.whitespaceBlocks, 0),
    thinking_blocks: source.rows.reduce((sum, row) => sum + row.thinkingBlocks, 0) });
  return {
    converted(payload) {
      try {
        converted = snapshot(payload?.messages, false); configIdentity = config(payload); mask |= 2;
        const comparison = compare(source, converted);
        incomplete ||= !comparison.coverage_complete;
        write("provider_converted", comparison);
      } catch { incomplete = true; write("provider_converted", { coverage_complete: false, observer_failed: true }); }
    },
    wire(payload) {
      try {
        chat = identity(payload?.kv_transfer_params?.qwen_chat?.id);
        generation = identity(payload?.kv_transfer_params?.qwen_chat?.generation);
        if (chat) key = fingerprint([chat, generation]);
        wire = snapshot(payload?.messages, false); mask |= 4;
        wireConfigIdentity = config(payload);
        incomplete ||= wire.truncated || wire.unsupported || !converted;
        const previous = key ? PREVIOUS.get(key) : undefined;
        const history = previous ? compare(previous.source, { ...source,
          total: Math.min(source.total, previous.source.total), rows: source.rows.slice(0, previous.source.total) }) : null;
        const nextAssistant = previous ? source.rows[previous.source.total] : null;
        write("provider_sdk_input", { ...(converted ? compare(converted, wire) : { coverage_complete: false }),
          config_changed: configIdentity === undefined ? null : configIdentity !== config(payload),
          previous_available: Boolean(previous), history_equal: history?.equal ?? null,
          history_first_changed_message: history?.first_changed_message ?? null,
          previous_output_equal: previous?.output ? nextAssistant?.digest === previous.output.digest : null,
          previous_output_text_equal: previous?.output && nextAssistant ? nextAssistant.textDigest === previous.output.textDigest : null,
          previous_output_reasoning_equal: previous?.output && nextAssistant ? nextAssistant.reasoningDigest === previous.output.reasoningDigest : null });
      } catch { incomplete = true; write("provider_sdk_input", { coverage_complete: false, observer_failed: true }); }
    },
    fetch(input, init) {
      try {
        // Read only the SDK's existing serialized string. Do not clone a
        // Request, consume stream bodies, or change arguments forwarded below.
        if (typeof init?.body !== "string" || Buffer.byteLength(init.body) > 8 * 1024 * 1024) {
          incomplete = true;
          write("provider_wire", { coverage_complete: false, body_unavailable: true });
        } else {
          const serialized = JSON.parse(init.body);
          const actual = snapshot(serialized?.messages, false);
          mask |= 32;
          const comparison = wire ? compare(wire, actual) : { coverage_complete: false };
          incomplete ||= !comparison.coverage_complete;
          write("provider_wire", { ...comparison,
            serialization_config_changed: wireConfigIdentity === undefined ? null : config(serialized) !== wireConfigIdentity });
        }
      } catch { incomplete = true; write("provider_wire", { coverage_complete: false, observer_failed: true }); }
      return (customFetch ?? globalThis.fetch)(input, init);
    },
    responseId(value) {
      if (typeof value !== "string" || !value || value.length > 512) return;
      if (value === rawResponseId) return;
      rawResponseId = value;
      const next = createHash("sha256").update(value).digest("hex");
      if (responseId === next) return;
      const changed = responseId !== null;
      responseId = next; mask |= 8;
      write("provider_response_identity", { response_id_changed: changed });
    },
    assembled(output, success = true) {
      if (completed) return;
      completed = true;
      try {
        const assembled = semantic(output, true); mask |= 16;
        write(success ? "provider_response_assembled" : "provider_response_failed", {
          success, text_bytes: assembled.textBytes, reasoning_bytes: assembled.reasoningBytes,
          thinking_blocks: assembled.thinkingBlocks,
          coverage_complete: mask === 63 && !incomplete && !assembled.unsupported && HEALTH.errors === 0 && HEALTH.dropped === 0 });
        if (success && key && wire) {
          PREVIOUS.delete(key);
          PREVIOUS.set(key, { source, output: assembled });
          if (PREVIOUS.size > MAX_CHATS) PREVIOUS.delete(PREVIOUS.keys().next().value);
        }
      } catch { write("provider_response_failed", { coverage_complete: false, observer_failed: true }); }
    },
  };
}
