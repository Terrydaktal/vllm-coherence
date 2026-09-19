import test from "node:test";
import assert from "node:assert/strict";
import install, { recoveryKind, SESSION_ID, PROJECT, CONTROL, RADIANCE_MODEL } from "../integrations/pi/qwen-searchtool-recovery.mjs";

const model = { provider: "qwen-r9700", api: "openai-completions", id: "fixture" };
const answer = (text, extra = {}) => ({ role: "assistant", ...model, model: model.id,
  content: [{ type: "thinking", thinking: "Reasoning retained." }, { type: "text", text }],
  stopReason: "stop", ...extra });
function fixture(overrides = {}, entries = []) {
  const handlers = new Map(), messages = [], notices = [], statuses = [];
  const ctx = { cwd: PROJECT, model,
    sessionManager: { getSessionId: () => SESSION_ID, getCwd: () => PROJECT, getBranch: () => entries },
    signal: new AbortController().signal, isIdle: () => false, hasPendingMessages: () => false,
    ui: { setStatus: (...args) => statuses.push(args), notify: (...args) => notices.push(args) }, ...overrides };
  install({ on: (name, fn) => handlers.set(name, fn), sendMessage: (...args) => messages.push(args) });
  const emit = (name, event = {}) => handlers.get(name)?.(event, ctx);
  emit("session_start");
  return { ctx, emit, messages, notices, statuses };
}

test("automatic follow-up preserves the stopped answer and never injects partial arguments", () => {
  const f = fixture();
  const message = answer("Let me read the file:");
  const before = structuredClone(message);
  assert.equal(f.emit("message_end", { message }), undefined);
  f.emit("agent_end");
  assert.deepEqual(message, before);
  assert.equal(f.messages.length, 1);
  assert.equal(f.messages[0][0].customType, CONTROL);
  assert.equal(f.messages[0][0].display, false);
  assert.deepEqual(f.messages[0][1], { deliverAs: "followUp", triggerTurn: true });
});

test("two recoveries per real user message; repeated callbacks and success do not refill budget", () => {
  const f = fixture();
  for (let n = 0; n < 5; n++) {
    f.emit("message_end", { message: answer("I'll run the tests:") });
    f.emit("agent_end");
    f.emit("agent_end");
    f.emit("agent_settled");
  }
  assert.equal(f.messages.length, 2);
  assert.ok(f.notices.length);
  f.emit("message_start", { message: { role: "user", content: "Try another task" } });
  f.emit("message_end", { message: answer("I'll inspect the file.") });
  f.emit("agent_end");
  assert.equal(f.messages.length, 3);
  assert.equal(f.messages[2][0].details.attempt, 1);
});

test("reload preserves the budget from durable custom messages", () => {
  const f = fixture({}, [{ type: "message", message: { role: "user" } },
    { type: "custom_message", customType: CONTROL }, { type: "custom_message", customType: CONTROL }]);
  f.emit("message_end", { message: answer("I'll inspect the file.") });
  f.emit("agent_end");
  assert.equal(f.messages.length, 0);
});

test("protocol failure bypasses destructive stock retry while retaining text", () => {
  const f = fixture();
  const message = answer("Earlier result", { stopReason: "error", errorMessage: "Stream ended without finish_reason" });
  message.content.push({ type: "toolCall", id: "UNTRUSTED_PARTIAL_ARGS", name: "read", arguments: {} });
  const result = f.emit("message_end", { message });
  assert.deepEqual(result.message.content, message.content.slice(0, 2));
  assert.equal(result.message.stopReason, "error");
  assert.doesNotMatch(result.message.errorMessage, /server.?error|ended without|500|retry delay/i);
  f.emit("agent_end");
  assert.equal(f.messages.length, 1);
  assert.doesNotMatch(f.messages[0][0].content, /Earlier result|UNTRUSTED_PARTIAL_ARGS/);
});

test("completed calls are blocked by arguments even if the retry uses a new ID", () => {
  const f = fixture();
  f.emit("tool_result", { toolName: "read", toolCallId: "old", input: { path: "a", limit: 2 }, isError: false });
  f.emit("message_end", { message: answer("I'll inspect the file:") });
  f.emit("agent_end");
  const calls = [{ type: "toolCall", id: "new-read", name: "read", arguments: { limit: 2, path: "a" } },
    { type: "toolCall", id: "new-task", name: "read", arguments: { path: "b" } }];
  f.emit("message_end", { message: answer("", { stopReason: "toolUse", content: calls }) });
  assert.equal(f.emit("tool_call", { toolCallId: "new-read" }).block, true);
  assert.equal(f.emit("tool_call", { toolCallId: "new-task" }), undefined);
  f.emit("tool_result", { toolName: "read", input: { path: "b" }, isError: false });
  assert.equal(f.emit("tool_call", { toolCallId: "new-read" }).block, true);
});

for (const [label, overrides] of [
  ["different chat", { sessionManager: { getSessionId: () => "other", getCwd: () => PROJECT, getBranch: () => [] } }],
  ["different working directory", { cwd: "/home/lewis/tasks/qwen" }],
  ["different provider", { model: { ...model, provider: "other" } }],
  ["cancelled request", { signal: AbortSignal.abort() }],
  ["queued user input", { hasPendingMessages: () => true }],
]) test(`no recovery for ${label}`, () => {
  const f = fixture(overrides);
  f.emit("message_end", { message: answer("I'll read the file:") });
  f.emit("agent_end");
  assert.equal(f.messages.length, 0);
});

test("steered input wins even before it reaches the agent queue", () => {
  const f = fixture();
  f.emit("input", { source: "interactive" });
  f.emit("message_end", { message: answer("I'll run the test:") });
  f.emit("agent_end");
  assert.equal(f.messages.length, 0);
});

for (const text of ["Done.", "Would you like me to run it?", "I'll run it if you approve.",
  'The example says: "Let me run the test:"',
  "```text\nLet me run the test:\n```", "Let me know if you need anything else."]) {
  test(`ordinary answer/quotation is not auto-continued: ${JSON.stringify(text)}`, () => {
    assert.equal(recoveryKind(answer(text)), undefined);
  });
}

test("abort, length, network and tool errors are not retried by this extension", () => {
  for (const stopReason of ["aborted", "length", "error"]) {
    assert.equal(recoveryKind(answer("I'll read the file:", { stopReason, errorMessage: "network error" })), undefined);
  }
  assert.equal(recoveryKind({ role: "toolResult", isError: true }), undefined);
});

test("reasoning without any final outcome receives bounded continuation", () => {
  assert.equal(recoveryKind(answer("")), "missing-outcome");
  assert.equal(recoveryKind(answer("", { content: [] })), undefined);
});

test("an announcement after introductory sentences is detected", () => {
  assert.equal(recoveryKind(answer("Resuming. The previous check is complete. Let me inspect the next file:")), "announcement");
});

test("money chat receives authorized recovery without a persistent readiness status", () => {
  const moneyProject = "/home/lewis/tasks/money";
  const moneySessionId = "01a067d9-69cf-761f-8503-6554ccfd7703";
  const f = fixture({
    cwd: moneyProject,
    sessionManager: { getSessionId: () => moneySessionId, getCwd: () => moneyProject, getBranch: () => [] },
  });
  assert.equal(f.statuses.at(-1)?.[0], "money-recovery");
  assert.equal(f.statuses.at(-1)?.[1], undefined);

  const message = answer("Earlier result", { stopReason: "error", errorMessage: "Stream ended without finish_reason" });
  message.content.push({ type: "toolCall", id: "UNTRUSTED", name: "bash", arguments: {} });
  const result = f.emit("message_end", { message });
  assert.equal(result.message.rawStopReason, "money_incomplete_output");
  assert.match(result.message.errorMessage, /Money structured-output failure/);

  f.emit("agent_end");
  assert.equal(f.messages.length, 1);
  assert.match(f.messages[0][0].content, /Continue the user's already-authorized money task/);
  assert.equal(f.messages[0][0].details.sessionId, moneySessionId);
});

test("moneychat chat receives authorized recovery without a persistent readiness status", () => {
  const moneychatProject = "/home/lewis/tasks/moneychat";
  const moneychatSessionId = "01a07934-a41f-7a80-8fb3-3b0e0b5d3ab5";
  const f = fixture({
    cwd: moneychatProject,
    sessionManager: { getSessionId: () => moneychatSessionId, getCwd: () => moneychatProject, getBranch: () => [] },
  });
  assert.equal(f.statuses.at(-1)?.[0], "moneychat-recovery");
  assert.equal(f.statuses.at(-1)?.[1], undefined);

  const message = answer("Let me check the balance:");
  f.emit("message_end", { message });
  f.emit("agent_end");
  assert.equal(f.messages.length, 1);
  assert.match(f.messages[0][0].content, /Continue the user's already-authorized moneychat task/);
  assert.equal(f.messages[0][0].details.sessionId, moneychatSessionId);
});

test("cross-directory or unauthorized session combinations reject recovery", () => {
  const moneyProject = "/home/lewis/tasks/money";
  const f1 = fixture({
    cwd: "/home/lewis/tasks/searchtool",
    sessionManager: { getSessionId: () => "01a067d9-69cf-761f-8503-6554ccfd7703", getCwd: () => moneyProject, getBranch: () => [] },
  });
  f1.emit("message_end", { message: answer("Let me check:") });
  f1.emit("agent_end");
  assert.equal(f1.messages.length, 0);

  const f2 = fixture({
    cwd: moneyProject,
    sessionManager: { getSessionId: () => "unauthorized-session-id", getCwd: () => moneyProject, getBranch: () => [] },
  });
  f2.emit("message_end", { message: answer("Let me check:") });
  f2.emit("agent_end");
  assert.equal(f2.messages.length, 0);
});

const radianceModel = { ...model, id: RADIANCE_MODEL };
const radianceAnswer = (text, extra = {}) => answer(text, { model: RADIANCE_MODEL, ...extra });
function radianceFixture(overrides = {}, entries = []) {
  return fixture({
    cwd: "/synthetic/new-radiance-project",
    model: radianceModel,
    sessionManager: { getSessionId: () => "new-radiance-session", getCwd: () => "/synthetic/new-radiance-project", getBranch: () => entries },
    ...overrides,
  });
}

test("any Radiance chat continues only after the complete run ends, without posing as a user", () => {
  const f = radianceFixture();
  const message = radianceAnswer("Results: \n\t");
  const before = structuredClone(message);
  f.emit("message_update", { assistantMessageEvent: { type: "text_delta", delta: ":", partial: message } });
  assert.equal(f.messages.length, 0, "streaming punctuation cannot trigger recovery");
  f.emit("message_end", { message });
  assert.equal(f.messages.length, 0, "a message end alone is not an idle run");
  f.emit("agent_end", { messages: [message] });
  f.emit("agent_end", { messages: [message] });
  assert.deepEqual(message, before);
  assert.equal(f.messages.length, 1);
  const [control, options] = f.messages[0];
  assert.equal(control.customType, CONTROL);
  assert.equal(control.display, false);
  assert.equal(control.role, undefined);
  assert.match(control.content, /already-authorized task/);
  assert.match(control.content, /without another announcement/);
  assert.equal(control.details.reason, "announcement");
  assert.equal(control.details.sessionId, "new-radiance-session");
  assert.deepEqual(options, { deliverAs: "followUp", triggerTurn: true });
  assert.ok(f.statuses.every(([, value]) => value === undefined), "recovery never pins footer text");
});

for (const text of [
  "Checking the file:", "Now running the tests:", "I'll run `git status`:",
  "I'll read the file `src/main.py`.", "I'm checking the file:", "I’m going to inspect the file:",
  "Next, I'll apply the patch:", "**Let me inspect the file:**", "I'll patch the file:",
  "The first check passed.\nLet me check the next file:",
]) test(`recognizes dangling action: ${JSON.stringify(text)}`, () => {
  assert.equal(recoveryKind(radianceAnswer(text)), "announcement");
});

for (const text of [
  ":", "Results:", "Results: \n\t", "The syntax uses a colon:", "Run this command:", "Here is the code:",
  "```python\nif ready:", "```text\nLet me run the test:",
  "> I'll inspect the file:", "Example:\n\n> I'll inspect the file:",
  "    I'll inspect the file:", "Example:\n\n    I'll inspect the file:",
  "I'm waiting for your permission.\n\nI'll run it after you approve:",
  "Should I proceed? I'll run the tests:", "If you want, I'll run the tests:",
  "Let me read `unterminated command:", "# Checking the file:",
]) test(`a final colon is sufficient regardless of wording: ${JSON.stringify(text)}`, () => {
  const f = radianceFixture();
  const message = radianceAnswer(text);
  f.emit("message_end", { message });
  f.emit("agent_end", { messages: [message] });
  assert.equal(f.messages.length, 1);
  assert.match(f.messages[0][0].content, /Do not perform anything awaiting user approval/);
});

for (const text of [
  "For example, `if ready:`", "```python\nif ready:\n```",
  "\"I'll run the tests:\"", "“I'll run the tests:”", "'Let me inspect the file:'",
  "`Let me inspect the file:`", "I will use the phrase `Let me inspect the file:`",
  "> I'll inspect the file.", "    I'll inspect the file.",
  "I'll run it after you approve.", "Should I proceed? I'll run the tests.",
  "If you want, I'll run the tests.", "Let me read `unterminated command.", "# Checking the file.",
]) test(`non-colon responses retain conservative announcement detection: ${JSON.stringify(text)}`, () => {
  const f = radianceFixture();
  const message = radianceAnswer(text);
  f.emit("message_end", { message });
  f.emit("agent_end", { messages: [message] });
  assert.equal(f.messages.length, 0);
});

test("the colon rule also applies to explicit final text, but never to a message containing a tool call", () => {
  assert.equal(recoveryKind(radianceAnswer("Results:", { rawStopReason: "qwen_json_outcome_final" })), "announcement");
  assert.equal(recoveryKind(radianceAnswer("Let me check the file.", { rawStopReason: "qwen_json_outcome_final" })), undefined);
  assert.equal(recoveryKind(radianceAnswer("Results:", { content: [
    { type: "text", text: "Results:" },
    { type: "toolCall", id: "one", name: "read", arguments: { path: "fixture" } },
  ] })), undefined);
});

test("a real tool call or later completed answer supersedes an announcement", () => {
  const f = radianceFixture();
  const announcement = radianceAnswer("Let me read the file:");
  const complete = radianceAnswer("Finished.");
  f.emit("message_end", { message: announcement });
  f.emit("agent_end", { messages: [announcement, complete] });
  assert.equal(f.messages.length, 0);
  const tool = radianceAnswer("Let me read the file:", { stopReason: "toolUse", content: [
    { type: "text", text: "Let me read the file:" },
    { type: "toolCall", id: "one", name: "read", arguments: { path: "fixture" } },
  ] });
  f.emit("message_end", { message: announcement });
  f.emit("message_start", { message: tool });
  f.emit("message_end", { message: tool });
  f.emit("agent_end", { messages: [announcement, tool] });
  assert.equal(f.messages.length, 0);
});

test("Radiance announcement recovery never retries cancellation, tool, protocol, or length errors", () => {
  for (const stopReason of ["aborted", "error", "length"]) {
    const f = radianceFixture();
    const message = radianceAnswer("Let me check the file:", { stopReason, errorMessage: "Stream ended without finish_reason" });
    f.emit("message_end", { message });
    f.emit("agent_end", { messages: [message] });
    assert.equal(f.messages.length, 0);
  }
  const f = radianceFixture();
  const message = radianceAnswer("", { content: [{ type: "thinking", thinking: "Synthetic reasoning." }] });
  f.emit("message_end", { message });
  f.emit("agent_end", { messages: [message] });
  assert.equal(f.messages.length, 0, "broader protocol/missing-outcome recovery remains in its existing scope");
});

test("new Radiance chats honor the limit, reload, cancellation and real user input", () => {
  for (const overrides of [
    { signal: AbortSignal.abort() }, { hasPendingMessages: () => true },
    { model: { ...radianceModel, provider: "other" } },
  ]) {
    const f = radianceFixture(overrides);
    f.emit("message_end", { message: radianceAnswer("Checking the file:") });
    f.emit("agent_end");
    assert.equal(f.messages.length, 0);
  }
  const f = radianceFixture({}, [{ type: "message", message: { role: "user" } },
    { type: "custom_message", customType: CONTROL }, { type: "custom_message", customType: CONTROL }]);
  f.emit("message_end", { message: radianceAnswer("Checking the file:") });
  f.emit("agent_end");
  assert.equal(f.messages.length, 0);
  assert.equal(f.notices.length, 1);
  f.emit("message_start", { message: { role: "user" } });
  f.emit("message_end", { message: radianceAnswer("Checking the file:") });
  f.emit("agent_end");
  assert.equal(f.messages.length, 1);
  assert.equal(f.messages[0][0].details.attempt, 1);
  f.emit("input", { source: "interactive" });
  f.emit("message_end", { message: radianceAnswer("Checking the file:") });
  f.emit("agent_end");
  assert.equal(f.messages.length, 1);
});
