import assert from "node:assert/strict";
import test from "node:test";
import { createPrefixLineage, prefixLineageHealth } from "../integrations/pi/qwen-prefix-lineage.mjs";

const user = (text) => ({ role: "user", content: text });
const answer = (text, thinking = []) => ({ role: "assistant", content: [
  ...thinking.map((value) => ({ type: "thinking", thinking: value, thinkingSignature: "reasoning_content" })),
  { type: "text", text },
] });
const chat = (letter, generation = "b") => ({ id: letter.repeat(64), generation: generation.repeat(64) });
const payload = (messages, qwenChat, kwargs = { preserve_thinking: true }) => ({
  model: "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate", messages,
  chat_template_kwargs: kwargs, kv_transfer_params: { qwen_chat: qwenChat },
});
function recorder(context, sessionId = "test-session") {
  const records = [];
  return { records, observer: createPrefixLineage({ context, sessionId, fetch: () => null, emit: (record) => records.push(record), now: () => 123 }) };
}
function finish(recorder, wire, output) {
  recorder.observer.converted(wire);
  recorder.observer.wire(wire);
  recorder.observer.fetch("http://127.0.0.1/v1/chat/completions", { body: JSON.stringify(wire) });
  recorder.observer.responseId("chatcmpl-test-random-id");
  recorder.observer.assembled(output);
}

test("every real provider boundary contributes to complete coverage", () => {
  const r = recorder({ messages: [user("private-input")] });
  finish(r, payload([user("private-input")], chat("a")), answer("private-output"));
  assert.deepEqual(r.records.map((record) => record.stage), ["context_before_conversion", "provider_converted",
    "provider_sdk_input", "provider_wire", "provider_response_identity", "provider_response_assembled"]);
  assert.equal(r.records[1].equal, true);
  assert.equal(r.records[2].equal, true);
  assert.equal(r.records.at(-1).hook_mask, 63);
  assert.equal(r.records.at(-1).coverage_complete, true);
  const serialized = JSON.stringify(r.records);
  assert.equal(serialized.includes("private-input"), false);
  assert.equal(serialized.includes("private-output"), false);
  assert.equal(serialized.includes("digest"), false);
  assert.equal(serialized.includes("token_ids"), false);
});

test("payload hooks are compared at the final wire boundary", () => {
  const r = recorder({ messages: [user("original")] });
  const before = payload([user("original")], chat("c"));
  r.observer.converted(before);
  r.observer.wire(payload([user("changed-by-payload-hook")], chat("c"), { preserve_thinking: false }));
  const record = r.records.at(-1);
  assert.equal(record.equal, false);
  assert.equal(record.first_changed_message, 0);
  assert.equal(record.config_changed, true);
});

test("SDK serialization is measured separately and forwarded without modification", () => {
  const records = [], calls = [];
  const observer = createPrefixLineage({ context: { messages: [user("before")] }, emit: (record) => records.push(record),
    fetch: (...args) => { calls.push(args); return "existing-fetch-result"; } });
  const expected = payload([user("before")], chat("5"));
  observer.converted(expected); observer.wire(expected);
  const options = { body: JSON.stringify(payload([user("after-sdk")], chat("5"))), method: "POST" };
  const endpoint = new URL("http://127.0.0.1/v1/chat/completions");
  assert.equal(observer.fetch(endpoint, options), "existing-fetch-result");
  assert.equal(calls[0][0], endpoint); assert.equal(calls[0][1], options);
  assert.equal(records.at(-1).stage, "provider_wire");
  assert.equal(records.at(-1).equal, false);
  assert.equal(records.at(-1).first_changed_message, 0);
});

test("provider inserted thinking separator is visible as a conversion change", () => {
  const r = recorder({ messages: [answer("answer", ["first", "second"])] });
  r.observer.converted(payload([{ role: "assistant", content: "answer", reasoning_content: "first\nsecond" }], chat("d")));
  assert.equal(r.records.at(-1).equal, false);
  assert.equal(r.records.at(-1).first_changed_message, 0);
  assert.equal(r.records[0].thinking_blocks, 2);
});

test("dropping a whitespace-only assistant block is detected", () => {
  const r = recorder({ messages: [{ role: "assistant", content: [
    { type: "text", text: "left" }, { type: "text", text: " " }, { type: "text", text: "right" },
  ] }] });
  r.observer.converted(payload([{ role: "assistant", content: "leftright" }], chat("e")));
  assert.equal(r.records[0].whitespace_only_blocks, 1);
  assert.equal(r.records.at(-1).equal, false);
});

test("previous assembled output and unchanged history are checked on continuation", () => {
  const qwenChat = chat("f");
  const output = answer("response", ["thought"]);
  finish(recorder({ messages: [user("question")] }), payload([user("question")], qwenChat), output);
  const next = recorder({ messages: [user("question"), output, user("next")] });
  const wire = payload([user("question"), { role: "assistant", content: "response", reasoning_content: "thought" }, user("next")], qwenChat);
  next.observer.converted(wire); next.observer.wire(wire);
  assert.equal(next.records.at(-1).previous_available, true);
  assert.equal(next.records.at(-1).history_equal, true);
  assert.equal(next.records.at(-1).previous_output_equal, true);
  const changed = recorder({ messages: [user("question"), answer("response changed", ["thought"]), user("next")] });
  changed.observer.converted(wire); changed.observer.wire(wire);
  assert.equal(changed.records.at(-1).previous_output_equal, false);
});

test("history changes and cache generation changes remain separate", () => {
  const qwenChat = chat("1");
  finish(recorder({ messages: [user("question")] }), payload([user("question")], qwenChat), answer("response"));
  const changed = recorder({ messages: [user("rewritten history"), answer("response"), user("next")] });
  const wire = payload([user("rewritten history"), { role: "assistant", content: "response" }, user("next")], qwenChat);
  changed.observer.converted(wire); changed.observer.wire(wire);
  assert.equal(changed.records.at(-1).history_equal, false);
  assert.equal(changed.records.at(-1).history_first_changed_message, 0);
  const fresh = recorder({ messages: [user("question"), answer("response"), user("next")] });
  const newGeneration = payload(wire.messages, chat("1", "c"));
  fresh.observer.converted(newGeneration); fresh.observer.wire(newGeneration);
  assert.equal(fresh.records.at(-1).previous_available, false);
});

test("missing hooks, unsupported images and truncation cannot claim complete", () => {
  const omitted = recorder({ messages: [user("input")] });
  omitted.observer.assembled(answer("output"));
  assert.equal(omitted.records.at(-1).coverage_complete, false);
  const image = recorder({ messages: [{ role: "user", content: [{ type: "image", data: "private-image" }] }] });
  finish(image, payload([], chat("2")), answer("output"));
  assert.equal(image.records.at(-1).coverage_complete, false);
  const truncated = recorder({ messages: Array.from({ length: 1025 }, () => user("x")) });
  finish(truncated, payload(Array.from({ length: 1025 }, () => user("x")), chat("3")), answer("output"));
  assert.equal(truncated.records[0].truncated, true);
  assert.equal(truncated.records.at(-1).coverage_complete, false);
  assert.equal(truncated.records[1].first_changed_message, null);
});

test("unavailable or malformed serialized bodies stay explicitly incomplete", () => {
  for (const body of [null, "not JSON", "x".repeat(8 * 1024 * 1024 + 1)]) {
    const r = recorder({ messages: [user("input")] });
    const wire = payload([user("input")], chat("6"));
    r.observer.converted(wire); r.observer.wire(wire);
    assert.doesNotThrow(() => r.observer.fetch("http://127.0.0.1/v1/chat/completions", { body }));
    assert.equal(r.records.at(-1).coverage_complete, false);
    r.observer.responseId("id"); r.observer.assembled(answer("output"));
    assert.equal(r.records.at(-1).coverage_complete, false);
  }
});

test("diagnostic failures cannot abort or change provider output", () => {
  const output = answer("unchanged");
  const original = JSON.stringify(output);
  const observer = createPrefixLineage({ context: { messages: [] }, emit: () => { throw new Error("observer failed"); } });
  observer.converted(payload([], chat("4"))); observer.wire(payload([], chat("4")));
  observer.responseId("random-id"); observer.assembled(output); observer.assembled(output);
  assert.equal(JSON.stringify(output), original);
  const malformed = createPrefixLineage({ context: { messages: [{ role: "assistant", content: [{ type: "text", text: 123 }] }] }, emit: () => {} });
  assert.doesNotThrow(() => malformed.assembled(output, false));
});

test("per-process previous chat records are bounded and no output text is retained", () => {
  for (let i = 0; i < 40; i++) {
    const qwenChat = { id: i.toString(16).padStart(64, "0"), generation: "a".repeat(64) };
    finish(recorder({ messages: [user(`question-${i}`)] }), payload([user(`question-${i}`)], qwenChat), answer(`answer-${i}`));
  }
  assert.ok(prefixLineageHealth().chats <= 32);
});
