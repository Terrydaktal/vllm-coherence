import assert from "node:assert/strict";
import test from "node:test";
import { TOOL_NAME_ALIASES, activateCanonicalTools, canonicalToolName,
  canonicalizeToolMessages, legacyToolName, toolNamesMatch } from "../integrations/pi/qwen-tool-names.mjs";

test("the naming contract is bijective and leaves unrelated tools alone", () => {
  const names = Object.values(TOOL_NAME_ALIASES);
  assert.equal(names.length, 13);
  assert.equal(new Set(names).size, names.length);
  assert.equal(TOOL_NAME_ALIASES.session_search, "pi_session_search");
  for (const [old, current] of Object.entries(TOOL_NAME_ALIASES)) {
    assert.equal(canonicalToolName(old), current);
    assert.equal(canonicalToolName(current), current);
    assert.equal(toolNamesMatch(old, current), true);
  }
  assert.equal(canonicalToolName("user_tool"), "user_tool");
  assert.equal(toolNamesMatch("read", "write_file"), false);
});

test("the previous archive-reader name migrates to rehydrate_tool_result without changing stored history", () => {
  assert.equal(canonicalToolName("read_archived_tool_result"), "rehydrate_tool_result");
  assert.equal(toolNamesMatch("read_archived_tool_result", "qwen_rehydrate_tool_turn"), true);
  assert.equal(legacyToolName("read_archived_tool_result"), "qwen_rehydrate_tool_turn");
  const messages = [
    { role: "assistant", content: [{ type: "toolCall", id: "archive", name: "read_archived_tool_result", arguments: {} }] },
    { role: "toolResult", toolCallId: "archive", toolName: "read_archived_tool_result", content: [] },
  ];
  const original = structuredClone(messages);
  const mapped = canonicalizeToolMessages(messages);
  assert.equal(mapped[0].content[0].name, "rehydrate_tool_result");
  assert.equal(mapped[1].toolName, "rehydrate_tool_result");
  assert.deepEqual(messages, original);
  let active = ["read_archived_tool_result", "rehydrate_tool_result", "qwen_rehydrate_tool_turn"];
  activateCanonicalTools({ getAllTools: () => [{ name: "rehydrate_tool_result" }], getActiveTools: () => active,
    setActiveTools: (tools) => { active = tools; } });
  assert.deepEqual(active, ["rehydrate_tool_result"]);
});

test("activation migrates selected names without enabling inactive or missing tools", () => {
  let active = ["read", "read_file", "bash", "user_tool", "search"];
  let updates = 0;
  const pi = { getAllTools: () => ["read_file", "run_shell_command", "write_file", "user_tool", "search"].map((name) => ({ name })),
    getActiveTools: () => [...active], setActiveTools: (names) => { updates++; active = names; } };
  activateCanonicalTools(pi);
  assert.deepEqual(active, ["read_file", "run_shell_command", "user_tool", "search"]);
  activateCanonicalTools(pi);
  assert.equal(updates, 1);
});

test("excluded native aliases migrate from old defaults without enabling inactive current selections", () => {
  let active = ["pi_session_search"];
  const pi = { getAllTools: () => ["read_file", "run_shell_command", "write_file", "pi_session_search"].map(name => ({ name })),
    getActiveTools: () => active, setActiveTools: names => { active = names; } };
  activateCanonicalTools(pi, ["read", "bash", "write_file", "find", "qwen_plan"]);
  assert.deepEqual(active, ["pi_session_search", "read_file", "run_shell_command"]);
});

test("historical tool identities are mapped without changing source messages, arguments or archive hints", () => {
  const tokenIds = Symbol.for("qwen-r9700:raw-completion-token-ids:v1");
  const messages = [
    { role: "user", content: "Use the old read name as a historical quotation." },
    { role: "assistant", content: [
      { type: "thinking", thinking: "history" },
      { type: "toolCall", id: "old-call", name: "qwen_rehydrate_tool_turn", arguments: { sha256: "a".repeat(64) } },
      { type: "toolCall", id: "custom-call", name: "user_tool", arguments: {} },
    ], [tokenIds]: [1, 2] },
    { role: "toolResult", toolName: "qwen_rehydrate_tool_turn", toolCallId: "old-call",
      content: [{ type: "text", text: "Archived bytes: session_search and qwen_plan are quoted history." }] },
  ];
  const before = structuredClone(messages);
  const mapped = canonicalizeToolMessages(messages);
  assert.equal(mapped[0], messages[0]);
  assert.equal(mapped[1].content[1].name, "rehydrate_tool_result");
  assert.equal(mapped[2].toolName, "rehydrate_tool_result");
  assert.equal(mapped[1].content[1].arguments, messages[1].content[1].arguments);
  assert.equal(mapped[1][tokenIds], messages[1][tokenIds]);
  assert.equal(mapped[2].content, messages[2].content);
  assert.deepEqual(structuredClone(messages), before);
  assert.equal(canonicalizeToolMessages(mapped), mapped);
});
