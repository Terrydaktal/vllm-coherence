import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import {
  TASK_PLAN_ENTRY, TASK_PLAN_CONTEXT, applyTaskPlanAction, filterTaskPlanForContext,
  installTaskPlan, isTrustedPlanTool, planRelevantPaths, planToolGate, renderTaskPlan, replayTaskPlan,
  taskPlanContext, taskPlanState, validateReadOnlyBash,
} from "../integrations/pi/qwen-task-plan.mjs";
import { CONTEXT_POLICY_ENTRY } from "../integrations/pi/qwen-context-policy.mjs";

const user = { type: "message", id: "user-1", parentId: null, message: { role: "user", content: "Synthetic task", timestamp: 1 } };
const result = { type: "message", id: "tool-1", parentId: "user-1", message: { role: "toolResult", toolName: "read", toolCallId: "read-1", content: [{ type: "text", text: "Synthetic result" }], timestamp: 2 } };
const branch = [user, result];
const initial = () => replayTaskPlan([]);
const create = (changes = {}) => applyTaskPlanAction(initial(), { action: "create", goal: "Implement synthetic feature",
  relevantFiles: ["src/synthetic.mjs"], constraints: [{ text: "Preserve fixture data" }], notes: [{ text: "Synthetic only" }],
  steps: [{ title: "Inspect source", status: "in_progress" }, { title: "Implement change" }, { title: "Verify behavior" }], ...changes }, branch);
const entry = (state, id = "plan-1") => ({ type: "custom", customType: TASK_PLAN_ENTRY, id, data: state });

test("create assigns stable IDs, bounded source-linked state, and model-reported completion", () => {
  let state = create();
  assert.deepEqual(state.plan.steps.map((step) => step.id), ["s1", "s2", "s3"]);
  assert.deepEqual(state.plan.goalSourceEntryIds, [user.id]);
  assert.deepEqual(state.plan.relevantFileSources["src/synthetic.mjs"], [user.id]);
  assert.deepEqual(state.plan.constraints[0].sourceEntryIds, [user.id]);
  state = applyTaskPlanAction(state, { action: "update", steps: [
    { id: "s1", status: "completed", evidence: [{ entryId: result.id, note: "Read fixture source" }] },
    { id: "s2", status: "in_progress" },
  ] }, branch);
  assert.equal(state.plan.steps[0].title, "Inspect source");
  assert.match(renderTaskPlan(state), /s1 \[completed; model reported\]/);
  assert.match(renderTaskPlan(state), /Evidence \(model cited\): tool-1/);
  assert.match(renderTaskPlan(state), /source:user-1/);
});

test("revise preserves named IDs and never reuses deleted automatically assigned IDs", () => {
  const first = create();
  const second = applyTaskPlanAction(first, { action: "revise", steps: [{ id: "s1", status: "pending" }, { id: "s3" }] }, branch);
  const third = applyTaskPlanAction(second, { action: "revise", steps: [{ id: "s1" }, { title: "New subtask" }] }, branch);
  assert.deepEqual(third.plan.steps.map((step) => step.id), ["s1", "s4"]);
  assert.equal(third.plan.steps[0].title, first.plan.steps[0].title);
  assert.equal(first.plan.steps.length, 3, "actions must not mutate their input");
});

test("schema validation rejects unsupported actions, accidental replacement and unknown evidence", () => {
  const state = create();
  const bad = [
    { action: "execute" }, { action: "create", goal: "Overwrite" },
    { action: "update", steps: [{ id: "missing", status: "completed" }] },
    { action: "update", steps: [{ id: "s1", status: "completed", evidence: [{ entryId: "sibling-evidence", note: "Unrelated" }] }] },
    { action: "update", steps: [{ id: "s2", status: "in_progress" }] },
    { action: "update", constraints: [{ text: "Bad source", sourceEntryIds: ["missing"] }] },
    { action: "update", relevantFileSources: { "not-in-plan": [] } },
    { action: "revise", steps: [{ id: "s1" }, { id: "s1" }] },
    { action: "revise", steps: [] }, { action: "update", mode: "execute" },
    { action: "show", goal: "Mutate" }, { action: "show", maxChars: -1 },
  ];
  for (const params of bad) assert.throws(() => applyTaskPlanAction(state, params, branch), undefined, JSON.stringify(params));
  assert.throws(() => create({ steps: [{ title: "x".repeat(281) }] }), /280/);
  assert.throws(() => create({ steps: Array.from({ length: 31 }, () => ({ title: "Too many" })) }), /30/);
  assert.throws(() => create({ goalSourceEntryIds: [user.id, user.id] }), /Duplicate/);
});

test("selected-branch replay ignores siblings, retains compaction ancestors and persists clear/mode", () => {
  const state = { ...create(), mode: "plan" };
  const ancestor = entry(state);
  const compacted = [...branch, ancestor, { type: "compaction", id: "compact-1", summary: "Synthetic summary" }];
  assert.deepEqual(replayTaskPlan(compacted), state);
  const sibling = applyTaskPlanAction(state, { action: "update", goal: "Sibling goal" }, branch);
  assert.equal(replayTaskPlan([...branch, entry(sibling)]).plan.goal, "Sibling goal");
  assert.equal(replayTaskPlan(compacted).plan.goal, "Implement synthetic feature");
  const cleared = applyTaskPlanAction(state, { action: "clear" }, branch);
  assert.equal(cleared.mode, "plan");
  assert.equal(replayTaskPlan([...compacted, entry(cleared, "plan-2")]).plan, null);
  assert.deepEqual(replayTaskPlan([]), initial(), "fresh sessions do not inherit a process-global plan");
  assert.throws(() => replayTaskPlan([...compacted, entry({ version: 99, mode: "execute", plan: null })]), /Unsupported/);
});

test("context exclusions suppress linked content without changing persistent state", () => {
  const state = create({ steps: [{ title: "Read exact source", sourceEntryIds: [result.id], relevantFiles: ["src/other.mjs"],
    status: "completed", evidence: [{ entryId: result.id, note: "Observed synthetic source" }] }] });
  const filtered = filterTaskPlanForContext(state, new Set([user.id]));
  assert.equal(filtered.plan.goal, "[Goal withheld by /context]");
  assert.deepEqual(filtered.plan.constraints, []);
  assert.deepEqual(filtered.plan.notes, []);
  assert.deepEqual(filtered.plan.relevantFiles, []);
  assert.equal(filtered.plan.steps.length, 1);
  assert.equal(state.plan.goal, "Implement synthetic feature");
  assert.deepEqual(filterTaskPlanForContext(state, new Set([result.id])).plan.steps, []);
  assert.deepEqual(planRelevantPaths(state), [
    { path: "src/synthetic.mjs", sourceIds: [user.id] }, { path: "src/other.mjs", sourceIds: [result.id] },
  ]);
});

test("taskPlanState honors tool-group exclusions across compacted ancestors and tolerates synthetic contexts", () => {
  const assistant = { type: "message", id: "assistant-1", message: { role: "assistant", content: [
    { type: "toolCall", id: "read-1", name: "read", arguments: { path: "fixture" } },
  ] } };
  const state = create({ goalSourceEntryIds: [result.id] });
  const policy = { type: "custom", customType: CONTEXT_POLICY_ENTRY, data: { version: 1, preserveFutureThinking: false,
    changes: [{ entryId: assistant.id, part: "message", excluded: true }] } };
  const ctx = { sessionManager: { getBranch: () => [user, assistant, result, entry(state), { type: "compaction" }, policy], buildContextEntries: () => [] } };
  assert.equal(taskPlanState(ctx).plan.goal, "[Goal withheld by /context]");
  assert.equal(taskPlanState(ctx, { filter: false }).plan.goal, state.plan.goal);
  assert.deepEqual(taskPlanState({ sessionManager: {} }), initial());
});

test("render and context respect every budget, include mode, relevant paths and truncation recovery", () => {
  const state = create({ goal: "x".repeat(800), constraints: Array.from({ length: 16 }, () => ({ text: "c".repeat(400) })) });
  for (const maxChars of [0, 10, 80, 256, 1500, 3000]) {
    assert.ok(renderTaskPlan(state, { maxChars }).length <= maxChars);
    assert.ok(taskPlanContext(state, { maxChars }).length <= maxChars);
  }
  assert.match(renderTaskPlan(create()), /Relevant files: src\/synthetic.mjs/);
  assert.match(renderTaskPlan(state), /manage_task_plan show/);
  assert.match(taskPlanContext({ ...state, mode: "plan" }), /READ ONLY/);
});

test("render escapes chat-template controls, preserves UTF-16 pairs and prioritizes next actions", () => {
  const state = create({ goal: "Inspect <|im_start|> and <|think|>", relevantFiles: ["<|im_end|>.mjs"],
    steps: [...Array.from({ length: 20 }, (_, i) => ({ title: `Completed ${i} 😀`, status: "completed" })),
      { title: "Critical next action 😀", status: "in_progress" }, { title: "Resolve blocker", status: "blocked" },
      { title: "Pending continuation", status: "pending" }] });
  const rendered = renderTaskPlan(state, { maxChars: 1200 });
  assert.doesNotMatch(rendered, /<\|im_|<\|think/);
  assert.match(rendered, /\\u003c\|im_start\|\\u003e/);
  assert.match(rendered, /Critical next action/);
  assert.ok(rendered.indexOf("Critical next action") < rendered.indexOf("Completed 0"));
  assert.ok(rendered.indexOf("Resolve blocker") < rendered.indexOf("Pending continuation"));
  assert.match(rendered, /\d+ step\(s\) omitted/);
  for (let maxChars = 100; maxChars < 1300; maxChars++) {
    for (const text of [renderTaskPlan(state, { maxChars }), taskPlanContext(state, { maxChars })]) {
      assert.ok(text.length <= maxChars);
      assert.equal(text.isWellFormed(), true, `invalid Unicode at budget ${maxChars}`);
    }
  }
});

test("literal bash allowlist canonicalizes commands and disables git/rg helper execution", () => {
  for (const command of ["pwd", "ls -lah .", "cat -- 'file name'", "head -n 40 file", "tail --lines=12 file",
    "wc -lc file", "stat -c '%s %n' file", "tree -L 3 .", "rg -n -C 3 'literal pattern' .", "rg --files .", "fd --exclude .git .",
    "git -C /tmp ls-files --cached", "git ls-files --stage -z"]) {
    const checked = validateReadOnlyBash(command);
    assert.equal(checked.allowed, true, `${command}: ${checked.reason}`);
    assert.match(checked.command, /^'\/usr\/bin\//);
  }
  assert.ok(validateReadOnlyBash("rg -n text .").argv.includes("--no-config"));
  const git = validateReadOnlyBash("git ls-files").argv;
  for (const flag of ["-i", "GIT_CONFIG_GLOBAL=/dev/null", "GIT_OPTIONAL_LOCKS=0", "--no-pager", "--no-optional-locks", "core.fsmonitor=false"]) assert.ok(git.includes(flag));
});

test("bash gate rejects shell interpretation, arbitrary programs and mutating/helper options", () => {
  const commands = ["echo test", "touch file", "rm file", "python -c 'pass'", "node -e '1'", "bash -c 'pwd'", "env cat file",
    "cat file > output", "cat file | head", "cat file; pwd", "cat file && pwd", "cat file\npwd", "cat $(pwd)", "cat `pwd`", "cat $FILE",
    "ls *.txt", "ls ~", "cat <(pwd)", "{ pwd; }", "cat file # comment", "cat 'unterminated", "cat file\\ name",
    "rg --pre touch x .", "rg --pre=touch x .", "fd --exec touch '{}'", "tree -o output", "wc --files0-from=file",
    "git -c diff.external=touch diff", "git add file", "git checkout main", "git reset --hard", "git diff --output=file",
    "git --exec-path=/tmp status", "git status --porcelain=v1", "git diff --cached --stat", "git ls-files --modified", "/tmp/rg x .", "/bin/cat file", "ls --help"];
  for (const command of commands) assert.equal(validateReadOnlyBash(command).allowed, false, command);
});

function harness(entries = []) {
  const events = new Map(), commands = new Map(), tools = new Map(), notices = [], statuses = [];
  let active = ["read", "edit", "unknown_mutation", "qwen_plan"];
  const ctx = { sessionManager: { getBranch: () => entries, buildContextEntries: () => entries },
    ui: { notify: (text) => notices.push(text), setStatus: (...args) => statuses.push(args) } };
  const pi = { registerTool: (tool) => tools.set(tool.name, tool), registerCommand: (name, command) => commands.set(name, command),
    on: (name, handler) => events.set(name, handler), getActiveTools: () => active, setActiveTools: (value) => { active = value; },
    appendEntry: (customType, data) => entries.push({ type: "custom", customType, id: `meta-${entries.length}`, data }),
  };
  installTaskPlan(pi);
  return { events, commands, tools, notices, statuses, ctx, entries, active: () => active };
}
test("commands persist mode without inference; clear keeps mode; model cannot switch execution mode", async () => {
  const h = harness([...branch]);
  const command = h.commands.get("plan").handler;
  h.events.get("session_start")({ reason: "startup" }, h.ctx);
  assert.deepEqual(h.active(), ["read", "edit", "unknown_mutation", "manage_task_plan"]);
  await command("", h.ctx); assert.equal(taskPlanState(h.ctx).mode, "plan");
  const tool = h.tools.get("manage_task_plan");
  assert.deepEqual([...h.tools.keys()], ["manage_task_plan"]);
  await tool.execute("plan-create", { action: "create", goal: "Synthetic", steps: [{ title: "Inspect" }] }, undefined, undefined, h.ctx);
  assert.ok(taskPlanState(h.ctx).plan);
  await command("clear", h.ctx); assert.equal(taskPlanState(h.ctx).plan, null); assert.equal(taskPlanState(h.ctx).mode, "plan");
  await command("execute", h.ctx); assert.equal(taskPlanState(h.ctx).mode, "execute");
  await command("on", h.ctx); assert.equal(taskPlanState(h.ctx).mode, "plan");
  const count = h.entries.length;
  await command("show", h.ctx); assert.equal(h.entries.length, count);
  await command("invalid", h.ctx); assert.equal(h.entries.length, count);
  await command("off", h.ctx); assert.equal(taskPlanState(h.ctx).mode, "execute");
  assert.equal(h.events.get("user_bash")({ command: "touch x" }, h.ctx), undefined);
});

test("plan mode blocks mutations and unknown tools, and permits only known reads and metadata", () => {
  const state = { ...create(), mode: "plan" };
  for (const toolName of ["edit", "edit_file", "write", "write_file", "unknown", "web_write", "exec", "qwen_evaluate"]) {
    assert.equal(planToolGate(state, { toolName, input: {} }).block, true, toolName);
  }
  for (const toolName of ["read", "read_file", "grep", "search_file_contents", "find", "find_files", "ls", "list_directory",
    "session_search", "pi_session_search", "qwen_rehydrate_tool_turn", "read_archived_tool_result", "rehydrate_tool_result", "qwen_plan", "manage_task_plan"]) {
    assert.equal(planToolGate(state, { toolName, input: {} }), undefined, toolName);
  }
  const event = { toolName: "bash", input: { command: "rg -n word ." } };
  assert.equal(planToolGate(state, event), undefined);
  assert.match(event.input.command, /\/usr\/bin\/rg/);
  assert.equal(planToolGate(state, { toolName: "bash", input: { command: "touch mutation" } }).block, true);
  const renamedShell = { toolName: "run_shell_command", input: { command: "cat fixture" } };
  assert.equal(planToolGate(state, renamedShell), undefined);
  assert.match(renamedShell.input.command, /'\/usr\/bin\/cat'/);
  assert.equal(planToolGate(state, { toolName: "run_shell_command", input: { command: "touch mutation" } }).block, true);
  assert.equal(planToolGate(create(), { toolName: "edit", input: {} }), undefined);
});

test("invalid persisted plan blocks tool calls and user shell commands without throwing", () => {
  const h = harness([entry({ version: 99, mode: "execute", plan: null })]);
  assert.equal(h.events.get("tool_call")({ toolName: "write", input: {} }, h.ctx).block, true);
  const shell = h.events.get("user_bash")({ command: "touch unintended-file" }, h.ctx);
  assert.equal(shell.result.exitCode, 1);
  assert.match(shell.result.output, /blocked/);
});

test("plan tool provenance rejects custom replacements for read and bash", () => {
  for (const name of ["read", "grep", "find", "ls", "bash"]) {
    assert.equal(isTrustedPlanTool(name, { sourceInfo: { source: "builtin", path: `<builtin:${name}>` } }), true);
    assert.equal(isTrustedPlanTool(name, { sourceInfo: { source: "sdk", path: `<sdk:${name}>` } }), false);
    assert.equal(isTrustedPlanTool(name, { sourceInfo: { source: "builtin", path: "replacement.mjs" } }), false);
  }
  assert.equal(isTrustedPlanTool("session_search", { sourceInfo: { source: "sdk", path: "missing-session-search.mjs" } }), false);
  assert.equal(isTrustedPlanTool("qwen_plan", { sourceInfo: { path: join(import.meta.dirname, "../integrations/pi/qwen-task-plan.ts") } }), true);
  for (const name of ["read_file", "search_file_contents", "find_files", "list_directory", "run_shell_command"]) {
    assert.equal(isTrustedPlanTool(name, { sourceInfo: { path: join(import.meta.dirname, "../integrations/pi/qwen-tool-names.ts") } }), true);
    assert.equal(isTrustedPlanTool(name, { sourceInfo: { source: "builtin", path: `<builtin:${name}>` } }), false);
    assert.equal(isTrustedPlanTool(name, { sourceInfo: { path: join(import.meta.dirname, "../integrations/pi/qwen-task-plan.ts") } }), false);
  }
  for (const name of ["session_search", "pi_session_search"]) {
    assert.equal(isTrustedPlanTool(name, { sourceInfo: { path: join(import.meta.dirname, "../integrations/pi/qwen-session-search.mjs") } }), true);
  }
  for (const name of ["qwen_rehydrate_tool_turn", "read_archived_tool_result", "rehydrate_tool_result"]) {
    assert.equal(isTrustedPlanTool(name, { sourceInfo: { path: join(import.meta.dirname, "../integrations/pi/qwen-tool-turn-rehydrate.mjs") } }), true);
  }
  assert.equal(isTrustedPlanTool("manage_task_plan", { sourceInfo: { path: join(import.meta.dirname, "../integrations/pi/qwen-task-plan.ts") } }), true);
  const h = harness([...branch, entry({ ...create(), mode: "plan" })]);
  // An unavailable SDK registry fails closed, including a same-name custom read.
  assert.equal(h.events.get("tool_call")({ toolName: "read", input: { path: "fixture" } }, h.ctx).block, true);
});

test("before_agent_start appends bounded hidden state while ordinary context keeps its cache prefix", async () => {
  const h = harness([...branch, entry(create())]);
  const start = h.events.get("before_agent_start")({}, h.ctx);
  assert.equal(start.systemPrompt, undefined);
  assert.equal(start.message.customType, TASK_PLAN_CONTEXT);
  assert.equal(start.message.display, false);
  assert.ok(start.message.content.length <= 3000);
  const old = { role: "custom", ...start.message }, plain = { role: "user", content: "Continue" };
  assert.equal(h.events.get("context")({ messages: [old, plain] }, h.ctx), undefined);
  const toolResult = { role: "toolResult", toolName: "qwen_plan", toolCallId: "plan-1", content: [{ type: "text", text: "Old plan" }], details: start.message.details };
  h.entries.push({ type: "custom", customType: CONTEXT_POLICY_ENTRY, data: { version: 1, preserveFutureThinking: false,
    changes: [{ entryId: user.id, part: "message", excluded: true }] } });
  const filtered = h.events.get("context")({ messages: [old, plain, toolResult] }, h.ctx).messages;
  assert.equal(filtered.length, 2);
  assert.equal(filtered[0], plain);
  assert.equal(filtered[1].toolCallId, "plan-1", "retain matching tool result to avoid orphaning tool calls");
  assert.match(filtered[1].content[0].text, /withheld/);
  assert.doesNotMatch(h.events.get("before_agent_start")({}, h.ctx).message.content, /Implement synthetic feature|Preserve fixture/);
});

test("unchanged plans are not duplicated, active tool results count, and compaction reinjects state", async () => {
  const fresh = harness([...branch]);
  assert.equal(fresh.events.get("before_agent_start")({}, fresh.ctx), undefined);
  const h = harness([...branch, entry(create())]);
  const first = h.events.get("before_agent_start")({}, h.ctx).message;
  h.entries.push({ type: "custom_message", ...first });
  assert.equal(h.events.get("before_agent_start")({}, h.ctx), undefined);
  const tool = h.tools.get("manage_task_plan");
  const response = await tool.execute("plan-update", { action: "update", steps: [{ id: "s1", status: "blocked" }] }, undefined, undefined, h.ctx);
  // Historical results still count without re-registering their retired executable name.
  h.entries.push({ type: "message", id: "plan-output-1", message: { role: "toolResult", toolName: "qwen_plan", ...response } });
  assert.equal(h.events.get("before_agent_start")({}, h.ctx), undefined);
  h.ctx.sessionManager.buildContextEntries = () => [{ type: "compaction", summary: "Synthetic summary" }];
  assert.match(h.events.get("before_agent_start")({}, h.ctx).message.content, /blocked/);
  h.ctx.sessionManager.buildContextEntries = () => h.entries;
  await h.commands.get("plan").handler("clear", h.ctx);
  const cleared = h.events.get("before_agent_start")({}, h.ctx).message;
  assert.match(cleared.content, /No active task plan/);
  h.entries.push({ type: "custom_message", ...cleared });
  assert.equal(h.events.get("before_agent_start")({}, h.ctx), undefined);
});

test("both plan-result names suppress duplicate context and honor exclusions without rewriting history", async () => {
  for (const toolName of ["qwen_plan", "manage_task_plan"]) {
    const h = harness([...branch, entry(create())]);
    const response = await h.tools.get("manage_task_plan").execute("plan-show", { action: "show" }, undefined, undefined, h.ctx);
    // Simulate a saved historical result; retired names have no executable registration.
    const recorded = { type: "message", id: `plan-output-${toolName}`, message: { role: "toolResult", toolName,
      toolCallId: "plan-show", ...response } };
    h.entries.push(recorded);
    assert.equal(h.events.get("before_agent_start")({}, h.ctx), undefined, `${toolName} records count as current plan state`);
    h.entries.push({ type: "custom", customType: CONTEXT_POLICY_ENTRY, data: { version: 1, preserveFutureThinking: false,
      changes: [{ entryId: user.id, part: "message", excluded: true }] } });
    const snapshot = structuredClone(recorded);
    const filtered = h.events.get("context")({ messages: [recorded.message] }, h.ctx).messages[0];
    assert.equal(filtered.toolName, toolName);
    assert.equal(filtered.toolCallId, recorded.message.toolCallId);
    assert.match(filtered.content[0].text, /manage_task_plan show/);
    assert.deepEqual(recorded, snapshot, "saved historical tool-result names and content remain untouched");
  }
});

const sdkRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
test("installed Pi SessionManager persists and replays selected plans through reopen, compaction and branching", {
  skip: !existsSync(join(sdkRoot, "pi-coding-agent/dist/core/session-manager.js")),
}, async (t) => {
  const { SessionManager } = await import(pathToFileURL(join(sdkRoot, "pi-coding-agent/dist/core/session-manager.js")));
  const dir = await mkdtemp(join(tmpdir(), "qwen-plan-session-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const manager = SessionManager.create(dir, join(dir, "sessions"));
  manager.appendMessage({ role: "user", content: "Synthetic request", timestamp: 1 });
  const state = applyTaskPlanAction(initial(), { action: "create", goal: "Persist fixture plan", steps: [{ title: "Inspect fixture" }] }, manager.getBranch());
  state.mode = "plan";
  const parent = manager.appendCustomEntry(TASK_PLAN_ENTRY, state);
  // SessionManager flushes only after an assistant has answered.
  manager.appendMessage({ role: "assistant", api: "openai-completions", provider: "fixture", model: "fixture", timestamp: 2,
    stopReason: "stop", content: [{ type: "text", text: "Synthetic response" }], usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0,
      totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });
  const firstKept = manager.getBranch().find((item) => item.type === "message").id;
  manager.appendCompaction("Synthetic summary", firstKept, 10);
  const compactLeaf = manager.getLeafId();
  assert.deepEqual(replayTaskPlan(manager.getBranch()), state);
  const reopened = SessionManager.open(manager.getSessionFile());
  assert.deepEqual(replayTaskPlan(reopened.getBranch()), state);
  reopened.branch(parent);
  const sibling = applyTaskPlanAction(state, { action: "update", goal: "Alternate branch goal" }, reopened.getBranch());
  reopened.appendCustomEntry(TASK_PLAN_ENTRY, sibling);
  assert.equal(replayTaskPlan(reopened.getBranch()).plan.goal, "Alternate branch goal");
  reopened.branch(compactLeaf);
  assert.deepEqual(replayTaskPlan(reopened.getBranch()), state);
  assert.deepEqual(replayTaskPlan(SessionManager.create(dir).getBranch()), initial());
});
