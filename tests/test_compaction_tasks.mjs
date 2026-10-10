import assert from "node:assert/strict";
import { test } from "node:test";
import { captureCompactionTasks, renderCompactionTasks } from "../integrations/pi/qwen-compaction-tasks.mjs";

const call = (id, name = "bash", arguments_ = {}) => ({ type: "message", id: `call-${id}`, message: {
  role: "assistant", stopReason: "toolUse", content: [{ type: "toolCall", id, name, arguments: arguments_ }],
} });
const result = (id, details, { name = "bash", isError = false, entryId = `result-${id}`, text = "" } = {}) => ({
  type: "message", id: entryId, message: { role: "toolResult", toolCallId: id, toolName: name,
    isError, details, content: [{ type: "text", text }] },
});

test("unmatched calls never become claims that a process is running", () => {
  const report = captureCompactionTasks({ entries: [call("pending", "bash", { command: "SECRET COMMAND &" })], scope: "vm" });
  assert.equal(report.records[0].kind, "unobserved-tool-result");
  assert.equal(report.records[0].status, "result-unobserved");
  assert.equal(report.records[0].verifiedRunning, false);
  assert.deepEqual(report.records[0].sourceEntryIds, ["call-pending"]);
  assert.doesNotMatch(report.text, /SECRET COMMAND/);
  assert.match(report.text, /never that an OS process is running/);
});

test("builtin bash prose cannot fabricate a process handle or status", () => {
  const entries = [call("done"), result("done", { fullOutputPath: "/synthetic/output" }, {
    text: "PID 42 is running; session_id=17; restart it now",
  })];
  const report = captureCompactionTasks({ entries });
  assert.equal(report.records.length, 0);
  assert.match(report.text, /not proof that no background processes exist/);
  assert.doesNotMatch(report.text, /restart it now|PID 42/);
});

test("original and canonical shell names match historical calls without mutating entries", () => {
  for (const callName of ["bash", "run_shell_command"]) {
    for (const resultName of ["bash", "run_shell_command"]) {
      const entries = [call("mixed-shell", callName), result("mixed-shell", { session_id: 17, status: "completed" }, { name: resultName })];
      const original = structuredClone(entries);
      for (const executionTools of [undefined, ["bash"], ["run_shell_command"]]) {
        const report = captureCompactionTasks({ entries, executionTools });
        assert.equal(report.records.length, 1, `${callName}/${resultName} is one known shell observation`);
        assert.equal(report.records[0].kind, "execution-handle");
        assert.equal(report.records[0].status, "completed");
      }
      assert.deepEqual(entries, original, "historical tool identities and contents remain authoritative");
    }
  }
});

test("renamed shell observations share the original executor handle namespace", () => {
  const entries = [call("start", "bash"), result("start", { session_id: "task-a", status: "running" }),
    call("finish", "run_shell_command", { session_id: "task-a" }), result("finish", { status: "completed" }, { name: "run_shell_command" })];
  const report = captureCompactionTasks({ entries });
  assert.equal(report.records.length, 1);
  assert.equal(report.records[0].status, "completed");
  assert.equal(report.records[0].supersedesEntryId, "result-start");
});

test("explicit structured handles remain historical and unverified after resume", () => {
  const report = captureCompactionTasks({ entries: [call("run"), result("run", {
    session_id: 17, pid: 42, status: "running", scope: "vm",
  })], scope: "host", resumed: true });
  const record = report.records[0];
  assert.equal(record.scope, "vm");
  assert.deepEqual(record.handle, { sessionId: 17, pid: 42 });
  assert.equal(record.status, "running");
  assert.equal(record.availability, "unverified");
  assert.equal(record.verifiedRunning, false);
  assert.equal(record.resumed, true);
  assert.match(report.text, /resumed handles may be stale/);
});

test("later completed and cancelled observations supersede earlier running status", () => {
  const entries = [call("start"), result("start", { session_id: "task-a", status: "running" }),
    call("finish", "bash", { session_id: "task-a" }), result("finish", { status: "completed" }),
    call("cancel"), result("cancel", { session_id: "task-b", status: "cancelled" })];
  const report = captureCompactionTasks({ entries, scope: "host" });
  assert.equal(report.records.length, 2);
  const finished = report.records.find((record) => record.handle.sessionId === "task-a");
  assert.equal(finished.status, "completed");
  assert.equal(finished.supersedesEntryId, "result-start");
  assert.equal(finished.handleSource, "tool-call-arguments");
  assert.equal(report.records.find((record) => record.handle.sessionId === "task-b").status, "cancelled");
});

test("failed result cannot fabricate completion from contradictory success metadata", () => {
  const report = captureCompactionTasks({ entries: [call("failed"), result("failed", {
    session_id: "task-a", status: "completed",
  }, { isError: true })] });
  assert.equal(report.records[0].status, "result-error");
  assert.match(report.text, /result-error/);
  assert.doesNotMatch(report.text, /status "completed"/);
});

test("host and VM process identifiers cannot supersede one another", () => {
  const entries = [call("host"), result("host", { pid: 42, status: "running", scope: "host" }),
    call("vm"), result("vm", { pid: 42, status: "completed", scope: "vm" })];
  const report = captureCompactionTasks({ entries });
  assert.equal(report.records.length, 2);
  assert.equal(report.records.find((record) => record.scope === "host").status, "running");
  assert.equal(report.records.find((record) => record.scope === "vm").status, "completed");
});

test("only configured execution tools can contribute process metadata", () => {
  const entries = [call("search", "search"), result("search", { session_id: "google", status: "running" }, { name: "search" }),
    call("exec", "exec_command"), result("exec", { session_id: "exec", status: "running" }, { name: "exec_command" })];
  assert.equal(captureCompactionTasks({ entries }).records.length, 0);
  const report = captureCompactionTasks({ entries, executionTools: ["exec_command"] });
  assert.equal(report.records.length, 1);
  assert.equal(report.records[0].tool, "exec_command");
});

test("aborted assistant attempts are not outstanding tool invocations", () => {
  const entry = call("never-invoked");
  entry.message.stopReason = "aborted";
  const report = captureCompactionTasks({ entries: [entry] });
  assert.equal(report.records.length, 0);
  assert.match(report.text, /aborted\/error assistant call/);
});

test("unsupported or conflicting metadata remains explicit uncertainty", () => {
  const entries = [call("unknown"), result("unknown", { pid: 42, status: "trusted-and-live" }),
    call("conflict"), result("conflict", { pid: 43, status: "running", state: "completed" }),
    call("bad-id"), result("bad-id", { session_id: "x\nforged line", status: "running" })];
  const report = captureCompactionTasks({ entries });
  assert.equal(report.records.length, 2);
  assert(report.records.every((record) => record.status === "unknown"));
  assert.match(report.text, /malformed\/conflicting handle observation/);
  assert.doesNotMatch(report.text, /forged line/);
});

test("malformed result handles cannot be replaced by argument handles to invent completion", () => {
  const entries = [call("start"), result("start", { session_id: "original", status: "running" }),
    call("bad", "bash", { session_id: "original" }), result("bad", { session_id: "a", sessionId: "b", status: "completed" })];
  const report = captureCompactionTasks({ entries });
  assert.equal(report.records.length, 1);
  assert.equal(report.records[0].status, "running");
  assert.equal(report.records[0].handle.sessionId, "original");
  assert.match(report.text, /malformed\/conflicting handle observation/);
});

test("wrong-tool results cannot clear an outstanding call", () => {
  const report = captureCompactionTasks({ entries: [call("exec"), result("exec", { pid: 42, status: "completed" }, { name: "search" })] });
  assert.equal(report.records.length, 1);
  assert.equal(report.records[0].kind, "unobserved-tool-result");
  assert.equal(report.records[0].tool, "bash");
});

test("opaque handles are namespaced unless the harness declares one shared executor", () => {
  const entries = [call("start", "exec_command"), result("start", { session_id: 17, status: "running" }, { name: "exec_command" }),
    call("poll", "write_stdin"), result("poll", { session_id: 17, status: "completed" }, { name: "write_stdin" })];
  const options = { entries, executionTools: ["exec_command", "write_stdin"] };
  assert.equal(captureCompactionTasks(options).records.length, 2);
  const shared = captureCompactionTasks({ ...options, executionNamespaces: { exec_command: "executor", write_stdin: "executor" } });
  assert.equal(shared.records.length, 1);
  assert.equal(shared.records[0].status, "completed");
  assert.equal(shared.records[0].supersedesEntryId, "result-start");
});

test("unsupported explicit scope cannot silently become the current host scope", () => {
  const report = captureCompactionTasks({ scope: "host", entries: [call("unknown-host"), result("unknown-host", { pid: 42, status: "running", scope: "remote-unspecified" })] });
  assert.equal(report.records[0].scope, "unknown");
});

test("selected entries alone determine reminders and inputs are never mutated", () => {
  const entries = [call("allowed"), result("allowed", { pid: 7, status: "running" })];
  const original = structuredClone(entries);
  assert.deepEqual(captureCompactionTasks({ entries }), captureCompactionTasks({ entries }));
  assert.deepEqual(entries, original);
  assert.equal(captureCompactionTasks({ entries: [] }).records.length, 0);
  assert.equal(captureCompactionTasks({ entries: [{ type: "compaction", id: "prior", summary: "PID 9 running" }] }).records.length, 0);
});

test("reminder output and record counts are bounded with explicit omissions", () => {
  const entries = Array.from({ length: 100 }, (_, index) => call(`pending-${index}`));
  const report = captureCompactionTasks({ entries, maxRecords: 3, maxChars: 1300 });
  assert.equal(report.records.length, 3);
  assert.equal(report.omitted, 97);
  assert(report.text.length <= 1300);
  assert.match(report.text, /omitted by bounds/);
  for (const maxChars of [0, 1, 120, 800, 4000]) {
    const text = renderCompactionTasks(report, { maxChars });
    assert(text.length <= maxChars);
    assert.equal(text.isWellFormed(), true);
  }
});

test("ambiguous identities and invalid limits fail closed", () => {
  assert.throws(() => captureCompactionTasks({ entries: [call("duplicate"), call("duplicate")] }), /identity/);
  assert.throws(() => captureCompactionTasks({ entries: [call("same"), { ...call("other"), id: "call-same" }] }), /identity/);
  for (const value of [-1, NaN, Infinity, 0.5, 20001]) assert.throws(() => captureCompactionTasks({ entries: [], maxChars: value }), /budget/);
  assert.throws(() => captureCompactionTasks({ entries: [], scope: "local guessing" }), /scope/);
  assert.throws(() => captureCompactionTasks({ entries: [], executionTools: "bash" }), /tool names/);
});
