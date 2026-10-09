// Legacy synthetic JSONL only: no private sessions, model calls or network.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const available = existsSync(join(root, "pi-coding-agent/dist/core/session-manager.js"));
const CONTRACT = "radiance-prefix-compaction-v1";
const STALE = 253792;
const mod = (path) => import(pathToFileURL(join(root, path)));

async function savedSession(t) {
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const dir = await mkdtemp(join(tmpdir(), "pi-compaction-token-count-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const manager = SessionManager.create(dir, join(dir, "sessions"));
  const user = manager.appendMessage({ role: "user", content: "Synthetic pre-compaction task", timestamp: 1 });
  const assistant = manager.appendMessage({ role: "assistant", provider: "fixture", model: "fixture", api: "openai-completions",
    content: [{ type: "text", text: "Synthetic completed response" }], timestamp: 2, stopReason: "stop",
    usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });
  return { manager, SessionManager, user, assistant };
}
const projectedCount = (manager) => manager.buildSessionContext().messages.find((message) => message.role === "compactionSummary")?.tokensBefore;

test("installed Pi projects recorded historical counts through reopen and tree without rewriting JSONL", { skip: !available }, async (t) => {
  const { manager, SessionManager, user, assistant } = await savedSession(t);
  const first = manager.appendCompaction("Synthetic old first checkpoint", user, STALE, { contract: CONTRACT, historicalTokens: 2 });
  const filename = manager.getSessionFile();
  assert.equal(projectedCount(manager), 2, "live projection must use exact selected-history count rather than stale assistant usage");
  assert.equal(manager.getEntry(first).tokensBefore, STALE, "display repair must not mutate original entry fields");
  manager.branch(assistant);
  const sibling = manager.appendCompaction("Synthetic alternate checkpoint", user, STALE, { contract: CONTRACT, historicalTokens: 7 });
  const original = await readFile(filename);
  const reopened = SessionManager.open(filename);
  assert.equal(projectedCount(reopened), 7);
  reopened.branch(first);
  assert.equal(projectedCount(reopened), 2);
  assert.equal(reopened.getEntry(first).tokensBefore, STALE);
  reopened.branch(sibling);
  assert.equal(projectedCount(reopened), 7);
  assert.deepEqual(await readFile(filename), original, "opening and navigating must preserve saved transcript bytes");
});

test("real collapsed and expanded UI display the selected historical count for a legacy checkpoint", { skip: !available }, async (t) => {
  const { manager, SessionManager, user } = await savedSession(t);
  const { CompactionSummaryMessageComponent } = await mod("pi-coding-agent/dist/modes/interactive/components/compaction-summary-message.js");
  const { initTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  initTheme("dark", false);
  manager.appendCompaction("Synthetic old checkpoint body", user, STALE, { contract: CONTRACT, historicalTokens: 2 });
  const filename = manager.getSessionFile(), original = await readFile(filename);
  const reopened = SessionManager.open(filename);
  const message = reopened.buildSessionContext().messages.find((message) => message.role === "compactionSummary");
  const component = new CompactionSummaryMessageComponent(message);
  for (const expanded of [false, true]) {
    await t.test(expanded ? "expanded" : "collapsed", () => {
      component.setExpanded(expanded);
      const rendered = component.render(160).join("\n");
      assert.match(rendered, /Compacted from 2 tokens/);
      assert.doesNotMatch(rendered, /253,792|253792/);
      if (expanded) assert.match(rendered, /Synthetic old checkpoint body/);
    });
  }
  assert.deepEqual(await readFile(filename), original);
});

test("projection override accepts only nonnegative safe integers from the Radiance compaction contract", { skip: !available }, async (t) => {
  const { manager, SessionManager, user, assistant } = await savedSession(t);
  const cases = [
    ["positive", { contract: CONTRACT, historicalTokens: 2 }, 2],
    ["zero", { contract: CONTRACT, historicalTokens: 0 }, 0],
    ["largest safe integer", { contract: CONTRACT, historicalTokens: Number.MAX_SAFE_INTEGER }, Number.MAX_SAFE_INTEGER],
    ["other contract", { contract: "other-contract", historicalTokens: 2 }, STALE],
    ["missing contract", { historicalTokens: 2 }, STALE],
    ["missing count", { contract: CONTRACT }, STALE],
    ["negative", { contract: CONTRACT, historicalTokens: -1 }, STALE],
    ["fractional", { contract: CONTRACT, historicalTokens: 2.5 }, STALE],
    ["unsafe integer", { contract: CONTRACT, historicalTokens: Number.MAX_SAFE_INTEGER + 1 }, STALE],
    ["string", { contract: CONTRACT, historicalTokens: "2" }, STALE],
    ["null", { contract: CONTRACT, historicalTokens: null }, STALE],
    ["boolean", { contract: CONTRACT, historicalTokens: true }, STALE],
    ["nonfinite", { contract: CONTRACT, historicalTokens: Infinity }, STALE],
  ];
  const entries = [];
  for (const [label, details, expected] of cases) {
    manager.branch(assistant);
    const id = manager.appendCompaction(`Synthetic ${label} checkpoint`, user, STALE, details);
    assert.equal(projectedCount(manager), expected, label);
    entries.push({ id, label, expected });
  }
  const filename = manager.getSessionFile(), original = await readFile(filename);
  const reopened = SessionManager.open(filename);
  for (const { id, label, expected } of entries) {
    reopened.branch(id);
    assert.equal(projectedCount(reopened), expected, `reopen: ${label}`);
  }
  assert.deepEqual(await readFile(filename), original);
});
