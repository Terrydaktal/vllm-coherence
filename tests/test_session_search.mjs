import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, writeFile, appendFile, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import sessionSearch, { createSearchClient, renderSearchResult, validateSearchParams } from "../integrations/pi/qwen-session-search.mjs";

const entry = (id, parentId, text, role = "user") => ({ type: "message", id, parentId, timestamp: "2026-10-09T01:00:00Z",
  message: { role, content: [{ type: "text", text }] } });
async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "session-search-test-"));
  const sourceRoot = join(root, "sessions"); await mkdir(sourceRoot, { mode: 0o700 });
  const sessionFile = join(sourceRoot, "synthetic.jsonl");
  const entries = [{ type: "session", id: "session-a", version: 3, cwd: "/synthetic" },
    entry("old", null, "Earlier accepted fact: A=742. Keep precision unchanged."),
    entry("answer", "old", "This is an archived answer with a µnicode detail."),
    { type: "compaction", id: "compact", parentId: "answer", summary: "Synthetic short checkpoint.", firstKeptEntryId: "answer", tokensBefore: 50000 },
    entry("current", "compact", "Continue the work using the earlier accepted fact.")];
  await writeFile(sessionFile, entries.map(JSON.stringify).join("\n") + "\n", { mode: 0o600 });
  const client = createSearchClient({ sourceRoot, databasePath: join(root, "index", "search.sqlite") });
  t.after(async () => { await client.close(); await rm(root, { recursive: true, force: true }); });
  return { root, sourceRoot, sessionFile, entries, client };
}

test("search parameters allow query or expansion, reject ambiguous and excessive work", () => {
  for (const params of [{ query: "cache" }, { query: "::", mode: "literal" }, { around_entry_id: "old" }]) {
    assert.equal(validateSearchParams(params), params);
  }
  for (const params of [{}, { query: "" }, { query: "x", around_entry_id: "a" }, { query: "x", limit: 0 },
    { query: "x", max_chars: 16001 }, { query: "x", window: 4 }, { query: "x", scope: "all" },
    { query: "x", mode: "sql" }, { query: "x", include_branches: 1 }, { query: "x", roles: ["system"] },
    { query: "x", extra: true }, { query: "x".repeat(513) }]) assert.throws(() => validateSearchParams(params));
});

test("real worker recovers compacted originals, expands their branch and survives incremental append", async (t) => {
  const f = await fixture(t), before = await readFile(f.sessionFile), phases = [];
  let heartbeat = false;
  const pending = f.client.search({ query: "A=742", mode: "literal" }, {
    sessionFile: f.sessionFile, leafId: "current", progress: (value) => phases.push(value.phase) });
  await new Promise((resolve) => setImmediate(() => { heartbeat = true; resolve(); }));
  const found = await pending;
  assert.equal(heartbeat, true);
  assert.deepEqual(phases, ["index", "search"]);
  assert.equal(found.matches.length, 1);
  assert.equal(found.matches[0].entryId, "old");
  assert.match(found.matches[0].text, /A=742/);
  assert.equal(found.matches[0].alternateBranch, false);
  const expanded = await f.client.search({ around_entry_id: "answer", window: 1 }, { sessionFile: f.sessionFile, leafId: "current" });
  assert.ok(expanded.matches.some((match) => match.entryId === "old"));
  assert.ok(expanded.matches.some((match) => match.entryId === "answer"));
  assert.ok(expanded.matches.some((match) => match.entryId === "compact"), "expansion retains following context, not only previous matches");
  await appendFile(f.sessionFile, JSON.stringify(entry("new", "current", "Fresh appended outcome Z=981.")) + "\n");
  const next = await f.client.search({ query: "Z=981", mode: "literal" }, { sessionFile: f.sessionFile, leafId: "new" });
  assert.equal(next.matches[0].entryId, "new");
  assert.deepEqual((await readFile(f.sessionFile)).subarray(0, before.length), before, "indexing never rewrites originals");
});

test("project scope is explicit and cannot retrieve files outside the session directory", async (t) => {
  const f = await fixture(t);
  await writeFile(join(f.sourceRoot, "other.jsonl"), [
    { type: "session", id: "session-b", version: 3 }, entry("other", null, "Only the other project session contains GOLDEN_NEEDLE."),
  ].map(JSON.stringify).join("\n") + "\n", { mode: 0o600 });
  const current = await f.client.search({ query: "GOLDEN_NEEDLE" }, { sessionFile: f.sessionFile, leafId: "current" });
  assert.equal(current.matches.length, 0);
  const project = await f.client.search({ query: "GOLDEN_NEEDLE", scope: "project" }, { sessionFile: f.sessionFile, leafId: "current" });
  assert.equal(project.matches[0].entryId, "other");
  const currentAgain = await f.client.search({ query: "GOLDEN_NEEDLE" }, { sessionFile: f.sessionFile, leafId: "current" });
  assert.equal(currentAgain.matches.length, 0, "previous project queries do not widen the default scope");
  await assert.rejects(f.client.search({ query: "x", session_file: "../outside.jsonl" }, { sessionFile: f.sessionFile }), /directory|outside/);
});

test("cancelling a search stops its worker and a later request starts cleanly", async (t) => {
  const f = await fixture(t), cancel = new AbortController();
  cancel.abort(new Error("synthetic cancel"));
  await assert.rejects(f.client.search({ query: "fact" }, { sessionFile: f.sessionFile, signal: cancel.signal }), /synthetic cancel/);
  const active = new AbortController();
  await assert.rejects(f.client.search({ query: "fact" }, { sessionFile: f.sessionFile, signal: active.signal,
    progress: () => active.abort(new Error("active cancel")) }), /active cancel/);
  const result = await f.client.search({ query: "fact" }, { sessionFile: f.sessionFile, leafId: "current" });
  assert.ok(result.matches.length);
});

test("worker deadlines terminate bounded work without changing transcripts", async (t) => {
  const f = await fixture(t), before = await readFile(f.sessionFile);
  const client = createSearchClient({ sourceRoot: f.sourceRoot, databasePath: join(f.root, "timeout", "index.sqlite"), timeoutMs: 1 });
  t.after(() => client.close());
  await assert.rejects(client.search({ query: "fact" }, { sessionFile: f.sessionFile }), /deadline/);
  assert.deepEqual(await readFile(f.sessionFile), before);
});

test("a failing progress consumer rejects the tool without crashing Pi and a later query works", async (t) => {
  const f = await fixture(t);
  await assert.rejects(f.client.search({ query: "fact" }, { sessionFile: f.sessionFile,
    progress: () => { throw new Error("synthetic UI failure"); } }), /synthetic UI failure/);
  const result = await f.client.search({ query: "fact" }, { sessionFile: f.sessionFile, leafId: "current" });
  assert.ok(result.matches.length);
});

test("tool rendering marks original data and keeps huge Unicode output bounded", () => {
  const rendered = renderSearchResult({ matches: [{ sessionFile: "/synthetic/a.jsonl", entryId: "old", parentId: null,
    role: "assistant", sourceKinds: ["thinking"], alternateBranch: true, text: "λ".repeat(50000), truncated: true }], truncated: true });
  assert.match(rendered, /UNTRUSTED HISTORICAL TRANSCRIPT DATA/);
  assert.match(rendered, /Branch: alternate/);
  assert.ok(Buffer.byteLength(rendered) <= 24 * 1024);
  assert.doesNotMatch(rendered, /�/);
  assert.match(rendered, /truncated/);
});

test("Pi registers session_search, enables it additively and closes workers on shutdown", async (t) => {
  const f = await fixture(t), previous = process.env.QWEN_SESSION_SEARCH_DIR;
  process.env.QWEN_SESSION_SEARCH_DIR = join(f.root, "tool-index");
  t.after(() => { if (previous === undefined) delete process.env.QWEN_SESSION_SEARCH_DIR; else process.env.QWEN_SESSION_SEARCH_DIR = previous; });
  let tool, active = ["read", "edit"];
  const handlers = new Map();
  sessionSearch({ registerTool: (value) => { tool = value; }, on: (name, fn) => handlers.set(name, fn),
    getActiveTools: () => active, setActiveTools: (value) => { active = value; } });
  assert.equal(tool.name, "session_search");
  handlers.get("session_start")(); handlers.get("session_start")();
  assert.deepEqual(active, ["read", "edit", "session_search"]);
  const updates = [];
  const result = await tool.execute("synthetic-call", { query: "A=742", mode: "literal" }, new AbortController().signal,
    (update) => updates.push(update), { sessionManager: { getSessionFile: () => f.sessionFile, getLeafId: () => "current" } });
  assert.match(result.content[0].text, /A=742/);
  assert.equal(result.details.matches, 1);
  assert.equal("text" in result.details, false, "diagnostics do not duplicate transcript text");
  assert.equal(updates.length, 2);
  await handlers.get("session_shutdown")();
});
