import test from "node:test";
import assert from "node:assert/strict";
import { appendFileSync, chmodSync, linkSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, symlinkSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { DatabaseSync } from "node:sqlite";
import { TranscriptIndex } from "../integrations/pi/qwen-session-index.mjs";

const stamp = "2026-10-09T00:00:00.000Z";
const message = (id, parentId, role, content) => ({ type: "message", id, parentId, timestamp: stamp, message: { role, content } });
const serialize = (sessionId, entries) => [JSON.stringify({ type: "session", version: 3, id: sessionId, timestamp: stamp }), ...entries.map(JSON.stringify)].join("\n") + "\n";

function fixture(t, entries = []) {
  const temporary = mkdtempSync(join(tmpdir(), "session-index-"));
  const sourceRoot = join(temporary, "sources");
  mkdirSync(sourceRoot, { mode: 0o700 });
  const databasePath = join(temporary, "cache", "index.sqlite");
  const file = join(sourceRoot, "one.jsonl");
  writeFileSync(file, serialize("session-one", entries), { mode: 0o600 });
  const index = new TranscriptIndex({ databasePath, sourceRoot });
  t.after(() => { index.close(); rmSync(temporary, { recursive: true, force: true }); });
  return { temporary, sourceRoot, databasePath, file, index,
    write: (items, path = file, sessionId = "session-one") => writeFileSync(path, serialize(sessionId, items), { mode: 0o600 }),
    append: (entry, path = file) => appendFileSync(path, JSON.stringify(entry) + "\n"),
    search: (query, options = {}) => index.search({ query, sessionFile: file, ...options }) };
}

test("contentless postings preserve phrase and BM25 ranking parity without storing bodies", (t) => {
  const bodies = ["alpha beta", "alpha alpha beta", "beta elsewhere alpha", "no matching terms"];
  const entries = bodies.map((body, i) => message(`entry-${i}`, i ? `entry-${i - 1}` : null, "user", body));
  const f = fixture(t, entries);
  const report = f.index.sync([f.file]);
  assert.equal(report.indexedEntries, bodies.length);
  const ordinary = new DatabaseSync(":memory:");
  t.after(() => ordinary.close());
  ordinary.exec("CREATE VIRTUAL TABLE full USING fts5(body,detail=full,columnsize=1)");
  bodies.forEach((body, i) => ordinary.prepare("INSERT INTO full(rowid,body) VALUES(?,?)").run(i + 1, body));
  for (const query of ["alpha", '"alpha beta"', "alpha AND beta", "NEAR(alpha beta, 1)"]) {
    const expected = ordinary.prepare("SELECT rowid,rank FROM full WHERE full MATCH ? ORDER BY rank,rowid").all(query);
    const result = f.search(query, { mode: "fts", limit: 10 });
    assert.deepEqual(result.matches.map((match) => match.entryId), expected.map((row) => `entry-${row.rowid - 1}`));
    assert.deepEqual(result.matches.map((match) => match.rank), expected.map((row) => row.rank));
  }
  const expectedWords = ordinary.prepare('SELECT rowid,rank FROM full WHERE full MATCH ? ORDER BY rank,rowid').all('"alpha" AND "beta"');
  const actualWords = f.search("alpha beta").matches;
  assert.deepEqual(actualWords.map((match) => match.entryId), expectedWords.map((row) => `entry-${row.rowid - 1}`));
  assert.deepEqual(actualWords.map((match) => match.rank), expectedWords.map((row) => row.rank));
  const db = new DatabaseSync(f.databasePath);
  t.after(() => db.close());
  const schema = db.prepare("SELECT sql FROM sqlite_master WHERE name='postings'").get().sql;
  assert.match(schema, /content=''/);
  assert.match(schema, /contentless_delete=1/);
  assert.equal(db.prepare("SELECT name FROM sqlite_master WHERE name='postings_content'").get(), undefined);
  assert.equal(db.prepare("SELECT body FROM postings WHERE rowid=1").get().body, null);
  assert.equal(lstatSync(f.databasePath).mode & 0o777, 0o600);
  assert.equal(lstatSync(join(f.temporary, "cache")).mode & 0o777, 0o700);
});

test("indexes Unicode, unredacted thinking, tool arguments/results and summaries with source labels", (t) => {
  const f = fixture(t, [
    message("u", null, "user", "café 你好 λ"),
    message("a", "u", "assistant", [{ type: "thinking", thinking: "authoritative insight" },
      { type: "thinking", thinking: "redactedsecret", redacted: true }, { type: "text", text: "visible response" },
      { type: "toolCall", name: "lookup", arguments: { path: "unique-tool-argument" } }]),
    message("t", "a", "toolResult", [{ type: "text", text: "unique tool evidence" }, { type: "image", data: "hiddenimagepayload" }]),
    { type: "compaction", id: "c", parentId: "t", timestamp: stamp, summary: "checkpoint archive", firstKeptEntryId: "a" },
    message("empty", "c", "assistant", []),
  ]);
  f.index.sync([f.file]);
  assert.equal(f.search("cafe").matches[0].entryId, "u");
  assert.equal(f.search("你好").matches[0].entryId, "u");
  const insight = f.search("insight").matches[0];
  assert.equal(insight.role, "assistant");
  assert.deepEqual(insight.sourceKinds, ["thinking", "text", "toolCall"]);
  assert.equal(insight.text.includes("redactedsecret"), false);
  assert.equal(f.search("redactedsecret").matches.length, 0);
  assert.equal(f.search("unique-tool-argument").matches[0].entryId, "a");
  assert.equal(f.search("evidence", { roles: ["toolResult"] }).matches[0].entryId, "t");
  assert.equal(f.search("hiddenimagepayload").matches.length, 0);
  assert.equal(f.search("archive", { roles: ["summary"] }).matches[0].entryId, "c");
});

test("incremental appends defer partial JSONL lines and do not reparse unchanged files", (t) => {
  const f = fixture(t, [message("a", null, "user", "firstword")]);
  f.index.sync([f.file]);
  assert.equal(f.index.sync([f.file]).parsedEntries, 0);
  const next = JSON.stringify(message("b", "a", "assistant", "secondword"));
  appendFileSync(f.file, next.slice(0, -3));
  assert.equal(f.index.sync([f.file]).parsedEntries, 0);
  assert.equal(f.search("secondword").matches.length, 0);
  appendFileSync(f.file, next.slice(-3) + "\n");
  assert.equal(f.index.sync([f.file]).parsedEntries, 1);
  assert.equal(f.search("firstword").matches[0].entryId, "a");
  assert.equal(f.search("secondword").matches[0].entryId, "b");
  f.append(message("c", "b", "user", "thirdword"));
  assert.equal(f.index.sync([f.file]).parsedEntries, 1);
});

test("rewrite plus append, truncation and replacement discard obsolete postings", (t) => {
  const f = fixture(t, [message("a", null, "user", "obsoleteword")]);
  f.index.sync([f.file]);
  f.write([message("a", null, "user", "rewrittenword"), message("b", "a", "assistant", "appendedword")]);
  assert.equal(f.index.sync([f.file]).parsedEntries, 2);
  assert.equal(f.search("obsoleteword").matches.length, 0);
  assert.equal(f.search("rewrittenword").matches.length, 1);
  f.write([message("c", null, "user", "shortword")]);
  f.index.sync([f.file]);
  assert.equal(f.search("appendedword").matches.length, 0);
  const replacement = join(f.sourceRoot, "replacement.jsonl");
  f.write([message("d", null, "user", "replacementword")], replacement);
  renameSync(replacement, f.file);
  f.index.sync([f.file]);
  assert.equal(f.search("shortword").matches.length, 0);
  assert.equal(f.search("replacementword").matches[0].entryId, "d");
});

test("scoped sync preserves other sessions and removes genuinely deleted sources", (t) => {
  const f = fixture(t, [message("a", null, "user", "sharedword firstsession")]);
  const other = join(f.sourceRoot, "two.jsonl");
  f.write([message("b", null, "assistant", "sharedword othersession")], other, "session-two");
  f.index.sync([f.file, other]);
  f.index.sync([f.file]);
  assert.equal(f.index.search({ query: "sharedword" }).matches.length, 2);
  unlinkSync(other);
  assert.equal(f.index.sync([f.file]).removedFiles, 1);
  assert.equal(f.index.search({ query: "othersession" }).matches.length, 0);
});

test("default ancestors include compacted history while alternate branches require opt-in", (t) => {
  const f = fixture(t, [
    message("a", null, "user", "topic ancestor"),
    message("b", "a", "assistant", "topic abandoned"),
    message("c", "a", "assistant", "topic current"),
    { type: "compaction", id: "summary", parentId: "c", timestamp: stamp, summary: "topic checkpoint", firstKeptEntryId: "c" },
    message("d", "summary", "user", "topic newest"),
  ]);
  f.index.sync([f.file]);
  const current = f.search("topic", { limit: 10 });
  assert.deepEqual(new Set(current.matches.map((match) => match.entryId)), new Set(["a", "c", "summary", "d"]));
  const all = f.search("topic", { includeBranches: true, limit: 10 });
  assert.equal(all.matches.find((match) => match.entryId === "b").alternateBranch, true);
  assert.ok(all.matches.filter((match) => match.entryId !== "b").every((match) => !match.alternateBranch));
  assert.deepEqual(new Set(f.search("topic", { leafId: "b", limit: 10 }).matches.map((match) => match.entryId)), new Set(["a", "b"]));
  assert.deepEqual(f.index.search({ sessionFile: f.file, aroundEntryId: "summary", window: 1 }).matches.map((match) => match.entryId), ["c", "summary", "d"]);
  assert.deepEqual(f.index.search({ sessionFile: f.file, aroundEntryId: "summary", window: 0 }).matches.map((match) => match.entryId), ["summary"]);
  assert.deepEqual(f.index.search({ sessionFile: f.file, aroundEntryId: "b", includeBranches: true, window: 1 }).matches.map((match) => match.entryId), ["a", "b"]);
  assert.equal(f.search("topic", { leafId: "missing" }).warnings.length, 1);
});

test("owned historical-session hardlinks are searchable without duplicate project hits", (t) => {
  const f = fixture(t, [message("a", null, "user", "historicalword")]);
  const historical = join(f.sourceRoot, "historical.jsonl");
  linkSync(f.file, historical);
  assert.equal(lstatSync(historical).nlink, 2);
  f.index.sync([f.file, historical]);
  assert.equal(f.index.search({ query: "historicalword" }).matches.length, 1);
  assert.equal(f.search("historicalword").matches[0].entryId, "a");
  assert.equal(f.index.search({ query: "historicalword", sessionFile: historical }).matches[0].entryId, "a");
});

test("long messages retain all searchable words and excerpts center on the verified match", (t) => {
  const body = "initial text ".repeat(10000) + "needle-at-the-end café 😀";
  const f = fixture(t, [message("long", null, "assistant", body)]);
  f.index.sync([f.file]);
  for (const [query, mode] of [["needle", "words"], ['"needle at the end"', "fts"], ["needle-at-the-end", "literal"], ["cafe", "words"]]) {
    const result = f.search(query, { mode, maxChars: 80 });
    const match = result.matches[0];
    assert.ok(match.text.includes(query === "cafe" ? "café" : "needle"));
    assert.equal(match.text, body.slice(match.charStart, match.charEnd));
    assert.ok(match.charStart > 100000);
    assert.equal(result.truncated, true);
    assert.ok(match.text.length <= 80);
    assert.equal(match.text.includes("\uFFFD"), false);
  }
  assert.equal(f.search("--", { mode: "literal" }).matches.length, 0);
});

test("result count and total character budgets are enforced", (t) => {
  const entries = Array.from({ length: 12 }, (_, i) => message(`e${i}`, i ? `e${i - 1}` : null, "user", `common ${"padding ".repeat(20)} ${i}`));
  const f = fixture(t, entries);
  f.index.sync([f.file]);
  const result = f.search("common", { limit: 10, maxChars: 99 });
  assert.equal(result.matches.length, 10);
  assert.ok(result.matches.reduce((sum, match) => sum + match.text.length, 0) <= 99);
  assert.equal(result.truncated, true);
});

test("excerpts use actual whole-word or prefix hits rather than unrelated substrings", (t) => {
  const body = "concatenation ".repeat(10000) + "cat targetprefixword";
  const f = fixture(t, [message("long", null, "assistant", body)]);
  f.index.sync([f.file]);
  for (const [query, mode] of [["cat", "words"], ["targetprefix*", "fts"]]) {
    const match = f.search(query, { mode, maxChars: 60 }).matches[0];
    assert.ok(match.charStart > 100000);
    assert.ok(match.text.includes(query === "cat" ? "cat targetprefixword" : "targetprefixword"));
    assert.equal(match.text, body.slice(match.charStart, match.charEnd));
  }
});

test("hydration refuses modified sources and digest mismatches without exposing stale text", (t) => {
  const f = fixture(t, [message("a", null, "user", "oldsecret needle")]);
  f.index.sync([f.file]);
  f.write([message("a", null, "user", "newsecret needle")]);
  const stale = f.search("needle");
  assert.equal(stale.matches.length, 0);
  assert.ok(stale.stale > 0);
  assert.equal(JSON.stringify(stale).includes("oldsecret"), false);
  f.index.sync([f.file]);
  assert.ok(f.search("needle").matches[0].text.includes("newsecret"));
  const db = new DatabaseSync(f.databasePath);
  db.prepare("UPDATE entries SET sha256=?").run("0".repeat(64)); db.close();
  assert.equal(f.search("needle").matches.length, 0);
});

test("invalid arguments and malformed records fail without quoting transcript content", (t) => {
  const f = fixture(t, [message("a", null, "user", "validword")]);
  f.index.sync([f.file]);
  for (const options of [{ query: "!!!" }, { query: "x", mode: "sql" }, { query: "x", limit: 11 },
    { query: "x", maxChars: 16001 }, { query: "x", window: 4 }, { query: "x", roles: ["invalid"] },
    { query: "x", leafId: "a" }, { aroundEntryId: "a" }, { query: '"unterminated', mode: "fts" }]) {
    assert.throws(() => f.index.search(options));
  }
  appendFileSync(f.file, '{"privatecontents": BREAK_SECRET}\n');
  assert.throws(() => f.index.sync([f.file]), (error) => !error.message.includes("BREAK_SECRET") && /JSONL record at byte/.test(error.message));
  assert.equal(f.search("validword").matches.length, 0);
});

test("source and database symlinks, escapes and unsafe permissions are rejected", (t) => {
  const f = fixture(t, [message("a", null, "user", "needle")]);
  f.index.sync([f.file]);
  const link = join(f.sourceRoot, "link.jsonl"); symlinkSync(f.file, link);
  assert.throws(() => f.index.sync([link]));
  assert.throws(() => f.index.sync([join(f.temporary, "outside.jsonl")]));
  const directoryLink = join(f.sourceRoot, "directory-link"); symlinkSync(f.sourceRoot, directoryLink);
  assert.throws(() => f.index.sync([join(directoryLink, "one.jsonl")]));
  chmodSync(f.file, 0o666);
  f.index.sync([]);
  assert.equal(f.index.search({ query: "needle" }).matches.length, 0);
  const dbLink = join(f.temporary, "cache", "linked.sqlite"); symlinkSync(f.databasePath, dbLink);
  assert.throws(() => new TranscriptIndex({ databasePath: dbLink, sourceRoot: f.sourceRoot }));
  assert.throws(() => new TranscriptIndex({ databasePath: join(f.temporary, "cache", "other.sqlite"), sourceRoot: directoryLink }));
  f.index.close();
  assert.throws(() => f.index.search({ query: "needle" }), /closed/);
});

test("private session roots accept new SDK-style sources created with umask 0007", (t) => {
  const f = fixture(t);
  const file = join(f.sourceRoot, "sdk-default.jsonl");
  const previousUmask = process.umask(0o007);
  try { writeFileSync(file, serialize("sdk-session", [message("a", null, "user", "defaultpermissionword")])); }
  finally { process.umask(previousUmask); }
  assert.equal(lstatSync(f.sourceRoot).mode & 0o777, 0o700);
  assert.equal(lstatSync(file).mode & 0o777, 0o660);
  f.index.sync([file]);
  assert.equal(f.index.search({ query: "defaultpermissionword", sessionFile: file }).matches[0].entryId, "a");
});

test("group-writable sources require a private session root", (t) => {
  const f = fixture(t, [message("a", null, "user", "grouppermissionword")]);
  chmodSync(f.file, 0o660);
  f.index.sync([f.file]);
  assert.equal(f.search("grouppermissionword").matches.length, 1);
  chmodSync(f.sourceRoot, 0o755);
  assert.throws(() => f.index.sync([f.file]), /safe permissions/);
  f.index.sync([]);
  assert.equal(f.index.search({ query: "grouppermissionword" }).matches.length, 0);
});

test("world-writable sources remain rejected even within a private root", (t) => {
  const f = fixture(t, [message("a", null, "user", "worldpermissionword")]);
  chmodSync(f.file, 0o662);
  assert.throws(() => f.index.sync([f.file]), /safe permissions/);
  assert.equal(f.index.search({ query: "worldpermissionword" }).matches.length, 0);
});

test("multiple connections refresh transactionally and reopen the persistent contentless index", (t) => {
  const f = fixture(t, [message("a", null, "user", "firstword")]);
  const other = new TranscriptIndex({ databasePath: f.databasePath, sourceRoot: f.sourceRoot });
  t.after(() => other.close());
  f.index.sync([f.file]);
  assert.equal(other.search({ query: "firstword" }).matches.length, 1);
  f.append(message("b", "a", "assistant", "secondword"));
  other.sync([f.file]);
  assert.equal(f.search("secondword").matches.length, 1);
  assert.equal(readFileSync(f.file, "utf8").includes("secondword"), true);
});
