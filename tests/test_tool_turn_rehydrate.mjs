import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { chmodSync, existsSync, linkSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import rehydrate from "../integrations/pi/qwen-tool-turn-rehydrate.mjs";

function fixture(t, body, historical = false) {
  const root = mkdtempSync(join(tmpdir(), "rehydrate-contract-"));
  const variables = ["QWEN_PI_TOOL_RESULT_DIR", "QWEN_PI_TOOL_TURN_ARCHIVE_ROOT"];
  const previous = variables.map((key) => process.env[key]);
  t.after(() => {
    variables.forEach((key, index) => previous[index] === undefined ? delete process.env[key] : process.env[key] = previous[index]);
    rmSync(root, { recursive: true, force: true });
  });
  const live = join(root, "live"), history = join(root, "history");
  process.env[variables[0]] = live;
  process.env[variables[1]] = history;
  const data = historical ? JSON.stringify({ archive_schema: "qwen-pi-archived-tool-turn-v1", tool_name: "fixture",
    tool_call_id: "original-call", tool_result_entry: { message: { role: "toolResult", content: [{ type: "text", text: body }] } } }) : body;
  const digest = createHash("sha256").update(data).digest("hex");
  const directory = join(historical ? history : live, "sha256", digest.slice(0, 2));
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  const path = join(directory, `${digest}.${historical ? "json" : "txt"}`);
  writeFileSync(path, data, { mode: 0o400 });
  let tool;
  rehydrate({ registerTool(value) { tool = value; } });
  return { root, digest, directory, path, tool, data, call: (params = {}, signal) => tool.execute("fixture-call", { sha256: digest, ...params }, signal) };
}

const textOf = (result) => result.content[0].text;

test("a one-line literal budget returns the actual match rather than preceding context", async (t) => {
  const f = fixture(t, "before-1\nbefore-2\nbefore-3\nEXACT-NEEDLE\nafter-1\n");
  const result = await f.call({ pattern: "EXACT-NEEDLE", max_lines: 1 });
  assert.deepEqual(result.details.selectedLines, [4]);
  assert.match(textOf(result), /4: EXACT-NEEDLE/);
  assert.equal(result.details.selectionTruncated, true);
});

test("literal matches take priority over neighboring context and stay in source order", async (t) => {
  const f = fixture(t, "before\nneedle first\nbetween\nneedle second\nafter\n");
  const result = await f.call({ pattern: "NEEDLE", context: 20, max_lines: 2 });
  assert.deepEqual(result.details.selectedLines, [2, 4]);
  assert.match(textOf(result), /2: needle first/);
  assert.match(textOf(result), /4: needle second/);
});

test("a literal at the end of a long Unicode line survives the byte budget", async (t) => {
  const f = fixture(t, "prefix-" + "🦜".repeat(12000) + "TAIL-NEEDLE\n");
  const result = await f.call({ pattern: "TAIL-NEEDLE", max_lines: 1, context: 0 });
  assert.ok(textOf(result).includes("TAIL-NEEDLE"), "the bounded line excerpt must retain the requested match");
  assert.ok(Buffer.byteLength(textOf(result)) <= 32768);
  assert.ok(!textOf(result).includes("\uFFFD"));
  assert.equal(result.details.outputTruncated, true);
  assert.deepEqual(result.details.returnedLines, [1]);
  assert.deepEqual(result.details.clippedLines, [1]);
  assert.equal(readFileSync(f.path, "utf8"), f.data);
});

test("long context cannot crowd the matching line out of a bounded response", async (t) => {
  const f = fixture(t, "x".repeat(50000) + "\n" + "y".repeat(50000) + "\nimportant needle\nafter\n");
  const result = await f.call({ pattern: "needle", context: 2 });
  assert.ok(textOf(result).includes("3: important needle"), "long neighboring lines must not hide the requested match");
  assert.ok(Buffer.byteLength(textOf(result)) <= 32768);
  assert.deepEqual(result.details.returnedLines, [1, 2, 3, 4]);
});

test("preview, range and no-match preserve their declared boundaries", async (t) => {
  const f = fixture(t, Array.from({ length: 30 }, (_, index) => `row-${index + 1}`).join("\n") + "\n");
  const preview = await f.call();
  assert.deepEqual(preview.details.selectedLines, Array.from({ length: 12 }, (_, index) => index + 1));
  assert.ok(Buffer.byteLength(textOf(preview)) <= 1536);
  assert.equal(preview.details.selectionTruncated, true);
  const range = await f.call({ start_line: 27, end_line: 30 });
  assert.deepEqual(range.details.selectedLines, [27, 28, 29, 30]);
  assert.equal(range.details.selectionTruncated, false);
  const absent = await f.call({ pattern: "absent", context: 0 });
  assert.deepEqual(absent.details.selectedLines, []);
  assert.match(textOf(absent), /No literal matches/);
  for (const params of [{ start_line: 0 }, { end_line: 31 }, { start_line: 5, end_line: 4 }, { pattern: "" }, { context: 21 }, { max_lines: 201 }]) {
    await assert.rejects(f.call(params));
  }
});

test("historical JSON works without a live store and retains original tool identity", async (t) => {
  const f = fixture(t, "historical one\nhistorical two\n", true);
  const result = await f.call({ start_line: 2, end_line: 2 });
  assert.equal(result.details.archiveKind, "historical");
  assert.match(textOf(result), /Tool: fixture\nTool call: original-call/);
  assert.match(textOf(result), /2: historical two/);
  assert.equal(readFileSync(f.path, "utf8"), f.data);
});

test("corrupt, writable, symlinked and unexplained hardlinked archives are refused", async (t) => {
  const f = fixture(t, "authenticated evidence\n");
  chmodSync(f.path, 0o600);
  await assert.rejects(f.call(), /mode/);
  writeFileSync(f.path, "tampered evidence\n");
  chmodSync(f.path, 0o400);
  await assert.rejects(f.call(), /SHA-256 mismatch/);
  rmSync(f.path);
  const alternate = join(f.root, "alternate");
  writeFileSync(alternate, f.data, { mode: 0o400 });
  symlinkSync(alternate, f.path);
  await assert.rejects(f.call(), /non-symlink/);
  rmSync(f.path);
  linkSync(alternate, f.path);
  await assert.rejects(f.call(), /exactly one link/);
});

test("pre-cancelled retrieval does not touch the archive", async (t) => {
  const f = fixture(t, "untouched evidence\n");
  const controller = new AbortController(); controller.abort();
  await assert.rejects(f.call({}, controller.signal), /cancelled/);
  assert.equal(readFileSync(f.path, "utf8"), f.data);
});

test("empty archived results contain zero lines rather than an invented blank line", async (t) => {
  const f = fixture(t, "", true);
  const result = await f.call();
  assert.match(textOf(result), /Original result: 0 lines \/ 0 bytes/);
  assert.deepEqual(result.details.selectedLines, []);
  assert.deepEqual(result.details.returnedLines, []);
});

test("case-folded matches retain their correct position after expanding Unicode characters", async (t) => {
  const f = fixture(t, "İ".repeat(20000) + "TaIl NeEdLe\n");
  const result = await f.call({ pattern: "tail needle", context: 0, max_lines: 1 });
  assert.ok(textOf(result).toLocaleLowerCase("en-US").includes("tail needle"), "the excerpt must use original-text positions after case folding");
  assert.ok(Buffer.byteLength(textOf(result)) <= 32768);
  assert.ok(!textOf(result).includes("\uFFFD"));
});

test("line and byte budget combinations preserve matching anchors without changing the archive", async (t) => {
  const matches = [6, 57, 300];
  const body = Array.from({ length: 320 }, (_, index) => matches.includes(index + 1)
    ? "🦜".repeat(9000) + `MATCH-${index + 1}` : `context-${index + 1}`).join("\n") + "\n";
  const f = fixture(t, body);
  for (const max_lines of [1, 2, 3, 5, 80, 200]) for (const context of [0, 1, 20]) {
    const result = await f.call({ pattern: "MATCH-", max_lines, context, start_line: 4, end_line: 310 });
    assert.ok(result.details.selectedLines.length <= max_lines);
    assert.deepEqual([...new Set(result.details.selectedLines)].sort((a, b) => a - b), result.details.selectedLines);
    for (const line of matches.slice(0, max_lines)) {
      assert.ok(result.details.selectedLines.includes(line), `line budget ${max_lines} must select matching line ${line}`);
      assert.ok(result.details.returnedLines.includes(line), `byte budget must return matching line ${line}`);
      assert.ok(textOf(result).includes(`MATCH-${line}`), `rendered excerpt must include match ${line}`);
    }
    assert.ok(result.details.selectedLines.every((line) => line >= 4 && line <= 310));
    assert.ok(Buffer.byteLength(textOf(result)) <= 32768);
    assert.ok(!textOf(result).includes("\uFFFD"));
  }
  assert.equal(readFileSync(f.path, "utf8"), body);
});

test("recovery removes only the authenticated publisher's interrupted temporary link", async (t) => {
  const f = fixture(t, "durable evidence\n");
  const temporary = join(f.directory, `.${f.digest}.999.123e4567-e89b-42d3-a456-426614174000.tmp`);
  linkSync(f.path, temporary);
  const result = await f.call({ start_line: 1, end_line: 1 });
  assert.match(textOf(result), /1: durable evidence/);
  assert.equal(existsSync(temporary), false);
  assert.equal(statSync(f.path).nlink, 1);
  assert.equal(statSync(f.path).mode & 0o777, 0o400);
  assert.equal(readFileSync(f.path, "utf8"), f.data);
});

test("malformed historical JSON errors do not quote archive contents", async (t) => {
  const f = fixture(t, "valid historical body", true);
  const malformed = "SYNTHETIC_PRIVATE_SENTINEL is not JSON";
  const digest = createHash("sha256").update(malformed).digest("hex");
  const directory = join(f.root, "history", "sha256", digest.slice(0, 2));
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  writeFileSync(join(directory, `${digest}.json`), malformed, { mode: 0o400 });
  await assert.rejects(f.tool.execute("bad-json", { sha256: digest }), (error) => {
    assert.equal(error.message, "archived tool-turn JSON is invalid");
    assert.ok(!error.message.includes("SYNTHETIC"));
    return true;
  });
});
