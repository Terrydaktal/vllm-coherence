import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { existsSync } from "node:fs";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { searchFilenames, withFilesOnlySearch } from "../integrations/pi/qwen-files-only-search.mjs";

const textOf = (result) => result.content.map((item) => item.text ?? "").join("\n");
async function fixture(t) {
  const cwd = await mkdtemp(join(tmpdir(), "qwen-files-only-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  await mkdir(join(cwd, ".git"));
  await writeFile(join(cwd, ".gitignore"), "ignored.txt\n");
  await writeFile(join(cwd, "alpha.txt"), "Emma\nEmma\n");
  await writeFile(join(cwd, "beta.txt"), "emma\n");
  await writeFile(join(cwd, "gamma.md"), "Emma\n");
  await writeFile(join(cwd, ".hidden.txt"), "Emma\n");
  await writeFile(join(cwd, "ignored.txt"), "Emma\n");
  return cwd;
}

test("filenames-only uses native hidden/ignore/glob/literal/case semantics and returns each file once", async (t) => {
  const cwd = await fixture(t);
  const result = await searchFilenames({ pattern: "emma", ignoreCase: true, literal: true, context: 10 }, { cwd });
  assert.deepEqual(textOf(result).split("\n").sort(), [".hidden.txt", "alpha.txt", "beta.txt", "gamma.md"]);
  assert.equal(result.details.returnedFiles, 4);
  // Native rg --glob is an explicit whitelist and can override .gitignore.
  const globbed = await searchFilenames({ pattern: "emma", ignoreCase: true, literal: true, glob: "*.txt" }, { cwd });
  assert.deepEqual(textOf(globbed).split("\n").sort(), [".hidden.txt", "alpha.txt", "beta.txt", "ignored.txt"]);
  assert.equal(result.details.exitCode, 0);
  assert.equal(textOf(await searchFilenames({ pattern: "Emma", path: "alpha.txt", literal: true }, { cwd })), "alpha.txt");
  assert.equal(textOf(await searchFilenames({ pattern: "Emma", path: "@alpha.txt", literal: true }, { cwd })), "alpha.txt");
  assert.equal(textOf(await searchFilenames({ pattern: "Emma", path: pathToFileURL(join(cwd, "alpha.txt")).href }, { cwd })), "alpha.txt");
  await writeFile(join(cwd, "literal.txt"), "a.b\n");
  assert.equal(textOf(await searchFilenames({ pattern: "a.b", literal: true, path: "literal.txt" }, { cwd })), "literal.txt");
  const missing = await searchFilenames({ pattern: "missing", literal: true }, { cwd });
  assert.equal(textOf(missing), "No matches found");
  assert.equal(missing.details.exitCode, 1, "no-match status remains distinct from rg errors");
  await assert.rejects(searchFilenames({ pattern: "[", path: "alpha.txt" }, { cwd }), /regex parse error/i);
  await assert.rejects(searchFilenames({ pattern: "Emma", path: "not-found" }, { cwd }), /Path not found/);
});

test("filenames-only has bounded output and escapes newline filenames without leaking matched contents", async (t) => {
  const cwd = await fixture(t);
  const capped = await searchFilenames({ pattern: "Emma", limit: 1 }, { cwd });
  assert.equal(capped.details.returnedFiles, 1);
  assert.equal(capped.details.matchLimitReached, 1);
  assert.match(textOf(capped), /files limit reached/);
  await mkdir(join(cwd, "weird"));
  await writeFile(join(cwd, "weird", "two\nlines.txt"), "Emma private match contents\n");
  const weird = await searchFilenames({ pattern: "Emma", path: "weird" }, { cwd });
  assert.equal(textOf(weird), '"two\\nlines.txt"');
  assert.ok(!textOf(weird).includes("private match contents"));
  await mkdir(join(cwd, "many"));
  await Promise.all(Array.from({ length: 300 }, (_, index) => writeFile(join(cwd, "many", `${index}-${"x".repeat(195)}.txt`), "match\n")));
  const bytes = await searchFilenames({ pattern: "match", path: "many", limit: 1000 }, { cwd });
  assert.equal(bytes.details.truncation.truncated, true);
  assert.ok(Buffer.byteLength(textOf(bytes).split("\n\n[")[0]) <= 50 * 1024);
  await assert.rejects(searchFilenames({ pattern: "Emma", limit: 0 }, { cwd }), /positive integer/);
});

test("deadline and abort stop the spawned search and cannot report false completion", async (t) => {
  const cwd = await fixture(t);
  const children = [];
  const spawnProcess = () => {
    const child = new EventEmitter();
    child.stdout = new PassThrough(); child.stderr = new PassThrough();
    child.kill = (signal) => { child.killedWith = signal; queueMicrotask(() => child.emit("close", null)); };
    children.push(child); return child;
  };
  await assert.rejects(searchFilenames({ pattern: "Emma" }, { cwd, timeoutMs: 10, spawnProcess }), /exceeded/);
  assert.equal(children[0].killedWith, "SIGKILL");
  const controller = new AbortController();
  const pending = searchFilenames({ pattern: "Emma" }, { cwd, signal: controller.signal, spawnProcess });
  await new Promise((resolve) => setTimeout(resolve, 10));
  controller.abort();
  await assert.rejects(pending, /Operation aborted/);
  assert.equal(children[1].killedWith, "SIGKILL");
  await assert.rejects(searchFilenames({ pattern: "Emma" }, { cwd, signal: controller.signal, spawnProcess }), /Operation aborted/);
  assert.equal(children.length, 2, "an already-aborted call never spawns a search");
  const failure = searchFilenames({ pattern: "Emma" }, { cwd, spawnProcess });
  await new Promise((resolve) => setTimeout(resolve, 10));
  children[2].stdout.write(Buffer.from(join(cwd, "alpha.txt") + "\0"));
  children[2].stderr.write("permission denied on another file");
  children[2].emit("close", 2);
  await assert.rejects(failure, /permission denied/, "partial matches do not conceal a search error");
});

const sdkRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
test("schema extension validates with installed TypeBox and default execution still delegates unchanged", {
  skip: !existsSync(join(sdkRoot, "pi-coding-agent/dist/core/tools/grep.js")),
}, async (t) => {
  const cwd = await fixture(t);
  const { createGrepToolDefinition } = await import(pathToFileURL(join(sdkRoot, "pi-coding-agent/dist/core/tools/grep.js")));
  const { validateToolArguments } = await import(pathToFileURL(join(sdkRoot, "pi-ai/dist/utils/validation.js")));
  const native = createGrepToolDefinition(cwd);
  const tool = withFilesOnlySearch(native, { cwd });
  const { files_only, ...properties } = tool.parameters.properties;
  assert.deepEqual(properties, native.parameters.properties);
  assert.deepEqual(tool.parameters.required, native.parameters.required);
  const { description: _description, ...filesOnlyType } = files_only;
  const { description: _nativeDescription, ...nativeType } = native.parameters.properties.literal;
  assert.deepEqual(filesOnlyType, nativeType, "native TypeBox Boolean/Optional metadata survives");
  for (const key of Reflect.ownKeys(native.parameters.properties.literal)) {
    if (key !== "description") assert.equal(files_only[key], native.parameters.properties.literal[key]);
  }
  for (const key of Reflect.ownKeys(native.parameters)) {
    if (key !== "properties") assert.equal(tool.parameters[key], native.parameters[key]);
  }
  validateToolArguments(tool, { name: "grep", arguments: { pattern: "Emma", files_only: true } });
  validateToolArguments(tool, { name: "grep", arguments: { pattern: "Emma" } });
  assert.throws(() => validateToolArguments(tool, { name: "grep", arguments: { pattern: "Emma", files_only: "yes" } }), /boolean/);
  const args = { pattern: "Emma", path: "alpha.txt" };
  assert.deepEqual(await tool.execute("default", args), await native.execute("native", args));
  const result = await tool.execute("files-only", { ...args, files_only: true });
  assert.equal(textOf(result), "alpha.txt");
  assert.equal(result.details.filesOnly, true);
  const elsewhere = await mkdtemp(join(tmpdir(), "qwen-files-only-ctx-"));
  t.after(() => rm(elsewhere, { recursive: true, force: true }));
  await writeFile(join(elsewhere, "session-cwd.txt"), "Emma\n");
  assert.equal(textOf(await tool.execute("ctx", { pattern: "Emma", files_only: true }, undefined, undefined, { cwd: elsewhere })), "session-cwd.txt");
});
