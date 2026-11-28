import assert from "node:assert/strict";
import fs from "node:fs";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { syncBuiltinESMExports } from "node:module";
import { availableParallelism, homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import { findPaths, searchContents, searchExecutable } from "../integrations/pi/qwen-dedicated-search.mjs";

const sdkRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const sdkAvailable = fs.existsSync(join(sdkRoot, "pi-coding-agent/dist/core/tools/index.js"));
const linuxSdk = { skip: process.platform !== "linux" || !sdkAvailable };
const sdk = async () => import(pathToFileURL(join(sdkRoot, "pi-coding-agent/dist/core/tools/index.js")));
async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "qwen-dedicated-search-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test("rg tool preserves actual Pi matches, truncation, overlapping and trailing context, encodings and line endings", linuxSdk, async t => {
  const root = await fixture(t), formatting = await sdk();
  const native = formatting.createGrepToolDefinition(root);
  const cases = [
    ["adjacent", "a\nHIT one\nHIT two\nb\n", [0, 1, 2], 100],
    ["disjoint", "HIT one\n" + "unmatched\n".repeat(15000) + "HIT two\n", [0, 1, 3], 2],
    ["last-only", "a\nHIT one\nHIT two\ncontext after matches\n", [1, 3], 1],
    ["cr-bom", "\ufeffline one\rline two\nHIT\r\nlast\r", [0, 1, 2], 3],
    ["bare-cr-at-chunk-end", "a".repeat(65535) + "\r\nHIT\nlast", [1], 1],
    ["invalid-utf8", Buffer.from([72, 73, 84, 255, 10, 122, 10]), [0, 1], 2],
    ["utf16", Buffer.from("\ufeffpre\nHIT\n", "utf16le"), [0, 2], 1],
    ["overlong", ("HIT" + "long".repeat(300) + "\n").repeat(115), [0, 1], 110],
    ["empty", "", [0, 1], 1],
    ["no-terminal-newline", "HIT", [1], 1],
  ];
  for (const [path, contents, contexts, limit] of cases) {
    await writeFile(join(root, path), contents);
    for (const context of contexts) {
      const args = { pattern: "hit", literal: true, ignoreCase: true, path, context, limit };
      const expected = await native.execute("oracle", args);
      assert.deepEqual(await searchContents("rg", args, root, undefined, formatting), expected, path + "/" + context);
    }
  }
  await assert.rejects(searchContents("rg", { path: ".", pattern: "[" }, root, undefined, formatting), /regex parse error/);
  await assert.rejects(searchContents("rg", { path: "missing", pattern: "x" }, root, undefined, formatting), /Path not found/);
});

test("one early hit with trailing context reads only a bounded prefix of a large file", linuxSdk, async t => {
  const root = await fixture(t), formatting = await sdk();
  const path = join(root, "large.txt");
  await writeFile(path, "needle\nsecond\nthird\n" + "0123456789012345678901234567890\n".repeat(262144));
  const openStream = fs.createReadStream;
  const reads = [];
  const mock = t.mock.method(fs, "createReadStream", (file, options) => {
    const stream = openStream(file, options);
    if (file === path) reads.push(stream);
    return stream;
  });
  syncBuiltinESMExports();
  t.after(() => { mock.mock.restore(); syncBuiltinESMExports(); });
  const args = { pattern: "needle", path, context: 2, limit: 1 };
  const actual = await searchContents("rg", args, root, undefined, formatting);
  assert.deepEqual(actual, await formatting.createGrepToolDefinition(root).execute("native", args));
  assert.equal(reads.length, 1);
  assert.ok(reads[0].bytesRead > 0 && reads[0].bytesRead <= 256 * 1024);
});

test("fd and rg choose workers without an extra executable invocation, keep Pi native outputs and handle aborts", linuxSdk, async t => {
  const root = await fixture(t), formatting = await sdk();
  const native = formatting.createFindToolDefinition(root);
  await mkdir(join(root, "src"));
  await mkdir(join(root, ".git"));
  await mkdir(join(root, "bin"));
  await writeFile(join(root, ".gitignore"), "ignored.txt\n");
  await writeFile(join(root, "ignored.txt"), "hit");
  await writeFile(join(root, "src/file.txt"), "hit");
  const binaries = {};
  for (const name of ["fd", "rg"]) {
    const real = searchExecutable(name, join(root, "empty"));
    assert.ok(real);
    // Managed binaries capture argv while the real tool still performs every search.
    const script = '#!/bin/sh\nprintf "%s\\n" "$*" >> "' + root + '/calls"\nexec "' + real + '" "$@"\n';
    await writeFile(join(root, "bin", name), script, { mode: 0o700 });
    binaries[name] = searchExecutable(name, root);
    assert.equal(binaries[name], join(root, "bin", name));
  }
  const savedAgentDir = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = root;
  t.after(() => {
    if (savedAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = savedAgentDir;
  });
  const { DefaultResourceLoader } = await import(pathToFileURL(join(sdkRoot, "pi-coding-agent/dist/core/resource-loader.js")));
  const { SettingsManager } = await import(pathToFileURL(join(sdkRoot, "pi-coding-agent/dist/core/settings-manager.js")));
  const loader = new DefaultResourceLoader({ cwd: root, agentDir: root, settingsManager: SettingsManager.inMemory(),
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [fileURLToPath(new URL("../integrations/pi/qwen-tool-names.ts", import.meta.url))] });
  await loader.reload();
  const loaded = loader.getExtensions();
  assert.deepEqual(loaded.errors, []);
  const invoke = (name, args) => loaded.extensions[0].tools.get(name).definition.execute(
    "test", args, undefined, undefined, { cwd: root, model: { provider: "qwen-r9700" } });
  for (const pattern of ["src/*.txt", "file.txt", "missing", "src/"]) {
    const args = { pattern, limit: 1 };
    assert.deepEqual(await invoke("find_files", args), await native.execute("native", args));
  }
  const ctl = new AbortController(); ctl.abort();
  await assert.rejects(findPaths(binaries.fd, { pattern: "*" }, root, ctl.signal, formatting), /Operation aborted/);
  await assert.rejects(searchContents(binaries.rg, { pattern: "x" }, root, ctl.signal, formatting), /Operation aborted/);
  await invoke("search_file_contents", { pattern: "hit", path: "src/file.txt" });
  await invoke("search_file_contents", { pattern: "hit", path: "src" });
  await invoke("search_file_contents", { pattern: "hit", path: "src", files_only: true });
  const calls = fs.readFileSync(join(root, "calls"), "utf8").trimEnd().split("\n");
  assert.equal(calls.length, 7);
  assert.ok(calls.slice(0, 2).every(c => c.startsWith("--threads 1 ")));
  assert.ok(calls.slice(2, 4).every(c => c.startsWith("--threads ")));
  assert.ok(calls[4].startsWith("--threads 1 "));
  assert.ok(calls.slice(5).every(c => c.startsWith("--threads " + Math.min(4, availableParallelism()) + " ")));
  assert.ok(calls.every(c => !c.includes("--version")));
});
