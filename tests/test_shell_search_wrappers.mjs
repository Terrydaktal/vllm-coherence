// Exercise the deployed combination, not just either standalone planner.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

// A qualification build can test an exact disposable runtime copy instead of
// overwriting the binaries used by already installed Pi sessions.
const sourceRoot = resolve(process.env.COHERENCE_SEARCH_TEST_ROOT ?? fileURLToPath(new URL("../", import.meta.url)));
const { withSearchWrappers } = await import(pathToFileURL(join(sourceRoot, "integrations/pi/qwen-shell-search-wrappers.mjs")));

const env = { ...process.env, LC_ALL: "C" };
delete env.GREP2RG_EXPERIMENTAL_RG;
delete env.FIND2FD_EXPERIMENTAL_FD;
const execute = (command, cwd, wrapped = true) => {
  const context = { command, env };
  const source = wrapped ? withSearchWrappers(context, "/usr/bin/bash").command : command;
  const result = spawnSync("/usr/bin/bash", ["--noprofile", "--norc", "-c", source],
    { cwd, env, encoding: "utf8", timeout: 10000, maxBuffer: 4 * 1024 * 1024 });
  assert.equal(result.error, undefined);
  return { status: result.status, signal: result.signal, stdout: result.stdout, stderr: result.stderr };
};

test("combined search wrappers preserve GNU bytes, diagnostics, status and Bash scope", t => {
  const cwd = mkdtempSync(join(tmpdir(), "coherence-search-wrappers-"));
  t.after(() => rmSync(cwd, { recursive: true, force: true }));
  mkdirSync(join(cwd, "tree")); mkdirSync(join(cwd, "tree", ".hidden"));
  writeFileSync(join(cwd, "tree", "a.txt"), "alpha\nbeta 12\nalpha two\n");
  writeFileSync(join(cwd, "tree", ".hidden", "space name.txt"), "beta 34\n");
  writeFileSync(join(cwd, "tree", "binary"), Buffer.from("alpha\0tail\n"));
  for (const command of [
    "grep -nF alpha tree/a.txt",
    "grep -rn -E 'alpha|beta' tree",
    "grep -r -I -c beta tree",
    "grep --color=always -n alpha tree/a.txt",
    "grep -lF alpha tree/binary tree/a.txt",
    "grep -LF absent tree/a.txt",
    "grep -q alpha missing tree/a.txt",
    "grep --does-not-exist alpha tree/a.txt",
    "printf 'alpha\\nbeta 12\\n' | grep -E '[0-9]+'",
    "grep -F -f <(printf 'alpha\\n') tree/a.txt",
    "find tree -type f -name '*.txt'",
    "find tree -type f -printf '%p:%s\\n'",
    "find tree -name .hidden -prune -o -type f -print",
    "find tree -maxdepth 1 -type f -exec wc -c '{}' +",
    "find missing -print",
    "find tree -type invalid",
    "set -eu; grep -F alpha tree/a.txt; find tree -name a.txt",
    "command grep -nF alpha tree/a.txt; command find tree -name a.txt",
    "/usr/bin/grep -nF alpha tree/a.txt; /usr/bin/find tree -name a.txt",
    "bash --noprofile --norc -c 'type -t grep; type -t find'",
  ]) assert.deepEqual(execute(command, cwd), execute(command, cwd, false), command);
  assert.equal(execute("type -t grep; type -t find", cwd).stdout, "function\nfunction\n");
  assert.equal(execute("type -t grep; type -t find", cwd, false).stdout, "file\nfile\n");
  assert.deepEqual(withSearchWrappers({ command: "grep alpha file" }, "/usr/bin/fish"),
    { command: "grep alpha file" });
});

test("pinned wrapper runtime has hashes for every bundled source", () => {
  const root = pathToFileURL(join(sourceRoot, "integrations/pi/search-wrappers/"));
  const manifest = JSON.parse(readFileSync(new URL("source-manifest.json", root), "utf8"));
  assert.equal(manifest.schema, "coherence-search-wrapper-source-v1");
  for (const name of ["grep2rg", "find2fd"]) {
    assert.ok(manifest.packages[name].files["src/" + (name === "grep2rg" ? "grep" : "find") + "-wrapper.mjs"]);
    assert.ok(manifest.packages[name].files["bin/" + name]);
    assert.ok(manifest.packages[name].files.LICENSE);
    for (const [path, digest] of Object.entries(manifest.packages[name].files)) {
      const data = readFileSync(new URL(`${name}/${path}`, root));
      assert.equal(createHash("sha256").update(data).digest("hex"), digest, `${name}/${path}`);
    }
  }
});

test("explicit legacy C Pi hook really accelerates a qualified search", t => {
  const cwd = mkdtempSync(join(tmpdir(), "coherence-search-auto-"));
  t.after(() => rmSync(cwd, { recursive: true, force: true }));
  mkdirSync(join(cwd, "tree"));
  const payload = Buffer.from("plain filler line\n".repeat(Math.ceil(256 * 1024 / 18))).subarray(0, 256 * 1024);
  payload.write("\nalpha\n", 128 * 1024); payload[payload.length - 1] = 10;
  for (let i = 0; i < 256; i++) writeFileSync(join(cwd, "tree", `part-${i}.txt`), payload);
  const command = "LC_ALL=C GREP2RG_ENGINE=c GREP2RG_ROUTE_LOG=route.log grep -raFc alpha tree";
  assert.deepEqual(execute(command, cwd), execute(command, cwd, false));
  assert.match(readFileSync(join(cwd, "route.log"), "utf8"), /rg-native files=256 bytes=67108864 mode=count/,
    "a native GNU-only default would pass the oracle but must fail route qualification");
});

test("legacy C long-record searches retain the GNU route", t => {
  const cwd = mkdtempSync(join(tmpdir(), "coherence-search-long-records-"));
  t.after(() => rmSync(cwd, { recursive: true, force: true }));
  mkdirSync(join(cwd, "tree"));
  const payload = Buffer.alloc(256 * 1024, 120);
  payload.write("\nalpha\n", 128 * 1024); payload[payload.length - 1] = 10;
  for (let i = 0; i < 256; i++) writeFileSync(join(cwd, "tree", `part-${i}.txt`), payload);
  for (const args of ["-raFc", "-ranF"]) {
    const command = `LC_ALL=C GREP2RG_ENGINE=c GREP2RG_ROUTE_LOG=route.log grep ${args} alpha tree`;
    assert.deepEqual(execute(command, cwd), execute(command, cwd, false));
    assert.equal(existsSync(join(cwd, "route.log")), false,
      "large files with long records must not enter the slower parallel route");
  }
});

test("explicit JavaScript Pi experiment invokes real fd and preserves GNU order", t => {
  const cwd = mkdtempSync(join(tmpdir(), "coherence-find-auto-"));
  t.after(() => rmSync(cwd, { recursive: true, force: true }));
  const discovery = spawnSync("/usr/bin/bash", ["--noprofile", "--norc", "-c", "command -v fd || command -v fdfind"],
    { env, encoding: "utf8", timeout: 10000 });
  assert.equal(discovery.error, undefined);
  assert.equal(discovery.status, 0, "the installed release must provide real fd or fdfind; route qualification cannot be skipped");
  const realFd = discovery.stdout.trim();
  assert.ok(realFd.startsWith("/"), "record the real executable before placing the owned recorder on PATH");

  const bin = join(cwd, "bin"), trace = join(cwd, "fd-calls.jsonl");
  mkdirSync(bin);
  const recorder = `#!${process.execPath}
const { appendFileSync } = require("node:fs");
const { spawnSync } = require("node:child_process");
const args = process.argv.slice(2);
appendFileSync(process.env.COHERENCE_FD_TRACE, JSON.stringify(args) + "\\n");
const child = spawnSync(process.env.COHERENCE_REAL_FD, args, { stdio: "inherit" });
if (child.error) throw child.error;
process.exit(child.status ?? 1);
`;
  for (const name of ["fd", "fdfind"]) writeFileSync(join(bin, name), recorder, { mode: 0o755 });

  mkdirSync(join(cwd, "layered"));
  for (let top = 0; top < 64; top++) {
    const parent = join(cwd, "layered", `top-${top}`); mkdirSync(parent);
    for (let child = 0; child < 400; child++) {
      const directory = join(parent, `child-${child}`); mkdirSync(directory);
      writeFileSync(join(directory, "common.txt"), "owned\n");
      if (top % 17 === 0 && child === 0) writeFileSync(join(directory, "rare-result.dat"), "owned\n");
    }
  }
  const testEnv = { ...env, PATH: `${bin}:${env.PATH}`, COHERENCE_REAL_FD: realFd, COHERENCE_FD_TRACE: trace };
  testEnv.FIND2FD_EXPERIMENTAL_FD = "1";
  delete testEnv.FIND2FD_ACCELERATION;
  const calls = () => readFileSync(trace, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse)
    .filter(argv => !argv.includes("--version") && !argv.includes("--help"));
  const invoke = (command, wrapped) => {
    const source = wrapped ? withSearchWrappers({ command, env: testEnv }, "/usr/bin/bash").command : command;
    const result = spawnSync("/usr/bin/bash", ["--noprofile", "--norc", "-c", source],
      { cwd, env: testEnv, timeout: 20000, maxBuffer: 4 * 1024 * 1024 });
    assert.equal(result.error, undefined);
    return { status: result.status, signal: result.signal, stdout: result.stdout, stderr: result.stderr };
  };
  writeFileSync(trace, "");
  for (const command of [
    "find layered -name rare-result.dat -print0",
    "find layered -name 'rare*.dat' -print",
    "find -P layered -type f -name rare-result.dat -print0",
    "find layered -mindepth 1 -maxdepth 3 -name never-a-match -print0",
  ]) {
    const expected = invoke(command, false), before = calls().length;
    assert.equal(expected.status, 0);
    assert.deepEqual(invoke(command, true), expected, `complete GNU bytes, diagnostics, status and order: ${command}`);
    assert.equal(calls().length - before, 1, "the explicit JavaScript experiment must actually invoke real fd");
  }
  t.diagnostic("25,600 owned directories; four GNU byte/order comparisons; four explicit experimental fd searches");
});

test("default combined model hook launches both Rust engines without Node, fd or rg", t => {
  const cwd = mkdtempSync(join(tmpdir(), "coherence-search-rust-"));
  t.after(() => rmSync(cwd, { recursive: true, force: true }));
  mkdirSync(join(cwd, "tree"));
  writeFileSync(join(cwd, "tree", "a.txt"), "alpha\nbeta\nalpha\n");
  const command = "grep -aFn alpha tree/a.txt; find tree -type f -name '*.txt' -print0";
  const context = withSearchWrappers({ command, env }, "/usr/bin/bash");
  const trace = join(cwd, "exec.trace");
  const observed = spawnSync("/usr/bin/strace", ["-f", "-qq", "-e", "execve", "-o", trace,
    "/usr/bin/bash", "--noprofile", "--norc", "-c", context.command],
  { cwd, env, encoding: "utf8", timeout: 10000 });
  assert.equal(observed.error, undefined);
  assert.deepEqual({ status: observed.status, signal: observed.signal, stdout: observed.stdout, stderr: observed.stderr },
    execute(command, cwd, false));
  const executions = readFileSync(trace, "utf8");
  assert.match(executions, /rust\/target\/release\/grep2rg-rust/);
  assert.match(executions, /rust\/target\/release\/find2fd-rust/);
  assert.doesNotMatch(executions, /execve\("[^"\n]*\/(?:node|fd|fdfind|rg|grep2rg-native)"/);
  assert.equal(execute("type -t grep; type -t find", cwd, false).stdout, "file\nfile\n");
});
