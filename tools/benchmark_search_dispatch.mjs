#!/usr/bin/env node
// Shipping Pi shell hooks, owned fixtures, GNU oracle, full entry overhead.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { cpus, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const source = process.argv[2] ? resolve(process.argv[2]) : fileURLToPath(new URL("../", import.meta.url));
const { withSearchWrappers } = await import(pathToFileURL(join(source, "integrations/pi/qwen-shell-search-wrappers.mjs")));
const cwd = mkdtempSync(join(tmpdir(), "coherence-dispatch-speed-"));
const env = { ...process.env, LC_ALL: "C", LANG: "C" };
for (const name of ["GREP2RG_EXPERIMENTAL_RG", "FIND2FD_EXPERIMENTAL_FD", "GREP2RG_ACCELERATION", "FIND2FD_ACCELERATION", "GREP2RG_ROUTE_LOG", "POSIXLY_CORRECT"]) delete env[name];
const run = (command, automatic) => {
  const began = process.hrtime.bigint();
  const context = automatic ? withSearchWrappers({ command, env }, "/usr/bin/bash") : { command, env };
  const result = spawnSync("/usr/bin/bash", ["--noprofile", "--norc", "-c", context.command],
    { cwd, env: context.env, timeout: 20000, maxBuffer: 16 * 1024 * 1024 });
  assert.equal(result.error, undefined, "search did not finish normally");
  assert.equal(result.signal, null);
  return { result, ms: Number(process.hrtime.bigint() - began) / 1e6 };
};
const equal = (actual, expected) => {
  assert.equal(actual.status, expected.status); assert.deepEqual(actual.stdout, expected.stdout); assert.deepEqual(actual.stderr, expected.stderr);
};
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
try {
  mkdirSync(join(cwd, "corpus")); mkdirSync(join(cwd, "long-lines")); mkdirSync(join(cwd, "layered"));
  writeFileSync(join(cwd, "small.txt"), "alpha\nbeta\n");
  const body = Buffer.from("plain filler line\n".repeat(Math.ceil(256 * 1024 / 18))).subarray(0, 256 * 1024);
  body.write("\nalpha\n", 128 * 1024); body[body.length - 1] = 10;
  const longBody = Buffer.alloc(256 * 1024, 120); longBody.write("\nalpha\n", 128 * 1024); longBody[longBody.length - 1] = 10;
  for (let i = 0; i < 256; i++) {
    writeFileSync(join(cwd, "corpus", `part-${i}.txt`), body);
    writeFileSync(join(cwd, "long-lines", `part-${i}.txt`), longBody);
  }
  for (let i = 0; i < 64; i++) {
    const parent = join(cwd, "layered", `top-${i}`); mkdirSync(parent);
    for (let j = 0; j < 400; j++) {
      const child = join(parent, `child-${j}`); mkdirSync(child); writeFileSync(join(child, "common.txt"), "owned\n");
      if (i % 17 === 0 && j === 0) writeFileSync(join(child, "rare-result.dat"), "owned\n");
    }
  }
  const cases = [
    ["large-literal-counts", "grep -raFc alpha corpus", "/usr/bin/grep -raFc alpha corpus"],
    ["large-sparse-literal-lines", "grep -ranF alpha corpus", "/usr/bin/grep -ranF alpha corpus"],
    ["long-record-counts", "grep -raFc alpha long-lines", "/usr/bin/grep -raFc alpha long-lines"],
    ["long-record-matching-lines", "grep -ranF alpha long-lines", "/usr/bin/grep -ranF alpha long-lines"],
    ["small-literal-lines", "grep -nF alpha small.txt", "/usr/bin/grep -nF alpha small.txt"],
    ["large-sparse-filenames", "find layered -type f -name rare-result.dat -print0", "/usr/bin/find layered -type f -name rare-result.dat -print0"],
    ["small-filenames", "find corpus -name part-0.txt -print0", "/usr/bin/find corpus -name part-0.txt -print0"],
  ];
  const results = [];
  for (const [id, automatic, gnu] of cases) {
    const reference = run(gnu, false).result;
    equal(run(automatic, true).result, reference);
    const samples = { gnu: [], automatic: [] };
    for (let i = 0; i < 7; i++) for (const engine of i % 2 ? ["automatic", "gnu"] : ["gnu", "automatic"]) {
      const value = run(engine === "gnu" ? gnu : automatic, engine === "automatic");
      equal(value.result, reference); samples[engine].push(value.ms);
    }
    results.push({ id, median_ms: { gnu: median(samples.gnu), automatic: median(samples.automatic) }, samples_ms: samples,
      exact_gnu_comparisons: 15, stdout_bytes: reference.stdout.length,
      stdout_sha256: createHash("sha256").update(reference.stdout).digest("hex"), status: reference.status });
  }
  console.log(JSON.stringify({ schema: "coherence-search-dispatch-speed-v1", captured_at: new Date().toISOString(),
    scope: "Actual combined Pi Bash hook on owned warm-cache fixtures; shell, dispatch, preparation, output included; seven alternating pairs, one warmup.",
    platform: process.platform, arch: process.arch, logical_cpus: cpus().length,
    versions: Object.fromEntries(["grep", "find", "rg", "fd"].map(program => {
      const value = spawnSync(program, ["--version"], { env, encoding: "utf8", timeout: 5000 });
      assert.equal(value.status, 0, `${program} version unavailable`);
      return [program, value.stdout.split("\n")[0]];
    })),
    source_manifest_sha256: createHash("sha256").update(readFileSync(join(source, "integrations/pi/search-wrappers/source-manifest.json"))).digest("hex"),
    cases: results }, null, 2));
} finally { rmSync(cwd, { recursive: true, force: true }); }
