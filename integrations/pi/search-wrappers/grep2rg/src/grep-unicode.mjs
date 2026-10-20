// Native GNU supplies scalar membership; rg still searches the actual operands.
// No Unicode-version or locale case-folding table is guessed by this adapter.
import { spawnSync } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { accessSync, closeSync, constants, lstatSync, mkdirSync, openSync, readFileSync, realpathSync, renameSync, rmSync, statSync, writeSync } from "node:fs";
import { delimiter, join } from "node:path";
import { tmpdir } from "node:os";
import { nulRecordPattern, translateGrepPattern } from "./grep-options.mjs";
import { inheritedGrepStdio } from "./grep-descriptors.mjs";

const CORPUS_SIZE = 5494654;
const CORPUS_HASH = "975786db53b020f0eb218668c84ba316c35bda92421027175bc503a2e1c2db0d";
const memory = new Map();
const hash = (value) => createHash("sha256").update(value).digest("hex");
const pointEscape = (point) => `\\x{${point.toString(16)}}`;

function writeAll(descriptor, bytes) {
  let offset = 0;
  while (offset < bytes.length) {
    const length = writeSync(descriptor, bytes, offset, bytes.length - offset);
    if (!length) throw new Error("Unicode cache write made no progress");
    offset += length;
  }
}

function privateCache(env) {
  const path = env.GREP2RG_UNICODE_CACHE || join(tmpdir(), `grep2rg-unicode-${process.getuid()}`);
  try { mkdirSync(path, { mode: 0o700 }); } catch (error) { if (error.code !== "EEXIST") throw error; }
  const info = lstatSync(path);
  if (!info.isDirectory() || info.isSymbolicLink() || info.uid !== process.getuid() || (info.mode & 0o077))
    throw new Error("unsafe Unicode cache directory");
  return path;
}

function temporary(path) {
  return `${path}.${process.pid}.${randomBytes(8).toString("hex")}.tmp`;
}

function corpus(cache) {
  const path = join(cache, "unicode-scalars-v1.nul");
  try {
    const info = lstatSync(path);
    if (info.isFile() && !info.isSymbolicLink() && info.uid === process.getuid() && info.size === CORPUS_SIZE &&
        hash(readFileSync(path)) === CORPUS_HASH) return path;
  } catch {}
  const staging = temporary(path);
  let descriptor;
  try {
    descriptor = openSync(staging, "wx", 0o600);
    const chunk = Buffer.allocUnsafe(65536);
    let length = 0;
    for (let point = 1; point <= 0x10ffff; point++) {
      if (point >= 0xd800 && point <= 0xdfff) continue;
      if (length + 5 > chunk.length) { writeAll(descriptor, chunk.subarray(0, length)); length = 0; }
      length += chunk.write(String.fromCodePoint(point), length, undefined, "utf8");
      chunk[length++] = 0;
    }
    if (length) writeAll(descriptor, chunk.subarray(0, length));
    closeSync(descriptor); descriptor = undefined;
    if (hash(readFileSync(staging)) !== CORPUS_HASH) throw new Error("invalid Unicode scalar corpus");
    renameSync(staging, path);
  } finally {
    if (descriptor !== undefined) closeSync(descriptor);
    rmSync(staging, { force: true });
  }
  return path;
}

function metadata(path) {
  try {
    const info = statSync(path, { bigint: true });
    return [realpathSync(path), ...["dev", "ino", "size", "mtimeNs", "ctimeNs"].map((field) => String(info[field]))];
  } catch { return [path, "missing"]; }
}

function gnuIdentity(env) {
  let executable;
  for (const directory of (env.PATH ?? "").split(delimiter)) {
    try {
      const path = join(directory || ".", "grep");
      accessSync(path, constants.X_OK);
      if (statSync(path).isFile()) { executable = realpathSync(path); break; }
    } catch {}
  }
  if (!executable) throw new Error("GNU grep unavailable for Unicode proof");
  const version = spawnSync(executable, ["--version"], { stdio: inheritedGrepStdio(["pipe", "pipe", "pipe"]), env, maxBuffer: 65536, timeout: 5000 });
  if (version.error || version.status !== 0 || !version.stdout.includes(Buffer.from("GNU grep")))
    throw new Error("Unicode proof requires GNU grep");
  const ctype = env.LC_ALL || env.LC_CTYPE || env.LANG || "C";
  const collate = env.LC_ALL || env.LC_COLLATE || env.LANG || "C";
  const libraries = readFileSync("/proc/self/maps", "utf8").split("\n")
    .map((line) => /\s(\/[^\n]*\/libc(?:[-.]|\.so)[^\n]*)$/.exec(line)?.[1]).filter(Boolean);
  const localeNames = [...new Set([ctype, collate, ctype.replace(/UTF-8$/i, "utf8"), collate.replace(/UTF-8$/i, "utf8")])];
  const localeFiles = ["/usr/lib/locale/locale-archive", "/usr/lib/locale/locale.alias",
    ...localeNames.flatMap((name) => ["/usr/lib/locale", "/lib/locale"].flatMap((base) =>
      [join(base, name, "LC_CTYPE"), join(base, name, "LC_COLLATE")]))];
  // Custom loaders/locale paths may resolve outside these observed dependencies.
  // Their tables remain correct, but are reprobed rather than persisted.
  const persistent = !env.LD_PRELOAD && !env.LD_LIBRARY_PATH && !env.LOCPATH;
  const identity = hash(JSON.stringify({ format: 2, executable: metadata(executable), binary: hash(readFileSync(executable)),
    version: version.stdout.toString(), ctype, collate, lang: env.LANG ?? "", libc: [...new Set(libraries)].map(metadata),
    locale: localeFiles.map(metadata), loader: [env.LD_PRELOAD, env.LD_LIBRARY_PATH, env.LOCPATH] }));
  return { executable, identity, persistent };
}

function ranges(points) {
  const result = [];
  for (const point of points) {
    const last = result.at(-1);
    if (last && last[1] + 1 === point) last[1] = point;
    else result.push([point, point]);
  }
  return result;
}

function rangePattern(value) {
  if (!value.length) return "(?!)";
  return `[${value.map(([first, last]) => first === last ? pointEscape(first) : `${pointEscape(first)}-${pointEscape(last)}`).join("")}]`;
}

function scalarTable(context, kind, pattern) {
  const key = hash(JSON.stringify([context.identity, kind, pattern]));
  if (context.tables.has(key)) return context.tables.get(key);
  if (context.persistent && memory.has(key)) return memory.get(key);
  const path = join(context.cache, `${key}.json`);
  if (context.persistent) {
    try {
      const info = lstatSync(path);
      if (!info.isFile() || info.isSymbolicLink() || info.uid !== process.getuid() || info.size > 4 * 1024 * 1024) throw new Error("invalid cached table");
      const bytes = readFileSync(path);
      const value = JSON.parse(bytes), payload = JSON.stringify(value.ranges);
      if (value.key !== key || value.checksum !== hash(payload) || !Array.isArray(value.ranges) ||
          !value.ranges.every((entry) => Array.isArray(entry) && entry.length === 2 && Number.isInteger(entry[0]) && Number.isInteger(entry[1]) &&
            entry[0] >= 0 && entry[1] >= entry[0] && entry[1] <= 0x10ffff && !(entry[0] <= 0xdfff && entry[1] >= 0xd800)))
        throw new Error("invalid cached ranges");
      const translated = rangePattern(value.ranges);
      context.tables.set(key, translated); memory.set(key, translated); return translated;
    } catch {}
  }
  const args = ["--text", "--null-data", "--line-regexp", "--color=never", "--no-filename",
    kind === "fold" ? "--fixed-strings" : "--extended-regexp", ...(kind === "fold" || kind === "class-fold" ? ["--ignore-case"] : []),
    "--regexp", pattern, "--", context.corpus];
  const result = spawnSync(context.executable, args, { stdio: inheritedGrepStdio(["pipe", "pipe", "pipe"]), env: context.env, timeout: 10000, maxBuffer: 16 * 1024 * 1024 });
  if (result.error || result.signal || result.status > 1 || result.stderr.length) throw new Error("GNU Unicode scalar probe failed");
  const records = result.stdout.toString("utf8").split("\0");
  if (records.pop() !== "") throw new Error("invalid Unicode scalar probe terminator");
  const points = records.map((record) => {
    const chars = [...record];
    if (chars.length !== 1) throw new Error("non-scalar GNU membership result");
    return chars[0].codePointAt(0);
  });
  if (kind !== "fold") {
    // NUL cannot itself be a member record in the NUL-delimited corpus. Test
    // that one scalar separately using an ordinary LF record and GNU -a.
    const nul = spawnSync(context.executable, ["--text", "--line-regexp", "--quiet", "--extended-regexp",
      ...(kind === "class-fold" ? ["--ignore-case"] : []), "--regexp", pattern],
      { stdio: inheritedGrepStdio(["pipe", "pipe", "pipe"]), env: context.env, input: Buffer.from([0, 10]), timeout: 5000, maxBuffer: 65536 });
    if (nul.error || nul.signal || nul.status > 1 || nul.stderr.length) throw new Error("GNU NUL scalar probe failed");
    if (nul.status === 0) points.unshift(0);
  }
  if (points.some((point, index) => index && point <= points[index - 1])) throw new Error("unordered Unicode membership result");
  const value = ranges(points), translated = rangePattern(value);
  context.tables.set(key, translated);
  if (context.persistent) {
    memory.set(key, translated);
    const staging = temporary(path);
    let descriptor;
    try {
      descriptor = openSync(staging, "wx", 0o600);
      writeAll(descriptor, Buffer.from(JSON.stringify({ key, ranges: value, checksum: hash(JSON.stringify(value)) })));
      closeSync(descriptor); descriptor = undefined;
      renameSync(staging, path);
    } finally { if (descriptor !== undefined) closeSync(descriptor); rmSync(staging, { force: true }); }
  }
  return translated;
}

/** Fixed GNU matching on fully validated UTF-8 subjects, with native tables. */
export function unicodeMatcherArgs(options, patterns, { env = process.env } = {}) {
  const cache = privateCache(env), identity = gnuIdentity(env);
  // Custom locale/loader dependencies may change between API calls. Deduplicate
  // their probes only within this translation, never across requests.
  const context = { ...identity, cache, env, corpus: corpus(cache), tables: new Map() };
  const definitions = new Map();
  const characterClass = (kind, pattern) => {
    const name = `G2RC${definitions.size}`;
    const value = scalarTable(context, kind, pattern);
    // Named subroutines share large classes and preserve user capture indices:
    // definitions are appended after all user expressions, never prepended.
    for (const [existing, classValue] of definitions) if (classValue === value) return `(?&${existing})`;
    definitions.set(name, value);
    return `(?&${name})`;
  };
  let word;
  const wordClass = () => word ??= characterClass("class", "[[:alnum:]_]");
  const spaceClass = () => characterClass("class", "[[:space:]]");
  const wordEscape = (name) => {
    if (name === "s") return spaceClass();
    if (name === "S") return `(?!${spaceClass()})[\\s\\S]`;
    const value = wordClass();
    if (name === "w") return value;
    if (name === "W") return `(?!${value})[\\s\\S]`;
    if (name === "<") return `(?<!${value})(?=${value})`;
    if (name === ">") return `(?<=${value})(?!${value})`;
    if (name === "b") return `(?:(?<=${value})(?!${value})|(?<!${value})(?=${value}))`;
    return `(?:(?<=${value})(?=${value})|(?<!${value})(?!${value}))`;
  };
  const args = ["--encoding", "none", "--unicode", "--pcre2"];
  if (options.nullData) args.push("--null-data");
  if (options.invertMatch) args.push("--invert-match");
  if (options.lineRegexp && !options.nullData) args.push("--line-regexp");
  const expressions = patterns.map((pattern) => options.matcher === "fixed" ? [...pattern].map((char) =>
    options.ignoreCase ? scalarTable(context, "fold", char) : pointEscape(char.codePointAt(0))).join("") :
    translateGrepPattern(pattern, options.matcher, {
      bracketTransform: (value) => characterClass(options.ignoreCase ? "class-fold" : "class", value),
      caseTransform: options.ignoreCase ? (char) => scalarTable(context, "fold", char) : undefined,
      wordTransform: wordEscape,
    }));
  // Share the large word class across all fixed alternatives rather than
  // duplicating it for every -e operand and wasting the OS argument budget.
  let translated = expressions.length > 1 ? `(?|${expressions.map((expression) => `(?:${expression})`).join("|")})` : expressions[0];
  if (options.wordRegexp && !options.lineRegexp) translated = `(?<!${wordClass()})(?:${translated})(?!${wordClass()})`;
  if (options.nullData && options.matcher !== "fixed") translated = nulRecordPattern(translated);
  if (options.nullData && options.lineRegexp) translated = `\\A(?:${translated})\\z`;
  if (definitions.size) translated += `(?(DEFINE)${[...definitions].map(([name, value]) => `(?<${name}>${value})`).join("")})`;
  args.push("-e", translated);
  return args;
}
