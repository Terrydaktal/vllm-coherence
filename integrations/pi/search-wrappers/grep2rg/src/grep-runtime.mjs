// GNU grep is the compatibility boundary; rg executes qualified searches.
// Selection happens before consuming stdin or publishing any result. Unknown
// options and irreducible semantics go to GNU unchanged, without an LLM retry.
import { spawn, spawnSync } from "node:child_process";
import { accessSync, constants, lstatSync, readFileSync, readlinkSync, statSync } from "node:fs";
import { open, stat } from "node:fs/promises";
import { isAscii, isUtf8 } from "node:buffer";
import { delimiter, join } from "node:path";
import { parseGrepOptions, resolveGrepPatterns, buildGrepMatcherArgs, grepMatcherFallbackReasons } from "./grep-options.mjs";
import { enumerateInputs, loadGrepFilters, requiresNativeGrepFileSelection } from "./grep-files.mjs";
import { createGrepFormatter } from "./grep-output.mjs";
import { createPosixSpanBroker, gnuSpanArguments } from "./grep-posix-spans.mjs";
import { unicodeMatcherArgs } from "./grep-unicode.mjs";
import { inheritedGrepStdio } from "./grep-descriptors.mjs";
import { seekableStdin, prepareSeekableStdin } from "./grep-stdin.mjs";

const PATTERN_LIMIT = 1024 * 1024;
const INPUT_LIMIT = 100000;
const INPUT_BYTE_LIMIT = 16 * 1024 * 1024;
const ARG_LIMIT = 48 * 1024;
// Default GNU binary detection is retained unless the complete regular-file
// contents can be proven to be text before any matching output is published.
const TEXT_PREFLIGHT_CHUNK = 64 * 1024;
const argumentBytes = (args) => args.reduce((bytes, value) => bytes + Buffer.byteLength(value) + 1, 0);
const literalBasic = (pattern) => !/[.*^$[\\]/.test(pattern);
const literalExtended = (pattern) => !/[.*^$[\\+?(){}|]/.test(pattern);
const nativePlan = (argv, reason) => ({ backend: "grep", args: argv, reason });

function rgAvailable() {
  return (process.env.PATH ?? "").split(delimiter).some((directory) => {
    try {
      const path = join(directory || ".", "rg");
      accessSync(path, constants.X_OK);
      return statSync(path).isFile();
    } catch { return false; }
  });
}

function matcherQualification(options, patterns, env = process.env) {
  const normalized = { ...options, patternSources: patterns.map((value) => ({ type: "regexp", value })) };
  if (normalized.color === "auto" && process.stdout.isTTY && env.TERM !== "dumb") normalized.color = "always";
  let reasons = grepMatcherFallbackReasons(normalized, { env });
  const locale = env.LC_ALL || env.LC_CTYPE || env.LANG || "C";
  const collate = env.LC_ALL || env.LC_COLLATE || env.LANG || "C";
  const safeAsciiFold = /^(?:C|en(?:_[A-Z]{2})?)[.-]UTF-?8$/i.test(locale) &&
    patterns.every((pattern) => /^[\x00-\x7f]*$/.test(pattern) && !/[ik]/i.test(pattern));
  if (safeAsciiFold) reasons = reasons.filter((reason) => !reason.startsWith("GNU locale case folding"));
  // In the C UTF-8 locale, ASCII subjects and ASCII patterns have the same
  // folding and word/class arithmetic as C. That fact must be established from
  // every input byte before selecting byte-mode rg; Unicode subjects retain
  // the existing libc-dependent fallback rather than guessing from the pattern.
  const asciiCandidate = /^C[.-]UTF-?8$/i.test(locale) && patterns.every((pattern) => isAscii(Buffer.from(pattern)));
  const asciiReason = (reason) => reason.startsWith("GNU locale case folding") ||
    reason === "GNU locale word boundaries differ from PCRE2 Unicode classes" ||
    reason === "GNU locale character classes, ranges or word boundaries";
  const requiresUnicodeSubject = options.matcher !== "perl" && /[.-]UTF-?8$/i.test(locale) && reasons.some(asciiReason);
  if (requiresUnicodeSubject) {
    reasons = reasons.filter((reason) => !asciiReason(reason));
    // Scalar probes cannot see multi-character collating elements, including
    // implicit elements admitted by negated brackets in locales such as Czech.
    if (options.matcher !== "fixed" && patterns.some((pattern) => pattern.includes("[")) &&
        !["C", "POSIX"].includes(collate) && !/^C[.-]UTF-?8$/i.test(collate))
      reasons.push("GNU multi-character locale collation requires native bracket matching");
    if (options.ignoreCase && options.matcher !== "fixed" && patterns.some((pattern) => /\\[1-9]/.test(pattern)))
      reasons.push("GNU locale-insensitive backreference comparison requires native matching");
  }
  const requiresAsciiSubject = !requiresUnicodeSubject && asciiCandidate && reasons.some(asciiReason);
  if (requiresAsciiSubject) reasons = reasons.filter((reason) => !asciiReason(reason));
  const literalCandidate = options.matcher === "fixed" ||
    (options.matcher === "basic" && patterns.every(literalBasic)) ||
    (options.matcher === "extended" && patterns.every(literalExtended));
  const spanOutput = !options.quiet && !options.count && !options.filesMode && (options.onlyMatching || normalized.color === "always");
  const requiresLiteralLongest = spanOutput && literalCandidate && !options.ignoreCase && !options.wordRegexp && !options.lineRegexp;
  const spanReason = (reason) => reason === "GNU fixed patterns use leftmost-longest match selection for extraction and highlighting" ||
    reason === "GNU BRE/ERE uses leftmost-longest matching for -o" ||
    reason === "GNU BRE/ERE uses leftmost-longest matching for highlighted spans";
  const requiresPosixSpans = spanOutput && options.matcher !== "perl" && !requiresLiteralLongest &&
    (reasons.some(spanReason) || (options.wordRegexp && options.lineRegexp));
  reasons = reasons.filter((reason) => !spanReason(reason));
  return { reasons, requiresAsciiSubject, requiresUnicodeSubject, requiresLiteralLongest, requiresPosixSpans };
}

/** Pure routing for argv; file pattern contents are resolved by runGrep. */
export function planGrep(argv, { env = process.env, stat = statSync } = {}) {
  if (env.GREP2RG_EXPERIMENTAL_RG !== "1")
    return nativePlan(argv, "GNU default: experimental rg requires explicit opt-in");
  const options = parseGrepOptions(argv, { env });
  if (options.errors.length || options.help || options.version)
    return nativePlan(argv, "GNU option diagnostics, help or version");
  // GNU -U changes text-mode translation on Windows and is a no-op on Linux.
  if ((options.binaryMode && process.platform !== "linux") || process.platform === "win32")
    return nativePlan(argv, "platform-specific GNU byte handling");
  if (options.maxCount === 0) return nativePlan(argv, "GNU zero-count validation and output");
  const stdin = options.paths.includes("-") || (!options.paths.length && !options.recursive);
  const requiresSeekableStdin = stdin && options.binary === "text" && options.maxCount > 0 &&
    !options.quiet && !options.filesMode && !options.recursive &&
    (!options.paths.length || (options.paths.length === 1 && options.paths[0] === "-")) && Boolean(seekableStdin());
  if (options.patternSources.some((p) => p.type === "file" && p.value === "-"))
    return nativePlan(argv, "stdin pattern file must preserve GNU stream ownership");
  if (stdin && (options.maxCount !== null || options.quiet || options.filesMode) && !requiresSeekableStdin)
    return nativePlan(argv, "early exits must preserve GNU's stdin position");
  // A stdin pre-scan would consume the descriptor GNU may need.
  if (options.binary !== "text" && stdin)
    return nativePlan(argv, "GNU binary reporting for stdin");
  const patterns = options.patternSources.filter((p) => p.type === "regexp").flatMap((p) => p.value.split("\n"));
  if (options.invertMatch && !options.lineRegexp && !options.wordRegexp && patterns.includes(""))
    return nativePlan(argv, "GNU inverted universal-pattern shortcut");
  const { reasons, requiresAsciiSubject, requiresUnicodeSubject, requiresPosixSpans } = matcherQualification(options, patterns, env);
  if (reasons.length) return nativePlan(argv, reasons.join("; "));
  if (requiresAsciiSubject && stdin) return nativePlan(argv, "GNU locale matching requires proof of ASCII input bytes");
  if (requiresUnicodeSubject && stdin) return nativePlan(argv, "GNU locale matching requires proof of UTF-8 input bytes");
  if (requiresNativeGrepFileSelection(options, { env })) return nativePlan(argv, "locale-specific filename matching");
  let matcher = options.matcher;
  if (patterns.length && ((matcher === "basic" && patterns.every(literalBasic)) ||
      (matcher === "extended" && patterns.every(literalExtended)))) matcher = "fixed";
  // An explicitly listed regular-file filename search needs no presentation
  // adapter. Keep its streams inherited and its short-circuit fast path.
  const renderColor = options.color === "always" || (options.color === "auto" && process.stdout.isTTY && env.TERM !== "dumb");
  let direct = options.filesMode === "with" && !options.quiet && !renderColor &&
    !requiresAsciiSubject && !requiresUnicodeSubject &&
    options.binary !== "without-match" &&
    (options.binary === "text" || !options.invertMatch) &&
    !options.filters.length && !options.patternSources.some((p) => p.type === "file") &&
    !stdin && options.paths.length > 0 && matcher === "fixed" && !options.nullData && !options.wordRegexp && !options.lineRegexp;
  for (const path of options.paths) {
    try { if (!stat(path).isFile()) direct = false; } catch { direct = false; }
  }
  const normalized = { ...options, matcher };
  let args = direct ? ["--no-config", "--no-ignore", "--hidden", "--threads", "1", "--color", "never",
    options.binary === "text" ? "--text" : "--binary", "--files-with-matches",
    ...buildGrepMatcherArgs(normalized, patterns, { env }), ...(options.nullNames ? ["--null"] : []),
    ...(options.silent ? ["--no-messages"] : []), "--", ...options.paths] : undefined;
  if (direct && argumentBytes(args) > ARG_LIMIT) { direct = false; args = undefined; }
  const requiresTextPreflight = !direct && options.binary !== "text";
  const locale = env.LC_ALL || env.LC_CTYPE || env.LANG || "C";
  if (requiresTextPreflight && !["C", "POSIX"].includes(locale) && !/UTF-?8$/i.test(locale))
    return nativePlan(argv, "GNU binary reporting in an unclassified encoding");
  return { backend: "rg", mode: direct ? "direct" : "json", args, options, patterns, requiresTextPreflight, requiresAsciiSubject, requiresUnicodeSubject, requiresPosixSpans, requiresSeekableStdin };
}

function runInherited(program, args) {
  return new Promise((resolve) => {
    const child = spawn(program, args, { stdio: inheritedGrepStdio() });
    child.once("error", (error) => {
      process.stderr.write(`grep: could not start ${program}: ${error.code ?? "spawn failure"}\n`);
      resolve(2);
    });
    child.once("exit", (code, signal) => resolve(code ?? (signal === "SIGPIPE" ? 141 : 2)));
  });
}

function readablePattern(path) {
  const info = lstatSync(path);
  // Following ordinary symlinks is safe; never open a FIFO for a speculative
  // parser read, and leave large or non-UTF-8 pattern files to GNU unchanged.
  if (!(info.isFile() || (info.isSymbolicLink() && statSync(path).isFile()))) throw new Error("special pattern file");
  if (statSync(path).size > PATTERN_LIMIT) throw new Error("large pattern file");
  const data = readFileSync(path);
  if (!isUtf8(data) || data.includes(0)) throw new Error("byte-oriented pattern file");
  return data;
}

function sameFileVersion(before, after) {
  return ["dev", "ino", "size", "mtimeNs", "ctimeNs", "mode"].every((field) => before[field] === after[field]);
}

/**
 * Classify complete regular-file payloads in fixed memory, never a prefix or a
 * consumed stdin. File length and operand multiplicity do not restrict this
 * proof; limits on metadata enumeration are independent of payload inspection.
 * The nonblocking open plus descriptor stat also prevents a file-to-FIFO race
 * from hanging the speculative read. No descriptor survives native fallback.
 * Metadata checks detect observed replacement/mutation, not an atomic snapshot
 * between this preparation and rg opening the original operands.
 */
async function classifyTextInputs(inputs, { env = process.env, requireText = true, requireAscii = false, nullData = false, subjectInfo } = {}) {
  if (inputs.some((input) => input.kind !== "file" || input.stdin || input.special)) return false;
  const locale = env.LC_ALL || env.LC_CTYPE || env.LANG || "C";
  const byteLocale = ["C", "POSIX"].includes(locale);
  if (!byteLocale && !/UTF-?8$/i.test(locale)) return false;
  const buffer = Buffer.allocUnsafe(TEXT_PREFLIGHT_CHUNK);
  for (const input of inputs) {
    let file;
    try {
      file = await open(input.path, constants.O_RDONLY | constants.O_NONBLOCK | constants.O_NOCTTY);
      const before = await file.stat({ bigint: true });
      if (!before.isFile()) return false;
      const decoder = byteLocale ? undefined : new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
      let fileBytes = 0n;
      for (;;) {
        const { bytesRead } = await file.read(buffer, 0, buffer.length, null);
        if (!bytesRead) break;
        fileBytes += BigInt(bytesRead);
        const chunk = buffer.subarray(0, bytesRead);
        if (subjectInfo && !isAscii(chunk)) subjectInfo.ascii = false;
        if ((requireText && !nullData && chunk.includes(0)) || (requireAscii && !isAscii(chunk))) return false;
        decoder?.decode(chunk, { stream: true });
      }
      decoder?.decode();
      const after = await file.stat({ bigint: true });
      const named = await stat(input.path, { bigint: true });
      if (fileBytes !== before.size || !sameFileVersion(before, after) || !sameFileVersion(after, named)) return false;
      // -T prefix widths must describe the same payload that was classified.
      input.size = Number(after.size);
    } catch { return false; }
    finally { await file?.close().catch(() => {}); }
  }
  return true;
}

async function prepare(plan, argv) {
  const options = plan.options;
  let patterns, filters;
  try {
    patterns = await resolveGrepPatterns(options, readablePattern);
    // Exclusion files are checked before the traversal helper reads them.
    for (const path of options.excludeFiles) readablePattern(path);
    filters = await loadGrepFilters(options);
  } catch { return nativePlan(argv, "GNU pattern/exclusion file handling"); }
  if (!patterns.length) return nativePlan(argv, "GNU empty pattern-file shortcuts");
  if (options.invertMatch && !options.lineRegexp && !options.wordRegexp && patterns.includes(""))
    return nativePlan(argv, "GNU inverted universal-pattern shortcut");
  const { reasons, requiresAsciiSubject, requiresUnicodeSubject, requiresLiteralLongest, requiresPosixSpans } = matcherQualification(options, patterns);
  if (reasons.length || requiresNativeGrepFileSelection({ ...options, filters }))
    return nativePlan(argv, reasons.join("; ") || "locale-specific filename matching");
  let matcher = options.matcher;
  if (patterns.length && ((matcher === "basic" && patterns.every(literalBasic)) ||
      (matcher === "extended" && patterns.every(literalExtended)))) matcher = "fixed";
  const effective = { ...options, matcher, filters, excludeFiles: [] };
  if (requiresLiteralLongest) effective.fixedLongestPatterns = patterns;
  if (requiresPosixSpans) {
    effective.posixSpanArgs = gnuSpanArguments(effective, patterns);
    // Resolve GNU-only compilation/diagnostic behavior before consuming input.
    // Subsequent GNU matching sees only the records selected by rg.
    const check = spawnSync("grep", [...effective.posixSpanArgs, "--", "/dev/null"],
      { stdio: inheritedGrepStdio(["pipe", "pipe", "pipe"]), maxBuffer: 1024 * 1024, timeout: 5000 });
    if (check.error || check.status > 1 || check.signal || check.stderr.length)
      return nativePlan(argv, "GNU POSIX span matcher compilation or diagnostics");
  }
  const matchingEnv = requiresAsciiSubject ? { ...process.env, LC_ALL: "C" } : process.env;
  let matchingArgs;
  try { matchingArgs = requiresUnicodeSubject ? undefined : buildGrepMatcherArgs(effective, patterns, { env: matchingEnv }); }
  catch { return nativePlan(argv, "GNU matcher diagnostics"); }
  // A small -f operand can expand into thousands of argv entries. Fall back
  // before touching input rather than risking E2BIG in the translated search.
  if (matchingArgs && argumentBytes(matchingArgs) > ARG_LIMIT) return nativePlan(argv, "GNU large pattern-file handling");
  // Compile before walking files or consuming stdin; GNU remains responsible
  // for exact regex diagnostics and extensions that PCRE2 cannot compile.
  if (matchingArgs?.includes("--pcre2")) {
    const check = spawnSync("rg", ["--no-config", "--text", ...matchingArgs, "--", "-"],
      { stdio: inheritedGrepStdio(["pipe", "pipe", "pipe"]), input: Buffer.alloc(0), maxBuffer: 1024 * 1024, timeout: 5000 });
    if (check.error || check.status > 1 || check.signal) return nativePlan(argv, "GNU regex compilation or unavailable PCRE2");
  }
  const inputs = [];
  let inputBytes = 0;
  try {
    for await (const input of enumerateInputs(effective)) {
      inputBytes += Buffer.byteLength(input.label ?? "");
      if (input.kind === "native" || input.kind === "error" || input.special || inputs.length >= INPUT_LIMIT || inputBytes > INPUT_BYTE_LIMIT)
        return nativePlan(argv, "GNU special files, byte filenames or large traversal");
      inputs.push(input);
    }
  } catch { return nativePlan(argv, "GNU filesystem diagnostics"); }
  const subjectInfo = { ascii: true };
  if (plan.requiresTextPreflight || requiresAsciiSubject || requiresUnicodeSubject) {
    if (!await classifyTextInputs(inputs, { requireText: plan.requiresTextPreflight, requireAscii: requiresAsciiSubject, nullData: options.nullData, subjectInfo }))
      return nativePlan(argv, "GNU complete-file binary/encoding or ASCII-subject classification");
    if (plan.requiresTextPreflight) effective.binary = "text";
  }
  if (requiresUnicodeSubject) {
    try {
      const locale = process.env.LC_ALL || process.env.LC_CTYPE || process.env.LANG || "C";
      const collate = process.env.LC_ALL || process.env.LC_COLLATE || process.env.LANG || "C";
      // C.UTF-8's ASCII-only subjects need no Unicode corpus probe at all.
      matchingArgs = /^C[.-]UTF-?8$/i.test(locale) && (["C", "POSIX"].includes(collate) || /^C[.-]UTF-?8$/i.test(collate)) &&
        subjectInfo.ascii && patterns.every((pattern) => isAscii(Buffer.from(pattern))) ?
        buildGrepMatcherArgs(effective, patterns, { env: { ...process.env, LC_ALL: "C" } }) : unicodeMatcherArgs(effective, patterns);
    } catch { return nativePlan(argv, "GNU locale scalar membership proof unavailable"); }
    if (argumentBytes(matchingArgs) > ARG_LIMIT) return nativePlan(argv, "GNU Unicode table exceeds rg argument capacity");
    const check = spawnSync("rg", ["--no-config", "--text", ...matchingArgs, "--", "-"],
      { stdio: inheritedGrepStdio(["pipe", "pipe", "pipe"]), input: Buffer.alloc(0), maxBuffer: 1024 * 1024, timeout: 5000 });
    if (check.error || check.status > 1 || check.signal) return nativePlan(argv, "GNU Unicode translation or unavailable PCRE2");
  }
  let seekableInput;
  if (plan.requiresSeekableStdin) {
    try { seekableInput = prepareSeekableStdin(); }
    catch { return nativePlan(argv, "GNU seekable stdin stream or cursor could not be established"); }
  }
  return { options: effective, inputs, matchingArgs, seekableInput };
}

function chunks(inputs) {
  const batches = [];
  let batch = [], bytes = 0, labels = new Set();
  const flush = () => { if (batch.length) batches.push(batch); batch = []; bytes = 0; labels = new Set(); };
  for (const input of inputs) {
    if (input.kind !== "file" || bytes + Buffer.byteLength(input.label) > ARG_LIMIT || labels.has(input.label)) flush();
    if (input.kind !== "file") batches.push([input]);
    else { batch.push(input); bytes += Buffer.byteLength(input.label) + 1; labels.add(input.label); }
  }
  flush();
  return batches;
}

async function searchBatch(batch, args, formatter, options, state) {
  for (const input of batch) formatter.registerFile(input.stdin ? "<stdin>" : input.label,
    { size: input.size, stdin: input.stdin, filenameRequired: input.filenameRequired });
  const operands = batch.map((input) => input.stdin ? "-" : input.label);
  let broker;
  try { if (options.posixSpanArgs) broker = createPosixSpanBroker(options.posixSpanArgs, options); }
  catch {
    state.searchError = true;
    for (const input of batch) formatter.markError(input.stdin ? "<stdin>" : input.label);
    if (!options.silent) process.stderr.write("grep: could not prepare GNU span adapter\n");
    return;
  }
  // stdin uses the original fd, not Node buffering; multiple '-' operands must
  // share the same consumed descriptor. Reading here never changes the source.
  const child = spawn("rg", [...args, "--", ...operands], { stdio: inheritedGrepStdio([state.seekableInput ? "pipe" : "inherit", "pipe", "pipe"]) });
  state.child = child;
  const feed = state.seekableInput?.feed(child.stdin, () => state.brokenPipe || Boolean(state.signal))
    .then(() => undefined, (error) => error);
  let pending = "", diagnostic = "", stopped = false;
  const status = await new Promise((resolve) => {
    const consume = (record) => {
      if (stopped || state.brokenPipe || state.signal || !record) return;
      try {
        const message = JSON.parse(record);
        state.seekableInput?.record(message, options.maxCount);
        if (broker) broker.capture(message, record);
        else formatter.event(message);
        if (options.quiet && formatter.selected) { stopped = true; child.kill("SIGTERM"); }
      } catch {
        stopped = true; state.protocolError = true; child.kill("SIGTERM");
        if (!options.silent) process.stderr.write("grep: invalid search response from rg\n");
      }
    };
    child.once("error", (error) => { diagnostic = `grep: rg: ${error.code ?? "spawn failure"}\n`; resolve(2); });
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (data) => {
      pending += data;
      let newline;
      while ((newline = pending.indexOf("\n")) >= 0) {
        const record = pending.slice(0, newline); pending = pending.slice(newline + 1);
        consume(record);
      }
    });
    child.stderr.on("data", (data) => { diagnostic += data; if (diagnostic.length > 65536) diagnostic = diagnostic.slice(0, 65536); });
    child.once("close", (code, signal) => {
      // JSONL may omit the final newline. A complete final event must still be
      // consumed; a truncated one is an error rather than a zero-match result.
      consume(pending);
      resolve(stopped && options.quiet && formatter.selected ? 0 : code ?? (signal === "SIGPIPE" ? 141 : 2));
    });
  });
  state.child = null;
  const feedError = await feed;
  if (feedError && !state.signal && !state.brokenPipe) {
    state.searchError = true;
    if (!options.silent) process.stderr.write(`grep: seekable stdin stream: ${feedError.message}\n`);
  }
  try {
    if (broker && !state.protocolError && !state.brokenPipe && !state.signal) await broker.replay(formatter,
      { setChild: (child) => { state.child = child; }, stopped: () => state.brokenPipe || Boolean(state.signal), waitForOutput: state.waitForOutput });
  } catch (error) {
    state.searchError = true;
    for (const input of batch) formatter.markError(input.stdin ? "<stdin>" : input.label);
    if (!options.silent) process.stderr.write(`grep: GNU span adapter: ${error.message.trim()}\n`);
  } finally { broker?.close(); }
  if (diagnostic && !options.silent) process.stderr.write(diagnostic);
  if ((status > 1 || state.protocolError) && !(options.quiet && formatter.selected)) {
    state.searchError = true;
    // A failed batch must not invent counts or nonmatching filenames for files
    // rg could not read. Its own diagnostic supplies the failing operation.
    for (const input of batch) formatter.markError(input.stdin ? "<stdin>" : input.label);
  }
  for (const input of batch) formatter.finishFile(input.stdin ? "<stdin>" : input.label);
}

export async function runGrep(argv) {
  if (process.env.GREP2RG_EXPERIMENTAL_RG !== "1") return runInherited("grep", argv);
  // Node replaces invalid UTF-8 in cwd names. Resolving operands against that
  // decoded path could classify a different directory from the one inherited
  // by rg. Native GNU inherits the exact OS cwd, including deleted directories.
  try {
    process.cwd();
    if (!isUtf8(readlinkSync("/proc/self/cwd", { encoding: "buffer" }))) return runInherited("grep", argv);
  } catch { return runInherited("grep", argv); }
  const plan = planGrep(argv);
  if (plan.backend === "grep" || !rgAvailable()) return runInherited("grep", argv);
  if (plan.mode === "direct") return runInherited("rg", plan.args);
  const prepared = await prepare(plan, argv);
  if (prepared.backend === "grep") return runInherited("grep", argv);
  const { options, inputs, matchingArgs, seekableInput } = prepared;
  const state = { child: null, searchError: false, protocolError: false, brokenPipe: false, waitingDrain: false, signal: null, seekableInput };
  const signalStatuses = { SIGHUP: 129, SIGINT: 130, SIGTERM: 143 };
  let killTimer;
  const signalHandlers = Object.entries(signalStatuses).map(([signal]) => {
    const handler = () => {
      state.signal ??= signal;
      process.stdout.destroy();
      const child = state.child;
      child?.kill(signal);
      // A paused search pipe must not keep cancellation waiting for its drain.
      child?.stdout?.destroy();
      if (child && !killTimer) {
        killTimer = setTimeout(() => { if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL"); }, 250);
        killTimer.unref();
      }
    };
    process.on(signal, handler);
    return [signal, handler];
  });
  const pipeError = (error) => {
    if (error.code !== "EPIPE") throw error;
    state.brokenPipe = true; state.child?.kill("SIGTERM");
  };
  const write = (data) => {
    if (state.brokenPipe || state.signal) return;
    if (!process.stdout.write(data) && state.child && !state.waitingDrain) {
      const child = state.child;
      state.waitingDrain = true;
      child.stdout.pause();
      process.stdout.once("drain", () => { state.waitingDrain = false; child.stdout.resume(); });
    }
  };
  process.stdout.on("error", pipeError);
  state.waitForOutput = async () => {
    if (!process.stdout.writableNeedDrain || state.brokenPipe || state.signal) return;
    await new Promise((resolve) => {
      const done = () => {
        for (const event of ["drain", "error", "close"]) process.stdout.removeListener(event, done);
        for (const signal of Object.keys(signalStatuses)) process.removeListener(signal, done);
        resolve();
      };
      for (const event of ["drain", "error", "close"]) process.stdout.once(event, done);
      for (const signal of Object.keys(signalStatuses)) process.once(signal, done);
    });
  };
  const formatter = createGrepFormatter(options, { stdout: write, tty: Boolean(process.stdout.isTTY) });
  const args = ["--no-config", "--no-ignore", "--hidden", "--threads", "1", "--json", "--color", "never", "--text", ...matchingArgs];
  if (options.filesMode || options.quiet) args.push("--max-count", "1");
  else if (options.maxCount !== null) args.push("--max-count", String(options.maxCount));
  if (!options.filesMode && !options.count && !options.quiet) {
    if (options.beforeContext) args.push("--before-context", String(options.beforeContext));
    if (options.afterContext) args.push("--after-context", String(options.afterContext));
  }
  try {
    for (const batch of chunks(inputs)) {
      if (state.signal || state.brokenPipe || (options.quiet && formatter.selected)) break;
      await searchBatch(batch, args, formatter, options, state);
    }
    const result = formatter.finish();
    if (state.signal) return signalStatuses[state.signal];
    if (state.brokenPipe) return 141;
    if (seekableInput && !state.searchError && !state.protocolError) {
      try { await seekableInput.advance(options.maxCount, () => Boolean(state.signal) || state.brokenPipe); }
      catch (error) {
        state.searchError = true;
        if (!options.silent) process.stderr.write(`grep: seekable stdin cursor: ${error.message}\n`);
      }
    }
    if (options.quiet && result.selected) return 0;
    return state.searchError || state.protocolError ? 2 : result.status;
  } finally {
    seekableInput?.close();
    // Leave the handler active until pending output callbacks have completed;
    // a short pipeline consumer may close after the final search event.
    if (!state.signal) await new Promise((resolve) => process.stdout.write(Buffer.alloc(0), resolve));
    process.stdout.removeListener("error", pipeError);
    for (const [signal, handler] of signalHandlers) process.removeListener(signal, handler);
    if (killTimer) clearTimeout(killTimer);
    if (state.signal) return signalStatuses[state.signal];
    if (state.brokenPipe) return 141;
  }
}
