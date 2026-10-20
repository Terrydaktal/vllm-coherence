// GNU grep's option language is parsed here; filesystem walking and rendering
// live in separate modules. Qualified matching uses ripgrep; unqualified GNU
// semantics retain the original GNU invocation before input or output begins.

const LONG_OPTIONS = {
  "extended-regexp": ["E", "none"], "fixed-strings": ["F", "none"],
  "basic-regexp": ["G", "none"], "perl-regexp": ["P", "none"],
  regexp: ["e", "required"], file: ["f", "required"],
  "ignore-case": ["i", "none"], "no-ignore-case": ["no-ignore-case", "none"],
  "word-regexp": ["w", "none"], "line-regexp": ["x", "none"],
  "null-data": ["z", "none"], "no-messages": ["s", "none"],
  "invert-match": ["v", "none"], version: ["V", "none"], help: ["help", "none"],
  "max-count": ["m", "required"], "byte-offset": ["b", "none"],
  "line-number": ["n", "none"], "line-buffered": ["line-buffered", "none"],
  "with-filename": ["H", "none"], "no-filename": ["h", "none"],
  label: ["label", "required"], "only-matching": ["o", "none"],
  quiet: ["q", "none"], silent: ["q", "none"],
  "binary-files": ["binary-files", "required"], text: ["a", "none"],
  directories: ["d", "required"], devices: ["D", "required"],
  recursive: ["r", "none"], "dereference-recursive": ["R", "none"],
  include: ["include", "required"], exclude: ["exclude", "required"],
  "exclude-from": ["exclude-from", "required"], "exclude-dir": ["exclude-dir", "required"],
  "files-without-match": ["L", "none"], "files-with-matches": ["l", "none"],
  count: ["c", "none"], "initial-tab": ["T", "none"], null: ["Z", "none"],
  "before-context": ["B", "required"], "after-context": ["A", "required"],
  context: ["C", "required"], "group-separator": ["group-separator", "required"],
  "no-group-separator": ["no-group-separator", "none"],
  color: ["color", "optional"], colour: ["color", "optional"], binary: ["U", "none"],
};
const SHORT_VALUES = new Set(["e", "f", "m", "d", "D", "A", "B", "C"]);
const SHORT_FLAGS = new Set("EFGPiywxzsvVbnHhoqaIrRLlcTZU".split(""));
const MATCHERS = { E: "extended", F: "fixed", G: "basic", P: "perl" };
const INTEGER_FIELDS = { m: "maxCount", A: "afterContext", B: "beforeContext", C: "context" };

function integer(value) {
  if (!/^\+?\d+$/.test(value)) return undefined;
  const number = BigInt(value);
  return Number(number > BigInt(Number.MAX_SAFE_INTEGER) ? BigInt(Number.MAX_SAFE_INTEGER) : number);
}

/** Parse GNU grep argv, including getopt permutation and long abbreviations. */
export function parseGrepOptions(argv, { env = process.env } = {}) {
  const p = {
    matcher: "basic", patternSources: [], paths: [], recursive: false, follow: false,
    directories: "read", devices: "read", devicesExplicit: false,
    ignoreCase: false, wordRegexp: false, lineRegexp: false, nullData: false,
    invertMatch: false, maxCount: null, byteOffset: false, lineNumber: false,
    filename: undefined, label: "(standard input)", onlyMatching: false, quiet: false,
    filesMode: null, count: false, initialTab: false, nullNames: false,
    binary: "binary", binaryMode: false, lineBuffered: false, color: "never", silent: false,
    beforeContext: 0, afterContext: 0, contextRequested: false, groupSeparator: "--",
    include: [], exclude: [], excludeDir: [], excludeFiles: [], filters: [],
    help: false, version: false, errors: [], errorExitCode: 2, numericNative: false,
  };
  let options = true, explicitPatterns = false, selectedMatcher;
  let explicitBeforeContext = false, explicitAfterContext = false;
  const fail = (message) => p.errors.push(message);
  const set = (flag, value) => {
    if (Object.hasOwn(MATCHERS, flag)) {
      if (selectedMatcher && selectedMatcher !== flag) fail("conflicting matchers specified");
      selectedMatcher = flag;
      p.matcher = MATCHERS[flag];
      return;
    }
    if (Object.hasOwn(INTEGER_FIELDS, flag)) {
      const num = integer(value);
      if (num === undefined) { fail(`invalid ${flag === "m" ? "max count" : "context length"} argument: ${value}`); return; }
      // Keep runtime counters bounded, but never substitute the saturated JS
      // value for the original GNU argument when its exact integer is lost.
      if (BigInt(value) > BigInt(Number.MAX_SAFE_INTEGER)) p.numericNative = true;
      if (flag !== "m") p.contextRequested = true;
      // GNU treats -C/-NUM as defaults for each side. An explicit -A/-B
      // remains authoritative even when generic context occurs later.
      if (flag === "C") {
        if (!explicitBeforeContext) p.beforeContext = num;
        if (!explicitAfterContext) p.afterContext = num;
      } else {
        p[INTEGER_FIELDS[flag]] = num;
        if (flag === "A") explicitAfterContext = true;
        if (flag === "B") explicitBeforeContext = true;
      }
      return;
    }
    switch (flag) {
      case "e": explicitPatterns = true; p.patternSources.push({ type: "regexp", value }); break;
      case "f": explicitPatterns = true; p.patternSources.push({ type: "file", value }); break;
      case "i": case "y": p.ignoreCase = true; break;
      case "no-ignore-case": p.ignoreCase = false; break;
      case "w": p.wordRegexp = true; break;
      case "x": p.lineRegexp = true; break;
      case "z": p.nullData = true; break;
      case "s": p.silent = true; break;
      case "v": p.invertMatch = true; break;
      case "V": p.version = true; break;
      case "help": p.help = true; break;
      case "b": p.byteOffset = true; break;
      case "n": p.lineNumber = true; break;
      case "line-buffered": p.lineBuffered = true; break;
      case "H": p.filename = true; break;
      case "h": p.filename = false; break;
      case "label": p.label = value; break;
      case "o": p.onlyMatching = true; break;
      case "q": p.quiet = true; break;
      case "a": p.binary = "text"; break;
      case "I": p.binary = "without-match"; break;
      case "binary-files":
        if (["binary", "text", "without-match"].includes(value)) p.binary = value;
        else fail(`unknown binary-files type: ${value}`);
        break;
      case "d":
        if (["read", "recurse", "skip"].includes(value)) {
          p.directories = value; p.recursive = value === "recurse";
        } else { p.errorExitCode = 1; fail(`invalid argument '${value}' for '--directories'`); }
        break;
      case "D":
        if (["read", "skip"].includes(value)) { p.devices = value; p.devicesExplicit = true; }
        else { p.errorExitCode = 1; fail(`unknown devices method: ${value}`); }
        break;
      case "r": case "R":
        p.recursive = true; p.directories = "recurse";
        if (flag === "R") p.follow = true;
        break;
      case "include": case "exclude":
        p[flag].push(value); p.filters.push({ type: flag, value }); break;
      case "exclude-dir":
        p.excludeDir.push(value); p.filters.push({ type: "exclude-dir", value }); break;
      case "exclude-from":
        p.excludeFiles.push(value); p.filters.push({ type: "exclude-file", value }); break;
      case "L": p.filesMode = "without"; break;
      case "l": p.filesMode = "with"; break;
      case "c": p.count = true; break;
      case "T": p.initialTab = true; break;
      case "Z": p.nullNames = true; break;
      case "group-separator": p.groupSeparator = value; break;
      case "no-group-separator": p.groupSeparator = null; break;
      case "color":
        if (value === undefined) p.color = "auto";
        else if (["never", "auto", "always"].includes(value)) p.color = value;
        else p.help = true; // GNU grep prints help and succeeds for an invalid WHEN.
        break;
      case "U": p.binaryMode = true; break;
    }
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (typeof arg !== "string") { fail("arguments must be strings"); continue; }
    if (options && arg === "--") { options = false; continue; }
    if (options && arg.startsWith("--")) {
      const eq = arg.indexOf("="), name = arg.slice(2, eq < 0 ? undefined : eq);
      let value = eq < 0 ? undefined : arg.slice(eq + 1);
      const names = Object.hasOwn(LONG_OPTIONS, name) ? [name] : Object.keys(LONG_OPTIONS).filter((key) => key.startsWith(name));
      if (!names.length) { fail(`unrecognized option '${arg}'`); continue; }
      const forms = new Set(names.map((key) => LONG_OPTIONS[key].join(":")));
      if (forms.size > 1) { fail(`option '--${name}' is ambiguous`); continue; }
      const [flag, kind] = LONG_OPTIONS[names[0]];
      if (kind === "none" && value !== undefined) { fail(`option '--${name}' doesn't allow an argument`); continue; }
      if (kind === "required" && value === undefined) value = argv[++index];
      if (kind === "required" && value === undefined) { fail(`option '--${name}' requires an argument`); continue; }
      set(flag, value);
    } else if (options && arg.startsWith("-") && arg !== "-") {
      for (let offset = 1; offset < arg.length; offset++) {
        const flag = arg[offset];
        if (/\d/.test(flag)) {
          let end = offset + 1;
          while (/\d/.test(arg[end] ?? "")) end++;
          // Every contiguous digit run is a new generic-context option. A
          // letter ends that run: GNU -1n2 means context 2, not context 12.
          set("C", arg.slice(offset, end)); offset = end - 1;
        } else if (SHORT_VALUES.has(flag)) {
          const value = arg.slice(offset + 1) || argv[++index];
          if (value === undefined) fail(`option requires an argument -- '${flag}'`);
          else set(flag, value);
          break;
        } else if (SHORT_FLAGS.has(flag)) set(flag);
        else fail(`invalid option -- '${flag}'`);
      }
    } else {
      p.paths.push(arg);
      if (env.POSIXLY_CORRECT !== undefined) options = false;
    }
  }
  if (!explicitPatterns && !p.help && !p.version) {
    if (p.paths.length) p.patternSources.push({ type: "regexp", value: p.paths.shift() });
    else fail("missing search pattern");
  }
  // GNU recursively skips devices unless -D explicitly requests reading them.
  if (p.recursive && !p.follow && !p.devicesExplicit) p.devices = "skip";
  return p;
}

/** Resolve -e/-f in order. readFile may return strings or Buffers, including '-'. */
export async function resolveGrepPatterns(options, readFile) {
  const patterns = [];
  for (const source of options.patternSources) {
    if (source.type === "regexp") patterns.push(...source.value.split("\n"));
    else {
      const value = await readFile(source.value);
      const text = typeof value === "string" ? value : value.toString("utf8");
      if (text.length) {
        const lines = text.split("\n");
        if (lines.at(-1) === "") lines.pop();
        patterns.push(...lines);
      }
    }
  }
  return [...new Set(patterns)];
}

const escapeRegex = (value) => value.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&").replaceAll("\0", "\\x00");
const GNU_CHARACTER_CLASSES = new Set(["alnum", "alpha", "blank", "cntrl", "digit", "graph", "lower", "print", "punct", "space", "upper", "xdigit"]);

// Bracket expressions are almost shared, but backslashes in GNU bracket
// expressions are ordinary characters; PCRE2 would interpret them as escapes.
function bracket(pattern, start) {
  let end = start + 1;
  if (pattern[end] === "^") end++;
  if (pattern[end] === "]") end++;
  for (; end < pattern.length; end++) {
    if (pattern[end] === "[" && [":", ".", "="].includes(pattern[end + 1])) {
      const tag = pattern[end + 1], close = pattern.indexOf(`${tag}]`, end + 2);
      if (tag === ":" && close >= 0 && !GNU_CHARACTER_CLASSES.has(pattern.slice(end + 2, close)))
        throw new Error("Invalid character class name");
      if (close >= 0) { end = close + 1; continue; }
    }
    if (pattern[end] === "]") break;
  }
  if (end >= pattern.length) throw new Error("Unmatched [, [^, [:, [., or [=");
  let inner = pattern.slice(start + 1, end).replaceAll("\\", "\\\\");
  // C-locale single-character collating/equivalence elements. Non-C collation
  // requires locale-specific tables and is reported by grepCompatibilityNotes.
  inner = inner.replace(/\[([.=])([^\]]+)\1\]/g, (_, kind, value) => {
    if ([...value].length !== 1) throw new Error(`Invalid ${kind === "." ? "collation character" : "equivalence class"}`);
    return value.replace(/[\\\]\-^]/g, "\\$&");
  });
  return { value: `[${inner}]`, end };
}

/** Translate GNU basic/extended regex syntax to PCRE2 without shell parsing. */
export function translateGrepPattern(pattern, matcher = "basic", { bracketTransform, caseTransform, wordTransform } = {}) {
  if (matcher === "fixed") return escapeRegex(pattern);
  if (matcher === "perl") return pattern.replaceAll("\0", "\\x00");
  const basic = matcher === "basic";
  let result = "", branchStart = true, anchorAllowed = true;
  for (let index = 0; index < pattern.length; index++) {
    let char = pattern[index];
    if (caseTransform && char.codePointAt(0) >= 0xd800 && char.codePointAt(0) <= 0xdbff && index + 1 < pattern.length) {
      char = String.fromCodePoint(pattern.codePointAt(index));
      index += char.length - 1;
    }
    if (char === "[") {
      const value = bracket(pattern, index); result += bracketTransform ? bracketTransform(pattern.slice(index, value.end + 1)) : value.value;
      index = value.end; branchStart = anchorAllowed = false; continue;
    }
    if (char === "\\") {
      const next = pattern[++index];
      if (next === undefined) throw new Error("Trailing backslash");
      if (basic && "()+?|{}".includes(next)) {
        result += next; branchStart = anchorAllowed = next === "(" || next === "|"; continue;
      }
      if (/[1-9]/.test(next)) result += `\\${next}`;
      else if (next === "<") result += wordTransform ? wordTransform(next) : "(?<!\\w)(?=\\w)";
      else if (next === ">") result += wordTransform ? wordTransform(next) : "(?<=\\w)(?!\\w)";
      else if (next === "`") result += "^";
      else if (next === "'") result += "$";
      else if ("wWsSbB".includes(next)) result += wordTransform ? wordTransform(next) : `\\${next}`;
      else result += caseTransform && /[a-z]|[^\x00-\x7f]/i.test(next) ? caseTransform(next) : escapeRegex(next);
      branchStart = anchorAllowed = false; continue;
    }
    if (basic && "+?|(){}".includes(char)) result += `\\${char}`;
    else if (basic && char === "^" && !anchorAllowed) result += "\\^";
    else if (basic && char === "$" && index !== pattern.length - 1 && !/^\\[)|]/.test(pattern.slice(index + 1))) result += "\\$";
    else if (basic && char === "*" && branchStart) result += "\\*";
    else if (char === "\0") result += "\\x00";
    else result += caseTransform && /[a-z]|[^\x00-\x7f]/i.test(char) ? caseTransform(char) : char;
    if (char === "^" && branchStart && anchorAllowed) { anchorAllowed = false; continue; }
    branchStart = anchorAllowed = !basic && (char === "(" || char === "|");
  }
  return result;
}

// NUL records can contain LF bytes. rg/PCRE2's usual multiline anchors and dot
// therefore need an explicit complete-record interpretation for GNU BRE/ERE.
// Escaped anchors and bracket literals are retained; POSIX class subexpressions
// do not close their surrounding bracket while scanning.
export function nulRecordPattern(pattern) {
  let result = "", inClass = false, classPosition = 0;
  for (let index = 0; index < pattern.length; index++) {
    const char = pattern[index];
    if (char === "\\") { result += char + (pattern[++index] ?? ""); if (inClass) classPosition = 2; continue; }
    if (!inClass && char === "[") { inClass = true; classPosition = 0; result += char; continue; }
    if (inClass) {
      if (char === "[" && pattern[index + 1] === ":") {
        const end = pattern.indexOf(":]", index + 2);
        if (end >= 0) { result += pattern.slice(index, end + 2); index = end + 1; classPosition = 2; continue; }
      }
      if (char === "]" && classPosition >= 2) inClass = false;
      classPosition = char === "^" && classPosition === 0 ? 1 : 2;
      result += char; continue;
    }
    result += char === "^" ? "\\A" : char === "$" ? "\\z" : char;
  }
  return `(?s-m:${result})`;
}

/** Arguments controlling matching only; traversal and printing are independent. */
export function buildGrepMatcherArgs(options, patterns, { env = process.env } = {}) {
  const locale = env.LC_ALL || env.LC_CTYPE || env.LANG || "C";
  const ascii = ["C", "POSIX"].includes(locale);
  const args = ["--encoding", "none", ascii ? "--no-unicode" : "--unicode"];
  if (options.ignoreCase) args.push("--ignore-case");
  if (options.nullData) args.push("--null-data");
  // GNU -w uses adjacent word-character tests and can match non-word patterns.
  // A pair of \b boundaries would incorrectly reject a pattern such as '@'.
  const useFixed = patterns.length > 0 && options.matcher === "fixed" && !options.wordRegexp && !(options.nullData && options.lineRegexp);
  args.push(useFixed ? "--fixed-strings" : "--pcre2");
  if (options.lineRegexp && !options.nullData) args.push("--line-regexp");
  if (options.invertMatch) args.push("--invert-match");
  if (options.matcher === "perl" && patterns.length > 1) throw new Error("the -P option only supports a single pattern");
  if (!patterns.length) args.push("-e", "(?!)");
  const translatedPatterns = [];
  for (const pattern of patterns) {
    let translated = useFixed ? pattern : translateGrepPattern(pattern, options.matcher);
    if (options.wordRegexp && !options.lineRegexp) translated = `(?<!\\w)(?:${translated})(?!\\w)`;
    if (options.nullData && !useFixed) {
      translated = nulRecordPattern(translated);
      if (options.lineRegexp) translated = `\\A(?:${translated})\\z`;
    }
    translatedPatterns.push(translated);
  }
  // rg joins separate PCRE patterns into one alternation, which renumbers
  // captures in later -e operands. GNU backreferences are local to each
  // pattern; a branch-reset group preserves those indices in the union.
  if (!useFixed && options.matcher !== "perl" && patterns.length > 1 && patterns.some((pattern) => /\\[1-9]/.test(pattern)))
    args.push("-e", `(?|${translatedPatterns.map((pattern) => `(?:${pattern})`).join("|")})`);
  else for (const translated of translatedPatterns) args.push("-e", translated);
  return args;
}

/** Deliberate limits of rg/PCRE2 compared with a locale-dependent POSIX engine. */
export function grepCompatibilityNotes(options, patterns, { env = process.env } = {}) {
  const locale = env.LC_ALL || env.LC_CTYPE || env.LANG || "C";
  const notes = [];
  if (!["C", "POSIX"].includes(locale) && patterns.some((pattern) => /\[[.=]/.test(pattern)))
    notes.push("Non-C locale collating/equivalence elements use Unicode character identity rather than libc collation tables.");
  if (options.ignoreCase && !["C", "POSIX"].includes(locale))
    notes.push("Unicode case folding is performed by ripgrep/PCRE2 rather than the host libc locale tables.");
  return notes;
}

/**
 * Decide whether matching needs GNU before opening stdin or emitting output.
 * This deliberately selects the reference implementation for unqualified
 * syntax instead of discovering an incompatible match after output has begun.
 * Runtime/output handling can add its own independent fallback reasons.
 */
export function grepMatcherFallbackReasons(options, { env = process.env } = {}) {
  const reasons = [];
  const locale = env.LC_ALL || env.LC_CTYPE || env.LANG || "C";
  const ascii = ["C", "POSIX"].includes(locale);
  const patterns = options.patternSources.filter((source) => source.type === "regexp").flatMap((source) => source.value.split("\n"));
  if (options.patternSources.some((source) => source.type === "file"))
    reasons.push("GNU pattern-file byte and error semantics");
  if (options.errors.length) reasons.push("GNU option error diagnostics");
  if (options.numericNative) reasons.push("GNU numeric option exceeds exact JavaScript integer range");
  if (options.ignoreCase && !ascii)
    reasons.push("GNU locale case folding differs from ripgrep/PCRE2");
  // Perl permits inline modifiers and PCRE2-specific newline conventions;
  // those cannot use the BRE/ERE complete-NUL-record translation.
  if (options.nullData && options.matcher === "perl")
    reasons.push("GNU NUL-record Perl matching semantics differ from ripgrep");
  if (!ascii && options.matcher !== "fixed" && !/[.-]UTF-?8$/i.test(locale))
    reasons.push("GNU non-UTF-8 locale regex encoding differs from PCRE2 UTF-8");
  if (options.matcher === "perl") {
    // GNU grep's handling of malformed UTF-8 differs across GNU/PCRE2
    // releases. The Pi-opsec GNU 3.11 / rg 14.1.1 pair disagrees on a UTF-8
    // surrogate byte sequence even with -a; decide before reading the subject.
    if (!ascii) reasons.push("GNU -P malformed UTF-8 behavior depends on the GNU/PCRE2 release");
    if (new Set(patterns).size > 1) reasons.push("GNU -P accepts only one distinct pattern");
    if (!ascii && patterns.some((pattern) => /\\[dD]/.test(pattern)))
      reasons.push("GNU -P uses ASCII digit classes in UTF-8 locales");
    if (options.wordRegexp && !ascii)
      reasons.push("GNU locale word boundaries differ from PCRE2 Unicode classes");
    return [...new Set(reasons)];
  }
  if (options.matcher === "fixed") {
    if (options.wordRegexp && !ascii)
      reasons.push("GNU locale word boundaries differ from PCRE2 Unicode classes");
    if (patterns.some((pattern) => pattern.includes("\0"))) reasons.push("NUL pattern argument cannot be passed to ripgrep");
    if (new Set(patterns).size > 1 && (options.onlyMatching || options.color === "always"))
      reasons.push("GNU fixed patterns use leftmost-longest match selection for extraction and highlighting");
    return [...new Set(reasons)];
  }
  if (options.onlyMatching) reasons.push("GNU BRE/ERE uses leftmost-longest matching for -o");
  if (options.color === "always") reasons.push("GNU BRE/ERE uses leftmost-longest matching for highlighted spans");
  if (options.wordRegexp && !ascii)
    reasons.push("GNU locale word boundaries differ from PCRE2 Unicode classes");
  for (const pattern of patterns) {
    if (pattern.includes("\0")) reasons.push("NUL pattern argument cannot be passed to ripgrep");
    if (/\[[.=]/.test(pattern)) reasons.push("GNU POSIX collation and equivalence elements");
    if (!ascii && (/\[/.test(pattern) || /\\[wWsSbB<>]/.test(pattern)))
      reasons.push("GNU locale character classes, ranges or word boundaries");
    if (options.matcher === "extended" && /\(\?/.test(pattern))
      reasons.push("PCRE2 group extensions are not GNU ERE syntax");
    if (/\\[1-9]\d/.test(pattern)) reasons.push("GNU numbered-backreference parsing differs from PCRE2");
    // Nested/leading quantifiers and invalid intervals can have GNU warnings or
    // interpretation unlike PCRE2. Escaped pairs are inspected in order so a
    // literal backslash cannot accidentally turn into a regex operator.
    let priorAtom = false, groupDepth = 0;
    for (let index = 0; index < pattern.length; index++) {
      let char = pattern[index], escaped = false;
      if (char === "[") {
        try { const value = bracket(pattern, index); index = value.end; priorAtom = true; }
        catch { reasons.push("GNU invalid bracket-expression diagnostics"); }
        continue;
      }
      if (char === "\\") {
        char = pattern[++index]; escaped = true;
        if (char === undefined) { reasons.push("GNU trailing-backslash diagnostics"); break; }
        const special = "123456789wWsSbB<>`'\\.^$*[]";
        const operator = options.matcher === "basic" && "()+?|{}".includes(char);
        if (!operator && !special.includes(char) && !"()+?|{}".includes(char))
          reasons.push("GNU stray-backslash warning semantics");
        if (!operator) { priorAtom = !"bB<>`'".includes(char); continue; }
      }
      const operator = options.matcher === "basic" ? escaped : !escaped;
      if (operator && char === "(") { groupDepth++; priorAtom = false; }
      else if (operator && char === ")") {
        if (--groupDepth < 0) reasons.push("GNU unmatched-group diagnostics");
        priorAtom = true;
      } else if (operator && char === "|") priorAtom = false;
      else if ((char === "*" && !escaped) || (operator && "+?".includes(char))) {
        if (!priorAtom) reasons.push("GNU leading or repeated quantifier semantics");
        priorAtom = false;
      } else if (operator && char === "{") {
        const rest = pattern.slice(index + 1);
        const interval = options.matcher === "basic" ? /^(\d+)(?:,(\d*))?\\\}/.exec(rest) : /^(\d+)(?:,(\d*))?\}/.exec(rest);
        if (!interval || !priorAtom) reasons.push("GNU interval syntax or warning semantics");
        else {
          if (Number(interval[1]) > 32767 || (interval[2] && Number(interval[2]) > 32767))
            reasons.push("GNU interval repetition limit");
          if (interval[2] && Number(interval[1]) > Number(interval[2])) reasons.push("GNU invalid interval ordering");
          index += interval[0].length;
        }
        priorAtom = false;
      } else if (char !== "^") priorAtom = true;
    }
    if (groupDepth !== 0) reasons.push("GNU unmatched-group diagnostics");
    try { translateGrepPattern(pattern, options.matcher); }
    catch { reasons.push("GNU invalid regex diagnostics"); }
  }
  return [...new Set(reasons)];
}

export const grepLongOptionNames = Object.freeze(Object.keys(LONG_OPTIONS));
