import { lstatSync, statSync } from "node:fs";
import { basename, isAbsolute, resolve } from "node:path";
import { validateEvaluation } from "./find-evaluate.mjs";
import { inheritedStdio, unusedInputFd } from "./find-descriptors.mjs";

const native = (args, reason) => ({ eligible: false, args, reason });
const ascii = (value) => typeof value === "string" && /^[\x01-\x7f]*$/.test(value);
const isExpression = (value) => value.startsWith("-") || ["!", "(", ")", ","].includes(value);
const TYPE_METHODS = { f: "isFile", d: "isDirectory", l: "isSymbolicLink", p: "isFIFO", s: "isSocket", b: "isBlockDevice", c: "isCharacterDevice" };
const TYPES = Object.keys(TYPE_METHODS);
export const MAX_EXPRESSION_NODES = 256;
export const MAX_EXPRESSION_DEPTH = 64;
const MAX_FILTER_ALTERNATIVES = 64;
const ALL = { names: null, types: null }, NONE = { names: [], types: [] };

export function normalizedRoot(value, allowUnicode = false) {
  if (!value || (allowUnicode ? typeof value !== "string" || value.includes("\0") : !ascii(value))) return false;
  if ([".", "./", "/"].includes(value)) return true;
  let path = value;
  if (path.startsWith("./")) path = path.slice(2);
  else if (isAbsolute(path)) path = path.slice(1);
  if (path.endsWith("/")) path = path.slice(0, -1);
  return path.length > 0 && path.split("/").every((part) => part && part !== "." && part !== "..");
}

export function rootName(root) {
  return Buffer.from(root === "/" ? "/" : [".", "./"].includes(root) ? "." : basename(root));
}

function byteLocale() {
  const locale = process.env.LC_ALL || process.env.LC_CTYPE || process.env.LANG || "C";
  return locale === "C" || locale === "POSIX";
}

function safePattern(pattern, fullPath, allowQuestion) {
  return ascii(pattern) && pattern.length > 0 && pattern.length <= 8192 &&
    !/[\[\]{}\\]/.test(pattern) && (fullPath || !pattern.includes("/")) &&
    (allowQuestion || !pattern.includes("?"));
}

export function safeName(pattern, allowQuestion = false) {
  return safePattern(pattern, false, allowQuestion);
}

const fold = (byte) => byte >= 65 && byte <= 90 ? byte + 32 : byte;
function matchBytes(name, pattern, ignoreCase) {
  let cursor = 0, index = 0, star = -1, retry = 0;
  while (cursor < name.length) {
    if (index < pattern.length && pattern[index] !== 42 &&
        (pattern[index] === 63 || (ignoreCase ? fold(pattern[index]) === fold(name[cursor]) : pattern[index] === name[cursor]))) {
      cursor++; index++;
    } else if (pattern[index] === 42) { star = index++; retry = cursor; }
    else if (star >= 0) { index = star + 1; cursor = ++retry; }
    else return false;
  }
  while (pattern[index] === 42) index++;
  return index === pattern.length;
}

export function nameMatches(name, pattern, ignoreCase = false) {
  return matchBytes(name, Buffer.from(pattern), ignoreCase);
}

const matcherPatterns = new WeakMap();
export function evaluateFind(expression, { name, path, entry }, actions = []) {
  const evaluate = (node) => {
    switch (node.kind) {
      case "and": return evaluate(node.left) && evaluate(node.right);
      case "or": return evaluate(node.left) || evaluate(node.right);
      case "not": return !evaluate(node.child);
      case "comma": evaluate(node.left); return evaluate(node.right);
      case "true": return true;
      case "false": return false;
      case "type": return typeMatches(entry, node.type);
      case "print": actions.push(node.action); return true;
      case "name": case "path": {
        let pattern = matcherPatterns.get(node);
        if (!pattern) { pattern = Buffer.from(node.pattern); matcherPatterns.set(node, pattern); }
        return matchBytes(node.kind === "name" ? name : path, pattern, node.ignoreCase);
      }
      default: throw new Error("unknown qualified expression");
    }
  };
  const value = evaluate(expression);
  return { value, actions };
}

function intersect(left, right) {
  const types = left.types === null ? right.types : right.types === null ? left.types : left.types.filter((type) => right.types.includes(type));
  let names;
  if (left.names === null) names = right.names;
  else if (right.names === null) names = left.names;
  else names = left.names.length <= right.names.length ? left.names : right.names;
  return types?.length === 0 || names?.length === 0 ? NONE : { types, names };
}

function union(left, right) {
  const types = left.types === null || right.types === null ? null : [...new Set([...left.types, ...right.types])];
  let names = left.names === null || right.names === null ? null : [...new Map([...left.names, ...right.names].map((node) => [`${node.ignoreCase}:${node.pattern}`, node])).values()];
  if (names?.length > MAX_FILTER_ALTERNATIVES) names = null;
  return { types, names };
}

// Each summary is a superset of entries producing true, false, or output.
// Print conditions matter separately: a print can run even if its branch fails.
function summarize(node) {
  switch (node.kind) {
    case "name": return { yes: node.pattern.includes("?") ? ALL : { names: [node], types: null }, no: ALL, output: NONE };
    case "type": return { yes: { names: null, types: [node.type] }, no: { names: null, types: TYPES.filter((type) => type !== node.type) }, output: NONE };
    case "true": return { yes: ALL, no: NONE, output: NONE };
    case "false": return { yes: NONE, no: ALL, output: NONE };
    case "path": return { yes: ALL, no: ALL, output: NONE };
    case "gnu": return { yes: ALL, no: ALL, output: NONE };
    case "print": return { yes: ALL, no: NONE, output: ALL };
    case "not": { const child = summarize(node.child); return { yes: child.no, no: child.yes, output: child.output }; }
    default: {
      const left = summarize(node.left), right = summarize(node.right);
      if (node.kind === "and") return { yes: intersect(left.yes, right.yes), no: union(left.no, intersect(left.yes, right.no)), output: union(left.output, intersect(left.yes, right.output)) };
      if (node.kind === "or") return { yes: union(left.yes, intersect(left.no, right.yes)), no: intersect(left.no, right.no), output: union(left.output, intersect(left.no, right.output)) };
      return { yes: right.yes, no: right.no, output: union(left.output, right.output) };
    }
  }
}

/** Parse bounded pure listing expressions; unknown semantics remain GNU argv. */
function parseFastListing(argv) {
  if (!Array.isArray(argv) || !argv.every(ascii)) return native(argv, "non-ASCII or invalid argument bytes");
  let index = argv[0] === "-P" ? 1 : 0;
  const roots = [];
  while (index < argv.length && !isExpression(argv[index])) roots.push(argv[index++]);
  if (!roots.length) roots.push(".");
  if (!roots.every((root) => normalizedRoot(root))) return native(argv, "unnormalized starting paths");
  const options = { roots, minDepth: 0, maxDepth: null, names: [], type: null, action: "print" };
  while (["-mindepth", "-maxdepth"].includes(argv[index])) {
    const flag = argv[index++], value = argv[index++];
    if (!/^\d+$/.test(value ?? "") || value.length > 10 || Number(value) > 1000000)
      return native(argv, "GNU depth validation or unbounded depth");
    options[flag === "-mindepth" ? "minDepth" : "maxDepth"] = Number(value);
  }
  let count = 0;
  const printActions = new Set(), byteMatching = byteLocale();
  const node = (value) => {
    if (++count > MAX_EXPRESSION_NODES) throw new Error("expression node bound");
    return value;
  };
  const primary = (depth) => {
    if (depth > MAX_EXPRESSION_DEPTH) throw new Error("expression nesting bound");
    const flag = argv[index++];
    if (flag === "!" || flag === "-not") return node({ kind: "not", child: primary(depth + 1) });
    if (flag === "(") {
      if (argv[index] === ")") throw new Error("GNU empty expression group");
      const expression = comma(depth + 1);
      if (argv[index++] !== ")") throw new Error("GNU unmatched expression group");
      return expression;
    }
    if (["-print", "-print0"].includes(flag)) {
      const action = flag === "-print0" ? "print0" : "print";
      printActions.add(action); return node({ kind: "print", action });
    }
    if (["-true", "-false"].includes(flag)) return node({ kind: flag.slice(1) });
    if (flag === "-type") {
      const type = argv[index++];
      if (!/^[fdlpsbc]$/.test(type ?? "")) throw new Error("GNU file type syntax");
      return node({ kind: "type", type });
    }
    if (["-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename"].includes(flag)) {
      const pattern = argv[index++], ignoreCase = ["-iname", "-ipath", "-iwholename"].includes(flag);
      const fullPath = !["-name", "-iname"].includes(flag);
      if ((ignoreCase && !byteMatching) || !safePattern(pattern, fullPath, byteMatching))
        throw new Error("GNU filename pattern requires native locale or syntax");
      // GNU warns for some trailing-slash path patterns depending on the roots.
      // Let it preserve those diagnostics before either implementation traverses.
      if (fullPath && pattern.endsWith("/"))
        throw new Error("GNU trailing-slash path diagnostics require native execution");
      return node({ kind: fullPath ? "path" : "name", pattern, ignoreCase });
    }
    throw new Error("GNU expression, diagnostic or action outside the listing subset");
  };
  const and = (depth) => {
    let expression = primary(depth);
    while (index < argv.length && !["-o", "-or", ")", ","].includes(argv[index])) {
      if (["-a", "-and"].includes(argv[index])) index++;
      expression = node({ kind: "and", left: expression, right: primary(depth) });
    }
    return expression;
  };
  const or = (depth) => {
    let expression = and(depth);
    while (["-o", "-or"].includes(argv[index])) {
      index++; expression = node({ kind: "or", left: expression, right: and(depth) });
    }
    return expression;
  };
  const comma = (depth) => {
    let expression = or(depth);
    while (argv[index] === ",") { index++; expression = node({ kind: "comma", left: expression, right: or(depth) }); }
    return expression;
  };
  try {
    let expression = index < argv.length ? comma(0) : node({ kind: "true" });
    if (index !== argv.length) throw new Error("GNU expression syntax");
    if (!printActions.size) {
      printActions.add("print"); expression = node({ kind: "and", left: expression, right: node({ kind: "print", action: "print" }) });
    }
    options.expression = expression;
    options.printActions = [...printActions];
    options.action = printActions.has("print") ? "print" : "print0";
    options.candidateFilter = summarize(expression).output;
    options.names = options.candidateFilter.names?.filter((item) => !item.ignoreCase).map((item) => item.pattern) ?? [];
    options.type = options.candidateFilter.types?.length === 1 ? options.candidateFilter.types[0] : null;
    return { eligible: true, options };
  } catch (error) { return native(argv, error.message); }
}

const GNU_ARGUMENT_TESTS = new Set([
  "-amin", "-anewer", "-atime", "-cmin", "-cnewer", "-context", "-ctime", "-fstype", "-gid", "-group",
  "-ilname", "-iname", "-inum", "-iwholename", "-iregex", "-links", "-lname", "-mmin", "-mtime",
  "-name", "-newer", "-path", "-perm", "-regex", "-samefile", "-size", "-type", "-uid", "-used", "-user",
  "-wholename", "-ipath", "-xtype",
]);
const GNU_ZERO_TESTS = new Set(["-empty", "-nouser", "-nogroup", "-readable", "-writable", "-executable", "-true", "-false"]);
const GNU_POSITIONAL = new Set(["-daystart", "-follow", "-warn", "-nowarn", "-regextype"]);
const TRAVERSAL_OPTIONS = new Set(["-mindepth", "-maxdepth", "-depth", "-xdev", "-mount", "-noleaf", "-noignore_readdir_race"]);

function formatContains(format, directives) {
  // A doubled percent consumes its partner before directives are considered.
  for (let index = 0; index < format.length; index++) {
    if (format[index] === "\\") {
      if (format[index + 1] === "c") break;
      index++; continue;
    }
    if (format[index] !== "%") continue;
    if (format[index + 1] === "%") { index++; continue; }
    const directive = /^%[#0 +\-]*\d*(?:\.\d*)?([A-Za-z@])/.exec(format.slice(index));
    if (directive && directives.includes(directive[1])) return true;
  }
  return false;
}

function parseGnuListing(argv, configuration) {
  if (!Array.isArray(argv) || !argv.every((value) => typeof value === "string" && !value.includes("\0")))
    return native(argv, "invalid argument representation");
  let index = 0;
  const options = { roots: [], minDepth: 0, maxDepth: null, depth: false, xdev: false, dereference: "P",
    names: [], type: null, action: "print", mode: "gnu-evaluate", candidateFilter: ALL };
  while (["-P", "-H", "-L"].includes(argv[index]) || /^-O\d+$/.test(argv[index] ?? "")) {
    const flag = argv[index++];
    if (flag.length > 8192) return native(argv, "GNU optimization argument byte bound");
    if (flag.startsWith("-O")) options.optimization = flag;
    else options.dereference = flag.slice(1);
  }
  if (argv[index] === "--") index++;
  options.evaluationDereference = options.dereference;
  while (index < argv.length && !isExpression(argv[index])) options.roots.push(argv[index++]);
  if (!options.roots.length) options.roots.push(".");
  if (!options.roots.every((root) => normalizedRoot(root, true))) return native(argv, "unnormalized starting paths");
  while (TRAVERSAL_OPTIONS.has(argv[index])) {
    const flag = argv[index++];
    if (["-mindepth", "-maxdepth"].includes(flag)) {
      const value = argv[index++];
      if (!/^\d+$/.test(value ?? "") || value.length > 10 || Number(value) > 1000000)
        return native(argv, "GNU depth validation or unbounded depth");
      options[flag === "-mindepth" ? "minDepth" : "maxDepth"] = Number(value);
    } else if (flag === "-depth") options.depth = true;
    else if (["-xdev", "-mount"].includes(flag)) options.xdev = true;
  }
  const expressionStart = index;
  let nodes = 0;
  const printActions = new Set(), references = [], printfContext = [], controls = [];
  let accessTimeSensitive = false;
  const node = (value) => { if (++nodes > MAX_EXPRESSION_NODES) throw new Error("expression node bound"); return value; };
  const primary = (depth) => {
      if (depth > MAX_EXPRESSION_DEPTH) throw new Error("expression nesting bound");
      const flag = argv[index++];
      if (["-prune", "-quit"].includes(flag)) {
        controls.push({ index: index - 1 - expressionStart, kind: flag.slice(1) });
        return node({ kind: "gnu" });
      }
      if (["!", "-not"].includes(flag)) return node({ kind: "not", child: primary(depth + 1) });
      if (flag === "(") {
        if (argv[index] === ")") throw new Error("GNU empty expression group");
        const expression = comma(depth + 1);
        if (argv[index++] !== ")") throw new Error("GNU unmatched expression group");
        return expression;
      }
      const reference = ["-newer", "-anewer", "-cnewer", "-samefile"].includes(flag) || /^-newer[aBcm][aBcm]$/.test(flag);
      const withArgument = GNU_ARGUMENT_TESTS.has(flag) || /^-newer[aBcm][aBcmt]$/.test(flag);
      if (withArgument || flag === "-printf" || flag === "-regextype") {
        const value = argv[index++];
        if (value === undefined) throw new Error("GNU missing predicate argument");
        if (Buffer.byteLength(value) > 8192) throw new Error("GNU predicate argument byte bound");
        if (["-path", "-ipath", "-wholename", "-iwholename"].includes(flag) && value.endsWith("/"))
          throw new Error("GNU trailing-slash path diagnostics require native execution");
        if (flag === "-printf") {
          if (formatContains(value, ["d", "H", "P"])) printfContext.push(index - 1 - expressionStart);
          if (formatContains(value, ["a", "A"])) accessTimeSensitive = true;
          printActions.add("printf");
        }
        if (["-amin", "-atime", "-anewer", "-used"].includes(flag) || /^-newera[aBcmt]$/.test(flag)) accessTimeSensitive = true;
        if (reference) references.push({ path: value, accessTime: /^-newer[aBcm]a$/.test(flag), dereference: options.dereference });
        return node(flag === "-printf" ? { kind: "print", action: "printf" } :
          flag === "-type" && /^[fdlpsbc]$/.test(value) ? { kind: "type", type: value } : { kind: "gnu" });
      }
      if (["-print", "-print0", "-ls"].includes(flag)) { printActions.add(flag.slice(1)); return node({ kind: "print", action: flag.slice(1) }); }
      if (GNU_ZERO_TESTS.has(flag)) return node(["-true", "-false"].includes(flag) ? { kind: flag.slice(1) } : { kind: "gnu" });
      if (GNU_POSITIONAL.has(flag)) {
        if (flag === "-follow") options.dereference = "L";
        return node({ kind: "true" });
      }
      throw new Error("GNU action or traversal control requires native execution");
  };
  const and = (depth) => {
    let expression = primary(depth);
    while (index < argv.length && !["-o", "-or", ")", ","].includes(argv[index])) {
      if (["-a", "-and"].includes(argv[index])) index++;
      expression = node({ kind: "and", left: expression, right: primary(depth) });
    }
    return expression;
  };
  const or = (depth) => {
    let expression = and(depth);
    while (["-o", "-or"].includes(argv[index])) { index++; expression = node({ kind: "or", left: expression, right: and(depth) }); }
    return expression;
  };
  const comma = (depth) => {
    let expression = or(depth);
    while (argv[index] === ",") { index++; expression = node({ kind: "comma", left: expression, right: or(depth) }); }
    return expression;
  };
  try {
    let expression = index < argv.length ? comma(0) : node({ kind: "true" });
    if (index !== argv.length) throw new Error("GNU expression syntax");
    if (!printActions.size) expression = node({ kind: "and", left: expression, right: node({ kind: "print", action: "print" }) });
    // With -H, GNU's reference-file dereference policy must survive splitting
    // original roots from descendant roots in separate evaluation requests.
    const incremental = controls.length > 0;
    if (incremental && (options.depth || options.xdev || options.dereference !== "P" || printfContext.length))
      throw new Error("incremental controls require physical preorder and noncontextual output");
    if (!incremental && options.dereference === "H" && references.some(({ path }) => {
      try { return lstatSync(resolve(configuration.cwd ?? process.cwd(), path)).isSymbolicLink(); }
      catch { return true; }
    })) throw new Error("GNU -H symbolic-link reference context requires native execution");
    if (!incremental && references.some(({ path, accessTime, dereference }) => {
      if (!accessTime) return false;
      try {
        const referencePath = resolve(configuration.cwd ?? process.cwd(), path);
        const entry = lstatSync(referencePath);
        return entry.isSymbolicLink() || (["L", "H"].includes(dereference) ? statSync(referencePath) : entry).isDirectory();
      }
      catch { return true; }
    })) throw new Error("GNU directory or symlink reference access-time snapshot requires native execution");
    const outputTypes = summarize(expression).output.types;
    if (!incremental && accessTimeSensitive && (!outputTypes || outputTypes.some((type) => type !== "f")))
      throw new Error("GNU directory or symlink access-time snapshot requires native execution");
    const stdio = inheritedStdio(undefined, configuration.callerDescriptors ?? []);
    if (references.some(({ path }) => {
      const match = /^\/(?:dev\/fd|proc\/(?:self|thread-self)\/fd)\/(\d+)$/.exec(path);
      return match && Number(match[1]) >= 3 && stdio[Number(match[1])] === undefined;
    })) throw new Error("GNU closed caller descriptor reference requires native execution");
    options.evaluationInputFd = unusedInputFd(stdio);
    options.callerDescriptors = configuration.callerDescriptors ?? [];
    options.evaluationArgs = argv.slice(expressionStart);
    if (incremental) {
      options.mode = "gnu-incremental";
      options.controls = controls;
      options.implicitPrint = printActions.size === 0;
    }
    options.expression = expression;
    options.printfContext = printfContext;
    options.contextualPrintf = printfContext.length > 0;
    options.printActions = printActions.size ? [...printActions] : ["print"];
    options.action = options.printActions.includes("print") ? "print" : options.printActions[0];
    if (!validateEvaluation(options, configuration)) throw new Error("GNU predicate syntax, reference or dependency validation");
    if (options.evaluationHasWarnings && (options.dereference === "H" || options.contextualPrintf))
      throw new Error("GNU split-evaluation parse diagnostics require native execution");
    return { eligible: true, options };
  } catch (error) { return native(argv, error.message); }
}

/** Keep common byte-safe listings in-process; GNU owns richer predicate semantics. */
export function parseFindOptions(argv, configuration = {}) {
  const fast = parseFastListing(argv);
  return fast.eligible ? fast : parseGnuListing(argv, configuration);
}

export function typeMatches(entry, type) {
  return !type || entry[TYPE_METHODS[type]]();
}

function coveringGlob(names, ignoreCase) {
  if (names.length === 1) return names[0].pattern;
  const patterns = names.map((node) => ignoreCase ? node.pattern.toLowerCase() : node.pattern);
  const common = (values) => {
    let length = Math.min(...values.map((value) => value.length)), index = 0;
    while (index < length && values.every((value) => value[index] === values[0][index])) index++;
    return values[0].slice(0, index);
  };
  const prefix = common(patterns.map((pattern) => pattern.split("*")[0]));
  const suffix = common(patterns.map((pattern) => pattern.split("*").at(-1).split("").reverse().join(""))).split("").reverse().join("");
  // A shared prefix or suffix covers every alternative without requiring both
  // to fit non-overlapping portions of an exact literal filename.
  return prefix.length >= suffix.length ? `${prefix}*` : `*${suffix}`;
}

export function fdArguments(path, options) {
  const args = ["--hidden", "--no-ignore", "--no-ignore-parent", "--show-errors", "--print0", "--color", "never", "--threads", "8", "--absolute-path"];
  if (options.xdev) args.push("--one-file-system");
  if (options.dereference === "L") args.push("--follow");
  const filter = options.candidateFilter;
  const ignoreCase = filter.names?.some((node) => node.ignoreCase) ?? false;
  args.push(ignoreCase ? "--ignore-case" : "--case-sensitive");
  for (const type of filter.types ?? []) args.push("--type", type);
  if (options.minDepth) args.push("--min-depth", String(options.minDepth));
  if (options.maxDepth !== null) args.push("--max-depth", String(options.maxDepth));
  let pattern = "";
  if (filter.names?.length === 0 || filter.types?.length === 0) pattern = "\\A\\z.";
  else if (filter.names) { args.push("--glob"); pattern = coveringGlob(filter.names, ignoreCase); }
  args.push("--search-path", path, "--", pattern);
  return args;
}
