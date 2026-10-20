// GNU grep operand selection. Matching is performed by the caller's rg process;
// this module preserves the filesystem rules that differ from rg's defaults.
import { lstat, stat, opendir, readFile } from "node:fs/promises";
import { isUtf8 } from "node:buffer";
import { basename, resolve } from "node:path";

const quoteRegex = (value) => value.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
const classes = {
  alnum: "A-Za-z0-9", alpha: "A-Za-z", blank: " \\t", cntrl: "\\x00-\\x1f\\x7f",
  digit: "0-9", graph: "\\x21-\\x7e", lower: "a-z", print: "\\x20-\\x7e",
  punct: "!-/:-@[-`{-~", space: "\\t\\n\\v\\f\\r ", upper: "A-Z", xdigit: "A-Fa-f0-9",
};

function bracket(glob, start) {
  let index = start + 1;
  let source = "";
  if (glob[index] === "!" || glob[index] === "^") { source += "^"; index++; }
  if (glob[index] === "]") { source += "\\]"; index++; }
  const contentStart = index;
  for (; index < glob.length; index++) {
    const char = glob[index];
    if (char === "]") {
      if (index === contentStart && !source.includes("\\]")) return undefined;
      try { new RegExp(`[${source}]`, "u"); } catch { return { source: "(?!)", end: index }; }
      return { source: `[${source}]`, end: index };
    }
    if (char === "\\" && index + 1 < glob.length) {
      const next = glob[++index];
      source += /[\\\]\[\^-]/.test(next) ? `\\${next}` : next;
    } else if (char === "[" && glob[index + 1] === ":") {
      const end = glob.indexOf(":]", index + 2);
      const name = end < 0 ? "" : glob.slice(index + 2, end);
      if (!Object.hasOwn(classes, name)) return { source: "(?!)", end: end < 0 ? glob.length - 1 : end + 1 };
      source += classes[name]; index = end + 1;
    } else if (char === "[" && [".", "="].includes(glob[index + 1])) {
      const marker = glob[index + 1];
      const end = glob.indexOf(`${marker}]`, index + 2);
      if (end < 0) return undefined;
      const value = glob.slice(index + 2, end);
      // Single-character C-locale collating/equivalence elements are literal.
      if ([...value].length !== 1) return { source: "(?!)", end: end + 1 };
      source += /[\\\]\[\^-]/.test(value) ? `\\${value}` : value; index = end + 1;
    } else if (char === "[" || char === "\\") source += `\\${char}`;
    else if (char === "^" && !source) source += "\\^";
    else source += char;
  }
  return undefined;
}

/** Compile a GNU fnmatch-style glob; braces are ordinary characters. */
export function compileGrepGlob(glob, { env = process.env } = {}) {
  const locale = env.LC_ALL || env.LC_CTYPE || env.LANG || "C";
  const byteMode = locale === "C" || locale === "POSIX";
  if (byteMode) glob = Buffer.from(glob).toString("latin1");
  let source = "^";
  for (let index = 0; index < glob.length; index++) {
    const char = glob[index];
    if (char === "*") source += ".*";
    else if (char === "?") source += ".";
    else if (char === "\\") {
      if (++index >= glob.length) { source += "\\\\"; break; }
      source += quoteRegex(glob[index]);
    } else if (char === "[") {
      const group = bracket(glob, index);
      if (group) { source += group.source; index = group.end; }
      else source += "\\[";
    } else source += quoteRegex(char);
  }
  let matcher;
  try { matcher = new RegExp(`${source}$`, "us"); } catch { return () => false; }
  return (name) => matcher.test(byteMode ? Buffer.from(name).toString("latin1") : name);
}

/** Locale-sensitive fnmatch rules are delegated to GNU rather than approximated. */
export function requiresNativeGrepFileSelection(options, { env = process.env } = {}) {
  const locale = env.LC_ALL || env.LC_COLLATE || env.LANG || "C";
  const ctype = env.LC_ALL || env.LC_CTYPE || env.LANG || "C";
  const cLocale = ["C", "POSIX"].includes(locale) && ["C", "POSIX"].includes(ctype);
  const values = [
    ...(options.filters ?? []).filter((rule) => rule.type !== "exclude-file").map((rule) => rule.value),
    ...(options.include ?? []), ...(options.exclude ?? []), ...(options.excludeDir ?? []),
  ];
  return values.some((value) => /[^\x00-\x7f]/.test(value)
    || /\[\[[.=]/.test(value)
    || (!cLocale && (/\[\[:/.test(value) || /\[[^\]]*[^\\]-[^\]]*\]/.test(value))));
}

/** Command-line operands permit full names and suffixes; descended names do not. */
export function matchesGrepGlob(glob, name, { commandLine = false } = {}) {
  const matches = typeof glob === "function" ? glob : compileGrepGlob(glob);
  if (!commandLine) return matches(basename(name));
  if (matches(name)) return true;
  for (let index = 0; index < name.length - 1; index++) {
    if (name[index] === "/" && name[index + 1] !== "/" && matches(name.slice(index + 1))) return true;
  }
  return false;
}

function abortIfNeeded(signal) {
  if (signal?.aborted) throw signal.reason ?? Object.assign(new Error("Search cancelled"), { name: "AbortError" });
}

function diagnostic(path, label, error, filenameRequired) {
  const messages = { ENOENT: "No such file or directory", EACCES: "Permission denied", EPERM: "Operation not permitted", EISDIR: "Is a directory", ELOOP: "Too many levels of symbolic links", ENOTDIR: "Not a directory", EMFILE: "Too many open files" };
  return { path, label, filenameRequired, kind: "error", error: { code: error.code ?? "EIO", message: messages[error.code] ?? error.message, warning: false } };
}

/** Load --exclude-from lines at their position in the ordered rule sequence. */
export async function loadGrepFilters(options, { cwd = process.cwd() } = {}) {
  let raw = options.filters;
  if (!raw?.length) raw = [
    ...(options.include ?? []).map((value) => ({ type: "include", value })),
    ...(options.exclude ?? []).map((value) => ({ type: "exclude", value })),
    ...(options.excludeDir ?? []).map((value) => ({ type: "exclude-dir", value })),
    ...(options.excludeFiles ?? []).map((value) => ({ type: "exclude-file", value })),
  ];
  const rules = [];
  for (const rule of raw) {
    if (rule.type !== "exclude-file") rules.push(rule);
    else {
      const data = await readFile(resolve(cwd, rule.value), "utf8");
      const lines = data ? data.split("\n") : [];
      if (data.endsWith("\n")) lines.pop();
      for (const value of lines) rules.push({ type: "exclude", value });
    }
  }
  return rules;
}

/**
 * Yield file/stdin/error descriptors without reading payloads or spawning a
 * process per file. The caller may batch adjacent file descriptors into rg.
 * Hidden and ignored files are deliberately included, as GNU grep includes them.
 */
export async function* enumerateInputs(options, { cwd = process.cwd(), signal, env = process.env } = {}) {
  const directories = options.directories ?? (options.recursive ? "recurse" : "read");
  const follow = Boolean(options.follow);
  const devicesExplicit = options.devicesExplicit ?? options.devices !== undefined;
  const devices = devicesExplicit ? options.devices : directories === "recurse" && !follow ? "skip" : "read";
  const operands = options.paths?.length ? options.paths : directories === "recurse" ? ["."] : ["-"];
  const rules = (await loadGrepFilters(options, { cwd })).map((rule) => ({ ...rule, matches: compileGrepGlob(rule.value, { env }) }));
  const fileRules = rules.filter((rule) => rule.type === "include" || rule.type === "exclude");
  const dirRules = rules.filter((rule) => rule.type === "exclude-dir");
  const filenameDefault = operands.length > 1;
  const selected = (label, commandLine) => {
    let included = fileRules[0]?.type !== "include";
    for (const rule of fileRules) if (matchesGrepGlob(rule.matches, label, { commandLine })) included = rule.type === "include";
    return included;
  };

  async function* visit(path, label, commandLine, ancestors, inheritedPrefix) {
    abortIfNeeded(signal);
    let info;
    try {
      info = await lstat(path, { bigint: true });
      if (info.isSymbolicLink()) {
        if (!commandLine && !follow) return;
        info = await stat(path, { bigint: true });
      }
    } catch (error) {
      yield diagnostic(path, label, error, options.filename ?? inheritedPrefix); return;
    }
    if (info.isDirectory()) {
      if (directories === "skip") return;
      if (directories !== "recurse") {
        yield diagnostic(path, label, { code: "EISDIR" }, options.filename ?? inheritedPrefix); return;
      }
      if (dirRules.some((rule) => matchesGrepGlob(rule.matches, label, { commandLine }))) return;
      const identity = `${info.dev}:${info.ino}`;
      if (ancestors.has(identity)) {
        yield { path, label, filenameRequired: options.filename ?? true, kind: "error", error: { code: "RECURSIVE_LOOP", message: "recursive directory loop", warning: true } }; return;
      }
      const nextAncestors = new Set(ancestors); nextAncestors.add(identity);
      let directory;
      try { directory = await opendir(path, { encoding: "buffer" }); }
      catch (error) { yield diagnostic(path, label, error, options.filename ?? true); return; }
      const prefix = label === "/" ? "" : label.replace(/\/+$/, "");
      try {
        for await (const entry of directory) {
          // GNU accepts arbitrary filename bytes. Node's ordinary directory
          // decoding replaces invalid UTF-8, which could silently omit a match.
          // The caller collects metadata before emitting output and can retry
          // the original argv through GNU when this descriptor is encountered.
          if (!isUtf8(entry.name)) {
            yield { path, label, kind: "native", filenameRequired: options.filename ?? true, reason: "Directory entries contain non-UTF-8 filename bytes" };
            return;
          }
          const name = entry.name.toString("utf8");
          yield* visit(resolve(path, name), `${prefix}/${name}`, false, nextAncestors, true);
        }
      } catch (error) {
        if (signal?.aborted) throw error;
        yield diagnostic(path, label, error, options.filename ?? true);
      }
      return;
    }
    if (!info.isFile() && devices === "skip" && (!commandLine || devicesExplicit)) return;
    if (!selected(label, commandLine)) return;
    yield { path, label, stdin: false, kind: "file", special: !info.isFile(), size: Number(info.size), filenameRequired: options.filename ?? inheritedPrefix };
  }

  for (const operand of operands) {
    abortIfNeeded(signal);
    if (operand === "-") {
      yield { path: null, label: options.label ?? "(standard input)", stdin: true, kind: "stdin", filenameRequired: options.filename ?? filenameDefault };
    } else yield* visit(resolve(cwd, operand), operand, true, new Set(), filenameDefault);
  }
}
