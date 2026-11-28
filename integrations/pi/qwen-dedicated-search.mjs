// Radiance's Linux fd/rg execution. The SDK still supplies the public tool schemas,
// renderers, and output truncation. Other platforms use the stock definitions.
import { spawn } from "node:child_process";
import { accessSync, constants, createReadStream, existsSync, statSync } from "node:fs";
import { stat } from "node:fs/promises";
import { availableParallelism } from "node:os";
import { basename, delimiter, dirname, isAbsolute, join, relative, resolve } from "node:path";
import { createInterface } from "node:readline";
import { resolveSearchPath, ripgrepThreads } from "./qwen-files-only-search.mjs";

const executables = new Map();
const slowFindRoots = new Map();

function executable(path) {
  try { accessSync(path, constants.X_OK); return statSync(path).isFile(); }
  catch { return false; }
}

export function searchExecutable(name, agentDir) {
  const managed = join(agentDir, "bin", name);
  if (executable(managed)) return managed;
  // Relative PATH entries resolve against Pi's process cwd, as in the stock SDK.
  const path = process.env.PATH ?? "/usr/bin:/bin";
  const key = JSON.stringify([name, path, process.cwd()]);
  const cached = executables.get(key);
  if (cached && executable(cached)) return cached;
  const directories = path.split(delimiter);
  for (const candidate of name === "fd" ? ["fd", "fdfind"] : [name]) {
    for (const dir of directories) {
      const file = resolve(dir, candidate);
      if (executable(file)) {
        if (executables.size >= 16) executables.clear();
        executables.set(key, file);
        return file;
      }
    }
  }
}

function checkAbort(signal) {
  if (signal?.aborted) throw new Error("Operation aborted");
}

// Terminate on the match limit, but wait for exit before resolving. Always drain
// both pipes and propagate search failures unless the result limit stopped us.
function runLines(program, args, signal, visit) {
  checkAbort(signal);
  return new Promise((resolveResult, reject) => {
    const child = spawn(program, args, { stdio: ["ignore", "pipe", "pipe"] });
    const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
    let stderr = "", stopped = false, failure;
    const abort = () => child.kill("SIGKILL");
    signal?.addEventListener("abort", abort, { once: true });
    child.stderr.setEncoding("utf8").on("data", chunk => { stderr += chunk; });
    lines.on("line", line => {
      if (stopped || failure || signal?.aborted) return;
      try { if (visit(line) === false) { stopped = true; child.kill(); } }
      catch (error) { failure = error; child.kill(); }
    });
    child.once("error", error => { failure = error; });
    child.once("close", code => {
      signal?.removeEventListener("abort", abort);
      lines.close();
      if (signal?.aborted) reject(new Error("Operation aborted"));
      else if (failure) reject(failure);
      else resolveResult({ code, stderr: stderr.trim(), stopped });
    });
  });
}

function result(text, details) {
  return { content: [{ type: "text", text }], details };
}

function limitedOutput(lines, details, notices, sdk) {
  const truncation = sdk.truncateHead(lines.join("\n"), { maxLines: Number.MAX_SAFE_INTEGER });
  if (truncation.truncated) {
    details.truncation = truncation;
    notices.push(`${sdk.formatSize(sdk.DEFAULT_MAX_BYTES)} limit reached`);
  }
  if (details.linesTruncated) notices.push("Some lines truncated to 500 chars. Use read tool to see full lines");
  const suffix = notices.length ? "\n\n[" + notices.join(". ") + "]" : "";
  return result(truncation.content + suffix, Object.keys(details).length ? details : undefined);
}

export async function findPaths(program, params, cwd, signal, sdk) {
  checkAbort(signal);
  const root = resolveSearchPath(params.path, cwd);
  const limit = params.limit ?? 1000;
  const widePattern = ["*", "**", "**/*"].includes(params.pattern) && limit >= 1000;
  const workers = widePattern || slowFindRoots.has(root) ? Math.min(4, availableParallelism()) : 1;
  const args = ["--threads", String(workers), "--glob", "--color=never", "--hidden"];
  let ancestor = root;
  while (!existsSync(join(ancestor, ".git"))) {
    const parent = dirname(ancestor);
    if (parent === ancestor) { args.push("--no-require-git"); break; }
    ancestor = parent;
  }
  args.push("--max-results", String(limit));
  let pattern = params.pattern;
  if (pattern.includes("/")) {
    args.push("--full-path");
    if (!pattern.startsWith("/") && !pattern.startsWith("**/")) pattern = "**/" + pattern;
  }
  args.push("--", pattern, root);
  const paths = [];
  const started = performance.now();
  const { code, stderr } = await runLines(program, args, signal, line => {
    const raw = line.trim();
    if (!raw) return;
    const path = isAbsolute(raw) ? relative(root, raw) : raw;
    paths.push(raw.endsWith("/") && !path.endsWith("/") ? path + "/" : path);
  });
  if (code !== 0 && !paths.length) throw new Error(stderr || `fd exited with code ${code}`);
  // A completed, slow single-worker walk is evidence for widening subsequent
  // walks. Capped queries may finish early and say little about total tree size.
  if (workers === 1 && paths.length < limit && performance.now() - started >= 50) {
    if (slowFindRoots.size >= 64) slowFindRoots.delete(slowFindRoots.keys().next().value);
    slowFindRoots.set(root, true);
  }
  if (!paths.length) return result("No files found matching pattern");
  const capped = paths.length >= limit;
  return limitedOutput(paths, capped ? { resultLimitReached: limit } : {},
    capped ? [`${limit} results limit reached. Use limit=${limit * 2} for more, or refine pattern`] : [], sdk);
}

// Match the SDK's UTF-8 decoding and CRLF/lone-CR normalization. rg numbers LF
// lines, while Pi indexes that normalized text, so rg context records alone
// cannot reproduce its output for all files. Stream only up to the last wanted
// row and retain only requested rows, including the terminal empty row.
async function readContext(file, ranges, signal, sdk) {
  const rows = new Map();
  const stream = createReadStream(file, { encoding: "utf8", signal });
  let number = 1, index = 0, pending = "", skipLf = false;
  try {
    for await (let chunk of stream) {
      if (skipLf && chunk.startsWith("\n")) chunk = chunk.slice(1);
      skipLf = chunk.endsWith("\r");
      const segments = chunk.includes("\r") ? chunk.split(/\r\n|[\r\n]/) : chunk.split("\n");
      const trailingNumber = number + segments.length - 1;
      // Jump over rows outside the requested windows. Each chunk retains at
      // most one unfinished row, capped at the SDK limit plus one code unit.
      for (; index < ranges.length; index++) {
        const { start, end } = ranges[index];
        for (let row = Math.max(number, start); row <= Math.min(end, trailingNumber - 1); row++) {
          const segment = segments[row - number];
          rows.set(row, sdk.truncateLine(row === number ? pending + segment : segment));
        }
        if (end >= trailingNumber) break;
      }
      if (index === ranges.length) return rows;
      if (trailingNumber >= ranges[index].start) {
        const prefix = trailingNumber === number ? pending : "";
        pending = prefix + segments.at(-1).slice(0, 501 - prefix.length);
      } else pending = "";
      number = trailingNumber;
    }
    if (number >= ranges[index].start) rows.set(number, sdk.truncateLine(pending));
    return rows;
  } catch {
    checkAbort(signal);
    return undefined;
  }
}

export async function searchContents(program, params, cwd, signal, sdk) {
  checkAbort(signal);
  const root = resolveSearchPath(params.path, cwd);
  let directory;
  try { directory = (await stat(root)).isDirectory(); }
  catch { throw new Error(`Path not found: ${root}`); }
  const context = Math.max(0, params.context ?? 0);
  const limit = Math.max(1, params.limit ?? 100);
  // Workers can buffer a file's output until EOF. This per-file ceiling lets
  // rg stop reading as soon as that file could satisfy the global match limit.
  const args = ["--threads", ripgrepThreads(directory), "--max-count", String(limit),
    "--json", "--line-number", "--color=never", "--hidden"];
  if (params.ignoreCase) args.push("--ignore-case");
  if (params.literal) args.push("--fixed-strings");
  if (params.glob) args.push("--glob", params.glob);
  args.push("--", params.pattern, root);
  const matches = [], ranges = new Map();
  let count = 0;
  const { code, stopped, stderr } = await runLines(program, args, signal, line => {
    let event;
    try { event = JSON.parse(line); }
    catch { return; } // The SDK ignores non-event stdout from a configured rg.
    if (event.type !== "match") return;
    count++;
    const { path, line_number: number, lines } = event.data;
    if (path.text && typeof number === "number") {
      // Context is recovered from source bytes below. Drop rg's complete long
      // lines here instead of keeping all of them live until the child exits.
      const inline = !context && lines.text !== undefined
        ? sdk.truncateLine(lines.text.replace(/\r\n/g, "\n").replace(/\r/g, "").replace(/\n$/, ""))
        : undefined;
      matches.push({ file: path.text, number, inline });
      if (!inline) {
        const windows = ranges.get(path.text) ?? [];
        const start = Math.max(1, number - context), end = number + context;
        const last = windows.at(-1);
        if (last && start <= last.end + 1) last.end = Math.max(last.end, end);
        else windows.push({ start, end });
        ranges.set(path.text, windows);
      }
    }
    if (count >= limit) return false;
  });
  if (!stopped && code !== 0 && code !== 1) throw new Error(stderr || `ripgrep exited with code ${code}`);
  if (count === 0) return result("No matches found");
  const fileRows = new Map();
  for (const [file, windows] of ranges) {
    checkAbort(signal);
    fileRows.set(file, await readContext(file, windows, signal, sdk));
  }
  const output = [], details = {}, notices = [];
  if (stopped) {
    details.matchLimitReached = limit;
    notices.push(`${limit} matches limit reached. Use limit=${limit * 2} for more, or refine pattern`);
  }
  for (const { file, number, inline } of matches) {
    const rel = relative(root, file);
    const label = directory && rel && !rel.startsWith("..") ? rel.replace(/\\/g, "/") : basename(file);
    if (inline) {
      if (inline.wasTruncated) details.linesTruncated = true;
      output.push(`${label}:${number}: ${inline.text}`);
    } else {
      const rows = fileRows.get(file);
      if (!rows) {
        output.push(`${label}:${number}: (unable to read file)`);
        continue;
      }
      for (let row = Math.max(1, number - context); row <= number + context; row++) {
        const entry = rows.get(row);
        if (!entry) break;
        if (entry.wasTruncated) details.linesTruncated = true;
        output.push(row === number ? `${label}:${row}: ${entry.text}` : `${label}-${row}- ${entry.text}`);
      }
    }
  }
  return limitedOutput(output, details, notices, sdk);
}
