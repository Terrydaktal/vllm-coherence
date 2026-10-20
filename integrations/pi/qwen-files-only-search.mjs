import { spawn } from "node:child_process";
import { stat } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const MAX_OUTPUT_BYTES = 50 * 1024;
const MAX_FILES = 1000;
const DEFAULT_LIMIT = 100;
const DEFAULT_TIMEOUT_MS = 20000;

function extendSchema(schema, updates) {
  // Recent TypeBox versions store ~kind/~optional as non-enumerable fields;
  // older ones use symbols. Preserve both rather than spreading the schema.
  const clone = Object.create(Object.getPrototypeOf(schema), Object.getOwnPropertyDescriptors(schema));
  for (const [key, value] of Object.entries(updates)) {
    Object.defineProperty(clone, key, { value, writable: true, enumerable: true, configurable: true });
  }
  return clone;
}

// Extend the SDK's own optional Boolean schema, including its TypeBox metadata.
// The existing parameter properties/required list and default execution survive.
export function withFilesOnlySearch(definition, { cwd, rgPath = "rg", timeoutMs = DEFAULT_TIMEOUT_MS,
  spawnProcess = spawn } = {}) {
  const schema = definition.parameters;
  if (schema?.properties?.literal?.type !== "boolean") throw new Error("Unsupported native search schema");
  return {
    ...definition,
    parameters: extendSchema(schema, { properties: { ...schema.properties,
      files_only: extendSchema(schema.properties.literal, { description: "Return matching filenames once each instead of matching lines (default: false). limit bounds filenames; context is ignored." }),
    } }),
    description: definition.description + " Set files_only: true to return matching filenames once each, without returning matching lines. Filename output is limited to 1,000 files or 50KB and the search has a 20-second deadline.",
    execute(id, params, signal, onUpdate, ctx) {
      if (params.files_only !== true) return definition.execute(id, params, signal, onUpdate, ctx);
      return searchFilenames(params, { cwd: ctx?.cwd ?? cwd, signal, rgPath, timeoutMs, spawnProcess });
    },
  };
}

function resolveSearchPath(rawPath, cwd) {
  let searchPath = rawPath || ".";
  // Match the native SDK's pasted-path conveniences and home expansion.
  searchPath = searchPath.replace(/[\u00a0\u2000-\u200a\u202f\u205f\u3000]/g, " ");
  if (searchPath.startsWith("@")) searchPath = searchPath.slice(1);
  if (searchPath === "~" || searchPath.startsWith("~/")) searchPath = homedir() + searchPath.slice(1);
  if (searchPath.startsWith("file://")) searchPath = fileURLToPath(searchPath);
  return resolve(cwd, searchPath);
}

export async function searchFilenames(params, { cwd, signal, rgPath = "rg", timeoutMs = DEFAULT_TIMEOUT_MS,
  spawnProcess = spawn } = {}) {
  if (signal?.aborted) throw new Error("Operation aborted");
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error("Invalid filename-search deadline");
  if (typeof params.pattern !== "string") throw new Error("Search pattern must be a string");
  if (params.limit !== undefined && (!Number.isSafeInteger(params.limit) || params.limit < 1)) {
    throw new Error("Filename limit must be a positive integer");
  }
  const limit = Math.min(params.limit ?? DEFAULT_LIMIT, MAX_FILES);
  const searchPath = resolveSearchPath(params.path, cwd);
  let directory;
  try { directory = (await stat(searchPath)).isDirectory(); }
  catch { throw new Error(`Path not found: ${searchPath}`); }
  if (signal?.aborted) throw new Error("Operation aborted");
  const args = ["--files-with-matches", "--null", "--color=never", "--hidden"];
  if (params.ignoreCase) args.push("--ignore-case");
  if (params.literal) args.push("--fixed-strings");
  if (params.glob) args.push("--glob", params.glob);
  args.push("--", params.pattern, searchPath);

  return new Promise((resolveResult, reject) => {
    const child = spawnProcess(rgPath, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let pending = Buffer.alloc(0), stderr = "", outputBytes = 0, stopped, settled = false;
    const filenames = [];
    const stop = (reason) => { if (!stopped) { stopped = reason; child.kill("SIGKILL"); } };
    const onAbort = () => stop("aborted");
    const timer = setTimeout(() => stop("timeout"), timeoutMs);
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
    const cleanup = () => { clearTimeout(timer); signal?.removeEventListener("abort", onAbort); };
    const settle = (action) => { if (!settled) { settled = true; cleanup(); action(); } };
    const addFilename = (raw) => {
      const fullPath = raw.toString("utf8");
      const display = directory ? relative(searchPath, fullPath) : basename(fullPath);
      // Keep one path per output line even for filenames containing newlines.
      const text = /[\x00-\x1f\x7f]/.test(display) ? JSON.stringify(display) : display;
      const bytes = Buffer.byteLength(text, "utf8") + (filenames.length ? 1 : 0);
      if (filenames.length >= limit) { stop("limit"); return; }
      if (outputBytes + bytes > MAX_OUTPUT_BYTES) { stop("bytes"); return; }
      filenames.push(text); outputBytes += bytes;
    };
    child.stdout.on("data", (chunk) => {
      if (stopped) return;
      pending = Buffer.concat([pending, Buffer.from(chunk)]);
      let separator;
      while (!stopped && (separator = pending.indexOf(0)) >= 0) {
        addFilename(pending.subarray(0, separator));
        pending = pending.subarray(separator + 1);
      }
      if (pending.length > MAX_OUTPUT_BYTES) stop("bytes");
    });
    child.stderr.on("data", (chunk) => { stderr = (stderr + chunk.toString()).slice(0, 8192); });
    child.on("error", (error) => settle(() => reject(new Error(`Failed to run ripgrep: ${error.message}`))));
    child.on("close", (code) => settle(() => {
      if (stopped === "aborted") return reject(new Error("Operation aborted"));
      if (stopped === "timeout") return reject(new Error(`Filename search exceeded ${timeoutMs / 1000} seconds; narrow the path or pattern`));
      if (!stopped && code !== 0 && code !== 1) return reject(new Error(stderr.trim() || `ripgrep exited with code ${code}`));
      const details = { filesOnly: true, returnedFiles: filenames.length, exitCode: stopped ? null : code };
      let text = filenames.length ? filenames.join("\n") : "No matches found";
      if (stopped === "limit") {
        details.matchLimitReached = limit;
        text += `\n\n[${limit} files limit reached. Refine the path or pattern, or increase limit up to ${MAX_FILES}.]`;
      } else if (stopped === "bytes") {
        details.truncation = { truncated: true, maxBytes: MAX_OUTPUT_BYTES };
        text += "\n\n[50KB filename output limit reached. Refine the path or pattern.]";
      }
      return resolveResult({ content: [{ type: "text", text }], details });
    }));
  });
}
