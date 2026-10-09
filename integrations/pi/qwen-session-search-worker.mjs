import { parentPort, workerData } from "node:worker_threads";
import { readdirSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { TranscriptIndex } from "./qwen-session-index.mjs";

// All SQLite work and JSONL parsing stay outside Pi's UI/agent event loop.
let index;
parentPort.on("message", ({ id, params, sessionFile, leafId }) => {
  try {
    const sourceRoot = resolve(workerData.sourceRoot);
    const current = resolve(sessionFile);
    if (dirname(current) !== sourceRoot) throw new Error("session is outside the transcript search directory");
    index ??= new TranscriptIndex({ databasePath: workerData.databasePath, sourceRoot });
    const selected = params.session_file ? resolve(sourceRoot, params.session_file) : current;
    if (dirname(selected) !== sourceRoot || !basename(selected).endsWith(".jsonl")) {
      throw new Error("session_file must name a JSONL session in the current session directory");
    }
    const project = params.scope === "project" && !params.session_file && !params.around_entry_id;
    const files = project ? readdirSync(sourceRoot, { withFileTypes: true })
      .filter((entry) => entry.isFile() && entry.name.endsWith(".jsonl"))
      .map((entry) => join(sourceRoot, entry.name)) : [selected];
    parentPort.postMessage({ id, event: "progress", phase: "index", files: files.length });
    const started = performance.now();
    const refresh = index.sync(files);
    const refreshed = performance.now();
    parentPort.postMessage({ id, event: "progress", phase: "search" });
    const query = {
      query: params.query, mode: params.mode ?? "words",
      limit: params.limit ?? (params.around_entry_id ? 2 * (params.window ?? 2) + 1 : 3),
      sessionFile: project ? undefined : selected,
      leafId: !project && selected === current ? leafId : undefined,
      includeBranches: params.include_branches === true, roles: params.roles,
      maxChars: params.max_chars ?? 6000, aroundEntryId: params.around_entry_id,
      window: params.window ?? 2,
    };
    let result = index.search(query);
    // A writer may replace a source between refresh and hydration. Refresh once
    // rather than return a stale excerpt; further changes remain explicit.
    if (result.stale > 0) { index.sync(files); result = index.search(query); }
    parentPort.postMessage({ id, result: { ...result, refresh,
      timings: { refreshMs: refreshed - started, searchMs: performance.now() - refreshed } } });
  } catch (error) {
    parentPort.postMessage({ id, error: { name: error.name, message: error.message } });
  }
});
