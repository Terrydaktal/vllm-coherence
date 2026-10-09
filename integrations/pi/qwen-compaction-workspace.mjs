import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { isAbsolute } from "node:path";

// A bounded repository metadata snapshot. Git may inspect tracked contents to
// determine status, but no source contents are returned. Never invoke a shell
// or let a repository's fsmonitor run during compaction.
export async function captureCompactionWorkspace(cwd, { signal, timeoutMs = 750 } = {}) {
  signal?.throwIfAborted();
  if (typeof cwd !== "string" || !isAbsolute(cwd)) return { available: false, reason: "workspace-unavailable" };
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 5000) throw new Error("invalid workspace snapshot deadline");
  const capturedAt = new Date().toISOString();
  const deadline = performance.now() + timeoutMs;
  // -C must always name the requested workspace. Inherited Git overrides can
  // otherwise redirect status to another repository/index or inject config.
  const env = Object.fromEntries(Object.entries(process.env).filter(([name]) => !name.startsWith("GIT_")));
  Object.assign(env, { GIT_OPTIONAL_LOCKS: "0", GIT_TERMINAL_PROMPT: "0", GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null", LC_ALL: "C" });
  const runGit = (args) => new Promise((resolve) => {
    const remainingMs = Math.floor(deadline - performance.now());
    if (remainingMs < 1) { resolve({ available: false, reason: "deadline-exceeded" }); return; }
    execFile("git", ["--no-optional-locks", "-c", "core.fsmonitor=false", "-C", cwd,
      ...args],
    { signal, timeout: remainingMs, killSignal: "SIGKILL", maxBuffer: 16384,
      env },
    (error, stdout) => resolve(error ? { available: false, code: error.code, reason: error.code === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER" ? "output-too-large" : error.killed ? "deadline-exceeded" : "git-state-unavailable" }
      : { available: true, status: stdout }));
  });
  // Refreshing status can invoke configured clean/process filters. Detect
  // those without inspecting source contents and omit the snapshot rather
  // than run repository commands or alter their numerical/status semantics.
  const filters = await runGit(["config", "--null", "--name-only", "--get-regexp", "^filter\\..*\\.(clean|process)$"]);
  signal?.throwIfAborted();
  if (filters.available && filters.status.length) return { available: false, reason: "repository-filters-unsupported" };
  if (!filters.available && filters.code !== 1) return { available: false, reason: filters.reason };
  const result = await runGit(["status", "--porcelain=v2", "--branch", "--untracked-files=no", "--ignore-submodules=all"]);
  signal?.throwIfAborted();
  if (!result.available) return { available: false, reason: result.reason };
  return { ...result, capturedAt,
    digest: createHash("sha256").update(result.status).digest("hex"),
    scope: "tracked files and branch only; untracked files and submodule contents excluded" };
}

export function workspaceMemoryText(workspace) {
  if (!workspace?.available) return "";
  const quoted = JSON.stringify(workspace.status).replace(/</g, "\\u003c").replace(/>/g, "\\u003e");
  return `\n\n### Workspace state at compaction\nRead-only Git snapshot at ${workspace.capturedAt}; ` +
    `${workspace.scope}. This records repository state, not test success or task completion. ` +
    `Recheck files before editing and Git before committing. Porcelain v2 snapshot (JSON string):\n${quoted}`;
}
