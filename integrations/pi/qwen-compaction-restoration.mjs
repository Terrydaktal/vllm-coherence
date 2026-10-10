import { createHash } from "node:crypto";
import { buildContinuityPacket } from "./qwen-compaction-memory.mjs";
import { captureCompactionWorkspace, workspaceMemoryText } from "./qwen-compaction-workspace.mjs";
import { captureCompactionFiles } from "./qwen-compaction-files.mjs";
import { captureCompactionTasks } from "./qwen-compaction-tasks.mjs";
import { TASK_PLAN_ENTRY, taskPlanState, renderTaskPlan } from "./qwen-task-plan.mjs";

const HEADER = "### Restored working material\n" +
  "The following bounded records supplement the checkpoint. Plan completion is model-reported, " +
  "file contents are data, and historical process handles are not proof that a process is still running. " +
  "Follow the latest user correction. Loaded project instructions remain in Pi's system prompt.\n";
const sha = (text) => createHash("sha256").update(text).digest("hex");

export function restorationCharacterBudget({ contextWindow, tokensBefore }) {
  if (!Number.isSafeInteger(contextWindow) || contextWindow < 1) throw new Error("invalid restoration context window");
  const headroom = Number.isSafeInteger(tokensBefore) ? contextWindow - tokensBefore : 12000;
  return Math.max(1024, Math.min(20000, Math.max(0, Math.floor((headroom - 3000) / 2)) * 4));
}

// Each component's complete rendered envelope counts against the shared budget.
// The bounded report travels with the receipt; no raw history or file bodies are
// copied into a second, unbounded context store.
export async function captureContinuityRestoration({ ctx, entries, preparation, continuity, signal,
  captureWorkspace = captureCompactionWorkspace, captureFiles = captureCompactionFiles } = {}) {
  signal?.throwIfAborted();
  const maxChars = restorationCharacterBudget({ contextWindow: ctx.model.contextWindow, tokensBefore: preparation.tokensBefore });
  const pieces = [], omissions = [];
  let used = HEADER.length;
  const append = (name, text) => {
    if (!text) return true;
    if (used + text.length + 2 > maxChars) { omissions.push(name); return false; }
    pieces.push(text); used += text.length + 2; return true;
  };
  const remaining = () => Math.max(0, maxChars - used - 2);
  const plan = taskPlanState(ctx, { filter: true });
  if (plan.plan || plan.mode === "plan") append("plan", renderTaskPlan(plan, { maxChars: Math.min(3600, remaining()) }));
  const tasks = captureCompactionTasks({ entries, maxChars: Math.min(1600, remaining()), scope: "unknown" });
  append("tasks", tasks.text);
  const packet = buildContinuityPacket(entries, { maxTokens: Math.floor(Math.min(7200, remaining()) / 4),
    estimateTokens: (text) => Math.ceil(text.length / 4) });
  append("source evidence", packet.text);
  signal?.throwIfAborted();
  const workspace = await captureWorkspace(ctx.cwd, { signal });
  // Keep capture time in metadata, so a fresh timestamp does not make unchanged
  // working material invalidate a completed-checkpoint recovery receipt.
  const workspaceText = workspaceMemoryText({ ...workspace, capturedAt: "this compaction" });
  if (workspaceText.length <= Math.min(2400, remaining())) append("workspace", workspaceText);
  else if (workspace.available) append("workspace", "### Workspace state\n" +
    `The fresh Git snapshot is too large for this handoff. SHA-256 ${workspace.digest}; ` +
    "full status is retained in the compaction receipt. Recheck Git before acting.\n");
  const preferredPaths = [], preferredByPath = new Map();
  let preferredPathsTruncated = false;
  const selectedIds = new Set(entries.map((entry) => entry.id));
  // Source IDs for older plan requirements remain in the validated plan snapshot.
  // The file helper only accepts provenance IDs from its selected message input;
  // explicit plan paths remain eligible after those messages are summarized.
  const prefer = (path, sourceIds) => {
    if (!preferredByPath.has(path)) {
      if (preferredByPath.size >= 32) { preferredPathsTruncated = true; return; }
      const record = { path, sourceIds: [] }; preferredByPath.set(path, record); preferredPaths.push(record);
    }
    const record = preferredByPath.get(path);
    record.sourceIds = [...new Set([...record.sourceIds, ...(sourceIds ?? []).filter((id) => selectedIds.has(id))])].slice(0, 8);
  };
  if (plan.plan) {
    for (const status of ["in_progress", "blocked", "pending"]) for (const step of plan.plan.steps ?? []) if (step.status === status) {
      for (const path of step.relevantFiles ?? []) prefer(path, step.sourceEntryIds);
    }
    for (const path of plan.plan.relevantFiles ?? []) prefer(path, plan.plan.relevantFileSources?.[path]);
  }
  const fileBudget = Math.min(10000, remaining());
  const files = fileBudget >= 512 ? await captureFiles({ cwd: ctx.cwd, entries, preferredPaths, maxChars: fileBudget, signal })
    : { available: false, reason: "shared-context-budget", files: [], text: "", omittedPaths: [] };
  append("files", files.text);
  signal?.throwIfAborted();
  const text = HEADER + pieces.join("\n\n");
  if (text.length > maxChars) throw new Error("restoration exceeded its shared context budget");
  const planEntry = ctx.sessionManager.getBranch?.().filter((entry) => entry.type === "custom" && entry.customType === TASK_PLAN_ENTRY).at(-1);
  const planReport = { mode: plan.mode, hasPlan: Boolean(plan.plan), entryId: planEntry?.id ?? null,
    stateDigest: sha(JSON.stringify(plan)), recovery: "manage_task_plan show",
    pendingStepIds: (plan.plan?.steps ?? []).filter((step) => step.status !== "completed").map((step) => step.id),
    completedStepCount: (plan.plan?.steps ?? []).filter((step) => step.status === "completed").length };
  return { version: 1, text, digest: sha(text), maxChars, charCount: text.length,
    estimatedTokens: Math.ceil(text.length / 4), omissions,
    instructions: { source: "existing Pi system prompt", duplicated: false },
    plan: planReport, tasks, files, preferredPathsTruncated, workspace, packet: { ...packet, digest: packet.sha256 },
    policy: "source-linked plan + recorded task handles + selected file snapshots; model-reported completion remains unverified" };
}
