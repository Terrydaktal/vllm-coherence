import test from "node:test";
import assert from "node:assert/strict";
import { captureContinuityRestoration, restorationCharacterBudget } from "../integrations/pi/qwen-compaction-restoration.mjs";
import { TASK_PLAN_ENTRY, applyTaskPlanAction, replayTaskPlan } from "../integrations/pi/qwen-task-plan.mjs";

const message = (id, text) => ({ type: "message", id, message: { role: "user", content: text } });
const ctx = (entries = []) => ({ cwd: "/synthetic-workspace", model: { contextWindow: 32768 },
  sessionManager: { getBranch: () => entries } });
const emptyFiles = async () => ({ available: true, files: [], text: "", omittedPaths: [] });
const noWorkspace = async () => ({ available: false, reason: "not-a-repository" });

test("restoration enforces a shared envelope budget through tight and normal headroom", async () => {
  const entries = [message("latest-user", "Keep the exact current arithmetic. ".repeat(1000))];
  for (const tokensBefore of [0, 20000, 28000, 30720, 32000, 32768]) {
    let fileAllowance = 0;
    const report = await captureContinuityRestoration({ ctx: ctx(), entries, preparation: { tokensBefore },
      captureWorkspace: noWorkspace,
      captureFiles: async ({ maxChars }) => { fileAllowance = maxChars; return { available: true, files: [], text: "x".repeat(maxChars), omittedPaths: [] }; },
    });
    assert.ok(report.text.length <= report.maxChars);
    assert.equal(report.charCount, report.text.length);
    assert.equal(report.estimatedTokens, Math.ceil(report.text.length / 4));
    assert.ok(fileAllowance <= 10000);
    assert.equal(report.instructions.duplicated, false);
  }
});

test("capture timestamps do not invalidate an otherwise identical restoration receipt", async () => {
  const entries = [message("u", "Current exact requirement.")];
  const base = { ctx: ctx(), entries, preparation: { tokensBefore: 10 }, captureFiles: emptyFiles };
  const status = "# branch.head main\n# branch.oid abc123\n";
  const first = await captureContinuityRestoration({ ...base, captureWorkspace: async () => ({ available: true, status, capturedAt: "2026-01-01T00:00:00Z", digest: "same", scope: "tracked paths" }) });
  const second = await captureContinuityRestoration({ ...base, captureWorkspace: async () => ({ available: true, status, capturedAt: "2026-01-02T00:00:00Z", digest: "same", scope: "tracked paths" }) });
  assert.equal(first.digest, second.digest);
  assert.notEqual(first.workspace.capturedAt, second.workspace.capturedAt);
  assert.equal(first.text, second.text);
});

test("oversized Git state is an explicit reference and cannot crowd out file restoration", async () => {
  let reads = 0;
  const report = await captureContinuityRestoration({ ctx: ctx(), entries: [message("u", "Continue the current task.")], preparation: { tokensBefore: 10 },
    captureWorkspace: async () => ({ available: true, status: "branch state ".repeat(2000), digest: "full-status-digest", scope: "tracked paths" }),
    captureFiles: async () => { reads++; return { available: true, files: [], text: "### Current relevant file\nexport const current = 1;", omittedPaths: [] }; },
  });
  assert.match(report.text, /Git snapshot is too large/);
  assert.match(report.text, /full-status-digest/);
  assert.match(report.text, /export const current = 1/);
  assert.equal(reads, 1);
});

test("pre-aborted restoration performs neither workspace nor file capture", async () => {
  const controller = new AbortController(); controller.abort(new Error("cancelled synthetic capture"));
  let captures = 0;
  await assert.rejects(captureContinuityRestoration({ ctx: ctx(), entries: [], preparation: { tokensBefore: 0 }, signal: controller.signal,
    captureWorkspace: async () => { captures++; }, captureFiles: async () => { captures++; },
  }), /cancelled synthetic capture/);
  assert.equal(captures, 0);
});

test("invalid context windows are rejected before restoration", () => {
  for (const contextWindow of [0, -1, 1.5, NaN, undefined]) assert.throws(() => restorationCharacterBudget({ contextWindow, tokensBefore: 0 }), /context window/);
});

test("large persisted plans prioritize pending files and bound their receipt metadata", async () => {
  const source = message("older-user", "Complete this multi-file task.");
  const paths = Array.from({ length: 40 }, (_, i) => `src/file-${i}.mjs`);
  const state = applyTaskPlanAction(replayTaskPlan([]), { action: "create", goal: "Preserve the current task",
    relevantFiles: paths, steps: [{ title: "Check pending repair", relevantFiles: [paths[39]] },
      ...Array.from({ length: 29 }, (_, i) => ({ title: `Other step ${i}`, relevantFiles: paths }))],
  }, [source]);
  const branch = [source, { type: "custom", customType: TASK_PLAN_ENTRY, id: "plan-snapshot", data: state },
    { type: "compaction", id: "compact", summary: "Older task retained in summary." }, message("new-user", "Continue.")];
  let preferred;
  const report = await captureContinuityRestoration({ ctx: ctx(branch), entries: branch.slice(-2), preparation: { tokensBefore: 100 },
    captureWorkspace: noWorkspace, captureFiles: async ({ preferredPaths }) => { preferred = preferredPaths; return emptyFiles(); },
  });
  assert.equal(preferred.length, 32);
  assert.equal(preferred[0].path, paths[39]);
  assert.ok(preferred.every((ref) => ref.sourceIds.length === 0), "archived IDs are not passed as current-message provenance");
  assert.equal(report.preferredPathsTruncated, true);
  assert.equal(report.plan.entryId, "plan-snapshot");
  assert.equal(report.plan.recovery, "qwen_plan show");
  assert.equal(report.plan.plan, undefined, "full canonical state remains in the session instead of every receipt");
  assert.ok(Buffer.byteLength(JSON.stringify(report)) < 50_000);
  assert.equal(state.plan.steps.length, 30);
});
