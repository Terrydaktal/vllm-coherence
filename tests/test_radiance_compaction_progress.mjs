import assert from "node:assert/strict";
import test from "node:test";
import { randomUUID } from "node:crypto";
import { appendCompactionTiming, COMPACTION_TIMING_ENTRY, startCompactionProgress,
  getCompactionProgress, clearCompactionProgress } from "../integrations/pi/qwen-radiance-compaction-progress.mjs";

const currentChat = { chat_id: "a".repeat(64), generation: "b".repeat(64) };
const otherChat = { chat_id: "c".repeat(64), generation: "d".repeat(64) };

function fixture(t, { observationAt = () => ({ available: false, configured: false }) } = {}) {
  let clock = 0, tick, intervalMs, stopped = false, schedulerStopped = false;
  const widgets = [], statuses = [], messages = [], reports = [], notices = [], abort = new AbortController();
  const sessionFile = `/synthetic-${randomUUID()}.jsonl`;
  const ctx = { sessionManager: { getSessionFile: () => sessionFile }, ui: {
    setWidget: (_key, lines) => widgets.push(lines?.join("\n")), setStatus: (_key, value) => statuses.push(value),
    notify: (message) => notices.push(message),
    setWorkingMessage: (message) => messages.push(message),
  } };
  const scheduler = {
    start() {},
    stop() { schedulerStopped = true; },
    read: () => observationAt(clock),
  };
  const progress = startCompactionProgress(ctx, { signal: abort.signal, now: () => clock,
    save: (report) => reports.push(report), scheduler,
    schedule: (fn, ms) => { tick = fn; intervalMs = ms; return 1; }, unschedule: () => { stopped = true; } });
  t.after(async () => {
    await clearCompactionProgress(ctx);
    assert.ok(widgets.every((value) => value === undefined), "no pinned compaction widget");
    assert.ok(statuses.every((value) => value === undefined), "no pinned compaction footer");
    assert.equal(messages.at(-1), undefined, "no message left for the next working spinner");
  });
  return { ctx, progress, abort, widgets, statuses, messages, reports, notices,
    stopped: () => stopped, schedulerStopped: () => schedulerStopped, intervalMs: () => intervalMs,
    advance: (ms, render = true) => { clock += ms; if (!stopped && render) tick(); } };
}

test("context preparation separately times GPU queue, cache restore, prefill and first-token work", async (t) => {
  const f = fixture(t, { observationAt: (clock) => {
    if (clock < 13_000) return { available: true,
      request: { ...currentChat, state: "paused", computed_tokens: 0, input_tokens: 241_594 },
      activeChat: otherChat, workerAvailable: true, otherRunningChatId: otherChat.chat_id,
      otherRunningRequest: { ...otherChat, state: "running", computed_tokens: 40_000, input_tokens: 50_000 } };
    if (clock < 33_000) return { available: true,
      request: { ...currentChat, state: "queued", computed_tokens: 0, input_tokens: 241_594 },
      activeChat: currentChat, workerAvailable: true };
    if (clock < 63_000) return { available: true,
      request: { ...currentChat, state: "running", computed_tokens: 235_664, input_tokens: 241_594 },
      activeChat: currentChat, workerAvailable: true };
    return { available: true,
      request: { ...currentChat, state: "running", computed_tokens: 241_594, input_tokens: 241_594 },
      activeChat: currentChat, workerAvailable: true };
  } });
  f.advance(1000);
  f.progress.update({ phase: "tokenize" });
  f.advance(2000);
  f.progress.update({ phase: "wait", inputTokens: 241594 });
  assert.match(f.messages.at(-1), /cached\/uncached split pending/);
  f.advance(10000);
  f.advance(20000);
  f.advance(30000);
  f.advance(5000);
  assert.match(f.messages.at(-1), /GPU queue: 10.0s · another chat c{12} has the GPU and is prefilling 40,000 \/ 50,000 prompt tokens/);
  assert.match(f.messages.at(-1), /Find and load cached context: 20.0s/);
  assert.match(f.messages.at(-1), /Prompt prefill: 30.0s/);
  assert.match(f.messages.at(-1), /Await first checkpoint output: 5.0s/);
  assert.match(f.messages.at(-1), /1m 8s total/);
  f.progress.update({ cacheRead: 0 });
  assert.match(f.messages.at(-1), /cached 0 \(0.0%\) · uncached 241,594/);
  f.progress.update({ phase: "generate", outputTokens: 1, cacheRead: 0 });
  assert.match(f.messages.at(-1), /cached 0 \(0.0%\) · uncached 241,594/);
  assert.match(f.messages.at(-1), /Await first checkpoint output: 5.0s/);
  f.advance(10000);
  f.progress.update({ outputTokens: 400 });
  f.advance(1000);
  assert.match(f.messages.at(-1), /400 tokens · 26.6 t\/s/);
  f.progress.update({ phase: "validate" });
  await f.progress.finish("failed");
  assert.match(f.messages.at(-2), /Failed at: Validate and save checkpoint/);
  assert.equal(f.messages.at(-1), undefined);
  assert.equal(f.reports[0].phases.wait.elapsedMs, 65000);
  assert.equal(f.reports[0].waitPhases.queue.elapsedMs, 10000);
  assert.equal(f.reports[0].waitPhases.restore.elapsedMs, 20000);
  assert.equal(f.reports[0].waitPhases.prefill.elapsedMs, 30000);
  assert.equal(f.reports[0].waitPhases.first_token.elapsedMs, 5000);
  assert.equal(f.reports[0].tokens.cacheRead, 0);
  assert.equal(f.reports[0].transcriptAppended, false);
  assert.equal(f.stopped(), true);
});

test("checkpoint progress displays its effective output ceiling and retains a bounded finish reason", async (t) => {
  const f = fixture(t);
  f.progress.update({ phase: "submit", inputTokens: 239028, outputTokenLimit: 12288 });
  f.progress.update({ phase: "generate", outputTokens: 7000 });
  assert.match(f.messages.at(-1), /Checkpoint: 7,000 \/ 12,288 tokens/);
  f.progress.update({ phase: "validate", outputTokens: 12288, finishReason: "length" });
  await f.progress.finish("failed");
  assert.equal(f.reports[0].tokens.outputTokenLimit, 12288);
  assert.equal(f.reports[0].tokens.outputTokens, 12288);
  assert.equal(f.reports[0].finishReason, "length");
  assert.equal(f.reports[0].transcriptAppended, false);
  assert.equal(f.messages.at(-1), undefined);
});

test("compaction queue names the response boundary it is waiting for", async (t) => {
  const f = fixture(t, { observationAt: () => ({
    available: true, policy: "response_boundary",
    request: { ...currentChat, state: "queued", computed_tokens: 0, input_tokens: 240_000 },
    activeChat: otherChat, workerAvailable: true, otherRunningChatId: otherChat.chat_id,
    otherRunningRequest: { ...otherChat, state: "running", computed_tokens: 51_000, input_tokens: 50_000 },
  }) });
  f.progress.update({ phase: "wait", inputTokens: 240_000 });
  f.advance(20_000);
  assert.match(f.messages.at(-1), /GPU queue.*another chat c{12} has the GPU and is generating; waiting for its response to finish/);
  assert.doesNotMatch(f.messages.at(-1), /0 \/ 240,000/);
});

test("compaction distinguishes tool grace from a cache-bank transfer", (t) => {
  const f = fixture(t, { observationAt: () => ({
    available: true, request: { ...currentChat, state: "queued", computed_tokens: 0, input_tokens: 240_000 },
    activeChat: otherChat, workerAvailable: true,
    toolGrace: { ...otherChat, phase: "tool_grace", remaining_seconds: 1.4 },
  }) });
  f.progress.update({ phase: "wait", inputTokens: 240_000 });
  f.advance(1000);
  assert.match(f.messages.at(-1), /GPU queue.*chat c{12}'s tool.*1.4s grace remaining/);
  assert.doesNotMatch(f.messages.at(-1), /transfer is still completing|0 \/ 240,000/);
});

test("checkpoint shows separate three-second and total generation averages", (t) => {
  const f = fixture(t);
  f.progress.update({ phase: "wait", inputTokens: 238880, outputTokenLimit: 12288 });
  f.advance(166000);
  f.progress.update({ phase: "generate", outputTokens: 1 });
  f.advance(271000, false);
  f.progress.update({ outputTokens: 4430 });
  for (const outputTokens of [4470, 4510, 4550]) {
    f.advance(1000, false);
    f.progress.update({ outputTokens });
  }
  assert.match(f.messages.at(-1), /Checkpoint: 4,550 \/ 12,288 tokens · 40.0 t\/s, 16.6 t\/s avg/);
  assert.doesNotMatch(f.messages.at(-1), /round |acceptance/);
  assert.equal(f.progress.snapshot().phases.generate.elapsedMs, 274000);
});

test("sparse counters interpolate the window boundary and a silent stream decays to zero", (t) => {
  const f = fixture(t);
  f.progress.update({ phase: "generate", outputTokens: 1 });
  f.advance(10000, false);
  f.progress.update({ outputTokens: 401 });
  assert.match(f.messages.at(-1), /40.0 t\/s, 40.0 t\/s avg/);
  f.advance(1000);
  assert.match(f.messages.at(-1), /26.7 t\/s/);
  f.advance(2000);
  assert.match(f.messages.at(-1), /0.0 t\/s, 40.0 t\/s avg/);
  f.progress.update({ outputTokens: 401 });
  f.progress.update({ outputTokens: 300 });
  f.advance(1000);
  assert.match(f.messages.at(-1), /401 tokens · 0.0 t\/s, 40.0 t\/s avg/);
  assert.equal(f.progress.snapshot().tokens.outputTokens, 401);
});

test("GPU pauses show the blocker and freeze the rate window without changing wall timers", (t) => {
  let observation = { available: true, workerAvailable: true, activeChat: currentChat,
    request: { ...currentChat, state: "running", computed_tokens: 240000, input_tokens: 239000 } };
  const f = fixture(t, { observationAt: () => observation });
  f.progress.update({ phase: "generate", outputTokens: 1 });
  for (const outputTokens of [41, 81, 121, 161]) {
    f.advance(1000, false);
    f.progress.update({ outputTokens });
  }
  observation = { ...observation, activeChat: otherChat, otherRunningChatId: otherChat.chat_id,
    request: { ...observation.request, state: "paused" } };
  f.advance(0);
  assert.match(f.messages.at(-1), /Checkpoint generation paused/);
  assert.match(f.messages.at(-1), /paused: another chat c{12}.*GPU/);
  assert.doesNotMatch(f.messages.at(-1), /t\/s/);
  f.advance(15000);
  observation = { ...observation, activeChat: currentChat, otherRunningChatId: undefined,
    request: { ...observation.request, state: "running" } };
  f.advance(0);
  assert.match(f.messages.at(-1), /40.0 t\/s, 40.0 t\/s avg/);
  f.advance(1000, false);
  f.progress.update({ outputTokens: 201 });
  assert.match(f.messages.at(-1), /40.0 t\/s, 40.0 t\/s avg/);
  assert.equal(f.progress.snapshot().phases.generate.elapsedMs, 20000);
  f.progress.update({ phase: "validate" });
  f.advance(10000);
  assert.doesNotMatch(f.messages.at(-1), /t\/s/);
});

test("character-only progress never invents a token rate and late counters start their own window", (t) => {
  const f = fixture(t);
  f.progress.update({ phase: "generate", characters: 100 });
  f.advance(20000);
  assert.doesNotMatch(f.messages.at(-1), /t\/s/);
  f.progress.update({ outputTokens: 1000 });
  f.advance(1000, false);
  f.progress.update({ outputTokens: 1040 });
  assert.match(f.messages.at(-1), /40.0 t\/s, 40.0 t\/s avg/);
});

test("concurrent compactions have independent rolling windows", (t) => {
  const first = fixture(t), second = fixture(t);
  first.progress.update({ phase: "generate", outputTokens: 1 });
  second.progress.update({ phase: "generate", outputTokens: 1 });
  first.advance(1000, false);
  second.advance(1000, false);
  first.progress.update({ outputTokens: 41 });
  second.progress.update({ outputTokens: 21 });
  assert.match(first.messages.at(-1), /40.0 t\/s, 40.0 t\/s avg/);
  assert.match(second.messages.at(-1), /20.0 t\/s, 20.0 t\/s avg/);
});

test("compaction refreshes round and three-second acceptance at the normal 100 ms cadence", (t) => {
  const timing = { phase: "generate", phase_elapsed_ms: 0, last_round_ms: 44.12,
    acceptance_rate_3s: 0.6, acceptance_rate: 0.9, last_acceptance_rate: 1 };
  const observation = { available: true, requestPhase: timing };
  const f = fixture(t, { observationAt: () => observation });
  assert.equal(f.intervalMs(), 100);
  f.progress.update({ phase: "generate", outputTokens: 1 });
  f.advance(1000, false);
  f.progress.update({ outputTokens: 41 });
  assert.match(f.messages.at(-1), /40.0 t\/s, 40.0 t\/s avg · round 44.1 ms • acceptance 60.0%/);
  assert.doesNotMatch(f.messages.at(-1), /acceptance (90.0|100.0)%/);
  timing.last_round_ms = 46.23;
  timing.acceptance_rate_3s = 0.65;
  f.advance(100);
  assert.match(f.messages.at(-1), /round 46.2 ms • acceptance 65.0%/);
  // The final telemetry row may move to recent before the last SSE frame.
  observation.requestPhase = undefined;
  observation.lastRequestTiming = { ...timing, last_round_ms: 45.5, acceptance_rate_3s: 0 };
  f.advance(100);
  assert.match(f.messages.at(-1), /round 45.5 ms • acceptance 0.0%/);
  observation.lastRequestTiming = { last_round_ms: null, acceptance_rate_3s: null };
  f.advance(100);
  assert.doesNotMatch(f.messages.at(-1), /round |acceptance/);
});

test("paused compaction retains round telemetry without charging the GPU wait to average throughput", (t) => {
  let observation = { available: true, activeChat: currentChat,
    request: { ...currentChat, state: "running", computed_tokens: 240000, input_tokens: 239000 },
    lastRequestTiming: { last_round_ms: 43.7, acceptance_rate_3s: 0.57 } };
  const f = fixture(t, { observationAt: () => observation });
  f.progress.update({ phase: "generate", outputTokens: 1 });
  f.advance(1000, false);
  f.progress.update({ outputTokens: 41 });
  observation = { ...observation, activeChat: otherChat, otherRunningChatId: otherChat.chat_id,
    request: { ...observation.request, state: "paused" } };
  f.advance(0);
  f.advance(15000);
  assert.match(f.messages.at(-1), /paused: another chat c{12}.*round 43.7 ms • acceptance 57.0%/);
  assert.doesNotMatch(f.messages.at(-1), /t\/s/);
  observation = { ...observation, activeChat: currentChat, otherRunningChatId: undefined,
    request: { ...observation.request, state: "running" } };
  f.advance(0);
  f.advance(1000, false);
  f.progress.update({ outputTokens: 81 });
  assert.match(f.messages.at(-1), /40.0 t\/s, 40.0 t\/s avg/);
});

test("a partial cache hit reports physical prefill separately from the cached-token split", async (t) => {
  const f = fixture(t, { observationAt: () => ({ available: true,
    request: { ...currentChat, state: "running", computed_tokens: 235_664, input_tokens: 239_265 },
    activeChat: currentChat, workerAvailable: true }) });
  f.progress.update({ phase: "wait", inputTokens: 239265, cacheRead: 235664 });
  assert.match(f.messages.at(-1), /Radiance compaction: Prompt prefill/);
  assert.match(f.messages.at(-1), /235,664 \/ 239,265 prompt tokens prepared/);
  assert.match(f.messages.at(-1), /cached 235,664 \(98.5%\) · uncached 3,601/);
});

test("a fully prepared prompt waits for first output without claiming isolated GPU generation time", async (t) => {
  const f = fixture(t, { observationAt: () => ({ available: true,
    request: { ...currentChat, state: "running", computed_tokens: 239_265, input_tokens: 239_265 },
    activeChat: currentChat, workerAvailable: true }) });
  f.progress.update({ phase: "wait", inputTokens: 239265, cacheRead: 239265 });
  assert.match(f.messages.at(-1), /Radiance compaction: Await first checkpoint output/);
  assert.match(f.messages.at(-1), /the prompt is ready; checkpoint output has not reached Pi yet/);
  assert.match(f.messages.at(-1), /Prompt prefill: not observed yet/);
  assert.match(f.messages.at(-1), /cached 239,265 \(100.0%\) · uncached 0/);
});

test("first-output waits missed between samples do not leave an unobserved or zero-duration row", (t) => {
  const f = fixture(t, { observationAt: (clock) => ({ available: true,
    request: { ...currentChat, state: "running", computed_tokens: clock < 500 ? 235_664 : 239_265, input_tokens: 239_265 },
    activeChat: currentChat, workerAvailable: true }) });
  f.progress.update({ phase: "wait", inputTokens: 239265 });
  assert.doesNotMatch(f.messages.at(-1), /Await first checkpoint output|Generate first checkpoint token/);
  // The final prompt-ready observation arrives in the same callback that
  // announces the first generated token; there is no separately timed wait.
  f.advance(500, false);
  f.progress.update({ phase: "generate", outputTokens: 1 });
  assert.doesNotMatch(f.messages.at(-1), /Await first checkpoint output|Generate first checkpoint token/);
  assert.equal(f.progress.snapshot().waitPhases.first_token.elapsedMs, 0);
});

test("actual cached/uncached breakdown and saved-checkpoint recovery are explicit", async (t) => {
  const f = fixture(t);
  f.progress.update({ phase: "validate", reusedCheckpoint: true, inputTokens: 241594, cacheRead: 237312, outputTokens: 3626 });
  assert.match(f.messages.at(-1), /cached 237,312 \(98.2%\) · uncached 4,282/);
  assert.match(f.messages.at(-1), /Saved checkpoint reused; no new generation/);
  assert.match(f.messages.at(-1), /Generate checkpoint: skipped/);
  f.progress.update({ phase: "commit" });
  f.advance(5000);
  assert.equal(f.progress.snapshot().state, "running");
  assert.equal(f.reports.length, 0, "a saved receipt is not yet a committed compaction");
  f.abort.abort();
  await f.progress.finish("cancelled");
  assert.equal(f.reports[0].state, "cancelled");
  assert.equal(f.stopped(), true);
});

test("a committed transcript remains committed if Escape arrives during cleanup", async (t) => {
  const f = fixture(t);
  f.progress.update({ phase: "commit" });
  f.progress.markAppended();
  f.progress.markCommitted();
  f.progress.update({ phase: "cleanup" });
  f.advance(3000);
  f.abort.abort();
  assert.equal(f.progress.snapshot().state, "running");
  assert.equal(f.stopped(), false);
  await f.progress.finish("cleanup_pending");
  assert.match(f.messages.at(-2), /Compacted; snapshot cleanup pending/);
  assert.equal(f.messages.at(-1), undefined);
  assert.equal(f.reports[0].durableCommit, true);
  assert.equal(f.reports[0].phases.cleanup.elapsedMs, 3000);
});

test("successful cleanup clears the spinner override and saves only bounded diagnostic fields", async (t) => {
  const f = fixture(t);
  f.progress.update({ phase: "commit", summary: "PRIVATE FIXTURE SUMMARY", title: "PRIVATE TITLE",
    finishReason: "PRIVATE FIXTURE REASON" });
  f.progress.markAppended();
  f.progress.markCommitted();
  f.progress.update({ phase: "cleanup" });
  f.advance(2000);
  f.progress.update({ removedBytes: 2 ** 30 });
  await f.progress.finish("complete");
  const rendered = f.messages.at(-2);
  assert.match(rendered, /Compacted/);
  assert.match(rendered, /freed 1.00 GiB/);
  assert.equal(f.messages.at(-1), undefined);
  assert.equal(f.reports[0].removedBytes, 2 ** 30);
  assert.doesNotMatch(JSON.stringify(f.reports[0]), /PRIVATE|synthetic-/);
  const renders = f.messages.length;
  f.advance(60000);
  assert.equal(f.messages.length, renders, "the completed timer must stop");
  await clearCompactionProgress(f.ctx);
  assert.equal(getCompactionProgress(f.ctx), undefined);
  assert.equal(f.widgets.at(-1), undefined);
});

test("a timing-report write failure cannot leave stale progress on screen", async (t) => {
  const messages = [], notices = [];
  const sessionFile = `/synthetic-${randomUUID()}.jsonl`;
  const ctx = { sessionManager: { getSessionFile: () => sessionFile },
    ui: { setWorkingMessage: (message) => messages.push(message), notify: (message) => notices.push(message) } };
  const progress = startCompactionProgress(ctx, { save: () => { throw new Error("Synthetic disk full"); } });
  t.after(() => clearCompactionProgress(ctx));
  await progress.finish("complete");
  assert.equal(messages.at(-1), undefined);
  assert.deepEqual(notices, ["Could not save compaction timing metadata."]);
});

test("shutdown does not call an appended but unflushed conversation durable", async (t) => {
  const f = fixture(t);
  f.progress.update({ phase: "commit" });
  f.progress.markAppended();
  await clearCompactionProgress(f.ctx);
  assert.equal(f.reports[0].state, "commit_unconfirmed");
  assert.equal(f.reports[0].durableCommit, false);
  assert.equal(f.stopped(), true);
});

test("the exact final total is linked to the compaction without storing chat content", async (t) => {
  let clock = 0;
  const entries = [];
  const compactionEntry = { type: "compaction", id: "compacted", details: { elapsedMs: 1000 } };
  entries.push(compactionEntry);
  const ctx = { sessionManager: { getSessionFile: () => `/synthetic-${randomUUID()}.jsonl`,
    getEntries: () => entries }, ui: { setWorkingMessage() {} } };
  const progress = startCompactionProgress(ctx, { now: () => clock, save: () => {},
    schedule: () => 1, unschedule: () => {} });
  t.after(() => clearCompactionProgress(ctx));
  progress.markAppended();
  clock = 94321;
  await progress.finish("complete");
  const pi = { appendEntry(customType, data) {
    entries.push({ type: "custom", customType, data, parentId: compactionEntry.id });
  } };

  assert.equal(appendCompactionTiming(pi, { compactionEntry }, ctx, progress), true);
  assert.equal(compactionEntry.details.elapsedMs, 94321);
  assert.deepEqual(entries.at(-1), { type: "custom", customType: COMPACTION_TIMING_ENTRY,
    data: { compactionEntryId: "compacted", elapsedMs: 94321 }, parentId: "compacted" });
  assert.equal(appendCompactionTiming(pi, { compactionEntry }, ctx, progress), false, "duplicates are rejected");
  assert.doesNotMatch(JSON.stringify(entries.at(-1)), /summary|message|content/i);
});

test("backend timings preserve cache steps that finish between spinner samples", (t) => {
  let observation = { available: true, request: { ...currentChat, state: "queued", input_tokens: 1000, computed_tokens: 0 },
    phaseObservedAt: Date.now(), requestPhase: { phase: "prefill", phase_elapsed_ms: 10, cached_tokens: 900,
      input_tokens: 1000, computed_tokens: 920, timings_ms: { admission: 1, cache_update: 237, cache_lookup: 3, cache_restore: 20, prefill: 10 } } };
  const f = fixture(t, { observationAt: () => observation });
  f.progress.update({ phase: "wait" });
  const report = f.progress.snapshot();
  assert.equal(report.waitPhases.cache_update.elapsedMs, 237);
  assert.equal(report.waitPhases.cache_lookup.elapsedMs, 3);
  assert.equal(report.waitPhases.restore.elapsedMs, 20);
  assert.equal(report.waitPhases.queue.observed, false);
  assert.match(f.messages.at(-1), /Finish previous cache update: 0.2s/);
  assert.match(f.messages.at(-1), /Check reusable context: 0.0s/);
});
