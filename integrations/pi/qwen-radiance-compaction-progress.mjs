import { randomUUID } from "node:crypto";

import { createSchedulerTelemetry, toolGraceDescription, requestPhaseStatus } from "./qwen-radiance-scheduler-telemetry.mjs";

const STAGES = [
  ["prepare", "Prepare checkpoint"],
  ["tokenize", "Prepare prompt for cache reuse"],
  ["submit", "Submit checkpoint request"],
  ["wait", "Prepare model context"],
  ["generate", "Generate checkpoint"],
  ["validate", "Validate and save checkpoint"],
  ["flush", "Flush pending KV tail"],
  ["commit", "Commit conversation"],
  ["cleanup", "Retire old snapshots"],
];
const WAIT_STAGES = [
  ["admission", "Scheduler admission"],
  ["queue", "GPU queue"],
  ["handover", "GPU cache-bank handover"],
  ["ram_allocation", "Allocate RAM for chat handover"],
  ["cache_update", "Finish previous cache update"],
  ["cache_lookup", "Check reusable context"],
  ["restore", "Find and load cached context"],
  ["prefill", "Prompt prefill"],
  ["first_token", "Await first checkpoint output"],
  ["unknown", "Unclassified backend wait"],
];
const ALWAYS_VISIBLE_WAIT_STAGES = new Set(["queue", "restore", "prefill"]);
const KEY = "qwen-compaction";
export const COMPACTION_TIMING_ENTRY = "qwen-radiance-compaction-timing-v1";
// Pi loads the TS compactor and the JS cache extension separately. Share the
// controller even when their loaders instantiate this module more than once.
const registry = globalThis[Symbol.for("qwen.radiance.compaction.progress")] ??= new Map();
const sessionKey = (ctx) => ctx.sessionManager.getSessionFile() ?? ctx.sessionManager.getSessionId();
const duration = (ms) => ms < 60000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
const count = (value) => value.toLocaleString("en-GB");
const validCount = (value) => Number.isSafeInteger(value) && value >= 0;
const RATE_WINDOW_MS = 3000;
const BACKEND_WAIT_PHASES = {
  admission: "admission", gpu_queue: "queue", priority_wait: "queue", priority_preempt: "queue", tool_grace: "queue", handover: "handover",
  ram_allocation: "ram_allocation", cache_update: "cache_update", cache_lookup: "cache_lookup",
  cache_restore: "restore", prefill: "prefill",
};

const stageLabel = (id) => STAGES.find(([stage]) => stage === id)[1];
const waitStageLabel = (id) => WAIT_STAGES.find(([stage]) => stage === id)[1];
const shortChat = (id) => id?.slice(0, 12);

function sameIdentity(left, right) {
  return left?.chat_id === right?.chat_id && left?.generation === right?.generation;
}

export function classifyCompactionWait(observation) {
  if (!observation?.available) {
    return {
      id: "unknown",
      detail: observation?.configured === false
        ? "scheduler telemetry is not configured"
        : "scheduler telemetry is unavailable",
    };
  }
  const request = observation.request;
  const exact = requestPhaseStatus(observation);
  if (exact) return {
    id: BACKEND_WAIT_PHASES[observation.requestPhase.phase] ?? "first_token",
    detail: `${exact.phase} · ${exact.detail}`,
  };
  if (observation.toolGrace && request?.state !== "running") {
    return { id: "queue", detail: toolGraceDescription(observation.toolGrace) };
  }
  const active = observation.activeChat;
  const otherRunning = observation.otherRunningChatId;
  const otherRequest = observation.otherRunningRequest;
  const differentPhysicalBank = request && active && !sameIdentity(active, request);
  const sameChatPredecessor = differentPhysicalBank && active.chat_id === request.chat_id;
  const blocker = otherRunning ?? (differentPhysicalBank ? active.chat_id : undefined);
  const blockerDetail = sameChatPredecessor
    ? "the previous generation of this chat still owns the GPU bank"
    : blocker
      ? otherRequest
        ? `another chat ${shortChat(blocker)} has the GPU and is ` +
          (otherRequest.computed_tokens < otherRequest.input_tokens
            ? `prefilling ${count(otherRequest.computed_tokens)} / ${count(otherRequest.input_tokens)} prompt tokens`
            : "generating")
        : `another chat ${shortChat(blocker)} still owns the GPU bank`
      : undefined;

  if (request?.state === "paused") {
    return {
      id: "queue",
      detail: blockerDetail ?? "the fair scheduler paused this request between GPU time slices",
    };
  }
  if (request?.state === "queued" && blocker) {
    return { id: otherRunning ? "queue" : "handover", detail: blockerDetail +
      (otherRunning && observation.policy === "response_boundary" ? "; waiting for its response to finish" : "") };
  }
  if (request?.state === "running" && differentPhysicalBank) {
    return { id: "handover", detail: `${blockerDetail}; the RAM/GPU bank transfer is still completing` };
  }
  if (request?.state === "queued") {
    if (observation.workerAvailable && sameIdentity(active, request)) {
      return {
        id: "restore",
        detail: "this chat has the GPU; finding and loading its cached context",
      };
    }
    return {
      id: "admission",
      detail: observation.workerAvailable
        ? "the request is queued and no other chat currently owns the GPU"
        : "the request is queued; physical GPU-bank telemetry is unavailable",
    };
  }
  if (request?.state === "running") {
    if (request.computed_tokens < request.input_tokens) {
      return {
        id: "prefill",
        detail: `${count(request.computed_tokens)} / ${count(request.input_tokens)} prompt tokens prepared`,
      };
    }
    return { id: "first_token", detail: "the prompt is ready; checkpoint output has not reached Pi yet" };
  }
  if (otherRunning) {
    return {
      id: "queue",
      detail: `another chat ${shortChat(otherRunning)} has the GPU; ` +
        (observation.policy === "response_boundary"
          ? "waiting for its response to finish"
          : "the compaction request is awaiting admission"),
    };
  }
  return {
    id: "admission",
    detail: "the HTTP request was submitted but has not appeared in the scheduler yet",
  };
}

export function getCompactionProgress(ctx) {
  return registry.get(sessionKey(ctx));
}

export function appendCompactionTiming(pi, event, ctx, progressOrReport) {
  const report = typeof progressOrReport?.snapshot === "function" ? progressOrReport.snapshot() : progressOrReport;
  const compactionEntry = event?.compactionEntry;
  if (!report?.transcriptAppended || typeof compactionEntry?.id !== "string" ||
      !Number.isSafeInteger(report.elapsedMs) || report.elapsedMs < 0) return false;
  const activeEntries = ctx.sessionManager.getBranch?.() ?? ctx.sessionManager.getEntries();
  const duplicate = activeEntries.some((entry) =>
    entry.type === "custom" && entry.customType === COMPACTION_TIMING_ENTRY &&
    entry.data?.compactionEntryId === compactionEntry.id);
  if (duplicate) return false;
  // The result and saved entry share this details object until compaction_end,
  // allowing the freshly rendered summary to use the exact final total too.
  if (compactionEntry.details && typeof compactionEntry.details === "object") {
    compactionEntry.details.elapsedMs = report.elapsedMs;
  }
  pi.appendEntry(COMPACTION_TIMING_ENTRY, {
    compactionEntryId: compactionEntry.id,
    elapsedMs: report.elapsedMs,
  });
  return true;
}

export async function clearCompactionProgress(ctx) {
  const progress = getCompactionProgress(ctx);
  if (progress) {
    const report = progress.snapshot();
    await progress.finish(report.durableCommit ? "cleanup_pending" : report.transcriptAppended ? "commit_unconfirmed" : "cancelled");
  }
  registry.delete(sessionKey(ctx));
  ctx.ui.setWidget?.(KEY, undefined);
  ctx.ui.setStatus?.(KEY, undefined);
}

export function startCompactionProgress(ctx, {
  signal, save = async () => {}, now = () => performance.now(),
  schedule = setInterval, unschedule = clearInterval,
  scheduler = createSchedulerTelemetry(), schedulerNow = () => Date.now(),
} = {}) {
  const started = now(), key = sessionKey(ctx);
  // Remove the panel/footer left by an earlier extension version or attempt.
  ctx.ui.setWidget?.(KEY, undefined);
  ctx.ui.setStatus?.(KEY, undefined);
  const stages = new Map(STAGES.map(([id]) => [id, { state: "pending", elapsedMs: 0 }]));
  const waitPhases = new Map(WAIT_STAGES.map(([id]) => [id, { observed: false, elapsedMs: 0 }]));
  const waitDetails = new Map();
  let phase = "prepare", phaseStarted = started, state = "running", lastRender = -Infinity, timer, saving, finishedAt;
  let activeWaitId, activeWaitStarted, schedulerStopped = false;
  let transcriptAppended = false, durableCommit = false, reusedCheckpoint = false;
  const tokens = {};
  const rateSamples = [];
  let firstRateAt, ratePausedAt, excludedRateWaitMs = 0, generationWait;
  let removedBytes, finishReason;
  let unsubscribe;
  stages.get(phase).state = "running";
  const identity = { schema: "urn:qwen-r9700:radiance-compaction-progress:v1", attemptId: randomUUID(), startedAt: new Date().toISOString() };

  function readScheduler() {
    try {
      return scheduler.read?.(schedulerNow()) ?? { available: false, configured: false };
    } catch {
      return { available: false, configured: true };
    }
  }

  function rateClock(at) {
    return (ratePausedAt ?? at) - excludedRateWaitMs;
  }

  function resumeRate(at) {
    if (ratePausedAt !== undefined) excludedRateWaitMs += at - ratePausedAt;
    ratePausedAt = undefined;
    generationWait = undefined;
  }

  function observeGeneration(at, outputAdvanced) {
    if (state !== "running" || phase !== "generate") return;
    // A new stream delta wins over a scheduler sample awaiting its next refresh.
    const activity = outputAdvanced ? undefined : classifyCompactionWait(readScheduler());
    if (activity && ["queue", "handover"].includes(activity.id)) {
      ratePausedAt ??= at;
      generationWait = activity;
    } else {
      resumeRate(at);
    }
  }

  function trimRateSamples(cutoff) {
    let discard = 0;
    // Retain one sample at/before the boundary for sparse-counter interpolation.
    while (discard + 1 < rateSamples.length && rateSamples[discard + 1].at <= cutoff) discard++;
    if (discard) rateSamples.splice(0, discard);
  }

  function recordRate(at, outputTokens) {
    const activeAt = rateClock(at);
    firstRateAt ??= activeAt;
    const previous = rateSamples.at(-1);
    if (previous?.at === activeAt) previous.tokens = outputTokens;
    else rateSamples.push({ at: activeAt, tokens: outputTokens });
    trimRateSamples(activeAt - RATE_WINDOW_MS);
  }

  function rollingRate(at) {
    if (firstRateAt === undefined || rateSamples.length === 0) return undefined;
    const activeAt = rateClock(at), start = Math.max(firstRateAt, activeAt - RATE_WINDOW_MS);
    const seconds = (activeAt - start) / 1000;
    if (seconds < 0.5) return undefined;
    trimRateSamples(start);
    const [before, after] = rateSamples;
    let initialTokens = before.tokens;
    if (after && before.at < start) {
      initialTokens += (after.tokens - before.tokens) * (start - before.at) / (after.at - before.at);
    }
    return Math.max(0, tokens.outputTokens - initialTokens) / seconds;
  }

  function observeWait(at) {
    if (state !== "running" || phase !== "wait") return;
    const observation = readScheduler();
    const activity = classifyCompactionWait(observation);
    if (activity.id !== activeWaitId) {
      if (activeWaitId !== undefined && activeWaitStarted !== undefined) {
        waitPhases.get(activeWaitId).elapsedMs += at - activeWaitStarted;
      }
      activeWaitId = activity.id;
      activeWaitStarted = at;
      waitPhases.get(activeWaitId).observed = true;
    }
    waitDetails.set(activity.id, activity.detail);
    if (observation.requestPhase) {
      const measured = new Map();
      for (const [backend, elapsedMs] of Object.entries(observation.requestPhase.timings_ms)) {
        const id = BACKEND_WAIT_PHASES[backend];
        if (id) measured.set(id, (measured.get(id) ?? 0) + elapsedMs);
      }
      for (const [id, elapsedMs] of measured) {
        Object.assign(waitPhases.get(id), { observed: true, elapsedMs });
      }
      if (measured.has(activeWaitId)) activeWaitStarted = at;
    }
  }

  function closeWait(at) {
    if (phase !== "wait") return;
    observeWait(at);
    if (activeWaitId !== undefined && activeWaitStarted !== undefined) {
      waitPhases.get(activeWaitId).elapsedMs += at - activeWaitStarted;
      activeWaitStarted = undefined;
    }
  }

  function stopScheduler() {
    if (schedulerStopped) return;
    schedulerStopped = true;
    unsubscribe?.();
    try { scheduler.stop?.(); } catch { /* Compaction completion must not depend on diagnostics. */ }
  }

  function snapshot(at = now()) {
    const phaseTimings = Object.fromEntries([...stages].map(([id, value]) => [id, { ...value,
      elapsedMs: Math.round(value.elapsedMs + (state === "running" && id === phase ? at - phaseStarted : 0)) }]));
    const waitPhaseTimings = Object.fromEntries([...waitPhases].map(([id, value]) => [id, { ...value,
      elapsedMs: Math.round(value.elapsedMs + (state === "running" && phase === "wait" &&
        id === activeWaitId && activeWaitStarted !== undefined ? at - activeWaitStarted : 0)) }]));
    return { ...identity, state, phase, elapsedMs: Object.values(phaseTimings).reduce((sum, value) => sum + value.elapsedMs, 0),
      phases: phaseTimings, waitPhases: waitPhaseTimings, tokens: { ...tokens }, reusedCheckpoint,
      transcriptAppended, durableCommit,
      ...(finishReason ? { finishReason } : {}),
      ...(finishedAt ? { finishedAt } : {}),
      ...(removedBytes === undefined ? {} : { removedBytes }) };
  }

  function pushWaitLines(lines, report) {
    for (const [id, label] of WAIT_STAGES) {
      const timing = report.waitPhases[id];
      if (!ALWAYS_VISIBLE_WAIT_STAGES.has(id) && !timing.observed) continue;
      const current = state === "running" && phase === "wait" && id === activeWaitId && activeWaitStarted !== undefined;
      // The prompt-ready sample and first stream output can arrive together.
      // Keep a useful live wait, but omit an unmeasured or zero-duration row.
      if (id === "first_token" && !current && timing.elapsedMs === 0) continue;
      const marker = current ? "›" : timing.observed ? "✓" : "·";
      const value = timing.observed
        ? duration(timing.elapsedMs)
        : state === "running" && phase === "wait" ? "not observed yet" : "not observed";
      const detail = current || id === "queue" || id === "handover" ? waitDetails.get(id) : undefined;
      lines.push(`  ${marker} ${label}: ${value}${detail ? ` · ${detail}` : ""}`);
    }
  }

  function render(force = false, outputAdvanced = false) {
    // Continuous usage arrives at token cadence. Refresh at most once a second,
    // except at phase transitions, and keep ticking during a silent HTTP wait.
    const at = now();
    const wasPaused = Boolean(generationWait);
    observeGeneration(at, outputAdvanced);
    force ||= wasPaused !== Boolean(generationWait);
    if (!force && at - lastRender < 1000) return;
    lastRender = at;
    observeWait(at);
    const report = snapshot(at);
    const label = phase === "generate" && generationWait ? "Checkpoint generation paused" :
      phase === "wait" && activeWaitId ? waitStageLabel(activeWaitId) : stageLabel(phase);
    const outcome = state === "running" ? label : {
      complete: "Compacted", failed: `Failed at: ${label}`, cancelled: "Cancelled; transcript retained",
      cleanup_pending: "Compacted; snapshot cleanup pending", commit_unconfirmed: "Conversation commit unconfirmed",
    }[state];
    const headline = `Radiance compaction: ${outcome} · ${duration(report.elapsedMs)} total`;
    const lines = [headline];
    if (state === "running") {
      for (const [id] of STAGES) {
        const stage = report.phases[id];
        if (id === "wait" && !["pending", "skipped"].includes(stage.state)) {
          pushWaitLines(lines, report);
        } else {
          const name = stageLabel(id);
          lines.push(`  ${id === phase ? "›" : stage.state === "done" ? "✓" : "·"} ${name}: ` +
            (stage.state === "pending" ? "waiting" : stage.state === "skipped" ? "skipped" : duration(stage.elapsedMs)));
        }
      }
    } else {
      const completed = [];
      for (const [id] of STAGES.filter(([stage]) => report.phases[stage].state !== "pending")) {
        if (id === "wait" && report.phases[id].state !== "skipped") {
          for (const [waitId, waitLabel] of WAIT_STAGES) {
            const timing = report.waitPhases[waitId];
            if (timing.observed) completed.push(`${waitLabel}: ${duration(timing.elapsedMs)}`);
          }
        } else {
          completed.push(`${stageLabel(id)}: ${report.phases[id].state === "skipped" ? "skipped" : duration(report.phases[id].elapsedMs)}`);
        }
      }
      lines.push(completed.join(" · "));
    }
    if (validCount(tokens.inputTokens)) {
      lines.push(validCount(tokens.cacheRead) ?
        `Input: ${count(tokens.inputTokens)} tok · cached ${count(tokens.cacheRead)} (${(100 * tokens.cacheRead / Math.max(1, tokens.inputTokens)).toFixed(1)}%) · uncached ${count(tokens.inputTokens - tokens.cacheRead)}` :
        `Input: ${count(tokens.inputTokens)} tok · cached/uncached split pending`);
    }
    if (reusedCheckpoint) lines.push("Saved checkpoint reused; no new generation" +
      (tokens.outputTokens ? ` (${count(tokens.outputTokens)} tokens)` : "") + ". Counts describe the saved request.");
    if (!reusedCheckpoint && (tokens.outputTokens || tokens.characters)) {
      const rate = state === "running" && phase === "generate" ? rollingRate(at) : undefined;
      const rateText = state === "running" && phase === "generate" && generationWait
        ? ` · paused: ${generationWait.detail}`
        : rate === undefined ? "" : ` · ${rate.toFixed(1)} tok/s (3s)`;
      lines.push(tokens.outputTokens ? `Checkpoint: ${count(tokens.outputTokens)}` +
        (tokens.outputTokenLimit ? ` / ${count(tokens.outputTokenLimit)}` : "") + " tokens" +
        rateText :
        `Checkpoint: ${count(tokens.characters)} characters; token count pending`);
    }
    if (state !== "running" && removedBytes !== undefined) lines.push(`Old snapshots: freed ${(removedBytes / 1024 ** 3).toFixed(2)} GiB`);
    // The pinned runtime forwards this message to its native compaction spinner.
    // Pi removes that component at compaction_end, including cancellation/failure.
    ctx.ui.setWorkingMessage?.(lines.join("\n"));
  }

  function update(value) {
    if (state !== "running") return;
    const at = now();
    const changed = value.phase && value.phase !== phase;
    const previousCacheRead = tokens.cacheRead;
    const tokensAdvanced = validCount(value.outputTokens) && value.outputTokens > (tokens.outputTokens ?? 0);
    const outputAdvanced = tokensAdvanced || (validCount(value.characters) && value.characters > (tokens.characters ?? 0));
    if (changed) {
      const nextIndex = STAGES.findIndex(([id]) => id === value.phase);
      if (nextIndex < 0) throw new Error("unknown compaction phase");
      if (nextIndex < STAGES.findIndex(([id]) => id === phase)) throw new Error("compaction progress moved backwards");
      if (phase === "wait") closeWait(at);
      const previous = stages.get(phase);
      previous.elapsedMs += at - phaseStarted;
      previous.state = "done";
      for (const [id] of STAGES.slice(0, nextIndex)) {
        if (stages.get(id).state === "pending") stages.get(id).state = "skipped";
      }
      phase = value.phase; phaseStarted = at;
      stages.get(phase).state = "running";
      if (phase === "wait") observeWait(at);
    }
    for (const field of ["inputTokens", "outputTokens", "outputTokenLimit", "characters"]) {
      if (validCount(value[field]) && (field !== "outputTokens" || value[field] >= (tokens[field] ?? 0))) tokens[field] = value[field];
    }
    if (phase === "generate" && outputAdvanced) resumeRate(at);
    if (phase === "generate" && tokensAdvanced) recordRate(at, tokens.outputTokens);
    if (validCount(value.cacheRead) && value.cacheRead <= tokens.inputTokens) tokens.cacheRead = value.cacheRead;
    if (value.reusedCheckpoint) reusedCheckpoint = true;
    if (["stop", "length", "content_filter", "tool_calls", "error"].includes(value.finishReason)) finishReason = value.finishReason;
    if (validCount(value.removedBytes)) removedBytes = value.removedBytes;
    render(changed || tokens.cacheRead !== previousCacheRead, outputAdvanced);
    return snapshot();
  }

  function finish(outcome) {
    if (state !== "running") return saving;
    const at = now();
    if (phase === "wait") closeWait(at);
    stages.get(phase).elapsedMs += at - phaseStarted;
    stages.get(phase).state = outcome === "complete" ? "done" : outcome;
    state = outcome;
    finishedAt = new Date().toISOString();
    unschedule(timer);
    signal?.removeEventListener("abort", onAbort);
    stopScheduler();
    render(true);
    // Do not carry the completed breakdown into the next ordinary working row.
    ctx.ui.setWorkingMessage?.();
    const report = snapshot();
    saving = Promise.resolve().then(() => save(report)).catch(() => {
      ctx.ui.notify?.("Could not save compaction timing metadata.", "warning");
    });
    return saving;
  }

  const onAbort = () => { void finish("cancelled"); };
  const controller = { update, snapshot, finish, markAppended() {
    transcriptAppended = true;
    // Escape can no longer undo an entry that Pi has already appended.
    signal?.removeEventListener("abort", onAbort);
  }, markCommitted() { durableCommit = true; } };
  registry.set(key, controller);
  try {
    if (typeof scheduler.start === "function") scheduler.start(ctx);
    else scheduler.bind?.(ctx);
  } catch { /* The spinner reports unavailable telemetry while compaction continues. */ }
  unsubscribe = scheduler.subscribe?.(() => { if (state === "running") render(true); });
  timer = schedule(() => render(), 1000);
  timer?.unref?.();
  signal?.addEventListener("abort", onAbort, { once: true });
  render(true);
  if (signal?.aborted) onAbort();
  return controller;
}
