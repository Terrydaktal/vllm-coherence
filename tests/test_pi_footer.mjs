import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { startCompactionProgress, clearCompactionProgress } from "../integrations/pi/qwen-radiance-compaction-progress.mjs";
import installRequestProgress from "../integrations/pi/qwen-progress.mjs";
import { formatCacheBreakdown } from "../integrations/pi/qwen-cache-residency.mjs";
import { formatTemperatureStatus } from "../integrations/pi/qwen-gpu-temperature.mjs";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
test("installed footer wraps with one VRAM cache label and bare memory usage after fan", {
  skip: !existsSync(join(root, "pi-coding-agent/dist/modes/interactive/components/footer.js")),
}, async () => {
  const mod = path => import(pathToFileURL(join(root, path)));
  const { FooterComponent } = await mod("pi-coding-agent/dist/modes/interactive/components/footer.js");
  const { initTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { stripTerminalSequences, visibleWidth } = await mod("pi-tui/dist/index.js");
  initTheme("dark", false);
  const statuses = new Map([
    ["qwen-gpu-temperature", "40°C · 38°C · 25% · 28.5 / 31.9 GiB"],
    ["qwen-cache-residency", "VRAM 60,000 · RAM 0 · Disk 58,000 · Cold 160"],
    ["other", "another extension status"],
  ]);
  const session = {
    state: { model: { id: "radiance-model-with-a-long-name", provider: "qwen-r9700", reasoning: true, contextWindow: 253_792 }, thinkingLevel: "xhigh" },
    sessionManager: { getEntries: () => [
      { type: "message", message: { role: "assistant", usage: { input: 1_100_000, output: 0, cacheRead: 0, cacheWrite: 0, cost: { total: 0 } } } },
      { type: "message", message: { role: "assistant", usage: { input: 3_700_000, output: 845_000, cacheRead: 109_000_000, cacheWrite: 0, cost: { total: 0 } } } },
    ], getCwd: () => "/synthetic/project", getSessionName: () => "Synthetic session" },
    getContextUsage: () => ({ tokens: 66_160, contextWindow: 253_792, percent: 26.1 }),
    modelRuntime: { isUsingSubscription: () => false },
  };
  const footer = new FooterComponent(session, {
    getGitBranch: () => "main", getExtensionStatuses: () => statuses, getAvailableProviderCount: () => 2,
  });
  footer.setAutoCompactEnabled(true);
  const wide = footer.render(1000).map(stripTerminalSequences);
  assert.equal(wide.length, 1, "no hard break for path, model, or extension statuses");
  assert.ok(wide[0].indexOf("Cold 160") < wide[0].indexOf("40°C"));
  assert.ok(wide[0].includes("(auto) • VRAM 60,000 · RAM 0"));
  assert.ok(wide[0].includes("Cold 160 • 40°C · 38°C"));
  assert.ok(wide[0].includes("25% · 28.5 / 31.9 GiB • ↑4.8M"));
  assert.doesNotMatch(wide[0], /Cache [≈=]|\btok\b|\bGPU\b/);
  assert.equal(wide[0].match(/\bVRAM\b/g)?.length, 1);
  assert.ok(wide[0].includes("Disk 58,000"));
  assert.ok(wide[0].indexOf("Disk 58,000") < wide[0].indexOf("40°C"));
  assert.ok(wide[0].includes("↑4.8M ↓845k R109M CH96.7%"));
  assert.ok(wide[0].indexOf("40°C") < wide[0].indexOf("↑4.8M"));
  assert.ok(wide[0].indexOf("↑4.8M") < wide[0].indexOf("(qwen-r9700)"));
  assert.ok(wide[0].includes("66,160 / 253,792 (26.1%)"));
  assert.ok(wide[0].includes("(qwen-r9700) radiance-model-with-a-long-name"));
  for (const width of [40, 80, 110, 180]) {
    const lines = footer.render(width);
    assert.ok(lines.every(line => visibleWidth(line) <= width));
    assert.equal(lines.map(stripTerminalSequences).join("").replace(/\s/g, ""), wide[0].replace(/\s/g, ""),
      `no missing or duplicated characters at width ${width}`);
  }
});

async function compactionFooterFixture(t, { inMemory = false, postCompaction = false } = {}) {
  const mod = path => import(pathToFileURL(join(root, path)));
  const { FooterComponent } = await mod("pi-coding-agent/dist/modes/interactive/components/footer.js");
  const { initTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { stripTerminalSequences } = await mod("pi-tui/dist/index.js");
  initTheme("dark", false);
  const sessionId = randomUUID();
  const entries = [{ type: "message", message: { role: "assistant", usage: {
    input: 217_000, output: 366, cacheRead: 0, cacheWrite: 0, cost: { total: 0 },
  } } }];
  if (postCompaction) entries.push({ type: "compaction", id: randomUUID() });
  const originalEntries = structuredClone(entries);
  const contextUsage = { tokens: postCompaction ? null : 217_366, contextWindow: 253_792,
    percent: postCompaction ? null : 217_366 / 253_792 * 100 };
  const model = { id: "synthetic-compaction-model", provider: "qwen-r9700", api: "openai-completions", contextWindow: 253_792 };
  const sessionManager = {
    getSessionFile: () => inMemory ? undefined : `/synthetic-${sessionId}.jsonl`,
    getSessionId: () => sessionId,
    getEntries: () => entries,
    getCwd: () => "/synthetic/project",
    getSessionName: () => "Synthetic compaction",
  };
  const session = { state: { model }, sessionManager, getContextUsage: () => contextUsage,
    modelRuntime: { isUsingSubscription: () => false } };
  const footer = new FooterComponent(session, {
    getGitBranch: () => undefined, getExtensionStatuses: () => new Map(), getAvailableProviderCount: () => 1,
  });
  const abort = new AbortController();
  const ctx = { mode: "tui", model, sessionManager, getContextUsage: session.getContextUsage,
    ui: { setWidget() {}, setStatus() {}, setWorkingMessage() {} } };
  const progress = startCompactionProgress(ctx, { signal: abort.signal,
    scheduler: { read: () => ({ available: false, configured: false }) },
    schedule: () => 1, unschedule() {},
  });
  t.after(() => clearCompactionProgress(ctx));
  return { session, entries, originalEntries, progress, abort, ctx, contextUsage,
    rendered: () => footer.render(1000).map(stripTerminalSequences).join(""),
  };
}

const installedFooterAvailable = existsSync(join(root, "pi-coding-agent/dist/modes/interactive/components/footer.js"));

test("installed footer preserves minimal Cold padding and safely wraps compact sensor values", {
  skip: !installedFooterAvailable,
}, async () => {
  const mod = path => import(pathToFileURL(join(root, path)));
  const { FooterComponent } = await mod("pi-coding-agent/dist/modes/interactive/components/footer.js");
  const { initTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { stripTerminalSequences, visibleWidth } = await mod("pi-tui/dist/index.js");
  initTheme("dark", false);
  const statuses = new Map([["other", " \r\nalpha  \tbeta\rgamma\n  delta \t "]]);
  const session = {
    state: { model: { id: "synthetic-stable-footer", provider: "qwen-r9700", contextWindow: 253_792 } },
    sessionManager: {
      getEntries: () => [{ type: "message", message: { role: "assistant", usage: {
        input: 217_000, output: 366, cacheRead: 0, cacheWrite: 0, cost: { total: 0 },
      } } }],
      getSessionFile: () => "/synthetic-stable-footer.jsonl",
      getCwd: () => "/synthetic/project", getSessionName: () => "Synthetic session",
    },
    getContextUsage: () => ({ tokens: 66_160, contextWindow: 253_792, percent: 26.1 }),
    modelRuntime: { isUsingSubscription: () => false },
  };
  const footer = new FooterComponent(session, {
    getGitBranch: () => "main", getExtensionStatuses: () => statuses, getAvailableProviderCount: () => 2,
  });
  const offsets = text => {
    const sensorSeparator = text.indexOf(" • ", text.indexOf("Cold"));
    return { separators: [...text.slice(0, sensorSeparator + 3).matchAll(/[•·]/g)].map(match => match.index),
      firstSensor: sensorSeparator + 3 };
  };
  let expectedOffsets;
  for (const cold of [0, 1, 9, 10, 70, 99, 100, 999]) {
    for (const junction_millicelsius of [99_000, 99_500, 100_000, 100_500]) {
      for (const fan_percent of [9, 74, 100]) {
        for (const usedGiB of [9.9, 10, 31.8]) {
          const label = `Cold ${cold}, ${junction_millicelsius / 1000}°C, fan ${fan_percent}%, ${usedGiB} GiB`;
          const cacheStatus = formatCacheBreakdown({ gpu: 60_000, ram: 0, diskSaved: 58_000, cold });
          const temperatureStatus = formatTemperatureStatus({ junction_millicelsius,
            edge_millicelsius: 60_000, fan_percent,
            vram_used_bytes: Math.round(usedGiB * 1024 ** 3), vram_total_bytes: 32 * 1024 ** 3,
          });
          statuses.set("qwen-cache-residency", cacheStatus);
          statuses.set("qwen-gpu-temperature", temperatureStatus);
          const wideLines = footer.render(1000).map(stripTerminalSequences);
          assert.equal(wideLines.length, 1);
          const wide = wideLines[0];
          assert.ok(wide.includes(cacheStatus), `cache padding survives native sanitization: ${label}`);
          assert.ok(wide.includes(temperatureStatus), `compact sensor values survive native sanitization: ${label}`);
          assert.doesNotMatch(temperatureStatus, /\u00a0| {2}/, `sensor values have no reserved blank columns: ${label}`);
          const renderedCold = wide.match(/Cold ([\u00a0]*)(\d+)/);
          assert.equal(Number(renderedCold[2]), cold);
          assert.ok(renderedCold[1].length <= 2, "Cold reserves at most two blank columns");
          assert.ok(wide.endsWith("alpha beta gamma delta"), "ordinary statuses still normalize whitespace");
          assert.doesNotMatch(wide, /[\r\n\t]/, "status control characters cannot alter the footer layout");
          expectedOffsets ??= offsets(wide);
          assert.deepEqual(offsets(wide), expectedOffsets, `Cold changes keep the first sensor column stable: ${label}`);
          for (const width of [20, 40, 80, 110, 180]) {
            const lines = footer.render(width);
            assert.ok(lines.every(line => visibleWidth(line) <= width), `bounded wrapping: ${label}, width ${width}`);
            assert.equal(lines.map(stripTerminalSequences).join("").replace(/ /g, ""), wide.replace(/ /g, ""),
              `all text and reserved nonbreaking padding survive wrapping: ${label}, width ${width}`);
          }
        }
      }
    }
  }
  const example = formatTemperatureStatus({ junction_millicelsius: 91_000,
    edge_millicelsius: 41_000, fan_percent: 29,
    vram_used_bytes: Math.round(31.8 * 1024 ** 3), vram_total_bytes: Math.round(31.9 * 1024 ** 3) });
  assert.equal(example, "91°C · 41°C · 29% · 31.8 / 31.9 GiB");
  statuses.set("qwen-cache-residency", formatCacheBreakdown({ gpu: 60_000, ram: 0, diskSaved: 58_000, cold: 68 }));
  statuses.set("qwen-gpu-temperature", example);
  assert.ok(footer.render(1000).map(stripTerminalSequences).join("").includes(`Cold \u00a068 • ${example}`),
    "only the Cold counter retains a single reserved blank in the reported example");
});

test("installed footer follows live checkpoint input and output without changing transcript usage", {
  skip: !installedFooterAvailable,
}, async (t) => {
  const f = await compactionFooterFixture(t);
  assert.match(f.rendered(), /217,366 \/ 253,792 \(85\.6%\)/);
  f.progress.update({ phase: "submit", inputTokens: 218_328, outputTokenLimit: 12_288 });
  assert.match(f.rendered(), /218,328 \/ 253,792 \(86\.0%\)/);
  f.progress.update({ phase: "generate", outputTokens: 9_162, cacheRead: 200_000 });
  assert.match(f.rendered(), /227,490 \/ 253,792 \(89\.6%\)/);
  f.progress.update({ outputTokens: 9_500 });
  assert.match(f.rendered(), /227,828 \/ 253,792 \(89\.8%\)/);
  assert.equal(f.session.getContextUsage().tokens, 217_366, "compaction only overrides the footer display");
  assert.deepEqual(f.entries, f.originalEntries, "checkpoint counters do not change persisted message usage");
  assert.match(f.rendered(), /↑217k ↓366/, "cumulative transcript totals remain unchanged");
});

function requestProgressClock(t) {
  let now = 1000, nextTimer = 0;
  const timers = new Map();
  t.mock.method(globalThis.performance, "now", () => now);
  t.mock.method(globalThis, "setInterval", (callback) => {
    const timer = ++nextTimer;
    timers.set(timer, callback);
    return timer;
  });
  t.mock.method(globalThis, "clearInterval", (timer) => timers.delete(timer));
  return { advance(ms = 100) { now += ms; for (const callback of [...timers.values()]) callback(); } };
}

async function requestFooterFixture(t, options) {
  const f = await compactionFooterFixture(t, { ...options, postCompaction: true });
  const handlers = new Map();
  let observation = { available: true, requestPhase: {
    request_id: "a".repeat(64), phase: "generate", input_tokens: 217_000,
    computed_tokens: 217_366, first_token_ms: 1000,
  } };
  const scheduler = { start() {}, bind() {}, clear() {}, stop() {}, read: () => observation };
  installRequestProgress({ on: (name, handler) => handlers.set(name, handler), registerCommand() {} }, { scheduler });
  const emit = (name, event = {}) => handlers.get(name)?.(event, f.ctx);
  t.after(() => emit("session_shutdown"));
  return { ...f, emit,
    publish(inputTokens, requestId = "b".repeat(64)) {
      observation = { available: true, request: { state: "running", input_tokens: inputTokens,
        computed_tokens: 32_000 }, phaseObservedAt: Date.now(), requestPhase: {
        request_id: requestId, phase: "prefill", input_tokens: inputTokens, computed_tokens: 32_000,
        cached_tokens: 30_000, first_token_ms: null, elapsed_ms: 100, phase_elapsed_ms: 100,
        timings_ms: { prefill: 100 },
      } };
    },
  };
}

test("installed footer shows the new request's exact prompt during post-compaction prefill before streamed usage", {
  skip: !installedFooterAvailable,
}, async (t) => {
  const clock = requestProgressClock(t);
  const f = await requestFooterFixture(t);
  f.emit("before_provider_request", { payload: { stream: true } });
  clock.advance();
  assert.match(f.rendered(), /\? \/ 253,792 \(\?%\)/, "the previous request is excluded at the request boundary");
  f.publish(43_521);
  clock.advance();
  assert.match(f.rendered(), /43,521 \/ 253,792 \(17\.1%\)/,
    "the footer displays the full prompt, including cached input, before any message_update");
  assert.equal(f.session.getContextUsage().tokens, null, "prefill only overrides the displayed count");
  assert.deepEqual(f.entries, f.originalEntries, "prefill telemetry does not create or mutate transcript usage");
  assert.match(f.rendered(), /↑217k ↓366/);
  f.contextUsage.tokens = 43_526;
  f.contextUsage.percent = 43_526 / 253_792 * 100;
  f.emit("message_update", { assistantMessageEvent: { type: "usage_update", partial: {
    usage: { input: 13_521, cacheRead: 30_000, cacheWrite: 0, output: 5 },
  } } });
  assert.match(f.rendered(), /43,526 \/ 253,792/, "measured streamed usage replaces the prefill override");
});

test("installed footer isolates ordinary prefill counts and clears them when a request ends or retries", {
  skip: !installedFooterAvailable,
}, async (t) => {
  const clock = requestProgressClock(t);
  const first = await requestFooterFixture(t);
  const second = await requestFooterFixture(t, { inMemory: true });
  first.emit("before_provider_request", { payload: { stream: true } });
  second.emit("before_provider_request", { payload: { stream: true } });
  first.publish(43_521);
  clock.advance();
  assert.match(first.rendered(), /43,521 \/ 253,792/);
  assert.match(second.rendered(), /\? \/ 253,792/, "another session cannot inherit this prompt count");
  second.publish(55_000, "c".repeat(64));
  clock.advance();
  assert.match(second.rendered(), /55,000 \/ 253,792/);
  assert.match(first.rendered(), /43,521 \/ 253,792/);
  first.emit("message_end", { message: { role: "assistant", usage: { output: 0 } } });
  assert.match(first.rendered(), /\? \/ 253,792/, "an ended request cannot retain the prefill override");
  second.emit("before_provider_request", { payload: { stream: true } });
  clock.advance();
  assert.match(second.rendered(), /\? \/ 253,792/, "a retry excludes the previous request's telemetry");
  second.publish(56_000, "d".repeat(64));
  clock.advance();
  assert.match(second.rendered(), /56,000 \/ 253,792/);
  second.emit("agent_end");
  assert.match(second.rendered(), /\? \/ 253,792/, "agent completion clears the current request's override");
  assert.deepEqual(first.entries, first.originalEntries);
  assert.deepEqual(second.entries, second.originalEntries);
});

test("installed footer isolates simultaneous compaction counts by session file and in-memory session ID", {
  skip: !installedFooterAvailable,
}, async (t) => {
  const first = await compactionFooterFixture(t);
  const second = await compactionFooterFixture(t, { inMemory: true });
  first.progress.update({ phase: "generate", inputTokens: 218_328, outputTokens: 9_162 });
  assert.match(first.rendered(), /227,490 \/ 253,792/);
  assert.match(second.rendered(), /217,366 \/ 253,792/, "another session cannot inherit an active checkpoint");
  second.progress.update({ phase: "wait", inputTokens: 40_000 });
  assert.match(second.rendered(), /40,000 \/ 253,792/);
  second.progress.update({ phase: "generate", outputTokens: 200 });
  assert.match(second.rendered(), /40,200 \/ 253,792/);
  assert.match(first.rendered(), /227,490 \/ 253,792/, "another active checkpoint cannot replace the first");
});

test("installed footer restores transcript context after validation, failure, cancellation or checkpoint reuse", {
  skip: !installedFooterAvailable,
}, async (t) => {
  for (const outcome of ["validate", "failed", "cancelled", "reused", "appended"]) {
    await t.test(outcome, async (t) => {
      const f = await compactionFooterFixture(t);
      f.progress.update({ phase: "generate", inputTokens: 218_328, outputTokens: 9_162 });
      assert.match(f.rendered(), /227,490 \/ 253,792/);
      if (outcome === "validate") f.progress.update({ phase: "validate" });
      else if (outcome === "cancelled") f.abort.abort();
      else if (outcome === "failed") await f.progress.finish("failed");
      else if (outcome === "reused") f.progress.update({ reusedCheckpoint: true });
      else f.progress.markAppended();
      assert.match(f.rendered(), /217,366 \/ 253,792 \(85\.6%\)/);
      assert.deepEqual(f.entries, f.originalEntries);
    });
  }
});
