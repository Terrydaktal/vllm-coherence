// Exercise the installed Pi UI with an in-memory terminal and synthetic metadata.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { startCompactionProgress, clearCompactionProgress } from "../integrations/pi/qwen-radiance-compaction-progress.mjs";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");

test("collapsed compaction summaries retain the exact final duration", {
  skip: !existsSync(join(root, "pi-coding-agent/dist/modes/interactive/interactive-mode.js")),
}, async () => {
  const mod = (path) => import(pathToFileURL(join(root, path)));
  const { InteractiveMode } = await mod("pi-coding-agent/dist/modes/interactive/interactive-mode.js");
  const { CompactionSummaryMessageComponent } = await mod("pi-coding-agent/dist/modes/interactive/components/compaction-summary-message.js");
  const { initTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { Container, stripTerminalSequences: stripAnsi } = await mod("pi-tui/dist/index.js");
  initTheme("dark", false);
  const chatContainer = new Container();
  const component = new CompactionSummaryMessageComponent({
    role: "compactionSummary", summary: "Synthetic checkpoint", tokensBefore: 238332, timestamp: Date.now(),
  });
  chatContainer.addChild(component);
  const mode = Object.assign(Object.create(InteractiveMode.prototype), { chatContainer });
  mode.addCustomEntryToChat({ type: "custom", customType: "qwen-radiance-compaction-timing-v1",
    parentId: "later-message", data: { compactionEntryId: "compacted", elapsedMs: 94321 } });

  assert.match(stripAnsi(chatContainer.render(120).join("\n")),
    /Compacted from 238,332 tokens in 1m 34s \(.*to expand\)/);
  component.setExpanded(true);
  assert.match(stripAnsi(chatContainer.render(120).join("\n")), /Compacted from 238,332 tokens in 1m 34s/);
});

test("native manual and automatic compaction spinners update in place and disappear on every outcome", {
  skip: !existsSync(join(root, "pi-coding-agent/dist/modes/interactive/interactive-mode.js")), timeout: 20000,
}, async (t) => {
  const agentDir = await mkdtemp(join(tmpdir(), "radiance-spinner-"));
  const previous = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = agentDir;
  t.after(async () => {
    if (previous === undefined) delete process.env.PI_CODING_AGENT_DIR; else process.env.PI_CODING_AGENT_DIR = previous;
    await rm(agentDir, { recursive: true, force: true });
  });
  const mod = (path) => import(pathToFileURL(join(root, path)));
  const { InteractiveMode } = await mod("pi-coding-agent/dist/modes/interactive/interactive-mode.js");
  const { IdleStatus, WorkingStatusIndicator } = await mod("pi-coding-agent/dist/modes/interactive/components/status-indicator.js");
  const { initTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { Container, stripTerminalSequences: stripAnsi, visibleWidth } = await mod("pi-tui/dist/index.js");
  initTheme("dark", false);
  for (const reason of ["manual", "threshold"]) {
    for (const outcome of ["complete", "cancelled", "failed", "cleanup_pending"]) {
      let clock = 0, tick, stopped = false, aborted = false;
      const reports = [], notices = [], originalEscape = () => {};
      const mode = Object.assign(Object.create(InteractiveMode.prototype), {
        isInitialized: true, options: { tuiMode: "regular" },
        ui: { requestRender() {}, getClearOnShrink: () => true },
        footer: { invalidate() {} }, runtimeHost: { session: {
          settingsManager: { getShowTerminalProgress: () => false }, abortCompaction: () => { aborted = true; },
        } }, defaultEditor: { onEscape: originalEscape },
        statusContainer: new Container(), chatContainer: new Container(), idleStatus: new IdleStatus(),
        defaultWorkingMessage: "Working...", flushCompactionQueue: async () => {}, rebuildChatFromMessages() {}, addMessageToChat() {},
        showError: (message) => notices.push(message), showStatus: (message) => notices.push(message),
        setExtensionWidget: (_key, value) => assert.equal(value, undefined),
        setExtensionStatus: (_key, value) => assert.equal(value, undefined),
      });
      const ctx = { sessionManager: { getSessionFile: () => join(agentDir, `${reason}-${outcome}.jsonl`) }, ui: mode.createExtensionUIContext() };
      try {
        await mode.handleEvent({ type: "compaction_start", reason });
        const spinner = mode.activeStatusIndicator;
        assert.equal(spinner.kind, "compaction");
        mode.defaultEditor.onEscape();
        assert.equal(aborted, true, "the compaction cancellation control remains connected");
        const scheduler = { start() {}, stop() {}, read: () => ({ available: true,
          lastRequestTiming: { last_round_ms: 44.2, acceptance_rate_3s: 0.6 },
          request: { chat_id: "a".repeat(64), generation: "b".repeat(64), state: "paused",
            computed_tokens: 0, input_tokens: 239265 },
          activeChat: { chat_id: "c".repeat(64), generation: "d".repeat(64) }, workerAvailable: true,
          otherRunningChatId: "c".repeat(64), otherRunningRequest: { chat_id: "c".repeat(64),
            generation: "d".repeat(64), state: "running", computed_tokens: 239266, input_tokens: 239265 } }) };
        const progress = startCompactionProgress(ctx, { now: () => clock, scheduler,
          save: (report) => reports.push(report),
          schedule: (fn) => { tick = fn; return 1; }, unschedule: () => { stopped = true; } });
        progress.update({ phase: "wait", inputTokens: 239265 });
        clock = 65000; tick();
        assert.equal(mode.activeStatusIndicator, spinner, "updates reuse the existing spinner");
        const rendered = mode.statusContainer.render(100).map(stripAnsi);
        assert.match(rendered.join("\n"), /GPU queue: 1m 5s/);
        assert.match(rendered.join("\n"), /another chat c{12} has the GPU and is generating/);
        assert.match(rendered.join("\n"), /Find and load cached context: not observed yet/);
        assert.match(rendered.join("\n"), /Input: 239,265 tok/);
        assert.ok(rendered.every((line) => visibleWidth(line) <= 100), "multiline spinner respects terminal width");
        progress.update({ phase: "generate", outputTokens: 3922, cacheRead: 235664 });
        clock += 1000;
        progress.update({ outputTokens: 3962 });
        const generation = mode.statusContainer.render(100).map(stripAnsi);
        assert.match(generation.join("\n"), /40.0 t\/s, 40.0 t\/s avg/);
        assert.match(generation.join("\n"), /round 44.2 ms/);
        assert.match(generation.join("\n"), /acceptance 60.0%/);
        assert.ok(generation.every((line) => visibleWidth(line) <= 100), "metrics wrap inside the compaction spinner");
        if (["complete", "cleanup_pending"].includes(outcome)) {
          progress.update({ phase: "commit" }); progress.markAppended(); progress.markCommitted();
          progress.update({ phase: "cleanup", removedBytes: 2 ** 30 });
        }
        await progress.finish(outcome);
        assert.equal(mode.workingMessage, undefined);
        assert.equal(stopped, true);
        await mode.handleEvent({ type: "compaction_end", reason,
          ...(outcome === "cancelled" ? { aborted: true } : outcome === "failed" ? { errorMessage: "Synthetic validation failure" } :
            { result: { summary: "Synthetic checkpoint", tokensBefore: 239265 } }) });
        assert.equal(mode.activeStatusIndicator, undefined);
        assert.equal(spinner.intervalId, null, "Pi disposes the animation timer");
        assert.equal(mode.defaultEditor.onEscape, originalEscape);
        assert.ok(mode.statusContainer.render(100).every((line) => !line.trim()), "no pinned breakdown survives completion");
        assert.equal(reports[0].state, outcome);
        mode.showStatusIndicator(new WorkingStatusIndicator(mode.ui, mode.workingMessage ?? mode.defaultWorkingMessage));
        assert.match(stripAnsi(mode.statusContainer.render(100).join("\n")), /Working/);
        assert.doesNotMatch(stripAnsi(mode.statusContainer.render(100).join("\n")), /Radiance compaction/);
      } finally {
        await clearCompactionProgress(ctx);
        mode.clearStatusIndicator();
      }
    }
  }
});
