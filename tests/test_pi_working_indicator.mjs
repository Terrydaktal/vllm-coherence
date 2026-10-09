import assert from "node:assert/strict";
import test from "node:test";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const root = process.env.QWEN_TEST_PI_ROOT ??
	join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const interactiveModeUrl = pathToFileURL(
	join(root, "pi-coding-agent/dist/modes/interactive/interactive-mode.js"),
).href;
const themeUrl = pathToFileURL(
	join(root, "pi-coding-agent/dist/modes/interactive/theme/theme.js"),
).href;
const statusIndicatorUrl = pathToFileURL(
	join(root, "pi-coding-agent/dist/modes/interactive/components/status-indicator.js"),
).href;
const { InteractiveMode } = await import(interactiveModeUrl);
const { initTheme } = await import(themeUrl);
const { RetryStatusIndicator } = await import(statusIndicatorUrl);
initTheme("dark");

function modeFixture(InteractiveMode) {
	const mode = Object.create(InteractiveMode.prototype);
	mode.workingMessage = undefined;
	mode.workingVisible = true;
	mode.workingIndicatorOptions = undefined;
	mode.defaultWorkingMessage = "Working...";
	mode.activeStatusIndicator = undefined;
	mode.runtimeHost = { session: { isStreaming: true } };
	mode.options = { tuiMode: "fullscreen" };
	mode.footer = { invalidate() {} };
	mode.ui = { requestRender() {} };
	mode.statusContainer = {
		child: undefined,
		clear() { this.child = undefined; },
		addChild(child) { this.child = child; },
	};
	return mode;
}

test("provider progress restores the working spinner after inter-turn compaction", () => {
	const mode = modeFixture(InteractiveMode);
	const extensionUi = mode.createExtensionUIContext();

	// compaction_end has removed its indicator, while the surrounding agent tool
	// loop is still streaming and begins another provider request.
	extensionUi.setWorkingMessage("Qwen KV lookup/prefill (~150.0K ctx) • 0s");

	assert.equal(mode.activeStatusIndicator?.kind, "working");
	assert.equal(mode.statusContainer.child, mode.activeStatusIndicator);
	assert.equal(mode.workingMessage, "Qwen KV lookup/prefill (~150.0K ctx) • 0s");
	mode.clearStatusIndicator();
});

test("working progress does not replace another live status or revive an idle spinner", () => {
	const mode = modeFixture(InteractiveMode);
	const extensionUi = mode.createExtensionUIContext();
	const compaction = {
		kind: "compaction",
		message: undefined,
		setMessage(message) { this.message = message; },
	};
	mode.activeStatusIndicator = compaction;

	extensionUi.setWorkingMessage("Radiance compaction: Generate checkpoint • 12s");
	assert.equal(mode.activeStatusIndicator, compaction);
	assert.equal(compaction.message, "Radiance compaction: Generate checkpoint • 12s");

	mode.activeStatusIndicator = undefined;
	mode.session.isStreaming = false;
	extensionUi.setWorkingMessage("stale progress");
	assert.equal(mode.activeStatusIndicator, undefined);

	mode.session.isStreaming = true;
	extensionUi.setWorkingMessage(undefined);
	assert.equal(mode.activeStatusIndicator, undefined);
});

test("native retry countdown stays on one content line at every tick", (t) => {
	t.mock.timers.enable({ apis: ["setInterval"] });
	const indicator = new RetryStatusIndicator({ requestRender() {} }, 1, 3, 3000);
	t.after(() => indicator.dispose());
	for (const seconds of [3, 2, 1]) {
		assert.match(indicator.message, new RegExp(`Retrying \\(1/3\\) in ${seconds}s`));
		assert.equal(indicator.message.split("\n").length, 1);
		assert.equal(indicator.render(200).length, 2);
		t.mock.timers.tick(1000);
	}
	assert.match(indicator.message, /Retrying \(1\/3\) in 0s/);
	assert.equal(indicator.countdown, undefined);
});

test("native retry preserves Escape cancellation and releases its single status line", async (t) => {
	t.mock.timers.enable({ apis: ["setInterval"] });
	const mode = modeFixture(InteractiveMode);
	mode.isInitialized = true;
	const priorEscape = () => {};
	mode.defaultEditor = { onEscape: priorEscape };
	const errors = [];
	mode.showError = (message) => errors.push(message);
	let cancellations = 0;
	mode.session.abortRetry = () => { cancellations += 1; };
	t.after(() => mode.clearStatusIndicator());

	await mode.handleEvent({ type: "auto_retry_start", attempt: 1, maxAttempts: 3, delayMs: 3000 });
	const retry = mode.activeStatusIndicator;
	assert.equal(retry?.kind, "retry");
	const extensionUi = mode.createExtensionUIContext();
	extensionUi.setWorkingMessage("stale previous response progress");
	assert.notEqual(retry.message, "stale previous response progress");
	t.mock.timers.tick(1000);
	assert.match(retry.message, /Retrying \(1\/3\) in 2s/);

	mode.defaultEditor.onEscape();
	assert.equal(cancellations, 1);
	await mode.handleEvent({ type: "auto_retry_end", success: false, attempt: 1, finalError: "Retry cancelled" });
	assert.equal(mode.defaultEditor.onEscape, priorEscape);
	assert.equal(mode.activeStatusIndicator, undefined);
	assert.equal(retry.countdown, undefined);
	assert.equal(retry.intervalId, null);
	assert.deepEqual(errors, ["Retry failed after 1 attempts: Retry cancelled"]);

	extensionUi.setWorkingMessage("Qwen reasoning: 170 tok • 53.9 t/s");
	assert.equal(mode.activeStatusIndicator?.kind, "working");
	assert.equal(mode.activeStatusIndicator.render(200).length, 2);
});

test("successful native retry end clears countdown without reporting an error", async (t) => {
	t.mock.timers.enable({ apis: ["setInterval"] });
	const mode = modeFixture(InteractiveMode);
	mode.isInitialized = true;
	mode.defaultEditor = { onEscape() {} };
	mode.showError = () => assert.fail("successful retry must not report a failure");
	t.after(() => mode.clearStatusIndicator());
	await mode.handleEvent({ type: "auto_retry_start", attempt: 2, maxAttempts: 3, delayMs: 5000 });
	const retry = mode.activeStatusIndicator;
	await mode.handleEvent({ type: "auto_retry_end", success: true, attempt: 2 });
	assert.equal(mode.activeStatusIndicator, undefined);
	assert.equal(retry.countdown, undefined);
	assert.equal(retry.intervalId, null);
});
