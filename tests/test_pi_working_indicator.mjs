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
const { InteractiveMode } = await import(interactiveModeUrl);
const { initTheme } = await import(themeUrl);
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
