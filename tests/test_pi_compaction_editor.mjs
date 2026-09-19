// Synthetic editor state only. No live terminals, private drafts or model calls.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const available = existsSync(join(root, "pi-coding-agent/dist/modes/interactive/interactive-mode.js"));

async function fixture() {
  const mod = (path) => import(pathToFileURL(join(root, path)));
  const { InteractiveMode } = await mod("pi-coding-agent/dist/modes/interactive/interactive-mode.js");
  const { CustomEditor } = await mod("pi-coding-agent/dist/modes/interactive/components/custom-editor.js");
  const { KeybindingsManager } = await mod("pi-coding-agent/dist/core/keybindings.js");
  const { IdleStatus } = await mod("pi-coding-agent/dist/modes/interactive/components/status-indicator.js");
  const { initTheme, getEditorTheme, getMarkdownTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { Container, Text, TuiMainScreen } = await mod("pi-tui/dist/index.js");
  initTheme("dark", false);
  const writes = [], submitted = [];
  const terminal = { columns: 100, rows: 24, write: (text) => writes.push(text), hideCursor() {}, showCursor() {}, stop() {} };
  const ui = new TuiMainScreen(terminal, false);
  const editor = new CustomEditor(ui, getEditorTheme(), new KeybindingsManager());
  const session = { isCompacting: true, isStreaming: false,
    settingsManager: { getShowTerminalProgress: () => false, getShowCacheMissNotices: () => false },
    sessionManager: { buildContextEntries: () => [] }, abortCompaction() {}, prompt: async (text) => submitted.push(text) };
  const mode = Object.assign(Object.create(InteractiveMode.prototype), {
    isInitialized: true, options: { tuiMode: "regular" }, ui, footer: { invalidate() {} }, runtimeHost: { session },
    editor, defaultEditor: editor, editorContainer: new Container(), statusContainer: new Container(),
    chatContainer: new Container(), pendingTools: new Map(), idleStatus: new IdleStatus(), compactionQueuedMessages: [],
    getMarkdownThemeWithSettings: getMarkdownTheme, updatePendingMessagesDisplay() {}, showError() {}, showStatus() {},
  });
  editor.onAction("app.clear", () => mode.clearEditor());
  editor.onSubmit = (text) => submitted.push(text);
  mode.editorContainer.addChild(editor);
  ui.addChild(mode.chatContainer); ui.addChild(mode.statusContainer); ui.addChild(mode.editorContainer); ui.setFocus(editor);
  mode.chatContainer.addChild(new Text("Synthetic old transcript\n".repeat(200), 0, 0));
  return { mode, ui, editor, submitted, writes,
    close() { mode.releaseCompactionDraftGuard?.(); mode.clearStatusIndicator(); ui.stop({ preserveScreen: true }); } };
}

function draftState(editor) {
  return { text: editor.getText(), expanded: editor.getExpandedText(), state: structuredClone(editor.state),
    pastes: [...editor.pastes], undo: structuredClone(editor.undoStack) };
}

test("compaction completion protects the latest multiline draft from deferred clears", { skip: !available }, async (t) => {
  for (const reason of ["manual", "threshold", "overflow"]) {
    for (const outcome of ["complete", "cancelled", "failed"]) {
      await t.test(`${reason}: ${outcome}`, async () => {
        const { mode, ui, editor, submitted, close } = await fixture();
        try {
          editor.setText("Unsent draft before compaction");
          await mode.handleEvent({ type: "compaction_start", reason });
          mode.activeStatusIndicator.setMessage("Synthetic compaction phase\n".repeat(18));
          ui.handleTerminalInput(`\x1b[200~\n${"Synthetic pasted line\n".repeat(15)}Typed during compaction\x1b[201~`);
          ui.handleTerminalInput("\x1b[D");
          ui.renderNow();
          const before = draftState(editor);
          assert.ok(before.pastes.length, "exercise expanded paste data, not just plain text");
          mode.session.isCompacting = false;
          await mode.handleEvent({ type: "compaction_end", reason,
            ...(outcome === "cancelled" ? { aborted: true } : outcome === "failed" ? { errorMessage: "Synthetic failure" } :
              { result: { summary: "Synthetic checkpoint", tokensBefore: 240000 } }) });
          // A delayed UI caller finishing the old operation must not discard
          // what the user typed while that operation was awaiting completion.
          await new Promise((resolve) => setImmediate(resolve));
          editor.setText("");
          ui.renderNow();
          assert.deepEqual(draftState(editor), before);
          assert.deepEqual(submitted, [], "preserving a draft must never send it");
          assert.equal(ui.inputListeners.size, 1);
          ui.handleTerminalInput("continued");
          await new Promise((resolve) => setImmediate(resolve));
          const continued = draftState(editor);
          editor.setText("");
          assert.deepEqual(draftState(editor), continued, "continued typing must not expose the draft to late clears");
          ui.handleTerminalInput("\r");
          await Promise.resolve();
          assert.deepEqual(submitted, [continued.expanded.trim()]);
          assert.equal(editor.getText(), "");
          assert.equal(ui.inputListeners.size, 0, "submitting the draft releases protection");
        } finally { close(); }
      });
    }
  }
});

test("compaction draft protection allows explicit clearing and ends when the editor changes", { skip: !available }, async () => {
  const { mode, ui, editor, close } = await fixture();
  try {
    editor.setText("Synthetic draft");
    await mode.handleEvent({ type: "compaction_end", reason: "threshold", aborted: true });
    await mode.handleEvent({ type: "compaction_end", reason: "threshold", aborted: true });
    assert.equal(ui.inputListeners.size, 1, "repeated completion events cannot leak listeners");
    ui.handleTerminalInput("\x03");
    await Promise.resolve();
    assert.equal(editor.getText(), "", "Ctrl+C still deliberately clears the draft");
    assert.equal(ui.inputListeners.size, 0);
    editor.setText("New synthetic draft");
    await mode.handleEvent({ type: "compaction_end", reason: "threshold", aborted: true });
    mode.setCustomEditorComponent(undefined);
    assert.equal(ui.inputListeners.size, 0);
    editor.setText("");
    assert.equal(editor.getText(), "", "session/editor replacement must not inherit an old guard");
    editor.setText("Synthetic external-editor draft");
    await mode.handleEvent({ type: "compaction_end", reason: "threshold", aborted: true });
    mode.settingsManager.getExternalEditorCommand = () => { throw new Error("Synthetic stop before launching an editor"); };
    await assert.rejects(mode.handleOpenExternalEditor(), /Synthetic stop/);
    assert.equal(ui.inputListeners.size, 0, "an explicit external edit authorizes replacing or clearing the draft");
    editor.setText("");
    assert.equal(editor.getText(), "");
  } finally { close(); }
});

test("a message submitted during compaction is queued while the next unsent draft is preserved", { skip: !available }, async () => {
  const { mode, ui, editor, submitted, close } = await fixture();
  try {
    mode.setupEditorSubmitHandler();
    mode.isExtensionCommand = () => false;
    await mode.handleEvent({ type: "compaction_start", reason: "threshold" });
    editor.setText("Synthetic queued message");
    ui.handleTerminalInput("\r");
    assert.equal(mode.compactionQueuedMessages.length, 1);
    assert.equal(editor.getText(), "");
    ui.handleTerminalInput("Synthetic next draft");
    mode.session.isCompacting = false;
    await mode.handleEvent({ type: "compaction_end", reason: "threshold",
      result: { summary: "Synthetic checkpoint", tokensBefore: 240000 } });
    await new Promise((resolve) => setImmediate(resolve));
    editor.setText("");
    assert.deepEqual(submitted, ["Synthetic queued message"]);
    assert.equal(editor.getText(), "Synthetic next draft");
  } finally { close(); }
});
