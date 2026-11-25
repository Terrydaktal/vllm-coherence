import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";

const root = process.env.QWEN_TEST_PI_ROOT ??
  join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const installed = existsSync(join(root, "pi-tui/dist/index.js"));
const renderChunks = Symbol.for("pi.tui.renderChunks");
const modules = async () => {
  const load = (name) => import(pathToFileURL(join(root, name)));
  const { initTheme, getMarkdownTheme } = await load(
    "pi-coding-agent/dist/modes/interactive/theme/theme.js",
  );
  initTheme("dark", false);
  const tui = await load("pi-tui/dist/index.js");
  const { AssistantMessageComponent } = await load(
    "pi-coding-agent/dist/modes/interactive/components/assistant-message.js",
  );
  return { ...tui, markdownTheme: getMarkdownTheme(), AssistantMessageComponent };
};

const line = (index) => "Step " + index + " checks the **input** and records the result in the cache.";
const thinking = (text) => ({ role: "assistant", content: [{ type: "thinking", thinking: text }] });

test("streaming Markdown matches Pi's canonical render at syntax transitions", { skip: !installed }, async () => {
  const { Markdown, markdownTheme } = await modules();
  const style = { color: markdownTheme.italic, italic: true };
  const texts = [
    ["A heading on", " two lines\nsecond line", "\n---\n", "\n**after**"],
    ["Stable paragraph.\n\n| a | b |\n", "|---|", "---|\n| x | y |\n", "| p | q |\n", "\nA second paragraph.\n"],
    ["Prose paragraph.\n\n- item one\n", "\n  continued item\n", "  - nested **item**\n", "- second\n", "\nFollow up.\n"],
    ["Opening text.\n\n\`\`\`js\nlet val", "ue = 0;\n\`", "\`", "\`\n", "\nA paragraph.\n"],
    ["See [existing][ref].\n\nText.\n\nAnother paragraph.\n", "\nFourth paragraph.\n\n", "[ref]: https://example.com\n"],
    ["A *long\n", "spanning emphasis*.\n\n", "> quote line\n", "> second line\n\n", "Unicode e\u0301 界 🤝.\n"],
    ["Long\tword with tabs.\r\n\r\nNext text.\r", "\n", "\nNew block.\n"],
    ["An equation with $x^", "2$ and inline code \`x", " + y\`.\n\n", "$$\nx+1\n", "$$\n\nNew paragraph."],
  ];
  for (const parts of texts) {
    const stream = new Markdown("", 1, 0, markdownTheme, style, { streamingAppend: true });
    let text = "";
    for (const part of parts) {
      text += part;
      const baseline = new Markdown(text, 1, 0, markdownTheme, style);
      stream.setStreamingText(text);
      for (const width of [110, 41, 110]) {
        assert.deepEqual(stream[renderChunks](width).flat(), baseline.render(width), JSON.stringify(text));
      }
    }
    stream.setStreamingText("Replacement after compaction.");
    assert.deepEqual(stream.render(110), new Markdown("Replacement after compaction.", 1, 0, markdownTheme, style).render(110));
  }
});

test("one growing long paragraph reuses displayed rows without truncating it", { skip: !installed }, async () => {
  const { Markdown, markdownTheme } = await modules();
  const stream = new Markdown("", 1, 0, markdownTheme, undefined, { streamingAppend: true });
  const text = Array.from({ length: 1400 }, (_, i) => line(i)).join("\n");
  stream.setStreamingText(text);
  const first = stream[renderChunks](110);
  const extended = text + "\n" + line(1400);
  stream.setStreamingText(extended);
  const second = stream[renderChunks](110);
  const unchangedRows = second.filter((rows) => Object.isFrozen(rows) && first.includes(rows))
    .reduce((count, rows) => count + rows.length, 0);
  assert.ok(unchangedRows >= 1272, String(unchangedRows));
  const baseline = new Markdown(extended, 1, 0, markdownTheme);
  assert.deepEqual(second.flat(), baseline.render(110));
  // A new Setext underline may legitimately change the whole open paragraph.
  stream.setStreamingText(extended + "\n---\n");
  baseline.setText(extended + "\n---\n");
  assert.deepEqual(stream.render(110), baseline.render(110));
});

test("assistant streaming stays in normal scrollback with stock terminal output", { skip: !installed }, async () => {
  const { AssistantMessageComponent, Container, Text, TuiMainScreen } = await modules();
  function frame(plain) {
    const writes = [];
    const terminal = { columns: 106, rows: 19, write: (text) => writes.push(text), hideCursor() {}, showCursor() {} };
    const ui = new TuiMainScreen(terminal);
    if (plain) {
      // Use the legacy frame path while preserving its ordinary child output.
      ui.render = (width) => Container.prototype.render.call(ui, width);
    }
    const chat = new Container();
    const assistant = new AssistantMessageComponent();
    const editor = new Text("Editor", 1, 0);
    chat.addChild(assistant);
    ui.addChild(new Text("Synthetic session", 1, 0));
    ui.addChild(chat);
    ui.addChild(editor);
    return { ui, assistant, terminal, writes, editor };
  }
  const incremental = frame(false);
  const baseline = frame(true);
  let text = Array.from({ length: 240 }, (_, index) => line(index)).join("\n");
  const first = thinking(text);
  for (const view of [incremental, baseline]) {
    view.assistant.updateContent(first, true);
    view.ui.renderNow();
  }
  assert.ok(incremental.writes.join("").includes("Step 0 checks"));
  let shared = 0;
  for (let index = 240; index < 246; index++) {
    const prior = incremental.assistant[renderChunks](106);
    text += "\n" + line(index);
    const message = thinking(text);
    for (const view of [incremental, baseline]) {
      view.writes.length = 0;
      view.assistant.updateContent(message, true);
      view.editor.setText("Editor " + index);
      view.ui.renderNow();
    }
    shared += incremental.assistant[renderChunks](106).filter((rows) => prior.includes(rows)).length;
    assert.equal(incremental.assistant.lastMessage, message);
    assert.deepEqual(incremental.ui.previousLines, baseline.ui.previousLines);
    assert.deepEqual(incremental.writes, baseline.writes);
    assert.ok(!incremental.writes.join("").includes("\x1b[3J"));
    assert.ok(!incremental.writes.join("").includes("Step 0 checks"));
    assert.ok(incremental.writes.join("").includes("Step " + index + " checks"));
  }
  assert.ok(shared > 0);
  for (const view of [incremental, baseline]) {
    view.writes.length = 0;
    view.assistant.updateContent(thinking(text), false);
    view.terminal.columns = 81;
    view.ui.renderNow();
    view.assistant.setHideThinkingBlock(true);
    view.ui.renderNow();
    view.assistant.setHideThinkingBlock(false);
    view.ui.renderNow();
  }
  assert.deepEqual(incremental.ui.previousLines, baseline.ui.previousLines);
  assert.deepEqual(incremental.writes, baseline.writes);
});
