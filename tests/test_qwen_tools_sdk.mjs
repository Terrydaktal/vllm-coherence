// Synthetic offline SDK integration: /tools must never send an inference request.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { stripVTControlCharacters } from "node:util";

const sdkRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");

// Track effective foreground colors after nested resets and the real Text renderer.
function foregroundCells(ansi) {
  const chunks = [], foreground = [];
  let color = "default", offset = 0;
  const append = (chunk) => {
    chunks.push(chunk);
    foreground.push(...Array(chunk.length).fill(color));
  };
  for (const match of ansi.matchAll(/\x1b\[([\d;]*)m/g)) {
    append(ansi.slice(offset, match.index));
    const codes = match[1].split(";").map(Number);
    for (let index = 0; index < codes.length; index++) {
      const code = codes[index];
      if (code === 0 || code === 39) color = "default";
      else if (code === 38 && codes[index + 1] === 2) {
        color = `rgb:${codes.slice(index + 2, index + 5).join(",")}`;
        index += 4;
      } else if (code === 38 && codes[index + 1] === 5) {
        color = `indexed:${codes[index + 2]}`;
        index += 2;
      } else if ((code >= 30 && code <= 37) || (code >= 90 && code <= 97)) color = `ansi:${code}`;
    }
    offset = match.index + match[0].length;
  }
  append(ansi.slice(offset));
  return { text: chunks.join(""), foreground };
}

test("shared compaction extension registers /tools and lists real SDK builtin and extension tools without inference", {
  skip: !existsSync(join(sdkRoot, "pi-coding-agent/dist/core/sdk.js")), timeout: 20000,
}, async (t) => {
  const mod = (path) => import(pathToFileURL(join(sdkRoot, path)));
  const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
  const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
  const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
  const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
  const { getThemeByName } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { Text } = await mod("pi-tui/dist/components/text.js");
  const theme = getThemeByName("dark");
  assert.ok(theme, "installed Pi dark theme is available");
  const colors = Object.fromEntries(["text", "dim", "success", "error"]
    .map((name) => [name, foregroundCells(theme.fg(name, "x")).foreground[0]]));
  assert.notEqual(colors.text, colors.dim, "tool names and descriptions use distinct colors");
  const agentDir = await mkdtemp(join(tmpdir(), "qwen-tools-sdk-"));
  const previous = Object.fromEntries(["PI_CODING_AGENT_DIR", "PI_OFFLINE", "QWEN_RADIANCE_CACHE_ABI"].map((key) => [key, process.env[key]]));
  process.env.PI_CODING_AGENT_DIR = agentDir;
  process.env.PI_OFFLINE = "1";
  delete process.env.QWEN_RADIANCE_CACHE_ABI;
  t.after(async () => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
    await rm(agentDir, { recursive: true, force: true });
  });
  let networkRequests = 0;
  let inferenceRequests = 0;
  t.mock.method(globalThis, "fetch", () => {
    networkRequests++;
    throw new Error("Network forbidden in tool-listing SDK fixture");
  });
  const model = { id: "synthetic-tools", name: "Synthetic tools", provider: "fixture", api: "openai-completions",
    baseUrl: "http://fixture.invalid/v1", reasoning: false, input: ["text"], contextWindow: 32768, maxTokens: 8192,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
    isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "fixture" }, env: {} }),
    streamSimple: () => { inferenceRequests++; throw new Error("Inference forbidden in tool-listing SDK fixture"); } };
  const settingsManager = SettingsManager.inMemory({ defaultTools: ["read", "bash"],
    compaction: { enabled: false }, retry: { enabled: false } }, { projectTrusted: true });
  const resourceLoader = new DefaultResourceLoader({ cwd: agentDir, agentDir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    additionalExtensionPaths: [resolve("integrations/pi/qwen-radiance-compaction.ts"), resolve("integrations/pi/qwen-task-plan.ts"), resolve("integrations/pi/qwen-tool-names.ts")],
    systemPrompt: "Stable synthetic tool-listing prefix." });
  await resourceLoader.reload();
  assert.deepEqual(resourceLoader.getExtensions().errors, []);
  const listingCommands = resourceLoader.getExtensions().extensions.flatMap((extension) => [...extension.commands.keys()])
    .filter((name) => name === "tools");
  assert.deepEqual(listingCommands, ["tools"], "shared wrapper registers /tools once");
  const sessionManager = SessionManager.create(agentDir, join(agentDir, "sessions"));
  const { session } = await createAgentSession({ cwd: agentDir, agentDir, model, modelRuntime, settingsManager,
    resourceLoader, sessionManager });
  t.after(async () => {
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    session.dispose();
  });
  const notices = [];
  await session.bindExtensions({ uiContext: {
    theme,
    notify: (message, level) => notices.push({ message, level }),
    setWidget: () => {}, setStatus: () => {}, setWorkingMessage: () => {},
  } });
  assert.ok(session.getAllTools().some(({ name }) => name === "read"));
  assert.ok(session.getAllTools().some(({ name }) => name === "write"));
  assert.ok(session.getAllTools().some(({ name }) => name === "manage_task_plan"));
  assert.ok(!session.getAllTools().some(({ name }) => name === "qwen_plan"));
  assert.ok(session.getActiveToolNames().includes("manage_task_plan"));
  const normalize = (text) => text.replace(/\s+/g, " ").trim();
  const invoke = async () => {
    notices.length = 0;
    const tools = session.getAllTools();
    const active = session.getActiveToolNames();
    const prompt = session.systemPrompt;
    const entries = structuredClone(sessionManager.getEntries());
    const messages = structuredClone(session.agent.state.messages);
    await session.prompt("/tools");
    const output = stripVTControlCharacters(notices.map(({ message }) => message).join("\n"));
    const rendered = foregroundCells(notices.map(({ message }) =>
      new Text(theme.fg("dim", message), 1, 0).render(120).join("\n")).join("\n"));
    for (const tool of tools) {
      const name = new RegExp(`\\b${tool.name}\\b`);
      const status = new RegExp(`\\b${active.includes(tool.name) ? "enabled" : "disabled"}\\b`, "i");
      assert.ok(output.split("\n").some((line) => name.test(line) && status.test(line)), `${tool.name} has the current activation status`);
      assert.ok(normalize(output).includes(normalize(tool.description)), `${tool.name} has its actual SDK description`);
      const statusText = active.includes(tool.name) ? "enabled" : "disabled";
      const label = `[${statusText}] ${tool.name} — `;
      const start = rendered.text.indexOf(label);
      assert.notEqual(start, -1, `${tool.name} has a rendered status and name`);
      const assertColor = (offset, length, expected, label) => {
        assert.deepEqual(rendered.foreground.slice(start + offset, start + offset + length),
          Array(length).fill(expected), `${tool.name}: ${label}`);
      };
      assertColor(0, 1, colors.dim, "status opening bracket stays dim");
      assertColor(1, statusText.length, colors[statusText === "enabled" ? "success" : "error"],
        `${statusText} status uses the theme's ${statusText === "enabled" ? "green" : "red"}`);
      assertColor(statusText.length + 1, 2, colors.dim, "status closing bracket stays dim");
      assertColor(statusText.length + 3, tool.name.length, colors.text, "tool name uses normal foreground");
      assertColor(label.length - 3, 3, colors.dim, "description separator stays dim");
      assertColor(label.length, 1, colors.dim, "description starts dim after the tool name reset");
    }
    assert.deepEqual(session.getActiveToolNames(), active, "listing does not mutate active tools");
    assert.deepEqual(session.getAllTools(), tools, "listing does not mutate the registry");
    assert.deepEqual(sessionManager.getEntries(), entries, "listing is not saved to the conversation");
    assert.deepEqual(session.agent.state.messages, messages, "listing is not sent to the model context");
    assert.equal(session.systemPrompt, prompt);
    assert.ok(notices.every(({ level }) => level !== "error"));
    assert.equal(networkRequests, 0);
    assert.equal(inferenceRequests, 0);
  };
  await invoke();
  session.setActiveToolsByName(["write"]);
  await invoke();
});
