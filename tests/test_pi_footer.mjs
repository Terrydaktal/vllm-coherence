import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";

const root = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
test("installed footer is one continuous wrapping line with usage after temperatures", {
  skip: !existsSync(join(root, "pi-coding-agent/dist/modes/interactive/components/footer.js")),
}, async () => {
  const mod = path => import(pathToFileURL(join(root, path)));
  const { FooterComponent } = await mod("pi-coding-agent/dist/modes/interactive/components/footer.js");
  const { initTheme } = await mod("pi-coding-agent/dist/modes/interactive/theme/theme.js");
  const { stripTerminalSequences, visibleWidth } = await mod("pi-tui/dist/index.js");
  initTheme("dark", false);
  const statuses = new Map([
    ["qwen-gpu-temperature", "40°C · 38°C · 25%"],
    ["qwen-cache-residency", "Cache ≈ GPU 60,000 · RAM 0 · Disk 58,000 · Cold 160 tok"],
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
  const wide = footer.render(1000).map(stripTerminalSequences);
  assert.equal(wide.length, 1, "no hard break for path, model, or extension statuses");
  assert.ok(wide[0].indexOf("Cold 160 tok") < wide[0].indexOf("40°C"));
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
