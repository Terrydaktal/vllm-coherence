import assert from "node:assert/strict";
import test from "node:test";
import runningBuildTitle, { expandTitleCwd, installRunningBuildTitle } from "../integrations/pi/qwen-running-build-title.mjs";
import { homedir } from "node:os";
import { join } from "node:path";

function fixture(t) {
  class Terminal { setTitle(title) { this.writes.push(title); } constructor() { this.writes = []; } }
  let onChange, starts = 0, stops = 0;
  const monitor = { outdated: false, start: async () => { starts++; }, stop: () => { stops++; } };
  const options = { registerExitHook: false, monitorFactory(value) { onChange = value.onChange; return monitor; } };
  const guard = installRunningBuildTitle(Terminal, options);
  t.after(() => guard.dispose());
  return { Terminal, guard, monitor, options, starts: () => starts, stops: () => stops,
    stale(value) { monitor.outdated = value; onChange(); } };
}

test("native titles use the whole home-relative cwd without changing session names or custom titles", () => {
  const home = "/home/lewis", cwd = `${home}/tasks/emma`;
  assert.equal(expandTitleCwd("π - emma", cwd, home), "π - ~/tasks/emma");
  assert.equal(expandTitleCwd("π - My session - emma", cwd, home), "π - My session - ~/tasks/emma");
  assert.equal(expandTitleCwd("π - lewis", home, home), "π - ~");
  assert.equal(expandTitleCwd("π - emma", "/home/lewis-other/emma", home), "π - /home/lewis-other/emma");
  assert.equal(expandTitleCwd("π - ", "/", home), "π - /");
  for (const title of ["", "custom title", "π - Other project", "π - ~/tasks/emma"]) {
    assert.equal(expandTitleCwd(title, cwd, home), title);
  }
});

test("cwd updates keep the stale marker and distinguish projects with identical basenames", (t) => {
  const f = fixture(t), terminal = new f.Terminal();
  f.guard.setCwd(join(homedir(), "tasks", "emma"));
  terminal.setTitle("π - emma");
  assert.equal(terminal.writes.at(-1), "π - ~/tasks/emma");
  f.stale(true);
  assert.equal(terminal.writes.at(-1), "π\u0336 - ~/tasks/emma");
  f.guard.setCwd(join(homedir(), "repos", "emma"));
  assert.equal(terminal.writes.at(-1), "π\u0336 - ~/repos/emma");
  terminal.setTitle("π - Named session - emma");
  assert.equal(terminal.writes.at(-1), "π\u0336 - Named session - ~/repos/emma");
  f.stale(false);
  assert.equal(terminal.writes.at(-1), "π - Named session - ~/repos/emma");
});

test("native title updates keep session text and immediately reflect stale and restored builds", (t) => {
  const f = fixture(t), terminal = new f.Terminal();
  terminal.setTitle("π - fixture - project");
  f.stale(true);
  assert.equal(terminal.writes.at(-1), "π\u0336 - fixture - project");
  terminal.setTitle("π - renamed - project");
  assert.equal(terminal.writes.at(-1), "π\u0336 - renamed - project");
  f.stale(false);
  assert.equal(terminal.writes.at(-1), "π - renamed - project");
  const count = terminal.writes.length;
  terminal.setTitle("π - renamed - project");
  assert.equal(terminal.writes.length, count, "unchanged titles add no terminal writes");
});

test("extension reload retains the same monitor baseline rather than blessing an old process", async (t) => {
  const f = fixture(t), terminal = new f.Terminal();
  terminal.setTitle("Pi - project");
  f.stale(true);
  const again = installRunningBuildTitle(f.Terminal, { monitorFactory() { assert.fail("baseline recaptured"); } });
  assert.equal(again, f.guard);
  await again.start();
  assert.equal(f.starts(), 1);
  terminal.setTitle("Pi - next session");
  assert.equal(terminal.writes.at(-1), "P\u0336i\u0336 - next session");
});

test("disabled titles stay empty; custom titles retain their text with a stale app suffix", (t) => {
  const f = fixture(t), terminal = new f.Terminal();
  terminal.setTitle("");
  f.stale(true);
  assert.deepEqual(terminal.writes, [""]);
  terminal.setTitle("custom project");
  assert.match(terminal.writes.at(-1), /^custom project.*π\u0336$/);
  f.stale(false);
  assert.equal(terminal.writes.at(-1), "custom project");
});

test("disposing stops monitoring and restores the original terminal setter", (t) => {
  const f = fixture(t), terminal = new f.Terminal();
  terminal.setTitle("π - fixture");
  f.stale(true);
  f.guard.dispose();
  f.guard.dispose();
  assert.equal(f.stops(), 1);
  terminal.setTitle("π - final");
  assert.equal(terminal.writes.at(-1), "π - final");
  f.stale(true);
  assert.equal(terminal.writes.at(-1), "π - final");
});

test("noninteractive sessions register no terminal monitor or additional tools", async () => {
  const handlers = new Map();
  runningBuildTitle({ on(name, handler) { handlers.set(name, handler); } });
  assert.deepEqual([...handlers.keys()], ["session_start"]);
  await handlers.get("session_start")({}, { hasUI: false });
});
