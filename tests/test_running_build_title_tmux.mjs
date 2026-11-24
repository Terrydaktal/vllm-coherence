// Private tmux server + actual Pi terminal writer. No existing windows, model
// requests, user configuration or session transcripts are accessed.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rename, rm, symlink, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const sdk = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");
const factory = fileURLToPath(new URL("../integrations/pi/qwen-running-build-title.mjs", import.meta.url));
const tmuxAvailable = spawnSync("tmux", ["-V"], { encoding: "utf8", timeout: 2000 }).status === 0;

async function waitFor(operation, predicate, label) {
  const deadline = Date.now() + 5000;
  let last;
  do {
    last = await operation();
    if (predicate(last)) return last;
    await new Promise((done) => setTimeout(done, 20));
  } while (Date.now() < deadline);
  assert.fail(`${label}; last observed ${JSON.stringify(last)}`);
}

test("private tmux preserves normal, stale and restored real Pi titles without replacing its process", {
  timeout: 20000,
  skip: tmuxAvailable ? false : "tmux unavailable; install it to qualify title forwarding",
}, async (t) => {
  const terminalModule = join(sdk, "pi-tui/dist/terminal.js");
  assert.ok(existsSync(terminalModule), "install the supported Pi SDK or set QWEN_TEST_PI_ROOT");
  const directory = await mkdtemp(join(tmpdir(), "pi-title-tmux-"));
  const socket = `pi-title-${process.pid}-${directory.split("/").at(-1)}`;
  const env = { ...process.env, HOME: directory, TMUX: "", NODE_OPTIONS: "" };
  const command = (...args) => spawnSync("tmux", ["-L", socket, "-f", "/dev/null", ...args], {
    env, encoding: "utf8", timeout: 3000, maxBuffer: 32768,
  });
  t.after(async () => {
    command("kill-server");
    await rm(directory, { recursive: true, force: true });
  });
  const cli = join(directory, "cli.mjs");
  const alternate = join(directory, "alternate.mjs");
  const selector = join(directory, "pi");
  const child = join(directory, "child.mjs");
  const ready = join(directory, "ready.json");
  const initialCode = 'export const version = "0.84.2"; export const build = "original";\n';
  await writeFile(cli, initialCode);
  await writeFile(alternate, 'export const version = "0.84.2"; export const build = "alternate";\n');
  await symlink(cli, selector);
  await writeFile(child, `
    import { writeFile } from 'node:fs/promises';
    import { pathToFileURL } from 'node:url';
    const { ProcessTerminal } = await import(pathToFileURL(${JSON.stringify(terminalModule)}));
    const { installRunningBuildTitle } = await import(pathToFileURL(${JSON.stringify(factory)}));
    const guard = installRunningBuildTitle(ProcessTerminal, { monitorOptions: {
      entrypoint: ${JSON.stringify(cli)}, launcher: ${JSON.stringify(selector)},
      releaseRoot: undefined, extensionRoots: [], debounceMs: 15, pollMs: 50
    }});
    await guard.start();
    const terminal = new ProcessTerminal();
    terminal.setTitle('π - Synthetic session - project');
    await writeFile(${JSON.stringify(ready)}, JSON.stringify({pid:process.pid}));
    setInterval(() => {}, 1000);
    process.on('SIGTERM', () => { guard.dispose(); process.exit(0); });
  `);
  const started = command("new-session", "-d", "-s", "title", "-c", directory, process.execPath, child);
  assert.equal(started.status, 0, started.stderr || String(started.error));
  const childPid = await waitFor(async () => {
    try { return JSON.parse(await readFile(ready, "utf8")).pid; } catch { return undefined; }
  }, (value) => Number.isInteger(value), "owned child did not become ready");
  const pane = () => {
    const result = command("display-message", "-p", "-t", "title:0.0", "#{pane_pid}\t#{pane_title}");
    assert.equal(result.status, 0, result.stderr || String(result.error));
    const [pid, title] = result.stdout.trim().split("\t");
    return { pid: Number(pid), title };
  };
  const original = await waitFor(pane, (value) => value.title === "π - Synthetic session - project", "normal title not recorded by tmux");
  assert.equal(original.pid, childPid, "tmux must directly own the same Node process");
  const expect = async (title) => {
    const result = await waitFor(pane, (value) => value.title === title, `tmux did not record ${title}`);
    assert.equal(result.pid, childPid, "title transitions must retain the running process");
  };
  // Both replacement mechanisms must update an otherwise idle application's title.
  const stagedCli = join(directory, "cli.next.mjs");
  await writeFile(stagedCli, 'export const version = "0.84.2"; export const build = "atomic";\n');
  await rename(stagedCli, cli);
  await expect("π̶ - Synthetic session - project");
  await writeFile(stagedCli, initialCode);
  await rename(stagedCli, cli);
  await expect("π - Synthetic session - project");
  const stagedLink = join(directory, "pi.next");
  await symlink(alternate, stagedLink);
  await rename(stagedLink, selector);
  await expect("π̶ - Synthetic session - project");
  await symlink(cli, stagedLink);
  await rename(stagedLink, selector);
  await expect("π - Synthetic session - project");
});
