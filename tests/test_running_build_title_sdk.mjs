// Offline release-path integration: use the installed Pi loader and terminal
// class in an owned child, never attach to a terminal or submit a model prompt.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const sdkRoot = process.env.QWEN_TEST_PI_ROOT ?? join(homedir(), ".local/share/qwen-r9700/pi/0.84.2/node_modules/@earendil-works");

test("installed Pi extension binds through its launcher symlink and decorates the real terminal without model calls", {
  timeout: 20000,
}, async (t) => {
  assert.ok(existsSync(join(sdkRoot, "pi-coding-agent/dist/core/sdk.js")),
    "Install the supported Pi SDK or set QWEN_TEST_PI_ROOT to its @earendil-works directory");
  const directory = await mkdtemp(join(tmpdir(), "pi-running-build-sdk-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const source = `
    import assert from "node:assert/strict";
    import { mkdir, rename, symlink, writeFile } from "node:fs/promises";
    import { join } from "node:path";
    import { pathToFileURL } from "node:url";
    const cwd = ${JSON.stringify(directory)};
    const sdkRoot = ${JSON.stringify(sdkRoot)};
    const extensionSource = ${JSON.stringify(join(root, "integrations/pi/qwen-running-build-title.mjs"))};
    const extension = join(cwd, "title-extension.mjs");
    await symlink(extensionSource, extension);
    const phase = (name) => process.stderr.write("title-sdk: " + name + "\\n");
    phase("import SDK");
    const mod = (path) => import(pathToFileURL(join(sdkRoot, path)));
    const { ProcessTerminal } = await mod("pi-tui/dist/terminal.js");
    const { createAgentSession } = await mod("pi-coding-agent/dist/core/sdk.js");
    const { SettingsManager } = await mod("pi-coding-agent/dist/core/settings-manager.js");
    const { SessionManager } = await mod("pi-coding-agent/dist/core/session-manager.js");
    const { DefaultResourceLoader } = await mod("pi-coding-agent/dist/core/resource-loader.js");
    const { InteractiveMode } = await mod("pi-coding-agent/dist/modes/interactive/interactive-mode.js");
    const selector = join(cwd, "pi");
    const originalEntry = join(sdkRoot, "pi-coding-agent/dist/cli.js");
    const nextEntry = join(cwd, "entry-next.js");
    await writeFile(nextEntry, "export const build = 'new-install';\\n");
    await symlink(originalEntry, selector);
    // Node preserves the user-facing launcher path in argv[1]. Resolve SDK
    // dependencies from its actual target without replacing this launch evidence.
    process.argv[1] = selector;
    process.env.QWEN_PI_LATEST_LAUNCHER = selector;
    delete process.env.QWEN_PI_RUNNING_RELEASE;
    const agentDir = join(cwd, "agent");
    await mkdir(agentDir);
    await mkdir(join(cwd, "runtime"), { mode: 0o700 });
    let networkRequests = 0, inferenceRequests = 0;
    globalThis.fetch = () => { networkRequests++; throw new Error("Network forbidden in title SDK fixture"); };
    const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
    const loader = new DefaultResourceLoader({ cwd, agentDir, settingsManager,
      noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
      systemPrompt: "Synthetic title regression fixture.", additionalExtensionPaths: [extension] });
    await loader.reload();
    phase("extensions loaded");
    assert.deepEqual(loader.getExtensions().errors, []);
    assert.equal(loader.getExtensions().extensions.length, 1);
    assert.equal(loader.getExtensions().extensions[0].tools.size, 0, "title status adds no model tools");
    const model = { id: "synthetic-title-fixture", name: "Synthetic title fixture", provider: "qwen-r9700",
      api: "openai-completions", baseUrl: "http://fixture.invalid/v1", reasoning: false, input: ["text"],
      contextWindow: 32768, maxTokens: 8192, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
    const modelRuntime = { getModel: () => model, getAvailableSnapshot: () => [model], hasConfiguredAuth: () => true,
      isUsingOAuth: () => false, getAuth: async () => ({ auth: { apiKey: "synthetic" }, env: {} }),
      streamSimple: () => { inferenceRequests++; throw new Error("Inference forbidden in title SDK fixture"); } };
    const { session } = await createAgentSession({ cwd, agentDir, model, modelRuntime, resourceLoader: loader,
      sessionManager: SessionManager.inMemory(cwd), settingsManager, tools: [] });
    phase("session created");
    const originalPrompt = session.agent.state.systemPrompt;
    const originalMessages = structuredClone(session.agent.state.messages);
    const originalTools = session.getAllTools().map((tool) => tool.name);
    const errors = [];
    session.extensionRunner.onError((error) => errors.push(error));
    const key = Symbol.for("coherence.pi.running-build-title.v1");
    const originalSetter = ProcessTerminal.prototype.setTitle;
    let guard, originalTitleWrites, nativeTitleChecks = 0;
    const writes = [];
    const stdoutWrite = process.stdout.write;
    try {
      await session.bindExtensions({});
      assert.equal(ProcessTerminal.prototype[key], undefined, "headless SDK binding must not alter terminal titles");
      await session.bindExtensions({ mode: "interactive", uiContext: { notify() {}, setWidget() {}, setStatus() {},
        setWorkingMessage() {}, setTitle() {} } });
      phase("interactive bound");
      assert.deepEqual(errors, [], "extension initialization must succeed through the installed SDK loader");
      guard = ProcessTerminal.prototype[key];
      assert.ok(guard, "the actual SDK terminal prototype must carry the extension guard");
      assert.notEqual(ProcessTerminal.prototype.setTitle, originalSetter);
      await guard.monitor.ready;
      phase("identity captured");
      assert.equal(process.argv[1], selector, "binding must preserve the real symlink invocation");
      assert.equal(guard.monitor.outdated, false);
      process.stdout.write = (chunk, encoding, callback) => {
        writes.push(Buffer.isBuffer(chunk) ? chunk.toString("utf8") : String(chunk));
        if (typeof encoding === "function") encoding(); else callback?.();
        return true;
      };
      const terminal = new ProcessTerminal();
      terminal.setTitle("π - Synthetic session - project-alpha");
      assert.equal(writes.at(-1), "\\x1b]0;π - Synthetic session - project-alpha\\x07");
      const activate = async (entry) => {
        const staged = join(cwd, "pi.next");
        await symlink(entry, staged);
        await rename(staged, selector);
        await guard.monitor.check();
      };
      await activate(nextEntry);
      phase("replacement checked");
      assert.equal(process.argv[1], selector, "a changed selector must not rewrite how the running process was invoked");
      assert.equal(guard.monitor.outdated, true);
      assert.equal(writes.at(-1), "\\x1b]0;π\\u0336 - Synthetic session - project-alpha\\x07");
      terminal.setTitle("π - Renamed session - project-beta");
      assert.equal(writes.at(-1), "\\x1b]0;π\\u0336 - Renamed session - project-beta\\x07",
        "later native title updates must keep the stale marker while preserving useful dynamic text");
      await session.bindExtensions({ mode: "interactive", uiContext: { notify() {}, setWidget() {}, setStatus() {},
        setWorkingMessage() {}, setTitle() {} } });
      assert.equal(ProcessTerminal.prototype[key], guard, "rebinding must not recapture an already stale baseline");
      assert.equal(guard.monitor.outdated, true);
      await activate(originalEntry);
      phase("restoration checked");
      assert.equal(guard.monitor.outdated, false);
      assert.equal(writes.at(-1), "\\x1b]0;π - Renamed session - project-beta\\x07");
      terminal.setTitle("π - Fresh session - project-gamma");
      assert.equal(writes.at(-1), "\\x1b]0;π - Fresh session - project-gamma\\x07");
      originalTitleWrites = writes.length;
      // Drive the installed SDK's actual title producer against an owned
      // session manager. Its cwd intentionally differs from the SDK runner's
      // initial cwd, so this also checks the session_start context source.
      let selectedCwd = join(cwd, "tasks/emma"), selectedName;
      session.sessionManager.getCwd = () => selectedCwd;
      session.sessionManager.getSessionName = () => selectedName;
      const interactive = { sessionManager: session.sessionManager, ui: { terminal } };
      const rebind = () => session.bindExtensions({ mode: "interactive", uiContext: {
        notify() {}, setWidget() {}, setStatus() {}, setWorkingMessage() {}, setTitle() {} } });
      const expectNativeTitle = (title) => {
        InteractiveMode.prototype.updateTerminalTitle.call(interactive);
        assert.equal(writes.at(-1), "\\x1b]0;" + title + "\\x07");
        nativeTitleChecks++;
      };
      await rebind();
      expectNativeTitle("π - ~/tasks/emma");
      selectedName = "Synthetic review";
      expectNativeTitle("π - Synthetic review - ~/tasks/emma");
      selectedName = "Renamed review";
      expectNativeTitle("π - Renamed review - ~/tasks/emma");
      selectedCwd = "/workspace/project-alpha";
      await rebind();
      expectNativeTitle("π - Renamed review - /workspace/project-alpha");
      selectedCwd = cwd;
      selectedName = undefined;
      await rebind();
      expectNativeTitle("π - ~");
      selectedCwd = join(cwd, "tasks/emma");
      await rebind();
      expectNativeTitle("π - ~/tasks/emma");
      await activate(nextEntry);
      expectNativeTitle("π\\u0336 - ~/tasks/emma");
      selectedName = "Another session";
      expectNativeTitle("π\\u0336 - Another session - ~/tasks/emma");
      await activate(originalEntry);
      expectNativeTitle("π - Another session - ~/tasks/emma");
      assert.equal(ProcessTerminal.prototype[key], guard, "native session changes retain the captured build baseline");
      phase("native SDK title lifecycle checked");
      assert.deepEqual(errors, []);
      assert.equal(session.agent.state.systemPrompt, originalPrompt, "title status must not change the model prompt");
      assert.deepEqual(session.agent.state.messages, originalMessages, "title status must not append conversation messages");
      assert.deepEqual(session.getAllTools().map((tool) => tool.name), originalTools);
      assert.equal(networkRequests, 0);
      assert.equal(inferenceRequests, 0);
    } finally {
      guard?.dispose();
      process.stdout.write = stdoutWrite;
      session.dispose();
    }
    assert.equal(ProcessTerminal.prototype.setTitle, originalSetter, "teardown restores the real terminal prototype");
    assert.equal(ProcessTerminal.prototype[key], undefined);
    process.stdout.write(JSON.stringify({ networkRequests, inferenceRequests, titles: originalTitleWrites, nativeTitleChecks }) + "\\n");
  `;
  const result = spawnSync(process.execPath, ["--input-type=module", "--eval", source], {
    cwd: directory, encoding: "utf8", timeout: 18000, maxBuffer: 1024 * 1024,
    env: { ...process.env, HOME: directory, PI_CODING_AGENT_DIR: join(directory, "agent"),
      XDG_RUNTIME_DIR: join(directory, "runtime"), PI_OFFLINE: "1", PI_TUI_WRITE_LOG: "" },
  });
  assert.equal(result.error?.code, undefined, result.stderr || result.error?.message);
  assert.equal(result.status, 0, result.stderr || result.stdout);
  const receipt = JSON.parse(result.stdout.trim());
  assert.deepEqual(receipt, { networkRequests: 0, inferenceRequests: 0, titles: 5, nativeTitleChecks: 9 });
});


for (const exitCommand of ["/quit", "/exit"]) {
test(`actual Pi CLI launched through a symlink shows the project title and exits cleanly with ${exitCommand}`, {
  timeout: 20000,
}, () => {
  // Exercise Node's real argv[1], rather than emulating launch arguments in an
  // SDK harness. Own the PTY, config, model, home and selector; never use a chat.
  const source = `
import json, os, pathlib, pty, select, signal, subprocess, tempfile, time
root = pathlib.Path(${JSON.stringify(root)})
cli = pathlib.Path(${JSON.stringify(join(sdkRoot, "pi-coding-agent/dist/cli.js"))})
exit_command = ${JSON.stringify(exitCommand)}
with tempfile.TemporaryDirectory(prefix="pi-title-real-cli-") as td:
    directory = pathlib.Path(td)
    cwd = directory / "tasks" / "emma"
    cwd.mkdir(parents=True)
    agent = directory / "agent"
    agent.mkdir()
    runtime = directory / "runtime"
    runtime.mkdir(mode=0o700)
    selector = directory / "pi"
    selector.symlink_to(cli)
    network_log = directory / "network.log"
    guard = directory / "no-network.mjs"
    guard.write_text('import net from "node:net"; import fs from "node:fs"; const fail=()=>{fs.appendFileSync(process.env.TITLE_NETWORK_LOG,"attempt"); throw new Error("Network forbidden in title CLI fixture")}; globalThis.fetch=fail; net.Socket.prototype.connect=fail;')
    model = {"id":"title-fixture", "name":"Synthetic title fixture", "reasoning":False,
        "input":["text"], "contextWindow":32768, "maxTokens":8192,
        "cost":{"input":0, "output":0, "cacheRead":0, "cacheWrite":0}}
    (agent / "models.json").write_text(json.dumps({"providers":{"title-fixture":{
        "baseUrl":"http://127.0.0.1:9/v1", "api":"openai-completions",
        "apiKey":"synthetic", "models":[model]}}}))
    env = {"PATH":os.environ["PATH"], "HOME":str(directory), "TERM":"xterm-256color",
        "PI_CODING_AGENT_DIR":str(agent), "PI_OFFLINE":"1", "PI_TELEMETRY":"0",
        "XDG_RUNTIME_DIR":str(runtime),
        "QWEN_PI_LATEST_LAUNCHER":str(selector), "TITLE_NETWORK_LOG":str(network_log)}
    args = [${JSON.stringify(process.execPath)}, "--import", str(guard), str(selector),
        "--offline", "--no-session", "--no-tools", "--no-extensions", "--no-skills",
        "--no-prompt-templates", "--no-themes", "--no-context-files",
        "--provider", "title-fixture", "--model", "title-fixture",
        "--extension", str(root / "integrations/pi/qwen-running-build-title.mjs")]
    if exit_command == "/exit":
        args.extend(["--extension", str(root / "integrations/pi/qwen-client-commands.mjs")])
    master, slave = pty.openpty()
    process = subprocess.Popen(args, cwd=cwd, env=env, stdin=slave, stdout=slave,
        stderr=slave, start_new_session=True)
    os.close(slave)
    captured = b""
    expected = bytes([27]) + "]0;π - ~/tasks/emma".encode() + bytes([7])
    deadline = time.monotonic() + 12
    record = None
    try:
        while time.monotonic() < deadline and (expected not in captured or record is None):
            if select.select([master], [], [], .1)[0]:
                try: captured += os.read(master, 65536)
                except OSError: break
            if process.poll() is not None: break
            records = list((runtime / "pi/processes").glob(str(process.pid) + "-*.json"))
            if records:
                record_path = records[0]
                record = json.loads(record_path.read_text())
        assert expected in captured, repr(captured[-6000:])
        assert b"Cannot find module" not in captured, repr(captured[-6000:])
        assert record is not None, "Actual CLI must publish its startup build identity"
        ticks = pathlib.Path('/proc/' + str(process.pid) + '/stat').read_text().rsplit(')', 1)[1].split()[19]
        assert record["schema"] == "coherence.pi.process-build.v1"
        assert record["pid"] == process.pid and record["start_ticks"] == ticks
        assert record["entry_point"] == str(cli.resolve())
        assert record["version"] == "0.84.2"
        assert record_path.stat().st_mode & 0o777 == 0o600
        installed = cli.parent.parent / 'coherence-build.json'
        if installed.is_file():
            manifest = json.loads(installed.read_text())
            assert record["running"]["sdk_fingerprint"] == manifest["sdk_fingerprint"], "Independent installer and process SDK digests must agree"
            assert record["latest"]["status"] == "current", record["latest"]
        else:
            assert record["latest"]["status"] == "unknown", "No installed manifest must never be called current"
        os.write(master, exit_command.encode() + bytes([13]))
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], .1)[0]:
                try: captured += os.read(master, 65536)
                except OSError: break
        # PTY EOF/EIO can arrive before waitpid observes the child exiting.
        # Reap the owned process with a bound rather than racing poll() here.
        assert process.wait(timeout=2) == 0, repr(captured[-6000:])
        assert not record_path.exists(), "Normal shutdown must remove the process record"
        assert not network_log.exists(), "No request or startup network operation is allowed"
        print(json.dumps({"title":"π - ~/tasks/emma", "networkRequests":0, "exitCode":0}))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        os.close(master)
  `;
  const result = spawnSync("python3", ["-c", source], {
    encoding: "utf8", timeout: 19000, maxBuffer: 1024 * 1024,
  });
  assert.equal(result.error?.code, undefined, result.stderr || result.error?.message);
  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.deepEqual(JSON.parse(result.stdout), {
    title: "π - ~/tasks/emma", networkRequests: 0, exitCode: 0,
  });
});
}
