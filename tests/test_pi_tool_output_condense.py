import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PI_DIR = ROOT / "integrations" / "pi"
EXTENSION = PI_DIR / "qwen-tool-output-condense.mjs"
REHYDRATE = PI_DIR / "qwen-tool-turn-rehydrate.mjs"
LAUNCHER = ROOT / "scripts" / "pi-remote-qwen"


def test_launcher_loads_tool_output_condensation_extension() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    extension = EXTENSION.read_text(encoding="utf-8")

    definition = (
        'tool_output_condense_extension="$project_root/integrations/pi/'
        'qwen-tool-output-condense.mjs"'
    )
    invocation = '--extension "$tool_output_condense_extension"'
    progress = '--extension "$progress_extension"'
    repetition = '--extension "$repetition_guard_extension"'

    assert definition in launcher
    assert launcher.count(invocation) == 1
    assert launcher.index(progress) < launcher.index(invocation) < launcher.index(repetition)
    assert 'pi.on("tool_result"' in extension
    assert 'pi.on("tool_call"' in extension
    assert 'pi.on("tool_execution_start"' in extension
    assert 'pi.on("before_agent_start"' in extension
    assert "rg with a narrow pattern" in extension
    assert "jq projections" in extension
    assert "sed -n" in extension
    assert "literal pattern or tight start_line/end_line range" in extension
    assert "event.toolName === REHYDRATE_TOOL_NAME" in extension
    assert "export const DEFAULT_MAX_CONTEXT_BYTES = 8 * 1024;" in extension


def test_tool_result_condensation_lifecycle(tmp_path: Path) -> None:
    extension_url = EXTENSION.as_uri()
    harness = f"""
import {{ createHash }} from "node:crypto";
import {{
  chmodSync,
  existsSync,
  linkSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
}} from "node:fs";
import condense, {{ DEFAULT_MAX_CONTEXT_BYTES }} from {json.dumps(extension_url)};
import rehydrate from {json.dumps(REHYDRATE.as_uri())};

let now = 1000;
Object.defineProperty(globalThis, "performance", {{
  value: {{ now() {{ return now; }} }},
  configurable: true,
}});

const handlers = new Map();
condense({{ on(name, handler) {{ handlers.set(name, handler); }} }});
let rehydrateDefinition;
rehydrate({{ registerTool(value) {{ rehydrateDefinition = value; }} }});
if (!rehydrateDefinition || rehydrateDefinition.name !== "qwen_rehydrate_tool_turn") {{
  process.exit(41);
}}

const ctx = {{ model: {{ provider: "qwen-r9700" }} }};
delete process.env.QWEN_PI_FIXED_SLOT_RESUME;

const blockedRootFind = await handlers.get("tool_call")({{
  toolName: "bash",
  input: {{ command: 'find / -name "pi-coding-agent" -type d 2>/dev/null | head' }},
}});
if (!blockedRootFind?.block || !blockedRootFind.reason.includes("unbounded find /")) {{
  process.exit(42);
}}
if (await handlers.get("tool_call")({{
  toolName: "bash",
  input: {{ command: 'find /home/lewis/tasks -name "pi-coding-agent" -type d' }},
}}) !== undefined) process.exit(43);
if (await handlers.get("tool_call")({{
  toolName: "bash",
  input: {{ command: "cd /tmp && find / -name '*.json'" }},
}})?.block !== true) process.exit(44);
const prompt = await handlers.get("before_agent_start")({{
  systemPrompt: "base prompt",
}}, ctx);
if (!prompt.systemPrompt.includes("Qwen bounded tool-output discipline")) process.exit(9);
if (!prompt.systemPrompt.includes("rg with a narrow pattern")) process.exit(10);
const duplicate = await handlers.get("before_agent_start")({{
  systemPrompt: prompt.systemPrompt,
}}, ctx);
if (duplicate !== undefined) process.exit(11);

process.env.QWEN_PI_FIXED_SLOT_RESUME = "1";
const snapshotPrompt = await handlers.get("before_agent_start")({{
  systemPrompt: "snapshot-bound prompt",
}}, ctx);
if (snapshotPrompt !== undefined) process.exit(27);
delete process.env.QWEN_PI_FIXED_SLOT_RESUME;

await handlers.get("tool_execution_start")({{ toolCallId: "interrupted-before-result" }});
await handlers.get("session_shutdown")();
if (existsSync(process.env.QWEN_PI_TOOL_RESULT_DIR)) {{
  throw new Error(
    "session shutdown during tool output published an archive before a result existed",
  );
}}

const small = {{
  toolName: "bash",
  toolCallId: "small",
  input: {{ command: "printf short" }},
  content: [{{ type: "text", text: "short" }}],
  details: undefined,
  isError: false,
}};
await handlers.get("tool_execution_start")({{ toolCallId: "small" }});
if (await handlers.get("tool_result")(small) !== undefined) process.exit(12);

const lines = [];
for (let index = 1; index <= 180; index += 1) {{
  if (index === 91) lines.push("WARNING: bounded diagnostic marker");
  else lines.push(`line-${{String(index).padStart(3, "0")}} ${{"x".repeat(60)}}`);
}}
const fullText = lines.join("\\n") + "\\n";
await handlers.get("tool_execution_start")({{ toolCallId: "large/call" }});
now = 2750;
const large = await handlers.get("tool_result")({{
  toolName: "bash",
  toolCallId: "large/call",
  input: {{ command: "large-output" }},
  content: [{{ type: "text", text: fullText }}],
  details: undefined,
  isError: false,
}});
const summary = large.content[0].text;
if (Buffer.byteLength(summary, "utf8") > DEFAULT_MAX_CONTEXT_BYTES) process.exit(13);
if (!summary.includes("Exit status: 0")) process.exit(14);
if (!summary.includes("Duration: 1.75 s")) process.exit(15);
if (!summary.includes("Original size: 180 lines")) process.exit(16);
if (!summary.includes("WARNING: bounded diagnostic marker")) process.exit(17);
if (!summary.includes("1: line-001") || !summary.includes("180: line-180")) process.exit(18);
const retained = summary.match(/Content-addressed archive: (.+)/)?.[1];
if (!retained || readFileSync(retained, "utf8") !== fullText) process.exit(19);
const retainedStat = statSync(retained);
if ((retainedStat.mode & 0o777) !== 0o400 || retainedStat.nlink !== 1) process.exit(20);
if (large.details.fullOutputPath !== retained) process.exit(21);
const retainedSha = createHash("sha256").update(fullText).digest("hex");
if (!retained.endsWith(`/sha256/${{retainedSha.slice(0, 2)}}/${{retainedSha}}.txt`)) {{
  process.exit(28);
}}
if (large.details.qwenToolResultArchive.sha256 !== retainedSha) process.exit(29);
if (!summary.includes(`Archive SHA-256: ${{retainedSha}}`)) process.exit(30);
if (!summary.includes("literal pattern or tight start_line/end_line range")) process.exit(31);
if (!summary.includes("Do not infer omitted content")) process.exit(40);

const preview = await rehydrateDefinition.execute("preview", {{ sha256: retainedSha }});
const previewText = preview.content[0].text;
if (Buffer.byteLength(previewText, "utf8") > DEFAULT_MAX_CONTEXT_BYTES) process.exit(45);
if (!previewText.includes(`Archive path: ${{retained}}`)) process.exit(46);
if (!previewText.includes("digest-only preview (first 12 lines maximum)")) process.exit(47);
if (preview.details.archivePath !== retained || preview.details.selectedLines.length !== 12) {{
  process.exit(48);
}}
const repeatedPreview = await rehydrateDefinition.execute(
  "preview-again",
  {{ sha256: retainedSha }},
);
if (repeatedPreview.content[0].text !== previewText) process.exit(49);
const exactMatch = await rehydrateDefinition.execute("exact-match", {{
  sha256: retainedSha,
  pattern: "WARNING: bounded diagnostic marker",
  context: 0,
  max_lines: 1,
}});
if (exactMatch.details.selectedLines.join(",") !== "91") process.exit(57);
if (!exactMatch.content[0].text.includes("91: WARNING: bounded diagnostic marker")) {{
  process.exit(58);
}}

await handlers.get("tool_execution_start")({{ toolCallId: "rehydrated-preview" }});
if (await handlers.get("tool_result")({{
  toolName: "qwen_rehydrate_tool_turn",
  toolCallId: "rehydrated-preview",
  input: {{ sha256: retainedSha }},
  content: preview.content,
  details: preview.details,
  isError: false,
}}) !== undefined) process.exit(50);

const targeted = await rehydrateDefinition.execute("targeted", {{
  sha256: retainedSha,
  start_line: 1,
  end_line: 180,
  max_lines: 180,
}});
if (Buffer.byteLength(targeted.content[0].text, "utf8") <= DEFAULT_MAX_CONTEXT_BYTES) {{
  process.exit(51);
}}
if (!targeted.content[0].text.includes("Selection mode: explicit targeted retrieval")) {{
  process.exit(52);
}}
await handlers.get("tool_execution_start")({{ toolCallId: "rehydrated-targeted" }});
if (await handlers.get("tool_result")({{
  toolName: "qwen_rehydrate_tool_turn",
  toolCallId: "rehydrated-targeted",
  input: {{ sha256: retainedSha, start_line: 1, end_line: 180, max_lines: 180 }},
  content: targeted.content,
  details: targeted.details,
  isError: false,
}}) !== undefined) process.exit(53);

const secondaryArchiveFor = (text) => {{
  const digest = createHash("sha256").update(text).digest("hex");
  return [
    process.env.QWEN_PI_TOOL_RESULT_DIR,
    "sha256",
    digest.slice(0, 2),
    `${{digest}}.txt`,
  ].join("/");
}};
if (existsSync(secondaryArchiveFor(previewText))) process.exit(54);
if (existsSync(secondaryArchiveFor(targeted.content[0].text))) process.exit(59);
const retainedDirectory = retained.slice(0, retained.lastIndexOf("/"));
const archiveCount = () =>
  readdirSync(retainedDirectory).filter((name) => name.endsWith(".txt")).length;
if (archiveCount() !== 1) process.exit(55);

const controller = new AbortController();
controller.abort();
let cancelled = false;
try {{
  await rehydrateDefinition.execute(
    "cancelled",
    {{ sha256: retainedSha }},
    controller.signal,
  );
}}
catch (error) {{ cancelled = String(error).includes("cancelled"); }}
if (!cancelled || archiveCount() !== 1) {{
  process.exit(56);
}}

const interruptedLink = retained.replace(
  `${{retainedSha}}.txt`,
  `.${{retainedSha}}.999.123e4567-e89b-42d3-a456-426614174000.tmp`,
);
linkSync(retained, interruptedLink);
if (statSync(retained).nlink !== 2) process.exit(35);
await handlers.get("tool_execution_start")({{ toolCallId: "large-again" }});
const duplicateLarge = await handlers.get("tool_result")({{
  toolName: "bash",
  toolCallId: "large-again",
  input: {{ command: "large-output" }},
  content: [{{ type: "text", text: fullText }}],
  details: undefined,
  isError: false,
}});
if (duplicateLarge.details.fullOutputPath !== retained) process.exit(32);
if (existsSync(interruptedLink) || statSync(retained).nlink !== 1) process.exit(36);

const piRetained = {json.dumps(str(tmp_path / "pi-retained.log"))};
const original = Array.from({{ length: 2500 }}, (_, index) =>
  index === 1200 ? "ERROR: original retained failure" : `retained-${{index + 1}}`
).join("\\n") + "\\n";
writeFileSync(piRetained, original, {{ mode: 0o600 }});
chmodSync(piRetained, 0o600);
await handlers.get("tool_execution_start")({{ toolCallId: "pi-path" }});
now = 4000;
const fromPiPath = await handlers.get("tool_result")({{
  toolName: "bash",
  toolCallId: "pi-path",
  input: {{ command: "retained-output" }},
  content: [{{ type: "text", text: "truncated tail\\n\\nCommand exited with code 7" }}],
  details: {{ fullOutputPath: piRetained }},
  isError: true,
}});
const piSummary = fromPiPath.content[0].text;
if (!piSummary.includes("Original size: 2500 lines")) process.exit(22);
if (!piSummary.includes("ERROR: original retained failure")) process.exit(23);
if (piSummary.includes(piRetained)) process.exit(24);
if (!piSummary.includes("Exit status: 7")) process.exit(25);
const piArchive = fromPiPath.details.fullOutputPath;
if (readFileSync(piArchive, "utf8") !== original) process.exit(33);
if ((statSync(piArchive).mode & 0o777) !== 0o400) process.exit(34);

const image = {{
  toolName: "read",
  toolCallId: "image",
  content: [{{ type: "image", mimeType: "image/png", data: "AA==" }}],
  details: undefined,
  isError: false,
}};
if (await handlers.get("tool_result")(image) !== undefined) process.exit(26);

await handlers.get("agent_settled")();
await handlers.get("session_shutdown")();
if (!existsSync(retained) || readFileSync(retained, "utf8") !== fullText) process.exit(37);
if (statSync(retained).nlink !== 1 || (statSync(retained).mode & 0o777) !== 0o400) {{
  process.exit(38);
}}
if (!existsSync(piArchive) || readFileSync(piArchive, "utf8") !== original) process.exit(39);
if (statSync(piArchive).nlink !== 1 || (statSync(piArchive).mode & 0o777) !== 0o400) {{
  process.exit(40);
}}
"""
    environment = dict(os.environ)
    environment["QWEN_PI_TOOL_RESULT_DIR"] = str(tmp_path / "retained")
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", harness],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def test_sectional_budget_preserves_tail_complete_errors_and_tool_signals() -> None:
    extension_url = EXTENSION.as_uri()
    harness = f"""
import {{ buildCondensedSummary, inspectText }} from {json.dumps(extension_url)};

const diagnostic = [
  "prelude",
  "",
  "Traceback (most recent call last):",
  "  File \\\"worker.py\\\", line 9, in run",
  "    compile_target()",
  "  File \\\"worker.py\\\", line 4, in compile_target",
  "RuntimeError: exact-boom",
  "  exact-detail-" + "z".repeat(320),
  "",
  "pytest: 45 passed, 2 skipped in 1.20s",
  "median latency 81.4 ms throughput 61.9 tok/s",
  "diff --git a/kernel.py b/kernel.py",
  "@@ -1,2 +1,2 @@",
  '\"verified\": true,',
  "name    median    p95",
  ...Array.from({{ length: 120 }}, (_, index) => `body-${{index + 1}} ${{"x".repeat(40)}}`),
  "FINAL-SENTINEL",
].join("\\n");
const archive = {{
  bytes: Buffer.byteLength(diagnostic),
  path: "/tmp/qwen-tool-results/sha256/ab/" + "a".repeat(64) + ".txt",
  sha256: "a".repeat(64),
}};
const inspection = inspectText(diagnostic);
const normal = buildCondensedSummary(
  {{ archive, durationMs: 1250, inspection, status: "0", toolName: "bash" }},
  3600,
);
for (const expected of [
  "Traceback (most recent call last):",
  'File \\\"worker.py\\\", line 9, in run',
  "compile_target()",
  "RuntimeError: exact-boom",
  "exact-detail-" + "z".repeat(320),
  "45 passed, 2 skipped",
  "81.4 ms throughput 61.9 tok/s",
  "diff --git",
  "@@ -1,2 +1,2 @@",
  '\"verified\": true',
  "name    median    p95",
  "FINAL-SENTINEL",
]) {{
  if (!normal.includes(expected)) throw new Error(`missing normal-budget evidence: ${{expected}}`);
}}
if (Buffer.byteLength(normal) > 3600) throw new Error("normal summary exceeded budget");

const minimum = buildCondensedSummary(
  {{ archive, durationMs: 1250, inspection, status: "0", toolName: "bash" }},
  1024,
);
if (Buffer.byteLength(minimum) > 1024) throw new Error("minimum summary exceeded budget");
if (!minimum.includes("Tail (guaranteed):")) {{
  throw new Error("minimum summary omitted tail section");
}}
if (!minimum.includes("FINAL-SENTINEL")) throw new Error("minimum summary omitted final line");
if (!minimum.includes("Archive SHA-256: " + "a".repeat(64))) {{
  throw new Error("minimum summary omitted digest");
}}
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", harness],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_single_line_megabyte_json_inspection_is_bounded() -> None:
    extension_url = EXTENSION.as_uri()
    harness = f"""
import {{ inspectText }} from {json.dumps(extension_url)};

const huge = JSON.stringify({{
  tokensInUsd: Array.from({{ length: 80_000 }}, (_, index) => ({{
    token: `asset-${{index}}`,
    value: index,
  }})),
}});
const started = performance.now();
const inspection = inspectText(huge);
const elapsedMs = performance.now() - started;
if (inspection.lines !== 1 || inspection.bytes !== Buffer.byteLength(huge)) {{
  throw new Error("large single-line JSON accounting changed");
}}
if (!inspection.signals.json.length) throw new Error("JSON signal was not retained");
if (elapsedMs > 1_000) throw new Error(`inspection took ${{elapsedMs}} ms`);
console.log(JSON.stringify({{ bytes: inspection.bytes, elapsedMs }}));
"""
    started = time.monotonic()
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", harness],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 0, result.stderr
    assert elapsed < 5
    measurement = json.loads(result.stdout)
    assert measurement["bytes"] > 2_000_000
