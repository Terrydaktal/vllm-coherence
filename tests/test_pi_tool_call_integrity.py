from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "integrations" / "pi" / "qwen-tool-call-integrity.mjs"
MAIN_LAUNCHER = ROOT / "scripts" / "pi-remote-qwen"
AGGRESSIVE_LAUNCHER = ROOT / "scripts" / "pi-remote-qwen-aggressive"
RADIANCE_LAUNCHER = ROOT / "scripts" / "pi-remote-qwen-radiance"


def run_harness(source: str) -> subprocess.CompletedProcess[str]:
    harness = f"""
import integrity from {json.dumps(EXTENSION.as_uri())};

function instance() {{
  const handlers = new Map();
  integrity({{ on(name, handler) {{ handlers.set(name, handler); }} }});
  return handlers;
}}
const ctx = {{
  mode: "tui",
  model: {{ provider: "qwen-r9700", api: "openai-completions" }},
  ui: {{ setStatus() {{}} }},
}};
const payload = {{
  model: "qwen3.8-27b-philbert-aggressive",
  stream: true,
  temperature: 1,
  top_p: 0.95,
  top_k: 40,
  messages: [{{ role: "user", content: "perform the action" }}],
}};
function message(content, stopReason = "toolUse") {{
  const rawTokenIds = content.flatMap((block) =>
    block?.type === "toolCall" ? [248058, 248059] : []
  );
  const result = {{
    role: "assistant",
    provider: "qwen-r9700",
    api: "openai-completions",
    model: "qwen3.8-27b-philbert-aggressive",
    stopReason,
    rawStopReason: stopReason,
    content,
    usage: {{ input: 10, output: rawTokenIds.length }},
  }};
  result[Symbol.for("qwen-r9700:raw-completion-token-ids:v1")] = rawTokenIds;
  return result;
}}
async function begin(handlers, block) {{
  const initial = await handlers.get("before_provider_request")({{ payload }}, ctx);
  if (initial?.return_token_ids !== true) throw new Error("raw token capture was not requested");
  await handlers.get("message_start")({{ message: message([]) }}, ctx);
  await handlers.get("message_update")({{
    assistantMessageEvent: {{
      type: "toolcall_start", contentIndex: 0, partial: message([block]),
    }},
  }}, ctx);
}}
async function delta(handlers, block, text) {{
  await handlers.get("message_update")({{
    assistantMessageEvent: {{
      type: "toolcall_delta", contentIndex: 0, delta: text, partial: message([block]),
    }},
  }}, ctx);
}}
async function end(handlers, block) {{
  await handlers.get("message_update")({{
    assistantMessageEvent: {{
      type: "toolcall_end", contentIndex: 0, toolCall: block, partial: message([block]),
    }},
  }}, ctx);
  return handlers.get("message_end")({{ message: message([block]) }}, ctx);
}}
{source}
"""
    return subprocess.run(
        ["node", "--input-type=module", "--eval", harness],
        check=False,
        capture_output=True,
        text=True,
    )


def test_complete_fragmented_json_and_valid_shell_are_accepted_unchanged() -> None:
    result = run_harness(
        r"""
const handlers = instance();
const block = {
  type: "toolCall", id: "bash-1", name: "bash",
  arguments: { command: "cat <<'EOF'\ncomplete\nEOF\n" },
};
await begin(handlers, block);
await delta(handlers, block, `{"command":"cat <<'EOF'\\n`);
await delta(handlers, block, `complete\\nEOF\\n"}`);
const accepted = await end(handlers, block);
if (accepted !== undefined) process.exit(10);
"""
    )
    assert result.returncode == 0, result.stderr


def test_dflash_startup_warmup_covers_multi_token_shape_and_fails_on_inference_jit() -> None:
    source = MAIN_LAUNCHER.read_text(encoding="utf-8")
    assert '"max_tokens":24' in source
    assert "Output exactly sixteen short words separated by spaces" in source
    assert "JIT compilation during inference" in source
    assert "refusing to start Pi" in source
    assert "tail -c +$((jit_log_offset + 1))" in source
    assert "repeating warmup" in source
    assert "poll_attempt <= 20" in source


def test_radiance_warmup_fails_closed_if_remote_jit_log_cannot_be_checked() -> None:
    source = RADIANCE_LAUNCHER.read_text(encoding="utf-8")
    assert '"max_tokens":96' in source
    assert "Output exactly sixty-four short words separated by spaces" in source
    assert "JIT compilation during inference" in source
    assert "case $jit_status in" in source
    assert "Radiance startup warmup could not inspect the remote backend log" in source
    assert 'podman logs --since "$since" "$container"' in source
    assert '[[ $jit_container_id =~ ^[0-9a-f]{64}$ ]]' in source
    assert "repeating warmup" in source
    assert "for poll_attempt in {1..20}" in source
    assert "sleep 0.1" in source
    assert 'if [[ ${QWEN_PI_SKIP_WARMUP:-0} != 1 ]]; then' in source
    assert "reused Radiance backend validated; non-session startup warmup complete" in source
    assert "skipping mutating startup warmup" not in source


def test_partial_json_is_rejected_before_execution_and_retry_is_bounded() -> None:
    result = run_harness(
        r"""
const handlers = instance();
const block = {
  type: "toolCall", id: "write-1", name: "write",
  arguments: { path: "/tmp/example", content: "partial" },
};
await begin(handlers, block);
await delta(handlers, block, '{"path":"/tmp/example","content":"partial');
const rejected = await end(handlers, block);
if (rejected?.message?.rawStopReason !== "tool_call_integrity_violation") process.exit(20);
if (rejected.message.content.length !== 0) process.exit(21);
if (!rejected.message.errorMessage.includes("server error")) process.exit(22);
const internalRetry = Symbol.for("qwen-r9700:internal-guard-retry:v1");
if (rejected.message[internalRetry] !== true) process.exit(30);

const recovery = await handlers.get("before_provider_request")({ payload }, ctx);
if (
  recovery.temperature !== 1 || recovery.top_p !== 0.95 || recovery.top_k !== 40
) process.exit(23);
const recoveryMarker = Symbol.for("qwen-r9700:ephemeral-recovery-request:v1");
if (recovery[recoveryMarker] !== true) process.exit(28);
if (JSON.stringify(recovery).includes("ephemeral-recovery")) process.exit(29);
if (recovery.messages === payload.messages || recovery.messages.length !== 2) process.exit(24);
if (!recovery.messages[1].content.includes("complete tool call from scratch")) process.exit(25);
await handlers.get("message_start")({ message: message([]) }, ctx);
const omitted = await handlers.get("message_end")({
  message: message([{ type: "text", text: "I will do that next." }], "stop"),
}, ctx);
if (omitted?.message?.stopReason !== "error") process.exit(26);
if (omitted.message.errorMessage.includes("server error")) process.exit(27);
if (omitted.message[internalRetry] !== undefined) process.exit(31);
"""
    )
    assert result.returncode == 0, result.stderr


def test_repaired_arguments_and_invalid_shell_syntax_are_rejected() -> None:
    result = run_harness(
        r"""
for (const [ordinal, raw, command] of [
  [0, `{"command":"printf ok"}`, "printf changed"],
  [1, `{"command":"printf 'oops"}`, "printf 'oops"],
  [2, `{"command":"cat <<'EOF'\\nbody"}`, "cat <<'EOF'\nbody"],
]) {
  const handlers = instance();
  const block = { type: "toolCall", id: `bash-${ordinal}`, name: "bash", arguments: { command } };
  await begin(handlers, block);
  await delta(handlers, block, raw);
  const rejected = await end(handlers, block);
  if (
    rejected?.message?.rawStopReason !== "tool_call_integrity_violation"
  ) process.exit(30 + ordinal);
}
"""
    )
    assert result.returncode == 0, result.stderr


def test_missing_stream_events_duplicates_and_provider_errors_fail_closed() -> None:
    result = run_harness(
        r"""
{
  const handlers = instance();
  const block = { type: "toolCall", id: "read-1", name: "read", arguments: { path: "/tmp/x" } };
  await handlers.get("before_provider_request")({ payload }, ctx);
  const rejected = await handlers.get("message_end")({ message: message([block]) }, ctx);
  if (rejected?.message?.stopReason !== "error") process.exit(40);
}
{
  const handlers = instance();
  const block = { type: "toolCall", id: "read-2", name: "read", arguments: { path: "/tmp/x" } };
  await begin(handlers, block);
  await handlers.get("message_update")({
    assistantMessageEvent: { type: "toolcall_start", contentIndex: 0, partial: message([block]) },
  }, ctx);
  await delta(handlers, block, '{"path":"/tmp/x"}');
  const rejected = await end(handlers, block);
  if (rejected?.message?.stopReason !== "error") process.exit(41);
}
{
  const handlers = instance();
  const block = {
    type: "toolCall", id: "bash-error", name: "bash", arguments: { command: "false" },
  };
  await handlers.get("before_provider_request")({ payload }, ctx);
  const providerError = await handlers.get("message_end")({
    message: message([block], "error"),
  }, ctx);
  if (
    providerError.message.stopReason !== "error" ||
    providerError.message.content.length !== 0
  ) process.exit(42);
}
"""
    )
    assert result.returncode == 0, result.stderr


def test_patched_provider_incomplete_json_error_gets_one_corrective_retry() -> None:
    result = run_harness(
        r"""
const handlers = instance();
await handlers.get("before_provider_request")({ payload }, ctx);
const coreError = message([
  { type: "toolCall", id: "bash-core", name: "bash", arguments: { command: "printf" } },
], "error");
coreError.errorMessage =
  "Provider tool-call integrity server error: the stream ended before " +
  "a complete JSON argument object was emitted";
const first = await handlers.get("message_end")({ message: coreError }, ctx);
if (first.message.content.length !== 0 || !first.message.errorMessage.includes("server error")) {
  process.exit(45);
}
const recovery = await handlers.get("before_provider_request")({ payload }, ctx);
if (!recovery.messages[1].content.includes("complete tool call from scratch")) process.exit(46);
const second = await handlers.get("message_end")({ message: coreError }, ctx);
if (second.message.errorMessage.includes("server error")) process.exit(47);
"""
    )
    assert result.returncode == 0, result.stderr


def test_non_target_requests_and_plain_answers_are_untouched() -> None:
    result = run_harness(
        r"""
const handlers = instance();
const other = await handlers.get("before_provider_request")({ payload }, {
  ...ctx, model: { provider: "other", api: "openai-completions" },
});
if (other !== undefined) process.exit(50);
await handlers.get("before_provider_request")({ payload }, ctx);
const plain = await handlers.get("message_end")({
  message: message([{ type: "text", text: "Complete answer." }], "stop"),
}, ctx);
if (plain !== undefined) process.exit(51);
"""
    )
    assert result.returncode == 0, result.stderr


def test_raw_incomplete_tool_prefix_is_rejected_when_parser_emits_no_call() -> None:
    result = run_harness(
        r"""
const handlers = instance();
const request = await handlers.get("before_provider_request")({ payload }, ctx);
if (request?.return_token_ids !== true) process.exit(60);
const hidden = message([{ type: "text", text: "Let me execute:" }], "stop");
hidden.usage.output = 4;
hidden[Symbol.for("qwen-r9700:raw-completion-token-ids:v1")] = [10, 248058, 20, 248046];
const rejected = await handlers.get("message_end")({ message: hidden }, ctx);
if (rejected?.message?.rawStopReason !== "tool_call_integrity_violation") process.exit(61);
if (!rejected.message.errorMessage.includes("raw token stream ended inside")) process.exit(62);
if (!rejected.message.errorMessage.includes("retrying once")) process.exit(63);
const recovery = await handlers.get("before_provider_request")({ payload }, ctx);
if (recovery?.return_token_ids !== true) process.exit(64);
if (!recovery.messages.at(-1).content.includes("complete tool call from scratch")) process.exit(65);
"""
    )
    assert result.returncode == 0, result.stderr


def test_both_launchers_load_raw_integrity_before_heuristic_and_snapshot_handlers() -> None:
    main = MAIN_LAUNCHER.read_text(encoding="utf-8")
    aggressive = AGGRESSIVE_LAUNCHER.read_text(encoding="utf-8")
    integrity = '--extension "$tool_call_integrity_extension"'

    assert main.count(integrity) == 1
    assert main.index(integrity) < main.index('--extension "$repetition_guard_extension"')
    assert main.index(integrity) < main.index('--extension "$fixed_slot_resume_extension"')
    assert aggressive.count(integrity) == 1
    assert aggressive.index(integrity) < aggressive.index(
        '--extension "$repetition_guard_extension"'
    )
    assert aggressive.index(integrity) < aggressive.index(
        '--extension "$tool_turn_rehydrate_extension"'
    )
