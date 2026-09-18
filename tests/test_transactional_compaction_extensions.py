from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PI = ROOT / "integrations/pi"
NO_TOOLS = PI / "qwen-semantic-summary-no-tools.mjs"
TOKEN_JOURNAL = PI / "qwen-semantic-summary-token-journal.mjs"
SUCCESSOR_EXPORT = PI / "qwen-compaction-successor-payload-export.mjs"
TRANSACTIONAL = PI / "qwen-transactional-compaction.mjs"
HANDOFF = PI / "qwen-transactional-compaction-handoff.mjs"
COMPACTION_ABI_SHA256 = "a" * 64
PROMPT_ABI = "searchtool-toolarchive-v1"
OUTCOME_PROMPT_ABI = "searchtool-toolarchive-outcome-v1"
TRANSACTION_KEY_DOMAIN = b"qwen-r9700-transactional-compaction-key-v3\0"


def compaction_transaction_key(source_sha256: str, prompt_abi: str = PROMPT_ABI) -> str:
    return hashlib.sha256(
        TRANSACTION_KEY_DOMAIN
        + source_sha256.encode("ascii")
        + COMPACTION_ABI_SHA256.encode("ascii")
        + prompt_abi.encode("ascii")
    ).hexdigest()


def run_node(
    source: str, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module", "--eval", source],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_semantic_summary_disables_tools_without_changing_prompt_schema() -> None:
    harness = f"""
import noTools from {json.dumps(NO_TOOLS.as_uri())};
const handlers = new Map();
noTools({{ on(name, handler) {{ handlers.set(name, handler); }} }});
const ctx = {{ model: {{
  provider: "qwen-r9700", api: "openai-completions",
}} }};
const messages = [{{ role: "user", content: "summarize" }}];
const tools = [{{ type: "function", function: {{ name: "read" }} }}];
const payload = {{
  model: "qwen3.8-27b-frozenlock", max_tokens: 16384, messages, tools,
  chat_template_kwargs: {{
    enable_thinking: true, preserve_thinking: true, reasoning_effort: "xhigh",
  }},
  thinking_token_budget: 12345,
}};
const result = await handlers.get("before_provider_request")({{ payload }}, ctx);
if (result.tool_choice !== "none") process.exit(2);
if (result.max_tokens !== 1024) process.exit(7);
if (result.thinking_token_budget !== 768) process.exit(10);
if (result.return_token_ids !== true || result.include_reasoning !== true) process.exit(11);
if (result.stop !== undefined) process.exit(8);
if (result.messages !== messages || result.tools !== tools) process.exit(3);
if (payload.tool_choice !== undefined) process.exit(4);
const other = await handlers.get("before_provider_request")(
  {{ payload }}, {{ model: {{ provider: "other", api: "openai-completions" }} }},
);
if (other !== undefined) process.exit(5);
let rejected = false;
try {{
  await handlers.get("before_provider_request")(
    {{ payload: {{ ...payload, tool_choice: "auto" }} }}, ctx,
  );
}} catch {{ rejected = true; }}
if (!rejected) process.exit(6);
rejected = false;
try {{
  await handlers.get("before_provider_request")(
    {{ payload: {{ ...payload, stop: ["legacy-stop"] }} }}, ctx,
  );
}} catch {{ rejected = true; }}
if (!rejected) process.exit(9);
"""
    environment = os.environ.copy()
    environment["QWEN_PI_SEMANTIC_SUMMARY_MAX_TOKENS"] = "1024"
    environment["QWEN_PI_SEMANTIC_SUMMARY_PROMPT_THINKING"] = "xhigh"
    environment["QWEN_PI_SEMANTIC_SUMMARY_THINKING_TOKEN_BUDGET"] = "768"
    environment["QWEN_PI_SEMANTIC_SUMMARY_TOOL_NAMES"] = '["read"]'
    result = run_node(harness, environment)
    assert result.returncode == 0, result.stderr


def test_semantic_summary_requires_a_bounded_provider_limit() -> None:
    harness = f"""
import noTools from {json.dumps(NO_TOOLS.as_uri())};
noTools({{ on() {{}} }});
"""
    missing = os.environ.copy()
    missing.pop("QWEN_PI_SEMANTIC_SUMMARY_MAX_TOKENS", None)
    missing["QWEN_PI_SEMANTIC_SUMMARY_PROMPT_THINKING"] = "xhigh"
    missing["QWEN_PI_SEMANTIC_SUMMARY_THINKING_TOKEN_BUDGET"] = "2048"
    missing["QWEN_PI_SEMANTIC_SUMMARY_TOOL_NAMES"] = '["read"]'
    missing_result = run_node(harness, missing)
    assert missing_result.returncode != 0
    assert "is missing or outside 500..8000" in missing_result.stderr

    invalid = os.environ.copy()
    invalid["QWEN_PI_SEMANTIC_SUMMARY_MAX_TOKENS"] = "8001"
    invalid["QWEN_PI_SEMANTIC_SUMMARY_PROMPT_THINKING"] = "xhigh"
    invalid["QWEN_PI_SEMANTIC_SUMMARY_THINKING_TOKEN_BUDGET"] = "2048"
    invalid["QWEN_PI_SEMANTIC_SUMMARY_TOOL_NAMES"] = '["read"]'
    invalid_result = run_node(harness, invalid)
    assert invalid_result.returncode != 0
    assert "is missing or outside 500..8000" in invalid_result.stderr

    maximum = os.environ.copy()
    maximum["QWEN_PI_SEMANTIC_SUMMARY_MAX_TOKENS"] = "8000"
    maximum["QWEN_PI_SEMANTIC_SUMMARY_PROMPT_THINKING"] = "xhigh"
    maximum["QWEN_PI_SEMANTIC_SUMMARY_THINKING_TOKEN_BUDGET"] = "2048"
    maximum["QWEN_PI_SEMANTIC_SUMMARY_TOOL_NAMES"] = '["read"]'
    maximum_result = run_node(harness, maximum)
    assert maximum_result.returncode == 0, maximum_result.stderr


def test_semantic_summary_rejects_prompt_abi_or_budget_drift() -> None:
    harness = f"""
import noTools from {json.dumps(NO_TOOLS.as_uri())};
const handlers = new Map();
noTools({{ on(name, handler) {{ handlers.set(name, handler); }} }});
const ctx = {{ model: {{ provider: "qwen-r9700", api: "openai-completions" }} }};
const base = {{
  model: "qwen3.8-27b-frozenlock",
  chat_template_kwargs: {{
    enable_thinking: true, preserve_thinking: true, reasoning_effort: "low",
  }},
}};
let rejected = 0;
try {{ await handlers.get("before_provider_request")({{ payload: base }}, ctx); }}
catch {{ rejected += 1; }}
if (rejected !== 1) process.exit(2);
"""
    environment = os.environ.copy()
    environment["QWEN_PI_SEMANTIC_SUMMARY_MAX_TOKENS"] = "8000"
    environment["QWEN_PI_SEMANTIC_SUMMARY_PROMPT_THINKING"] = "xhigh"
    environment["QWEN_PI_SEMANTIC_SUMMARY_THINKING_TOKEN_BUDGET"] = "2048"
    environment["QWEN_PI_SEMANTIC_SUMMARY_TOOL_NAMES"] = '["read"]'
    mismatch = run_node(harness, environment)
    assert mismatch.returncode == 0, mismatch.stderr

    environment["QWEN_PI_SEMANTIC_SUMMARY_THINKING_TOKEN_BUDGET"] = "8000"
    invalid_budget = run_node(
        "import noTools from " + json.dumps(NO_TOOLS.as_uri()) + "; noTools({on(){}});",
        environment,
    )
    assert invalid_budget.returncode != 0
    assert "outside 0..max_tokens-1" in invalid_budget.stderr


def test_semantic_summary_token_journal_seals_exact_server_tokens(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    journal = tmp_path / "attempt-0000.tokens.jsonl"
    sse = "".join(
        (
            'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":'
            '{"reasoning_content":"think"},"finish_reason":null,"token_ids":[10,11]}]}\n\n',
            'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":'
            '{"content":"## Goal\\nKeep it.\\nCOMPACTION_SUMMARY_COMPLETE"},'
            '"finish_reason":"stop","token_ids":[12,13]}]}\n\n',
            'data: {"id":"chatcmpl-test","choices":[],"usage":{"completion_tokens":4}}\n\n',
            "data: [DONE]\n\n",
        )
    )
    harness = f"""
const source = {json.dumps(sse)};
globalThis.fetch = async () => new Response(new ReadableStream({{
  start(controller) {{
    const bytes = new TextEncoder().encode(source);
    controller.enqueue(bytes.slice(0, 37));
    controller.enqueue(bytes.slice(37, 101));
    controller.enqueue(bytes.slice(101));
    controller.close();
  }},
}}), {{ status: 200, headers: {{ "content-type": "text/event-stream" }} }});
const journalExtension = (await import({json.dumps(TOKEN_JOURNAL.as_uri())})).default;
journalExtension();
const body = JSON.stringify({{
  model: "qwen3.8-27b-frozenlock", stream: true, return_token_ids: true,
}});
const request = new Request("http://127.0.0.1:8000/v1/chat/completions", {{
  method: "POST", body,
}});
const response = await fetch(request);
if (await response.text() !== source) process.exit(2);
"""
    environment = os.environ.copy()
    environment["QWEN_PI_SEMANTIC_SUMMARY_TOKEN_JOURNAL"] = str(journal)
    result = run_node(harness, environment)
    assert result.returncode == 0, result.stderr
    assert stat.S_IMODE(journal.stat().st_mode) == 0o400
    assert journal.stat().st_nlink == 1
    assert not journal.with_name(journal.name + ".partial").exists()
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [record["type"] for record in records] == ["header", "batch", "complete"]
    assert records[1]["events"][0]["token_ids"] == [10, 11]
    assert records[1]["events"][1]["token_ids"] == [12, 13]
    terminal = records[-1]
    encoded = b"qwen-r9700-token-ids-u32be-v1\0" + b"".join(
        token.to_bytes(4, "big") for token in (10, 11, 12, 13)
    )
    assert terminal["token_ids_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert (
        terminal["content_sha256"]
        == hashlib.sha256(b"## Goal\nKeep it.\nCOMPACTION_SUMMARY_COMPLETE").hexdigest()
    )
    assert terminal["reasoning_sha256"] == hashlib.sha256(b"think").hexdigest()
    assert terminal["response_id"] == "chatcmpl-test"
    assert terminal["output_tokens"] == 4


def test_semantic_summary_token_journal_preserves_interrupted_partial(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    journal = tmp_path / "attempt-0000.tokens.jsonl"
    harness = f"""
const source = 'data: {{"id":"chatcmpl-interrupted","choices":[{{"index":0,' +
  '"delta":{{"content":"partial"}},"finish_reason":null,"token_ids":[7]}}]}}\\n\\n';
globalThis.fetch = async () => new Response(source, {{
  status: 200, headers: {{ "content-type": "text/event-stream" }},
}});
const journalExtension = (await import({json.dumps(TOKEN_JOURNAL.as_uri())})).default;
journalExtension();
const body = JSON.stringify({{
  model: "qwen3.8-27b-frozenlock", stream: true, return_token_ids: true,
}});
const response = await fetch("http://127.0.0.1:8000/v1/chat/completions", {{
  method: "POST", body,
}});
let rejected = false;
try {{ await response.text(); }} catch {{ rejected = true; }}
if (!rejected) process.exit(2);
"""
    environment = os.environ.copy()
    environment["QWEN_PI_SEMANTIC_SUMMARY_TOKEN_JOURNAL"] = str(journal)
    result = run_node(harness, environment)
    assert result.returncode == 0, result.stderr
    assert not journal.exists()
    partial = journal.with_name(journal.name + ".partial")
    assert stat.S_IMODE(partial.stat().st_mode) == 0o400
    assert [json.loads(line)["type"] for line in partial.read_text().splitlines()] == ["header"]


def test_successor_payload_export_uses_the_real_prompt_and_hides_control(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    output = tmp_path / "successor-payload.json"
    placeholder = "QWEN-COMPACTION-SUCCESSOR-PLACEHOLDER-" + "a" * 64
    harness = f"""
import successorExport from {json.dumps(SUCCESSOR_EXPORT.as_uri())};
const handlers = new Map();
let sent;
successorExport({{
  on(name, handler) {{ handlers.set(name, handler); }},
  sendMessage(message, options) {{ sent = {{ message, options }}; }},
}});
handlers.get("session_start")();
if (sent.message.display !== false || sent.options.triggerTurn !== true) process.exit(2);
const filtered = handlers.get("context")({{
  messages: [sent.message, {{ role: "user", content: "retained" }}],
}});
if (filtered.messages.length !== 1 || filtered.messages[0].content !== "retained") process.exit(3);
let aborts = 0;
let shutdowns = 0;
const payload = {{
  model: "qwen3.8-27b-frozenlock",
  messages: [
    {{ role: "system", content: "system" }},
    {{ role: "system", content: {json.dumps(placeholder)} }},
    {{ role: "user", content: "recent exact tail" }},
  ],
  tools: [{{ type: "function", function: {{ name: "read" }} }}],
  temperature: 0,
}};
const replacement = handlers.get("before_provider_request")({{ payload }}, {{
  abort() {{ aborts += 1; }}, shutdown() {{ shutdowns += 1; }},
}});
if (aborts !== 1 || shutdowns !== 1 || replacement.messages[0].content !== ".") process.exit(4);
"""
    environment = os.environ.copy()
    environment.update(
        {
            "QWEN_PI_COMPACTION_SUCCESSOR_PAYLOAD_EXPORT": str(output),
            "QWEN_PI_COMPACTION_SUCCESSOR_PLACEHOLDER": placeholder,
        }
    )
    result = run_node(harness, environment)
    assert result.returncode == 0, result.stderr
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    document = json.loads(output.read_text())
    assert document["schema"] == "urn:qwen-r9700:compaction-successor-payload-template:v1"
    assert document["payload"]["messages"][1]["content"] == placeholder
    assert document["payload"]["messages"][2]["content"] == "recent exact tail"
    assert "temperature" not in document["payload"]


def summary_environment(
    path: Path,
    payload: bytes,
    *,
    read_files: list[str] | None = None,
    modified_files: list[str] | None = None,
) -> dict[str, str]:
    file_operations = path.with_name("file-operations.json")
    file_operations_payload = (
        json.dumps(
            {
                "modifiedFiles": sorted(modified_files or []),
                "readFiles": sorted(read_files or []),
                "schema": "urn:qwen-r9700:deterministic-file-operations:v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    if file_operations.exists():
        assert file_operations.read_bytes() == file_operations_payload
    else:
        file_operations.write_bytes(file_operations_payload)
        file_operations.chmod(0o400)
    environment = os.environ.copy()
    environment.update(
        {
            "QWEN_PI_TRANSACTIONAL_COMPACTION_SUMMARY": str(path),
            "QWEN_PI_TRANSACTIONAL_COMPACTION_SUMMARY_BYTES": str(len(payload)),
            "QWEN_PI_TRANSACTIONAL_COMPACTION_SUMMARY_SHA256": hashlib.sha256(payload).hexdigest(),
            "QWEN_PI_TRANSACTIONAL_COMPACTION_FILE_OPERATIONS": str(file_operations),
            "QWEN_PI_TRANSACTIONAL_COMPACTION_FILE_OPERATIONS_BYTES": str(
                len(file_operations_payload)
            ),
            "QWEN_PI_TRANSACTIONAL_COMPACTION_FILE_OPERATIONS_SHA256": hashlib.sha256(
                file_operations_payload
            ).hexdigest(),
        }
    )
    return environment


def test_transactional_compaction_uses_one_authenticated_summary(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    summary = tmp_path / "summary.txt"
    payload = b"## Goal\nPreserve exact evidence.\n"
    expected_sha = hashlib.sha256(payload).hexdigest()
    summary.write_bytes(payload)
    summary.chmod(0o400)
    harness = f"""
import compact from {json.dumps(TRANSACTIONAL.as_uri())};
const handlers = new Map();
compact({{ on(name, handler) {{ handlers.set(name, handler); }} }});
const handler = handlers.get("session_before_compact");
let wrongRejected = false;
try {{
  await handler({{
    reason: "auto", willRetry: false,
    preparation: {{ firstKeptEntryId: "entry-1", tokensBefore: 1000 }},
  }});
}} catch {{ wrongRejected = true; }}
if (!wrongRejected) process.exit(2);
const result = await handler({{
  reason: "manual", willRetry: false,
  preparation: {{ firstKeptEntryId: "entry-1", tokensBefore: 1000 }},
}});
if (result.compaction.summary !== {json.dumps(payload.decode())}) process.exit(3);
if (result.compaction.firstKeptEntryId !== "entry-1") process.exit(4);
if (result.compaction.tokensBefore !== 1000) process.exit(5);
if (result.compaction.details.strategy !== "snapshot-prefix-continuation") process.exit(6);
if (result.compaction.details.summary_sha256 !== {json.dumps(expected_sha)}) process.exit(7);
const observedRead = JSON.stringify(result.compaction.details.readFiles);
const observedModified = JSON.stringify(result.compaction.details.modifiedFiles);
if (observedRead !== JSON.stringify(["read-only.txt"])) process.exit(9);
if (observedModified !== JSON.stringify(["changed.txt"])) process.exit(10);
let reuseRejected = false;
try {{ await handler({{
  reason: "manual", willRetry: false,
  preparation: {{ firstKeptEntryId: "entry-1", tokensBefore: 1000 }},
}}); }} catch {{ reuseRejected = true; }}
if (!reuseRejected) process.exit(8);
"""
    result = run_node(
        harness,
        summary_environment(
            summary,
            payload,
            read_files=["read-only.txt"],
            modified_files=["changed.txt"],
        ),
    )
    assert result.returncode == 0, result.stderr


def test_transactional_compaction_rejects_tamper_and_invalid_utf8(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    summary = tmp_path / "summary.txt"
    payload = b"valid summary\n"
    summary.write_bytes(payload)
    summary.chmod(0o400)
    harness = f"""
import compact from {json.dumps(TRANSACTIONAL.as_uri())};
compact({{ on() {{}} }});
"""
    environment = summary_environment(summary, payload)
    environment["QWEN_PI_TRANSACTIONAL_COMPACTION_SUMMARY_SHA256"] = "0" * 64
    mismatch = run_node(harness, environment)
    assert mismatch.returncode != 0
    assert "digest differs" in mismatch.stderr

    invalid = b"\xff\xfe"
    summary.chmod(0o600)
    summary.write_bytes(invalid)
    summary.chmod(0o400)
    invalid_result = run_node(harness, summary_environment(summary, invalid))
    assert invalid_result.returncode != 0
    assert "not valid UTF-8" in invalid_result.stderr


def handoff_environment(
    source: Path,
    requests: Path,
    transactions: Path,
    logical_id: str,
    prompt_abi: str = PROMPT_ABI,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "QWEN_PI_COMPACTION_HANDOFF_ABI_SHA256": COMPACTION_ABI_SHA256,
            "QWEN_PI_COMPACTION_HANDOFF_LOGICAL_SESSION_ID": logical_id,
            "QWEN_PI_COMPACTION_HANDOFF_PROMPT_ABI": prompt_abi,
            "QWEN_PI_COMPACTION_HANDOFF_REQUEST_ROOT": str(requests),
            "QWEN_PI_COMPACTION_HANDOFF_SOURCE": str(source),
            "QWEN_PI_COMPACTION_HANDOFF_TRANSACTION_ROOT": str(transactions),
        }
    )
    return environment


def test_compaction_handoff_fsyncs_request_before_shutdown(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    requests = tmp_path / "requests"
    transactions = tmp_path / "transactions"
    requests.mkdir(mode=0o700)
    transactions.mkdir(mode=0o700)
    source = tmp_path / "session.jsonl"
    source.write_text('{"type":"session","id":"session-one"}\n', encoding="utf-8")
    source.chmod(0o600)
    harness = f"""
import handoff from {json.dumps(HANDOFF.as_uri())};
const handlers = new Map();
let shutdowns = 0;
handoff({{ on(name, handler) {{ handlers.set(name, handler); }}, sendMessage() {{}} }});
const result = await handlers.get("session_before_compact")({{
  reason: "overflow", willRetry: true, customInstructions: "retain exact evidence",
  preparation: {{ firstKeptEntryId: "entry-1", tokensBefore: 200000 }},
  branchEntries: [{{ id: "user-entry" }}],
}}, {{
  sessionManager: {{ getSessionFile() {{ return {json.dumps(str(source))}; }} }},
  shutdown() {{ shutdowns += 1; }},
}});
if (result.cancel !== true || shutdowns !== 1) process.exit(2);
"""
    result = run_node(harness, handoff_environment(source, requests, transactions, "session-one"))
    assert result.returncode == 0, result.stderr
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    transaction_key = compaction_transaction_key(source_sha)
    request = requests / transaction_key / "request.json"
    assert stat.S_IMODE(request.stat().st_mode) == 0o600
    document = json.loads(request.read_text())
    assert document["will_retry"] is True
    assert document["reason"] == "overflow"
    assert document["custom_instructions"] == "retain exact evidence"
    assert document["source"]["sha256"] == source_sha
    assert document["compaction_abi_sha256"] == COMPACTION_ABI_SHA256
    assert document["prompt_abi"] == PROMPT_ABI
    assert document["schema"] == "urn:qwen-r9700:transactional-compaction-handoff:v3"
    assert document["output_root"] == str(transactions / transaction_key)


def test_compaction_handoff_requires_an_authenticated_source_abi(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    requests = tmp_path / "requests"
    transactions = tmp_path / "transactions"
    source = tmp_path / "session.jsonl"
    requests.mkdir(mode=0o700)
    transactions.mkdir(mode=0o700)
    source.write_text('{"type":"session","id":"session-one"}\n', encoding="utf-8")
    source.chmod(0o600)
    environment = handoff_environment(source, requests, transactions, "session-one")
    environment.pop("QWEN_PI_COMPACTION_HANDOFF_ABI_SHA256")
    result = run_node(
        f"import handoff from {json.dumps(HANDOFF.as_uri())}; handoff({{ on() {{}} }});",
        environment,
    )
    assert result.returncode != 0
    assert "COMPACTION_HANDOFF_ABI_SHA256 is missing or invalid" in result.stderr


def test_compaction_handoff_requires_and_isolates_prompt_abi(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    requests = tmp_path / "requests"
    transactions = tmp_path / "transactions"
    source = tmp_path / "session.jsonl"
    requests.mkdir(mode=0o700)
    transactions.mkdir(mode=0o700)
    source.write_text('{"type":"session","id":"session-one"}\n', encoding="utf-8")
    source.chmod(0o600)
    environment = handoff_environment(source, requests, transactions, "session-one")
    environment.pop("QWEN_PI_COMPACTION_HANDOFF_PROMPT_ABI")
    result = run_node(
        f"import handoff from {json.dumps(HANDOFF.as_uri())}; handoff({{ on() {{}} }});",
        environment,
    )
    assert result.returncode != 0
    assert "COMPACTION_HANDOFF_PROMPT_ABI is missing or invalid" in result.stderr

    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert compaction_transaction_key(source_sha) != compaction_transaction_key(
        source_sha, OUTCOME_PROMPT_ABI
    )


def test_overflow_resume_control_is_invisible_and_submitted_once(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    requests = tmp_path / "requests"
    transactions = tmp_path / "transactions"
    request_root = requests / ("a" * 64)
    request_root.mkdir(parents=True, mode=0o700)
    requests.chmod(0o700)
    transactions.mkdir(mode=0o700)
    source = tmp_path / "session.jsonl"
    source.write_text('{"type":"session","id":"session-one"}\n', encoding="utf-8")
    source.chmod(0o600)
    request = request_root / "request.json"
    request_payload = b'{"will_retry":true}\n'
    request.write_bytes(request_payload)
    request.chmod(0o600)
    submitted = request_root / "resume-submitted.json"
    environment = handoff_environment(source, requests, transactions, "session-one")
    environment.update(
        {
            "QWEN_PI_COMPACTION_RESUME_REQUEST": str(request),
            "QWEN_PI_COMPACTION_RESUME_REQUEST_SHA256": hashlib.sha256(request_payload).hexdigest(),
            "QWEN_PI_COMPACTION_RESUME_SUBMITTED": str(submitted),
        }
    )
    harness = f"""
import handoff from {json.dumps(HANDOFF.as_uri())};
const handlers = new Map();
let sent;
handoff({{
  on(name, handler) {{ handlers.set(name, handler); }},
  sendMessage(message, options) {{ sent = {{ message, options }}; }},
}});
await handlers.get("session_start")({{}}, {{}});
if (sent.message.display !== false || sent.options.triggerTurn !== true) process.exit(2);
const contextEvent = {{ messages: [sent.message, {{ role: "user" }}] }};
const context = await handlers.get("context")(contextEvent, {{}});
if (context.messages.length !== 1 || context.messages[0].role !== "user") process.exit(3);
await handlers.get("before_provider_request")({{ payload: {{}} }}, {{}});
await handlers.get("before_provider_request")({{ payload: {{}} }}, {{}});
"""
    result = run_node(harness, environment)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(submitted.read_text())
    assert evidence == {
        "request_sha256": hashlib.sha256(request_payload).hexdigest(),
        "schema": "urn:qwen-r9700:transactional-compaction-resume-submitted:v1",
    }


def test_installed_pi_hands_off_manual_compaction_before_appending(tmp_path: Path) -> None:
    pi = Path("/home/lewis/.local/bin/pi")
    if not pi.is_file():
        return
    tmp_path.chmod(0o700)
    logical_id = str(uuid.uuid4())
    source = tmp_path / "session.jsonl"
    records = [
        {
            "type": "session",
            "id": logical_id,
            "version": 3,
            "cwd": str(ROOT),
        },
        {
            "type": "message",
            "id": "user-entry",
            "parentId": None,
            "timestamp": "2026-09-02T00:00:00.000Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "old context " * 50_000}],
            },
        },
        {
            "type": "message",
            "id": "assistant-entry",
            "parentId": "user-entry",
            "timestamp": "2026-09-02T00:00:01.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "completed"}],
                "provider": "qwen-r9700",
                "model": "qwen3.8-27b-frozenlock",
                "usage": {"input": 100_000, "output": 10},
                "stopReason": "stop",
            },
        },
        {
            "type": "message",
            "id": "recent-user",
            "parentId": "assistant-entry",
            "timestamp": "2026-09-02T00:00:02.000Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "recent context " * 30_000}],
            },
        },
        {
            "type": "message",
            "id": "recent-assistant",
            "parentId": "recent-user",
            "timestamp": "2026-09-02T00:00:03.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "recent response"}],
                "provider": "qwen-r9700",
                "model": "qwen3.8-27b-frozenlock",
                "usage": {"input": 180_000, "output": 10},
                "stopReason": "stop",
            },
        },
        {
            "type": "thinking_level_change",
            "id": "thinking-entry",
            "parentId": "recent-assistant",
            "timestamp": "2026-09-02T00:00:04.000Z",
            "thinkingLevel": "medium",
        },
    ]
    source_payload = b"".join(
        json.dumps(record, separators=(",", ":")).encode() + b"\n" for record in records
    )
    source.write_bytes(source_payload)
    source.chmod(0o600)
    requests = tmp_path / "requests"
    transactions = tmp_path / "transactions"
    agent = tmp_path / "agent"
    for directory in (requests, transactions, agent):
        directory.mkdir(mode=0o700)
    models = json.loads((ROOT / "integrations/pi/models.json").read_text())
    models["providers"]["qwen-r9700"]["baseUrl"] = "http://127.0.0.1:9/v1"
    (agent / "models.json").write_text(json.dumps(models), encoding="utf-8")
    (agent / "models.json").chmod(0o600)
    (agent / "settings.json").write_text('{"httpIdleTimeoutMs":0}\n', encoding="utf-8")
    (agent / "settings.json").chmod(0o600)
    environment = handoff_environment(source, requests, transactions, logical_id)
    environment.update({"PI_CODING_AGENT_DIR": str(agent), "PI_OFFLINE": "1"})
    before = source.read_bytes()
    process = subprocess.Popen(
        [
            str(pi),
            "--model",
            "qwen-r9700/qwen3.8-27b-frozenlock",
            "--no-extensions",
            "--extension",
            str(HANDOFF),
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--mode",
            "rpc",
            "--session",
            str(source),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert process.stdin is not None
    process.stdin.write('{"id":"compact","type":"compact"}\n')
    process.stdin.flush()
    deadline = time.monotonic() + 10
    request = None
    while time.monotonic() < deadline:
        candidates = list(requests.glob("*/request.json"))
        if len(candidates) == 1:
            request = candidates[0]
            break
        if process.poll() is not None:
            break
        time.sleep(0.02)
    stdout, stderr = process.communicate(timeout=10)
    assert request is not None, f"stdout={stdout}\nstderr={stderr}"
    assert source.read_bytes() == before
    assert request.parent.name == compaction_transaction_key(hashlib.sha256(before).hexdigest())
    assert "durable manual compaction handoff recorded" in stderr
