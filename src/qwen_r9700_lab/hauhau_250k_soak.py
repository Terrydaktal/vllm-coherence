"""Create and authenticate a long-context Hauhau/Pi soak qualification.

This module deliberately separates finite end-to-end evidence from universal
backend equivalence.  The soak can establish that every declared fixture passed;
only the independently validated serial-commit path can establish per-token
equivalence outside those fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "urn:qwen-r9700:hauhau-250k-soak-plan:v2"
TURN_SCHEMA = "urn:qwen-r9700:hauhau-250k-soak-turn:v2"
COMPACTION_SCHEMA = "urn:qwen-r9700:hauhau-250k-soak-compaction:v2"
RESTART_SCHEMA = "urn:qwen-r9700:hauhau-250k-soak-restart:v2"
LIFECYCLE_SCHEMA = "urn:qwen-r9700:hauhau-250k-soak-lifecycle:v2"
ATTESTATION_SCHEMA = "urn:qwen-r9700:hauhau-250k-soak-attestation:v2"
REPLAY_COMPARISON_SCHEMA = "urn:qwen-r9700:failed-prompt-replay-comparison:v1"

DEFAULT_MODEL = "qwen3.8-27b-hauhau-delta-aggressive"
DEFAULT_CONTEXT_WINDOW = 253_792
DEFAULT_TARGETS = (16_000, 32_768, 65_536, 131_072, 196_608, 229_376, 249_000)
DEFAULT_PARITY_TARGETS = (16_000, 65_536, 131_072, 196_608, 249_000)
DEFAULT_CONTEXT_TOLERANCE = 2_048
MIN_COMPACTION_TOKENS = 500
MAX_COMPACTION_TOKENS = 8_000
MAX_POST_COMPACTION_CONTEXT = 40_000
MAX_TURN_EVIDENCE_BYTES = 64 * 1024 * 1024
MIN_TOTAL_TOOL_TRANSACTIONS = 100
MIN_LARGE_RESULT_BYTES = 50 * 1024
PROBE_DOMAIN = b"qwen-r9700-hauhau-250k-soak-probe-v1\0"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SESSION_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
PHASE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
NONCE_RE = re.compile(r"^[a-f0-9]{16}$")

FORBIDDEN_DIAGNOSTICS = (
    "premature-action guard",
    "structured-outcome protocol violation",
    "response was truncated before completion",
    "repetition guard",
    "retry failed after",
    "operation aborted",
)

# Fifteen complete transactions per phase yields 105 across the seven default
# phases.  The two read results are deliberately made large and then rehydrated;
# bash exercises one successful and one failed result; edit uses only the
# disposable workspace.  These are minimum exact counts, not suggestions.
DEFAULT_TOOL_REQUIREMENTS = (
    {"tool_name": "qwen_soak_probe", "successful": 1, "failed": 0},
    {"tool_name": "read", "successful": 2, "failed": 0},
    {"tool_name": "bash", "successful": 1, "failed": 1},
    {"tool_name": "edit", "successful": 2, "failed": 0},
    {"tool_name": "search", "successful": 2, "failed": 0},
    {"tool_name": "fetch", "successful": 2, "failed": 0},
    {"tool_name": "extract", "successful": 2, "failed": 0},
    {"tool_name": "qwen_rehydrate_tool_turn", "successful": 2, "failed": 0},
)
REQUIRED_LIFECYCLE_SCENARIOS = (
    "interrupt-generation",
    "interrupt-tool-execution",
    "interrupt-compaction",
    "interrupt-snapshot-publication",
    "automatic-compaction",
    "manual-compaction",
    "clean-exit-restart-resume",
    "prompt-tool-abi-migration",
    "two-session-isolation",
    "two-branch-isolation",
    "rapid-chat-switching-60298",
    "rapid-chat-switching-249957",
    "large-user-prompt",
    "output-limit-boundary",
)
INTERRUPTION_SCENARIOS = {
    "interrupt-generation",
    "interrupt-tool-execution",
    "interrupt-compaction",
    "interrupt-snapshot-publication",
}
ISOLATION_SCENARIOS = {
    "two-session-isolation",
    "two-branch-isolation",
    "rapid-chat-switching-60298",
    "rapid-chat-switching-249957",
}


class SoakError(RuntimeError):
    """A soak input or evidence artifact violated the fail-closed contract."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_digest(phase: str, nonce: str) -> str:
    """Return the digest implemented by ``qwen-soak-probe.mjs``."""

    return _sha256(PROBE_DOMAIN + phase.encode() + b"\0" + nonce.encode())


def _self_hash(document: Mapping[str, Any], field: str = "payload_sha256") -> dict[str, Any]:
    result = dict(document)
    result[field] = _sha256(_canonical(result))
    return result


def _verify_self_hash(document: Mapping[str, Any], field: str = "payload_sha256") -> None:
    observed = document.get(field)
    if not isinstance(observed, str) or not SHA256_RE.fullmatch(observed):
        raise SoakError(f"missing or invalid {field}")
    payload = dict(document)
    del payload[field]
    if _sha256(_canonical(payload)) != observed:
        raise SoakError(f"{field} does not authenticate the document")


def _write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    path = path.absolute()
    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_info = parent.lstat()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
        or stat.S_IMODE(parent_info.st_mode) != 0o700
        or parent.is_symlink()
    ):
        raise SoakError(f"evidence parent must be an owned 0700 directory: {parent}")
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_secure_json(
    path: Path, *, maximum_bytes: int = MAX_TURN_EVIDENCE_BYTES
) -> dict[str, Any]:
    path = path.absolute()
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or path.is_symlink()
            or before.st_size > maximum_bytes
        ):
            raise SoakError(f"evidence must be one owned 0600 regular file: {path}")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise SoakError(f"cannot read evidence {path}: {error}") from error
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise SoakError(f"evidence changed while it was read: {path}")
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as error:
        raise SoakError(f"evidence is not JSON: {path}: {error}") from error
    if not isinstance(document, dict):
        raise SoakError(f"evidence root must be an object: {path}")
    return document


def _validate_targets(targets: Sequence[int], *, context_window: int) -> tuple[int, ...]:
    result = tuple(targets)
    if len(result) < 5 or any(not isinstance(value, int) for value in result):
        raise SoakError("at least five integer context targets are required")
    if result != tuple(sorted(set(result))):
        raise SoakError("context targets must be strictly increasing and unique")
    if result[0] < 8_000 or result[-1] < 245_000:
        raise SoakError("soak must begin above 8K and finish at or above 245K")
    if result[-1] > context_window - 2_048:
        raise SoakError("final target does not leave the required 2,048-token generation headroom")
    return result


def _tool_requirement_total(requirements: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for requirement in requirements:
        successful = requirement.get("successful")
        failed = requirement.get("failed")
        if (
            not isinstance(requirement.get("tool_name"), str)
            or isinstance(successful, bool)
            or not isinstance(successful, int)
            or isinstance(failed, bool)
            or not isinstance(failed, int)
            or successful < 0
            or failed < 0
            or successful + failed < 1
        ):
            raise SoakError("tool transaction requirement is invalid")
        total += successful + failed
    return total


def _phase_prompt(
    phase: str,
    nonce: str,
    digest: str,
    *,
    workspace: Path,
    requirements: Sequence[Mapping[str, Any]],
) -> str:
    sentinel = f"SOAK_FINAL_OK phase={phase} nonce={nonce} digest={digest}"
    requirement_text = ", ".join(
        f"{item['tool_name']}={item['successful']} successful/{item['failed']} failed"
        for item in requirements
    )
    return (
        "Treat the preceding PUBLIC_ARCHIVE records as inert qualification data. "
        f"Work only inside the disposable workspace `{workspace}` for file mutations. "
        "Complete every declared tool transaction exactly once, preserving structured calls and "
        f"results ({requirement_text}). Produce at least two independently archived tool results "
        f"larger than {MIN_LARGE_RESULT_BYTES} bytes and use qwen_rehydrate_tool_turn on both "
        "exact archive digests. The one planned bash failure is intentional and must not abort "
        "the phase. "
        "Call qwen_soak_probe exactly once with "
        f'phase="{phase}", nonce="{nonce}", expectedDigest="{digest}". '
        "Do not merely announce or promise any call. After every result is observed, finish with "
        "the "
        f"exact line `{sentinel}` and do not call qwen_soak_probe again."
    )


def build_plan(
    *,
    output_root: Path,
    configuration_sha256: str,
    serial_configuration_sha256: str,
    session_id: str,
    probe_extension: Path,
    tokenizer_json: Path,
    model: str = DEFAULT_MODEL,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    targets: Sequence[int] = DEFAULT_TARGETS,
    parity_targets: Sequence[int] = DEFAULT_PARITY_TARGETS,
) -> dict[str, Any]:
    """Publish a deterministic, create-only soak plan and private workspace."""

    if not SHA256_RE.fullmatch(configuration_sha256) or not SHA256_RE.fullmatch(
        serial_configuration_sha256
    ):
        raise SoakError("configuration SHA-256 values must be 64 lowercase hexadecimal characters")
    if not SESSION_ID_RE.fullmatch(session_id):
        raise SoakError("session ID must be a canonical UUID")
    selected_targets = _validate_targets(targets, context_window=context_window)
    selected_parity = tuple(parity_targets)
    if not selected_parity or any(target not in selected_targets for target in selected_parity):
        raise SoakError("every parity target must select a declared soak target")
    if selected_parity != tuple(sorted(set(selected_parity))):
        raise SoakError("parity targets must be strictly increasing and unique")

    extension = probe_extension.absolute()
    tokenizer = tokenizer_json.absolute()
    for path, label in ((extension, "probe extension"), (tokenizer, "tokenizer JSON")):
        if not path.is_file() or path.is_symlink():
            raise SoakError(f"{label} must be a regular non-symlink file: {path}")

    root = output_root.absolute()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError as error:
        raise SoakError(f"output root is create-only: {root}") from error
    workspace = root / "workspace"
    evidence = root / "evidence"
    session = root / "session"
    for directory in (workspace, evidence, session):
        directory.mkdir(mode=0o700)

    requirements = tuple(dict(item) for item in DEFAULT_TOOL_REQUIREMENTS)
    transactions_per_phase = _tool_requirement_total(requirements)
    if transactions_per_phase * len(selected_targets) < MIN_TOTAL_TOOL_TRANSACTIONS:
        raise SoakError("default soak plan does not reach 100 complete tool transactions")

    phases = []
    for index, target in enumerate(selected_targets):
        phase = f"turn-{index + 1:02d}-{target}"
        nonce = hashlib.sha256(
            f"{configuration_sha256}\0{session_id}\0{phase}".encode()
        ).hexdigest()[:16]
        digest = probe_digest(phase, nonce)
        phases.append(
            {
                "index": index,
                "id": phase,
                "target_context_tokens": target,
                "minimum_context_tokens": target - DEFAULT_CONTEXT_TOLERANCE,
                "maximum_context_tokens": target + 512,
                "nonce": nonce,
                "probe_digest": digest,
                "final_sentinel": f"SOAK_FINAL_OK phase={phase} nonce={nonce} digest={digest}",
                "instruction": _phase_prompt(
                    phase,
                    nonce,
                    digest,
                    workspace=workspace,
                    requirements=requirements,
                ),
                "workspace": str(workspace),
                "tool_requirements": [dict(item) for item in requirements],
                "expected_tool_transactions": transactions_per_phase,
                "minimum_large_archived_results": 2,
                "minimum_rehydrated_results": 2,
                "restart_after": target == 131_072,
                "capture_parity_vector": target in selected_parity,
            }
        )

    # Each tool result causes another model request, followed by the final-answer
    # request. This is logical prompt exposure; prefix reuse should make physical
    # prefill much smaller and is measured separately.
    requests_per_phase = transactions_per_phase + 1
    logical_prompt_floor = requests_per_phase * sum(selected_targets)
    parity_prompt_floor = sum(selected_parity)
    document: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "public_synthetic_data_only": True,
        "model": model,
        "configuration_sha256": configuration_sha256,
        "session": {"id": session_id, "directory": str(session)},
        "workspace": str(workspace),
        "context": {
            "window_tokens": context_window,
            "targets": list(selected_targets),
            "final_generation_headroom_tokens": context_window - selected_targets[-1],
            "tolerance_tokens": DEFAULT_CONTEXT_TOLERANCE,
        },
        "files": {
            "probe_extension": {"path": str(extension), "sha256": _file_sha256(extension)},
            "tokenizer_json": {"path": str(tokenizer), "sha256": _file_sha256(tokenizer)},
        },
        "phases": phases,
        "boundaries": {
            "pi_restart_after_context_tokens": 131_072,
            "manual_compaction_after_context_tokens": selected_targets[-1],
            "automatic_compaction_required": True,
            "restart_after_compaction": True,
            "compaction_summary_tokens": {
                "minimum": MIN_COMPACTION_TOKENS,
                "maximum": MAX_COMPACTION_TOKENS,
            },
            "maximum_post_compaction_context_tokens": MAX_POST_COMPACTION_CONTEXT,
            "required_lifecycle_scenarios": list(REQUIRED_LIFECYCLE_SCENARIOS),
            "required_compaction_modes": ["manual", "automatic"],
        },
        "parity": {
            "targets": list(selected_parity),
            "escape_disabled": True,
            "required_arms": ["optimized-m8", "serial-m1"],
            "arm_configuration_sha256": {
                "optimized-m8": configuration_sha256,
                "serial-m1": serial_configuration_sha256,
            },
            "minimum_output_tokens_per_arm": 2_048,
            "require_identical_completion_token_ids": True,
        },
        "gates": {
            "minimum_total_tool_transactions": MIN_TOTAL_TOOL_TRANSACTIONS,
            "tool_transactions_per_phase": transactions_per_phase,
            "minimum_large_archived_results_per_phase": 2,
            "minimum_rehydrated_results_per_phase": 2,
            "allow_guard_retry": False,
            "allow_premature_eos": False,
            "allow_length_stop": False,
            "allow_repetition_stop": False,
            "allow_expected_tool_error": True,
            "allow_unexpected_tool_error": False,
            "allow_truncation_banner": False,
            "require_exact_final_sentinel": True,
            "require_compaction_then_restart_probe": True,
        },
        "cost_contract": {
            "primary_tool_transactions": transactions_per_phase * len(phases),
            "primary_user_phases": len(phases),
            "minimum_primary_generation_requests": requests_per_phase * len(phases),
            "logical_primary_prompt_tokens_floor": logical_prompt_floor,
            "serial_parity_prompt_tokens_floor": parity_prompt_floor,
            "serial_parity_output_tokens": len(selected_parity) * 2_048,
            "compaction_requests": 2,
            "minimum_pi_process_restarts": 3,
            "note": (
                "Logical prompt exposure is not physical prefill. With authenticated prefix reuse, "
                "the primary track should prefill roughly one unique 249K history; serial parity "
                "still incurs the declared target-only replays."
            ),
        },
        "claim_boundary": (
            "Passing proves only the declared finite fixtures and exact replay vectors. "
            "It does not "
            "prove behavior for every arbitrary prompt. Universal lossless serving additionally "
            "requires independent serial validation before each committed target token."
        ),
    }
    sealed = _self_hash(document)
    _write_new_json(root / "plan.json", sealed)
    return sealed


def _assistant_messages(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        event["message"]
        for event in events
        if event.get("type") == "message_end"
        and isinstance(event.get("message"), dict)
        and event["message"].get("role") == "assistant"
    ]


def _content_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    return "\n".join(
        item["text"]
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    )


def _archive_digest_from_result(result: object) -> str | None:
    if not isinstance(result, dict) or not isinstance(result.get("details"), dict):
        return None
    archive = result["details"].get("qwenToolResultArchive")
    if not isinstance(archive, dict):
        return None
    digest = archive.get("sha256")
    path_value = archive.get("path")
    byte_count = archive.get("bytes")
    if (
        not isinstance(digest, str)
        or not SHA256_RE.fullmatch(digest)
        or not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < MIN_LARGE_RESULT_BYTES
    ):
        raise SoakError("large-result archive metadata is incomplete")
    path = Path(path_value)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SoakError(f"large-result archive is unavailable: {path}: {error}") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_size != byte_count
        or _file_sha256(path) != digest
    ):
        raise SoakError("large-result archive failed byte and ownership authentication")
    return digest


def validate_turn_document(document: Mapping[str, Any], phase: Mapping[str, Any]) -> None:
    """Validate one complete RPC turn without accepting a guard/retry as success."""

    if document.get("schema") != TURN_SCHEMA:
        raise SoakError("turn evidence has the wrong schema")
    _verify_self_hash(document)
    if document.get("phase_id") != phase["id"]:
        raise SoakError("turn evidence phase does not match its plan")
    events = document.get("rpc_events")
    if (
        not isinstance(events, list)
        or not events
        or not all(isinstance(item, dict) for item in events)
    ):
        raise SoakError("turn evidence has no complete RPC event list")
    serialized = _canonical(events).decode(errors="replace").lower()
    for forbidden in FORBIDDEN_DIAGNOSTICS:
        if forbidden in serialized:
            raise SoakError(f"turn contains forbidden diagnostic: {forbidden}")

    starts = [event for event in events if event.get("type") == "tool_execution_start"]
    ends = [event for event in events if event.get("type") == "tool_execution_end"]
    expected_total = phase.get("expected_tool_transactions")
    if (
        isinstance(expected_total, bool)
        or not isinstance(expected_total, int)
        or expected_total < 1
        or len(starts) != expected_total
        or len(ends) != expected_total
    ):
        raise SoakError("turn does not contain the exact planned tool transaction count")
    start_by_id: dict[str, Mapping[str, Any]] = {}
    end_by_id: dict[str, Mapping[str, Any]] = {}
    for label, source, destination in (
        ("start", starts, start_by_id),
        ("end", ends, end_by_id),
    ):
        for event in source:
            call_id = event.get("toolCallId")
            if not isinstance(call_id, str) or not call_id or call_id in destination:
                raise SoakError(f"tool {label} IDs are missing or duplicated")
            destination[call_id] = event
    if set(start_by_id) != set(end_by_id):
        raise SoakError("tool transaction ledger contains an orphaned call or result")
    for call_id, start in start_by_id.items():
        if end_by_id[call_id].get("toolName") != start.get("toolName"):
            raise SoakError("tool transaction start/result names differ")

    requirements = phase.get("tool_requirements")
    if (
        not isinstance(requirements, list)
        or _tool_requirement_total(requirements) != expected_total
    ):
        raise SoakError("phase tool requirements are incomplete")
    expected_names = [item.get("tool_name") for item in requirements]
    if len(expected_names) != len(set(expected_names)):
        raise SoakError("phase tool requirements duplicate a tool name")
    for requirement in requirements:
        tool_name = requirement["tool_name"]
        matching = [event for event in ends if event.get("toolName") == tool_name]
        successes = sum(event.get("isError") is False for event in matching)
        failures = sum(event.get("isError") is True for event in matching)
        if successes != requirement["successful"] or failures != requirement["failed"]:
            raise SoakError(f"tool outcome counts differ for {tool_name}")
    if any(event.get("toolName") not in set(expected_names) for event in starts):
        raise SoakError("turn executed an undeclared tool")

    probe_starts = [event for event in starts if event.get("toolName") == "qwen_soak_probe"]
    probe_ends = [event for event in ends if event.get("toolName") == "qwen_soak_probe"]
    if len(probe_starts) != 1 or len(probe_ends) != 1:
        raise SoakError("turn must execute qwen_soak_probe exactly once")
    expected_arguments = {
        "phase": phase["id"],
        "nonce": phase["nonce"],
        "expectedDigest": phase["probe_digest"],
    }
    if probe_starts[0].get("args") != expected_arguments:
        raise SoakError("qwen_soak_probe arguments differ from the authenticated plan")
    if probe_ends[0].get("toolCallId") != probe_starts[0].get("toolCallId"):
        raise SoakError("qwen_soak_probe start/end IDs differ")
    if probe_ends[0].get("isError") is not False:
        raise SoakError("qwen_soak_probe returned an error")

    assistants = _assistant_messages(events)
    if len(assistants) < 2:
        raise SoakError("tool turn did not contain tool-call and final assistant messages")
    for message in assistants:
        if message.get("stopReason") in {"error", "aborted", "length"}:
            raise SoakError(
                f"assistant ended with forbidden stop reason: {message.get('stopReason')}"
            )
        if message.get("rawStopReason") in {"repetition", "premature_action_stop", "length"}:
            raise SoakError(
                f"assistant ended with forbidden raw stop reason: {message.get('rawStopReason')}"
            )
    if any(message.get("stopReason") != "toolUse" for message in assistants[:-1]):
        raise SoakError(
            "assistant emitted a premature terminal response before the phase completed"
        )
    if assistants[-1].get("stopReason") != "stop":
        raise SoakError("phase final assistant did not end with a normal explicit stop")
    tool_calls = [
        item
        for message in assistants
        for item in message.get("content", [])
        if isinstance(item, dict) and item.get("type") == "toolCall"
    ]
    if len(tool_calls) != expected_total:
        raise SoakError("assistant structured calls do not match the planned transaction count")
    if any(
        isinstance(item, dict) and item.get("type") == "toolCall"
        for item in assistants[-1].get("content", [])
    ):
        raise SoakError("phase final assistant still contains a tool call")
    structured_by_id: dict[str, Mapping[str, Any]] = {}
    for call in tool_calls:
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id or call_id in structured_by_id:
            raise SoakError("assistant structured tool-call IDs are missing or duplicated")
        structured_by_id[call_id] = call
    if set(structured_by_id) != set(start_by_id):
        raise SoakError("assistant calls and executed tool transactions differ")
    for call_id, call in structured_by_id.items():
        start = start_by_id[call_id]
        if call.get("name") != start.get("toolName") or call.get("arguments") != start.get("args"):
            raise SoakError("assistant structured call differs from executed tool input")

    probes = [item for item in tool_calls if item.get("name") == "qwen_soak_probe"]
    if len(probes) != 1:
        raise SoakError("assistant did not emit exactly one structured qwen_soak_probe call")
    if probes[0].get("arguments") != expected_arguments:
        raise SoakError("structured qwen_soak_probe call arguments differ from the plan")

    workspace = Path(phase.get("workspace", ""))
    if not workspace.is_absolute():
        raise SoakError("phase has no absolute disposable workspace")
    for event in starts:
        if event.get("toolName") != "edit":
            continue
        arguments = event.get("args")
        path_value = arguments.get("path") if isinstance(arguments, dict) else None
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise SoakError("edit transaction does not use an absolute disposable path")
        try:
            Path(path_value).relative_to(workspace)
        except ValueError as error:
            raise SoakError("edit transaction escaped the disposable workspace") from error

    archive_digests = {
        digest
        for event in ends
        if (digest := _archive_digest_from_result(event.get("result"))) is not None
    }
    minimum_archives = phase.get("minimum_large_archived_results")
    if not isinstance(minimum_archives, int) or len(archive_digests) < minimum_archives:
        raise SoakError("phase did not preserve enough authenticated large-result archives")
    rehydrate_starts = [
        event for event in starts if event.get("toolName") == "qwen_rehydrate_tool_turn"
    ]
    rehydrated_digests = {
        event.get("args", {}).get("sha256")
        for event in rehydrate_starts
        if isinstance(event.get("args"), dict)
    }
    minimum_rehydrated = phase.get("minimum_rehydrated_results")
    if (
        not isinstance(minimum_rehydrated, int)
        or len(rehydrated_digests) < minimum_rehydrated
        or not rehydrated_digests.issubset(archive_digests)
    ):
        raise SoakError("rehydration calls do not authenticate the phase's large results")

    final_text = _content_text(assistants[-1]).strip()
    if phase["final_sentinel"] not in final_text:
        raise SoakError("final assistant message omitted the exact phase sentinel")

    agent_ends = [event for event in events if event.get("type") == "agent_end"]
    settled = [event for event in events if event.get("type") == "agent_settled"]
    if len(agent_ends) != 1 or agent_ends[0].get("willRetry") is not False:
        raise SoakError("turn did not end once without a retry")
    if len(settled) != 1:
        raise SoakError("turn did not settle exactly once")
    context_tokens = document.get("context_tokens")
    if not isinstance(context_tokens, int) or not (
        phase["minimum_context_tokens"] <= context_tokens <= phase["maximum_context_tokens"]
    ):
        raise SoakError(
            f"turn context {context_tokens!r} is outside "
            f"[{phase['minimum_context_tokens']}, {phase['maximum_context_tokens']}]"
        )


def _validate_restart(document: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if document.get("schema") != RESTART_SCHEMA:
        raise SoakError("restart evidence has the wrong schema")
    _verify_self_hash(document)
    if document.get("plan_sha256") != plan["payload_sha256"]:
        raise SoakError("restart evidence is bound to another plan")
    before = document.get("process_before")
    after = document.get("process_after")
    if not isinstance(before, dict) or not isinstance(after, dict) or before == after:
        raise SoakError("restart evidence does not identify two distinct Pi processes")
    if (
        before.get("session_id") != plan["session"]["id"]
        or after.get("session_id") != before.get("session_id")
    ):
        raise SoakError("Pi restart did not retain the planned session identity")
    restart_indexes = [
        index for index, phase in enumerate(plan["phases"]) if phase.get("restart_after") is True
    ]
    if len(restart_indexes) != 1 or restart_indexes[0] + 1 >= len(plan["phases"]):
        raise SoakError("plan does not contain one usable mid-context restart boundary")
    expected_next = plan["phases"][restart_indexes[0] + 1]["id"]
    if document.get("first_post_restart_turn") != expected_next:
        raise SoakError("restart boundary was not followed by the declared continuation turn")


def _validate_compaction(document: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if document.get("schema") != COMPACTION_SCHEMA:
        raise SoakError("compaction evidence has the wrong schema")
    _verify_self_hash(document)
    if document.get("plan_sha256") != plan["payload_sha256"]:
        raise SoakError("compaction evidence is bound to another plan")
    if document.get("mode") not in set(plan["boundaries"]["required_compaction_modes"]):
        raise SoakError("compaction evidence has an undeclared mode")
    if document.get("success") is not True or document.get("interrupted") is not False:
        raise SoakError("compaction did not complete cleanly")
    tokens_before = document.get("tokens_before")
    if (
        not isinstance(tokens_before, int)
        or tokens_before < plan["phases"][-1]["minimum_context_tokens"]
    ):
        raise SoakError("compaction did not consume the qualified long-context state")
    summary_tokens = document.get("summary_tokens")
    if (
        not isinstance(summary_tokens, int)
        or not MIN_COMPACTION_TOKENS <= summary_tokens <= MAX_COMPACTION_TOKENS
    ):
        raise SoakError("compaction summary is outside the 500-8,000 token contract")
    if not isinstance(document.get("summary_sha256"), str) or not SHA256_RE.fullmatch(
        document["summary_sha256"]
    ):
        raise SoakError("compaction summary has no authenticated digest")
    if not isinstance(document.get("successor_prompt_sha256"), str) or not SHA256_RE.fullmatch(
        document["successor_prompt_sha256"]
    ):
        raise SoakError("compaction successor prompt has no authenticated digest")
    post_context = document.get("post_restart_context_tokens")
    if not isinstance(post_context, int) or post_context > MAX_POST_COMPACTION_CONTEXT:
        raise SoakError("post-compaction successor context is missing or unexpectedly large")
    if document.get("post_restart_probe_passed") is not True:
        raise SoakError("post-compaction restart probe did not pass")


def _validate_lifecycle(
    document: Mapping[str, Any], plan: Mapping[str, Any]
) -> list[Path]:
    if document.get("schema") != LIFECYCLE_SCHEMA:
        raise SoakError("lifecycle evidence has the wrong schema")
    _verify_self_hash(document)
    if document.get("plan_sha256") != plan["payload_sha256"]:
        raise SoakError("lifecycle evidence is bound to another plan")
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, list) or not all(isinstance(item, dict) for item in scenarios):
        raise SoakError("lifecycle evidence has no scenario list")
    required = tuple(plan["boundaries"]["required_lifecycle_scenarios"])
    observed = [item.get("id") for item in scenarios]
    if observed != list(required):
        raise SoakError("lifecycle scenarios are missing, duplicated, or reordered")

    artifacts: list[Path] = []
    for scenario in scenarios:
        scenario_id = scenario["id"]
        if scenario.get("passed") is not True:
            raise SoakError(f"lifecycle scenario did not pass: {scenario_id}")
        observations = scenario.get("observations")
        if not isinstance(observations, dict):
            raise SoakError(f"lifecycle scenario has no observations: {scenario_id}")
        if scenario_id in INTERRUPTION_SCENARIOS and observations != {
            "canonical_state_unchanged": True,
            "duplicate_side_effects": 0,
            "resumed_from_prior_durable_head": True,
        }:
            raise SoakError(f"interruption did not recover atomically: {scenario_id}")
        if scenario_id in ISOLATION_SCENARIOS and (
            observations.get("cross_contamination") is not False
            or observations.get("independent_identities", 0) < 2
        ):
            raise SoakError(f"session/branch scenario was not isolated: {scenario_id}")
        if scenario_id == "clean-exit-restart-resume" and (
            observations.get("cold_fallback") is not False
            or observations.get("authenticated_snapshot_reused") is not True
        ):
            raise SoakError("clean restart did not reuse an authenticated snapshot")
        if scenario_id == "prompt-tool-abi-migration" and (
            observations.get("migration_authenticated") is not True
            or observations.get("prior_head_unchanged") is not True
        ):
            raise SoakError("prompt/tool ABI migration was not failure-atomic")
        if scenario_id in {"automatic-compaction", "manual-compaction"} and (
            observations.get("pointer_authenticated") is not True
            or observations.get("successor_resumed") is not True
        ):
            raise SoakError(f"compaction lifecycle did not resume cleanly: {scenario_id}")
        if scenario_id == "large-user-prompt" and observations.get("prompt_tokens", 0) < 32_768:
            raise SoakError("large-user-prompt scenario was smaller than 32K tokens")
        if scenario_id == "output-limit-boundary" and observations != {
            "content_preserved": True,
            "explicit_stop_reason": True,
            "silent_truncation": False,
        }:
            raise SoakError("output-limit boundary was silent or lossy")

        evidence_path = scenario.get("evidence_path")
        evidence_sha256 = scenario.get("evidence_sha256")
        if (
            not isinstance(evidence_path, str)
            or not Path(evidence_path).is_absolute()
            or not isinstance(evidence_sha256, str)
            or not SHA256_RE.fullmatch(evidence_sha256)
        ):
            raise SoakError(f"scenario has no absolute authenticated evidence: {scenario_id}")
        path = Path(evidence_path)
        try:
            metadata = path.lstat()
        except OSError as error:
            raise SoakError(f"scenario evidence is unavailable: {path}: {error}") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
            or _file_sha256(path) != evidence_sha256
        ):
            raise SoakError(f"scenario evidence failed authentication: {scenario_id}")
        artifacts.append(path)
    return artifacts


def _load_replay_capture(path: Path) -> dict[str, Any]:
    capture = _read_secure_json(path)
    if capture.get("schema") != "urn:qwen-r9700:failed-prompt-replay:v1":
        raise SoakError("parity arm capture has the wrong schema")
    fingerprint = capture.get("capture_sha256")
    payload = dict(capture)
    payload.pop("capture_sha256", None)
    if not isinstance(fingerprint, str) or _sha256(_canonical(payload)) != fingerprint:
        raise SoakError("parity arm capture fingerprint is invalid")
    return capture


def _validate_parity(
    document: Mapping[str, Any], plan: Mapping[str, Any], *, target: int
) -> list[Path]:
    if document.get("schema") != REPLAY_COMPARISON_SCHEMA:
        raise SoakError("parity report has the wrong replay-comparison schema")
    fingerprint = document.get("comparison_sha256")
    payload = dict(document)
    payload.pop("comparison_sha256", None)
    if not isinstance(fingerprint, str) or _sha256(_canonical(payload)) != fingerprint:
        raise SoakError("parity comparison fingerprint is invalid")
    if document.get("all_completion_token_ids_identical") is not True:
        raise SoakError("optimized and serial parity arms did not emit identical token IDs")
    arms = document.get("arms")
    if not isinstance(arms, list) or len(arms) < 2:
        raise SoakError("parity report does not contain both serving arms")
    arm_ids = {arm.get("arm_id") for arm in arms if isinstance(arm, dict)}
    if not set(plan["parity"]["required_arms"]).issubset(arm_ids):
        raise SoakError("parity report is missing a required serving arm")
    capture_paths: list[Path] = []
    for arm in arms:
        if not isinstance(arm, dict):
            raise SoakError("parity arm is not an object")
        if arm.get("first_divergence_from_reference") is not None:
            raise SoakError("parity arm records a token divergence")
        minimum_output = plan["parity"]["minimum_output_tokens_per_arm"]
        if arm.get("completion_tokens", minimum_output) < minimum_output:
            raise SoakError("parity arm output is shorter than the declared floor")
        capture_path = arm.get("capture_path")
        if not isinstance(capture_path, str) or not Path(capture_path).is_absolute():
            raise SoakError("parity arm does not reference an absolute capture path")
        resolved_capture_path = Path(capture_path)
        capture = _load_replay_capture(resolved_capture_path)
        capture_paths.append(resolved_capture_path)
        arm_id = arm.get("arm_id")
        if capture.get("arm", {}).get("id") != arm_id:
            raise SoakError("parity report/capture arm identity differs")
        expected_configuration = plan["parity"]["arm_configuration_sha256"].get(arm_id)
        if capture.get("arm", {}).get("configuration_sha256") != expected_configuration:
            raise SoakError("parity capture used an unauthenticated serving configuration")
        prompt_count = capture.get("source", {}).get("prompt_token_count")
        phase = next(item for item in plan["phases"] if item["target_context_tokens"] == target)
        if not isinstance(prompt_count, int) or not (
            phase["minimum_context_tokens"] <= prompt_count <= phase["maximum_context_tokens"]
        ):
            raise SoakError("parity capture prompt length is outside its planned checkpoint")
        if capture.get("response", {}).get("completion_token_count") < minimum_output:
            raise SoakError("parity capture output is shorter than the declared floor")
    return capture_paths


def _validate_plan_contract(plan: Mapping[str, Any]) -> None:
    context = plan.get("context")
    phases = plan.get("phases")
    gates = plan.get("gates")
    boundaries = plan.get("boundaries")
    if not all(isinstance(item, dict) for item in (context, gates, boundaries)):
        raise SoakError("soak plan contract sections are incomplete")
    for field in ("configuration_sha256",):
        if not isinstance(plan.get(field), str) or not SHA256_RE.fullmatch(plan[field]):
            raise SoakError(f"soak plan {field} is invalid")
    session = plan.get("session")
    if (
        not isinstance(session, dict)
        or not isinstance(session.get("id"), str)
        or not SESSION_ID_RE.fullmatch(session["id"])
        or not isinstance(session.get("directory"), str)
        or not Path(session["directory"]).is_absolute()
    ):
        raise SoakError("soak plan session identity is invalid")
    workspace_value = plan.get("workspace")
    if not isinstance(workspace_value, str) or not Path(workspace_value).is_absolute():
        raise SoakError("soak plan workspace is not absolute")
    workspace = Path(workspace_value)
    try:
        workspace_info = workspace.lstat()
    except OSError as error:
        raise SoakError(f"soak workspace is unavailable: {workspace}: {error}") from error
    if (
        not stat.S_ISDIR(workspace_info.st_mode)
        or workspace.is_symlink()
        or workspace_info.st_uid != os.getuid()
        or stat.S_IMODE(workspace_info.st_mode) != 0o700
    ):
        raise SoakError("soak workspace is not an owned 0700 directory")

    files = plan.get("files")
    if not isinstance(files, dict) or set(files) != {"probe_extension", "tokenizer_json"}:
        raise SoakError("soak plan file bindings are incomplete")
    for label, record in files.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise SoakError(f"soak plan {label} binding is malformed")
        path_value = record["path"]
        if (
            not isinstance(path_value, str)
            or not Path(path_value).is_absolute()
            or not isinstance(record["sha256"], str)
            or not SHA256_RE.fullmatch(record["sha256"])
        ):
            raise SoakError(f"soak plan {label} binding is invalid")
        path = Path(path_value)
        if not path.is_file() or path.is_symlink() or _file_sha256(path) != record["sha256"]:
            raise SoakError(f"soak plan {label} binding changed")
    if not isinstance(phases, list) or not phases or not all(
        isinstance(phase, dict) for phase in phases
    ):
        raise SoakError("soak plan has no phase list")
    window_tokens = context.get("window_tokens")
    raw_targets = context.get("targets")
    if (
        isinstance(window_tokens, bool)
        or not isinstance(window_tokens, int)
        or not isinstance(raw_targets, list)
    ):
        raise SoakError("soak context window or targets are invalid")
    targets = _validate_targets(raw_targets, context_window=window_tokens)
    if [phase.get("target_context_tokens") for phase in phases] != list(targets):
        raise SoakError("soak phases do not cover the exact context targets")
    phase_ids = [phase.get("id") for phase in phases]
    if any(not isinstance(value, str) or not PHASE_RE.fullmatch(value) for value in phase_ids):
        raise SoakError("soak phase identity is invalid")
    if len(phase_ids) != len(set(phase_ids)):
        raise SoakError("soak phase identity is duplicated")
    total_transactions = 0
    for phase in phases:
        requirements = phase.get("tool_requirements")
        expected = phase.get("expected_tool_transactions")
        if not isinstance(requirements, list) or _tool_requirement_total(requirements) != expected:
            raise SoakError("soak phase tool contract is incomplete")
        total_transactions += expected
        if phase.get("workspace") != str(workspace):
            raise SoakError("soak phase workspace differs from the plan")
        if not isinstance(phase.get("nonce"), str) or not NONCE_RE.fullmatch(phase["nonce"]):
            raise SoakError("soak phase nonce is invalid")
        if (
            not isinstance(phase.get("probe_digest"), str)
            or phase["probe_digest"] != probe_digest(phase["id"], phase["nonce"])
        ):
            raise SoakError("soak phase probe digest is invalid")
        if phase.get("final_sentinel") not in phase.get("instruction", ""):
            raise SoakError("soak phase instruction omits its final sentinel")
        if phase.get("minimum_large_archived_results", 0) < 2:
            raise SoakError("soak phase does not require two large archived results")
        if phase.get("minimum_rehydrated_results", 0) < 2:
            raise SoakError("soak phase does not require two rehydrated results")
        names = {item["tool_name"] for item in requirements}
        if names != {item["tool_name"] for item in DEFAULT_TOOL_REQUIREMENTS}:
            raise SoakError("soak phase omits a required tool family")
    if total_transactions < MIN_TOTAL_TOOL_TRANSACTIONS:
        raise SoakError("soak plan contains fewer than 100 tool transactions")
    if gates.get("minimum_total_tool_transactions") != MIN_TOTAL_TOOL_TRANSACTIONS:
        raise SoakError("soak plan weakens the total tool-transaction gate")
    per_phase = gates.get("tool_transactions_per_phase")
    if (
        isinstance(per_phase, bool)
        or not isinstance(per_phase, int)
        or per_phase * len(phases) != total_transactions
    ):
        raise SoakError("soak plan transaction cost and phase contracts differ")
    if boundaries.get("required_lifecycle_scenarios") != list(REQUIRED_LIFECYCLE_SCENARIOS):
        raise SoakError("soak plan weakens the lifecycle-scenario gate")
    if boundaries.get("required_compaction_modes") != ["manual", "automatic"]:
        raise SoakError("soak plan does not require manual and automatic compaction")
    restart_phases = [phase for phase in phases if phase.get("restart_after") is True]
    if len(restart_phases) != 1 or restart_phases[0]["target_context_tokens"] != 131_072:
        raise SoakError("soak plan has no unique 131,072-token restart boundary")


def verify_evidence(root: Path, *, publish_attestation: bool = False) -> dict[str, Any]:
    """Deep-verify all required soak artifacts and optionally seal an attestation."""

    root = root.absolute()
    plan_path = root / "plan.json"
    plan = _read_secure_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise SoakError("soak plan has the wrong schema")
    _verify_self_hash(plan)
    _validate_plan_contract(plan)

    artifacts: list[dict[str, Any]] = []
    for phase in plan["phases"]:
        path = root / "evidence" / f"{phase['id']}.json"
        document = _read_secure_json(path)
        if document.get("plan_sha256") != plan["payload_sha256"]:
            raise SoakError(f"turn {phase['id']} is bound to another plan")
        validate_turn_document(document, phase)
        artifacts.append({"path": str(path), "sha256": _file_sha256(path)})

    restart_path = root / "evidence" / "restart-mid-context.json"
    restart = _read_secure_json(restart_path)
    _validate_restart(restart, plan)
    artifacts.append({"path": str(restart_path), "sha256": _file_sha256(restart_path)})

    for mode in plan["boundaries"]["required_compaction_modes"]:
        compaction_path = root / "evidence" / f"compaction-{mode}-and-restart.json"
        compaction = _read_secure_json(compaction_path)
        _validate_compaction(compaction, plan)
        if compaction.get("mode") != mode:
            raise SoakError("compaction evidence file/mode identity differs")
        artifacts.append({"path": str(compaction_path), "sha256": _file_sha256(compaction_path)})

    lifecycle_path = root / "evidence" / "lifecycle-scenarios.json"
    lifecycle = _read_secure_json(lifecycle_path)
    lifecycle_artifacts = _validate_lifecycle(lifecycle, plan)
    artifacts.append({"path": str(lifecycle_path), "sha256": _file_sha256(lifecycle_path)})
    artifacts.extend(
        {"path": str(path), "sha256": _file_sha256(path)} for path in lifecycle_artifacts
    )

    for target in plan["parity"]["targets"]:
        path = root / "evidence" / f"parity-{target}.json"
        report = _read_secure_json(path)
        capture_paths = _validate_parity(report, plan, target=target)
        artifacts.append({"path": str(path), "sha256": _file_sha256(path)})
        artifacts.extend(
            {"path": str(capture_path), "sha256": _file_sha256(capture_path)}
            for capture_path in capture_paths
        )

    session_root = Path(plan["session"]["directory"])
    transcripts = list(session_root.rglob(f"*{plan['session']['id']}*.jsonl"))
    if len(transcripts) != 1:
        raise SoakError("planned Pi transcript was not resolved uniquely")
    transcript = transcripts[0]
    transcript_info = transcript.lstat()
    if (
        not transcript.is_file()
        or transcript.is_symlink()
        or transcript_info.st_uid != os.getuid()
        or stat.S_IMODE(transcript_info.st_mode) != 0o600
        or transcript_info.st_nlink != 1
    ):
        raise SoakError("planned Pi transcript is missing or unsafe")
    artifacts.append({"path": str(transcript), "sha256": _file_sha256(transcript)})

    attestation = _self_hash(
        {
            "schema": ATTESTATION_SCHEMA,
            "qualified_at": datetime.now(UTC).isoformat(),
            "plan": {"path": str(plan_path), "sha256": _file_sha256(plan_path)},
            "plan_payload_sha256": plan["payload_sha256"],
            "model": plan["model"],
            "configuration_sha256": plan["configuration_sha256"],
            "final_context_tokens": plan["phases"][-1]["target_context_tokens"],
            "artifacts": artifacts,
            "finite_fixture_result": "pass",
            "universal_arbitrary_prompt_claim": False,
            "claim_boundary": plan["claim_boundary"],
        }
    )
    if publish_attestation:
        _write_new_json(root / "attestation.json", attestation)
    return attestation


def _parse_targets(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("targets must be comma-separated integers") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="create a private, immutable soak plan")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--configuration-sha256", required=True)
    prepare.add_argument("--serial-configuration-sha256", required=True)
    prepare.add_argument("--session-id", required=True)
    prepare.add_argument("--probe-extension", type=Path, required=True)
    prepare.add_argument("--tokenizer-json", type=Path, required=True)
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    prepare.add_argument("--targets", type=_parse_targets, default=DEFAULT_TARGETS)
    prepare.add_argument("--parity-targets", type=_parse_targets, default=DEFAULT_PARITY_TARGETS)
    verify = subparsers.add_parser("verify", help="authenticate a completed soak without mutation")
    verify.add_argument("--output-root", type=Path, required=True)
    finalize = subparsers.add_parser(
        "finalize", help="authenticate and create the final attestation"
    )
    finalize.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "prepare":
            result = build_plan(
                output_root=arguments.output_root,
                configuration_sha256=arguments.configuration_sha256,
                serial_configuration_sha256=arguments.serial_configuration_sha256,
                session_id=arguments.session_id,
                probe_extension=arguments.probe_extension,
                tokenizer_json=arguments.tokenizer_json,
                model=arguments.model,
                context_window=arguments.context_window,
                targets=arguments.targets,
                parity_targets=arguments.parity_targets,
            )
        else:
            result = verify_evidence(
                arguments.output_root, publish_attestation=arguments.command == "finalize"
            )
    except (OSError, SoakError) as error:
        print(f"qwen-hauhau-250k-soak: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
