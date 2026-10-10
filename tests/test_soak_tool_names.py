from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qwen_r9700_lab.hauhau_250k_soak import (
    DEFAULT_TOOL_REQUIREMENTS,
    TOOL_NAME_ALIASES,
    TURN_SCHEMA,
    SoakError,
    _self_hash,
    _validate_plan_contract,
    build_plan,
    probe_digest,
    validate_turn_document,
)

LEGACY_NAMES = {canonical: legacy for legacy, canonical in TOOL_NAME_ALIASES.items()}


def synthetic_turn(
    tmp_path: Path, naming: tuple[bool, bool, bool, bool]
) -> tuple[dict, dict]:
    legacy_requirements, legacy_calls, legacy_starts, legacy_ends = naming
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    archives = []
    for index in range(2):
        payload = f"Synthetic archive {index}\n".encode() * 4096
        digest = hashlib.sha256(payload).hexdigest()
        path = workspace / f"{digest}.txt"
        path.write_bytes(payload)
        path.chmod(0o400)
        archives.append({"sha256": digest, "path": str(path), "bytes": len(payload)})
    phase = {
        "id": "synthetic",
        "nonce": "0123456789abcdef",
        "workspace": str(workspace),
        "tool_requirements": [
            {
                **item,
                "tool_name": LEGACY_NAMES.get(item["tool_name"], item["tool_name"])
                if legacy_requirements
                else item["tool_name"],
            }
            for item in DEFAULT_TOOL_REQUIREMENTS
        ],
        "expected_tool_transactions": 15,
        "minimum_large_archived_results": 2,
        "minimum_rehydrated_results": 2,
        "minimum_context_tokens": 10,
        "maximum_context_tokens": 20,
        "final_sentinel": "SOAK_FINAL_OK synthetic",
    }
    phase["probe_digest"] = probe_digest(phase["id"], phase["nonce"])
    events = []
    read_index = archive_index = 0
    for requirement in DEFAULT_TOOL_REQUIREMENTS:
        name = requirement["tool_name"]
        for failed in [False] * requirement["successful"] + [True] * requirement[
            "failed"
        ]:
            call_id = f"synthetic-{len(events)}"
            arguments = {"synthetic": call_id}
            result = {"content": [{"type": "text", "text": "Synthetic result"}]}
            if name == "qwen_soak_probe":
                arguments = {
                    "phase": phase["id"],
                    "nonce": phase["nonce"],
                    "expectedDigest": phase["probe_digest"],
                }
            elif name == "edit_file":
                arguments = {"path": str(workspace / f"{call_id}.txt")}
            elif name == "read_file":
                result["details"] = {"qwenToolResultArchive": archives[read_index]}
                read_index += 1
            elif name == "rehydrate_tool_result":
                arguments = {"sha256": archives[archive_index]["sha256"]}
                archive_index += 1
            alias = LEGACY_NAMES.get(name, name)
            call = {
                "type": "toolCall",
                "id": call_id,
                "name": alias if legacy_calls else name,
                "arguments": arguments,
            }
            events.extend(
                [
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [call],
                            "stopReason": "toolUse",
                        },
                    },
                    {
                        "type": "tool_execution_start",
                        "toolCallId": call_id,
                        "toolName": alias if legacy_starts else name,
                        "args": arguments,
                    },
                    {
                        "type": "tool_execution_end",
                        "toolCallId": call_id,
                        "toolName": alias if legacy_ends else name,
                        "isError": failed,
                        "result": result,
                    },
                ]
            )
    events.extend(
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": phase["final_sentinel"]}],
                    "stopReason": "stop",
                },
            },
            {"type": "agent_end", "willRetry": False},
            {"type": "agent_settled"},
        ]
    )
    return _self_hash(
        {
            "schema": TURN_SCHEMA,
            "phase_id": phase["id"],
            "context_tokens": 15,
            "rpc_events": events,
        }
    ), phase


@pytest.mark.parametrize(
    "naming",
    [
        (False, False, False, False),
        (True, True, True, True),
        (True, False, False, False),
        (False, True, True, True),
        (True, False, True, False),
        (False, True, False, True),
    ],
)
def test_soak_accepts_canonical_legacy_and_mixed_names_without_rewriting_evidence(
    tmp_path: Path,
    naming: tuple[bool, bool, bool, bool],
) -> None:
    document, phase = synthetic_turn(tmp_path, naming)
    original = json.dumps([document, phase], sort_keys=True)
    validate_turn_document(document, phase)
    assert json.dumps([document, phase], sort_keys=True) == original


def test_soak_rejects_unknown_tool_identity(tmp_path: Path) -> None:
    document, phase = synthetic_turn(tmp_path, (False, False, False, False))
    for event in document["rpc_events"]:
        if event.get("toolName") == "google_ai_search":
            event["toolName"] = "google_ai_search_unknown"
    del document["payload_sha256"]
    with pytest.raises(SoakError, match="tool outcome counts differ"):
        validate_turn_document(_self_hash(document), phase)


def test_soak_accepts_previously_deployed_archive_reader_name(tmp_path: Path) -> None:
    document, phase = synthetic_turn(tmp_path, (False, False, False, False))
    for requirement in phase["tool_requirements"]:
        if requirement["tool_name"] == "rehydrate_tool_result":
            requirement["tool_name"] = "read_archived_tool_result"
    for event in document["rpc_events"]:
        if event.get("toolName") == "rehydrate_tool_result":
            event["toolName"] = "read_archived_tool_result"
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "toolCall" and block.get("name") == "rehydrate_tool_result":
                block["name"] = "read_archived_tool_result"
    del document["payload_sha256"]
    document = _self_hash(document)
    original = json.dumps([document, phase], sort_keys=True)
    validate_turn_document(document, phase)
    assert json.dumps([document, phase], sort_keys=True) == original


@pytest.mark.parametrize("legacy", [False, True])
def test_soak_checks_edit_workspace_for_both_names(
    tmp_path: Path, legacy: bool
) -> None:
    document, phase = synthetic_turn(tmp_path, (legacy, legacy, legacy, legacy))
    edit = next(
        event
        for event in document["rpc_events"]
        if event.get("type") == "tool_execution_start"
        and event.get("toolName") in {"edit", "edit_file"}
    )
    edit["args"]["path"] = str(tmp_path / "outside-workspace.txt")
    del document["payload_sha256"]
    with pytest.raises(SoakError, match="escaped the disposable workspace"):
        validate_turn_document(_self_hash(document), phase)


def test_soak_rejects_duplicate_alias_family(tmp_path: Path) -> None:
    document, phase = synthetic_turn(tmp_path, (False, False, False, False))
    read = next(
        item for item in phase["tool_requirements"] if item["tool_name"] == "read_file"
    )
    read["successful"] = 1
    phase["tool_requirements"].append(
        {"tool_name": "read", "successful": 1, "failed": 0}
    )
    with pytest.raises(SoakError, match="duplicate a tool name"):
        validate_turn_document(document, phase)


def test_new_soak_plan_prompts_use_canonical_names_and_legacy_plans_still_validate(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "probe.mjs"
    extension.write_text("export default () => {};\n")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}\n")
    plan = build_plan(
        output_root=tmp_path / "plan",
        configuration_sha256="a" * 64,
        serial_configuration_sha256="b" * 64,
        session_id="12345678-1234-4234-8234-123456789abc",
        probe_extension=extension,
        tokenizer_json=tokenizer,
    )
    _validate_plan_contract(plan)
    for phase in plan["phases"]:
        for item in DEFAULT_TOOL_REQUIREMENTS:
            assert f"{item['tool_name']}=" in phase["instruction"]
        for item in phase["tool_requirements"]:
            item["tool_name"] = LEGACY_NAMES.get(item["tool_name"], item["tool_name"])
    historical = json.dumps(plan, sort_keys=True)
    _validate_plan_contract(plan)
    assert json.dumps(plan, sort_keys=True) == historical
