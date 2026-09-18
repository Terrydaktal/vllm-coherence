"""Fail-closed receipts for independent serial validation before live commits.

Finite differential campaigns remain bounded qualification.  This module
defines the separate runtime evidence contract required for arbitrary live
inputs: an optimized transition is provisional, an independently implemented
serial transition is authoritative, and canonical state changes only through
one validated publication.  The verifier does not execute either transition;
therefore a capability cannot be promoted until the release-bound runtime
producer emits receipts satisfying this contract for every canonical commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Final

CAPABILITY_SCHEMA: Final = "urn:qwen-r9700:live-serial-commit-capability:v16"
RECEIPT_SCHEMA: Final = "urn:qwen-r9700:live-serial-commit-receipt:v2"
LEDGER_SCHEMA: Final = "urn:qwen-r9700:live-serial-commit-ledger:v2"
COMMIT_COUNTS: Final = tuple(range(9))
RELEASE_ARTIFACT_SCHEMA: Final = "urn:qwen-r9700:coding-turbo-artifact:v1"

TARGET_LAYER_COUNT: Final = 64
TARGET_KV_LAYER_COUNT: Final = 16
DRAFT_LAYER_COUNT: Final = 5
GDN_LAYER_COUNT: Final = 48
QUEST_LAYER_COUNT: Final = 16

TRANSITION_UNITS: Final = (
    "m8_construction_positions_rope",
    "w4a16_projections",
    "full_attention_qk_norm_rope",
    "quest_scoring",
    "quest_top96_selection",
    "quest_union_visibility",
    "quest_historical_attention",
    "quest_causal_tail_attention",
    "quest_softmax_reduction",
    "gdn_convolution",
    "gdn_recurrence",
    "gdn_gated_rmsnorm",
    "residual_stream",
    "swiglu_ffn",
    "final_rmsnorm",
    "lm_head",
    "target_verification",
    "sampler_rng_decoding",
)

STATE_DIGEST_FIELDS: Final = (
    "committed_token_prefix_sha256",
    "complete_target_logits_sha256",
    "logical_target_kv_sha256",
    "physical_target_kv_sha256",
    "logical_draft_kv_sha256",
    "physical_draft_kv_sha256",
    "gdn_state_sha256",
    "convolution_state_sha256",
    "residual_and_intermediates_sha256",
    "positions_sha256",
    "rope_indices_sha256",
    "ordered_quest_scores_sha256",
    "ordered_quest_pages_sha256",
    "quest_visibility_sha256",
    "quest_attention_outputs_sha256",
    "quantization_scales_sha256",
    "workspace_contents_sha256",
    "cache_mapping_sha256",
    "allocation_topology_sha256",
    "ownership_sha256",
    "pins_sha256",
    "refcounts_sha256",
    "sampler_rng_decoding_sha256",
    "transaction_state_sha256",
    "canonical_root_sha256",
)

LAYER_DIGEST_FIELDS: Final = {
    "logical_target_kv_sha256_by_layer": TARGET_KV_LAYER_COUNT,
    "physical_target_kv_sha256_by_layer": TARGET_KV_LAYER_COUNT,
    "logical_draft_kv_sha256_by_layer": DRAFT_LAYER_COUNT,
    "physical_draft_kv_sha256_by_layer": DRAFT_LAYER_COUNT,
    "gdn_state_sha256_by_layer": GDN_LAYER_COUNT,
    "convolution_state_sha256_by_layer": GDN_LAYER_COUNT,
    "residual_and_intermediates_sha256_by_layer": TARGET_LAYER_COUNT,
    "quest_scores_sha256_by_layer": QUEST_LAYER_COUNT,
    "quest_pages_sha256_by_layer": QUEST_LAYER_COUNT,
    "quest_visibility_sha256_by_layer": QUEST_LAYER_COUNT,
    "quest_attention_outputs_sha256_by_layer": QUEST_LAYER_COUNT,
}

STATE_KEYS: Final = {
    *STATE_DIGEST_FIELDS,
    *LAYER_DIGEST_FIELDS,
    "allocator_generation",
    "commit_epoch",
    "device_error_word",
    "logical_length",
    "nonfinite_count",
    "semantic_unit_sha256",
}

CAPABILITY_KEYS: Final = {
    "schema",
    "semantic_source",
    "release_artifact_manifest",
    "controller_runtime",
    "enforcement_runtime",
    "driver_runtime",
    "model_output_fingerprint_runtime",
    "coordinator_runtime",
    "scheduler_runtime",
    "release_runtime",
    "serial_oracle_runtime",
    "model",
    "hardware_runtime_contract_sha256",
    "compiler_contract_sha256",
    "candidate_implementation_family",
    "serial_implementation_family",
    "commit_counts",
    "runtime_invariants",
    "hook_boundaries",
    "classification",
    "universal_guarantee_mechanism",
    "capability_sha256",
}

RUNTIME_INVARIANTS: Final = {
    "engine_core_is_sole_transition_coordinator": True,
    "engine_core_source_authenticated": True,
    "scheduler_is_sole_canonical_state_authority": True,
    "scheduler_invokes_authenticated_private_root_preparation": True,
    "scheduler_captures_complete_physical_state_before_record_and_publication": True,
    "canonical_physical_root_is_rechecked_after_every_provisional_operation": True,
    "candidate_executor_is_engine_core_model_executor": True,
    "candidate_executor_distinct_from_coordinator_and_scheduler": True,
    "serial_executor_distinct_from_coordinator_scheduler_and_candidate": True,
    "candidate_and_serial_worker_rpc_methods_are_distinct": True,
    "private_worker_request_and_result_identities_are_echo_verified": True,
    "private_worker_rpc_requires_exactly_one_tp1_result": True,
    "worker_executors_cannot_publish_canonical_state": True,
    "scheduler_records_worker_results_before_comparison": True,
    "scheduler_fingerprints_complete_worker_output_at_record_publish_and_consume": True,
    "scheduler_extracts_and_matches_worker_output_tokens_at_record_publish_and_consume": True,
    "model_runner_output_contains_exactly_one_transaction_request": True,
    "scheduler_update_requires_one_shot_verified_advance": True,
    "scheduler_prepares_identity_bound_private_round_payload": True,
    "candidate_determines_live_commit_width": True,
    "serial_executes_exact_candidate_or_external_cap_width": True,
    "candidate_uses_private_state": True,
    "serial_uses_independent_private_state": True,
    "serial_executor_distinct_from_candidate_runner": True,
    "serial_hook_source_authenticated": True,
    "release_hooks_source_authenticated": True,
    "provisional_round_opened_before_candidate_forward": True,
    "provisional_abort_proves_canonical_unchanged": True,
    "canonical_unchanged_before_publication": True,
    "complete_state_compared": True,
    "atomic_root_publication": True,
    "serial_published_on_mismatch": True,
    "counterexample_preserved_before_fallback": True,
    "candidate_quarantined_on_mismatch": True,
    "serial_continues_after_quarantine": True,
    "authenticated_process_death_recovery": True,
    "recovery_is_idempotent": True,
    "runtime_bypass_forbidden": True,
}

HOOK_BOUNDARIES: Final = (
    "engine_core_enters_authenticated_serial_validation_before_model_execution",
    "scheduler_prepares_identity_bound_private_schedule_without_canonical_mutation",
    "scheduler_opens_private_branches_before_candidate_worker_execution",
    "candidate_worker_forward_between_private_open_and_validated_commit",
    "candidate_target_verification_determines_commit_width_zero_through_eight",
    "scheduler_cancellation_aborts_private_round",
    "after_candidate_before_serial",
    "serial_worker_executes_exact_verified_or_capped_candidate_width",
    "scheduler_records_private_worker_results",
    "after_serial_before_comparison",
    "after_comparison_before_publication",
    "scheduler_atomic_root_publication_before_request_advance",
    "engine_core_consumes_verified_advance_before_scheduler_update_from_output",
    "scheduler_update_from_output_accepts_only_verified_commit",
    "after_atomic_scheduler_publication",
    "serial_fallback_after_candidate_fault",
    "restart_resolves_prepublication_intent",
)

RECEIPT_KEYS: Final = {
    "schema",
    "capability_sha256",
    "session_id",
    "request_id",
    "round_index",
    "commit_count",
    "mode",
    "candidate_enabled_before",
    "candidate_enabled_after",
    "canonical_before",
    "candidate_state",
    "candidate_token_ids",
    "candidate_failure",
    "serial_state",
    "serial_token_ids",
    "canonical_probe_after_candidate_sha256",
    "canonical_probe_after_serial_sha256",
    "comparison_equal",
    "first_difference",
    "publication_source",
    "publication_intent_sha256",
    "canonical_after",
    "publication_atomic",
    "candidate_private_state_destroyed",
    "serial_private_state_destroyed",
    "counterexample_sha256",
    "comparison_sha256",
    "receipt_sha256",
}

CANDIDATE_FAILURE_KEYS: Final = {
    "stage",
    "exception_type",
    "message_sha256",
    "traceback_sha256",
}

LEDGER_KEYS: Final = {
    "schema",
    "capability_sha256",
    "session_id",
    "initial_state",
    "receipts",
    "final_state",
    "candidate_quarantined",
    "request_complete",
    "classification",
    "universal_guarantee_mechanism",
    "ledger_sha256",
}


class LiveCommitValidationError(RuntimeError):
    """A live serial-validation capability or receipt is unsafe."""


def _canonical(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise LiveCommitValidationError("document is not canonical JSON") from error
    if json.loads(payload) != value:
        raise LiveCommitValidationError("document does not round-trip through canonical JSON")
    return payload


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _document_digest(value: dict[str, Any], field: str) -> str:
    return _digest({key: item for key, item in value.items() if key != field})


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LiveCommitValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(character.isspace() for character in value)
    ):
        raise LiveCommitValidationError(f"{label} is invalid")
    return value


def _exact_dict(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LiveCommitValidationError(f"{label} keys must be exactly {sorted(keys)}")
    return json.loads(_canonical(value))


def _artifact(value: object, label: str) -> dict[str, str]:
    artifact = _exact_dict(value, {"path", "sha256"}, label)
    path_raw = artifact["path"]
    if not isinstance(path_raw, str) or not path_raw.startswith("/"):
        raise LiveCommitValidationError(f"{label}.path must be absolute")
    expected = _sha256(artifact["sha256"], f"{label}.sha256")
    path = Path(path_raw)
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        raise LiveCommitValidationError(
            f"{label} must be an owner-controlled, non-writable-by-others regular file"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    def identity(item: os.stat_result) -> tuple[int, int, int, int]:
        return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)

    if identity(before) != identity(after):
        raise LiveCommitValidationError(f"{label} changed during authentication")
    actual = digest.hexdigest()
    if actual != expected:
        raise LiveCommitValidationError(
            f"{label} SHA-256 differs: expected {expected}, found {actual}"
        )
    return {"path": str(path.absolute()), "sha256": actual}


def _release_manifest(value: object) -> tuple[dict[str, str], dict[str, Any]]:
    artifact = _artifact(value, "capability.release_artifact_manifest")
    try:
        manifest = json.loads(Path(artifact["path"]).read_bytes())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LiveCommitValidationError("release artifact manifest is invalid JSON") from error
    keys = {
        "artifact_id",
        "artifact_kind",
        "files",
        "schema",
        "semantic_contract",
        "semantic_source_sha256",
        "source_manifest_sha256",
        "tree_sha256",
    }
    manifest = _exact_dict(manifest, keys, "release artifact manifest")
    if (
        manifest["schema"] != RELEASE_ARTIFACT_SCHEMA
        or manifest["artifact_kind"] != "release"
    ):
        raise LiveCommitValidationError("release artifact manifest identity differs")
    _identifier(manifest["artifact_id"], "release artifact ID")
    if not isinstance(manifest["semantic_contract"], str) or not manifest[
        "semantic_contract"
    ]:
        raise LiveCommitValidationError("release semantic contract is invalid")
    for field in (
        "semantic_source_sha256",
        "source_manifest_sha256",
        "tree_sha256",
    ):
        manifest[field] = _sha256(manifest[field], f"release artifact manifest.{field}")
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise LiveCommitValidationError("release artifact manifest files are absent")
    paths: set[str] = set()
    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(manifest["files"]):
        entry = _exact_dict(
            raw,
            {"hot_path", "mode", "path", "sha256"},
            f"release artifact file[{index}]",
        )
        path = entry["path"]
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or path in paths
        ):
            raise LiveCommitValidationError("release artifact file path is unsafe or duplicated")
        paths.add(path)
        if not isinstance(entry["hot_path"], bool) or entry["mode"] not in {"0600", "0700"}:
            raise LiveCommitValidationError("release artifact file metadata is invalid")
        entry["sha256"] = _sha256(
            entry["sha256"], f"release artifact file[{index}].sha256"
        )
        entries.append(entry)
    identity = [{"path": item["path"], "sha256": item["sha256"]} for item in entries]
    tree_sha256 = hashlib.sha256(
        (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    if tree_sha256 != manifest["tree_sha256"]:
        raise LiveCommitValidationError("release artifact tree identity differs")
    manifest["files"] = entries
    return artifact, manifest


def _require_release_member(
    runtime: dict[str, str],
    manifest_artifact: dict[str, str],
    manifest: dict[str, Any],
    label: str,
) -> None:
    matches = [
        entry for entry in manifest["files"] if entry["sha256"] == runtime["sha256"]
    ]
    if len(matches) != 1:
        raise LiveCommitValidationError(
            f"{label} is not exactly one authenticated release artifact member"
        )
    member = Path(manifest_artifact["path"]).parent / "files" / matches[0]["path"]
    authenticated = _artifact(
        {"path": str(member), "sha256": matches[0]["sha256"]},
        f"{label} release member",
    )
    if authenticated["sha256"] != runtime["sha256"]:
        raise LiveCommitValidationError(f"{label} release member identity differs")


def normalize_state(value: object, label: str = "state") -> dict[str, Any]:
    state = _exact_dict(value, STATE_KEYS, label)
    for field in STATE_DIGEST_FIELDS:
        state[field] = _sha256(state[field], f"{label}.{field}")
    for field, count in LAYER_DIGEST_FIELDS.items():
        items = state[field]
        if not isinstance(items, list) or len(items) != count:
            raise LiveCommitValidationError(f"{label}.{field} must contain {count} digests")
        state[field] = [
            _sha256(item, f"{label}.{field}[{index}]")
            for index, item in enumerate(items)
        ]
    semantic_units = _exact_dict(
        state["semantic_unit_sha256"],
        set(TRANSITION_UNITS),
        f"{label}.semantic_unit_sha256",
    )
    state["semantic_unit_sha256"] = {
        unit: _sha256(
            semantic_units[unit], f"{label}.semantic_unit_sha256.{unit}"
        )
        for unit in TRANSITION_UNITS
    }
    for field in ("allocator_generation", "commit_epoch", "logical_length"):
        item = state[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise LiveCommitValidationError(f"{label}.{field} must be a non-negative integer")
    for field in ("device_error_word", "nonfinite_count"):
        if state[field] != 0:
            raise LiveCommitValidationError(f"{label}.{field} must be zero")
    return state


def state_sha256(value: object) -> str:
    return _digest(normalize_state(value))


def first_state_difference(left: object, right: object) -> dict[str, Any] | None:
    left_state = normalize_state(left, "left state")
    right_state = normalize_state(right, "right state")
    for field in sorted(STATE_KEYS):
        if left_state[field] != right_state[field]:
            if field in LAYER_DIGEST_FIELDS:
                for index, (left_item, right_item) in enumerate(
                    zip(left_state[field], right_state[field], strict=True)
                ):
                    if left_item != right_item:
                        return {
                            "field": f"{field}[{index}]",
                            "left": left_item,
                            "right": right_item,
                        }
            return {"field": field, "left": left_state[field], "right": right_state[field]}
    return None


def seal_capability(value: object) -> dict[str, Any]:
    capability = _exact_dict(value, CAPABILITY_KEYS - {"capability_sha256"}, "capability")
    capability["semantic_source"] = _artifact(
        capability["semantic_source"], "capability.semantic_source"
    )
    manifest_artifact, release_manifest = _release_manifest(
        capability["release_artifact_manifest"]
    )
    capability["release_artifact_manifest"] = manifest_artifact
    if (
        release_manifest["source_manifest_sha256"]
        != capability["semantic_source"]["sha256"]
    ):
        raise LiveCommitValidationError(
            "release artifact is bound to a different semantic source manifest"
        )
    capability["controller_runtime"] = _artifact(
        capability["controller_runtime"], "capability.controller_runtime"
    )
    capability["enforcement_runtime"] = _artifact(
        capability["enforcement_runtime"], "capability.enforcement_runtime"
    )
    capability["driver_runtime"] = _artifact(
        capability["driver_runtime"], "capability.driver_runtime"
    )
    capability["model_output_fingerprint_runtime"] = _artifact(
        capability["model_output_fingerprint_runtime"],
        "capability.model_output_fingerprint_runtime",
    )
    capability["coordinator_runtime"] = _artifact(
        capability["coordinator_runtime"], "capability.coordinator_runtime"
    )
    capability["scheduler_runtime"] = _artifact(
        capability["scheduler_runtime"], "capability.scheduler_runtime"
    )
    capability["release_runtime"] = _artifact(
        capability["release_runtime"], "capability.release_runtime"
    )
    capability["serial_oracle_runtime"] = _artifact(
        capability["serial_oracle_runtime"], "capability.serial_oracle_runtime"
    )
    capability["model"] = _artifact(capability["model"], "capability.model")
    runtime_paths = {
        capability["controller_runtime"]["path"],
        capability["enforcement_runtime"]["path"],
        capability["driver_runtime"]["path"],
        capability["model_output_fingerprint_runtime"]["path"],
        capability["coordinator_runtime"]["path"],
        capability["scheduler_runtime"]["path"],
        capability["release_runtime"]["path"],
        capability["serial_oracle_runtime"]["path"],
    }
    if len(runtime_paths) != 8:
        raise LiveCommitValidationError("all controller/enforcement/runtime paths must differ")
    for key in (
        "controller_runtime",
        "enforcement_runtime",
        "driver_runtime",
        "model_output_fingerprint_runtime",
        "coordinator_runtime",
        "scheduler_runtime",
        "release_runtime",
        "serial_oracle_runtime",
    ):
        _require_release_member(
            capability[key],
            manifest_artifact,
            release_manifest,
            f"capability.{key}",
        )
    if capability["release_runtime"]["path"] == capability["serial_oracle_runtime"]["path"]:
        raise LiveCommitValidationError("release and serial oracle runtime paths must differ")
    if capability["release_runtime"]["sha256"] == capability["serial_oracle_runtime"]["sha256"]:
        raise LiveCommitValidationError("release and serial oracle runtime identities must differ")
    candidate_family = _identifier(
        capability["candidate_implementation_family"], "candidate implementation family"
    )
    serial_family = _identifier(
        capability["serial_implementation_family"], "serial implementation family"
    )
    if candidate_family == serial_family:
        raise LiveCommitValidationError("candidate and serial implementation families must differ")
    capability["hardware_runtime_contract_sha256"] = _sha256(
        capability["hardware_runtime_contract_sha256"], "hardware/runtime contract"
    )
    capability["compiler_contract_sha256"] = _sha256(
        capability["compiler_contract_sha256"], "compiler contract"
    )
    if capability["schema"] != CAPABILITY_SCHEMA:
        raise LiveCommitValidationError("capability schema differs")
    if capability["commit_counts"] != list(COMMIT_COUNTS):
        raise LiveCommitValidationError("capability must support commit counts 0 through 8")
    if capability["runtime_invariants"] != RUNTIME_INVARIANTS:
        raise LiveCommitValidationError("capability runtime invariants are incomplete")
    if capability["hook_boundaries"] != list(HOOK_BOUNDARIES):
        raise LiveCommitValidationError("capability hook boundaries are incomplete or reordered")
    if capability["classification"] != "runtime_invariant_not_finite_qualification":
        raise LiveCommitValidationError("capability classification is unsafe")
    if capability["universal_guarantee_mechanism"] != "independent_serial_before_every_commit":
        raise LiveCommitValidationError("capability does not select the live serial mechanism")
    capability["capability_sha256"] = _document_digest(capability, "capability_sha256")
    return capability


def verify_capability(value: object) -> dict[str, Any]:
    capability = _exact_dict(value, CAPABILITY_KEYS, "capability")
    expected = _sha256(capability["capability_sha256"], "capability.capability_sha256")
    unsigned = {key: item for key, item in capability.items() if key != "capability_sha256"}
    sealed = seal_capability(unsigned)
    if sealed["capability_sha256"] != expected:
        raise LiveCommitValidationError("capability self-hash differs")
    return sealed


def _normalize_difference(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    difference = _exact_dict(value, {"field", "left", "right"}, "first difference")
    _identifier(difference["field"], "first difference field")
    return difference


def _normalize_candidate_failure(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    failure = _exact_dict(value, CANDIDATE_FAILURE_KEYS, "candidate failure")
    failure["stage"] = _identifier(failure["stage"], "candidate failure stage")
    failure["exception_type"] = _identifier(
        failure["exception_type"], "candidate failure exception type"
    )
    failure["message_sha256"] = _sha256(
        failure["message_sha256"], "candidate failure message"
    )
    failure["traceback_sha256"] = _sha256(
        failure["traceback_sha256"], "candidate failure traceback"
    )
    return failure


def _token_ids(value: object, label: str, *, count: int) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in value
        )
    ):
        raise LiveCommitValidationError(f"{label} must contain exactly {count} token IDs")
    return list(value)


def seal_receipt(value: object) -> dict[str, Any]:
    receipt = _exact_dict(value, RECEIPT_KEYS - {"receipt_sha256"}, "receipt")
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise LiveCommitValidationError("receipt schema differs")
    receipt["capability_sha256"] = _sha256(
        receipt["capability_sha256"], "receipt.capability_sha256"
    )
    receipt["publication_intent_sha256"] = _sha256(
        receipt["publication_intent_sha256"], "receipt.publication_intent_sha256"
    )
    _identifier(receipt["session_id"], "receipt.session_id")
    _identifier(receipt["request_id"], "receipt.request_id")
    if (
        isinstance(receipt["round_index"], bool)
        or not isinstance(receipt["round_index"], int)
        or receipt["round_index"] < 0
    ):
        raise LiveCommitValidationError("receipt.round_index is invalid")
    if receipt["commit_count"] not in COMMIT_COUNTS:
        raise LiveCommitValidationError("receipt.commit_count is outside 0 through 8")
    receipt["canonical_before"] = normalize_state(
        receipt["canonical_before"], "receipt.canonical_before"
    )
    receipt["serial_state"] = normalize_state(receipt["serial_state"], "receipt.serial_state")
    receipt["canonical_after"] = normalize_state(
        receipt["canonical_after"], "receipt.canonical_after"
    )
    before_sha256 = _digest(receipt["canonical_before"])
    for field in ("canonical_probe_after_candidate_sha256", "canonical_probe_after_serial_sha256"):
        if _sha256(receipt[field], f"receipt.{field}") != before_sha256:
            raise LiveCommitValidationError(
                f"{field} proves canonical state changed before publication"
            )
    for field in (
        "publication_atomic",
        "candidate_private_state_destroyed",
        "serial_private_state_destroyed",
        "candidate_enabled_before",
        "candidate_enabled_after",
        "comparison_equal",
    ):
        if not isinstance(receipt[field], bool):
            raise LiveCommitValidationError(f"receipt.{field} must be boolean")
    if not receipt["publication_atomic"] or not receipt["serial_private_state_destroyed"]:
        raise LiveCommitValidationError("receipt does not prove atomic publication/private cleanup")

    mode = receipt["mode"]
    candidate_state = receipt["candidate_state"]
    candidate_failure = _normalize_candidate_failure(receipt["candidate_failure"])
    receipt["candidate_failure"] = candidate_failure
    difference = _normalize_difference(receipt["first_difference"])
    counterexample = receipt["counterexample_sha256"]
    publication_source = receipt["publication_source"]
    commit_count = receipt["commit_count"]
    serial_token_ids = _token_ids(
        receipt["serial_token_ids"], "receipt.serial_token_ids", count=commit_count
    )
    receipt["serial_token_ids"] = serial_token_ids
    expected_serial_length = receipt["canonical_before"]["logical_length"] + commit_count
    if receipt["serial_state"]["logical_length"] != expected_serial_length:
        raise LiveCommitValidationError("serial state did not advance by the declared commit count")
    expected_serial_epoch = receipt["canonical_before"]["commit_epoch"] + (
        commit_count > 0
    )
    if receipt["serial_state"]["commit_epoch"] != expected_serial_epoch:
        raise LiveCommitValidationError("serial state has the wrong commit epoch")
    if commit_count == 0 and receipt["serial_state"] != receipt["canonical_before"]:
        raise LiveCommitValidationError("zero-commit serial state differs from canonical input")
    if (
        commit_count > 0
        and receipt["serial_state"]["committed_token_prefix_sha256"]
        == receipt["canonical_before"]["committed_token_prefix_sha256"]
    ):
        raise LiveCommitValidationError("serial state did not extend the token-prefix identity")

    if mode == "validated_candidate":
        if (
            not receipt["candidate_enabled_before"]
            or candidate_state is None
            or candidate_failure is not None
        ):
            raise LiveCommitValidationError(
                "validated-candidate receipt lacks a candidate execution"
            )
        candidate = normalize_state(candidate_state, "receipt.candidate_state")
        receipt["candidate_state"] = candidate
        candidate_token_ids = _token_ids(
            receipt["candidate_token_ids"],
            "receipt.candidate_token_ids",
            count=commit_count,
        )
        receipt["candidate_token_ids"] = candidate_token_ids
        actual_difference = (
            {
                "field": "committed_token_ids",
                "left": candidate_token_ids,
                "right": serial_token_ids,
            }
            if candidate_token_ids != serial_token_ids
            else first_state_difference(candidate, receipt["serial_state"])
        )
        actual_equal = actual_difference is None
        if receipt["comparison_equal"] != actual_equal or difference != actual_difference:
            raise LiveCommitValidationError(
                "receipt comparison result is not the complete state result"
            )
        if actual_equal:
            if counterexample is not None or not receipt["candidate_enabled_after"]:
                raise LiveCommitValidationError("equal candidate was incorrectly quarantined")
            expected_source = "unchanged" if commit_count == 0 else "candidate"
            expected_after = receipt["canonical_before"] if commit_count == 0 else candidate
        else:
            _sha256(counterexample, "receipt.counterexample_sha256")
            if receipt["candidate_enabled_after"]:
                raise LiveCommitValidationError("mismatching candidate was not quarantined")
            expected_source = "unchanged" if commit_count == 0 else "serial"
            expected_after = (
                receipt["canonical_before"] if commit_count == 0 else receipt["serial_state"]
            )
        if not receipt["candidate_private_state_destroyed"]:
            raise LiveCommitValidationError("candidate private state was not destroyed")
    elif mode == "candidate_fault_serial":
        if (
            not receipt["candidate_enabled_before"]
            or receipt["candidate_enabled_after"]
            or candidate_failure is None
        ):
            raise LiveCommitValidationError(
                "candidate-fault receipt has an invalid quarantine transition"
            )
        if (
            candidate_state is not None
            or receipt["candidate_token_ids"] is not None
            or receipt["comparison_equal"]
            or difference is not None
        ):
            raise LiveCommitValidationError(
                "candidate-fault receipt contains a completed candidate comparison"
            )
        _sha256(counterexample, "receipt.counterexample_sha256")
        expected_source = "unchanged" if commit_count == 0 else "serial"
        expected_after = (
            receipt["canonical_before"] if commit_count == 0 else receipt["serial_state"]
        )
        if not receipt["candidate_private_state_destroyed"]:
            raise LiveCommitValidationError("failed candidate private state was not destroyed")
    elif mode == "quarantined_serial":
        if receipt["candidate_enabled_before"] or receipt["candidate_enabled_after"]:
            raise LiveCommitValidationError("quarantined-serial receipt re-enabled the candidate")
        if (
            candidate_state is not None
            or receipt["candidate_token_ids"] is not None
            or candidate_failure is not None
            or receipt["comparison_equal"]
            or difference is not None
        ):
            raise LiveCommitValidationError(
                "quarantined-serial receipt contains a candidate comparison"
            )
        _sha256(counterexample, "receipt.counterexample_sha256")
        expected_source = "unchanged" if commit_count == 0 else "serial"
        expected_after = (
            receipt["canonical_before"] if commit_count == 0 else receipt["serial_state"]
        )
    else:
        raise LiveCommitValidationError("receipt mode is unsupported")

    if publication_source != expected_source or receipt["canonical_after"] != expected_after:
        raise LiveCommitValidationError(
            "canonical publication does not equal the safe selected state"
        )
    expected_epoch = receipt["canonical_before"]["commit_epoch"] + (commit_count > 0)
    if receipt["canonical_after"]["commit_epoch"] != expected_epoch:
        raise LiveCommitValidationError("canonical commit epoch did not advance exactly once")

    comparison_payload = {
        "candidate_state_sha256": (
            None if receipt["candidate_state"] is None else _digest(receipt["candidate_state"])
        ),
        "candidate_token_ids": receipt["candidate_token_ids"],
        "candidate_failure": receipt["candidate_failure"],
        "serial_state_sha256": _digest(receipt["serial_state"]),
        "serial_token_ids": receipt["serial_token_ids"],
        "comparison_equal": receipt["comparison_equal"],
        "first_difference": receipt["first_difference"],
    }
    expected_comparison = _digest(comparison_payload)
    if _sha256(receipt["comparison_sha256"], "receipt.comparison_sha256") != expected_comparison:
        raise LiveCommitValidationError("receipt comparison SHA-256 differs")
    receipt["receipt_sha256"] = _document_digest(receipt, "receipt_sha256")
    return receipt


def verify_receipt(value: object) -> dict[str, Any]:
    receipt = _exact_dict(value, RECEIPT_KEYS, "receipt")
    expected = _sha256(receipt["receipt_sha256"], "receipt.receipt_sha256")
    sealed = seal_receipt({key: item for key, item in receipt.items() if key != "receipt_sha256"})
    if sealed["receipt_sha256"] != expected:
        raise LiveCommitValidationError("receipt self-hash differs")
    return sealed


def seal_ledger(value: object) -> dict[str, Any]:
    ledger = _exact_dict(value, LEDGER_KEYS - {"ledger_sha256"}, "ledger")
    if ledger["schema"] != LEDGER_SCHEMA:
        raise LiveCommitValidationError("ledger schema differs")
    ledger["capability_sha256"] = _sha256(
        ledger["capability_sha256"], "ledger.capability_sha256"
    )
    session_id = _identifier(ledger["session_id"], "ledger.session_id")
    initial = normalize_state(ledger["initial_state"], "ledger.initial_state")
    final = normalize_state(ledger["final_state"], "ledger.final_state")
    if not isinstance(ledger["receipts"], list):
        raise LiveCommitValidationError("ledger.receipts must be a list")
    receipts = [verify_receipt(item) for item in ledger["receipts"]]
    previous = initial
    quarantined = False
    for index, receipt in enumerate(receipts):
        if receipt["session_id"] != session_id or receipt["capability_sha256"] != ledger[
            "capability_sha256"
        ]:
            raise LiveCommitValidationError("ledger receipt identity differs")
        if receipt["round_index"] != index:
            raise LiveCommitValidationError("ledger receipt round indexes are not contiguous")
        if receipt["canonical_before"] != previous:
            raise LiveCommitValidationError("ledger canonical state chain is broken")
        if quarantined and receipt["mode"] != "quarantined_serial":
            raise LiveCommitValidationError("ledger executed a candidate after quarantine")
        quarantined = quarantined or not receipt["candidate_enabled_after"]
        previous = receipt["canonical_after"]
    if previous != final:
        raise LiveCommitValidationError("ledger final state differs from its receipt chain")
    if ledger["candidate_quarantined"] is not quarantined:
        raise LiveCommitValidationError("ledger quarantine verdict differs from its receipts")
    if ledger["request_complete"] is not True:
        raise LiveCommitValidationError(
            "incomplete request ledger cannot satisfy the live invariant"
        )
    if ledger["classification"] != "runtime_invariant_not_finite_qualification":
        raise LiveCommitValidationError("ledger classification is unsafe")
    if ledger["universal_guarantee_mechanism"] != "independent_serial_before_every_commit":
        raise LiveCommitValidationError("ledger universal mechanism differs")
    ledger["initial_state"] = initial
    ledger["receipts"] = receipts
    ledger["final_state"] = final
    ledger["ledger_sha256"] = _document_digest(ledger, "ledger_sha256")
    return ledger


def verify_ledger(value: object) -> dict[str, Any]:
    ledger = _exact_dict(value, LEDGER_KEYS, "ledger")
    expected = _sha256(ledger["ledger_sha256"], "ledger.ledger_sha256")
    sealed = seal_ledger({key: item for key, item in ledger.items() if key != "ledger_sha256"})
    if sealed["ledger_sha256"] != expected:
        raise LiveCommitValidationError("ledger self-hash differs")
    return sealed


def _load(path: Path) -> object:
    status = path.lstat()
    if (
        not stat.S_ISREG(status.st_mode)
        or path.is_symlink()
        or status.st_uid != os.getuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise LiveCommitValidationError("input must be an owner-only regular file")
    return json.loads(path.read_bytes())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-live-commit-validation",
        description="Verify release-bound independent serial-before-commit evidence.",
    )
    parser.add_argument("kind", choices=("capability", "receipt", "ledger"))
    parser.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    verifier = {
        "capability": verify_capability,
        "receipt": verify_receipt,
        "ledger": verify_ledger,
    }[args.kind]
    try:
        result = verifier(_load(args.path))
    except (OSError, UnicodeError, json.JSONDecodeError, LiveCommitValidationError) as error:
        print(f"qwen-live-commit-validation: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
