# QWEN_ASSURANCE_ONLY_BEGIN: round-equivalence-certificate
"""Strict instance certificates for serial/M8 one-round refinement.

This contract does not infer universality from one fixture.  It makes every
supplied canonical state independently checkable and records the exact module
where an instance violates ``G(S, proposals, c) = F^c(S)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "urn:qwen-r9700:round-equivalence-evidence:v2"
CERTIFICATE_SCHEMA = "urn:qwen-r9700:round-equivalence-certificate:v4"
PRECOMMIT_HEADER_SCHEMA = "urn:qwen-r9700:round-equivalence-precommit-header:v2"
PRECOMMIT_EVENT_SCHEMA = "urn:qwen-r9700:round-equivalence-precommit:v4"
FORCED_PROPOSAL_SCHEMA = "urn:qwen-r9700:round-equivalence-proposal:v1"
COMMIT_COUNTS = tuple(range(9))
BOUNDED_ASSURANCE_CLASSIFICATION = "bounded_qualification"
DIFFERENTIAL_PROOF_BASIS = "independent_differential_execution"
ASSURANCE_CLAIM_KEYS = {
    "assurance_classification",
    "live_serial_validation_before_commit",
    "machine_checked_complete_domain",
    "proof_basis",
    "universal_equivalence_proven",
}
MODULE_ORDER = (
    "snapshot_restore",
    "m8_construction_positions",
    "w4a16_projections",
    "quest_scoring",
    "quest_top96_selection",
    "quest_union_visibility",
    "quest_historical_attention",
    "quest_causal_tail_attention",
    "quest_softmax_reduction",
    "provisional_gdn_convolution",
    "provisional_gdn_recurrence",
    "gdn_gated_rmsnorm",
    "residual_stream",
    "swiglu_activation",
    "lm_head",
    "target_verification",
    "provisional_state_isolation",
    "atomic_commit",
)
SNAPSHOT_KEYS = {
    "cache_generations_sha256",
    "cache_mapping_sha256",
    "cache_ownership_sha256",
    "canonical_gdn_layer_sha256",
    "committed_token_prefix_sha256",
    "convolution_layer_sha256",
    "decoding_state_sha256",
    "device_error_word",
    "draft_kv_scales_sha256",
    "draft_kv_sha256",
    "logical_length",
    "nonfinite_count",
    "positions_sha256",
    "quest_metadata_sha256",
    "rollback_generation",
    "rope_indices_sha256",
    "target_kv_scales_sha256",
    "target_kv_sha256",
}
BOUNDARY_KEYS = {"input_sha256", "module", "output_sha256", "state_sha256"}
TRANSITION_KEYS = {
    "commit_count",
    "committed_token_ids",
    "module_boundaries",
    "ordered_quest_pages_sha256",
    "positions_sha256",
    "quest_visibility_sha256",
    "rope_indices_sha256",
    "state",
}
SIBLING_KEYS = {
    "adversarial_row_sha256",
    "baseline_row_sha256",
    "mutated_rows",
    "protected_row",
}
ROW_PERMUTATION_KEYS = {
    "baseline_row_sha256",
    "permutation",
    "restored_row_sha256",
}
EVIDENCE_KEYS = {
    "canonical_state",
    "proposal_token_ids",
    "provisional_after",
    "provisional_before",
    "runtime_artifact_manifest_sha256",
    "row_permutation_trials",
    "schema",
    "semantic_source_sha256",
    "serial_transitions",
    "sibling_trials",
    "speculative_transitions",
}
AUTHORITATIVE_STATE_KEYS = {
    "cache_mapping_sha256",
    "canonical_gdn_layer_sha256",
    "convolution_layer_sha256",
    "device_error_word",
    "draft_kv_scales_sha256",
    "draft_kv_sha256",
    "nonfinite_count",
    "payload_producer_receipt_sha256",
    "rollback_generation",
    "target_kv_scales_sha256",
    "target_kv_sha256",
}
PRECOMMIT_EVENT_KEYS = {
    "authoritative_state",
    "event_sha256",
    "external_commit_count",
    "execution_mode",
    "phase",
    "positions",
    "post_target_authoritative_state",
    "request_id",
    "round_index",
    "sampler_approved_commit_count",
    "schema",
    "sequence_length",
    "target_argmax_token_ids",
    "target_logits_sha256",
    "transition_row_token_ids",
}
FORCED_PROPOSAL_KEYS = {
    "document_sha256",
    "prompt_tokens",
    "proposal_token_ids",
    "request_id",
    "runtime_artifact_manifest_sha256",
    "schema",
    "semantic_source_sha256",
    "snapshot_manifest_sha256",
}


class RoundEquivalenceError(RuntimeError):
    """Evidence is malformed or cannot establish one-round refinement."""


def bounded_assurance_claim() -> dict[str, object]:
    """Return the only claim supported by finite differential evidence.

    A passing instance or campaign proves the supplied executions matched.  It
    cannot establish the universal transition contract without either a
    machine-checked complete-domain construction or serial validation before
    every live commit.
    """

    return {
        "assurance_classification": BOUNDED_ASSURANCE_CLASSIFICATION,
        "live_serial_validation_before_commit": False,
        "machine_checked_complete_domain": False,
        "proof_basis": DIFFERENTIAL_PROOF_BASIS,
        "universal_equivalence_proven": False,
    }


def validate_bounded_assurance_claim(value: object, label: str) -> None:
    """Reject a finite artifact that represents itself as universal proof."""

    if not isinstance(value, dict):
        raise RoundEquivalenceError(f"{label} must be an object")
    claim = {key: value.get(key) for key in ASSURANCE_CLAIM_KEYS}
    if claim != bounded_assurance_claim():
        if value.get("universal_equivalence_proven") is True:
            raise RoundEquivalenceError(f"{label} makes an unsupported universal equivalence claim")
        raise RoundEquivalenceError(f"{label} bounded assurance claim is invalid")


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _document_digest(value: dict[str, Any], digest_key: str) -> str:
    unsigned = dict(value)
    unsigned.pop(digest_key, None)
    return _sha256(_canonical_json(unsigned))


def _line_digest(value: dict[str, Any], digest_key: str) -> str:
    unsigned = dict(value)
    unsigned.pop(digest_key, None)
    payload = json.dumps(unsigned, separators=(",", ":"), sort_keys=True).encode()
    return _sha256(payload)


def _exact_dict(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RoundEquivalenceError(f"{label} must be an object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise RoundEquivalenceError(f"{label} keys differ: missing={missing} extra={extra}")
    return dict(value)


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RoundEquivalenceError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise RoundEquivalenceError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _token_ids(value: object, label: str, *, length: int) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise RoundEquivalenceError(f"{label} must contain exactly {length} token IDs")
    result: list[int] = []
    for index, token in enumerate(value):
        result.append(_integer(token, f"{label}[{index}]", 0, 253_951))
    return result


def _layer_hashes(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 48:
        raise RoundEquivalenceError(f"{label} must contain exactly 48 layer hashes")
    return [_digest(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _authoritative_state(value: object, label: str) -> dict[str, Any]:
    state = _exact_dict(value, AUTHORITATIVE_STATE_KEYS, label)
    state["canonical_gdn_layer_sha256"] = _layer_hashes(
        state["canonical_gdn_layer_sha256"], f"{label}.canonical_gdn_layer_sha256"
    )
    state["convolution_layer_sha256"] = _layer_hashes(
        state["convolution_layer_sha256"], f"{label}.convolution_layer_sha256"
    )
    state["device_error_word"] = _integer(
        state["device_error_word"], f"{label}.device_error_word", 0, 2**32 - 1
    )
    state["nonfinite_count"] = _integer(
        state["nonfinite_count"], f"{label}.nonfinite_count", 0, 2**63 - 1
    )
    state["rollback_generation"] = _integer(
        state["rollback_generation"], f"{label}.rollback_generation", 0, 2**31 - 1
    )
    if state["device_error_word"] != 0 or state["nonfinite_count"] != 0:
        raise RoundEquivalenceError(f"{label} reports a device error or nonfinite value")
    for key in AUTHORITATIVE_STATE_KEYS - {
        "canonical_gdn_layer_sha256",
        "convolution_layer_sha256",
        "device_error_word",
        "nonfinite_count",
        "rollback_generation",
    }:
        state[key] = _digest(state[key], f"{label}.{key}")
    return state


def normalize_precommit_event(value: object) -> dict[str, Any]:
    """Validate one state witness captured after M8 and before canonical commit."""

    event = _exact_dict(value, PRECOMMIT_EVENT_KEYS, "precommit event")
    if event["schema"] != PRECOMMIT_EVENT_SCHEMA:
        raise RoundEquivalenceError("precommit event schema mismatch")
    if event["phase"] != "after-target-before-commit":
        raise RoundEquivalenceError("precommit event phase mismatch")
    execution_mode = event["execution_mode"]
    if execution_mode not in {"serial-m1", "speculative-m8"}:
        raise RoundEquivalenceError("precommit event execution mode is invalid")
    rows = 1 if execution_mode == "serial-m1" else 8
    request_id = event["request_id"]
    if (
        not isinstance(request_id, str)
        or not request_id
        or any(character.isspace() for character in request_id)
    ):
        raise RoundEquivalenceError("precommit event request_id is invalid")
    event["round_index"] = _integer(
        event["round_index"], "precommit event.round_index", 0, 1_000_000
    )
    event["sequence_length"] = _integer(
        event["sequence_length"], "precommit event.sequence_length", 1, 1_000_000
    )
    sampler_approved = _integer(
        event["sampler_approved_commit_count"],
        "precommit event.sampler_approved_commit_count",
        1,
        rows,
    )
    event["sampler_approved_commit_count"] = sampler_approved
    external_commit_count = _integer(
        event["external_commit_count"],
        "precommit event.external_commit_count",
        0,
        sampler_approved,
    )
    event["external_commit_count"] = external_commit_count
    if execution_mode == "serial-m1" and external_commit_count != 1:
        raise RoundEquivalenceError("serial M1 precommit must publish exactly one transition")
    event["transition_row_token_ids"] = _token_ids(
        event["transition_row_token_ids"],
        "precommit event.transition_row_token_ids",
        length=rows,
    )
    event["target_argmax_token_ids"] = _token_ids(
        event["target_argmax_token_ids"],
        "precommit event.target_argmax_token_ids",
        length=rows,
    )
    logits = event["target_logits_sha256"]
    if not isinstance(logits, list) or len(logits) != rows:
        raise RoundEquivalenceError(
            "precommit event.target_logits_sha256 must have one digest per transition row"
        )
    event["target_logits_sha256"] = [
        _digest(digest, f"precommit event.target_logits_sha256[{row}]")
        for row, digest in enumerate(logits)
    ]
    positions = event["positions"]
    if (
        not isinstance(positions, list)
        or len(positions) != rows
        or any(
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < 1_000_000
            for position in positions
        )
        or positions != list(range(positions[0], positions[0] + rows))
        or positions[0] != event["sequence_length"]
    ):
        raise RoundEquivalenceError(
            "precommit event positions must be consecutive absolute transition positions at S"
        )
    event["authoritative_state"] = _authoritative_state(
        event["authoritative_state"], "precommit event.authoritative_state"
    )
    event["post_target_authoritative_state"] = _authoritative_state(
        event["post_target_authoritative_state"],
        "precommit event.post_target_authoritative_state",
    )
    if (
        execution_mode == "speculative-m8"
        and event["post_target_authoritative_state"] != event["authoritative_state"]
    ):
        differing = next(
            key
            for key in sorted(AUTHORITATIVE_STATE_KEYS)
            if event["post_target_authoritative_state"][key] != event["authoritative_state"][key]
        )
        raise RoundEquivalenceError(
            "speculative target execution mutated canonical state before commit at " + differing
        )
    event["event_sha256"] = _digest(event["event_sha256"], "precommit event.event_sha256")
    if event["event_sha256"] != _line_digest(event, "event_sha256"):
        raise RoundEquivalenceError("precommit event self-hash mismatch")
    return event


def seal_precommit_event(value: object) -> dict[str, Any]:
    event = dict(_exact_dict(value, PRECOMMIT_EVENT_KEYS - {"event_sha256"}, "precommit event"))
    event["event_sha256"] = _line_digest(event, "event_sha256")
    return normalize_precommit_event(event)


def create_precommit_stream(path: Path, header: object) -> int:
    """Create an owner-only, create-once JSONL stream for precommit witnesses."""

    header_value = _exact_dict(
        header,
        {"capture", "max_rounds", "prompt_tokens", "request_id", "runtime_files"},
        "precommit header",
    )
    header_value = {**header_value, "schema": PRECOMMIT_HEADER_SCHEMA}
    path = path.absolute()
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or path.parent.is_symlink()
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) & 0o077
    ):
        raise RoundEquivalenceError("precommit stream parent must be an owner-only directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        _append_json_line(descriptor, {"header": header_value})
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _append_json_line(descriptor: int, value: object) -> None:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    offset = 0
    while offset < len(payload):
        offset += os.write(descriptor, payload[offset:])
    os.fsync(descriptor)


def append_precommit_event(descriptor: int, value: object) -> None:
    _append_json_line(descriptor, {"event": seal_precommit_event(value)})


def normalize_forced_proposal(value: object) -> dict[str, Any]:
    """Validate one assurance-only proposal set for an exact D7/M8 instance."""

    proposal = _exact_dict(value, FORCED_PROPOSAL_KEYS, "forced proposal")
    if proposal["schema"] != FORCED_PROPOSAL_SCHEMA:
        raise RoundEquivalenceError("forced proposal schema mismatch")
    request_id = proposal["request_id"]
    if (
        not isinstance(request_id, str)
        or not request_id
        or any(character.isspace() for character in request_id)
    ):
        raise RoundEquivalenceError("forced proposal request_id is invalid")
    proposal["prompt_tokens"] = _integer(
        proposal["prompt_tokens"], "forced proposal.prompt_tokens", 1, 253_792
    )
    proposal["proposal_token_ids"] = _token_ids(
        proposal["proposal_token_ids"], "forced proposal.proposal_token_ids", length=7
    )
    for key in (
        "document_sha256",
        "runtime_artifact_manifest_sha256",
        "semantic_source_sha256",
        "snapshot_manifest_sha256",
    ):
        proposal[key] = _digest(proposal[key], f"forced proposal.{key}")
    if proposal["document_sha256"] != _document_digest(proposal, "document_sha256"):
        raise RoundEquivalenceError("forced proposal self-hash mismatch")
    return proposal


def seal_forced_proposal(value: object) -> dict[str, Any]:
    proposal = dict(
        _exact_dict(value, FORCED_PROPOSAL_KEYS - {"document_sha256"}, "forced proposal")
    )
    proposal["document_sha256"] = _document_digest(proposal, "document_sha256")
    return normalize_forced_proposal(proposal)


def _snapshot(value: object, label: str) -> dict[str, Any]:
    snapshot = _exact_dict(value, SNAPSHOT_KEYS, label)
    snapshot["logical_length"] = _integer(
        snapshot["logical_length"], f"{label}.logical_length", 1, 1_000_000
    )
    snapshot["rollback_generation"] = _integer(
        snapshot["rollback_generation"], f"{label}.rollback_generation", 0, 2**31 - 1
    )
    snapshot["device_error_word"] = _integer(
        snapshot["device_error_word"], f"{label}.device_error_word", 0, 2**32 - 1
    )
    snapshot["nonfinite_count"] = _integer(
        snapshot["nonfinite_count"], f"{label}.nonfinite_count", 0, 2**63 - 1
    )
    if snapshot["device_error_word"] != 0 or snapshot["nonfinite_count"] != 0:
        raise RoundEquivalenceError(f"{label} reports a device error or nonfinite value")
    snapshot["canonical_gdn_layer_sha256"] = _layer_hashes(
        snapshot["canonical_gdn_layer_sha256"], f"{label}.canonical_gdn_layer_sha256"
    )
    snapshot["convolution_layer_sha256"] = _layer_hashes(
        snapshot["convolution_layer_sha256"], f"{label}.convolution_layer_sha256"
    )
    for key in SNAPSHOT_KEYS - {
        "canonical_gdn_layer_sha256",
        "convolution_layer_sha256",
        "device_error_word",
        "logical_length",
        "nonfinite_count",
        "rollback_generation",
    }:
        snapshot[key] = _digest(snapshot[key], f"{label}.{key}")
    return snapshot


def _boundaries(value: object, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != len(MODULE_ORDER):
        raise RoundEquivalenceError(
            f"{label} must contain all {len(MODULE_ORDER)} ordered semantic modules"
        )
    result: list[dict[str, str]] = []
    for index, (raw, expected_module) in enumerate(zip(value, MODULE_ORDER, strict=True)):
        boundary = _exact_dict(raw, BOUNDARY_KEYS, f"{label}[{index}]")
        if boundary["module"] != expected_module:
            raise RoundEquivalenceError(f"{label}[{index}] must describe module {expected_module}")
        for key in BOUNDARY_KEYS - {"module"}:
            boundary[key] = _digest(boundary[key], f"{label}[{index}].{key}")
        result.append(boundary)
    return result


def _transition(value: object, label: str, base_length: int) -> dict[str, Any]:
    transition = _exact_dict(value, TRANSITION_KEYS, label)
    commit_count = _integer(transition["commit_count"], f"{label}.commit_count", 0, 8)
    transition["commit_count"] = commit_count
    transition["committed_token_ids"] = _token_ids(
        transition["committed_token_ids"],
        f"{label}.committed_token_ids",
        length=commit_count,
    )
    transition["module_boundaries"] = _boundaries(
        transition["module_boundaries"], f"{label}.module_boundaries"
    )
    for key in (
        "ordered_quest_pages_sha256",
        "positions_sha256",
        "quest_visibility_sha256",
        "rope_indices_sha256",
    ):
        transition[key] = _digest(transition[key], f"{label}.{key}")
    transition["state"] = _snapshot(transition["state"], f"{label}.state")
    if transition["state"]["logical_length"] != base_length + commit_count:
        raise RoundEquivalenceError(f"{label} logical length does not equal S length plus c")
    return transition


def _transitions(value: object, label: str, base_length: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(COMMIT_COUNTS):
        raise RoundEquivalenceError(f"{label} must contain commit counts 0 through 8")
    result = [
        _transition(item, f"{label}[{index}]", base_length) for index, item in enumerate(value)
    ]
    if tuple(item["commit_count"] for item in result) != COMMIT_COUNTS:
        raise RoundEquivalenceError(f"{label} commit counts must be ordered 0 through 8")
    return result


def _sibling_trials(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RoundEquivalenceError("sibling_trials must be an array")
    result: list[dict[str, Any]] = []
    covered: set[int] = set()
    for index, raw in enumerate(value):
        trial = _exact_dict(raw, SIBLING_KEYS, f"sibling_trials[{index}]")
        protected = _integer(trial["protected_row"], f"sibling_trials[{index}].protected_row", 0, 6)
        mutated = trial["mutated_rows"]
        if (
            not isinstance(mutated, list)
            or not mutated
            or any(
                isinstance(row, bool) or not isinstance(row, int) or not protected < row <= 7
                for row in mutated
            )
            or mutated != sorted(set(mutated))
        ):
            raise RoundEquivalenceError(
                f"sibling_trials[{index}] may mutate only later non-visible rows"
            )
        trial["baseline_row_sha256"] = _digest(
            trial["baseline_row_sha256"], f"sibling_trials[{index}].baseline_row_sha256"
        )
        trial["adversarial_row_sha256"] = _digest(
            trial["adversarial_row_sha256"],
            f"sibling_trials[{index}].adversarial_row_sha256",
        )
        trial["protected_row"] = protected
        covered.add(protected)
        result.append(trial)
    if covered != set(range(7)):
        raise RoundEquivalenceError("sibling_trials must cover protected rows 0 through 6")
    return result


def _row_permutation_trials(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RoundEquivalenceError("row_permutation_trials must be a non-empty array")
    result: list[dict[str, Any]] = []
    identity = list(range(8))
    for index, raw in enumerate(value):
        trial = _exact_dict(raw, ROW_PERMUTATION_KEYS, f"row_permutation_trials[{index}]")
        permutation = trial["permutation"]
        if (
            not isinstance(permutation, list)
            or len(permutation) != 8
            or permutation == identity
            or any(isinstance(row, bool) or not isinstance(row, int) for row in permutation)
            or sorted(permutation) != identity
        ):
            raise RoundEquivalenceError(
                f"row_permutation_trials[{index}].permutation must reorder rows 0 through 7"
            )
        for key in ("baseline_row_sha256", "restored_row_sha256"):
            hashes = trial[key]
            if not isinstance(hashes, list) or len(hashes) != 8:
                raise RoundEquivalenceError(
                    f"row_permutation_trials[{index}].{key} must contain eight hashes"
                )
            trial[key] = [
                _digest(item, f"row_permutation_trials[{index}].{key}[{row}]")
                for row, item in enumerate(hashes)
            ]
        result.append(trial)
    return result


def validate_evidence(value: object) -> dict[str, Any]:
    evidence = _exact_dict(value, EVIDENCE_KEYS, "round evidence")
    if evidence["schema"] != EVIDENCE_SCHEMA:
        raise RoundEquivalenceError("round evidence schema mismatch")
    for key in ("runtime_artifact_manifest_sha256", "semantic_source_sha256"):
        evidence[key] = _digest(evidence[key], f"round evidence.{key}")
    evidence["proposal_token_ids"] = _token_ids(
        evidence["proposal_token_ids"], "proposal_token_ids", length=7
    )
    evidence["canonical_state"] = _snapshot(evidence["canonical_state"], "canonical_state")
    evidence["provisional_before"] = _snapshot(evidence["provisional_before"], "provisional_before")
    evidence["provisional_after"] = _snapshot(evidence["provisional_after"], "provisional_after")
    base_length = evidence["canonical_state"]["logical_length"]
    evidence["serial_transitions"] = _transitions(
        evidence["serial_transitions"], "serial_transitions", base_length
    )
    evidence["speculative_transitions"] = _transitions(
        evidence["speculative_transitions"], "speculative_transitions", base_length
    )
    evidence["sibling_trials"] = _sibling_trials(evidence["sibling_trials"])
    evidence["row_permutation_trials"] = _row_permutation_trials(evidence["row_permutation_trials"])
    return evidence


def _first_mapping_difference(
    left: dict[str, Any], right: dict[str, Any], *, prefix: str
) -> dict[str, Any] | None:
    for key in sorted(left):
        if left[key] != right[key]:
            return {
                "field": f"{prefix}.{key}",
                "left_value": left[key],
                "right_value": right[key],
            }
    return None


def diagnose_evidence(value: object) -> dict[str, Any]:
    evidence = validate_evidence(value)
    first_difference: dict[str, Any] | None = None
    if evidence["canonical_state"] != evidence["provisional_before"]:
        first_difference = _first_mapping_difference(
            evidence["canonical_state"],
            evidence["provisional_before"],
            prefix="provisional_before",
        )
        assert first_difference is not None
        first_difference["module"] = "snapshot_restore"
    elif evidence["provisional_before"] != evidence["provisional_after"]:
        first_difference = _first_mapping_difference(
            evidence["provisional_before"],
            evidence["provisional_after"],
            prefix="provisional_after",
        )
        assert first_difference is not None
        first_difference["module"] = "provisional_state_isolation"
    elif evidence["serial_transitions"][0]["state"] != evidence["canonical_state"]:
        first_difference = _first_mapping_difference(
            evidence["canonical_state"],
            evidence["serial_transitions"][0]["state"],
            prefix="serial_F0_state",
        )
        assert first_difference is not None
        first_difference.update({"commit_count": 0, "module": "snapshot_restore"})
    else:
        for serial, speculative in zip(
            evidence["serial_transitions"], evidence["speculative_transitions"], strict=True
        ):
            commit_count = serial["commit_count"]
            for field, module in (
                ("committed_token_ids", "target_verification"),
                ("positions_sha256", "m8_construction_positions"),
                ("rope_indices_sha256", "m8_construction_positions"),
                ("ordered_quest_pages_sha256", "quest_top96_selection"),
                ("quest_visibility_sha256", "quest_union_visibility"),
            ):
                if serial[field] != speculative[field]:
                    first_difference = {
                        "commit_count": commit_count,
                        "field": field,
                        "left_value": serial[field],
                        "module": module,
                        "right_value": speculative[field],
                    }
                    break
            if first_difference is not None:
                break
            for left_boundary, right_boundary in zip(
                serial["module_boundaries"],
                speculative["module_boundaries"],
                strict=True,
            ):
                if left_boundary != right_boundary:
                    first_difference = _first_mapping_difference(
                        left_boundary,
                        right_boundary,
                        prefix="module_boundary",
                    )
                    assert first_difference is not None
                    first_difference.update(
                        {"commit_count": commit_count, "module": left_boundary["module"]}
                    )
                    break
            if first_difference is not None:
                break
            if serial["state"] != speculative["state"]:
                first_difference = _first_mapping_difference(
                    serial["state"], speculative["state"], prefix="committed_state"
                )
                assert first_difference is not None
                state_field = first_difference["field"].removeprefix("committed_state.")
                first_difference.update(
                    {
                        "commit_count": commit_count,
                        "module": {
                            "canonical_gdn_layer_sha256": "provisional_gdn_recurrence",
                            "convolution_layer_sha256": "provisional_gdn_convolution",
                        }.get(state_field, "atomic_commit"),
                    }
                )
                break
    if first_difference is None:
        for lane in ("serial_transitions", "speculative_transitions"):
            transitions = evidence[lane]
            for commit_count in range(8):
                prefix = transitions[commit_count]["committed_token_ids"]
                extended = transitions[commit_count + 1]["committed_token_ids"]
                if prefix != extended[:commit_count]:
                    first_difference = {
                        "commit_count": commit_count + 1,
                        "field": f"{lane}.committed_token_ids",
                        "left_value": prefix,
                        "module": "target_verification",
                        "right_value": extended,
                    }
                    break
            if first_difference is not None:
                break
    if first_difference is None:
        for index, trial in enumerate(evidence["sibling_trials"]):
            if trial["baseline_row_sha256"] != trial["adversarial_row_sha256"]:
                first_difference = {
                    "field": f"sibling_trials[{index}].protected_row",
                    "left_value": trial["baseline_row_sha256"],
                    "module": "m8_construction_positions",
                    "right_value": trial["adversarial_row_sha256"],
                }
                break
    if first_difference is None:
        for trial_index, trial in enumerate(evidence["row_permutation_trials"]):
            for row, (baseline, restored) in enumerate(
                zip(
                    trial["baseline_row_sha256"],
                    trial["restored_row_sha256"],
                    strict=True,
                )
            ):
                if baseline != restored:
                    first_difference = {
                        "field": f"row_permutation_trials[{trial_index}].row[{row}]",
                        "left_value": baseline,
                        "module": "m8_construction_positions",
                        "right_value": restored,
                    }
                    break
            if first_difference is not None:
                break
    evidence_sha256 = _sha256(_canonical_json(evidence))
    return {
        **bounded_assurance_claim(),
        "commit_counts": list(COMMIT_COUNTS),
        "context_tokens": evidence["canonical_state"]["logical_length"],
        "evidence_sha256": evidence_sha256,
        "first_difference": first_difference,
        "module_order": list(MODULE_ORDER),
        "passed": first_difference is None,
        "runtime_artifact_manifest_sha256": evidence["runtime_artifact_manifest_sha256"],
        "semantic_source_sha256": evidence["semantic_source_sha256"],
    }


def seal_certificate(value: object) -> dict[str, Any]:
    report = diagnose_evidence(value)
    if not report["passed"]:
        difference = report["first_difference"]
        raise RoundEquivalenceError(
            f"round equivalence failed at {difference['module']}: {difference['field']}"
        )
    certificate = {
        **report,
        "schema": CERTIFICATE_SCHEMA,
    }
    certificate["certificate_sha256"] = _document_digest(certificate, "certificate_sha256")
    return certificate


def verify_certificate(value: object) -> dict[str, Any]:
    keys = {
        *ASSURANCE_CLAIM_KEYS,
        "certificate_sha256",
        "commit_counts",
        "context_tokens",
        "evidence_sha256",
        "first_difference",
        "module_order",
        "passed",
        "runtime_artifact_manifest_sha256",
        "schema",
        "semantic_source_sha256",
    }
    certificate = _exact_dict(value, keys, "round certificate")
    if certificate["schema"] != CERTIFICATE_SCHEMA:
        raise RoundEquivalenceError("round certificate schema mismatch")
    validate_bounded_assurance_claim(certificate, "round certificate")
    if (
        certificate["passed"] is not True
        or certificate["first_difference"] is not None
        or certificate["commit_counts"] != list(COMMIT_COUNTS)
        or certificate["module_order"] != list(MODULE_ORDER)
    ):
        raise RoundEquivalenceError("round certificate does not represent complete equivalence")
    if certificate["context_tokens"] not in {60_298, 249_957}:
        raise RoundEquivalenceError("round certificate context is not qualified")
    for key in (
        "certificate_sha256",
        "evidence_sha256",
        "runtime_artifact_manifest_sha256",
        "semantic_source_sha256",
    ):
        _digest(certificate[key], f"round certificate.{key}")
    if certificate["certificate_sha256"] != _document_digest(certificate, "certificate_sha256"):
        raise RoundEquivalenceError("round certificate self-hash mismatch")
    return certificate


def _load_private_json(path: Path, label: str) -> object:
    path = path.absolute()
    status = path.lstat()
    if (
        not stat.S_ISREG(status.st_mode)
        or path.is_symlink()
        or status.st_uid != os.getuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise RoundEquivalenceError(f"{label} must be an owner-only regular file")
    return json.loads(path.read_bytes())


def load_forced_proposal(path: Path) -> dict[str, Any]:
    """Load an owner-only proposal document through the authenticated contract."""

    return normalize_forced_proposal(_load_private_json(path, "forced proposal"))


def _write_create_only(path: Path, value: object) -> None:
    payload = _canonical_json(value)
    descriptor = os.open(path.absolute(), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Certify one serial/M8 round-equivalence fixture.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--evidence", type=Path, required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--evidence", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal_proposal = subparsers.add_parser("seal-proposal")
    seal_proposal.add_argument("--request-id", required=True)
    seal_proposal.add_argument("--prompt-tokens", type=int, required=True)
    seal_proposal.add_argument("--proposal-token-ids", required=True)
    seal_proposal.add_argument("--runtime-artifact-manifest-sha256", required=True)
    seal_proposal.add_argument("--semantic-source-sha256", required=True)
    seal_proposal.add_argument("--snapshot-manifest-sha256", required=True)
    seal_proposal.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--certificate", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "diagnose":
            result = diagnose_evidence(_load_private_json(args.evidence, "round evidence"))
        elif args.command == "seal":
            result = seal_certificate(_load_private_json(args.evidence, "round evidence"))
            _write_create_only(args.output, result)
        elif args.command == "seal-proposal":
            result = seal_forced_proposal(
                {
                    "prompt_tokens": args.prompt_tokens,
                    "proposal_token_ids": [
                        int(value) for value in args.proposal_token_ids.split(",")
                    ],
                    "request_id": args.request_id,
                    "runtime_artifact_manifest_sha256": (args.runtime_artifact_manifest_sha256),
                    "schema": FORCED_PROPOSAL_SCHEMA,
                    "semantic_source_sha256": args.semantic_source_sha256,
                    "snapshot_manifest_sha256": args.snapshot_manifest_sha256,
                }
            )
            _write_create_only(args.output, result)
        else:
            result = verify_certificate(_load_private_json(args.certificate, "round certificate"))
    except (
        OSError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
        RoundEquivalenceError,
    ) as error:
        print(f"qwen-round-equivalence: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.command == "seal-proposal" or result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
# QWEN_ASSURANCE_ONLY_END: round-equivalence-certificate
