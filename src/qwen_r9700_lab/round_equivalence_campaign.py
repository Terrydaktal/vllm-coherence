# QWEN_ASSURANCE_ONLY_BEGIN: round-equivalence-campaign
"""Assemble independent serial/M8 runs into one accepted-prefix state proof.

The full round-equivalence certificate also requires component-boundary,
sibling-independence, and row-permutation evidence.  This module performs the
independent-clone part of that proof: one serial eight-transition execution,
eight separately restored fully accepted M8 executions capped at commit widths
one through eight, and eight independently restored acceptance/rejection
executions must all start at the same canonical state and finish at the
matching serial state.
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

from qwen_r9700_lab import assurance_layer_diagnosis as layer_diagnosis
from qwen_r9700_lab import coding_turbo_oracle as oracle
from qwen_r9700_lab import round_equivalence as equivalence

CAMPAIGN_SCHEMA = "urn:qwen-r9700:round-equivalence-state-campaign:v6"
COMPONENT_CAMPAIGN_SCHEMA = "urn:qwen-r9700:round-equivalence-component-campaign:v2"
CAMPAIGN_COUNTS = tuple(range(9))
STATE_CHECKPOINT_KEYS = oracle.CHECKPOINT_KEYS - {"accepted_draft_count"}
DRAFT_STATE_CHECKPOINT_KEYS = {"draft_kv_scales_sha256", "draft_kv_sha256"}
TARGET_STATE_CHECKPOINT_KEYS = STATE_CHECKPOINT_KEYS - DRAFT_STATE_CHECKPOINT_KEYS
BASE_TARGET_STATE_KEYS = equivalence.AUTHORITATIVE_STATE_KEYS - DRAFT_STATE_CHECKPOINT_KEYS
TARGET_FORWARD_STATE_KEYS = {
    "canonical_gdn_layer_sha256",
    "convolution_layer_sha256",
}
DEFERRED_PUBLICATION_STATE_KEYS = {
    "cache_mapping_sha256",
    "draft_kv_sha256",
    "target_kv_sha256",
}
INVARIANT_STATE_KEYS = (
    equivalence.AUTHORITATIVE_STATE_KEYS
    - TARGET_FORWARD_STATE_KEYS
    - DEFERRED_PUBLICATION_STATE_KEYS
)
CAMPAIGN_TRANSITION_KEYS = {
    "acceptance_draft_state_sha256",
    "acceptance_target_state_sha256",
    "capped_draft_state_sha256",
    "capped_reachable",
    "capped_target_state_sha256",
    "commit_count",
    "committed_token_ids",
    "serial_target_state_sha256",
}
CAMPAIGN_KEYS = {
    *equivalence.ASSURANCE_CLAIM_KEYS,
    "acceptance_proposal_token_ids",
    "acceptance_run_ids",
    "base_draft_state_sha256",
    "base_target_state_sha256",
    "capped_proposal_token_ids",
    "capped_run_ids",
    "campaign_sha256",
    "commit_counts",
    "passed",
    "prompt_tokens",
    "runtime_artifact_manifest_sha256",
    "schema",
    "semantic_source_sha256",
    "serial_run_id",
    "snapshot_manifest_sha256",
    "transitions",
    "zero_commit_event_sha256",
    "zero_commit_proposal_token_ids",
    "zero_commit_run_id",
}
IDENTITY_IGNORED_KEYS = {
    "artifact_kind",
    "dflash_enabled",
    "run_id",
}
COMPONENT_CAMPAIGN_KEYS = {
    *equivalence.ASSURANCE_CLAIM_KEYS,
    "campaign_sha256",
    "diagnoses",
    "m8_run_id",
    "passed",
    "prompt_tokens",
    "runtime_artifact_manifest_sha256",
    "schema",
    "semantic_source_sha256",
    "serial_run_id",
    "snapshot_manifest_sha256",
    "state_campaign_sha256",
}


class CampaignError(RuntimeError):
    """Independent campaign artifacts cannot establish accepted-prefix parity."""


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _canonical_line(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object) -> str:
    return _sha256(_canonical_line(value))


def _document_digest(value: dict[str, Any], digest_key: str) -> str:
    unsigned = dict(value)
    unsigned.pop(digest_key, None)
    return _sha256(_canonical_json(unsigned))


def _private_file(path: Path, label: str, *, maximum_bytes: int) -> bytes:
    path = path.absolute()
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise CampaignError(f"{label} is missing: {path}") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise CampaignError(f"{label} must be an owner-only regular file")
    if not 0 < metadata.st_size <= maximum_bytes:
        raise CampaignError(f"{label} size is invalid")
    return path.read_bytes()


def _json_file(path: Path, label: str, *, maximum_bytes: int) -> object:
    try:
        return json.loads(_private_file(path, label, maximum_bytes=maximum_bytes))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"{label} is not valid JSON: {error}") from error


def _oracle_source(run: Path, label: str) -> dict[str, Any]:
    value = _json_file(
        run / "capture" / "oracle-source.json",
        f"{label} oracle source",
        maximum_bytes=64 << 20,
    )
    try:
        return oracle.validate_oracle(value, sealed=False)
    except oracle.OracleSafetyError as error:
        raise CampaignError(f"{label} oracle source is invalid: {error}") from error


def _precommit_events(run: Path, label: str) -> list[dict[str, Any]]:
    payload = _private_file(
        run / "capture" / "precommit.jsonl",
        f"{label} precommit stream",
        maximum_bytes=128 << 20,
    )
    try:
        rows = [json.loads(line) for line in payload.decode().splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"{label} precommit stream is invalid: {error}") from error
    if len(rows) < 2 or not isinstance(rows[0], dict) or set(rows[0]) != {"header"}:
        raise CampaignError(f"{label} precommit stream lacks a header or event")
    header = rows[0]["header"]
    if (
        not isinstance(header, dict)
        or header.get("schema") != equivalence.PRECOMMIT_HEADER_SCHEMA
        or not isinstance(header.get("request_id"), str)
    ):
        raise CampaignError(f"{label} precommit header is invalid")
    events: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows[1:]):
            if not isinstance(row, dict) or set(row) != {"event"}:
                raise CampaignError(f"{label} precommit row {index + 1} is invalid")
            event = equivalence.normalize_precommit_event(row["event"])
            if event["round_index"] != index or event["request_id"] != header["request_id"]:
                raise CampaignError(f"{label} precommit ordering or request identity differs")
            events.append(event)
    except equivalence.RoundEquivalenceError as error:
        raise CampaignError(f"{label} precommit event is invalid: {error}") from error
    return events


def _proposal(run: Path, label: str) -> dict[str, Any]:
    try:
        return equivalence.load_forced_proposal(run / "round-equivalence-proposal.json")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        equivalence.RoundEquivalenceError,
    ) as error:
        raise CampaignError(f"{label} forced proposal is invalid: {error}") from error


def _same_identity(serial: dict[str, Any], candidate: dict[str, Any], label: str) -> None:
    for key in sorted(oracle.IDENTITY_KEYS - IDENTITY_IGNORED_KEYS):
        if serial[key] != candidate[key]:
            raise CampaignError(f"{label} identity differs at {key}")
    if serial["artifact_kind"] != "assurance" or candidate["artifact_kind"] != "assurance":
        raise CampaignError(f"{label} must use assurance artifacts")
    if serial["dflash_enabled"] is not False or candidate["dflash_enabled"] is not True:
        raise CampaignError(f"{label} does not compare serial target-only with speculative M8")


def _target_checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: checkpoint[key] for key in sorted(TARGET_STATE_CHECKPOINT_KEYS)}


def _draft_checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: checkpoint[key] for key in sorted(DRAFT_STATE_CHECKPOINT_KEYS)}


def _authoritative_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: checkpoint[key] for key in sorted(equivalence.AUTHORITATIVE_STATE_KEYS)}


def _expected_serial_post_target_state(
    before: dict[str, Any], published: dict[str, Any]
) -> dict[str, Any]:
    """Model the real M1 boundary before scheduler/cache publication.

    Serial target execution advances the canonical recurrent and convolution
    state inside the model forward.  The accepted-path publication that follows
    advances logical KV visibility, cache mapping, and the draft cache.  The
    precommit witness is deliberately between those operations, so it must not
    be compared wholesale with the final F checkpoint.

    The final checkpoint still authenticates every state field.  This helper
    additionally proves that no delayed-publication field changed early and no
    supposedly invariant field changed at either phase.
    """

    for key in sorted(INVARIANT_STATE_KEYS):
        if before[key] != published[key]:
            raise CampaignError(f"serial invariant state changed across F publication at {key}")
    expected = dict(before)
    for key in TARGET_FORWARD_STATE_KEYS:
        expected[key] = published[key]
    return expected


def _serial_contract(
    source: dict[str, Any], events: list[dict[str, Any]]
) -> tuple[list[int], list[int], list[str]]:
    if source["identity"]["dflash_enabled"] is not False:
        raise CampaignError("serial source has DFlash enabled")
    if len(source["checkpoints"]) != 8 or len(events) != 8:
        raise CampaignError("serial run must contain exactly eight committed M1 transitions")
    tokens = source["committed_token_ids"]
    if len(tokens) != 8:
        raise CampaignError("serial run must expose exactly eight committed tokens")
    next_tokens: list[int] = []
    logits_sha256: list[str] = []
    base_length = source["identity"]["context_tokens"]
    if events[0]["request_id"] != source["identity"]["run_id"]:
        raise CampaignError("serial precommit request identity differs from its oracle")
    for index, event in enumerate(events):
        if event["execution_mode"] != "serial-m1":
            raise CampaignError("serial precommit stream contains a non-M1 event")
        if event["sequence_length"] != base_length + index:
            raise CampaignError("serial precommit sequence lengths are not consecutive")
        if event["transition_row_token_ids"] != [tokens[index]]:
            raise CampaignError("serial precommit input token differs from committed stream")
        if event["sampler_approved_commit_count"] != 1:
            raise CampaignError("serial precommit event did not approve exactly one token")
        if event["external_commit_count"] != 1:
            raise CampaignError("serial precommit event did not publish exactly one token")
        prior_state = event["authoritative_state"]
        if index > 0:
            if events[index - 1]["target_argmax_token_ids"] != [tokens[index]]:
                raise CampaignError("serial target sample does not chain into the next M1 input")
            preceding_published = _authoritative_from_checkpoint(source["checkpoints"][index - 1])
            if prior_state != preceding_published:
                raise CampaignError(
                    "serial precommit state does not chain from the preceding F transition"
                )
        published = _authoritative_from_checkpoint(source["checkpoints"][index])
        expected_post_target = _expected_serial_post_target_state(prior_state, published)
        if event["post_target_authoritative_state"] != expected_post_target:
            differing = next(
                key
                for key in sorted(equivalence.AUTHORITATIVE_STATE_KEYS)
                if event["post_target_authoritative_state"][key] != expected_post_target[key]
            )
            raise CampaignError(
                "serial post-target phase differs from the authenticated "
                f"target/publication boundary at {differing}"
            )
        next_tokens.append(event["target_argmax_token_ids"][0])
        logits_sha256.append(event["target_logits_sha256"][0])
    if source["committed_target_top1_token_ids"] != tokens:
        raise CampaignError("serial committed tokens differ from target top-1")
    return tokens, next_tokens, logits_sha256


def _proposal_document(
    identity: dict[str, Any], request_id: str, proposal_token_ids: list[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return equivalence.seal_forced_proposal(
        {
            "prompt_tokens": identity["context_tokens"],
            "proposal_token_ids": proposal_token_ids,
            "request_id": request_id,
            "runtime_artifact_manifest_sha256": identity["runtime_artifact_manifest_sha256"],
            "schema": equivalence.FORCED_PROPOSAL_SCHEMA,
            "semantic_source_sha256": identity["semantic_source_sha256"],
            "snapshot_manifest_sha256": identity["snapshot_manifest_sha256"],
        }
    )


def derive_forced_proposal(serial_run: Path, request_id: str) -> dict[str, Any]:
    """Derive P=T1..T7 from one independently authenticated serial F^8 run."""

    if not request_id or any(character.isspace() for character in request_id):
        raise CampaignError("proposal request ID is invalid")
    source = _oracle_source(serial_run, "serial")
    events = _precommit_events(serial_run, "serial")
    tokens, _next_tokens, _logits_sha256 = _serial_contract(source, events)
    return _proposal_document(source["identity"], request_id, tokens[1:8])


def derive_acceptance_proposal(
    serial_run: Path, request_id: str, commit_count: int
) -> dict[str, Any]:
    """Derive a proposal whose greedy verifier approves exactly ``commit_count`` rows.

    A committed width of eight uses the complete target-derived proposal.  For
    widths one through seven, proposal rows before the rejection are the serial
    target tokens and the next row is replaced by a deterministic in-vocabulary
    counterexample.  Later rows are irrelevant siblings and remain deterministic.
    """

    if not request_id or any(character.isspace() for character in request_id):
        raise CampaignError("proposal request ID is invalid")
    if isinstance(commit_count, bool) or not 1 <= commit_count <= 8:
        raise CampaignError("acceptance proposal commit count must be within 1..8")
    source = _oracle_source(serial_run, "serial")
    events = _precommit_events(serial_run, "serial")
    tokens, _next_tokens, _logits_sha256 = _serial_contract(source, events)
    proposal = list(tokens[1:8])
    if commit_count < 8:
        rejection_index = commit_count - 1
        expected = tokens[commit_count]
        proposal[rejection_index] = (expected + 1) % 253_952
        if proposal[rejection_index] == expected:  # pragma: no cover - defensive
            raise CampaignError("failed to construct a rejecting proposal token")
    return _proposal_document(source["identity"], request_id, proposal)


def _proposal_identity(
    candidate_identity: dict[str, Any], proposal: dict[str, Any], label: str
) -> None:
    if proposal["request_id"] != candidate_identity["run_id"]:
        raise CampaignError(f"{label} proposal request identity differs")
    for key in (
        "prompt_tokens",
        "runtime_artifact_manifest_sha256",
        "semantic_source_sha256",
        "snapshot_manifest_sha256",
    ):
        identity_key = "context_tokens" if key == "prompt_tokens" else key
        if proposal[key] != candidate_identity[identity_key]:
            raise CampaignError(f"{label} proposal differs from run identity at {key}")


def _proposal_identity_from_serial(
    serial_identity: dict[str, Any], proposal: dict[str, Any], label: str
) -> None:
    """Bind a no-publication run to the serial source without inventing an oracle."""

    for key in (
        "prompt_tokens",
        "runtime_artifact_manifest_sha256",
        "semantic_source_sha256",
        "snapshot_manifest_sha256",
    ):
        identity_key = "context_tokens" if key == "prompt_tokens" else key
        if proposal[key] != serial_identity[identity_key]:
            raise CampaignError(f"{label} proposal differs from serial identity at {key}")


def _sequence_difference(expected: list[int], observed: list[int]) -> str:
    shared = min(len(expected), len(observed))
    for index in range(shared):
        if expected[index] != observed[index]:
            return f"index={index} expected={expected[index]} observed={observed[index]}"
    return f"length expected={len(expected)} observed={len(observed)}"


def _candidate_checkpoint(
    *,
    candidate: dict[str, Any],
    serial_source: dict[str, Any],
    serial_tokens: list[int],
    commit_count: int,
    label: str,
) -> dict[str, Any]:
    expected_tokens = serial_tokens[:commit_count]
    if candidate["committed_token_ids"] != expected_tokens:
        detail = _sequence_difference(expected_tokens, candidate["committed_token_ids"])
        raise CampaignError(
            f"{label} committed token prefix differs from serial F^c: {detail}; "
            f"run_id={candidate['identity']['run_id']}"
        )
    if candidate["committed_target_top1_token_ids"] != expected_tokens:
        detail = _sequence_difference(expected_tokens, candidate["committed_target_top1_token_ids"])
        raise CampaignError(
            f"{label} target top-1 prefix differs from serial F^c: {detail}; "
            f"run_id={candidate['identity']['run_id']}"
        )
    if len(candidate["checkpoints"]) != 1:
        raise CampaignError(f"{label} must publish exactly one atomic M8 commit")
    serial_state = _target_checkpoint_state(serial_source["checkpoints"][commit_count - 1])
    candidate_checkpoint = candidate["checkpoints"][0]
    candidate_state = _target_checkpoint_state(candidate_checkpoint)
    if serial_state != candidate_state:
        differing = next(
            key for key in sorted(serial_state) if serial_state[key] != candidate_state[key]
        )
        raise CampaignError(f"{label} committed state differs from F^c at {differing}")
    return candidate_state, _draft_checkpoint_state(candidate_checkpoint)


def _capped_candidate(
    run: Path,
    *,
    serial_source: dict[str, Any],
    serial_identity: dict[str, Any],
    serial_tokens: list[int],
    serial_next: list[int],
    serial_logits: list[str],
    base_target_state: dict[str, Any],
    base_draft_state: dict[str, Any],
    commit_count: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Validate one scheduler-reachable full-accept external-cap transition."""

    capped_label = f"capped c={commit_count}"
    capped = _oracle_source(run, capped_label)
    capped_events = _precommit_events(run, capped_label)
    capped_proposal = _proposal(run, capped_label)
    _same_identity(serial_identity, capped["identity"], capped_label)
    if len(capped_events) != 1 or capped_events[0]["execution_mode"] != "speculative-m8":
        raise CampaignError(f"{capped_label} must contain exactly one M8 precommit event")
    capped_event = capped_events[0]
    if capped_event["request_id"] != capped["identity"]["run_id"]:
        raise CampaignError(f"{capped_label} precommit request identity differs from its oracle")
    if capped_event["sampler_approved_commit_count"] != 8:
        raise CampaignError(f"{capped_label} did not verify a full eight-token target prefix")
    if capped_event["external_commit_count"] != commit_count:
        raise CampaignError(
            f"{capped_label} external cap is wrong: expected={commit_count} "
            f"observed={capped_event['external_commit_count']}"
        )
    if capped_event["sequence_length"] != serial_identity["context_tokens"]:
        raise CampaignError(f"{capped_label} did not start at canonical state S")
    capped_precommit_target = {
        key: capped_event["authoritative_state"][key] for key in sorted(BASE_TARGET_STATE_KEYS)
    }
    if capped_precommit_target != base_target_state:
        raise CampaignError(f"{capped_label} target canonical state changed before atomic commit")
    capped_precommit_draft = {
        key: capped_event["authoritative_state"][key] for key in sorted(DRAFT_STATE_CHECKPOINT_KEYS)
    }
    if capped_precommit_draft != base_draft_state:
        raise CampaignError(f"{capped_label} did not start from canonical draft state S")
    if capped_event["transition_row_token_ids"] != serial_tokens:
        raise CampaignError(f"{capped_label} M8 row inputs differ from serial ancestor tokens")
    if capped_event["target_argmax_token_ids"] != serial_next:
        raise CampaignError(f"{capped_label} M8 target rows differ from standalone serial F")
    if capped_event["target_logits_sha256"] != serial_logits:
        raise CampaignError(f"{capped_label} LM-head logits differ from standalone serial F")
    if capped_proposal["proposal_token_ids"] != serial_tokens[1:8]:
        raise CampaignError(f"{capped_label} proposal differs from serial target sequence")
    _proposal_identity(capped["identity"], capped_proposal, capped_label)
    capped_state, capped_draft_state = _candidate_checkpoint(
        candidate=capped,
        serial_source=serial_source,
        serial_tokens=serial_tokens,
        commit_count=commit_count,
        label=capped_label,
    )
    return capped_state, capped_draft_state, capped["identity"]["run_id"]


def _zero_commit_candidate(
    run: Path,
    *,
    serial_identity: dict[str, Any],
    serial_tokens: list[int],
    serial_next: list[int],
    serial_logits: list[str],
    base_target_state: dict[str, Any],
) -> tuple[str, list[int], str, dict[str, Any]]:
    """Validate a real M8 execution stopped before any canonical publication."""

    label = "zero-commit c=0"
    events = _precommit_events(run, label)
    proposal = _proposal(run, label)
    if len(events) != 1 or events[0]["execution_mode"] != "speculative-m8":
        raise CampaignError(f"{label} must contain exactly one M8 precommit event")
    event = events[0]
    if event["request_id"] != proposal["request_id"]:
        raise CampaignError(f"{label} precommit request identity differs from its proposal")
    _proposal_identity_from_serial(serial_identity, proposal, label)
    if event["sequence_length"] != serial_identity["context_tokens"]:
        raise CampaignError(f"{label} did not start at canonical state S")
    if event["external_commit_count"] != 0:
        raise CampaignError(
            f"{label} published transitions: observed={event['external_commit_count']}"
        )
    if not 1 <= event["sampler_approved_commit_count"] <= 8:
        raise CampaignError(f"{label} has no valid completed target verification")
    zero_target_state = {
        key: event["authoritative_state"][key] for key in sorted(BASE_TARGET_STATE_KEYS)
    }
    if zero_target_state != base_target_state:
        differing = next(
            key
            for key in sorted(BASE_TARGET_STATE_KEYS)
            if zero_target_state[key] != base_target_state[key]
        )
        raise CampaignError(f"{label} did not start from S at {differing}")
    if event["post_target_authoritative_state"] != event["authoritative_state"]:
        differing = next(
            key
            for key in sorted(equivalence.AUTHORITATIVE_STATE_KEYS)
            if event["post_target_authoritative_state"][key] != event["authoritative_state"][key]
        )
        raise CampaignError(f"{label} provisional target mutated canonical state at {differing}")
    proposal_tokens = proposal["proposal_token_ids"]
    if event["transition_row_token_ids"] != [serial_tokens[0], *proposal_tokens]:
        raise CampaignError(f"{label} M8 rows differ from its authenticated proposal")
    # Only row zero belongs to F^0's observable boundary.  Later rows may be
    # arbitrary sibling proposals, but row zero must still equal standalone F.
    if event["target_argmax_token_ids"][0] != serial_next[0]:
        raise CampaignError(f"{label} row-zero target token differs from standalone serial F")
    if event["target_logits_sha256"][0] != serial_logits[0]:
        raise CampaignError(f"{label} row-zero logits differ from standalone serial F")
    zero_draft_state = {
        key: event["authoritative_state"][key] for key in sorted(DRAFT_STATE_CHECKPOINT_KEYS)
    }
    return event["request_id"], proposal_tokens, event["event_sha256"], zero_draft_state


def assemble_state_campaign(
    serial_run: Path,
    zero_commit_run: Path,
    capped_runs: dict[int, Path],
    acceptance_runs: dict[int, Path],
) -> dict[str, Any]:
    """Prove c=0..8 target parity and proposal-independent DFlash state.

    The target-only serial process defines ``F`` but does not own or advance a
    drafter.  Its draft-KV bytes therefore are not a valid post-transition
    oracle.  Complete draft KV is instead compared between two independently
    restored DFlash executions that commit the same target prefix through
    different proposal/acceptance mechanisms.
    """

    acceptance_counts = set(range(1, 9))
    capped_counts = set(range(2, 9))
    if set(capped_runs) != capped_counts:
        raise CampaignError("capped campaign must contain scheduler-reachable widths 2 through 8")
    if set(acceptance_runs) != acceptance_counts:
        raise CampaignError("acceptance campaign must contain commit widths 1 through 8")
    serial_source = _oracle_source(serial_run, "serial")
    serial_events = _precommit_events(serial_run, "serial")
    serial_tokens, serial_next, serial_logits = _serial_contract(serial_source, serial_events)
    serial_identity = serial_source["identity"]
    serial_base_state = serial_events[0]["authoritative_state"]
    base_target_state = {key: serial_base_state[key] for key in sorted(BASE_TARGET_STATE_KEYS)}
    zero_run_id, zero_proposal, zero_event_sha256, base_draft_state = _zero_commit_candidate(
        zero_commit_run,
        serial_identity=serial_identity,
        serial_tokens=serial_tokens,
        serial_next=serial_next,
        serial_logits=serial_logits,
        base_target_state=base_target_state,
    )
    transitions: list[dict[str, Any]] = [
        {
            "acceptance_draft_state_sha256": _digest(base_draft_state),
            "acceptance_target_state_sha256": _digest(base_target_state),
            "capped_draft_state_sha256": None,
            "capped_reachable": False,
            "capped_target_state_sha256": None,
            "commit_count": 0,
            "committed_token_ids": [],
            "serial_target_state_sha256": _digest(base_target_state),
        }
    ]
    capped_run_ids: dict[str, str] = {}
    acceptance_run_ids: dict[str, str] = {}
    acceptance_proposals: dict[str, list[int]] = {}
    for commit_count in range(1, 9):
        capped_state: dict[str, Any] | None = None
        capped_draft_state: dict[str, Any] | None = None
        capped_run_id: str | None = None
        if commit_count >= 2:
            capped_state, capped_draft_state, capped_run_id = _capped_candidate(
                capped_runs[commit_count],
                serial_source=serial_source,
                serial_identity=serial_identity,
                serial_tokens=serial_tokens,
                serial_next=serial_next,
                serial_logits=serial_logits,
                base_target_state=base_target_state,
                base_draft_state=base_draft_state,
                commit_count=commit_count,
            )

        acceptance_label = f"acceptance c={commit_count}"
        acceptance = _oracle_source(acceptance_runs[commit_count], acceptance_label)
        acceptance_events = _precommit_events(acceptance_runs[commit_count], acceptance_label)
        acceptance_proposal = _proposal(acceptance_runs[commit_count], acceptance_label)
        _same_identity(serial_identity, acceptance["identity"], acceptance_label)
        if (
            len(acceptance_events) != 1
            or acceptance_events[0]["execution_mode"] != "speculative-m8"
        ):
            raise CampaignError(f"{acceptance_label} must contain exactly one M8 precommit event")
        acceptance_event = acceptance_events[0]
        if acceptance_event["request_id"] != acceptance["identity"]["run_id"]:
            raise CampaignError(
                f"{acceptance_label} precommit request identity differs from its oracle"
            )
        if acceptance_event["sampler_approved_commit_count"] != commit_count:
            raise CampaignError(
                f"{acceptance_label} sampler did not approve exactly {commit_count} rows"
            )
        if acceptance_event["external_commit_count"] != commit_count:
            raise CampaignError(
                f"{acceptance_label} external commit differs: expected={commit_count} "
                f"observed={acceptance_event['external_commit_count']}"
            )
        if acceptance_event["sequence_length"] != serial_identity["context_tokens"]:
            raise CampaignError(f"{acceptance_label} did not start at canonical state S")
        acceptance_precommit_target = {
            key: acceptance_event["authoritative_state"][key]
            for key in sorted(BASE_TARGET_STATE_KEYS)
        }
        if acceptance_precommit_target != base_target_state:
            raise CampaignError(
                f"{acceptance_label} target canonical state changed before atomic commit"
            )
        acceptance_precommit_draft = {
            key: acceptance_event["authoritative_state"][key]
            for key in sorted(DRAFT_STATE_CHECKPOINT_KEYS)
        }
        if acceptance_precommit_draft != base_draft_state:
            raise CampaignError(f"{acceptance_label} did not start from canonical draft state S")
        proposal_tokens = acceptance_proposal["proposal_token_ids"]
        if acceptance_event["transition_row_token_ids"] != [serial_tokens[0], *proposal_tokens]:
            raise CampaignError(
                f"{acceptance_label} M8 row inputs differ from its authenticated proposal"
            )
        if acceptance_event["target_argmax_token_ids"][:commit_count] != serial_next[:commit_count]:
            raise CampaignError(
                f"{acceptance_label} accepted-path target rows differ from standalone serial F"
            )
        if acceptance_event["target_logits_sha256"][:commit_count] != serial_logits[:commit_count]:
            raise CampaignError(
                f"{acceptance_label} accepted-path LM-head logits differ from standalone serial F"
            )
        if proposal_tokens[: commit_count - 1] != serial_tokens[1:commit_count]:
            raise CampaignError(
                f"{acceptance_label} proposal does not preserve the accepted serial prefix"
            )
        if commit_count < 8 and proposal_tokens[commit_count - 1] == serial_tokens[commit_count]:
            raise CampaignError(
                f"{acceptance_label} proposal does not force the required rejection"
            )
        if commit_count == 8 and proposal_tokens != serial_tokens[1:8]:
            raise CampaignError(f"{acceptance_label} full-accept proposal differs from serial F")
        _proposal_identity(acceptance["identity"], acceptance_proposal, acceptance_label)
        acceptance_state, acceptance_draft_state = _candidate_checkpoint(
            candidate=acceptance,
            serial_source=serial_source,
            serial_tokens=serial_tokens,
            commit_count=commit_count,
            label=acceptance_label,
        )
        if capped_draft_state is not None and acceptance_draft_state != capped_draft_state:
            differing = next(
                key
                for key in sorted(DRAFT_STATE_CHECKPOINT_KEYS)
                if acceptance_draft_state[key] != capped_draft_state[key]
            )
            raise CampaignError(
                f"acceptance c={commit_count} draft state differs from capped DFlash "
                f"transition at {differing}"
            )
        serial_state = _target_checkpoint_state(serial_source["checkpoints"][commit_count - 1])
        transitions.append(
            {
                "acceptance_draft_state_sha256": _digest(acceptance_draft_state),
                "acceptance_target_state_sha256": _digest(acceptance_state),
                "capped_draft_state_sha256": (
                    _digest(capped_draft_state) if capped_draft_state is not None else None
                ),
                "capped_reachable": commit_count >= 2,
                "capped_target_state_sha256": (
                    _digest(capped_state) if capped_state is not None else None
                ),
                "commit_count": commit_count,
                "committed_token_ids": serial_tokens[:commit_count],
                "serial_target_state_sha256": _digest(serial_state),
            }
        )
        if capped_run_id is not None:
            capped_run_ids[str(commit_count)] = capped_run_id
        acceptance_run_ids[str(commit_count)] = acceptance["identity"]["run_id"]
        acceptance_proposals[str(commit_count)] = proposal_tokens

    all_run_ids = {
        serial_identity["run_id"],
        *capped_run_ids.values(),
        *acceptance_run_ids.values(),
        zero_run_id,
    }
    if len(all_run_ids) != 17:
        raise CampaignError("campaign runs do not have 17 independent request identities")

    result = {
        **equivalence.bounded_assurance_claim(),
        "acceptance_proposal_token_ids": acceptance_proposals,
        "acceptance_run_ids": acceptance_run_ids,
        "base_draft_state_sha256": _digest(base_draft_state),
        "base_target_state_sha256": _digest(base_target_state),
        "capped_proposal_token_ids": serial_tokens[1:8],
        "capped_run_ids": capped_run_ids,
        "commit_counts": list(CAMPAIGN_COUNTS),
        "passed": True,
        "prompt_tokens": serial_identity["context_tokens"],
        "runtime_artifact_manifest_sha256": serial_identity["runtime_artifact_manifest_sha256"],
        "schema": CAMPAIGN_SCHEMA,
        "semantic_source_sha256": serial_identity["semantic_source_sha256"],
        "serial_run_id": serial_identity["run_id"],
        "snapshot_manifest_sha256": serial_identity["snapshot_manifest_sha256"],
        "transitions": transitions,
        "zero_commit_event_sha256": zero_event_sha256,
        "zero_commit_proposal_token_ids": zero_proposal,
        "zero_commit_run_id": zero_run_id,
    }
    result["campaign_sha256"] = _document_digest(result, "campaign_sha256")
    return verify_state_campaign(result)


def verify_state_campaign(value: object) -> dict[str, Any]:
    """Validate a complete self-authenticating c=0..8 state campaign."""

    if not isinstance(value, dict) or set(value) != CAMPAIGN_KEYS:
        raise CampaignError("state campaign keys are invalid")
    result = dict(value)
    if result["schema"] != CAMPAIGN_SCHEMA or result["passed"] is not True:
        raise CampaignError("state campaign is not a passing v6 artifact")
    try:
        equivalence.validate_bounded_assurance_claim(result, "state campaign")
    except equivalence.RoundEquivalenceError as error:
        raise CampaignError(str(error)) from error
    if result["commit_counts"] != list(CAMPAIGN_COUNTS):
        raise CampaignError("state campaign commit counts are incomplete")
    transitions = result["transitions"]
    if not isinstance(transitions, list) or len(transitions) != 9:
        raise CampaignError("state campaign must contain nine transitions")
    for count, transition in enumerate(transitions):
        if not isinstance(transition, dict) or set(transition) != CAMPAIGN_TRANSITION_KEYS:
            raise CampaignError(f"state campaign transition {count} keys are invalid")
        if transition["commit_count"] != count:
            raise CampaignError("state campaign transitions are not ordered c=0..8")
        tokens = transition["committed_token_ids"]
        if (
            not isinstance(tokens, list)
            or len(tokens) != count
            or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens)
        ):
            raise CampaignError(f"state campaign transition {count} tokens are invalid")
        for key in (
            "acceptance_draft_state_sha256",
            "acceptance_target_state_sha256",
            "serial_target_state_sha256",
        ):
            value_sha = transition[key]
            if not isinstance(value_sha, str) or len(value_sha) != 64:
                raise CampaignError(f"state campaign transition {count} hash is invalid")
        if transition["serial_target_state_sha256"] != transition["acceptance_target_state_sha256"]:
            raise CampaignError(f"state campaign transition {count} is divergent")
        reachable = transition["capped_reachable"]
        if reachable is not (count >= 2):
            raise CampaignError(f"state campaign transition {count} cap reachability is invalid")
        if reachable:
            for key in ("capped_draft_state_sha256", "capped_target_state_sha256"):
                value_sha = transition[key]
                if not isinstance(value_sha, str) or len(value_sha) != 64:
                    raise CampaignError(f"state campaign transition {count} cap hash is invalid")
            if transition["capped_target_state_sha256"] != transition["serial_target_state_sha256"]:
                raise CampaignError(f"state campaign transition {count} cap state is divergent")
            if (
                transition["capped_draft_state_sha256"]
                != transition["acceptance_draft_state_sha256"]
            ):
                raise CampaignError(
                    f"state campaign transition {count} has proposal-dependent draft state"
                )
        elif (
            transition["capped_draft_state_sha256"] is not None
            or transition["capped_target_state_sha256"] is not None
        ):
            raise CampaignError(f"state campaign transition {count} has unreachable cap evidence")
    if any(
        not isinstance(result[key], str) or len(result[key]) != 64
        for key in (
            "base_draft_state_sha256",
            "base_target_state_sha256",
            "campaign_sha256",
            "runtime_artifact_manifest_sha256",
            "semantic_source_sha256",
            "snapshot_manifest_sha256",
            "zero_commit_event_sha256",
        )
    ):
        raise CampaignError("state campaign contains an invalid SHA-256")
    expected_acceptance_keys = {str(count) for count in range(1, 9)}
    expected_capped_keys = {str(count) for count in range(2, 9)}
    capped_run_ids = result["capped_run_ids"]
    acceptance_run_ids = result["acceptance_run_ids"]
    if (
        not isinstance(capped_run_ids, dict)
        or set(capped_run_ids) != expected_capped_keys
        or not isinstance(acceptance_run_ids, dict)
        or set(acceptance_run_ids) != expected_acceptance_keys
        or any(
            not isinstance(run_id, str) or not run_id
            for run_id in (*capped_run_ids.values(), *acceptance_run_ids.values())
        )
    ):
        raise CampaignError("state campaign run identities are invalid or reused")
    if (
        not isinstance(result["serial_run_id"], str)
        or not result["serial_run_id"]
        or not isinstance(result["zero_commit_run_id"], str)
        or not result["zero_commit_run_id"]
        or len(
            {
                result["serial_run_id"],
                result["zero_commit_run_id"],
                *capped_run_ids.values(),
                *acceptance_run_ids.values(),
            }
        )
        != 17
    ):
        raise CampaignError("state campaign run identities are invalid or reused")
    zero_proposal = result["zero_commit_proposal_token_ids"]
    if (
        not isinstance(zero_proposal, list)
        or len(zero_proposal) != 7
        or any(isinstance(token, bool) or not isinstance(token, int) for token in zero_proposal)
    ):
        raise CampaignError("state campaign zero-commit proposal is invalid")
    capped_proposal = result["capped_proposal_token_ids"]
    if (
        not isinstance(capped_proposal, list)
        or len(capped_proposal) != 7
        or any(isinstance(token, bool) or not isinstance(token, int) for token in capped_proposal)
    ):
        raise CampaignError("state campaign capped proposal is invalid")
    acceptance_proposals = result["acceptance_proposal_token_ids"]
    if (
        not isinstance(acceptance_proposals, dict)
        or set(acceptance_proposals) != expected_acceptance_keys
    ):
        raise CampaignError("state campaign acceptance proposals are incomplete")
    for count in range(1, 9):
        proposal = acceptance_proposals[str(count)]
        if (
            not isinstance(proposal, list)
            or len(proposal) != 7
            or any(isinstance(token, bool) or not isinstance(token, int) for token in proposal)
        ):
            raise CampaignError(f"state campaign acceptance proposal {count} is invalid")
        if proposal[: count - 1] != capped_proposal[: count - 1]:
            raise CampaignError(f"state campaign acceptance proposal {count} breaks its prefix")
        if count < 8 and proposal[count - 1] == capped_proposal[count - 1]:
            raise CampaignError(f"state campaign acceptance proposal {count} does not reject")
        if count == 8 and proposal != capped_proposal:
            raise CampaignError("state campaign full-accept proposal differs from capped proposal")
    if result["campaign_sha256"] != _document_digest(result, "campaign_sha256"):
        raise CampaignError("state campaign self-hash mismatch")
    return result


def assemble_component_campaign(
    *,
    serial_run: Path,
    m8_run: Path,
    state_campaign: object,
) -> dict[str, Any]:
    """Bind eight complete row diagnoses to the same c=8 state execution."""

    state = verify_state_campaign(state_campaign)
    serial_source = _oracle_source(serial_run, "component serial")
    m8_source = _oracle_source(m8_run, "component M8")
    serial_identity = serial_source["identity"]
    m8_identity = m8_source["identity"]
    _same_identity(serial_identity, m8_identity, "component campaign")
    if serial_identity["run_id"] == m8_identity["run_id"]:
        raise CampaignError("component serial and M8 runs reuse one request identity")
    for key in (
        "runtime_artifact_manifest_sha256",
        "semantic_source_sha256",
        "snapshot_manifest_sha256",
    ):
        if serial_identity[key] != state[key]:
            raise CampaignError(f"component campaign identity differs from state campaign at {key}")
    if serial_identity["context_tokens"] != state["prompt_tokens"]:
        raise CampaignError("component campaign prompt length differs from state campaign")
    transition_c8 = state["transitions"][8]
    if len(serial_source["checkpoints"]) != 8:
        raise CampaignError("component serial run must contain eight committed transitions")
    serial_final = _target_checkpoint_state(serial_source["checkpoints"][7])
    if _digest(serial_final) != transition_c8["serial_target_state_sha256"]:
        raise CampaignError("component serial final state differs from the state campaign")
    if len(m8_source["checkpoints"]) != 1:
        raise CampaignError("component M8 run must contain one atomic commit")
    m8_checkpoint = m8_source["checkpoints"][0]
    if (
        _digest(_target_checkpoint_state(m8_checkpoint))
        != transition_c8["capped_target_state_sha256"]
    ):
        raise CampaignError("component M8 final target state differs from the state campaign")
    if (
        _digest(_draft_checkpoint_state(m8_checkpoint))
        != transition_c8["capped_draft_state_sha256"]
    ):
        raise CampaignError("component M8 final draft state differs from the state campaign")

    serial_layers = serial_run / "capture" / "layers.jsonl"
    m8_layers = m8_run / "capture" / "layers.jsonl"
    diagnoses: list[dict[str, Any]] = []
    base_position = serial_identity["context_tokens"]
    for logical_row in range(8):
        position = base_position + logical_row
        try:
            diagnosis = layer_diagnosis.diagnose(
                serial_layers,
                m8_layers,
                target_position=position,
                require_gdn=True,
                allow_external_recurrence_state=True,
            )
        except layer_diagnosis.LayerDiagnosisError as error:
            raise CampaignError(
                f"component row {logical_row} evidence is invalid: {error}"
            ) from error
        if not diagnosis["passed"] or diagnosis["first_difference"] is not None:
            difference = diagnosis["first_difference"]
            module = difference.get("module") if isinstance(difference, dict) else "unknown"
            raise CampaignError(f"component row {logical_row} diverges first at {module}")
        if diagnosis["target_position"] != position:
            raise CampaignError(f"component row {logical_row} position differs")
        if diagnosis["selected_layer_indices"] != list(range(64)):
            raise CampaignError(f"component row {logical_row} does not cover all 64 layers")
        if diagnosis["serial_pass_count"] != 1 or diagnosis["candidate_pass_count"] != 1:
            raise CampaignError(
                f"component row {logical_row} must contain one serial and one M8 decoder pass"
            )
        if (
            diagnosis["serial_detailed_layer_count"] != 64
            or diagnosis["candidate_detailed_layer_count"] != 64
        ):
            raise CampaignError(
                f"component row {logical_row} lacks all 64 detailed decoder boundaries"
            )
        if (
            diagnosis["serial_quest_pass_count"] != 1
            or diagnosis["candidate_quest_pass_count"] != 1
            or diagnosis["candidate_quest_mode"] != "m8"
            or diagnosis["serial_scored_quest_layer_count"] != 16
            or diagnosis["candidate_scored_quest_layer_count"] != 16
        ):
            raise CampaignError(
                f"component row {logical_row} must contain one serial and one M8 Quest pass"
            )
        if (
            diagnosis["serial_gdn_pass_count"] != 1
            or diagnosis["candidate_gdn_pass_count"] != 1
            or diagnosis["candidate_gdn_mode"] != "m8"
        ):
            raise CampaignError(
                f"component row {logical_row} must contain one serial and one M8 GDN pass"
            )
        if diagnosis["gdn_recurrence_evidence"] not in {
            "captured_suboperations",
            "external_state_campaign",
        }:
            raise CampaignError(f"component row {logical_row} lacks GDN recurrence evidence")
        diagnoses.append({"logical_row": logical_row, **diagnosis})

    result = {
        **equivalence.bounded_assurance_claim(),
        "diagnoses": diagnoses,
        "m8_run_id": m8_identity["run_id"],
        "passed": True,
        "prompt_tokens": serial_identity["context_tokens"],
        "runtime_artifact_manifest_sha256": serial_identity["runtime_artifact_manifest_sha256"],
        "schema": COMPONENT_CAMPAIGN_SCHEMA,
        "semantic_source_sha256": serial_identity["semantic_source_sha256"],
        "serial_run_id": serial_identity["run_id"],
        "snapshot_manifest_sha256": serial_identity["snapshot_manifest_sha256"],
        "state_campaign_sha256": state["campaign_sha256"],
    }
    result["campaign_sha256"] = _document_digest(result, "campaign_sha256")
    return verify_component_campaign(result)


def verify_component_campaign(value: object) -> dict[str, Any]:
    """Validate one self-authenticating eight-row component campaign."""

    if not isinstance(value, dict) or set(value) != COMPONENT_CAMPAIGN_KEYS:
        raise CampaignError("component campaign keys are invalid")
    result = dict(value)
    if result["schema"] != COMPONENT_CAMPAIGN_SCHEMA or result["passed"] is not True:
        raise CampaignError("component campaign is not a passing v2 artifact")
    try:
        equivalence.validate_bounded_assurance_claim(result, "component campaign")
    except equivalence.RoundEquivalenceError as error:
        raise CampaignError(str(error)) from error
    for key in (
        "campaign_sha256",
        "runtime_artifact_manifest_sha256",
        "semantic_source_sha256",
        "snapshot_manifest_sha256",
        "state_campaign_sha256",
    ):
        digest = result[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CampaignError(f"component campaign {key} is invalid")
    if (
        isinstance(result["prompt_tokens"], bool)
        or not isinstance(result["prompt_tokens"], int)
        or result["prompt_tokens"] <= 0
    ):
        raise CampaignError("component campaign prompt length is invalid")
    for key in ("serial_run_id", "m8_run_id"):
        if (
            not isinstance(result[key], str)
            or not result[key]
            or any(character.isspace() for character in result[key])
        ):
            raise CampaignError(f"component campaign {key} is invalid")
    if result["serial_run_id"] == result["m8_run_id"]:
        raise CampaignError("component campaign reused one request identity")
    diagnoses = result["diagnoses"]
    if not isinstance(diagnoses, list) or len(diagnoses) != 8:
        raise CampaignError("component campaign must contain eight row diagnoses")
    for row, diagnosis in enumerate(diagnoses):
        if not isinstance(diagnosis, dict):
            raise CampaignError(f"component diagnosis {row} is invalid")
        if diagnosis.get("logical_row") != row:
            raise CampaignError("component campaign diagnoses are not ordered by logical row")
        if diagnosis.get("target_position") != result["prompt_tokens"] + row:
            raise CampaignError(f"component diagnosis {row} position is invalid")
        if diagnosis.get("passed") is not True or diagnosis.get("first_difference") is not None:
            raise CampaignError(f"component diagnosis {row} is divergent")
        if diagnosis.get("selected_layer_indices") != list(range(64)):
            raise CampaignError(f"component diagnosis {row} layer coverage is incomplete")
        if any(
            diagnosis.get(key) != 1
            for key in (
                "serial_pass_count",
                "candidate_pass_count",
                "serial_quest_pass_count",
                "candidate_quest_pass_count",
                "serial_gdn_pass_count",
                "candidate_gdn_pass_count",
            )
        ):
            raise CampaignError(f"component diagnosis {row} pass coverage is incomplete")
        if (
            diagnosis.get("serial_detailed_layer_count") != 64
            or diagnosis.get("candidate_detailed_layer_count") != 64
        ):
            raise CampaignError(f"component diagnosis {row} detail coverage is incomplete")
        if diagnosis.get("candidate_quest_mode") != "m8":
            raise CampaignError(f"component diagnosis {row} is not an M8 Quest comparison")
        if (
            diagnosis.get("serial_scored_quest_layer_count") != 16
            or diagnosis.get("candidate_scored_quest_layer_count") != 16
        ):
            raise CampaignError(f"component diagnosis {row} Quest score coverage is incomplete")
        if diagnosis.get("candidate_gdn_mode") != "m8":
            raise CampaignError(f"component diagnosis {row} is not an M8 GDN comparison")
    if result["campaign_sha256"] != _document_digest(result, "campaign_sha256"):
        raise CampaignError("component campaign self-hash mismatch")
    return result


def _write_create_only(path: Path, value: object) -> None:
    path = path.absolute()
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) & 0o077
    ):
        raise CampaignError("output parent must be an owner-only real directory")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        payload = _canonical_json(value)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _spec_argument(value: str) -> tuple[int, Path]:
    try:
        count_raw, path_raw = value.split("=", 1)
        count = int(count_raw)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("spec run must be COUNT=PATH") from error
    if not 1 <= count <= 8 or not path_raw:
        raise argparse.ArgumentTypeError("spec run count must be within 1..8")
    return count, Path(path_raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble independently restored serial/M8 round-equivalence runs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    proposal = subparsers.add_parser("derive-proposal")
    proposal.add_argument("--serial-run", type=Path, required=True)
    proposal.add_argument("--request-id", required=True)
    proposal.add_argument("--output", type=Path, required=True)
    acceptance_proposal = subparsers.add_parser("derive-acceptance-proposal")
    acceptance_proposal.add_argument("--serial-run", type=Path, required=True)
    acceptance_proposal.add_argument("--request-id", required=True)
    acceptance_proposal.add_argument("--commit-count", type=int, required=True)
    acceptance_proposal.add_argument("--output", type=Path, required=True)
    assemble = subparsers.add_parser("assemble-state")
    assemble.add_argument("--serial-run", type=Path, required=True)
    assemble.add_argument("--zero-commit-run", type=Path, required=True)
    assemble.add_argument("--capped-run", action="append", type=_spec_argument, required=True)
    assemble.add_argument("--acceptance-run", action="append", type=_spec_argument, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-state")
    verify.add_argument("--campaign", type=Path, required=True)
    components = subparsers.add_parser("assemble-components")
    components.add_argument("--state-campaign", type=Path, required=True)
    components.add_argument("--serial-run", type=Path, required=True)
    components.add_argument("--m8-run", type=Path, required=True)
    components.add_argument("--output", type=Path, required=True)
    verify_components = subparsers.add_parser("verify-components")
    verify_components.add_argument("--campaign", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "derive-proposal":
            result = derive_forced_proposal(args.serial_run, args.request_id)
        elif args.command == "derive-acceptance-proposal":
            result = derive_acceptance_proposal(args.serial_run, args.request_id, args.commit_count)
        elif args.command == "assemble-state":
            capped_pairs: list[tuple[int, Path]] = args.capped_run
            acceptance_pairs: list[tuple[int, Path]] = args.acceptance_run
            capped_runs = dict(capped_pairs)
            acceptance_runs = dict(acceptance_pairs)
            if len(capped_runs) != len(capped_pairs):
                raise CampaignError("capped run commit widths are duplicated")
            if len(acceptance_runs) != len(acceptance_pairs):
                raise CampaignError("acceptance run commit widths are duplicated")
            result = assemble_state_campaign(
                args.serial_run,
                args.zero_commit_run,
                capped_runs,
                acceptance_runs,
            )
        elif args.command == "assemble-components":
            result = assemble_component_campaign(
                serial_run=args.serial_run,
                m8_run=args.m8_run,
                state_campaign=_json_file(
                    args.state_campaign, "state campaign", maximum_bytes=4 << 20
                ),
            )
        elif args.command == "verify-components":
            result = verify_component_campaign(
                _json_file(args.campaign, "component campaign", maximum_bytes=16 << 20)
            )
        else:
            result = verify_state_campaign(
                _json_file(args.campaign, "state campaign", maximum_bytes=4 << 20)
            )
        if args.command not in {"verify-state", "verify-components"}:
            _write_create_only(args.output, result)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        CampaignError,
        equivalence.RoundEquivalenceError,
    ) as error:
        print(f"qwen-round-equivalence-campaign: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# QWEN_ASSURANCE_ONLY_END: round-equivalence-campaign
