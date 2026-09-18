# QWEN_ASSURANCE_ONLY_BEGIN: coding-turbo-oracle
"""Immutable Quest96 target oracle and exact DFlash/artifact comparison.

Speculative and target-only requests commit tokens at different round boundaries.
This contract therefore compares the flattened committed target stream first,
then authenticates target KV, GDN, convolution, and cache state at every logical
length published by the speculative candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

ORACLE_SCHEMA = "urn:qwen-r9700:coding-turbo-oracle:v1"
COMPARISON_SCHEMA = "urn:qwen-r9700:coding-turbo-oracle-comparison:v1"
CONTEXTS = {60_298, 249_957}
LIFECYCLES = {
    "fresh",
    "snapshot_restore",
    "full_cache_hit",
    "partial_cache_hit",
    "cancel_retry",
    "restart",
    "offload_restore",
    "page_reuse",
    "chat_switch",
    "two_sessions",
    "two_branches",
}
COMPARISON_KINDS = {
    "artifact_equivalence",
    "dflash_invariance",
    "lifecycle_equivalence",
    "snapshot_equivalence",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

IDENTITY_KEYS = {
    "artifact_kind",
    "context_tokens",
    "dflash_enabled",
    "lifecycle",
    "model_sha256",
    "payload_producer_receipt_sha256",
    "prompt_token_ids_sha256",
    "quest_page_budget",
    "run_id",
    "runtime_artifact_manifest_sha256",
    "sampling",
    "semantic_source_sha256",
    "snapshot_manifest_sha256",
    "tokenizer_sha256",
}
CHECKPOINT_KEYS = {
    "accepted_draft_count",
    "cache_mapping_sha256",
    "canonical_gdn_layer_sha256",
    "committed_token_count",
    "committed_token_prefix_sha256",
    "convolution_layer_sha256",
    "device_error_word",
    "draft_kv_scales_sha256",
    "draft_kv_sha256",
    "logical_length",
    "nonfinite_count",
    "payload_producer_receipt_sha256",
    "rollback_generation",
    "target_kv_scales_sha256",
    "target_kv_sha256",
    "target_top1_prefix_sha256",
}
SPARSE_CHECKPOINT_KEYS = CHECKPOINT_KEYS | {"captured_round_committed_count"}
ORACLE_KEYS = {
    "checkpoints",
    "committed_target_top1_token_ids",
    "committed_token_ids",
    "identity",
    "oracle_sha256",
    "schema",
}


class OracleSafetyError(RuntimeError):
    """Raised when oracle evidence is incomplete, mutable, or divergent."""


def _canonical_json(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _document_digest(document: dict[str, Any], digest_key: str) -> str:
    unsigned = dict(document)
    unsigned.pop(digest_key, None)
    return _sha256_bytes(_canonical_json(unsigned))


def _require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OracleSafetyError(f"{label} must be an object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise OracleSafetyError(f"{label} key mismatch: missing={missing} extra={extra}")
    return value


def _require_sha256(value: object, label: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise OracleSafetyError(f"{label} must be a lowercase SHA-256")
    return value


def _require_integer(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise OracleSafetyError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _integer_tokens(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise OracleSafetyError(f"{label} must be a non-empty token-ID array")
    return [
        _require_integer(token, f"{label}[{index}]", minimum=0, maximum=2**31 - 1)
        for index, token in enumerate(value)
    ]


def _layer_hashes(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 48:
        raise OracleSafetyError(f"{label} must contain exactly 48 layer hashes")
    return [str(_require_sha256(item, f"{label}[{index}]")) for index, item in enumerate(value)]


def _prefix_digest(tokens: list[int], count: int) -> str:
    return _sha256_bytes(_canonical_json(tokens[:count]))


def _validate_identity(value: object) -> dict[str, Any]:
    identity = _require_exact_keys(value, IDENTITY_KEYS, "identity")
    kind = identity["artifact_kind"]
    if kind not in {"assurance", "release"}:
        raise OracleSafetyError("identity.artifact_kind must be assurance or release")
    context = _require_integer(
        identity["context_tokens"], "identity.context_tokens", minimum=1, maximum=1_000_000
    )
    if context not in CONTEXTS:
        raise OracleSafetyError(f"identity.context_tokens must be one of {sorted(CONTEXTS)}")
    if not isinstance(identity["dflash_enabled"], bool):
        raise OracleSafetyError("identity.dflash_enabled must be boolean")
    if identity["lifecycle"] not in LIFECYCLES:
        raise OracleSafetyError(f"identity.lifecycle must be one of {sorted(LIFECYCLES)}")
    run_id = identity["run_id"]
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise OracleSafetyError("identity.run_id is invalid")
    if identity["quest_page_budget"] != 96:
        raise OracleSafetyError("identity.quest_page_budget must be exactly 96")
    if identity["sampling"] != {"temperature": 0.0, "top_k": 1, "top_p": 1.0}:
        raise OracleSafetyError("oracle sampling must be exact greedy 0/1/1")
    for key in (
        "model_sha256",
        "payload_producer_receipt_sha256",
        "prompt_token_ids_sha256",
        "runtime_artifact_manifest_sha256",
        "semantic_source_sha256",
        "tokenizer_sha256",
    ):
        _require_sha256(identity[key], f"identity.{key}")
    _require_sha256(
        identity["snapshot_manifest_sha256"],
        "identity.snapshot_manifest_sha256",
        nullable=identity["lifecycle"] == "fresh",
    )
    if identity["lifecycle"] != "fresh" and identity["snapshot_manifest_sha256"] is None:
        raise OracleSafetyError("non-fresh lifecycle requires a snapshot manifest")
    return dict(identity)


def validate_oracle(value: object, *, sealed: bool) -> dict[str, Any]:
    """Normalize and authenticate one target oracle document."""

    expected_keys = ORACLE_KEYS if sealed else ORACLE_KEYS - {"oracle_sha256"}
    oracle = _require_exact_keys(value, expected_keys, "oracle")
    if oracle["schema"] != ORACLE_SCHEMA:
        raise OracleSafetyError(f"oracle.schema must be {ORACLE_SCHEMA}")
    identity = _validate_identity(oracle["identity"])
    committed = _integer_tokens(oracle["committed_token_ids"], "committed_token_ids")
    target_top1 = _integer_tokens(
        oracle["committed_target_top1_token_ids"], "committed_target_top1_token_ids"
    )
    if committed != target_top1:
        raise OracleSafetyError("greedy committed tokens differ from authoritative target top-1")

    raw_checkpoints = oracle["checkpoints"]
    if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
        raise OracleSafetyError("checkpoints must be a non-empty array")
    checkpoints: list[dict[str, Any]] = []
    prior_count = 0
    context = identity["context_tokens"]
    sparse_checkpoints = (
        isinstance(raw_checkpoints[0], dict) and set(raw_checkpoints[0]) == SPARSE_CHECKPOINT_KEYS
    )
    for index, raw in enumerate(raw_checkpoints):
        label = f"checkpoints[{index}]"
        checkpoint = _require_exact_keys(
            raw,
            SPARSE_CHECKPOINT_KEYS if sparse_checkpoints else CHECKPOINT_KEYS,
            label,
        )
        count = _require_integer(
            checkpoint["committed_token_count"],
            f"{label}.committed_token_count",
            minimum=1,
            maximum=len(committed),
        )
        if count <= prior_count:
            raise OracleSafetyError("checkpoint committed counts must increase strictly")
        logical_length = _require_integer(
            checkpoint["logical_length"],
            f"{label}.logical_length",
            minimum=context + 1,
            maximum=context + len(committed),
        )
        if logical_length != context + count:
            raise OracleSafetyError(f"{label}.logical_length does not match committed count")
        accepted = _require_integer(
            checkpoint["accepted_draft_count"],
            f"{label}.accepted_draft_count",
            minimum=0,
            maximum=7,
        )
        emitted = count - prior_count
        captured_round_committed = (
            _require_integer(
                checkpoint["captured_round_committed_count"],
                f"{label}.captured_round_committed_count",
                minimum=1,
                maximum=8,
            )
            if sparse_checkpoints
            else emitted
        )
        if identity["dflash_enabled"]:
            if captured_round_committed != accepted + 1:
                raise OracleSafetyError(
                    f"{label} captured round count must equal accepted drafts plus target bonus"
                )
        elif accepted != 0 or captured_round_committed != 1:
            raise OracleSafetyError(
                "target-only oracle must checkpoint one token with acceptance zero"
            )
        for key in (
            "cache_mapping_sha256",
            "draft_kv_scales_sha256",
            "draft_kv_sha256",
            "payload_producer_receipt_sha256",
            "target_kv_scales_sha256",
            "target_kv_sha256",
        ):
            _require_sha256(checkpoint[key], f"{label}.{key}")
        gdn = _layer_hashes(
            checkpoint["canonical_gdn_layer_sha256"],
            f"{label}.canonical_gdn_layer_sha256",
        )
        convolution = _layer_hashes(
            checkpoint["convolution_layer_sha256"],
            f"{label}.convolution_layer_sha256",
        )
        if checkpoint["committed_token_prefix_sha256"] != _prefix_digest(committed, count):
            raise OracleSafetyError(f"{label}.committed_token_prefix_sha256 mismatch")
        if checkpoint["target_top1_prefix_sha256"] != _prefix_digest(target_top1, count):
            raise OracleSafetyError(f"{label}.target_top1_prefix_sha256 mismatch")
        if checkpoint["device_error_word"] != 0:
            raise OracleSafetyError(f"{label}.device_error_word must be zero")
        if checkpoint["nonfinite_count"] != 0:
            raise OracleSafetyError(f"{label}.nonfinite_count must be zero")
        if (
            checkpoint["payload_producer_receipt_sha256"]
            != identity["payload_producer_receipt_sha256"]
        ):
            raise OracleSafetyError(f"{label}.payload_producer_receipt_sha256 mismatch")
        rollback = _require_integer(
            checkpoint["rollback_generation"],
            f"{label}.rollback_generation",
            minimum=0,
            maximum=2**31 - 1,
        )
        checkpoints.append(
            {
                **checkpoint,
                "canonical_gdn_layer_sha256": gdn,
                "convolution_layer_sha256": convolution,
                "rollback_generation": rollback,
            }
        )
        prior_count = count
    if prior_count != len(committed):
        raise OracleSafetyError("final checkpoint does not cover the complete committed stream")

    normalized = {
        "checkpoints": checkpoints,
        "committed_target_top1_token_ids": target_top1,
        "committed_token_ids": committed,
        "identity": identity,
        "schema": ORACLE_SCHEMA,
    }
    if sealed:
        _require_sha256(oracle["oracle_sha256"], "oracle.oracle_sha256")
        normalized["oracle_sha256"] = oracle["oracle_sha256"]
        if _document_digest(normalized, "oracle_sha256") != oracle["oracle_sha256"]:
            raise OracleSafetyError("oracle self-hash mismatch")
    return normalized


def _lstat_regular_owned(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise OracleSafetyError(f"{label} does not exist: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise OracleSafetyError(f"{label} must be an owned regular non-symlink file: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise OracleSafetyError(f"{label} must be owner-only: {path}")
    return metadata


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _lstat_regular_owned(path, label)
    payload = path.read_bytes()
    if len(payload) > 64 * 1024 * 1024:
        raise OracleSafetyError(f"{label} exceeds 64 MiB")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OracleSafetyError(f"invalid {label} JSON: {error}") from error
    if not isinstance(value, dict):
        raise OracleSafetyError(f"{label} must be a JSON object")
    return value, payload


def _require_private_parent(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise OracleSafetyError(f"output parent does not exist: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise OracleSafetyError("output parent must be an owned real directory")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise OracleSafetyError("output parent must be owner-only")


def _write_create_only(path: Path, document: dict[str, Any]) -> None:
    path = path.absolute()
    _require_private_parent(path.parent)
    if path.exists() or path.is_symlink():
        raise OracleSafetyError(f"refusing to replace output: {path}")
    payload = _canonical_json(document)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def seal_oracle(source: Path, output: Path) -> dict[str, Any]:
    value, _payload = _load_json(source.absolute(), "oracle source")
    normalized = validate_oracle(value, sealed=False)
    normalized["oracle_sha256"] = _document_digest(normalized, "oracle_sha256")
    _write_create_only(output, normalized)
    return normalized


def load_oracle(path: Path) -> dict[str, Any]:
    value, payload = _load_json(path.absolute(), "sealed oracle")
    if payload != _canonical_json(value):
        raise OracleSafetyError("sealed oracle must use canonical JSON")
    return validate_oracle(value, sealed=True)


def _identity_comparison(left: dict[str, Any], right: dict[str, Any], kind: str) -> None:
    ignored = {"artifact_kind", "dflash_enabled", "run_id", "runtime_artifact_manifest_sha256"}
    if kind in {"lifecycle_equivalence", "snapshot_equivalence"}:
        ignored |= {"lifecycle", "snapshot_manifest_sha256"}
    for key in sorted(IDENTITY_KEYS - ignored):
        if left[key] != right[key]:
            raise OracleSafetyError(f"oracle identity diverged at {key}")
    if kind == "dflash_invariance":
        if left["artifact_kind"] != right["artifact_kind"]:
            raise OracleSafetyError("DFlash comparison requires the same artifact kind")
        if left["dflash_enabled"] is not False or right["dflash_enabled"] is not True:
            raise OracleSafetyError("DFlash comparison requires target-only left and DFlash right")
        if left["runtime_artifact_manifest_sha256"] != right["runtime_artifact_manifest_sha256"]:
            raise OracleSafetyError("DFlash comparison requires one runtime artifact")
    elif kind == "artifact_equivalence":
        if left["artifact_kind"] != "assurance" or right["artifact_kind"] != "release":
            raise OracleSafetyError("artifact comparison requires assurance left and release right")
        if left["dflash_enabled"] != right["dflash_enabled"]:
            raise OracleSafetyError("artifact comparison requires identical DFlash mode")
    elif kind == "snapshot_equivalence":
        if left["lifecycle"] != "fresh" or right["lifecycle"] != "snapshot_restore":
            raise OracleSafetyError(
                "snapshot comparison requires fresh left and snapshot_restore right"
            )
        if left["artifact_kind"] != right["artifact_kind"]:
            raise OracleSafetyError("snapshot comparison requires one artifact kind")
        if left["dflash_enabled"] != right["dflash_enabled"]:
            raise OracleSafetyError("snapshot comparison requires identical DFlash mode")
        if left["runtime_artifact_manifest_sha256"] != right["runtime_artifact_manifest_sha256"]:
            raise OracleSafetyError("snapshot comparison requires one runtime artifact")
        if right["snapshot_manifest_sha256"] is None:
            raise OracleSafetyError("snapshot comparison right side has no snapshot manifest")
        if (
            left["snapshot_manifest_sha256"] is not None
            and left["snapshot_manifest_sha256"] != right["snapshot_manifest_sha256"]
        ):
            raise OracleSafetyError("fresh source names a different snapshot manifest")
    else:
        if left["lifecycle"] != "fresh":
            raise OracleSafetyError("lifecycle comparison requires a fresh reference on the left")
        if left["artifact_kind"] != right["artifact_kind"]:
            raise OracleSafetyError("lifecycle comparison requires one artifact kind")
        if left["dflash_enabled"] != right["dflash_enabled"]:
            raise OracleSafetyError("lifecycle comparison requires identical DFlash mode")
        if left["runtime_artifact_manifest_sha256"] != right["runtime_artifact_manifest_sha256"]:
            raise OracleSafetyError("lifecycle comparison requires one runtime artifact")


def compare_oracles(left: dict[str, Any], right: dict[str, Any], *, kind: str) -> dict[str, Any]:
    """Prove exact committed target/state equivalence for one oracle pair."""

    left = validate_oracle(left, sealed=True)
    right = validate_oracle(right, sealed=True)
    if kind not in COMPARISON_KINDS:
        raise OracleSafetyError(f"comparison kind must be one of {sorted(COMPARISON_KINDS)}")
    _identity_comparison(left["identity"], right["identity"], kind)
    if left["committed_token_ids"] != right["committed_token_ids"]:
        raise OracleSafetyError("committed target token stream diverged")
    if left["committed_target_top1_token_ids"] != right["committed_target_top1_token_ids"]:
        raise OracleSafetyError("committed target top-1 stream diverged")

    left_by_length = {
        checkpoint["logical_length"]: checkpoint for checkpoint in left["checkpoints"]
    }
    compared_lengths: list[int] = []
    state_keys = (
        "cache_mapping_sha256",
        "canonical_gdn_layer_sha256",
        "committed_token_prefix_sha256",
        "convolution_layer_sha256",
        "device_error_word",
        "nonfinite_count",
        "payload_producer_receipt_sha256",
        "target_kv_scales_sha256",
        "target_kv_sha256",
        "target_top1_prefix_sha256",
    )
    if kind != "lifecycle_equivalence":
        state_keys += ("rollback_generation",)
    if kind in {"artifact_equivalence", "lifecycle_equivalence", "snapshot_equivalence"}:
        state_keys += (
            "draft_kv_scales_sha256",
            "draft_kv_sha256",
        )
    for candidate in right["checkpoints"]:
        logical_length = candidate["logical_length"]
        reference = left_by_length.get(logical_length)
        if reference is None:
            raise OracleSafetyError(
                f"left oracle has no state checkpoint at logical length {logical_length}"
            )
        for key in state_keys:
            if reference[key] != candidate[key]:
                raise OracleSafetyError(
                    f"target state diverged at logical length {logical_length}: {key}"
                )
        compared_lengths.append(logical_length)

    accepted_counts = sorted(
        {
            checkpoint["accepted_draft_count"]
            for checkpoint in right["checkpoints"]
            if right["identity"]["dflash_enabled"]
        }
    )
    output_tokens = len(right["committed_token_ids"])
    context = right["identity"]["context_tokens"]
    left_rollback_generation = left["checkpoints"][-1]["rollback_generation"]
    right_rollback_generation = right["checkpoints"][-1]["rollback_generation"]
    rollback_generation_delta = right_rollback_generation - left_rollback_generation
    if kind == "lifecycle_equivalence":
        if rollback_generation_delta < 0:
            raise OracleSafetyError("lifecycle rollback generation moved backwards")
        if right["identity"]["lifecycle"] in {"cancel_retry", "chat_switch"}:
            if rollback_generation_delta < 1:
                raise OracleSafetyError(
                    "cancel/chat lifecycle did not record a rollback-generation transition"
                )
        elif rollback_generation_delta != 0:
            raise OracleSafetyError("non-rollback lifecycle changed rollback generation")
    result = {
        "accepted_draft_counts": accepted_counts,
        "boundaries_reached": {
            str(boundary): output_tokens >= boundary for boundary in (34, 152, 512, 2048)
        },
        "compared_logical_lengths": compared_lengths,
        "comparison_kind": kind,
        "context_tokens": context,
        "left_lifecycle": left["identity"]["lifecycle"],
        "left_oracle_sha256": left["oracle_sha256"],
        "left_runtime_artifact_manifest_sha256": left["identity"][
            "runtime_artifact_manifest_sha256"
        ],
        "output_tokens": output_tokens,
        "passed": True,
        "right_lifecycle": right["identity"]["lifecycle"],
        "right_oracle_sha256": right["oracle_sha256"],
        "right_runtime_artifact_manifest_sha256": right["identity"][
            "runtime_artifact_manifest_sha256"
        ],
        "rollback_generation_delta": rollback_generation_delta,
        "schema": COMPARISON_SCHEMA,
        "semantic_source_sha256": left["identity"]["semantic_source_sha256"],
    }
    result["comparison_sha256"] = _document_digest(result, "comparison_sha256")
    return result


def verify_comparison(value: object) -> dict[str, Any]:
    keys = {
        "accepted_draft_counts",
        "boundaries_reached",
        "compared_logical_lengths",
        "comparison_kind",
        "comparison_sha256",
        "context_tokens",
        "left_lifecycle",
        "left_oracle_sha256",
        "left_runtime_artifact_manifest_sha256",
        "output_tokens",
        "passed",
        "right_lifecycle",
        "right_oracle_sha256",
        "right_runtime_artifact_manifest_sha256",
        "rollback_generation_delta",
        "schema",
        "semantic_source_sha256",
    }
    comparison = _require_exact_keys(value, keys, "comparison")
    if comparison["schema"] != COMPARISON_SCHEMA or comparison["passed"] is not True:
        raise OracleSafetyError("comparison identity/pass state is invalid")
    if comparison["comparison_kind"] not in COMPARISON_KINDS:
        raise OracleSafetyError("comparison kind is invalid")
    if comparison["left_lifecycle"] not in LIFECYCLES:
        raise OracleSafetyError("comparison left lifecycle is invalid")
    if comparison["right_lifecycle"] not in LIFECYCLES:
        raise OracleSafetyError("comparison right lifecycle is invalid")
    rollback_delta = _require_integer(
        comparison["rollback_generation_delta"],
        "comparison.rollback_generation_delta",
        minimum=0,
        maximum=2**31 - 1,
    )
    if comparison["comparison_kind"] == "lifecycle_equivalence":
        if comparison["left_lifecycle"] != "fresh":
            raise OracleSafetyError("lifecycle comparison left side is not fresh")
        if comparison["right_lifecycle"] in {"cancel_retry", "chat_switch"}:
            if rollback_delta < 1:
                raise OracleSafetyError("rollback lifecycle has no rollback transition")
        elif rollback_delta != 0:
            raise OracleSafetyError("non-rollback lifecycle has a rollback transition")
    elif rollback_delta != 0:
        raise OracleSafetyError("non-lifecycle comparison has a rollback transition")
    context = _require_integer(
        comparison["context_tokens"], "comparison.context_tokens", minimum=1, maximum=1_000_000
    )
    if context not in CONTEXTS:
        raise OracleSafetyError("comparison context is not qualified")
    output_tokens = _require_integer(
        comparison["output_tokens"], "comparison.output_tokens", minimum=1, maximum=1_000_000
    )
    accepted = comparison["accepted_draft_counts"]
    if (
        not isinstance(accepted, list)
        or accepted != sorted(set(accepted))
        or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 7
            for item in accepted
        )
    ):
        raise OracleSafetyError(
            "comparison accepted counts must be unique sorted integers in [0, 7]"
        )
    boundaries = _require_exact_keys(
        comparison["boundaries_reached"], {"34", "152", "512", "2048"}, "boundaries"
    )
    for boundary, reached in boundaries.items():
        if not isinstance(reached, bool) or reached != (output_tokens >= int(boundary)):
            raise OracleSafetyError(f"comparison boundary result is inconsistent at {boundary}")
    lengths = comparison["compared_logical_lengths"]
    if (
        not isinstance(lengths, list)
        or not lengths
        or any(isinstance(item, bool) or not isinstance(item, int) for item in lengths)
        or lengths != sorted(set(lengths))
        or lengths[-1] != context + output_tokens
    ):
        raise OracleSafetyError("comparison logical lengths are invalid or incomplete")
    for key in (
        "comparison_sha256",
        "left_oracle_sha256",
        "left_runtime_artifact_manifest_sha256",
        "right_oracle_sha256",
        "right_runtime_artifact_manifest_sha256",
        "semantic_source_sha256",
    ):
        _require_sha256(comparison[key], f"comparison.{key}")
    if _document_digest(comparison, "comparison_sha256") != comparison["comparison_sha256"]:
        raise OracleSafetyError("comparison self-hash mismatch")
    return dict(comparison)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal and compare immutable Quest96 target oracles."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--source", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--kind", choices=sorted(COMPARISON_KINDS), required=True)
    compare.add_argument("--output", type=Path, required=True)
    reduce_capture = subparsers.add_parser(
        "reduce", help="join token and post-commit state captures into an oracle source"
    )
    reduce_capture.add_argument("--identity", type=Path, required=True)
    reduce_capture.add_argument("--round-stream", type=Path, required=True)
    reduce_capture.add_argument("--state-stream", type=Path, required=True)
    reduce_capture.add_argument("--result", type=Path, required=True)
    reduce_capture.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--oracle", type=Path)
    verify.add_argument("--comparison", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal":
            result = seal_oracle(args.source, args.output)
        elif args.command == "compare":
            left = load_oracle(args.left)
            right = load_oracle(args.right)
            result = compare_oracles(left, right, kind=args.kind)
            _write_create_only(args.output, result)
        elif args.command == "reduce":
            from qwen_r9700_lab.coding_turbo_capture import (
                CaptureSafetyError,
                reduce_capture,
            )

            try:
                result = reduce_capture(
                    identity_path=args.identity,
                    round_stream=args.round_stream,
                    state_stream=args.state_stream,
                    result_path=args.result,
                    output=args.output,
                )
            except CaptureSafetyError as error:
                raise OracleSafetyError(str(error)) from error
        elif bool(args.oracle) == bool(args.comparison):
            raise OracleSafetyError("verify requires exactly one of --oracle or --comparison")
        elif args.oracle:
            result = load_oracle(args.oracle)
        else:
            value, payload = _load_json(args.comparison.absolute(), "comparison")
            if payload != _canonical_json(value):
                raise OracleSafetyError("comparison must use canonical JSON")
            result = verify_comparison(value)
    except OracleSafetyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# QWEN_ASSURANCE_ONLY_END: coding-turbo-oracle
