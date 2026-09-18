"""Immutable adversarial M8 evidence for row-local Quest96 semantics."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any

SCHEMA = "urn:qwen-r9700:coding-turbo-m8-row-invariance:v2"
CONTEXTS = {60_298, 249_957}
FIXED_ROWS = tuple(range(8))
MIN_RANDOMIZATIONS_PER_ROW = 128
MIN_PERMUTATIONS_PER_ROW = 8
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TRIAL_KINDS = {"row_permutation", "sibling_randomization"}

ROOT_KEYS = {
    "artifact_manifest_sha256",
    "context_tokens",
    "evidence_sha256",
    "fixed_rows",
    "schema",
    "semantic_source_sha256",
    "trials",
}
TRIAL_KEYS = {
    "fixed_row",
    "fixed_row_input_sha256",
    "m8_attention_output_sha256",
    "m8_cache_mapping_sha256",
    "m8_capture_sha256",
    "m8_selected_pages_sha256",
    "m8_target_top1",
    "perturbation_sha256",
    "standalone_attention_output_sha256",
    "standalone_cache_mapping_sha256",
    "standalone_capture_sha256",
    "standalone_selected_pages_sha256",
    "standalone_target_top1",
    "trial_index",
    "trial_kind",
}


class M8EvidenceError(RuntimeError):
    """Adversarial M8 evidence is incomplete, malformed, or divergent."""


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _document_digest(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("evidence_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _exact_object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise M8EvidenceError(f"{label} must be an object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise M8EvidenceError(f"{label} key mismatch: missing={missing} extra={extra}")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise M8EvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise M8EvidenceError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def validate_evidence(value: object, *, sealed: bool = True) -> dict[str, Any]:
    expected_root = ROOT_KEYS if sealed else ROOT_KEYS - {"evidence_sha256"}
    root = _exact_object(value, expected_root, "M8 evidence")
    if root["schema"] != SCHEMA:
        raise M8EvidenceError("M8 evidence schema mismatch")
    context = _integer(root["context_tokens"], "context_tokens", minimum=1, maximum=1_000_000)
    if context not in CONTEXTS:
        raise M8EvidenceError("M8 evidence context is not qualified")
    artifact_manifest = _sha256(root["artifact_manifest_sha256"], "artifact_manifest_sha256")
    semantic_source = _sha256(root["semantic_source_sha256"], "semantic_source_sha256")
    if root["fixed_rows"] != list(FIXED_ROWS):
        raise M8EvidenceError("M8 evidence must cover fixed rows 0 through 7")
    raw_trials = root["trials"]
    if not isinstance(raw_trials, list) or not raw_trials:
        raise M8EvidenceError("M8 evidence trials must be a non-empty array")

    normalized_trials: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    coverage: Counter[tuple[str, int]] = Counter()
    for index, value in enumerate(raw_trials):
        label = f"trials[{index}]"
        trial = dict(_exact_object(value, TRIAL_KEYS, label))
        kind = trial["trial_kind"]
        if kind not in TRIAL_KINDS:
            raise M8EvidenceError(f"{label}.trial_kind is invalid")
        fixed_row = _integer(trial["fixed_row"], f"{label}.fixed_row", minimum=0, maximum=7)
        trial_index = _integer(
            trial["trial_index"], f"{label}.trial_index", minimum=0, maximum=2**31 - 1
        )
        identity = (kind, fixed_row, trial_index)
        if identity in seen:
            raise M8EvidenceError(f"duplicate M8 trial identity: {identity}")
        seen.add(identity)
        coverage[(kind, fixed_row)] += 1
        for key in TRIAL_KEYS - {
            "fixed_row",
            "m8_target_top1",
            "standalone_target_top1",
            "trial_index",
            "trial_kind",
        }:
            trial[key] = _sha256(trial[key], f"{label}.{key}")
        standalone_top1 = _integer(
            trial["standalone_target_top1"],
            f"{label}.standalone_target_top1",
            minimum=0,
            maximum=2**31 - 1,
        )
        m8_top1 = _integer(
            trial["m8_target_top1"],
            f"{label}.m8_target_top1",
            minimum=0,
            maximum=2**31 - 1,
        )
        equal_pairs = (
            ("standalone_selected_pages_sha256", "m8_selected_pages_sha256"),
            ("standalone_attention_output_sha256", "m8_attention_output_sha256"),
            ("standalone_cache_mapping_sha256", "m8_cache_mapping_sha256"),
        )
        for standalone, m8 in equal_pairs:
            if trial[standalone] != trial[m8]:
                raise M8EvidenceError(f"{label} diverged at {m8}")
        if standalone_top1 != m8_top1:
            raise M8EvidenceError(f"{label} target top-1 diverged")
        normalized_trials.append(trial)

    for row in FIXED_ROWS:
        if coverage[("sibling_randomization", row)] < MIN_RANDOMIZATIONS_PER_ROW:
            raise M8EvidenceError(f"fixed row {row} has insufficient sibling randomizations")
        if coverage[("row_permutation", row)] < MIN_PERMUTATIONS_PER_ROW:
            raise M8EvidenceError(f"fixed row {row} has insufficient row permutations")

    normalized = {
        "artifact_manifest_sha256": artifact_manifest,
        "context_tokens": context,
        "fixed_rows": list(FIXED_ROWS),
        "schema": SCHEMA,
        "semantic_source_sha256": semantic_source,
        "trials": normalized_trials,
    }
    if sealed:
        normalized["evidence_sha256"] = _sha256(root["evidence_sha256"], "evidence_sha256")
        if normalized["evidence_sha256"] != _document_digest(normalized):
            raise M8EvidenceError("M8 evidence self-hash mismatch")
    return normalized


def seal_evidence(value: object) -> dict[str, Any]:
    normalized = validate_evidence(value, sealed=False)
    normalized["evidence_sha256"] = _document_digest(normalized)
    return validate_evidence(normalized, sealed=True)


def coverage_summary(value: object) -> dict[str, Any]:
    evidence = validate_evidence(value, sealed=True)
    counts = Counter((trial["trial_kind"], trial["fixed_row"]) for trial in evidence["trials"])
    return {
        "fixed_rows": list(FIXED_ROWS),
        "row_permutations": sum(counts[("row_permutation", row)] for row in FIXED_ROWS),
        "sibling_randomizations": sum(counts[("sibling_randomization", row)] for row in FIXED_ROWS),
    }
