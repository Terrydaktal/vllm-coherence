"""Verify exhaustive failure-atomic transaction fault campaigns.

The runtime producer injects one failure at every declared semantic write point
for every external commit width.  This module is deliberately only a verifier:
it cannot manufacture evidence and it never launches or mutates a model.
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

from qwen_r9700_lab.assurance_fault_contract import FAULT_SITES

CAMPAIGN_SCHEMA = "urn:qwen-r9700:failure-atomic-transaction-campaign:v2"
CONTEXTS = (60_298, 249_957)
COMMIT_COUNTS = tuple(range(9))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROLLBACK_FAULT_POINTS = FAULT_SITES
COMMITTED_FAULT_POINT = "commit.after_publication_before_ack"
CASE_KEYS = {
    "canonical_after_sha256",
    "canonical_before_sha256",
    "commit_count",
    "device_error_word",
    "failure_observed",
    "fault_point",
    "nonfinite_count",
    "outcome",
    "ownership_after_sha256",
    "ownership_before_sha256",
    "publication_epoch_after",
    "publication_epoch_before",
    "recovered_state_sha256",
    "serial_after_sha256",
}
CAMPAIGN_KEYS = {
    "campaign_sha256",
    "cases",
    "commit_counts",
    "context_tokens",
    "fault_points",
    "passed",
    "runtime_artifact_manifest_sha256",
    "schema",
    "semantic_source_sha256",
    "snapshot_manifest_sha256",
}


class TransactionCampaignError(RuntimeError):
    """A fault campaign is malformed, incomplete, or non-atomic."""


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _document_digest(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("campaign_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise TransactionCampaignError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise TransactionCampaignError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def verify_campaign(value: object) -> dict[str, Any]:
    """Require the complete context by commit-width by fault-point domain."""

    if not isinstance(value, dict) or set(value) != CAMPAIGN_KEYS:
        raise TransactionCampaignError("transaction campaign keys are invalid")
    campaign = dict(value)
    if campaign["schema"] != CAMPAIGN_SCHEMA or campaign["passed"] is not True:
        raise TransactionCampaignError("transaction campaign is not passing v2 evidence")
    context = _integer(campaign["context_tokens"], "context_tokens", 1, 1_000_000)
    if context not in CONTEXTS:
        raise TransactionCampaignError("transaction campaign context is not qualified")
    if campaign["commit_counts"] != list(COMMIT_COUNTS):
        raise TransactionCampaignError("transaction campaign commit widths are incomplete")
    if campaign["fault_points"] != [*ROLLBACK_FAULT_POINTS, COMMITTED_FAULT_POINT]:
        raise TransactionCampaignError("transaction campaign fault-point inventory differs")
    for key in (
        "campaign_sha256",
        "runtime_artifact_manifest_sha256",
        "semantic_source_sha256",
        "snapshot_manifest_sha256",
    ):
        _digest(campaign[key], key)

    cases = campaign["cases"]
    if not isinstance(cases, list):
        raise TransactionCampaignError("transaction campaign cases must be a list")
    required = {
        (commit_count, fault_point)
        for commit_count in COMMIT_COUNTS
        for fault_point in ROLLBACK_FAULT_POINTS
    } | {(commit_count, COMMITTED_FAULT_POINT) for commit_count in range(1, 9)}
    observed: set[tuple[int, str]] = set()
    for index, raw_case in enumerate(cases):
        if not isinstance(raw_case, dict) or set(raw_case) != CASE_KEYS:
            raise TransactionCampaignError(f"transaction case {index} keys are invalid")
        case = dict(raw_case)
        commit_count = _integer(case["commit_count"], f"case {index} commit_count", 0, 8)
        fault_point = case["fault_point"]
        key = (commit_count, fault_point)
        if key not in required or key in observed:
            raise TransactionCampaignError(
                f"transaction case {index} is unexpected or duplicated: {key}"
            )
        observed.add(key)
        if case["failure_observed"] is not True:
            raise TransactionCampaignError(f"transaction case {index} did not inject a failure")
        if case["device_error_word"] != 0 or case["nonfinite_count"] != 0:
            raise TransactionCampaignError(f"transaction case {index} observed an unsafe device")
        before = _digest(case["canonical_before_sha256"], f"case {index} before")
        after = _digest(case["canonical_after_sha256"], f"case {index} after")
        recovered = _digest(case["recovered_state_sha256"], f"case {index} recovered")
        serial_after = _digest(case["serial_after_sha256"], f"case {index} serial")
        owner_before = _digest(case["ownership_before_sha256"], f"case {index} owner before")
        owner_after = _digest(case["ownership_after_sha256"], f"case {index} owner after")
        epoch_before = _integer(
            case["publication_epoch_before"], f"case {index} epoch before", 0, 2**63 - 1
        )
        epoch_after = _integer(
            case["publication_epoch_after"], f"case {index} epoch after", 0, 2**63 - 1
        )
        if fault_point == COMMITTED_FAULT_POINT:
            if case["outcome"] != "committed":
                raise TransactionCampaignError(f"transaction case {index} lost committed outcome")
            publication_changed_once = epoch_after == epoch_before + 1
            if after != serial_after or recovered != serial_after or not publication_changed_once:
                raise TransactionCampaignError(
                    f"transaction case {index} did not recover the published serial transition"
                )
        else:
            if case["outcome"] != "rolled_back":
                raise TransactionCampaignError(f"transaction case {index} did not roll back")
            if after != before or recovered != before or owner_after != owner_before:
                raise TransactionCampaignError(
                    f"transaction case {index} changed canonical bytes or ownership"
                )
            if epoch_after != epoch_before:
                raise TransactionCampaignError(
                    f"transaction case {index} advanced publication during rollback"
                )
    if observed != required:
        missing = sorted(required - observed)
        raise TransactionCampaignError(f"transaction campaign cases are incomplete: {missing[:8]}")
    if campaign["campaign_sha256"] != _document_digest(campaign):
        raise TransactionCampaignError("transaction campaign self-hash mismatch")
    return campaign


def _load_private(path: Path) -> object:
    path = path.absolute()
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise TransactionCampaignError("campaign must be an owner-only regular file")
    return json.loads(path.read_bytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify one failure-atomic transaction campaign.")
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_campaign(_load_private(args.campaign))
    except (OSError, UnicodeError, json.JSONDecodeError, TransactionCampaignError) as error:
        print(f"qwen-transaction-campaign: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
