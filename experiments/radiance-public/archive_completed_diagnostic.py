"""Admission for reviewed, completed diagnostics; never classify a run as passing.

The admission document records already existing completion receipts, not a new
completion claim. Payload retirement is limited to captured SHA-named tensors.
"""

import hashlib
import os
import re
import stat
from pathlib import Path, PurePosixPath

from archive_conformance_run import owned_bytes, sealed
from archive_inactive_benchmark_cache import live_references

SCHEMA = "urn:qwen:completed-diagnostic-admission:v1"


def domain(root):
    root = Path(root).absolute()
    if root.resolve(strict=True) != root:
        raise ValueError("diagnostic path contains a symlink")
    if root.parent.name == "diagnostics" and re.fullmatch(r"[a-z0-9-]+-[0-9]{3}", root.name):
        return root, root.name, {"result.json", "probe-result.json"}
    if (
        root.name == "run"
        and root.parent.parent.name == "preflight"
        and re.fullmatch(r"reference-reuse-native-admission-[0-9]{3}", root.parent.name)
    ):
        return root.parent, root.parent.name, {"driver-result.json"}
    raise ValueError("outside the reviewed diagnostic namespace")


def finished_diagnostic_identity(root):
    root = Path(root).absolute()
    receipt_root, activity_name, allowed = domain(root)
    review = sealed(root / "archive-admission.json")
    info = root.lstat()
    if (
        review.get("schema") != SCHEMA
        or review.get("root") != str(root)
        or review.get("root_signature")
        != {"device": info.st_dev, "inode": info.st_ino, "owner": info.st_uid}
        or info.st_uid != os.getuid()
        or not stat.S_ISDIR(info.st_mode)
        or review.get("known_live_references") != []
    ):
        raise ValueError("diagnostic admission does not match this owned root")
    receipts = review.get("receipts", {})
    if not receipts or not set(receipts).issubset(allowed):
        raise ValueError("diagnostic completion receipts were not reviewed")
    terminal = False
    for name, expected in receipts.items():
        path = receipt_root / name
        result = sealed(path)
        if hashlib.sha256(owned_bytes(path)).hexdigest() != expected:
            raise ValueError("reviewed diagnostic completion changed")
        terminal |= type(result.get("returncode")) is int or result.get("status") in {
            "DISCREPANCY",
            "FAILED_OR_DISCREPANT",
            "EXPERIMENT_COMPLETED",
            "TESTED",
            "MEASURED",
            "PASSED",
            "DIAGNOSTIC_MEASURED",
        }
        arms = result.get("arms")
        terminal |= (
            result.get("complete") is True
            and isinstance(arms, list)
            and bool(arms)
            and all(isinstance(row, dict) and type(row.get("returncode")) is int for row in arms)
        )
    if not terminal:
        raise ValueError("no reviewed terminal diagnostic result")
    releases = {}
    for lease in sorted(root.rglob("gpu-lease")):
        if lease.is_symlink() or not lease.is_dir():
            raise ValueError("invalid diagnostic lease directory")
        release = sealed(lease / "released.json")
        releases[str(lease.relative_to(root))] = release["sha256"]
    if releases != review.get("gpu_releases", {}):
        raise ValueError("diagnostic GPU releases changed")
    activity = live_references(activity_name)
    if activity["references"]:
        raise ValueError("diagnostic still has live process references")
    return {
        "scope": SCHEMA,
        "review": review["sha256"],
        "receipts": receipts,
        "gpu_releases": releases,
    }


def diagnostic_payload_path(name, root_name, **_):
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.parts[0] != root_name
    ):
        raise ValueError("invalid archived diagnostic path")
    parts = relative.parts[1:]
    root_frame = len(parts) == 2 and re.fullmatch(r"frame-[0-9]{6}", parts[0])
    root_boundary = (
        len(parts) == 3
        and parts[0] in {"boundaries", "semantic"}
        and re.fullmatch(r"p[0-9]{9}-l[0-9]{3}-[a-zA-Z0-9_.-]+", parts[1])
    )
    if (
        len(parts) >= 2
        and re.fullmatch(r"[0-9a-f]{64}\.bin", parts[-1])
        and (root_frame or root_boundary or {"capture", "reference"}.intersection(parts[:-1]))
    ):
        return Path(*parts)
    return None
