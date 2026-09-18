"""Archive an explicitly reviewed abandoned queue without inventing completion.

The normal queue lock must be held by source_guard throughout streaming or
retirement. Live process references, changed metadata and unreleased GPU leases
reject admission. Original running/incomplete statuses are retained verbatim.
"""

import hashlib
import os
import re
import stat
from pathlib import Path, PurePosixPath

from archive_conformance_run import owned_bytes, sealed
from archive_inactive_benchmark_cache import live_references

SCHEMA = "urn:qwen:abandoned-conformance-run-admission:v1"
CASE = r"case-[0-9]{5}"
METADATA = (
    "input.json",
    "result.json",
    "worker-result.json",
    "process/invocation.json",
    "process/shutdown.json",
)


def inventory(root):
    """Bind original metadata presence as well as bytes, without reclassifying it."""
    root = Path(root).absolute()
    if (
        root.resolve(strict=True) != root
        or not re.fullmatch(r"run-[0-9]{3}(?:-cpu|-gpu)?", root.name)
        or not re.fullmatch(r"[0-9]{8}", root.parent.name)
        or root.parent.parent.name != "conformance"
    ):
        raise ValueError("outside the reviewed abandoned-queue namespace")
    checkpoint = sealed(root / "checkpoint.json")
    campaign = sealed(root / "campaign.json")
    if checkpoint.get("campaign") != campaign["sha256"] or checkpoint.get("status") not in {
        "running_case",
        "interrupted",
        "error",
    }:
        raise ValueError("not an abandoned incomplete queue")
    cases, leases = {}, {}
    for case in sorted(root.glob("case-*")):
        if not re.fullmatch(CASE, case.name) or case.is_symlink() or not case.is_dir():
            raise ValueError("unexpected abandoned case directory")
        if not (case / "input.json").exists():
            raise ValueError("abandoned case has no original input metadata")
        cases[case.name] = {
            name: hashlib.sha256(owned_bytes(case / name)).hexdigest()
            if (case / name).exists()
            else None
            for name in METADATA
        }
        lease = case / "gpu-lease"
        if lease.exists():
            if lease.is_symlink() or not lease.is_dir():
                raise ValueError("invalid abandoned GPU lease directory")
            leases[str(lease.relative_to(root))] = sealed(lease / "released.json")["sha256"]
    if not cases:
        raise ValueError("empty abandoned queue")
    return {
        "campaign": campaign["sha256"],
        "checkpoint": checkpoint["sha256"],
        "case_metadata": cases,
        "gpu_releases": leases,
    }


def abandoned_identity(root):
    root = Path(root).absolute()
    original = inventory(root)
    review = sealed(root / "archive-abandoned-admission.json")
    info = root.lstat()
    if (
        review.get("schema") != SCHEMA
        or review.get("root") != str(root)
        or review.get("root_signature")
        != {"device": info.st_dev, "inode": info.st_ino, "owner": info.st_uid}
        or info.st_uid != os.getuid()
        or not stat.S_ISDIR(info.st_mode)
        or review.get("original") != original
        or review.get("known_live_references") != []
        or review.get("preserve_incomplete_status") is not True
    ):
        raise ValueError("abandoned-queue review does not match the original evidence")
    if live_references(root.name)["references"]:
        raise ValueError("abandoned queue has live process references")
    return {"scope": SCHEMA, "review": review["sha256"], **original}


def abandoned_payload_path(name, root_name, **_):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or path.parts[0] != root_name:
        raise ValueError("invalid archived abandoned-queue path")
    parts = path.parts[1:]
    if (
        len(parts) >= 4
        and re.fullmatch(CASE, parts[0])
        and {"capture", "reference"}.intersection(parts[1:-1])
        and re.fullmatch(r"[0-9a-f]{64}\.bin", parts[-1])
    ):
        return Path(*parts)
    return None
