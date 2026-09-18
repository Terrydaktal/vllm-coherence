#!/usr/bin/env python3
"""Apply the reviewed upstream backports to an exact, stopped build tree.

This installs source changes. Native components still require a rebuild and
qualification; a successful source receipt is not a GPU qualification result.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

BUNDLE = Path(__file__).with_name("upstream-correctness")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def candidate_data_abi(
    parent_abi: str, *, bundle: Path = BUNDLE, native_bindings: dict | None = None
) -> str:
    """Keep state computed before the repairs out of the candidate namespace."""
    if not re.fullmatch(r"[0-9a-f]{64}", parent_abi):
        raise ValueError("parent snapshot ABI must be a SHA-256 identity")
    manifest = json.loads((bundle / "manifest.json").read_text())
    # Evidence labels and documentation do not change numerical state. Source
    # and installed binary changes do, even when their package versions agree.
    identities = {
        name: {entry["path"]: entry["after_sha256"] for entry in component["files"]}
        for name, component in manifest["components"].items()
    }
    contract = {
        "schema": "qwen-upstream-correctness-state-v1",
        "parent": parent_abi,
        "backports": identities,
        "native": native_bindings,
    }
    return sha256(json.dumps(contract, sort_keys=True).encode())


def apply_exact_patch(source: str, patch: str) -> str:
    """Apply unified hunks in order, without fuzzy context or external commands."""
    original = source.splitlines(keepends=True)
    result: list[str] = []
    cursor = 0
    lines = patch.splitlines(keepends=True)
    i = 0
    while i < len(lines) and not lines[i].startswith("@@ "):
        i += 1
    if i == len(lines):
        raise ValueError("patch has no hunks")
    while i < len(lines):
        match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n?", lines[i])
        if not match:
            raise ValueError("invalid unified hunk")
        start = int(match[1]) - (int(match[2] or 1) != 0)
        old_count, new_count = int(match[2] or 1), int(match[4] or 1)
        if start < cursor or start > len(original):
            raise ValueError("overlapping or out-of-range hunk")
        result.extend(original[cursor:start])
        before: list[str] = []
        after: list[str] = []
        i += 1
        while i < len(lines) and not lines[i].startswith("@@ "):
            line = lines[i]
            if line.startswith("\\"):
                # Our generated bundle only includes newline-terminated source.
                raise ValueError("unsupported unterminated source line")
            if not line or line[0] not in " +-":
                raise ValueError("invalid patch line")
            if line[0] in " -":
                before.append(line[1:])
            if line[0] in " +":
                after.append(line[1:])
            i += 1
        if len(before) != old_count or len(after) != new_count:
            raise ValueError("hunk length mismatch")
        if original[start : start + old_count] != before:
            raise ValueError("source context differs")
        result.extend(after)
        cursor = start + old_count
    result.extend(original[cursor:])
    return "".join(result)


def _path(root: Path, relative: str) -> Path:
    part = Path(relative)
    if part.is_absolute() or not part.parts or ".." in part.parts:
        raise ValueError(f"unsafe bundle path: {relative}")
    target = root / part
    for parent in (target, *target.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"symlink in source path: {relative}")
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"source escapes build root: {relative}")
    return target


def _replace(path: Path, data: bytes) -> None:
    mode = path.stat().st_mode & 0o777
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.backport-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def install(
    root: Path, component: str = "python", *, check_only: bool = False, bundle: Path = BUNDLE
) -> dict:
    root = root.resolve(strict=True)
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = manifest["components"][component]["files"]
    changes: list[tuple[Path, bytes, bytes]] = []
    receipt = {
        "schema": "urn:qwen:upstream-source-backport:v1",
        "component": component,
        "manifest_sha256": sha256(manifest_bytes),
        "check_only": check_only,
        "gpu_executed": False,
        "native_qualification": "NOT_RUN",
        "files": [],
    }
    # This lock serializes build installers. It does not make live patching safe;
    # callers must use a fresh container/source tree, never a running backend.
    with (root / ".qwen-upstream-backports.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for entry in entries:
            path = _path(root, entry["path"])
            before = path.read_bytes()
            digest = sha256(before)
            if digest == entry["after_sha256"]:
                status = "already_applied"
            elif digest == entry["before_sha256"]:
                patch_bytes = _path(bundle.resolve(), entry["patch"]).read_bytes()
                if sha256(patch_bytes) != entry["patch_sha256"]:
                    raise ValueError(f"backport payload differs: {entry['path']}")
                after = apply_exact_patch(before.decode(), patch_bytes.decode()).encode()
                if sha256(after) != entry["after_sha256"]:
                    raise ValueError(f"backport result differs: {entry['path']}")
                if path.suffix == ".py":
                    compile(after, entry["path"], "exec", dont_inherit=True)
                changes.append((path, before, after))
                status = "would_apply" if check_only else "applied"
            else:
                raise ValueError(f"unknown source preimage; refusing backport: {entry['path']}")
            receipt["files"].append(
                {"path": entry["path"], "status": status, "after_sha256": entry["after_sha256"]}
            )
        if not check_only:
            replaced: list[tuple[Path, bytes]] = []
            try:
                for path, before, after in changes:
                    if path.read_bytes() != before:
                        raise ValueError(f"source changed during installation: {path.name}")
                    _replace(path, after)
                    replaced.append((path, before))
            except BaseException:
                for path, before in reversed(replaced):
                    _replace(path, before)
                raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--component", default="python", choices=("python", "triton", "rocr", "xgrammar")
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = install(args.root, args.component, check_only=args.check_only)
    report = json.dumps(result, indent=2) + "\n"
    if args.receipt:
        args.receipt.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
