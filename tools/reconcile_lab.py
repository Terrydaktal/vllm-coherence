#!/usr/bin/env python3
"""Link a private lab's shared source files to this authoritative checkout.

Plan/apply/check operate on source paths only; they never open sessions, captures,
model weights or GPU state. Apply backs up originals before replacing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = (
    "src/qwen_r9700_lab/",
    "scripts/",
    "integrations/pi/",
    "experiments/radiance-public/",
    "tests/",
    "tools/",
    "configs/",
    "schemas/",
)
EXCLUDED = {
    "artifacts",
    "captures",
    "incidents",
    "sessions",
    ".pi",
    "node_modules",
    "__pycache__",
}
SCHEMA = "coherence-lab-links-v1"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def source_paths(root: Path) -> list[str]:
    names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted({name for name in names.split("\0") if admitted(name)})


def admitted(name: str) -> bool:
    path = Path(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and name.startswith(PREFIXES)
        and not EXCLUDED.intersection(path.parts)
    )


def fingerprint(path: Path) -> dict:
    if path.is_symlink():
        return {"kind": "link", "target": os.readlink(path)}
    if not path.exists():
        return {"kind": "missing"}
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"not a regular source file: {path}")
    return {
        "kind": "file",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": stat.S_IMODE(info.st_mode),
    }


def safe_path(root: Path, name: str) -> Path:
    if not admitted(name):
        raise ValueError(f"not an admitted source path: {name}")
    path = root / name
    for parent in path.parents:
        if parent == root:
            return path
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError(f"indirect source directory: {parent}")
    raise ValueError("source path outside root")


def expected_link(canonical: Path, lab: Path, name: str) -> dict:
    return {
        "kind": "link",
        "target": os.path.relpath(canonical / name, (lab / name).parent),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".reconcile-")
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_roots(canonical: Path, lab: Path) -> None:
    if canonical == lab or canonical in lab.parents or lab in canonical.parents:
        raise ValueError("canonical and lab must be separate, non-nested directories")
    if not canonical.is_dir() or not lab.is_dir():
        raise ValueError("both source directories must exist")


def make_plan(canonical: Path, lab: Path) -> dict:
    validate_roots(canonical, lab)
    entries = []
    for name in source_paths(canonical):
        source = fingerprint(safe_path(canonical, name))
        if source["kind"] != "file":
            raise ValueError(f"canonical source must be a regular file: {name}")
        entries.append(
            {
                "path": name,
                "source": source,
                "before": fingerprint(safe_path(lab, name)),
            }
        )
    return {
        "schema": SCHEMA,
        "canonical": str(canonical),
        "lab": str(lab),
        "canonical_head": git(canonical, "rev-parse", "HEAD").strip(),
        "lab_head": git(lab, "rev-parse", "HEAD").strip(),
        "canonical_status": git(canonical, "status", "--porcelain=v1"),
        "lab_status": git(lab, "status", "--porcelain=v1"),
        "entries": entries,
    }


def roots(plan: dict) -> tuple[Path, Path]:
    if plan["schema"] != SCHEMA:
        raise ValueError("unsupported reconciliation manifest")
    canonical, lab = Path(plan["canonical"]).resolve(), Path(plan["lab"]).resolve()
    validate_roots(canonical, lab)
    names = [row["path"] for row in plan["entries"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate plan entry")
    return canonical, lab


def replace_link(path: Path, target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".reconcile-")
    os.close(fd)
    tmp = Path(temporary)
    try:
        tmp.unlink()
        tmp.symlink_to(target)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def apply_plan(plan: dict, backup_root: Path) -> Path | None:
    canonical, lab = roots(plan)
    # Refuse stale plans before any source changes. This preserves concurrent work.
    if source_paths(canonical) != [row["path"] for row in plan["entries"]]:
        raise ValueError("canonical file list changed; generate a new plan")
    for row in plan["entries"]:
        name = row["path"]
        if fingerprint(safe_path(canonical, name)) != row["source"]:
            raise ValueError(f"canonical source changed since plan: {name}")
        if fingerprint(safe_path(lab, name)) != row["before"]:
            raise ValueError(f"lab source changed since plan: {name}")
    changes = [
        row
        for row in plan["entries"]
        if row["before"] != expected_link(canonical, lab, row["path"])
    ]
    if not changes:
        return None
    backup_root = backup_root.resolve()
    if any(p == backup_root or p in backup_root.parents for p in (canonical, lab)):
        raise ValueError("backups must be outside both repositories")
    backup_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    backup = Path(
        tempfile.mkdtemp(
            prefix=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-"),
            dir=backup_root,
        )
    )
    manifest = {**plan, "entries": changes, "state": "backing-up"}
    manifest_path = backup / "manifest.json"
    write_json(manifest_path, manifest)
    # Keep every original byte (including dirty/untracked source), not only diffs.
    for row in changes:
        if row["before"]["kind"] == "file":
            saved = backup / "originals" / row["path"]
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lab / row["path"], saved)
            if fingerprint(saved) != row["before"]:
                raise ValueError(f"original changed while backing up: {row['path']}")
    manifest["state"] = "applying"
    write_json(manifest_path, manifest)
    try:
        for row in changes:
            path = safe_path(lab, row["path"])
            if fingerprint(path) != row["before"]:
                raise ValueError(f"lab source changed during apply: {row['path']}")
            replace_link(path, expected_link(canonical, lab, row["path"])["target"])
    except Exception as exc:
        raise RuntimeError(
            f"partial reconciliation; originals retained in {manifest_path}"
        ) from exc
    manifest["state"] = "applied"
    write_json(manifest_path, manifest)
    return manifest_path


def restore(manifest_path: Path) -> int:
    plan = json.loads(manifest_path.read_text())
    canonical, lab = roots(plan)
    pending = []
    for row in plan["entries"]:
        current = fingerprint(safe_path(lab, row["path"]))
        if current == row["before"]:
            continue
        if current != expected_link(canonical, lab, row["path"]):
            raise ValueError(
                f"refusing to overwrite subsequent lab work: {row['path']}"
            )
        if row["before"]["kind"] == "file" and (
            fingerprint(manifest_path.parent / "originals" / row["path"])
            != row["before"]
        ):
            raise ValueError(f"missing or changed backup: {row['path']}")
        pending.append(row)
    for row in pending:
        path = safe_path(lab, row["path"])
        original = row["before"]
        if original["kind"] == "missing":
            path.unlink()
        elif original["kind"] == "link":
            replace_link(path, original["target"])
        else:
            fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".reconcile-")
            os.close(fd)
            try:
                shutil.copy2(
                    manifest_path.parent / "originals" / row["path"], temporary
                )
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
    plan["state"] = "restored"
    write_json(manifest_path, plan)
    return len(pending)


def check(canonical: Path, lab: Path) -> dict:
    validate_roots(canonical, lab)
    files = source_paths(canonical)
    drift = []
    for name in files:
        source = fingerprint(safe_path(canonical, name))
        if source["kind"] != "file" or fingerprint(
            safe_path(lab, name)
        ) != expected_link(canonical, lab, name):
            drift.append(name)
    return {
        "canonical": str(canonical),
        "lab": str(lab),
        "shared_files": len(files),
        "drift": drift,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "check"):
        sub = commands.add_parser(command)
        sub.add_argument("--lab", type=Path, required=True)
        if command == "plan":
            sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("apply")
    sub.add_argument("--plan", type=Path, required=True)
    sub.add_argument(
        "--backup-root",
        type=Path,
        default=Path.home() / ".local/state/vllm-coherence/reconciliation",
    )
    sub = commands.add_parser("restore")
    sub.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan = make_plan(ROOT, args.lab.resolve())
        write_json(args.output, plan)
        changed = [
            row["path"]
            for row in plan["entries"]
            if row["before"]["kind"] == "file" and row["before"] != row["source"]
        ]
        print(
            json.dumps(
                {
                    "plan": str(args.output),
                    "source_files": len(plan["entries"]),
                    "different_existing_files": changed,
                },
                indent=2,
            )
        )
    elif args.command == "apply":
        result = apply_plan(json.loads(args.plan.read_text()), args.backup_root)
        print(json.dumps({"manifest": str(result) if result else None}))
    elif args.command == "restore":
        print(json.dumps({"restored_files": restore(args.manifest)}))
    else:
        result = check(ROOT, args.lab.resolve())
        print(json.dumps(result, indent=2))
        return bool(result["drift"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
