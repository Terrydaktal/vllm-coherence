"""Retire archived capture/cache files, retaining case results and run metadata.

Only a guarded inactive source with a successful independent archive readback
is admitted. Completed, diagnostic and explicitly reviewed abandoned sources
have distinct admission rules. Each removed file must still match the archived SHA256. The archive
locator is durably written before removal; exceptions leave a partial retirement
with its remaining originals intact. This operation never forgets an archive.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path, PurePosixPath

from archive_conformance_run import owned_bytes, sealed, source_guard, write_new


def payload_path(name, root_name, *, case=False):
    relative = PurePosixPath(name)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[0] != root_name
    ):
        raise ValueError("invalid archived path")
    parts = relative.parts[1:]
    if not case and (len(parts) < 3 or not parts[0].startswith("case-")):
        return None
    directories = parts[:-1] if case else parts[1:-1]
    if not {"capture", "runtime", "data"}.intersection(directories):
        return None
    return Path(*parts)


def signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def reference_payload_path(name, root_name, *, case=False):
    """Select canonical CPU reference blobs while retaining every JSON descriptor."""
    relative = PurePosixPath(name)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[0] != root_name
    ):
        raise ValueError("invalid archived path")
    parts = relative.parts[1:]
    if not case and (len(parts) < 3 or not parts[0].startswith("case-")):
        return None
    content = parts if case else parts[1:]
    if (
        len(content) < 3
        or content[0] != "reference"
        or not re.fullmatch(r"[0-9a-f]{64}\.bin", content[-1])
    ):
        return None
    frame = len(content) == 3 and re.fullmatch(r"frame-[0-9]{6}", content[1])
    boundary = (
        len(content) == 4
        and content[1] in {"boundaries", "semantic"}
        and re.fullmatch(r"p[0-9]{9}-l[0-9]{3}-[a-zA-Z0-9_.-]+", content[2])
    )
    return Path(*parts) if frame or boundary else None


def retire(root, report, manifest, completion, repository, *, reference_captures_only=False):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("source root must not be a symlink")
    root = root.resolve(strict=True)
    report_bytes = Path(report).read_bytes()
    verified = json.loads(report_bytes)
    completed = json.loads(Path(completion).read_bytes())
    if (
        verified.get("schema") != "urn:qwen:conformance-archive-readback:v1"
        or verified.get("verified") is not True
        or completed.get("readback_report_sha256") != hashlib.sha256(report_bytes).hexdigest()
        or not re.fullmatch(r"[0-9a-f]{64}", completed.get("snapshot_id", ""))
    ):
        raise ValueError("missing verified archive completion")
    with Path(manifest).open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != verified["manifest_sha256"]:
            raise ValueError("archive member manifest changed")
    source = verified["stream_receipt"]
    if source["source"] != str(root) or source["root_name"] != root.name:
        raise ValueError("archive belongs to another source directory")
    scope = source.get("scope", "run")
    if scope not in {"run", "case", "paused-run", "diagnostic", "abandoned-run"}:
        raise ValueError("unknown archive scope")
    if scope in {"diagnostic", "abandoned-run"} and reference_captures_only:
        raise ValueError("this scope uses a single restricted tensor retirement policy")
    count = removed_bytes = previously_absent = 0
    with source_guard(
        root,
        case=scope == "case",
        paused=scope == "paused-run",
        diagnostic=scope == "diagnostic",
        abandoned=scope == "abandoned-run",
    ) as identity_of:
        if identity_of(root) != source["identity"]:
            raise ValueError("completed source identity changed")
        locator = {
            "schema": "urn:qwen:conformance-archive-locator:v1",
            "repository": repository,
            "snapshot_id": completed["snapshot_id"],
            "archive_path": f"/{root.name}.tar",
            "tar_sha256": source["tar_sha256"],
            "manifest_sha256": verified["manifest_sha256"],
            "identity": source["identity"],
            "retirement_started_ns": time.time_ns(),
        }
        prefix = "archive-reference" if reference_captures_only else "archive"
        locator_path = root / (prefix + "-locator.json")
        retirement_path = root / (prefix + "-retirement.json")
        if reference_captures_only:
            try:
                primary_bytes = owned_bytes(root / "archive-retirement.json")
            except FileNotFoundError as error:
                raise ValueError("missing primary retirement completion") from error
            primary = json.loads(primary_bytes)
            if (
                any(
                    primary.get(key) != value
                    for key, value in locator.items()
                    if key != "retirement_started_ns"
                )
                or primary.get("metadata_retained") is not True
                or not isinstance(primary.get("retirement_completed_ns"), int)
            ):
                raise ValueError("primary retirement belongs to another archive")
            locator.update(
                payload_policy="canonical-reference-binary-v1",
                primary_retirement_sha256=hashlib.sha256(primary_bytes).hexdigest(),
            )
        resume = locator_path.exists()
        if resume:
            prior = json.loads(locator_path.read_bytes())
            if {k: v for k, v in prior.items() if k != "retirement_started_ns"} != {
                k: v for k, v in locator.items() if k != "retirement_started_ns"
            }:
                raise ValueError("existing archive locator does not match")
            locator = prior
            if retirement_path.exists():
                if reference_captures_only or scope in {"diagnostic", "abandoned-run"}:
                    done = sealed(retirement_path)
                    if any(done.get(key) != value for key, value in locator.items()):
                        raise ValueError("restricted retirement receipt changed")
                    return done
                return json.loads(retirement_path.read_bytes())
        else:
            write_new(locator_path, locator)
        with gzip.open(manifest, "rt") as records:
            for line in records:
                member = json.loads(line)
                select = reference_payload_path if reference_captures_only else payload_path
                if scope == "diagnostic":
                    from archive_completed_diagnostic import diagnostic_payload_path

                    select = diagnostic_payload_path
                elif scope == "abandoned-run":
                    from archive_abandoned_run import abandoned_payload_path

                    select = abandoned_payload_path
                relative = select(member["path"], root.name, case=scope == "case")
                if relative is None or member["type"] != "0":
                    continue
                path = root / relative
                # Never traverse a replaced ancestor, including an internal symlink.
                parent = path.parent
                while parent != root:
                    if parent.is_symlink() or not parent.is_dir():
                        raise ValueError("source parent changed")
                    parent = parent.parent
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                except FileNotFoundError:
                    if not resume:
                        raise
                    previously_absent += 1
                    continue
                with os.fdopen(fd, "rb") as file:
                    before = os.fstat(file.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_size != member["size"]:
                        raise ValueError("source payload metadata changed")
                    if hashlib.file_digest(file, "sha256").hexdigest() != member["sha256"]:
                        raise ValueError(
                            "source payload differs from archive; keeping changed file"
                        )
                    after = os.fstat(file.fileno())
                current = path.lstat()
                if signature(before) != signature(after) or signature(after) != signature(current):
                    raise ValueError("source payload changed during verification")
                path.unlink()
                count += 1
                removed_bytes += before.st_size
                if count % 10000 == 0:
                    print(
                        json.dumps({"removed_files": count, "removed_bytes": removed_bytes}),
                        flush=True,
                    )
        if identity_of(root) != source["identity"]:
            raise ValueError("source completion changed during retirement")
        result = {
            **locator,
            "removed_files": count,
            "removed_bytes": removed_bytes,
            "previously_absent_files": previously_absent,
            "retirement_completed_ns": time.time_ns(),
            "metadata_retained": True,
        }
        if reference_captures_only or scope in {"diagnostic", "abandoned-run"}:
            result["sha256"] = hashlib.sha256(
                json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
        write_new(retirement_path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "report", "manifest", "completion"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument(
        "--reference-captures-only",
        action="store_true",
        help="Retire canonical CPU reference blobs after primary retirement; keep metadata",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            retire(
                args.root,
                args.report,
                args.manifest,
                args.completion,
                args.repository,
                reference_captures_only=args.reference_captures_only,
            )
        )
    )


if __name__ == "__main__":
    main()
