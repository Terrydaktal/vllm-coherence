"""Quarantine, archive and retire explicitly reviewed historical benchmark caches.

This is storage maintenance, never conformance evidence. Original paths must
match the reviewed inode and the narrow benchmark namespace. No GPU is queried.
Deletion is limited to quarantined regular files after full archive readback.
"""

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import json
import os
import re
import stat
import sys
import time
from pathlib import Path, PurePosixPath

from archive_conformance_run import owned_bytes, sealed, stream_admitted_tree, write_new

SCHEMA = "urn:qwen:inactive-benchmark-cache:v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def publish(path, value):
    write_new(path, {**value, "sha256": digest(value)})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(value):
    return tuple(
        getattr(value, key)
        for key in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


def directory(path, *, private=False):
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(), "directory is not owned")
    if private:
        require(info.st_mode & 0o077 == 0, "quarantine directory must be private")
    return info


def inventory(root):
    device = directory(root).st_dev
    records = {}
    for path in [root, *sorted(root.rglob("*"))]:
        info = path.lstat()
        require(info.st_dev == device and info.st_uid == os.getuid(), "foreign cache entry")
        require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), "unsupported cache entry")
        require(not stat.S_ISREG(info.st_mode) or info.st_nlink == 1, "cache file has other links")
        records[str(path.relative_to(root))] = signature(info)
    return records


def live_references(name, proc=Path("/proc")):
    """Inspect owned processes; unknown unreadable processes prevent quarantine."""
    found = []
    exceptions = []
    for process in proc.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            used = name.encode() in (process / "cmdline").read_bytes()
            try:
                for location in ("cwd", "root"):
                    with contextlib.suppress(FileNotFoundError):
                        used |= name in str((process / location).readlink())
                for fd in (process / "fd").iterdir():
                    with contextlib.suppress(FileNotFoundError):
                        used |= name in str(fd.readlink())
                used |= name in (process / "maps").read_text()
            except PermissionError:
                comm = (process / "comm").read_text().strip()
                require(
                    comm in {"systemd", "(sd-pam)", "sshd", "sshd-session"},
                    "unreadable owned process requires review",
                )
                exceptions.append({"pid": int(process.name), "comm": comm})
            if used:
                found.append(int(process.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return {"references": found, "service_metadata_exceptions": exceptions}


def freeze(root, entry, review):
    root, entry = Path(root).absolute(), Path(entry).absolute()
    require(root.resolve(strict=True) == root, "cache path contains a symlink")
    require(re.fullmatch(r"qwen-runtime-ab-[a-z0-9]+\d{8}", root.name), "not a benchmark cache")
    require(root.parent.name == "benchmarks", "unexpected benchmark parent")
    require(not entry.is_relative_to(root), "quarantine must be outside the cache")
    document = sealed(review)
    matches = [row for row in document["roots"] if row["root"] == str(root)]
    require(len(matches) == 1 and not document["known_live_references"], "cache was not reviewed")
    info = directory(root)
    require(
        matches[0]["root_signature"]
        == {
            "inode": info.st_ino,
            "device": info.st_dev,
            "owner": info.st_uid,
        },
        "reviewed cache directory changed",
    )
    activity = live_references(root.name)
    require(not activity["references"], "benchmark cache is still in use")
    before = inventory(root)
    require(any(stat.S_ISREG(row[2]) for row in before.values()), "empty benchmark cache")
    require(
        max(row[6] for row in before.values()) < time.time_ns() - 3600 * 10**9,
        "benchmark cache was recently modified",
    )
    entry.mkdir(mode=0o700)
    (entry / "payload").mkdir(mode=0o700)
    target = entry / "payload" / root.name
    publish(
        entry / "freeze-intent.json",
        {
            "schema": SCHEMA + "/freeze-intent",
            "original": str(root),
            "target": str(target),
            "review": document["sha256"],
            "activity": activity,
            "before": digest(before),
        },
    )
    root.rename(target)
    for parent in (root.parent, target.parent):
        fd = os.open(parent, os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    after = inventory(target)
    require(
        {k: v for k, v in before.items() if k != "."}
        == {k: v for k, v in after.items() if k != "."},
        "cache changed during quarantine",
    )
    frozen = {
        "schema": SCHEMA + "/frozen",
        "original": str(root),
        "target": str(target),
        "review": document["sha256"],
        "tree_identity": digest(after),
        "root_inode": info.st_ino,
        "root_device": info.st_dev,
        "regular_files": sum(stat.S_ISREG(row[2]) for row in after.values()),
        "logical_bytes": sum(row[5] for row in after.values() if stat.S_ISREG(row[2])),
        "frozen_ns": time.time_ns(),
    }
    publish(entry / "frozen.json", frozen)
    return sealed(entry / "frozen.json")


@contextlib.contextmanager
def guard(entry):
    entry = Path(entry).absolute()
    directory(entry, private=True)
    fd = os.open(entry / "archive.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = sealed(entry / "frozen.json")
        require(frozen["schema"] == SCHEMA + "/frozen", "unknown frozen cache")
        target = Path(frozen["target"])
        require(
            target == entry / "payload" / Path(frozen["original"]).name, "quarantine target changed"
        )
        directory(target.parent, private=True)
        info = directory(target)
        require(
            (info.st_ino, info.st_dev) == (frozen["root_inode"], frozen["root_device"]),
            "quarantine directory replaced",
        )
        yield target, frozen
    finally:
        os.close(fd)


def stream(entry, receipt, output):
    entry, receipt = Path(entry).absolute(), Path(receipt).absolute()
    with guard(entry) as (target, frozen):
        require(
            not receipt.is_relative_to(target) and not receipt.exists(), "invalid stream receipt"
        )
        require(not (entry / "retirement-intent.json").exists(), "cache retirement already started")

        def identity(root):
            require(digest(inventory(root)) == frozen["tree_identity"], "frozen cache changed")
            return {"scope": SCHEMA, "frozen": frozen["sha256"]}

        stream_admitted_tree(target, receipt, output, identity, scope="inactive-benchmark-cache")


def retire(entry, report, manifest, completion, repository):
    entry = Path(entry).absolute()
    raw = owned_bytes(report)
    verified, completed = json.loads(raw), json.loads(owned_bytes(completion))
    require(
        verified.get("schema") == "urn:qwen:conformance-archive-readback:v1"
        and verified.get("verified") is True
        and completed.get("readback_report_sha256") == hashlib.sha256(raw).hexdigest()
        and re.fullmatch(r"[0-9a-f]{64}", completed.get("snapshot_id", "")),
        "missing verified archive completion",
    )
    require(
        hashlib.sha256(owned_bytes(manifest)).hexdigest() == verified["manifest_sha256"],
        "archive member manifest changed",
    )
    with guard(entry) as (target, frozen):
        source = verified["stream_receipt"]
        require(
            source["source"] == str(target)
            and source["root_name"] == target.name
            and source["scope"] == "inactive-benchmark-cache"
            and source["identity"] == {"scope": SCHEMA, "frozen": frozen["sha256"]},
            "archive belongs to another frozen cache",
        )
        intent = {
            "schema": SCHEMA + "/retirement",
            "frozen": frozen["sha256"],
            "snapshot_id": completed["snapshot_id"],
            "repository": repository,
            "archive_path": f"/{target.name}.tar",
            "tar_sha256": source["tar_sha256"],
            "manifest_sha256": verified["manifest_sha256"],
        }
        prior = entry / "retirement-intent.json"
        resume = prior.exists()
        if resume:
            require(
                sealed(prior) == {**intent, "sha256": digest(intent)}, "retirement identity changed"
            )
        else:
            require(digest(inventory(target)) == frozen["tree_identity"], "frozen cache changed")
            publish(prior, intent)
        finished = entry / "retired.json"
        if finished.exists():
            done = sealed(finished)
            require(
                all(done.get(key) == value for key, value in intent.items()),
                "completed retirement identity changed",
            )
            require(
                not any(stat.S_ISREG(row[2]) for row in inventory(target).values()),
                "retired cache contains unexpected files",
            )
            return done
        count = total = absent = 0
        with gzip.open(manifest, "rt") as records:
            for row in map(json.loads, records):
                name = PurePosixPath(row["path"])
                require(
                    name.parts
                    and name.parts[0] == target.name
                    and not name.is_absolute()
                    and ".." not in name.parts,
                    "archive member escapes quarantine",
                )
                if row["type"] == "5":
                    continue
                require(row["type"] == "0" and len(name.parts) > 1, "unsupported archive member")
                path = target.joinpath(*name.parts[1:])
                parent = path.parent
                while parent != target:
                    directory(parent)
                    parent = parent.parent
                try:
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                except FileNotFoundError:
                    require(resume, "unretired payload disappeared")
                    absent += 1
                    continue
                with os.fdopen(descriptor, "rb") as data:
                    before = os.fstat(data.fileno())
                    require(
                        stat.S_ISREG(before.st_mode)
                        and before.st_uid == os.getuid()
                        and before.st_nlink == 1
                        and before.st_size == row["size"],
                        "source payload metadata changed",
                    )
                    require(
                        hashlib.file_digest(data, "sha256").hexdigest() == row["sha256"],
                        "source payload differs from archive",
                    )
                    after = os.fstat(data.fileno())
                require(
                    signature(before) == signature(after) == signature(path.lstat()),
                    "source payload changed during verification",
                )
                path.unlink()
                count += 1
                total += before.st_size
        require(
            not any(stat.S_ISREG(row[2]) for row in inventory(target).values()),
            "unarchived files remain in quarantine",
        )
        publish(
            finished,
            {
                **intent,
                "removed_files": count,
                "removed_bytes": total,
                "previously_absent_files": absent,
                "retired_ns": time.time_ns(),
            },
        )
        return sealed(finished)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    take = sub.add_parser("freeze")
    for key in ("root", "entry", "review"):
        take.add_argument("--" + key, type=Path, required=True)
    send = sub.add_parser("stream")
    for key in ("entry", "receipt"):
        send.add_argument("--" + key, type=Path, required=True)
    remove = sub.add_parser("retire")
    for key in ("entry", "report", "manifest", "completion"):
        remove.add_argument("--" + key, type=Path, required=True)
    remove.add_argument("--repository", required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        print(json.dumps(freeze(args.root, args.entry, args.review)))
    elif args.mode == "stream":
        stream(args.entry, args.receipt, sys.stdout.buffer)
    else:
        print(
            json.dumps(
                retire(args.entry, args.report, args.manifest, args.completion, args.repository)
            )
        )


if __name__ == "__main__":
    main()
