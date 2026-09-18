"""Stream and independently read back a completed run or immutable case attempt.

This operational tool does not change qualification sources, results, or GPU
state. Feed ``stream`` to restic's --stdin-from-command and feed ``restic dump``
to ``verify``. Both the producing command and verification must succeed before
any remote replica is eligible for removal. This tool never removes evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import gzip
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath


def write_new(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    parent = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def owned_bytes(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
            raise ValueError("evidence must be an owned regular file")
        raw = source.read()
        after = os.fstat(source.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("evidence changed while reading")
    return raw


def sealed(path):
    value = json.loads(owned_bytes(path))
    unsigned = {k: v for k, v in value.items() if k != "sha256"}
    expected = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if value.get("sha256") != expected:
        raise ValueError("evidence seal mismatch")
    return value


def completed_identity(root, *, paused=False):
    checkpoint = sealed(root / "checkpoint.json")
    campaign = sealed(root / "campaign.json")
    if paused:
        if checkpoint["status"] != "paused":
            raise ValueError("run is not paused at a case boundary")
    elif checkpoint["status"] not in {"complete", "stage_finished"} or checkpoint["current"]:
        raise ValueError("run has not finished")
    if checkpoint["campaign"] != campaign["sha256"]:
        raise ValueError("campaign mismatch")
    cases = {}
    for case in sorted(root.glob("case-*")):
        if not case.is_dir() or case.is_symlink():
            raise ValueError("unexpected case directory")
        if not (case / "input.json").is_file():
            raise ValueError("case input missing")
        result = sealed(case / "result.json")
        if result["campaign"] != campaign["sha256"]:
            raise ValueError("case belongs to another campaign")
        cases[case.name] = result["sha256"]
    if not cases:
        raise ValueError("empty campaign")
    return {"checkpoint": checkpoint["sha256"], "campaign": campaign["sha256"], "cases": cases}


def retained_reference(root, campaign, case):
    """Defer an origin until the live runner has evicted its disposable clone.

    Do not acquire the store lock: its writer deliberately uses nonblocking
    exclusion. Publication only names the currently executing, unfinished case;
    a finished non-origin can therefore never become a future origin. The
    immutable entry is atomically published. Any incomplete observation fails
    closed and can be reconsidered by the next archive inventory.
    """
    store = root.parent / "serial-reference-store"
    if not store.exists():
        return
    if sealed(store / "store.json").get("schema") != "urn:qwen:serial-reference-store:v1":
        raise ValueError("unrecognized serial reference store")
    if (store / "pending").exists():
        raise ValueError("serial reference publication is in progress")
    current = store / "current"
    if not current.exists():
        return
    entry = sealed(current / "entry.json")
    if (
        entry.get("schema") != "urn:qwen:serial-reference-store:v1/entry"
        or entry.get("campaign") != campaign["sha256"]
        or entry.get("case") not in campaign["cases"]
    ):
        raise ValueError("serial reference producer identity is not established")
    # Compare logical case identity, not host/container-specific absolute paths.
    # This conservatively also retains any earlier attempt of that same case.
    if entry["case"] == case:
        raise ValueError("case retains the active serial reference original")


def finished_case_identity(root):
    """Admit an immutable attempt only after its owned workers have exited.

    The queue creates a new directory for each retry. It never writes a finished
    attempt again. The parent controller may continue in a different directory;
    its changing current checkpoint is checked, but is not part of this identity.
    Incomplete/interrupted attempts deliberately require separate review.
    """
    match = re.fullmatch(r"case-(\d{5})(?:-attempt-\d{3,})?", root.name)
    if match is None:
        raise ValueError("invalid case attempt directory")
    parent = root.parent
    campaign = sealed(parent / "campaign.json")
    checkpoint = sealed(parent / "checkpoint.json")
    finished = sealed(parent / f"checkpoint-{root.name}.json")
    raw_input = owned_bytes(root / "input.json")
    payload = json.loads(raw_input)
    result = sealed(root / "result.json")
    worker = sealed(root / "worker-result.json")
    raw_shutdown = owned_bytes(root / "process/shutdown.json")
    shutdown = json.loads(raw_shutdown)
    index = int(match[1])
    if (
        checkpoint["campaign"] != campaign["sha256"]
        or finished["campaign"] != campaign["sha256"]
        or payload["campaign"] != campaign
        or result["campaign"] != campaign["sha256"]
        or index >= len(campaign["cases"])
        or payload["case"] != campaign["cases"][index]
        or result["case"] != payload["case"]
    ):
        raise ValueError("finished attempt does not match its campaign")
    if checkpoint.get("attempt") == root.name or (
        checkpoint.get("current") == result["case"]["id"] and not checkpoint.get("attempt")
    ):
        raise ValueError("controller has not advanced beyond this attempt")
    if (
        worker != result
        or result["status"] not in {"TESTED", "FAILED", "UNSUPPORTED"}
        or shutdown.get("cleanup_error", "missing") is not None
        or shutdown.get("returncode") != (0 if result["status"] == "TESTED" else 1)
    ):
        raise ValueError("owned worker completion or cleanup is not established")
    lease = root / "gpu-lease"
    release = sealed(lease / "released.json") if lease.exists() else None
    retained_reference(root, campaign, payload["case"])
    return {
        "scope": "finished-case-v1",
        "campaign": campaign["sha256"],
        "finished_checkpoint": finished["sha256"],
        "input_sha256": hashlib.sha256(raw_input).hexdigest(),
        "result": result["sha256"],
        "shutdown_sha256": hashlib.sha256(raw_shutdown).hexdigest(),
        "gpu_release": release["sha256"] if release is not None else None,
    }


@contextlib.contextmanager
def source_guard(root, *, case=False, paused=False, diagnostic=False, abandoned=False):
    # All archival/retirement operations for a run share this lock. It is
    # independent of the live queue lock, which is still mandatory for full runs.
    if sum((bool(case), bool(paused), bool(diagnostic), bool(abandoned))) > 1:
        raise ValueError("archive admission modes are mutually exclusive")
    parent = root.parent if case else root
    descriptors = []
    try:
        fd = os.open(
            parent / ".evidence-archive.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        descriptors.append(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not case and not diagnostic:
            fd = os.open(root / "queue.lock", os.O_RDWR | os.O_NOFOLLOW)
            descriptors.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if abandoned:
            from archive_abandoned_run import abandoned_identity

            yield abandoned_identity
        elif diagnostic:
            from archive_completed_diagnostic import finished_diagnostic_identity

            yield finished_diagnostic_identity
        else:
            yield (
                finished_case_identity
                if case
                else functools.partial(completed_identity, paused=paused)
            )
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def stream(root, receipt, output, *, case=False, paused=False, diagnostic=False, abandoned=False):
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or root.name in {"", ".", ".."}:
        raise ValueError("invalid source directory")
    root = root.resolve(strict=True)
    receipt = Path(receipt).absolute()
    if receipt.is_relative_to(root):
        raise ValueError("receipt must be outside the archived run")
    if receipt.exists():
        raise ValueError("receipt already exists")
    with source_guard(
        root, case=case, paused=paused, diagnostic=diagnostic, abandoned=abandoned
    ) as identity_of:
        stream_admitted_tree(
            root,
            receipt,
            output,
            identity_of,
            scope=(
                "abandoned-run"
                if abandoned
                else (
                    "diagnostic"
                    if diagnostic
                    else ("case" if case else ("paused-run" if paused else "run"))
                )
            ),
        )


def stream_admitted_tree(root, receipt, output, identity_of, *, scope):
    """Shared tar transport; caller holds its source-specific admission lock."""
    identity = identity_of(root)
    digest = hashlib.sha256()
    count = 0
    command = [
        "tar",
        "--sort=name",
        "--format=pax",
        "--pax-option=delete=atime,delete=ctime",
        "--one-file-system",
        "--create",
        "--file=-",
        "--directory",
        str(root.parent),
        "--",
        root.name,
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as child:
        assert child.stdout is not None
        try:
            while data := child.stdout.read(1024 * 1024):
                digest.update(data)
                count += len(data)
                output.write(data)
        except BaseException:
            child.stdout.close()
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        if child.wait() != 0:
            raise ValueError("tar failed; archive must not be published")
    output.flush()
    if identity_of(root) != identity:
        raise ValueError("source completion changed during archive")
    write_new(
        receipt,
        {
            "schema": "urn:qwen:conformance-archive-stream:v1",
            "source": str(root),
            "root_name": root.name,
            "scope": scope,
            "identity": identity,
            "tar_sha256": digest.hexdigest(),
            "tar_bytes": count,
            "completed_ns": time.time_ns(),
            "tar_command": command,
        },
    )


class HashingReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        value = self.source.read(size)
        self.digest.update(value)
        self.count += len(value)
        return value


def verify(source, receipt, manifest, report):
    expected = json.loads(Path(receipt).read_bytes())
    if expected.get("schema") != "urn:qwen:conformance-archive-stream:v1":
        raise ValueError("unknown receipt")
    reader = HashingReader(source)
    count = total = 0
    seen = set()
    # A failed verification leaves its partial manifest, but never a success report.
    fd = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as records:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or name.parts[0] != expected["root_name"]
                ):
                    raise ValueError("archive member escapes source root")
                if member.name in seen:
                    raise ValueError("duplicate archive member")
                seen.add(member.name)
                entry = {
                    "path": member.name,
                    "mode": member.mode,
                    "size": member.size,
                    "mtime": member.mtime,
                    "type": member.type.decode("ascii"),
                }
                if member.isfile():
                    digest = hashlib.sha256()
                    actual_size = 0
                    file = archive.extractfile(member)
                    if file is None:
                        raise ValueError("regular member has no payload")
                    while data := file.read(1024 * 1024):
                        digest.update(data)
                        actual_size += len(data)
                    if actual_size != member.size:
                        raise ValueError("truncated archive member")
                    entry["sha256"] = digest.hexdigest()
                    total += actual_size
                elif member.issym() or member.islnk():
                    entry["linkname"] = member.linkname
                elif not member.isdir():
                    raise ValueError("unsupported special file in evidence archive")
                records.write((json.dumps(entry, sort_keys=True) + "\n").encode())
                count += 1
        while reader.read(1024 * 1024):
            pass
    if reader.count != expected["tar_bytes"] or reader.digest.hexdigest() != expected["tar_sha256"]:
        raise ValueError("restored archive does not match source stream")
    manifest_path = Path(manifest)
    with manifest_path.open("rb") as file:
        manifest_sha = hashlib.file_digest(file, "sha256").hexdigest()
        os.fsync(file.fileno())
    write_new(
        report,
        {
            "schema": "urn:qwen:conformance-archive-readback:v1",
            "verified": True,
            "stream_receipt": expected,
            "members": count,
            "payload_bytes": total,
            "manifest": str(manifest_path.absolute()),
            "manifest_sha256": manifest_sha,
            "verified_ns": time.time_ns(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    send = sub.add_parser("stream")
    send.add_argument("--root", required=True, type=Path)
    send.add_argument("--receipt", required=True, type=Path)
    modes = send.add_mutually_exclusive_group()
    modes.add_argument("--case", action="store_true", help="Archive one finished immutable attempt")
    modes.add_argument(
        "--paused-run", action="store_true", help="Archive an inactive queue paused between cases"
    )
    modes.add_argument(
        "--diagnostic", action="store_true", help="Archive a reviewed completed diagnostic"
    )
    modes.add_argument(
        "--abandoned-run",
        action="store_true",
        help="Archive a reviewed inactive incomplete queue while retaining its original status",
    )
    check = sub.add_parser("verify")
    check.add_argument("--receipt", required=True, type=Path)
    check.add_argument("--manifest", required=True, type=Path)
    check.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "stream":
        stream(
            args.root,
            args.receipt,
            sys.stdout.buffer,
            case=args.case,
            paused=args.paused_run,
            diagnostic=args.diagnostic,
            abandoned=args.abandoned_run,
        )
    else:
        verify(sys.stdin.buffer, args.receipt, args.manifest, args.report)


if __name__ == "__main__":
    main()
