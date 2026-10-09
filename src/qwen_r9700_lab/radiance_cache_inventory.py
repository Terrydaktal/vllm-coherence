"""Private, bounded reuse of completed dashboard inventory metadata.

This stores the existing inspection report, never transcripts, token arrays or
tensor payloads. Cached inventory is historical; live residency still comes from
the separately validated telemetry feed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import stat
import time
from pathlib import Path

SCHEMA = "urn:coherence:cache-inventory:v1"
MAX_BYTES = 16 * 1024 * 1024
MAX_AGE_SECONDS = 10 * 60
ID = re.compile(r"[0-9a-f]{64}\Z")


def _scope(args):
    roots = getattr(args, "sessions_root", ())
    if not isinstance(roots, (list, tuple)):
        raise ValueError("invalid inventory session roots")  # noqa: TRY004 -- invalid cache configuration
    scope = {
        "host": str(getattr(args, "host", "ai")),
        "cache_root": str(getattr(args, "cache_root", "")),
        "abi": getattr(args, "abi", None),
        "verify": bool(getattr(args, "verify", False)),
        "stale_after": float(getattr(args, "stale_after", 300)),
        "sessions_root": [str(Path(root).expanduser().absolute()) for root in roots],
        "no_sessions": bool(getattr(args, "no_sessions", False)),
    }
    if not math.isfinite(scope["stale_after"]) or scope["stale_after"] < 0:
        raise ValueError("invalid inventory stale interval")
    return scope


def _filename(scope):
    encoded = json.dumps(scope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return f"inventory-{hashlib.sha256(encoded).hexdigest()}.json"


@contextlib.contextmanager
def _directory(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("inventory directory must be absolute")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.getuid() or info.st_mode & 0o7777 != 0o700:
            raise ValueError("unsafe inventory directory")
        yield descriptor
    finally:
        os.close(descriptor)


def _count(value):
    return type(value) is int and value >= 0


def _counts(value, *names):
    return isinstance(value, dict) and all(_count(value.get(name)) for name in names)


def _issues(value):
    return isinstance(value, list) and all(
        isinstance(row, dict) and all(isinstance(row.get(key), str) for key in ("severity", "code", "message"))
        for row in value
    )


def _processes(value):
    return isinstance(value, list) and all(
        isinstance(row, dict) and _counts(row, "pid", "port") for row in value
    )


def _summary(value):
    return isinstance(value, dict) and (
        value.get("generation") is None or ID.fullmatch(str(value["generation"]))
    ) and all(
        value.get(key) is None or isinstance(value[key], str)
        for key in ("title", "cwd", "session_id")
    ) and all(
        value.get(key) is None or _count(value[key]) for key in ("tokens", "last_turn_tokens")
    )


def _valid_report(report):
    if (
        not isinstance(report, dict)
        or not isinstance(report.get("chats"), list)
        or not isinstance(report.get("unsnapshotted_chats"), list)
        or not _issues(report.get("issues"))
        or not _counts(report.get("totals"), "file_bytes")
        or not _counts(report.get("io"), "written_file_bytes")
        or not _counts(report.get("filesystem"), "available_bytes")
    ):
        return False
    traffic = report["io"].get("lifetime", report["io"])
    if (
        not _counts(traffic, "written_file_bytes")
        or not _count(traffic.get("deleted_written_file_bytes", 0))
    ):
        return False
    for row in report["chats"]:
        if (
            not isinstance(row, dict)
            or not ID.fullmatch(str(row.get("id", "")))
            or not ID.fullmatch(str(row.get("abi", "")))
            or not _summary(row.get("metadata"))
            or not _counts(row.get("totals"), "file_bytes")
            or type(row.get("consistent")) is not bool
            or "expected_blocks" not in row
            or not _issues(row.get("issues"))
            or not _processes(row.get("active_processes", []))
            or (row.get("session") is not None and not _summary(row["session"]))
            or not isinstance(row.get("io", {}), dict)
            or (row.get("io", {}).get("written_file_bytes") is not None
                and not _count(row["io"]["written_file_bytes"]))
        ):
            return False
    for row in report["unsnapshotted_chats"]:
        if (
            not isinstance(row, dict)
            or not ID.fullmatch(str(row.get("id", "")))
            or not ID.fullmatch(str(row.get("generation", "")))
            or not isinstance(row.get("title"), str)
            or not _summary(row)
            or not _processes(row.get("active_processes", []))
        ):
            return False
    return True


def _invalid_constant(_value):
    raise ValueError("nonfinite inventory value")


def load(args, directory):
    """Return a safe matching completed report and its age, or a cache miss."""
    try:
        scope = _scope(args)
        with _directory(directory) as directory_fd:
            descriptor = os.open(
                _filename(scope), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
            )
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or info.st_mode & 0o7777 != 0o600
                    or info.st_size > MAX_BYTES
                ):
                    return None, None
                data = stream.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            return None, None
        cached = json.loads(data, parse_constant=_invalid_constant)
        if (
            not isinstance(cached, dict)
            or set(cached) != {"schema", "scope", "saved_at", "report"}
            or cached["schema"] != SCHEMA or cached["scope"] != scope
            or _filename(cached["scope"]) != _filename(scope)
            or type(cached["saved_at"]) not in (int, float)
            or not math.isfinite(cached["saved_at"])
            or not _valid_report(cached["report"])
        ):
            return None, None
        age = time.time() - cached["saved_at"]
        if not -1 <= age <= MAX_AGE_SECONDS:
            return None, None
        return cached["report"], max(0.0, age)
    except (OSError, ValueError, TypeError, RecursionError, OverflowError):
        return None, None


def save(args, directory, report):
    """Atomically replace one scoped report; errors leave the old cache intact."""
    if not _valid_report(report):
        raise ValueError("invalid completed inventory report")
    scope = _scope(args)
    try:
        data = json.dumps(
            {"schema": SCHEMA, "scope": scope, "saved_at": time.time(), "report": report},
            ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode()
    except (TypeError, RecursionError, OverflowError) as error:
        raise ValueError("inventory report is not JSON metadata") from error
    if len(data) > MAX_BYTES:
        raise ValueError("inventory report exceeds its size limit")
    filename = _filename(scope)
    temporary = f".{filename}.{secrets.token_hex(8)}"
    with _directory(directory) as directory_fd:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600, dir_fd=directory_fd,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(data)
            # The telemetry directory is tmpfs; publication needs no fsync.
            os.replace(temporary, filename, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
