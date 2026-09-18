"""Bounded, on-demand EngineCore traceback lookup; runnable over SSH using stdlib."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

SCHEMA = "urn:qwen-r9700:backend-error:v1"
CONTAINER = "vllm-coherence"
MAX_BYTES = 2 * 1024 * 1024
MAX_TRACE = 32768
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")
PREFIX = re.compile(r"\[([^\]\s]+\.py:\d+)\] ?")
FATAL = ("EngineCore encountered a fatal error.", "EngineCore failed to start.")


def clean(text: str) -> str:
    text = ANSI.sub("", text)
    return "".join(c for c in text if c in "\n\t" or (ord(c) >= 32 and not 127 <= ord(c) <= 159))


def parse_journal(raw: bytes) -> list[dict]:
    """Keep only the fatal logger's traceback, never adjacent input/config dumps."""
    incidents = []
    active = {}
    for line in raw.splitlines():
        try:
            entry = json.loads(line)
            timestamp = int(entry["__REALTIME_TIMESTAMP"]) // 1000
            container = entry.get("CONTAINER_ID_FULL") or entry["CONTAINER_ID"]
            message = entry["MESSAGE"]
            if isinstance(message, list):
                message = bytes(message).decode("utf-8", errors="replace")
            if not isinstance(message, str) or not re.fullmatch(r"[a-f0-9]{12,64}", container):
                continue
        except (ValueError, KeyError, TypeError):
            continue
        entry_key = None
        for part in clean(message).splitlines():
            prefix = PREFIX.search(part)
            if prefix:
                role = re.search(r"\((EngineCore[^)]*)\)", part[: prefix.start()])
                key = (container, role.group(1) if role else "", prefix.group(1))
                body = part[prefix.end() :]
                entry_key = key
            else:
                key, body = entry_key, part
            if key and body.strip() in FATAL:
                incident = {"timestamp": timestamp, "container_id": container, "lines": []}
                active[key] = incident
                incidents.append(incident)
            elif key in active:
                active[key]["lines"].append(body)

    results = []
    for incident in incidents[-8:]:
        lines = incident.pop("lines")
        beginning = next(
            (i for i, line in enumerate(lines) if line.startswith("Traceback (")), None
        )
        if beginning is None:
            continue
        lines = lines[beginning:]
        exceptions = [
            line
            for line in lines
            if re.match(r"^[\w.]*(?:Error|Exception|Interrupt|Exit|ExceptionGroup)(?::|$)", line)
        ]
        if not exceptions:
            continue
        exception = exceptions[-1]
        trace = "\n".join(lines).rstrip()
        truncated = len(trace) > MAX_TRACE
        if truncated:
            trace = (
                trace[: MAX_TRACE // 2]
                + "\n... traceback shortened ...\n"
                + trace[-MAX_TRACE // 2 :]
            )
        digest = hashlib.sha256(
            f"{incident['container_id']}:{incident['timestamp']}:{trace}".encode()
        ).hexdigest()
        results.append(
            {
                **incident,
                "id": digest,
                "traceback": trace,
                "summary": exception[:240],
                "exception_type": exception.split(":", 1)[0],
                "truncated": truncated,
            }
        )
    return results


def bounded_journal(since: int) -> tuple[bytes, str | None]:
    command = [
        "journalctl",
        "--user",
        f"CONTAINER_NAME={CONTAINER}",
        "--no-pager",
        "--output=json",
        "--output-fields=MESSAGE,CONTAINER_ID_FULL,CONTAINER_ID,__REALTIME_TIMESTAMP",
        "--grep=EngineCore",
        "--reverse",
        "--since",
        f"@{since // 1000}",
        "-n",
        "4096",
    ]
    chunks, size, reason = [], 0, None
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        deadline = time.monotonic() + 4
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "journal_timeout"
                    break
                if not selector.select(min(remaining, 0.2)):
                    continue
                chunk = os.read(process.stdout.fileno(), min(65536, MAX_BYTES - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size == MAX_BYTES:
                    reason = "journal_size_limit"
                    break
        if reason:
            process.kill()  # Stop only this bounded journal reader, never the backend.
        code = process.wait(timeout=1)
        if code not in (0, 1) and reason is None:
            reason = "journal_unavailable"
    # Read newest records first so an older huge input dump cannot hide a newer
    # failure behind the byte budget, then reconstruct each traceback in order.
    return b"\n".join(reversed(b"".join(chunks).splitlines())), reason


def backend_state() -> dict:
    try:
        result = subprocess.run(
            ["podman", "inspect", "--format", "{{json .State}}", CONTAINER],
            capture_output=True,
            text=True,
            timeout=2,
        )
        state = json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        state = {}
    ready = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=0.5) as response:
            ready = response.status == 200
    except OSError:
        pass
    return {
        "running": state.get("Running"),
        "ready": ready,
        "started_at": state.get("StartedAt"),
        "oom_killed": state.get("OOMKilled", False),
    }


def collect(now: int, directory: Path | None = None) -> dict:
    # A shared parsed result serves simultaneous failures, with no periodic probe.
    directory = directory or Path(f"/dev/shm/qwen-radiance-backend-errors-{os.getuid()}")
    directory.mkdir(mode=0o700, exist_ok=True)
    details = directory.lstat()
    if (
        not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.getuid()
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        raise ValueError("unsafe diagnostic cache directory")
    descriptor = os.open(directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        cache = directory / "report.json"
        if cache.exists() and not cache.is_symlink() and cache.stat().st_size < 512 * 1024:
            try:
                saved = json.loads(cache.read_text())
                if 0 <= now - saved["captured_at"] <= 2000:
                    return saved
            except (OSError, ValueError, KeyError, TypeError):
                pass
        raw, reason = bounded_journal(now - 86400000)
        value = {
            "captured_at": int(time.time() * 1000),
            "incidents": parse_journal(raw),
            "lookup_issue": reason,
            "backend": backend_state(),
        }
        fd, temporary_name = tempfile.mkstemp(prefix="report-", suffix=".tmp", dir=directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream)
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
        return value
    finally:
        os.close(descriptor)


def report_for_window(collected: dict, since: int, until: int, *, latest: bool = False) -> dict:
    matches = [
        row
        for row in collected["incidents"]
        if latest or since - 2000 <= row["timestamp"] <= until + 2000
    ]
    incident = max(matches, key=lambda row: row["timestamp"]) if matches else None
    return {
        "schema": SCHEMA,
        "status": "found" if incident else "unavailable",
        "incident": incident,
        "backend": collected["backend"],
        "captured_at": collected["captured_at"],
        "lookup_issue": collected["lookup_issue"],
        "since": since,
        "until": until,
        "latest": latest,
    }


def main() -> None:
    now = int(time.time() * 1000)
    since, until = int(sys.argv[1]), int(sys.argv[2])
    if since < now - 86400000 or until > now + 10000 or since > until:
        raise ValueError("invalid diagnostic time window")
    report = report_for_window(collect(now), since, until, latest="--latest" in sys.argv[3:])
    if "--metadata-only" in sys.argv[3:]:
        incident = report.pop("incident")
        report["incident_metadata"] = (
            None
            if not incident
            else {
                key: incident[key]
                for key in ("id", "timestamp", "container_id", "exception_type", "truncated")
            }
        )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
