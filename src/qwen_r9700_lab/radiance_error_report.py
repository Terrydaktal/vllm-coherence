"""Bounded request, container and host failure evidence; runnable over SSH using stdlib."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = "urn:qwen-r9700:backend-error:v1"
CONTAINER = "vllm-coherence"
MAX_BYTES = 2 * 1024 * 1024
MAX_TRACE = 32768
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")
PREFIX = re.compile(r"\[([^\]\s]+\.py:\d+)\] ?")
FATAL = ("EngineCore encountered a fatal error.", "EngineCore failed to start.")
CACHE_ROOT = Path.home() / ".cache/qwen-radiance-public-clean-snapshot-v1"


def clean(text: str) -> str:
    text = ANSI.sub("", text)
    return "".join(
        c for c in text if c in "\n\t" or (ord(c) >= 32 and not 127 <= ord(c) <= 159)
    )


def parse_journal(raw: bytes) -> list[dict]:
    """Keep logger tracebacks, including request failures, without adjacent input dumps."""
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
            if not isinstance(message, str) or not re.fullmatch(
                r"[a-f0-9]{12,64}", container
            ):
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
                role = re.match(r"\((APIServer[^)]*|EngineCore[^)]*)\) ?", part)
                body = part[role.end() :] if role else part
                key = entry_key or (
                    container,
                    role.group(1) if role else "stderr",
                    "plain",
                )
            if key and (
                body.strip() in FATAL
                or (body.startswith("Traceback (") and key not in active)
            ):
                incident = {
                    "timestamp": timestamp,
                    "container_id": container,
                    "lines": [],
                }
                if body.startswith("Traceback ("):
                    incident["lines"].append(body)
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
        lines = [
            line
            for line in lines[beginning:]
            if not line.strip()
            or line.startswith(
                (
                    "Traceback (",
                    " ",
                    "\t",
                    "During handling of the above exception",
                    "The above exception",
                )
            )
            or re.match(
                r"^[\w.]*(?:Error|Exception|Interrupt|Exit|ExceptionGroup)(?::|$)", line
            )
        ]
        exceptions = [
            line
            for line in lines
            if re.match(
                r"^[\w.]*(?:Error|Exception|Interrupt|Exit|ExceptionGroup)(?::|$)", line
            )
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


def bounded_output(
    command, *, maximum=MAX_BYTES, seconds=4
) -> tuple[bytes, str | None]:
    """Bound both a diagnostic child and its output, including unavailable journals."""
    chunks, size, reason = [], 0, None
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
    except OSError:
        return b"", "journal_unavailable"
    with process:
        deadline = time.monotonic() + seconds
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "journal_timeout"
                    break
                if not selector.select(min(remaining, 0.2)):
                    continue
                chunk = os.read(process.stdout.fileno(), min(65536, maximum - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size == maximum:
                    reason = "journal_size_limit"
                    break
        if reason:
            process.kill()  # Only the bounded reader, never the serving backend.
        try:
            code = process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code, reason = -9, "journal_timeout"
        if code not in (0, 1) and reason is None:
            reason = "journal_unavailable"
    return b"".join(chunks), reason


def bounded_journal(since: int) -> tuple[bytes, str | None]:
    command = [
        "journalctl",
        "--user",
        f"CONTAINER_NAME={CONTAINER}",
        "--no-pager",
        "--output=json",
        "--output-fields=MESSAGE,CONTAINER_ID_FULL,CONTAINER_ID,__REALTIME_TIMESTAMP",
        "--reverse",
        "--since",
        f"@{since // 1000}",
        "-n",
        "4096",
    ]
    raw, reason = bounded_output(command)
    # Read newest records first so an older huge input dump cannot hide a newer
    # failure behind the byte budget, then reconstruct each traceback in order.
    return b"\n".join(reversed(raw.splitlines())), reason


def parse_kernel_journal(raw: bytes) -> list[dict]:
    """Emit recognized numeric failure fields, never free-form kernel/log text."""
    events, machine = [], None
    rows = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            rows.append((int(row["__REALTIME_TIMESTAMP"]), row))
        except (ValueError, KeyError, TypeError):
            continue
    for microseconds, row in sorted(rows, key=lambda item: item[0]):
        try:
            stamp = microseconds // 1000
            text = row["MESSAGE"]
            boot = str(row.get("_BOOT_ID", "")).replace("-", "")
            if not isinstance(text, str):
                continue
        except (ValueError, KeyError, TypeError):
            continue
        if "[Hardware Error]: System Fatal error." in text:
            machine = {
                "kind": "cpu_machine_check",
                "timestamp": stamp,
                "fatal": True,
                "boot_id": boot,
            }
            events.append(machine)
        elif match := re.search(
            r"\[Hardware Error\]: CPU:(\d+).*MC(\d+)_STATUS.*: (0x[0-9a-fA-F]+)", text
        ):
            if (
                not machine
                or machine.get("boot_id") != boot
                or abs(stamp - machine["timestamp"]) > 1000
            ):
                machine = {
                    "kind": "cpu_machine_check",
                    "timestamp": stamp,
                    "fatal": False,
                    "boot_id": boot,
                }
                events.append(machine)
            machine.update(cpu=int(match[1]), bank=int(match[2]), status=match[3])
        elif match := re.search(
            r"\[Hardware Error\]: (Execution Unit) Ext. Error Code: (\d+)", text
        ):
            if (
                machine
                and machine.get("boot_id") == boot
                and abs(stamp - machine["timestamp"]) <= 1000
            ):
                machine.update(unit="execution", extended_code=int(match[2]))
        elif re.search(r"(?:Out of memory|oom-kill:).*|Killed process \d+", text):
            events.append({"kind": "host_oom", "timestamp": stamp, "boot_id": boot})
        elif re.search(
            r"amdgpu.*(?:GPU reset begin|GPU reset succeeded|ring .*timeout|GPU fault)",
            text,
            re.IGNORECASE,
        ):
            events.append({"kind": "gpu_fault", "timestamp": stamp, "boot_id": boot})
    return events[-32:]


def host_state(now: int) -> dict:
    state = {"kernel_events": []}
    try:
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        state.update(
            boot_id=Path("/proc/sys/kernel/random/boot_id")
            .read_text()
            .strip()
            .replace("-", ""),
            uptime_seconds=round(uptime, 2),
            boot_started_at=round(now - uptime * 1000),
        )
    except (OSError, ValueError, IndexError):
        pass
    raw, issue = bounded_output(
        [
            "journalctl",
            "-k",
            "--no-pager",
            "--output=json",
            "--output-fields=MESSAGE,__REALTIME_TIMESTAMP,_BOOT_ID",
            "--since",
            f"@{(now - 86400000) // 1000}",
            "--grep=Hardware Error|oom-kill:|Out of memory|Killed process|amdgpu.*(GPU reset|timeout|GPU fault)",
            "-n",
            "256",
        ],
        maximum=128 * 1024,
        seconds=1.5,
    )
    state.update(kernel_events=parse_kernel_journal(raw), kernel_lookup_issue=issue)
    filesystems = []
    for name, path in [("cache", CACHE_ROOT), ("system", Path("/"))]:
        try:
            fs = os.statvfs(path)
            filesystems.append(
                {
                    "name": name,
                    "available_bytes": fs.f_bavail * fs.f_frsize,
                    "total_bytes": fs.f_blocks * fs.f_frsize,
                    "available_inodes": fs.f_favail,
                }
            )
        except OSError:
            filesystems.append({"name": name, "unavailable": True})
    state["filesystems"] = filesystems
    return state


def backend_state() -> dict:
    try:
        result = subprocess.run(
            ["podman", "inspect", "--format", "{{json .State}}", CONTAINER],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        state = (
            json.loads(result.stdout)
            if result.returncode == 0
            else (
                {"Running": False, "Status": "absent"}
                if re.search(
                    r"no such (?:object|container)",
                    getattr(result, "stderr", ""),
                    re.IGNORECASE,
                )
                else {}
            )
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        state = {}
    ready = False
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8080/health", timeout=0.5
        ) as response:
            ready = response.status == 200
    except OSError:
        pass
    return {
        "running": state.get("Running"),
        "ready": ready,
        "started_at": state.get("StartedAt"),
        "oom_killed": state.get("OOMKilled", False),
        "status": state.get("Status"),
        "exit_code": state.get("ExitCode"),
        "finished_at": state.get("FinishedAt"),
    }


def diagnose(
    collected: dict, since: int, until: int, incident: dict | None, *, latest=False
) -> dict:
    """Separate incident-time evidence from current health; never guess a root cause."""
    host, backend = collected.get("host", {}), collected.get("backend", {})
    boot = host.get("boot_started_at")
    if isinstance(boot, (int, float)) and since - 2000 <= boot <= until + 2000:
        summary = (
            "AI host restarted in the diagnostic time window."
            if latest
            else "AI host restarted during this request."
        )
        fatal = next(
            (
                row
                for row in host.get("kernel_events", [])
                if row.get("fatal")
                and row.get("boot_id", host.get("boot_id")) == host.get("boot_id")
                and boot - 2000 <= row["timestamp"] <= boot + 120000
            ),
            None,
        )
        if fatal:
            summary += " Boot reported a fatal CPU error"
            if "cpu" in fatal:
                summary += f" on CPU {fatal['cpu']}"
            if fatal.get("unit") == "execution" and fatal.get("extended_code") == 0:
                summary += " (execution-unit watchdog timeout)"
            summary += ". The triggering operation is unconfirmed."
        return {
            "kind": "host_restarted",
            "summary": summary,
            "recovery": "Check /backend status; start it if stopped, then restart Pi to recreate its connection.",
        }
    exception = incident.get("summary", "") if incident else ""
    if re.search(r"\[Errno 28\]|No space left on device", exception, re.IGNORECASE):
        return {
            "kind": "disk_full",
            "summary": "Backend failed because the filesystem ran out of space.",
            "recovery": "Use qwen-radiance-cache to inspect storage, free space, then check /backend status and retry.",
        }
    if backend.get("oom_killed"):
        return {
            "kind": "host_oom",
            "summary": "The backend container was killed for exhausting system RAM.",
            "recovery": "Release system RAM, then use /backend start and retry.",
        }
    if re.search(r"out of memory|OutOfMemoryError", exception, re.IGNORECASE):
        return {
            "kind": "allocation_failed",
            "summary": f"Backend allocation failed: {exception}",
            "recovery": "Check GPU and system RAM usage; use /backend status before restarting or retrying.",
        }
    if incident:
        return {
            "kind": "backend_exception",
            "summary": f"Backend error: {exception}",
            "recovery": "Ctrl+O shows the recorded traceback and current backend status.",
        }
    if backend.get("running") is False:
        exit_code = backend.get("exit_code")
        return {
            "kind": "backend_stopped",
            "summary": "Backend container stopped"
            + (f" (exit {exit_code})." if exit_code is not None else ".")
            + " No matching Python traceback was retained.",
            "recovery": "Use /backend status, then /backend start and retry.",
        }
    if backend.get("ready"):
        if latest:
            if collected.get("lookup_issue"):
                return {
                    "kind": "history_unavailable",
                    "summary": "Backend is ready; error history could not be fully checked.",
                    "recovery": "Use /backend-error to retry the diagnostic lookup.",
                }
            return {
                "kind": "backend_ready",
                "summary": "Backend is ready. No recorded backend error was found.",
                "recovery": "",
            }
        return {
            "kind": "cause_unknown",
            "summary": "Backend is ready now; the cause of the interrupted request is unconfirmed.",
            "recovery": "Retry the request. If the local connection is refused, restart Pi to recreate its tunnel or relay.",
        }
    if backend.get("running"):
        return {
            "kind": "backend_not_ready",
            "summary": "Backend process is running but its health check failed.",
            "recovery": "Check /backend status; wait for startup or inspect the recorded failure before restarting.",
        }
    return {
        "kind": "cause_unknown",
        "summary": "Host answered the diagnostic lookup, but backend availability could not be confirmed.",
        "recovery": "Check /backend status. Ctrl+O shows the checks that succeeded and failed.",
    }


def collect(now: int, directory: Path | None = None) -> dict:
    # A shared parsed result serves simultaneous failures, with no periodic probe.
    container_key = hashlib.sha256(CONTAINER.encode()).hexdigest()[:16]
    directory = directory or Path(
        f"/dev/shm/qwen-radiance-backend-errors-{os.getuid()}-{container_key}"
    )
    directory.mkdir(mode=0o700, exist_ok=True)
    details = directory.lstat()
    if (
        not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.getuid()
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        raise ValueError("unsafe diagnostic cache directory")
    descriptor = os.open(
        directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        cache = directory / "report.json"
        if (
            cache.exists()
            and not cache.is_symlink()
            and cache.stat().st_size < 512 * 1024
        ):
            try:
                saved = json.loads(cache.read_text())
                if (
                    saved.get("collector_version") == 2
                    and 0 <= now - saved["captured_at"] <= 2000
                ):
                    return saved
            except (OSError, ValueError, KeyError, TypeError):
                pass
        with ThreadPoolExecutor(max_workers=3) as pool:
            journal = pool.submit(bounded_journal, now - 86400000)
            backend = pool.submit(backend_state)
            host = pool.submit(host_state, now)
            raw, reason = journal.result()
            backend, host = backend.result(), host.result()
        value = {
            "collector_version": 2,
            "captured_at": int(time.time() * 1000),
            "incidents": parse_journal(raw),
            "lookup_issue": reason,
            "backend": backend,
            "host": host,
        }
        fd, temporary_name = tempfile.mkstemp(
            prefix="report-", suffix=".tmp", dir=directory
        )
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


def report_for_window(
    collected: dict, since: int, until: int, *, latest: bool = False
) -> dict:
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
        "host": collected.get("host"),
        "diagnosis": diagnose(collected, since, until, incident, latest=latest),
    }


def main() -> None:
    global CONTAINER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("since", type=int)
    parser.add_argument("until", type=int)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--container", default=CONTAINER)
    options = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", options.container):
        raise ValueError("invalid backend container")
    CONTAINER = options.container
    now = int(time.time() * 1000)
    since, until = options.since, options.until
    if since < now - 86400000 or until > now + 10000 or since > until:
        raise ValueError("invalid diagnostic time window")
    report = report_for_window(collect(now), since, until, latest=options.latest)
    if options.metadata_only:
        incident = report.pop("incident")
        report["incident_metadata"] = (
            None
            if not incident
            else {
                key: incident[key]
                for key in (
                    "id",
                    "timestamp",
                    "container_id",
                    "exception_type",
                    "truncated",
                )
            }
        )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
