"""Terminal views and local Pi-session correlation for Radiance cache inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

from . import radiance_cache_audit as audit
from . import radiance_memory as memory

DEFAULT_ROOT = "/home/lewis/.cache/qwen-radiance-public-clean-snapshot-v1"
DEFAULT_SESSIONS = Path.home() / ".local/state/qwen-r9700/pi-remote"
DEFAULT_PROJECTS = Path.home() / "tasks"
MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"
ACTIVITY_SCHEMA = "urn:qwen-r9700:radiance-active-pi:v1"
HISTORY_SCHEMA = "urn:qwen-r9700:pi-project-history:v1"
AGENT_DIRECTORY = re.compile(r"agent-([1-9][0-9]{0,4})")
HELP = """NAME
    qwen-radiance-cache - inspect chat snapshot coverage, RAM tails, disk use, and cleanup

SYNOPSIS
    qwen-radiance-cache [OPTIONS] [status]
    qwen-radiance-cache [OPTIONS] show CHAT
    qwen-radiance-cache [OPTIONS] audit [--verify]
    qwen-radiance-cache [OPTIONS] watch [--interval SECONDS] [--count N]
    qwen-radiance-cache [OPTIONS] list [--json]
    qwen-radiance-cache [OPTIONS] flush --identity-json JSON [--timeout SECONDS]
    qwen-radiance-cache [OPTIONS] memory [--refresh | --map] [--json]

DESCRIPTION
    Show all labelled disk snapshots and local Radiance Pi chats across Pi profiles,
    including chats with no snapshot. Show the PID and local port of each active
    Pi process. Inspect published-block completeness, context counters, compressed/
    raw/allocated sizes, buffered RAM tails, tensor groups, old generations,
    abandoned temporary files, unreferenced blocks, and duplicate-content candidates.

OPTIONS
    --host HOST           SSH host (default: ai); local reads this machine.
    --cache-root PATH     Cache root on the selected host.
    --abi SHA256          Inspect one ABI namespace instead of all namespaces.
    --chat CHAT           Filter by chat ID prefix, session ID, title, or directory.
    --sessions-root PATH  Pi remote root, profile root, or sessions directory; repeatable.
    --no-sessions         Inspect remote cache only, without local chat discovery.
    --stale-after SECONDS Flag old unreferenced blocks (default: 300 seconds).
    --verify              Stream/decompress payloads and check SHA-256 checksums.
    --json                Emit structured JSON; watch emits one object per line.
    --interval SECONDS    Watch refresh interval, 1-60 seconds (default: 5).
    --count N             Stop watch after N refreshes; zero means until Ctrl-C.
    --identity-json JSON  Exact chat/generation identity for a mutating hook.
    --timeout SECONDS     Wait up to this long for a live backend tail flush.
    --refresh             Refresh the memory buffer inventory; wait up to 5 seconds.
    --map                 Request an allocator map with free block sizes and owners.
                          Requests within 10 seconds share the latest map.
    -h, --help            Show this help.

OPERATION
    status is the default. show expands a chat's generations, tensor groups,
    missing blocks and diagnostic paths. audit highlights issues and returns a
    nonzero status for warnings/errors. watch repeats the same read-only scan.
    Fast scans check headers under brief non-blocking chat locks. Busy chats are
    labelled BUSY. Full verification runs outside these locks and reports races.
    Block coverage measures the published manifest, not an exact token restore
    percentage. Published tokens and the last recorded Pi turn are separate
    counters; the hybrid model may reuse fewer tokens than either counter.
    memory reads the backend's shared one-second allocation report, without
    scanning chats or snapshots. It separates live tensor allocations, allocator
    reservations, fragmentation, peaks and known buffer sizes. --refresh requests
    one metadata-only buffer inventory; multiple requests share the reporter.
    Unused reservations are not a promise of safely reclaimable VRAM.

EXAMPLES
    qwen-radiance-cache
    qwen-radiance-cache show drainer
    qwen-radiance-cache audit
    qwen-radiance-cache audit --verify --json
    qwen-radiance-cache watch --interval 5
    qwen-radiance-cache --abi SHA256 flush --identity-json '{...}'
    qwen-radiance-cache --host local --cache-root /path/to/cache status
    qwen-radiance-cache memory
    qwen-radiance-cache memory --refresh --json
    qwen-radiance-cache memory --map

FILES
    snapshots/<ABI>/data/qwen-chat-cache-v1/<CHAT>/chat.json
    snapshots/<ABI>/data/qwen-chat-cache-v1/<CHAT>/generations/<GEN>/*.qkv
    Local Pi sessions: <PROJECT>/.pi/sessions/*.jsonl
    Legacy Pi sessions: agent-<PORT>/sessions/<PROJECT>/*.jsonl
    Active Pi markers: agent-<PORT>/radiance-active/<PID>.json
    Tail residency: /dev/shm/qwen-radiance-snapshot-tail.json
    Flush control: /dev/shm/qwen-radiance-snapshot-control-v1/
    Memory report/control: /dev/shm/qwen-radiance-memory-v1/

PATHS
    Default remote root:
      /home/lewis/.cache/qwen-radiance-public-clean-snapshot-v1
    Default local Pi root:
      ~/.local/state/qwen-r9700/pi-remote
    Default project root:
      ~/tasks/*/.pi/sessions

SECURITY NOTES
    status/show/audit/watch never delete, repair, or create cache files. SSH sends
    the inspector source, not model payloads. Symlinks are skipped/reported. Payload
    verification requires zstd on the storage host. Duplicate candidates use
    recorded content hashes; --verify checks the actual decoded bytes. Allocated
    bytes are stat block counts, not unique physical Btrfs/reflink ownership.
    Activity markers contain only chat identity, PID, port, and process start time;
    the PID and start time must still match a same-user live Pi process.
    compact and flush with --identity-json are reserved for authenticated Pi
    hooks and exact operator requests; both require --abi. Flush talks only to
    the live backend through an owner-only tmpfs control directory.
    memory reads sizes and counters only. --refresh writes an owner-only tmpfs
    request; it never profiles GPU execution, clears caches, or reads tensor values.

EXIT STATUS
    0  Scan completed (status/show may display issues).
    1  audit found warnings or errors.
    2  Invalid arguments, inaccessible storage, unavailable memory report, or SSH failure.
    130 Interrupted watch.

AUTHORS
    Qwen R9700 inference lab maintainers.
"""


class Parser(argparse.ArgumentParser):
    def format_help(self):
        return HELP


def parser():
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--host")
    common.add_argument("--cache-root")
    common.add_argument("--abi")
    common.add_argument("--chat")
    common.add_argument("--sessions-root", action="append")
    common.add_argument("--no-sessions", action="store_true")
    common.add_argument("--stale-after", type=float)
    common.add_argument("--verify", action="store_true")
    common.add_argument("--json", action="store_true")
    result = Parser(parents=[common])
    sub = result.add_subparsers(dest="command")
    for name in ("status", "show", "audit", "watch", "list", "compact", "flush", "memory"):
        command = sub.add_parser(name, parents=[common])
        if name == "show":
            command.add_argument("selector")
        if name == "watch":
            command.add_argument("--interval", type=float, default=5)
            command.add_argument("--count", type=int, default=0)
        if name == "compact":
            command.add_argument("--identity-json", required=True)
        if name == "flush":
            command.add_argument("--identity-json", required=True)
            command.add_argument("--timeout", type=float, default=120.0)
        if name == "memory":
            action = command.add_mutually_exclusive_group()
            action.add_argument("--refresh", action="store_true")
            action.add_argument("--map", action="store_true")
    return result


def js_digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def session_identity_path(path):
    absolute = path.absolute()
    if path.parent.name != "sessions" or path.parent.parent.name != ".pi":
        return str(absolute)
    index_path = path.parent / ".identity.json"
    if not index_path.exists():
        return str(absolute)
    index_info = index_path.lstat()
    if (
        not stat.S_ISREG(index_info.st_mode)
        or index_info.st_uid != os.getuid()
        or index_info.st_nlink != 1
        or index_info.st_mode & 0o022
    ):
        raise ValueError(f"unsafe project history identity index: {index_path}")
    with audit.open_regular(index_path) as stream:
        index = json.load(stream)
    if index.get("schema") != HISTORY_SCHEMA or not isinstance(index.get("sessions"), dict):
        raise ValueError(f"invalid project history identity index: {index_path}")
    record = index["sessions"].get(path.name)
    if record is None:
        return str(absolute)
    info = path.stat()
    if (
        not isinstance(record, dict)
        or record.get("device") != info.st_dev
        or record.get("inode") != info.st_ino
        or not isinstance(record.get("identity_path"), str)
        or not record["identity_path"].startswith("/")
    ):
        raise ValueError(f"project history identity does not match session: {path}")
    return str(Path(record["identity_path"]).absolute())


def session_info(path):
    summary = {
        "session_file": str(path.absolute()),
        "transcript_bytes": path.stat().st_size,
        "last_turn_tokens": None,
        "last_turn_timestamp": None,
        "pending_messages": False,
    }
    header = None
    generation = "initial"
    is_radiance = False
    partial = False
    with audit.open_regular(path) as stream:
        for line in stream:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                partial = True
                continue
            if not isinstance(entry, dict):
                continue
            kind = entry.get("type")
            if kind == "session" and isinstance(entry.get("id"), str):
                header = entry
            elif kind == "model_change":
                is_radiance |= entry.get("modelId") == MODEL
            elif kind == "session_info" and entry.get("name"):
                summary["title"] = str(entry["name"])
            elif kind == "compaction" and isinstance(entry.get("id"), str):
                generation = entry["id"]
                # The previous turn's counter is not the compacted context size.
                summary["last_turn_tokens"] = None
                summary["pending_messages"] = True
            elif kind == "message" and isinstance(entry.get("message"), dict):
                message = entry["message"]
                is_radiance |= message.get("model") == MODEL
                if message.get("role") in ("user", "toolResult"):
                    summary["pending_messages"] = True
                if message.get("role") == "assistant" and message.get("stopReason") not in (
                    "error",
                    "aborted",
                ):
                    usage = message.get("usage")
                    if isinstance(usage, dict):
                        values = [
                            usage.get(key, 0)
                            for key in ("input", "output", "cacheRead", "cacheWrite")
                        ]
                        if all(isinstance(v, (int, float)) and v >= 0 for v in values):
                            summary["last_turn_tokens"] = int(sum(values))
                            summary["last_turn_timestamp"] = entry.get("timestamp")
                            summary["pending_messages"] = False
    if not header or not is_radiance:
        return None
    summary.update(
        id=js_digest([session_identity_path(path), header["id"]]),
        session_id=header["id"],
        generation=js_digest(generation),
        cwd=header.get("cwd", ""),
        partial=partial,
    )
    summary.setdefault("title", summary["cwd"] or path.stem)
    return summary


def discover_sessions(roots):
    sessions, problems, seen = [], [], set()
    scan_roots = []
    for root in roots:
        profile_sessions = sorted(root.glob("agent-*/sessions"))
        nested_profile_sessions = sorted(root.glob("*/agent-*/sessions"))
        project_sessions = sorted(root.glob("*/.pi/sessions"))
        if (
            profile_sessions
            or nested_profile_sessions
            or project_sessions
            or root == DEFAULT_SESSIONS
        ):
            scan_roots.extend([*project_sessions, *profile_sessions, *nested_profile_sessions])
        elif (root / "sessions").is_dir():
            scan_roots.append(root / "sessions")
        else:
            scan_roots.append(root)
    for root in scan_roots:
        if not root.exists():
            continue
        if root.resolve() != root:
            problems.append(
                audit.issue("SESSION_SYMLINK", "Session root symlink skipped", path=root)
            )
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            current = Path(directory)
            dirs[:] = [
                d
                for d in dirs
                if not (current / d).is_symlink() and len(current.relative_to(root).parts) < 4
            ]
            for name in files:
                path = current / name
                if path.suffix != ".jsonl" or path.is_symlink():
                    continue
                try:
                    file_info = path.stat()
                    identity = (file_info.st_dev, file_info.st_ino)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    info = session_info(path)
                    if info:
                        sessions.append(info)
                except (OSError, ValueError, TypeError) as error:
                    problems.append(audit.issue("SESSION_SCAN_FAILED", str(error), path=path))
    return sessions, problems


def activity_directories(roots):
    directories = set()
    for root in roots:
        if root.name == "sessions" and AGENT_DIRECTORY.fullmatch(root.parent.name):
            directories.add(root.parent / "radiance-active")
        elif AGENT_DIRECTORY.fullmatch(root.name):
            directories.add(root / "radiance-active")
        else:
            directories.update(path / "radiance-active" for path in root.glob("agent-*"))
            directories.update(path / "radiance-active" for path in root.glob("*/agent-*"))
    return sorted(directories)


def live_pi_process(pid, start_ticks, proc_root=Path("/proc")):
    process = proc_root / str(pid)
    try:
        if process.stat().st_uid != os.getuid():
            return False
        if (process / "comm").read_text().strip() != "pi":
            return False
        stat = (process / "stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        return len(fields) > 19 and fields[19] == start_ticks
    except OSError:
        return False


def discover_active_processes(roots, proc_root=Path("/proc")):
    processes, problems, seen = [], [], set()
    for directory in activity_directories(roots):
        if not directory.exists():
            continue
        if directory.resolve() != directory:
            problems.append(
                audit.issue(
                    "ACTIVITY_SYMLINK", "Pi activity directory symlink skipped", path=directory
                )
            )
            continue
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink():
                problems.append(
                    audit.issue("ACTIVITY_SYMLINK", "Pi activity marker symlink skipped", path=path)
                )
                continue
            try:
                if path.stat().st_size > 4096:
                    raise ValueError("Pi activity marker exceeds 4096 bytes")
                with audit.open_regular(path) as stream:
                    marker = json.load(stream)
                pid = marker.get("pid")
                port = marker.get("port")
                chat_id = marker.get("chat_id")
                start_ticks = marker.get("process_start_ticks")
                if (
                    marker.get("schema") != ACTIVITY_SCHEMA
                    or not isinstance(pid, int)
                    or pid < 1
                    or path.stem != str(pid)
                    or not isinstance(port, int)
                    or not 1 <= port <= 65535
                    or not isinstance(chat_id, str)
                    or not audit.ID.fullmatch(chat_id)
                    or not isinstance(start_ticks, str)
                    or not start_ticks.isdigit()
                ):
                    raise ValueError("invalid Pi activity marker")
                key = (chat_id, pid, port)
                if key not in seen and live_pi_process(pid, start_ticks, proc_root):
                    seen.add(key)
                    processes.append({"chat_id": chat_id, "pid": pid, "port": port})
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                problems.append(audit.issue("BAD_ACTIVITY", str(error), path=path))
    return sorted(processes, key=lambda process: (process["port"], process["pid"])), problems


def correlate(report, sessions, active_processes=()):
    active_by_id = {}
    for process in active_processes:
        active_by_id.setdefault(process["chat_id"], []).append(process)
    for session in sessions:
        session["active_processes"] = active_by_id.get(session["id"], [])
    by_id = {session["id"]: session for session in sessions}
    found = set()
    for row in report["chats"]:
        row["active_processes"] = active_by_id.get(row["id"], [])
        session = by_id.get(row["id"])
        row["session"] = session
        if session:
            found.add(session["id"])
            if row["metadata"].get("generation") != session["generation"]:
                row["issues"].append(
                    audit.issue(
                        "GENERATION_BEHIND",
                        "Local compaction generation has no matching current snapshot",
                    )
                )
            published = row["metadata"].get("tokens")
            last_turn = session["last_turn_tokens"]
            if isinstance(published, int) and isinstance(last_turn, int) and last_turn > published:
                row["issues"].append(
                    audit.issue(
                        "NEWER_TURN",
                        f"Last recorded turn has {last_turn:,} tokens; "
                        f"published context has {published:,}",
                        severity="info",
                    )
                )
            if session["pending_messages"]:
                row["issues"].append(
                    audit.issue(
                        "NEW_LOCAL_MESSAGES",
                        "Local messages follow the last recorded model turn",
                        severity="info",
                    )
                )
    report["unsnapshotted_chats"] = [s for s in sessions if s["id"] not in found]
    report["local_transcript_bytes"] = sum(s["transcript_bytes"] for s in sessions)


def collect(args):
    if args.host in ("local", "localhost", "127.0.0.1"):
        result = audit.scan(
            args.cache_root, abi=args.abi, stale_after=args.stale_after, verify=args.verify
        )
    else:
        remote = [
            "python3",
            "-",
            "--cache-root",
            args.cache_root,
            "--stale-after",
            str(args.stale_after),
        ]
        if args.abi:
            remote += ["--abi", args.abi]
        if args.verify:
            remote.append("--verify")
        completed = subprocess.run(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "--",
                args.host,
                shlex.join(remote),
            ],
            input=Path(audit.__file__).read_text(),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise ValueError(
                completed.stderr.strip() or completed.stdout.strip() or "SSH inspection failed"
            )
        result = json.loads(completed.stdout)
        if result.get("error"):
            raise ValueError(result["error"])
    roots = [Path(p).expanduser().absolute() for p in args.sessions_root]
    if args.no_sessions:
        sessions, problems, active_processes = [], [], []
    else:
        sessions, problems = discover_sessions(roots)
        active_processes, activity_problems = discover_active_processes(roots)
        problems.extend(activity_problems)
    correlate(result, sessions, active_processes)
    result["active_processes"] = active_processes
    result["issues"].extend(problems)
    result["host"] = args.host
    result["scope"] = {"abi": args.abi, "local_sessions": not args.no_sessions}
    return result


def human(value):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError


def clean(value):
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(value or ""))


def number(value):
    return f"{value:,}" if isinstance(value, (int, float)) else "?"


def active(processes):
    return ",".join(f"{process['pid']}/{process['port']}" for process in processes) or "-"


def handover_residency(report, chat_id):
    residency = report.get("scheduler", {}).get("worker", {}).get("residency")
    if not isinstance(residency, dict):
        return None
    images = [image for image in residency.get("images", []) if image["chat_id"] == chat_id]
    active_bank = residency.get("active")
    return {
        "gpu": isinstance(active_bank, dict) and active_bank.get("chat_id") == chat_id,
        "data_bytes": sum(image["data_bytes"] for image in images),
        "allocated_bytes": sum(image["allocated_bytes"] for image in images),
        "images": len(images),
        "generations": [image["generation"] for image in images],
    }


def handover_cell(report, chat_id):
    residency = handover_residency(report, chat_id)
    if residency is None:
        return "?"
    if residency["gpu"]:
        return f"{human(residency['data_bytes'])} + GPU" if residency["data_bytes"] else "GPU"
    return human(residency["data_bytes"])


def tail_residency(report, chat_id):
    status = report.get("tail_journal", {})
    if not status.get("available"):
        return None
    rows = [row for row in status.get("chats", []) if row["chat_id"] == chat_id]
    return {
        "bytes": sum(row["bytes"] for row in rows),
        "blocks": sum(row["blocks"] for row in rows),
        "dirty_tokens": max((row["tokens"] - row["durable_tokens"] for row in rows), default=0),
        "generations": [row["generation"] for row in rows],
    }


def tail_cell(report, chat_id):
    residency = tail_residency(report, chat_id)
    return "?" if residency is None else human(residency["bytes"])


def title(row):
    return (
        row.get("session", {}).get("title")
        if row.get("session")
        else (
            row.get("metadata", {}).get("title") or row.get("metadata", {}).get("cwd") or row["id"]
        )
    )


def health(row):
    if not row["consistent"]:
        return "BUSY/CHANGED"
    if any(p["severity"] == "error" for p in row["issues"]):
        return "ERROR"
    if any(p["severity"] == "warning" for p in row["issues"]):
        return "WARN"
    if any(p["code"] in {"NEWER_TURN", "NEW_LOCAL_MESSAGES", "UNPUBLISHED"} for p in row["issues"]):
        return "UPDATING"
    return "SAVED" if row["expected_blocks"] else "EMPTY"


def match(selector, row):
    selector = selector.casefold()
    session = row.get("session") or {}
    return row["id"].startswith(selector) or any(
        selector in str(value).casefold()
        for value in (
            title(row),
            row.get("metadata", {}).get("cwd", ""),
            session.get("session_id", ""),
        )
    )


def selected(report, selector):
    if not selector:
        return report["chats"], report["unsnapshotted_chats"]
    chats = [row for row in report["chats"] if match(selector, row)]
    missing = [
        s for s in report["unsnapshotted_chats"] if match(selector, {"id": s["id"], "session": s})
    ]
    if not chats and not missing:
        raise ValueError(f"No chat matches {selector!r}")
    return chats, missing


def render(report, *, selector=None, details=False, audit_view=False):
    chats, missing = selected(report, selector)
    print(f"Radiance cache on {clean(report['host'])}  |  {report['scanned_at']}")
    print(
        f"Files: {human(report['totals']['file_bytes'])}"
        f"  |  allocated: {human(report['totals']['allocated_bytes'])}"
        f"  |  filesystem available: {human(report['filesystem']['available_bytes'])}"
    )
    print(
        "Storage: "
        + " | ".join(
            f"{name.replace('_', ' ')} {human(amount['file_bytes'])}"
            for name, amount in report["storage"].items()
            if amount["file_bytes"]
        )
    )
    io = report.get("io", {})
    print(
        f"Disk traffic: {human(io.get('written_file_bytes', 0))} in "
        f"{number(io.get('written_blocks', 0))} blocks; tracking "
        f"{number(io.get('tracked_chats', 0))}/{number(len(report.get('chats', [])))} chats"
    )
    tail = report.get("tail_journal", {})
    if tail.get("available"):
        tail_bytes = sum(row["bytes"] for row in tail["chats"])
        freshness = "live" if tail.get("active") else "stale"
        print(
            f"Snapshot tail RAM: {human(tail_bytes)} in {number(len(tail['chats']))}/"
            f"{number(tail['max_chats'])} chat(s); {number(tail['flush_tokens'])}-token "
            f"flush interval; {human(tail['max_bytes'])} limit; {freshness}"
        )
    elif tail.get("error"):
        print(f"Snapshot tail RAM: unreadable ({clean(tail['error'])})")
    scheduler = report.get("scheduler", {})
    if scheduler.get("available"):
        requests = scheduler.get("requests", [])
        states = (
            ", ".join(
                f"{row['chat_id'][:12]} {row['state']} {number(row['computed_tokens'])} tok"
                for row in requests
            )
            or "no active requests"
        )
        freshness = "live" if scheduler.get("active") else "stale" if requests else "idle"
        policy = (
            "switches between responses"
            if scheduler.get("policy") == "response_boundary"
            else f"{scheduler['quantum_seconds']:g}s slices"
        )
        print(
            f"GPU scheduler: {freshness}; {policy}; "
            f"{number(scheduler.get('switches'))} handovers; {states}"
        )
        worker = scheduler.get("worker", {})
        if worker:
            residency = worker.get("residency", {})
            resident_bytes = sum(image["data_bytes"] for image in residency.get("images", []))
            resident = (
                f"; {human(resident_bytes)} chat data in "
                f"{number(len(residency['images']))} RAM image(s)"
                if residency
                else ""
            )
            print(
                f"GPU handover RAM: {human(worker['allocated_bytes'])} allocated of "
                f"{human(worker['reserved_capacity_bytes'])}{resident}"
            )
            if "last_allocation_seconds" in worker:
                print(
                    "Pinned RAM allocation: last handover allocated "
                    f"{human(worker['last_allocation_bytes'])} in "
                    f"{worker['last_allocation_seconds']:.2f}s; "
                    f"{number(worker['allocation_events'])} allocation(s), "
                    f"{worker['allocation_seconds']:.2f}s total"
                )
                replacement = (
                    f"; {number(worker['generation_replacements'])} same-chat "
                    "generation replacement(s) discarded in place"
                )
            else:
                replacement = ""
            print(
                f"GPU cache transfer: last handover moved "
                f"{human(worker['last_transfer_bytes'])} in "
                f"{worker['last_transfer_seconds']:.2f}s; "
                f"{human(worker['transferred_bytes'])} in "
                f"{worker['transfer_seconds']:.2f}s total{replacement}"
            )
    elif scheduler.get("error"):
        print(f"GPU scheduler telemetry: unreadable ({clean(scheduler['error'])})")
    drive = report.get("drive", {})
    if drive.get("available"):
        print(
            f"Drive: {clean(drive.get('model', '?'))} ({clean(drive.get('device', '?'))}); "
            f"host writes since boot: {human(drive['host_written_bytes_since_boot'])}"
        )
        smart = drive.get("smart", {})
        if smart.get("available"):
            written = smart.get("lifetime_host_written_bytes")
            print(
                f"SMART: {number(smart.get('percentage_used'))}% endurance used; "
                f"lifetime host writes: {human(written) if isinstance(written, int) else '?'}; "
                f"media errors: {number(smart.get('media_errors'))}; "
                f"sample age: {int(smart.get('age_seconds') or 0):,}s"
            )
            if smart.get("critical_warnings"):
                print(f"SMART WARNING: {clean(str(smart['critical_warnings']))}")
        else:
            print(f"SMART: unavailable ({clean(smart.get('reason', 'unavailable'))})")
    verification = report["verification"]
    if verification["requested"]:
        print(
            f"Payload checks: {verification['verified_objects']}/{verification['objects']} "
            "scanned objects passed decompression and SHA-256 verification."
        )
    print()
    print(
        f"{'CHAT / ABI':23} {'STATE':12} {'PI PID / PORT':17} {'PUBLISHED / LAST TURN':>23}"
        f" {'BLOCKS':>18} {'DISK':>11} {'HANDOVER RAM':>15} {'TAIL RAM':>11}"
        f" {'DISK TRAFFIC':>13}  CHAT"
    )
    for row in chats:
        coverage = row["coverage_percent"]
        blocks = (
            f"{row['valid_blocks']}/{row['expected_blocks']} {coverage:.0f}%"
            if coverage is not None
            else "unknown / empty"
        )
        session = row.get("session") or {}
        tokens = (
            f"{number(row['metadata'].get('tokens'))} / {number(session.get('last_turn_tokens'))}"
        )
        process = active(row["active_processes"])
        io = row.get("io", {})
        write_traffic = human(io.get("written_file_bytes", 0)) if io.get("available") else "?"
        print(
            f"{row['id'][:12]}/{row['abi'][:8]:8}  {health(row):12} {process:17}"
            f" {tokens:>23} {blocks:>18}"
            f" {human(row['totals']['file_bytes']):>11}"
            f" {handover_cell(report, row['id']):>15} {tail_cell(report, row['id']):>11}"
            f" {write_traffic:>13}  "
            f"{clean(title(row))[:72]}"
        )
    for session in missing:
        tokens = f"0 / {number(session['last_turn_tokens'])}"
        process = active(session["active_processes"])
        print(
            f"{session['id'][:12]}/{'-':8}  {'NO SNAPSHOT':12} {process:17}"
            f" {tokens:>23} {'0 saved':>18}"
            f" {human(0):>11} {handover_cell(report, session['id']):>15}"
            f" {tail_cell(report, session['id']):>11} {'?':>13}  {clean(session['title'])[:72]}"
        )
    if not chats and not missing:
        print("No labelled chats found.")
    print(
        "\nBLOCKS = completeness of the published snapshot manifest; "
        + ("payload verification requested." if verification["requested"] else "headers checked.")
    )
    print(
        "Token columns are published context / last recorded Pi turn. "
        "Exact reusable tokens are not recorded."
    )
    print(
        "DISK TRAFFIC = cumulative completed snapshot payload writes since tracking began, "
        "including subsequently deleted blocks; excludes metadata and incomplete writes."
    )
    print(
        "HANDOVER RAM = useful bytes in a chat's pinned handover image; GPU marks the active "
        "bank. The shared content-addressed CPU offload tier is excluded because its blocks "
        "cannot be assigned uniquely to one chat."
    )
    print(
        "TAIL RAM = the newest changing recurrent blocks held by the snapshot tier. "
        "They become durable at the configured token interval or a forced flush."
    )
    print(
        f"Local transcripts: {human(report['local_transcript_bytes'])}; "
        "listed separately from remote cache."
    )
    for problem in report["issues"]:
        print(
            f"{problem['severity'].upper()}: {problem['code']}: {clean(problem['message'])}"
            + (f" [{clean(problem['path'])}]" if problem.get("path") else "")
        )
    for row in chats:
        if row["issues"] or details:
            print(f"\n{row['id'][:12]} [{row['abi'][:8]}] {clean(title(row))}")
        for problem in row["issues"]:
            print(
                f"  {problem['severity'].upper()}: {problem['code']}: {clean(problem['message'])}"
            )
            if (details or audit_view) and problem.get("path"):
                print(f"    {clean(problem['path'])}")
        if details:
            io = row.get("io", {})
            residency = handover_residency(report, row["id"])
            if residency is None:
                print("  Handover cache: per-chat telemetry unavailable from this backend")
            else:
                locations = []
                if residency["gpu"]:
                    locations.append("active on GPU")
                if residency["images"]:
                    locations.append(
                        f"{human(residency['data_bytes'])} useful data in "
                        f"{human(residency['allocated_bytes'])} pinned buffer space"
                    )
                print(f"  Handover cache: {'; '.join(locations) or '0.0 B in pinned RAM'}")
            tail = tail_residency(report, row["id"])
            if tail is None:
                print("  Snapshot tail RAM: telemetry unavailable")
            else:
                print(
                    f"  Snapshot tail RAM: {human(tail['bytes'])} in "
                    f"{number(tail['blocks'])} blocks; {number(tail['dirty_tokens'])} "
                    "tokens newer than the disk head"
                )
            if io.get("available"):
                print(
                    f"  Disk traffic since {io.get('since', '?')}: "
                    f"{human(io.get('written_file_bytes', 0))} in "
                    f"{number(io.get('written_blocks', 0))} blocks; "
                    f"{number(io.get('reused_blocks', 0))} existing blocks skipped; "
                    f"{number(io.get('write_failures', 0))} failed batches. "
                    "Crash-interrupted batches may be undercounted."
                )
            publication = row["metadata"].get("publication", {})
            gc = row["metadata"].get("gc", {})
            if publication:
                print(
                    f"  Latest publication: {clean(publication.get('result', '?'))}; "
                    f"requested tokens: {number(publication.get('tokens'))}; "
                    f"missing blocks: {len(publication.get('missing_keys', []))}; "
                    f"invalid blocks: {len(publication.get('invalid_keys', []))}"
                )
            if gc:
                print(
                    f"  Garbage collection: {clean(gc.get('status', '?'))}; "
                    f"removed {gc.get('removed_files', 0)} files "
                    f"({human(gc.get('removed_file_bytes', 0))})"
                )
            print(
                f"  Metadata updated: {row['metadata'].get('updated_at', '?')}; "
                f"status: {row['metadata'].get('status', '?')}"
            )
            print(
                f"  Payloads verified: {row['verified_blocks']}/{row['expected_blocks']} "
                "published blocks"
            )
            raw = row["totals"]["raw_bytes"]
            object_bytes = sum(o["file_bytes"] for o in row["objects"])
            savings = f"{100 * (1 - object_bytes / raw):.1f}%" if raw else "n/a"
            print(f"  Raw tensor bytes: {human(raw)}; compression saving: {savings}")
            transcript = (row.get("session") or {}).get("session_file") or row["metadata"].get(
                "session_file"
            )
            print(f"  Transcript: {clean(transcript)}")
            print("  Generations:")
            for generation in row["generations"]:
                print(
                    f"    {'CURRENT' if generation['current'] else 'STALE':7} {generation['id']}"
                    f"  {generation['files']} files  {human(generation['file_bytes'])}"
                )
            print("  Tensor groups:       published / other      disk       raw")
            configurations = next(
                (a["groups"] for a in report["abis"] if a["id"] == row["abi"]), []
            )
            for group in row["groups"]:
                config = next((c for c in configurations if c["id"] == group["id"]), {})
                layer = (config.get("layers") or [""])[0]
                print(
                    f"    g{group['id']:<4} {group['published_blocks']:>15}"
                    f" / {group['other_blocks']:<5} {human(group['file_bytes']):>11}"
                    f" {human(group['raw_bytes']):>11}  {clean(layer)}"
                )
            for label, objects in (
                ("Missing", row["missing_blocks"]),
                ("Invalid", row["invalid_blocks"]),
            ):
                for name in objects[:20]:
                    print(f"  {label}: {name}")
                if len(objects) > 20:
                    print(f"  ... {len(objects) - 20} more; use --json for every path")
    for snapshot in report["duplicate_snapshots"]:
        if not any(row["id"] == snapshot["chat_id"] for row in chats):
            continue
        print(
            f"\nDUPLICATE SNAPSHOTS: {snapshot['chat_id'][:12]}, "
            f"{len(snapshot['copies'])} generation copies with matching manifests"
        )
        if details or audit_view:
            for copy in snapshot["copies"]:
                print(f"  {clean(copy['path'])}  {human(copy['file_bytes'])}")
    duplicates = [d for d in report["duplicates"] if any(r["id"] == d["chat_id"] for r in chats)]
    if duplicates:
        print(
            f"\nDuplicate-content candidates: {len(duplicates)} sets,"
            f" {human(sum(d['duplicate_file_bytes'] for d in duplicates))} extra file bytes."
        )
        print(
            "Equal blocks required by one current head are excluded; "
            "these are not automatic deletion recommendations."
        )
        if details or audit_view:
            for duplicate in duplicates[:20]:
                verification = (
                    "verified payloads" if duplicate["verified"] else "recorded digests; unverified"
                )
                print(f"  g{duplicate['group']} {duplicate['digest'][:16]}  {verification}")
                for path in duplicate["copies"]:
                    print(f"    {clean(path)}")
            if len(duplicates) > 20:
                print("  Additional duplicate paths are available with --json.")


def collect_memory(args):
    operation = (
        "request_allocation_map"
        if args.map
        else "refresh_report"
        if args.refresh
        else "read_report"
    )
    if args.host in ("local", "localhost", "127.0.0.1"):
        return getattr(memory, operation)()
    source = Path(memory.__file__).read_text()
    source += f"\nprint(json.dumps({operation}()))\n"
    result = subprocess.run(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "--",
            args.host,
            "python3 -",
        ],
        input=source,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    if result.returncode:
        raise ValueError("SSH memory inspection failed")
    return json.loads(result.stdout)


def render_memory(report):
    if not report["available"]:
        messages = {
            "not_enabled": (
                "not enabled in the running backend; available after the updated backend starts"
            ),
            "stale": "stale; the backend is no longer publishing current samples",
            "refresh_timeout": "buffer inventory refresh timed out",
            "backend_changed": "backend changed during the refresh; retry the command",
            "map_not_enabled": (
                "allocation map needs the updated backend; counters remain available"
            ),
            "map_capture_failed": "allocation map capture failed; counters remain available",
            "map_unavailable": "allocation map unavailable",
            "map_timeout": "allocation map request timed out; retry the command",
        }
        print("Radiance GPU memory: " + messages.get(report["state"], "unavailable"))
        return
    if report.get("schema") == memory.MAP_SCHEMA:
        render_allocation_map(report)
        return
    print(f"Radiance GPU memory | sample age {report['age_seconds']:.1f}s | one shared reporter")
    counts = report["allocator"]
    for label, key in (
        ("Live tensor allocations", "allocated_bytes"),
        ("Allocator reservation", "reserved_bytes"),
        ("Unused reservation", "unused_reserved_bytes"),
        ("Waiting for pending frees", "pending_free_bytes"),
        ("Fragmented unused blocks", "inactive_split_bytes"),
        ("Peak tensor allocations", "allocated_peak_bytes"),
        ("Peak allocator reservation", "reserved_peak_bytes"),
    ):
        value = counts.get(key)
        print(f"  {label:29} {human(value) if value is not None else '?':>12}")
    print(
        f"  Allocation retries: {number(counts['allocation_retries'])}; "
        f"OOM events: {number(counts['out_of_memory_events'])}"
    )
    inventory = report["inventory"]
    age = max(0, time.time() - inventory["collected_at"])
    print(f"\nKnown GPU buffer storage (partial inventory, {age:.1f}s old):")
    for key, label in memory.GROUPS.items():
        row = inventory["groups"][key]
        if row["bytes"]:
            print(f"  {label:37} {human(row['bytes']):>12}")
    print(f"  {'Deduplicated total':37} {human(inventory['known_storage_bytes']):>12}")
    if inventory["truncated"]:
        print("  Inventory reached its time/size bound; some known buffers are omitted.")
    overhead = report["overhead"]

    def elapsed(value):
        return "?" if value is None else f"{value:.3f} ms"

    print(
        f"\nCollection: {elapsed(overhead['sample_ms'])}; "
        f"highest {elapsed(overhead['max_sample_ms'])}; "
        f"previous report write {elapsed(overhead['previous_write_ms'])}; "
        f"buffer inventory {elapsed(inventory['duration_ms'])}"
    )
    print(
        "Unused reservation includes fragmented/pending blocks and is not guaranteed reclaimable."
    )
    print(
        "Peaks are since the allocator's last reset; this report never resets them. "
        "Buffer sizes are metadata only "
        "and count shared storage once."
    )


def render_allocation_map(report):
    reused = "shared recent map" if report.get("reused") else "new map"
    totals = report["totals"]
    print(f"Radiance GPU allocation map | {report['age_seconds']:.1f}s old | {reused}")
    print(
        f"  {number(totals['segments'])} segments, {number(totals['blocks'])} blocks; "
        f"reserved {human(totals['reserved_bytes'])}; "
        f"live allocations {human(totals['allocated_bytes'])}"
    )
    print(
        f"  Free pieces in partly occupied segments: {human(totals['fragmented_bytes'])}; "
        f"pending frees: {human(totals['pending_bytes'])}"
    )
    if report["truncated"]:
        print("  PARTIAL MAP: scan limit reached; totals cover only the scanned segments.")
    print("\nPool        Inactive bytes    Largest free block    Fully inactive segments")
    for pool, values in report["pools"].items():
        if values["segments"]:
            print(
                f"{pool:10} {human(values['inactive_bytes']):>14}  "
                f"{human(values['largest_free_block_bytes']):>20}  "
                f"{human(values['fully_inactive_bytes']):>23}"
            )
    print("\nFree block sizes across pools       Blocks          Bytes")
    bounds = (0, *memory.FREE_BINS)
    for index, row in enumerate(report["free_block_histogram"]):
        upper = f"under {memory.FREE_BINS[index]}" if index < len(memory.FREE_BINS) else "and up"
        label = (
            f"{bounds[index]} MiB to {upper} MiB"
            if index < len(memory.FREE_BINS)
            else f"{bounds[index]} MiB {upper}"
        )
        print(f"  {label:32} {number(row['blocks']):>6}  {human(row['bytes']):>13}")
    print("\nSegments with the most unused space (fragmented segments first):")
    for row in report["segments"]:
        print(
            f"  S{row['segment']} {row['pool']}: {human(row['reserved_bytes'])} total; "
            f"{human(row['inactive_bytes'])} free; "
            f"largest piece {human(row['largest_free_block_bytes'])}"
        )
        owners = [
            f"{memory.GROUPS[key]} {human(value)}" for key, value in row["owners"].items() if value
        ]
        if row["unattributed_allocated_bytes"]:
            owners.append(
                f"unattributed/allocation padding {human(row['unattributed_allocated_bytes'])}"
            )
        if row["pending_bytes"]:
            owners.append(f"pending frees {human(row['pending_bytes'])}")
        print("    Occupied by: " + ("; ".join(owners) if owners else "no live allocations"))
    if report["omitted_segments"]:
        print(f"  {number(report['omitted_segments'])} other segments included in totals.")
    if report["ownership_truncated"]:
        print("  Owner inventory reached its bound; some known owners are unlabelled.")
    print(
        f"\nCPU collection: allocator {report['snapshot_ms']:.3f} ms; "
        f"owner inventory {report['inventory_ms']:.3f} ms; analysis {report['analysis_ms']:.3f} ms"
    )
    print(
        "Owners are best-effort storage matches; "
        "unlabelled bytes include padding and working buffers."
    )
    print(
        "Free pieces are not a reclaimable-byte promise. "
        "Private pools may restrict reuse or release."
    )


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    defaults = {
        "host": "ai",
        "cache_root": DEFAULT_ROOT,
        "abi": None,
        "chat": None,
        "sessions_root": [str(DEFAULT_PROJECTS), str(DEFAULT_SESSIONS)],
        "no_sessions": False,
        "stale_after": 300,
        "verify": False,
        "json": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.command = args.command or "status"
    if args.abi and not audit.ID.fullmatch(args.abi):
        cli.error("--abi must be a SHA256 identifier")
    if args.stale_after < 0 or not args.stale_after < float("inf"):
        cli.error("--stale-after must be finite and nonnegative")
    if args.command == "memory":
        try:
            report = collect_memory(args)
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                render_memory(report)
            return 0 if report["available"] else 2
        except (OSError, ValueError, subprocess.TimeoutExpired):
            if args.json:
                print(json.dumps({"available": False, "state": "inspection_failed"}))
            else:
                print("Radiance GPU memory inspection failed", file=sys.stderr)
            return 2
    if args.command in ("list", "compact", "flush"):
        from .radiance_cache import main as legacy_main

        legacy = ["--host", args.host, "--cache-root", args.cache_root]
        if args.abi:
            legacy += ["--abi", args.abi]
        legacy.append(args.command)
        if args.command in ("compact", "flush"):
            legacy += ["--identity-json", args.identity_json]
            if args.command == "flush":
                legacy += ["--timeout", str(args.timeout)]
        elif args.json:
            legacy.append("--json")
        return legacy_main(legacy)
    if args.command == "watch" and (not 1 <= args.interval <= 60 or args.count < 0):
        cli.error("watch requires --interval between 1 and 60 and nonnegative --count")
    selector = getattr(args, "selector", None) or args.chat
    iteration = 0
    try:
        while True:
            report = collect(args)
            rows, missing = selected(report, selector)
            if (
                args.command == "show"
                and len({r["id"] for r in rows} | {s["id"] for s in missing}) > 1
            ):
                raise ValueError("Chat selector is ambiguous; use the chat ID prefix from status")
            if args.json:
                if selector:
                    report = {
                        **report,
                        "chats": rows,
                        "unsnapshotted_chats": missing,
                        "scope": {
                            **report["scope"],
                            "chat": selector,
                            "storage_totals": "whole scanned root",
                        },
                    }
                print(json.dumps(report, indent=None if args.command == "watch" else 2), flush=True)
            else:
                if args.command == "watch" and sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                render(
                    report,
                    selector=selector,
                    details=args.command == "show",
                    audit_view=args.command == "audit",
                )
                sys.stdout.flush()
            problems = [*report["issues"], *(p for row in rows for p in row["issues"])]
            if args.command != "watch":
                return int(
                    args.command == "audit"
                    and any(p["severity"] in ("warning", "error") for p in problems)
                )
            iteration += 1
            if args.count and iteration >= args.count:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as error:
        if args.json:
            print(json.dumps({"error": str(error)}))
        else:
            print(f"qwen-radiance-cache: {error}", file=sys.stderr)
        return 2
