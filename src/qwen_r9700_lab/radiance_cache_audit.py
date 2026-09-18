"""Read-only snapshot inspection; standalone so it can be sent over SSH."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

FORMAT = "qwen-chat-cache-v1"
HEADER = struct.Struct(">8sQ32s")
ID = re.compile(r"[0-9a-f]{64}\Z")
KEY = re.compile(r"g([0-9]+)-[0-9a-f]{16,128}\.qkv\Z")
MAGICS = {b"QWENKV1Z": "zstd", b"QWENKV1R": "raw"}
CATEGORIES = (
    "published",
    "fallback",
    "unreferenced",
    "old_generations",
    "temporary",
    "legacy",
    "metadata",
    "other_snapshot_files",
    "other_cache_files",
)
FAIR_STATUS_PREFIX = Path("/dev/shm/qwen-radiance-fair-public")
TAIL_STATUS_PATH = Path("/dev/shm/qwen-radiance-snapshot-tail.json")


def issue(code, message, *, severity="warning", path=None):
    return {
        "code": code,
        "severity": severity,
        "message": message,
        **({"path": str(path)} if path is not None else {}),
    }


def totals():
    return {"files": 0, "file_bytes": 0, "allocated_bytes": 0, "raw_bytes": 0}


def add(total, item):
    total["files"] += 1
    for key in ("file_bytes", "allocated_bytes", "raw_bytes"):
        total[key] += item.get(key, 0)


def real_directory(path):
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f"not a real directory: {path}")


@contextlib.contextmanager
def open_regular(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"not a regular file: {path}")
        yield stream


def read_json(path):
    with open_regular(path) as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("metadata exceeds 1 MiB")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    return value, hashlib.sha256(data).hexdigest()


def walk_files(root, problems):
    """Never follow a symlink, including directory links inside the cache."""

    def failed(error):
        problems.append(issue("SCAN_ERROR", str(error), severity="error"))

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                dirs.remove(name)
                problems.append(issue("SYMLINK", "Directory symlink skipped", path=path))
        for name in files:
            path = Path(directory) / name
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    problems.append(issue("UNSAFE_FILE", "Non-regular file skipped", path=path))
                    continue
                yield path, info
            except OSError as error:
                problems.append(issue("SCAN_CHANGED", str(error), severity="info", path=path))


def file_info(path, info):
    return {
        "path": str(path),
        "name": path.name,
        "file_bytes": info.st_size,
        "allocated_bytes": info.st_blocks * 512,
        "mtime": info.st_mtime,
        "inode": [info.st_dev, info.st_ino],
        "raw_bytes": 0,
    }


def block_info(path, info):
    result = file_info(path, info)
    result.update(valid=False, verified=False, group=int(KEY.fullmatch(path.name)[1]))
    try:
        with open_regular(path) as stream:
            header = stream.read(HEADER.size)
        magic, raw_size, digest = HEADER.unpack(header)
        if magic not in MAGICS or not 0 < raw_size <= 1024**3:
            raise ValueError("invalid encoding or uncompressed block size")
        if info.st_size <= HEADER.size:
            raise ValueError("empty block payload")
        if magic == b"QWENKV1R" and info.st_size != HEADER.size + raw_size:
            raise ValueError("raw block payload length mismatch")
        result.update(valid=True, raw_bytes=raw_size, digest=digest.hex(), encoding=MAGICS[magic])
    except (OSError, ValueError, struct.error) as error:
        result["error"] = str(error)
    return result


def verify_block(record):
    """Stream decoded bytes with bounded memory. Never write or repair a block."""
    digest = hashlib.sha256()
    count = 0
    process = None
    try:
        with open_regular(record["path"]) as source:
            before = os.fstat(source.fileno())
            if [before.st_dev, before.st_ino] != record["inode"] or before.st_size != record[
                "file_bytes"
            ]:
                raise FileNotFoundError("snapshot object changed during inspection")
            source.seek(HEADER.size)
            if record["encoding"] == "zstd":
                executable = shutil.which("zstd")
                if not executable:
                    return "unavailable", "zstd executable is required for --verify on this host"
                process = subprocess.Popen(
                    [executable, "--decompress", "--stdout", "--quiet"],
                    stdin=source,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                output = process.stdout
            else:
                output = source
            while chunk := output.read(1024 * 1024):
                count += len(chunk)
                if count > record["raw_bytes"]:
                    raise ValueError("decoded payload exceeds the declared size")
                digest.update(chunk)
            if process and process.wait(timeout=10) != 0:
                raise ValueError("Zstd decompression/checksum failed")
            if count != record["raw_bytes"] or digest.hexdigest() != record["digest"]:
                raise ValueError("decoded size or SHA-256 checksum mismatch")
            after = os.fstat(source.fileno())
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise FileNotFoundError("snapshot object changed while being verified")
        return "verified", None
    except FileNotFoundError as error:
        return "changed", str(error)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return "corrupt", str(error)
    finally:
        if process:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


@contextlib.contextmanager
def inspection_lock(directory):
    """A brief, non-blocking exclusive lock excludes both publication and writes."""
    try:
        descriptor = os.open(directory / ".lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        yield "missing"
        return
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("chat lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield "busy"
        else:
            yield "locked"
    finally:
        os.close(descriptor)


def scan_chat(directory, abi, *, stale_after=300, verify=False):
    row = {
        "id": directory.name,
        "abi": abi,
        "path": str(directory),
        "issues": [],
        "objects": [],
        "generations": [],
        "groups": [],
        "metadata": {},
        "io": {"available": False},
        "consistent": True,
        "coverage_percent": None,
        "expected_blocks": None,
        "present_blocks": 0,
        "valid_blocks": 0,
        "verified_blocks": 0,
        "missing_blocks": [],
        "invalid_blocks": [],
        "inspection": "headers",
        "storage": {name: totals() for name in CATEGORIES},
        "totals": totals(),
    }
    problems = row["issues"]
    fingerprint = None
    with inspection_lock(directory) as lock:
        row["lock"] = lock
        if lock == "busy":
            row["consistent"] = False
            problems.append(
                issue(
                    "BUSY", "Snapshot is being read, written, or collected; retry", severity="info"
                )
            )
        if lock == "missing":
            problems.append(
                issue("MISSING_LOCK", "Chat has no writer lock; ownership is incomplete")
            )
        metadata = directory / "chat.json"
        try:
            if metadata.is_symlink():
                raise ValueError("chat metadata is a symlink")
            info, fingerprint = read_json(metadata)
            if info.get("format") != FORMAT:
                raise ValueError("unsupported chat metadata format")
            if info.get("id") != directory.name or not ID.fullmatch(
                str(info.get("generation", ""))
            ):
                raise ValueError("chat identity or current generation is invalid")
            head = info.get("head")
            if not isinstance(head, list) or any(
                not isinstance(k, str) or not KEY.fullmatch(k) for k in head
            ):
                raise ValueError("invalid published head manifest")
            if len(head) != len(set(head)):
                problems.append(
                    issue("DUPLICATE_MANIFEST_KEYS", "Published head repeats object names")
                )
            row["metadata"] = info
            row["expected_blocks"] = len(set(head))
            if info["generation"] in info.get("retired_generations", []):
                problems.append(
                    issue(
                        "RETIRED_CURRENT", "Current generation is marked retired", severity="error"
                    )
                )
        except (OSError, ValueError, TypeError) as error:
            problems.append(issue("BAD_METADATA", str(error), severity="error", path=metadata))
            info, head = {}, []
        current = info.get("generation")
        fallback = info.get("fallback") or {}
        fallback_generation = fallback.get("generation")
        if fallback and (
            not ID.fullmatch(str(fallback_generation))
            or not isinstance(fallback.get("head"), list)
            or any(not isinstance(k, str) or not KEY.fullmatch(k) for k in fallback["head"])
        ):
            problems.append(
                issue("BAD_FALLBACK", "Fallback snapshot manifest is invalid", severity="error")
            )
            fallback, fallback_generation = {}, None
        if fallback:
            problems.append(
                issue(
                    "FALLBACK_RETAINED",
                    "Previous complete snapshot retained until its replacement is verified",
                    severity="info",
                )
            )
        try:
            row["io"], _ = read_json(directory / "io.json")
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError):
            problems.append(
                issue("IO_COUNTERS_UNREADABLE", "Cumulative snapshot I/O counters are unreadable")
            )
        head_set = set(head)
        generation_root = directory / "generations"
        if generation_root.is_dir() and not generation_root.is_symlink():
            for path in sorted(generation_root.iterdir()):
                if path.is_dir() and not path.is_symlink():
                    row["generations"].append(
                        {"id": path.name, "current": path.name == current, **totals()}
                    )
                    if not ID.fullmatch(path.name):
                        problems.append(
                            issue("UNKNOWN_GENERATION", "Unexpected generation name", path=path)
                        )
        generation_rows = {g["id"]: g for g in row["generations"]}
        if current and current not in generation_rows:
            problems.append(
                issue(
                    "MISSING_GENERATION",
                    "Current generation directory is missing",
                    severity="error",
                )
            )
        old = [g for g in row["generations"] if g["id"] not in {current, fallback_generation}]
        if old and lock == "locked":
            problems.append(
                issue(
                    "GC_LEFTOVERS", f"{len(old)} non-current generation(s) remain after retirement"
                )
            )
        for path, attributes in walk_files(directory, problems):
            relative = path.relative_to(directory)
            parts = relative.parts
            generation = parts[1] if len(parts) >= 3 and parts[0] == "generations" else None
            is_object = generation is not None and len(parts) == 3 and KEY.fullmatch(path.name)
            record = block_info(path, attributes) if is_object else file_info(path, attributes)
            if (
                generation == fallback_generation
                and is_object
                and path.name in fallback.get("head", [])
            ):
                category = "fallback"
            elif generation and generation != current:
                category = "old_generations"
            elif path.name.startswith(".pending-"):
                category = "temporary"
            elif is_object:
                category = "published" if path.name in head_set else "unreferenced"
            elif path.name in ("chat.json", ".lock", "io.json", ".io.lock"):
                category = "metadata"
            else:
                category = "other_snapshot_files"
            record.update(category=category, generation=generation)
            add(row["storage"][category], record)
            add(row["totals"], record)
            if generation in generation_rows:
                add(generation_rows[generation], record)
            if is_object:
                row["objects"].append(record)
                if not record["valid"]:
                    problems.append(
                        issue("BAD_BLOCK_HEADER", record["error"], severity="error", path=path)
                    )
        if row["storage"]["temporary"]["files"] and lock == "locked":
            problems.append(
                issue(
                    "TEMP_LEFTOVERS", "Temporary files remain with no writer holding the chat lock"
                )
            )
        extras = [o for o in row["objects"] if o["category"] == "unreferenced"]
        if extras:
            age = max(0, time.time() - max(o["mtime"] for o in extras))
            old_extras = age >= stale_after and lock == "locked"
            problems.append(
                issue(
                    "UNREFERENCED_OLD" if old_extras else "UNPUBLISHED",
                    f"{len(extras)} object(s) outside the published head; "
                    "may be an unfinished request or failed cleanup",
                    severity="warning" if old_extras else "info",
                )
            )
        if info.get("status") == "incomplete":
            problems.append(
                issue(
                    "INCOMPLETE_PUBLICATION",
                    "Latest publication failed; the manifest describes the previous head",
                )
            )
        gc_status = info.get("gc", {}).get("status")
        if gc_status in ("pending", "failed"):
            problems.append(
                issue(
                    "GC_FAILED" if gc_status == "failed" else "GC_PENDING",
                    "Snapshot garbage collection has not finished; automatic retry is pending",
                )
            )
        if info.get("status") == "ready" and not head:
            problems.append(
                issue(
                    "EMPTY_HEAD",
                    "Ready metadata contains no restorable snapshot blocks",
                    severity="info",
                )
            )
    # Full verification deliberately runs outside the writer lock.
    if verify and lock == "locked":
        row["inspection"] = "payloads"
        for record in row["objects"]:
            if not record["valid"]:
                continue
            status, error = verify_block(record)
            record["verified"] = status == "verified"
            if status == "corrupt":
                record["valid"] = False
                problems.append(
                    issue("CORRUPT_PAYLOAD", error, severity="error", path=record["path"])
                )
            elif status == "changed":
                row["consistent"] = False
                problems.append(issue("SCAN_CHANGED", error, severity="info", path=record["path"]))
            elif status == "unavailable":
                row["inspection"] = "partial"
                if not any(p["code"] == "VERIFY_UNAVAILABLE" for p in problems):
                    problems.append(issue("VERIFY_UNAVAILABLE", error, severity="error"))
        try:
            if read_json(metadata)[1] != fingerprint:
                row["consistent"] = False
        except (OSError, ValueError):
            row["consistent"] = False
        if not row["consistent"]:
            problems.append(
                issue(
                    "SCAN_CHANGED",
                    "Publication changed during verification; repeat the audit",
                    severity="info",
                )
            )
    present = {o["name"]: o for o in row["objects"] if o["category"] == "published"}
    row["present_blocks"] = len(present)
    row["valid_blocks"] = sum(o["valid"] for o in present.values())
    row["verified_blocks"] = sum(o["verified"] for o in present.values())
    row["missing_blocks"] = sorted(head_set - present.keys())
    row["invalid_blocks"] = sorted(k for k, o in present.items() if not o["valid"])
    if row["consistent"] and row["missing_blocks"]:
        problems.append(
            issue(
                "MISSING_BLOCKS",
                f"{len(row['missing_blocks'])} published block(s) are missing",
                severity="error",
            )
        )
    if row["consistent"] and row["expected_blocks"]:
        row["coverage_percent"] = 100 * row["valid_blocks"] / row["expected_blocks"]
    groups = {}
    for record in row["objects"]:
        group = groups.setdefault(
            record["group"],
            {"id": record["group"], **totals(), "published_blocks": 0, "other_blocks": 0},
        )
        add(group, record)
        group["published_blocks" if record["category"] == "published" else "other_blocks"] += 1
    row["groups"] = sorted(groups.values(), key=lambda g: g["id"])
    if not row["consistent"]:
        # An observation without the lock must not diagnose an intermediate GC state.
        for problem in problems:
            if problem["code"] in {"BAD_METADATA", "MISSING_GENERATION", "BAD_BLOCK_HEADER"}:
                problem["severity"] = "info"
    return row


def duplicate_report(chats):
    buckets = defaultdict(list)
    for row in chats:
        if not row["consistent"]:
            continue
        for obj in row["objects"]:
            if obj["valid"] and obj["category"] != "fallback":
                buckets[(row["id"], obj["group"], obj["raw_bytes"], obj["digest"])].append(
                    (row, obj)
                )
    duplicates = []
    for (chat_id, group, raw_size, digest), copies in buckets.items():
        if len(copies) < 2:
            continue
        redundant = [o for _, o in copies if o["category"] != "published"]
        locations = {(r["abi"], o["generation"]) for r, o in copies}
        if not redundant and len(locations) == 1:
            # Equal states referenced by the same published head are legitimate.
            continue
        unique_inodes = {}
        for _, obj in copies:
            unique_inodes.setdefault(tuple(obj["inode"]), obj)
        duplicate_bytes = sum(o["file_bytes"] for o in unique_inodes.values()) - max(
            o["file_bytes"] for o in unique_inodes.values()
        )
        duplicates.append(
            {
                "chat_id": chat_id,
                "group": group,
                "raw_bytes": raw_size,
                "digest": digest,
                "copies": [o["path"] for _, o in copies],
                "duplicate_file_bytes": duplicate_bytes,
                "verified": all(o["verified"] for _, o in copies),
            }
        )
    return duplicates


def duplicate_snapshots(chats):
    buckets = defaultdict(list)
    for row in chats:
        if not row["consistent"]:
            continue
        for generation in row["generations"]:
            if generation["id"] == (row["metadata"].get("fallback") or {}).get("generation"):
                continue
            objects = [o for o in row["objects"] if o["generation"] == generation["id"]]
            if not objects or any(not o["valid"] for o in objects):
                continue
            signature = tuple(sorted((o["name"], o["raw_bytes"], o["digest"]) for o in objects))
            buckets[(row["id"], signature)].append(
                {
                    "abi": row["abi"],
                    "generation": generation["id"],
                    "path": str(Path(row["path"]) / "generations" / generation["id"]),
                    "objects": len(objects),
                    "file_bytes": sum(o["file_bytes"] for o in objects),
                    "verified": all(o["verified"] for o in objects),
                }
            )
    return [
        {"chat_id": chat_id, "copies": copies, "verified": all(c["verified"] for c in copies)}
        for (chat_id, _), copies in buckets.items()
        if len(copies) > 1
    ]


def command_json(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=4, check=True)
    return json.loads(result.stdout)


def drive_health(cache_root):
    """Read host writes and cached NVMe health, without privileges or chat data."""
    result = {"available": False, "smart": {"available": False, "reason": "unavailable"}}
    try:
        source = command_json(["findmnt", "-J", "-T", str(cache_root), "-o", "SOURCE"])[
            "filesystems"
        ][0]["source"]
        source = source.split("[", 1)[0]
        devices = command_json(["lsblk", "-J", "-s", "-p", "-o", "NAME,TYPE,MODEL", source])[
            "blockdevices"
        ]
        disks = []

        def visit(nodes):
            for node in nodes:
                if node.get("type") == "disk":
                    disks.append(node)
                visit(node.get("children", []))

        visit(devices)
        if len(disks) != 1:
            return {**result, "reason": "filesystem_does_not_map_to_one_disk"}
        device = disks[0]["name"]
        name = Path(device).name
        counters = (Path("/sys/class/block") / name / "stat").read_text().split()
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        result.update(
            available=True,
            device=device,
            model=disks[0].get("model", "").strip(),
            host_written_bytes_since_boot=int(counters[6]) * 512,
            uptime_seconds=uptime,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        )
        if shutil.which("busctl"):
            base = [
                "busctl",
                "--allow-interactive-authorization=no",
                "--json=short",
                "call",
                "org.freedesktop.UDisks2",
            ]
            objects = command_json(
                [
                    *base,
                    "/org/freedesktop/UDisks2",
                    "org.freedesktop.DBus.ObjectManager",
                    "GetManagedObjects",
                ]
            )["data"][0]
            block = objects.get("/org/freedesktop/UDisks2/block_devices/" + name, {})
            drive = block.get("org.freedesktop.UDisks2.Block", {}).get("Drive", {}).get("data")
            interface = "org.freedesktop.UDisks2.NVMe.Controller"
            props = objects.get(drive, {}).get(interface, {})
            if props:
                attributes = command_json(
                    [*base, drive, interface, "SmartGetAttributes", "a{sv}", "0"]
                )["data"][0]
                updated = props.get("SmartUpdated", {}).get("data", 0)
                result["smart"] = {
                    "available": bool(updated),
                    "source": "udisks2",
                    "updated_at_unix": updated,
                    "age_seconds": max(0, time.time() - updated) if updated else None,
                    "percentage_used": attributes.get("percent_used", {}).get("data"),
                    "lifetime_host_written_bytes": attributes.get("total_data_written", {}).get(
                        "data"
                    ),
                    "available_spare_percent": attributes.get("avail_spare", {}).get("data"),
                    "media_errors": attributes.get("media_errors", {}).get("data"),
                    "critical_warnings": props.get("SmartCriticalWarning", {}).get("data", []),
                }
        if not result["smart"]["available"] and shutil.which("smartctl"):
            # Nonzero smartctl exit codes can also report an unhealthy drive;
            # parse its JSON health values rather than discarding that evidence.
            proc = subprocess.run(
                ["smartctl", "-a", "-j", device], capture_output=True, text=True, timeout=4
            )
            smart = json.loads(proc.stdout).get("nvme_smart_health_information_log", {})
            if smart:
                units = smart.get("data_units_written")
                result["smart"] = {
                    "available": True,
                    "source": "smartctl",
                    "age_seconds": 0,
                    "percentage_used": smart.get("percentage_used"),
                    "lifetime_host_written_bytes": units * 512_000
                    if isinstance(units, int)
                    else None,
                    "media_errors": smart.get("media_errors"),
                    "critical_warnings": smart.get("critical_warning"),
                }
            else:
                result["smart"]["reason"] = "permission_denied_or_unsupported"
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        result["reason"] = type(error).__name__
    return result


def fair_scheduler_status(prefix=FAIR_STATUS_PREFIX):
    """Read content-free scheduler telemetry from tmpfs."""
    result = {"available": False, "active": False}
    prefix = Path(prefix)
    try:
        scheduler, _ = read_json(Path(str(prefix) + "-scheduler.json"))
    except FileNotFoundError:
        return result
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return {**result, "error": type(error).__name__}
    try:
        updated = float(scheduler["updated_at"])
        quantum = float(scheduler["quantum_seconds"])
        requests = scheduler.get("requests", [])
        if (
            not (quantum == 0 or 0.1 <= quantum <= 120)
            or not isinstance(requests, list)
            or any(
                not isinstance(row, dict)
                or row.get("state") not in {"running", "paused", "queued"}
                or not ID.fullmatch(str(row.get("chat_id", "")))
                or not ID.fullmatch(str(row.get("generation", "")))
                or not isinstance(row.get("computed_tokens"), int)
                or not isinstance(row.get("input_tokens"), int)
                for row in requests
            )
        ):
            raise ValueError("invalid fair-scheduler telemetry")
        age = max(0.0, time.time() - updated)
        result.update(
            available=True,
            active=age <= max(5, (quantum or 15) * 2),
            age_seconds=age,
            quantum_seconds=quantum,
            policy="response_boundary" if quantum == 0 else "time_slice",
            switches=int(scheduler.get("switches", 0)),
            cached_chats=int(scheduler.get("cached_chats", 0)),
            max_cached_chats=int(scheduler.get("max_cached_chats", 0)),
            requests=requests,
        )
        try:
            worker, _ = read_json(Path(str(prefix) + "-worker.json"))
            worker_updated = float(worker.get("updated_at", 0))
            fields = (
                "allocated_bytes",
                "reserved_capacity_bytes",
                "cached_chats",
                "switches",
                "last_transfer_bytes",
                "last_transfer_seconds",
                "transferred_bytes",
                "transfer_seconds",
            )
            if any(
                not isinstance(worker.get(key), (int, float)) or worker[key] < 0 for key in fields
            ):
                raise ValueError("invalid fair-scheduler worker telemetry")
            worker_result = {
                **{key: worker[key] for key in fields},
                "age_seconds": max(0.0, time.time() - worker_updated),
            }
            allocation_fields = (
                "last_allocation_bytes",
                "last_allocation_seconds",
                "allocation_events",
                "allocation_seconds",
                "generation_replacements",
            )
            allocation_present = [key in worker for key in allocation_fields]
            if any(allocation_present):
                if not all(allocation_present) or any(
                    not isinstance(worker[key], (int, float)) or worker[key] < 0
                    for key in allocation_fields
                ):
                    raise ValueError("invalid fair-scheduler allocation telemetry")
                handover = worker.get("last_handover")
                if handover not in {
                    "status",
                    "activate",
                    "drop",
                    "swap",
                    "generation-replace",
                }:
                    raise ValueError("invalid fair-scheduler handover telemetry")
                worker_result.update(
                    {key: worker[key] for key in allocation_fields},
                    last_handover=handover,
                )
            residency = worker.get("residency")
            if residency is not None:
                active = residency.get("active") if isinstance(residency, dict) else None
                images = residency.get("images") if isinstance(residency, dict) else None
                byte_fields = ("free_buffer_bytes", "staging_buffer_bytes")

                def valid_identity(value):
                    return (
                        isinstance(value, dict)
                        and ID.fullmatch(str(value.get("chat_id", ""))) is not None
                        and ID.fullmatch(str(value.get("generation", ""))) is not None
                    )

                if (
                    not isinstance(residency, dict)
                    or (active is not None and not valid_identity(active))
                    or not isinstance(images, list)
                    or any(
                        not valid_identity(image)
                        or not isinstance(image.get("data_bytes"), int)
                        or not isinstance(image.get("allocated_bytes"), int)
                        or not 0 <= image["data_bytes"] <= image["allocated_bytes"]
                        for image in images
                    )
                    or any(
                        not isinstance(residency.get(key), int) or residency[key] < 0
                        for key in byte_fields
                    )
                ):
                    raise ValueError("invalid fair-scheduler RAM residency telemetry")
                identities = [(image["chat_id"], image["generation"]) for image in images]
                if (
                    len(identities) != len(set(identities))
                    or (
                        active is not None
                        and (active["chat_id"], active["generation"]) in identities
                    )
                    or worker["cached_chats"] != len(images) + int(active is not None)
                    or worker["allocated_bytes"]
                    != sum(image["allocated_bytes"] for image in images)
                    + residency["free_buffer_bytes"]
                    + residency["staging_buffer_bytes"]
                ):
                    raise ValueError("inconsistent fair-scheduler RAM residency telemetry")
                worker_result["residency"] = {
                    "active": (
                        {
                            "chat_id": active["chat_id"],
                            "generation": active["generation"],
                        }
                        if active is not None
                        else None
                    ),
                    "images": [
                        {
                            "chat_id": image["chat_id"],
                            "generation": image["generation"],
                            "data_bytes": image["data_bytes"],
                            "allocated_bytes": image["allocated_bytes"],
                        }
                        for image in images
                    ],
                    **{key: residency[key] for key in byte_fields},
                }
            result["worker"] = worker_result
        except FileNotFoundError:
            pass
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        result.update(error=type(error).__name__)
    return result


def snapshot_tail_status(path=TAIL_STATUS_PATH):
    """Read content-free dirty-tail residency from the backend's tmpfs status."""
    result = {"available": False, "active": False, "chats": []}
    try:
        payload, _ = read_json(Path(path))
        updated = float(payload["updated_at"])
        chats = payload.get("chats")
        if (
            payload.get("schema") != "urn:qwen-r9700:radiance-tail-residency:v1"
            or not isinstance(payload.get("pid"), int)
            or not isinstance(payload.get("flush_tokens"), int)
            or payload["flush_tokens"] < 1
            or not isinstance(payload.get("max_bytes"), int)
            or payload["max_bytes"] < 1
            or not isinstance(payload.get("max_chats"), int)
            or payload["max_chats"] < 1
            or not isinstance(chats, list)
            or any(
                not isinstance(row, dict)
                or not ID.fullmatch(str(row.get("chat_id", "")))
                or not ID.fullmatch(str(row.get("generation", "")))
                or any(
                    not isinstance(row.get(field), int) or row[field] < 0
                    for field in ("tokens", "durable_tokens", "blocks", "bytes")
                )
                or row["durable_tokens"] > row["tokens"]
                for row in chats
            )
        ):
            raise ValueError("invalid snapshot-tail telemetry")
        identities = [(row["chat_id"], row["generation"]) for row in chats]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate snapshot-tail residency identity")
        age = max(0.0, time.time() - updated)
        last = payload.get("last_flush")
        last_result = None
        if last is not None:
            if (
                not isinstance(last, dict)
                or not ID.fullmatch(str(last.get("chat_id", "")))
                or not ID.fullmatch(str(last.get("generation", "")))
                or not isinstance(last.get("tokens"), int)
                or last["tokens"] < 0
                or last.get("reason")
                not in {"forced", "token_interval", "ram_eviction", "explicit"}
            ):
                raise ValueError("invalid snapshot-tail flush telemetry")
            last_result = {
                field: last[field] for field in ("chat_id", "generation", "tokens", "reason")
            }
        result.update(
            available=True,
            active=age <= 15,
            age_seconds=age,
            pid=payload["pid"],
            flush_tokens=payload["flush_tokens"],
            max_bytes=payload["max_bytes"],
            max_chats=payload["max_chats"],
            chats=chats,
        )
        if last_result is not None:
            result["last_flush"] = last_result
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        result["error"] = type(error).__name__
    return result


def scan(cache_root, *, abi=None, stale_after=300, verify=False):
    root = Path(cache_root).expanduser().absolute()
    real_directory(root)
    if root.resolve() != root:
        raise ValueError("cache root must not contain symlinks")
    output = {
        "schema": "qwen-radiance-cache-inspection-v1",
        "cache_root": str(root),
        "scanned_at": datetime.now(UTC).isoformat(),
        "chats": [],
        "issues": [],
        "storage": {name: totals() for name in CATEGORIES},
        "totals": totals(),
        "abis": [],
        "duplicates": [],
        "verify_requested": verify,
    }
    problems = output["issues"]
    snapshots = root / "snapshots"
    if snapshots.exists() or snapshots.is_symlink():
        real_directory(snapshots)
        roots = [snapshots / abi] if abi else sorted(snapshots.iterdir())
        root_names = {path.name for path in roots}
        for namespace in roots:
            if not ID.fullmatch(namespace.name):
                problems.append(
                    issue("UNKNOWN_ABI", "Unexpected snapshot namespace", path=namespace)
                )
                if namespace.is_dir() and not namespace.is_symlink():
                    for path, attributes in walk_files(namespace, problems):
                        add(output["storage"]["other_snapshot_files"], file_info(path, attributes))
                continue
            try:
                real_directory(namespace)
                data = namespace / "data"
                data_abi = namespace.name
                alias = False
                if not data.is_dir() or data.is_symlink():
                    manifest, _ = read_json(namespace / "abi.json")
                    data_abi = str(manifest.get("storage", {}).get("data_abi", ""))
                    if not ID.fullmatch(data_abi):
                        raise ValueError("runtime ABI has no valid snapshot data reference")
                    data = snapshots / data_abi / "data"
                    alias = True
                real_directory(data)
            except (OSError, ValueError) as error:
                problems.append(issue("BAD_ABI", str(error), severity="error", path=namespace))
                continue
            description = {
                "id": namespace.name,
                "data_abi": data_abi,
                "data_alias": alias,
                "groups": [],
            }
            configs = list(data.glob("*/config.json"))
            if (
                len(configs) == 1
                and not configs[0].is_symlink()
                and not configs[0].parent.is_symlink()
            ):
                try:
                    config, _ = read_json(configs[0])
                    description["groups"] = [
                        {
                            "id": index,
                            "tokens_per_block": group.get("tokens_per_block"),
                            "layers": group.get("layer_names", []),
                        }
                        for index, group in enumerate(config.get("kv_cache_groups", []))
                    ]
                except (OSError, ValueError, TypeError, AttributeError) as error:
                    problems.append(issue("BAD_CONFIG", str(error), path=configs[0]))
            output["abis"].append(description)
            # Runtime ABIs intentionally point at the compatible data ABI. In a
            # whole-cache scan, let that data namespace own the chat rows once.
            if alias and not abi and data_abi in root_names:
                for path, attributes in walk_files(namespace, problems):
                    add(output["storage"]["metadata"], file_info(path, attributes))
                continue
            managed = data / FORMAT
            counted = set()
            if managed.exists() or managed.is_symlink():
                try:
                    real_directory(managed)
                    for directory in sorted(managed.iterdir()):
                        if not ID.fullmatch(directory.name):
                            continue
                        try:
                            real_directory(directory)
                            row = scan_chat(
                                directory, namespace.name, stale_after=stale_after, verify=verify
                            )
                            output["chats"].append(row)
                            counted.add(directory)
                        except (OSError, ValueError) as error:
                            problems.append(
                                issue(
                                    "CHAT_SCAN_FAILED", str(error), severity="error", path=directory
                                )
                            )
                except (OSError, ValueError) as error:
                    problems.append(
                        issue("BAD_MANAGED_ROOT", str(error), severity="error", path=managed)
                    )
            for path, attributes in walk_files(namespace, problems):
                if any(directory in path.parents for directory in counted):
                    continue
                category = (
                    "legacy"
                    if path.suffix == ".bin"
                    else (
                        "metadata"
                        if path.suffix == ".json" or path.name == ".engine.lock"
                        else "other_snapshot_files"
                    )
                )
                add(output["storage"][category], file_info(path, attributes))
    # Whole-cache overhead is included only for an unfiltered scan.
    if not abi:
        for child in root.iterdir():
            if child.name == "snapshots":
                continue
            if child.is_symlink():
                problems.append(issue("SYMLINK", "Cache symlink skipped", path=child))
            elif child.is_dir():
                for path, attributes in walk_files(child, problems):
                    add(output["storage"]["other_cache_files"], file_info(path, attributes))
            elif child.is_file():
                add(output["storage"]["other_cache_files"], file_info(child, child.stat()))
    for row in output["chats"]:
        for category, amounts in row["storage"].items():
            for key, value in amounts.items():
                output["storage"][category][key] += value
    for amounts in output["storage"].values():
        for key, value in amounts.items():
            output["totals"][key] += value
    output["duplicates"] = duplicate_report(output["chats"])
    output["duplicate_snapshots"] = duplicate_snapshots(output["chats"])
    objects = [o for row in output["chats"] for o in row["objects"]]
    output["verification"] = {
        "requested": verify,
        "objects": len(objects),
        "verified_objects": sum(o["verified"] for o in objects),
        "invalid_objects": sum(not o["valid"] for o in objects),
    }
    by_chat = defaultdict(list)
    for row in output["chats"]:
        by_chat[row["id"]].append(row)
    for rows in by_chat.values():
        if len(rows) > 1:
            for row in rows:
                row["issues"].append(
                    issue("MULTIPLE_ABIS", f"This chat has snapshots in {len(rows)} ABI namespaces")
                )
    for row in output["chats"]:
        complete_copies = [d for d in output["duplicate_snapshots"] if d["chat_id"] == row["id"]]
        if complete_copies:
            row["issues"].append(
                issue(
                    "DUPLICATE_SNAPSHOTS",
                    f"{len(complete_copies)} set(s) of generation directories "
                    "have identical object manifests",
                )
            )
        candidates = [d for d in output["duplicates"] if d["chat_id"] == row["id"]]
        if candidates:
            row["issues"].append(
                issue(
                    "DUPLICATE_CONTENT",
                    f"{len(candidates)} block-content duplicate candidate(s); see audit details",
                )
            )
    if output["storage"]["legacy"]["files"]:
        problems.append(
            issue("UNASSIGNED_LEGACY", "Unlabelled legacy blocks have no reliable chat ownership")
        )
    filesystem = os.statvfs(root)
    output["filesystem"] = {
        "available_bytes": filesystem.f_bavail * filesystem.f_frsize,
        "total_bytes": filesystem.f_blocks * filesystem.f_frsize,
    }
    output["drive"] = drive_health(root)
    output["scheduler"] = fair_scheduler_status()
    output["tail_journal"] = snapshot_tail_status()
    tracked = [row["io"] for row in output["chats"] if row["io"].get("available")]
    output["io"] = {
        "tracked_chats": len(tracked),
        "untracked_chats": len(output["chats"]) - len(tracked),
        "written_file_bytes": sum(item.get("written_file_bytes", 0) for item in tracked),
        "written_blocks": sum(item.get("written_blocks", 0) for item in tracked),
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--abi")
    parser.add_argument("--stale-after", type=float, default=300)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.abi and not ID.fullmatch(args.abi):
        parser.error("--abi must be a SHA256 identifier")
    try:
        result = scan(
            args.cache_root, abi=args.abi, stale_after=args.stale_after, verify=args.verify
        )
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
