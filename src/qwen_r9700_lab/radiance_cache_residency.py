"""Small, read-only shared probe; transports identities and counters, never chat text.

This file is also streamed to the backend's stdlib Python over the existing SSH
monitor connection. It must not import project or model-runtime packages.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path


class StatusChanges:
    """One event-driven tmpfs watch for all Pi windows; no faster polling."""

    def __init__(self, directory="/dev/shm"):
        self.fd = -1
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if (
                self.fd < 0
                or libc.inotify_add_watch(
                    self.fd,
                    os.fsencode(directory),
                    0x80 | 0x08,  # MOVED_TO | CLOSE_WRITE
                )
                < 0
            ):
                self.close()
        except (OSError, AttributeError):
            self.close()

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def wait(self, timeout):
        if self.fd < 0:
            time.sleep(timeout)
            return
        import select

        names = {
            b"qwen-radiance-fair-public-scheduler.json",
            b"qwen-radiance-fair-public-worker.json",
            b"qwen-radiance-fair-public-phases.json",
            # The round stream is the authoritative source for the latest
            # per-round acceptance.  Phase publication can be overwritten by
            # a final target-only/EOS round, so wake the reader for appends too.
            b"qwen-radiance-fair-public-rounds.jsonl",
        }
        deadline = time.monotonic() + timeout
        while select.select([self.fd], [], [], max(0, deadline - time.monotonic()))[0]:
            data = os.read(self.fd, 65536)
            offset, changed = 0, False
            while offset + 16 <= len(data):
                _, mask, _, size = struct.unpack_from("iIII", data, offset)
                name = data[offset + 16 : offset + 16 + size].split(b"\0", 1)[0]
                changed |= name in names or bool(mask & 0x4000)  # queue overflow: resample
                offset += 16 + size
            if changed:
                return
            if time.monotonic() >= deadline:
                return


SCHEMA = "urn:qwen-r9700:cache-residency:v2"
LEGACY_SCHEMA = "urn:qwen-r9700:cache-residency:v1"
HEX = re.compile(r"[0-9a-f]{64}\Z")
OBJECT = re.compile(r"g[0-9]+-[0-9a-f]+\.qkv\Z")
DEFAULT_ROOT = "/home/lewis/.cache/qwen-radiance-public-clean-snapshot-v1"
DEFAULT_ABI = "74aef30706ffab186ee2fb89d3827c9895496918bdfb3d61188196daeced5b94"
BACKEND_CONTAINER = "vllm-coherence"


def backend_namespace(metadata, root):
    """Authenticate the active cache path from serving arguments, never environment/payloads."""
    if not isinstance(metadata, dict) or metadata.get("running") is not True:
        raise ValueError("backend is not running")
    mounts = [
        row
        for row in metadata.get("mounts", [])
        if isinstance(row, dict) and row.get("Destination") == "/cache"
    ]
    if len(mounts) != 1 or mounts[0].get("Source") != str(Path(root)):
        raise ValueError("backend cache mount differs")
    args = metadata["args"]
    if not isinstance(args, list) or args.count("--kv-transfer-config") != 1:
        raise ValueError("backend cache configuration is ambiguous")
    config = json.loads(args[args.index("--kv-transfer-config") + 1])
    if not isinstance(config, dict) or config.get("kv_connector") != "OffloadingConnector":
        raise ValueError("backend cache connector differs")
    tiers = config["kv_connector_extra_config"]["secondary_tiers"]
    paths = [
        row.get("root_dir")
        for row in tiers
        if isinstance(row, dict) and row.get("type") == "qwen_chat_fs"
    ]
    if len(paths) != 1 or not isinstance(paths[0], str):
        raise ValueError("backend snapshot path is ambiguous")
    match = re.fullmatch(r"/cache/snapshots/([0-9a-f]{64})/data", paths[0])
    if match is None:
        raise ValueError("backend snapshot path differs")
    return match[1]


def inspect_backend_namespace(root):
    result = subprocess.run(
        [
            "podman",
            "inspect",
            "--format",
            '{"running":{{json .State.Running}},"args":{{json .Args}},"mounts":{{json .Mounts}}}',
            BACKEND_CONTAINER,
        ],
        capture_output=True,
        check=True,
        timeout=2,
    )
    if len(result.stdout) > 65_536:
        raise ValueError("oversized backend metadata")
    return backend_namespace(json.loads(result.stdout), root)


class LiveCacheNamespace:
    """Inspect once per backend instance; all Pi windows share the result and probe."""

    def __init__(self, root):
        self.root = root
        self.instance = None
        self.abi = None
        self.last_attempt = None

    def read(self):
        report = optional_json("/dev/shm/qwen-radiance-memory-v1/report.json") or {}
        instance = report.get("instance_id")
        if not isinstance(instance, str) or not re.fullmatch(r"[0-9a-f]{32}", instance):
            instance = None
        if instance != self.instance:
            self.instance, self.abi, self.last_attempt = instance, None, None
        now = time.monotonic()
        if self.abi is not None and instance is not None:
            return self.abi
        if self.last_attempt is not None and now - self.last_attempt < 10:
            return self.abi
        self.last_attempt = now
        try:
            self.abi = inspect_backend_namespace(self.root)
        except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError):
            self.abi = None
        return self.abi


def count(value):
    return type(value) is int and 0 <= value <= 2**53 - 1


def identity(value):
    if (
        isinstance(value, dict)
        and HEX.fullmatch(str(value.get("chat_id", "")))
        and HEX.fullmatch(str(value.get("generation", "")))
    ):
        return value["chat_id"], value["generation"]
    return None


def read_json(path, limit=65_536):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("invalid metadata file")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(data) > limit or (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("metadata changed while reading")
    return json.loads(data)


def optional_json(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def _round_acceptance_event(value):
    """Return only safe numeric round telemetry, never payload text."""
    if not isinstance(value, dict) or value.get("schema") != "urn:qwen-r9700:decode-rounds:v1":
        return None
    identity_value = identity(value)
    pid = value.get("pid")
    observed_at_ms = value.get("observed_at_ms")
    draft_tokens = value.get("draft_tokens")
    accepted_tokens = value.get("accepted_tokens")
    round_ms = value.get("round_ms")
    rate = value.get("acceptance_rate")
    if (
        identity_value is None
        or type(pid) is not int
        or pid <= 0
        or type(observed_at_ms) is not int
        or observed_at_ms <= 0
        or type(draft_tokens) is not int
        or draft_tokens < 0
        or type(accepted_tokens) is not int
        or accepted_tokens < 0
        or accepted_tokens > draft_tokens
        or type(round_ms) not in (int, float)
        or not math.isfinite(round_ms)
        or round_ms < 0
    ):
        return None
    if draft_tokens and (
        type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1
    ):
        return None
    return identity_value, pid, observed_at_ms, draft_tokens, accepted_tokens


class RoundAcceptance:
    """Incrementally read the content-free decode-round telemetry stream."""

    PATH = "/dev/shm/qwen-radiance-fair-public-rounds.jsonl"
    MAX_INITIAL_READ = 8 * 1024 * 1024
    WINDOW_MS = 3_000

    def __init__(self, path=PATH):
        self.path = path
        self.inode = None
        self.offset = 0
        self.pending = b""
        self.latest = {}

    def _consume(self, data):
        complete, separator, pending = data.rpartition(b"\n")
        self.pending = pending if separator else data
        for line in complete.splitlines() if separator else ():
            try:
                event = _round_acceptance_event(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if event is not None:
                identity_value, pid, observed_at_ms, draft_tokens, accepted_tokens = event
                rows = self.latest.setdefault(identity_value, [])
                rows.append((pid, observed_at_ms, draft_tokens, accepted_tokens))
                cutoff = observed_at_ms - self.WINDOW_MS
                self.latest[identity_value] = [row for row in rows if row[1] >= cutoff]

    def _read_rotated(self):
        """Read the bounded predecessor so a recent chat survives log rotation."""
        try:
            fd = os.open(str(self.path) + ".1", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            return
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > self.MAX_INITIAL_READ:
                return
            data = stream.read(self.MAX_INITIAL_READ + 1)
            after = os.fstat(stream.fileno())
        if before.st_ino == after.st_ino and before.st_size == after.st_size:
            self._consume(data)

    def read(self):
        initial = self.inode is None
        if initial:
            self._read_rotated()
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            return self.latest
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                return self.latest
            if initial or self.inode != before.st_ino or before.st_size < self.offset:
                # The backend rotates the bounded log by rename.  The current
                # file is authoritative for new events, while its predecessor
                # retains recent rows that may belong to another chat.
                self.pending = b""
                if not initial:
                    self._read_rotated()
                self.inode = before.st_ino
                self.offset = 0
            if self.offset == 0 and before.st_size > self.MAX_INITIAL_READ:
                stream.seek(before.st_size - self.MAX_INITIAL_READ)
                self.offset = before.st_size - self.MAX_INITIAL_READ
            else:
                stream.seek(self.offset)
            data = stream.read()
            after = os.fstat(stream.fileno())
        if before.st_ino != after.st_ino or before.st_size != after.st_size:
            # An append raced the read.  Leave the offset unchanged and retry
            # on the next event rather than accepting a partial record.
            return self.latest
        self.offset = after.st_size
        data = self.pending + data
        complete, separator, pending = data.rpartition(b"\n")
        self.pending = pending if separator else data
        for line in complete.splitlines() if separator else ():
            try:
                event = _round_acceptance_event(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if event is not None:
                identity_value, pid, observed_at_ms, draft_tokens, accepted_tokens = event
                rows = self.latest.setdefault(identity_value, [])
                rows.append((pid, observed_at_ms, draft_tokens, accepted_tokens))
                cutoff = observed_at_ms - self.WINDOW_MS
                self.latest[identity_value] = [row for row in rows if row[1] >= cutoff]
        return self.latest

    def rolling_rate(self, identity_value, pid, at_ms):
        """Return weighted accepted/draft tokens in the same three-second window."""
        rows = self.latest.get(identity_value, [])
        cutoff = at_ms - self.WINDOW_MS
        eligible = [
            row for row in rows if row[0] == pid and cutoff <= row[1] <= at_ms
        ]
        self.latest[identity_value] = [row for row in rows if row[1] >= cutoff]
        draft_tokens = sum(row[2] for row in eligible)
        if not draft_tokens:
            return None
        return sum(row[3] for row in eligible) / draft_tokens


def restore_last_round_acceptance(phases, round_acceptance):
    """Attach weighted three-second acceptance from the numeric round feed."""
    if not isinstance(phases, dict) or not hasattr(round_acceptance, "rolling_rate"):
        return phases
    backend_pid = phases.get("pid")
    updated_at = phases.get("updated_at")
    at_ms = updated_at * 1000 if type(updated_at) in (int, float) and math.isfinite(updated_at) else None
    result = dict(phases)
    for key in ("requests", "recent"):
        rows = phases.get(key)
        if not isinstance(rows, list):
            continue
        copied = []
        for row in rows:
            if not isinstance(row, dict):
                copied.append(row)
                continue
            rate = (
                round_acceptance.rolling_rate(identity(row), backend_pid, int(at_ms))
                if at_ms is not None
                else None
            )
            copied.append({**row, "acceptance_rate_3s": rate})
        result[key] = copied
    return result


def fresh(value, now, max_age):
    updated = value.get("updated_at") if isinstance(value, dict) else None
    return (
        type(updated) in (int, float) and math.isfinite(updated) and -5 <= now - updated <= max_age
    )


def observe_idle_scheduler(scheduler, tail, now=None):
    """Keep an unchanged idle scheduler observable via its worker heartbeat.

    Scheduler publication is event-driven and stops between requests. The
    tail writer already publishes a same-process heartbeat during idle time.
    It cannot refresh a running/queued request or a different backend's state.
    No timestamps are written back to the backend's source files.
    """
    now = time.time() if now is None else now
    if (
        isinstance(scheduler, dict)
        and scheduler.get("requests") == []
        and count(scheduler.get("pid"))
        and scheduler["pid"] > 0
        and isinstance(tail, dict)
        and tail.get("pid") == scheduler["pid"]
        and fresh(tail, now, 5)
        and type(scheduler.get("updated_at")) in (int, float)
        and math.isfinite(scheduler["updated_at"])
        and tail["updated_at"] > scheduler["updated_at"]
    ):
        return {**scheduler, "updated_at": tail["updated_at"]}
    return scheduler


def legacy_sample(sample):
    """Keep already running Pi clients compatible with the shared probe."""
    if sample is None:
        return None
    return {
        **sample,
        "schema": LEGACY_SCHEMA,
        "chats": [
            {key: value for key, value in row.items() if key != "disk_saved_tokens"}
            for row in sample["chats"]
        ],
    }


class ResidencyProbe:
    def __init__(self, root, abi):
        self.abi = abi
        self.managed = Path(root) / "snapshots" / abi / "data" / "qwen-chat-cache-v1"
        self.manifests = {}
        self.progress = {}
        self.pid = None
        self.switches = None

    def disk_heads(self):
        """Validate publication fingerprints with stat, without reading KV payloads."""
        rows, complete, present = {}, True, set()
        try:
            if self.managed.is_symlink() or not self.managed.is_dir():
                return {}, False
            directories = [p for p in self.managed.iterdir() if HEX.fullmatch(p.name)]
            if len(directories) > 256:
                return {}, False
        except OSError:
            return {}, False
        for directory in directories:
            present.add(directory.name)
            key = None
            try:
                if directory.is_symlink() or not directory.is_dir():
                    raise ValueError("invalid chat directory")
                path = directory / "chat.json"
                info = path.lstat()
                signature = (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                cached = self.manifests.get(directory.name)
                if cached is None or cached[0] != signature:
                    metadata = read_json(path, 2 * 1024 * 1024)
                    cached = self.manifests[directory.name] = (signature, metadata)
                metadata = cached[1]
                key = identity(
                    {"chat_id": metadata.get("id"), "generation": metadata.get("generation")}
                )
                if key is None or key[0] != directory.name:
                    raise ValueError("invalid snapshot identity")
                rows[key] = None
                tokens, head = metadata.get("tokens"), metadata.get("head")
                if not count(tokens) or not isinstance(head, list) or len(head) > 4096:
                    raise ValueError("invalid snapshot head")
                if tokens == 0 and head == []:
                    rows[key] = 0
                    continue
                verified = metadata.get("verified_head", {})
                generation = directory / "generations" / key[1]
                if (
                    not head
                    or not isinstance(verified, dict)
                    or generation.is_symlink()
                    or generation.parent.is_symlink()
                ):
                    raise ValueError("unverified snapshot head")
                for name in head:
                    if not isinstance(name, str) or not OBJECT.fullmatch(name):
                        raise ValueError("invalid snapshot object")
                    details = (generation / name).lstat()
                    fingerprint = [
                        details.st_ino,
                        details.st_size,
                        details.st_mtime_ns,
                        details.st_ctime_ns,
                    ]
                    if not stat.S_ISREG(details.st_mode) or fingerprint != verified.get(name):
                        raise ValueError("snapshot object changed since publication")
                rows[key] = tokens
            except (OSError, ValueError, TypeError, AttributeError):
                # A broken head has its own unknown row. It must not hide the
                # known absence of a different chat from a complete inventory.
                if key not in rows:
                    complete = False
        self.manifests = {key: value for key, value in self.manifests.items() if key in present}
        return rows, complete

    def sample(self, scheduler, worker, tail, now=None, *, disk_inventory=None):
        now = time.time() if now is None else now
        disk, complete = self.disk_heads() if disk_inventory is None else disk_inventory
        pid = scheduler.get("pid") if isinstance(scheduler, dict) else None
        tail_live = isinstance(tail, dict) and tail.get("pid") == pid and fresh(tail, now, 15)
        live = count(pid) and pid > 0 and (fresh(scheduler, now, 5) or tail_live)
        if pid != self.pid or not live:
            self.progress.clear()
            self.pid = pid
            self.switches = None
        residency = (
            worker.get("residency", {})
            if isinstance(worker, dict) and worker.get("pid") == pid
            else {}
        )
        coherent = (
            live
            and isinstance(residency, dict)
            and isinstance(residency.get("images"), list)
            and count(scheduler.get("switches"))
            and count(worker.get("switches"))
            and worker.get("switches") == scheduler.get("switches")
        )
        active = identity(residency.get("active")) if coherent else None
        images = (
            {key for row in residency.get("images", []) if (key := identity(row))}
            if coherent
            else set()
        )
        banks = images | ({active} if active else set())
        if coherent:
            switches = worker["switches"]
            if self.switches is not None and switches not in (self.switches, self.switches + 1):
                # A missed eviction and reactivation can reuse a chat identity.
                # Do not carry a count across unobserved bank lifetimes.
                self.progress.clear()
            self.switches = switches
            # Admission and worker handover publish separate status files. Their
            # temporary disagreement is not evidence that any bank was evicted.
            # Hide uncertain residency during the transition, then retain counts
            # only for generations still present in the confirmed inventory.
            self.progress = {key: value for key, value in self.progress.items() if key in banks}
        tail_rows = tail.get("chats") if tail_live else None
        request_rows = scheduler.get("requests") if live else None
        tails = (
            {
                identity(row): row["tokens"]
                for row in tail_rows
                if identity(row) and count(row.get("tokens"))
            }
            if isinstance(tail_rows, list)
            else {}
        )
        requests = (
            {
                identity(row): row
                for row in request_rows
                if identity(row)
                and count(row.get("computed_tokens"))
                and count(row.get("input_tokens"))
            }
            if isinstance(request_rows, list)
            else {}
        )
        executing = {
            key
            for key, row in requests.items()
            if coherent
            and (
                (row.get("state") == "running" and key == active)
                or (row.get("state") == "paused" and key in images)
            )
        }
        for key in executing:
            # Keep the last observed computed count across idle periods, but only
            # while that exact generation still has a physical cache bank.
            # Queued async restores reserve this count before their KV is loaded.
            self.progress[key] = requests[key]["computed_tokens"]
        rows = []
        for key in sorted(disk.keys() | tails.keys() | requests.keys() | banks):
            settled = max(disk.get(key) or 0, tails.get(key, 0))
            row = requests.get(key)
            known = max(settled, self.progress.get(key, 0)) or None
            if key in executing:
                # Zero is authoritative once prefix lookup has completed and
                # execution begins, including a genuine miss after a warm turn.
                known = row["computed_tokens"]
            elif row is not None and row.get("state") == "queued" and key in banks:
                # Each HTTP request starts at zero before prefix lookup. The
                # sticky last_handover label does not mean this bank is empty
                # again. Preserve confirmed residency across tool continuations;
                # a larger async-restore reservation is not additional GPU data.
                known = self.progress.get(key)
                if known is not None and row["computed_tokens"] > 0:
                    known = min(known, row["computed_tokens"])
            gpu = known if key == active else 0
            ram = known if key in images else 0
            # Durable backup coverage is independent of residency, the buffered
            # tail, and whether this snapshot matches the current request prefix.
            disk_saved_tokens = disk.get(key, 0 if complete else None)
            # The buffered tail can extend a restore that still needs disk
            # attention blocks. Do not call that complete RAM coverage.
            disk_tokens = (
                max(disk[key], tails.get(key, 0))
                if disk.get(key) is not None
                else (0 if key not in disk and complete else None)
            )
            if key in executing:
                # vLLM finishes the initial prefix restore before GPU computation;
                # FairScheduler parks only running requests after that boundary.
                # Any remaining prompt tokens therefore require prefill. An old
                # snapshot for this generation need not match the resumed prompt.
                disk_tokens = min(disk_tokens or 0, row["computed_tokens"])
            rows.append(
                {
                    "chat_id": key[0],
                    "generation": key[1],
                    "gpu_tokens": gpu if coherent else None,
                    "ram_tokens": ram if coherent else None,
                    "disk_tokens": disk_tokens,
                    "disk_saved_tokens": disk_saved_tokens,
                    "input_tokens": row["input_tokens"] if row else None,
                }
            )
        return {
            "schema": SCHEMA,
            "observed_at_ms": int(now * 1000),
            "abi": self.abi,
            "live": bool(coherent),
            "complete": complete,
            "chats": rows,
        }


def main():
    global BACKEND_CONTAINER
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--once", action="store_true")
    options.add_argument("--container", default=BACKEND_CONTAINER)
    options = options.parse_args(sys.argv[4:])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", options.container):
        raise SystemExit("invalid backend container")
    BACKEND_CONTAINER = options.container
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    root = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ROOT
    abi = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_ABI
    if (
        not 0.1 <= interval <= 10
        or not Path(root).is_absolute()
        or not (abi == "auto" or HEX.fullmatch(abi))
    ):
        raise SystemExit("invalid probe configuration")
    namespace = LiveCacheNamespace(root) if abi == "auto" else None
    probe = None if namespace else ResidencyProbe(root, abi)
    changes = StatusChanges()
    round_acceptance = RoundAcceptance()
    disk_inventory, last_disk_check = None, 0.0
    while True:
        scheduler = optional_json("/dev/shm/qwen-radiance-fair-public-scheduler.json")
        worker = optional_json("/dev/shm/qwen-radiance-fair-public-worker.json")
        phases = optional_json("/dev/shm/qwen-radiance-fair-public-phases.json")
        if namespace:
            selected = namespace.read()
            if selected != (probe.abi if probe else None):
                probe = ResidencyProbe(root, selected) if selected else None
                disk_inventory, last_disk_check = None, 0.0
        # Memory counters follow the half-second shared feed; disk verification
        # retains its existing one-second cadence and cost across all Pi windows.
        if probe and (disk_inventory is None or time.monotonic() - last_disk_check >= 1):
            disk_inventory = probe.disk_heads()
            last_disk_check = time.monotonic()
        tail = optional_json("/dev/shm/qwen-radiance-snapshot-tail.json")
        scheduler = observe_idle_scheduler(scheduler, tail)
        round_acceptance.read()
        phases = restore_last_round_acceptance(phases, round_acceptance)
        coverage = (
            probe.sample(scheduler, worker, tail, disk_inventory=disk_inventory) if probe else None
        )
        print(
            "\t".join(
                json.dumps(value, separators=(",", ":"))
                for value in (scheduler, worker, legacy_sample(coverage), coverage, phases)
            ),
            flush=True,
        )
        if options.once:
            changes.close()
            break
        changes.wait(interval)


if __name__ == "__main__":
    with suppress(BrokenPipeError):
        main()
