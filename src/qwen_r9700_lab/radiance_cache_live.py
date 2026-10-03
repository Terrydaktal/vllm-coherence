"""Live cache dashboard consuming the same singleton tmpfs telemetry as Pi."""

from __future__ import annotations

import curses
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import threading
import time
from pathlib import Path

SCHEMA = "urn:qwen-r9700:telemetry:v1"
REFRESH_SECONDS = 0.1
INVENTORY_SECONDS = 30.0
STALE_MS = 5_000
ROOT = Path(__file__).resolve().parents[2]
PHASES = {
    "admission": "Preparing response",
    "gpu_queue": "Queued for GPU",
    "priority_wait": "Waiting for higher-priority chat",
    "priority_preempt": "Priority takeover requested",
    "tool_grace": "Waiting for another chat's tool",
    "cache_lookup": "Checking reusable context",
    "cache_update": "Finishing previous cache update",
    "handover": "Moving cache to GPU",
    "ram_allocation": "Allocating handover RAM",
    "cache_restore": "Loading cached context",
    "prefill": "Prompt prefill",
    "generate": "Generating",
    "complete": "Response complete",
}


def count(value):
    return type(value) is int and 0 <= value <= 2**53 - 1


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def identity(row):
    from .radiance_cache_audit import ID

    if isinstance(row, dict) and all(
        ID.fullmatch(str(row.get(key, ""))) for key in ("chat_id", "generation")
    ):
        return row["chat_id"], row["generation"]
    return None


def private_directory(path, *, create=False):
    if create:
        path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o777 != 0o700:
        raise ValueError(f"unsafe telemetry directory: {path}")


def read_sample(path, now_ms=None):
    """Read a bounded owner-only atomic snapshot; never follow a sample symlink."""
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_nlink != 1 or info.st_mode & 0o022 or info.st_size > 256 * 1024
        ):
            raise ValueError("unsafe combined telemetry sample")
        data = stream.read(256 * 1024 + 1)
    if len(data) > 256 * 1024:
        raise ValueError("oversized combined telemetry sample")
    sample = json.loads(data)
    if (
        not isinstance(sample, dict) or sample.get("schema") != SCHEMA
        or not count(sample.get("observed_at_ms"))
        or not -1_000 <= now_ms - sample["observed_at_ms"] <= STALE_MS
        or not isinstance(sample.get("scheduler"), dict)
        or any(sample.get(key) is not None and not isinstance(sample[key], dict)
               for key in ("worker", "cache", "phases", "temperature"))
    ):
        raise ValueError("combined telemetry unavailable or stale")
    scheduler = sample["scheduler"]
    if not count(scheduler.get("pid")) or not isinstance(scheduler.get("requests"), list):
        raise ValueError("invalid scheduler telemetry")
    worker = sample.get("worker")
    if worker and (
        not count(worker.get("pid")) or not count(worker.get("allocated_bytes"))
        or not count(worker.get("cached_chats"))
    ):
        raise ValueError("invalid worker telemetry")
    cache = sample.get("cache")
    if cache:
        from .radiance_cache_audit import ID

        if (
            cache.get("schema") != "urn:qwen-r9700:cache-residency:v2"
            or not ID.fullmatch(str(cache.get("abi", "")))
            or type(cache.get("live")) is not bool or type(cache.get("complete")) is not bool
            or not isinstance(cache.get("chats"), list) or len(cache["chats"]) > 272
            or any(not identity(row) or any(
                row.get(key) is not None and not count(row[key])
                for key in ("gpu_tokens", "ram_tokens", "disk_tokens", "disk_saved_tokens", "input_tokens")
            ) for row in cache["chats"])
            or len({identity(row) for row in cache["chats"]}) != len(cache["chats"])
        ):
            raise ValueError("invalid cache residency telemetry")
    phases = sample.get("phases")
    if phases and any(
        not isinstance(phases.get(key), list) or len(phases[key]) > 16
        or any(not identity(row) for row in phases[key]) for key in ("requests", "recent")
    ):
        raise ValueError("invalid request phase telemetry")
    return sample


def shared_state_directory(runtime, host):
    """Join an existing Pi reader for the same implicit/explicit SSH user."""
    root = runtime / "qwen-radiance-gpu-temperature"
    original = root / hashlib.sha256(host.encode()).hexdigest()
    if "@" in host or host in ("local", "localhost", "127.0.0.1"):
        return original
    try:
        # Config inspection makes no connection and runs only at startup.
        # `ai` and `lewis@ai` must not create separate hardware/metadata probes
        # when Pi already has a valid reader for that effective SSH user.
        config = subprocess.run(["ssh", "-G", "--", host], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, check=True, timeout=1)
        if len(config.stdout) > 65_536:
            return original
        users = [line.split()[1] for line in config.stdout.splitlines()
                 if len(line.split()) == 2 and line.split()[0] == "user"]
        if len(users) != 1 or not re.fullmatch(r"[A-Za-z0-9_.-]+", users[0]):
            return original
        candidate = root / hashlib.sha256(f"{users[0]}@{host}".encode()).hexdigest()
        private_directory(candidate)
        private_directory(candidate / "clients")
        read_sample(candidate / "telemetry-v1.json")
        return candidate
    except (OSError, ValueError, subprocess.SubprocessError):
        return original



class SharedTelemetry:
    """Register one normal Pi consumer, sharing the existing locked monitors."""

    def __init__(self, args):
        configured = args.telemetry_state or os.environ.get("QWEN_RADIANCE_GPU_TEMPERATURE_STATE")
        self.explicit_state = bool(configured)
        self.runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        self.directory = Path(configured) if configured else shared_state_directory(self.runtime, args.host)
        self.host = args.host
        self.cache_root = args.cache_root
        self.marker = None
        self.last_heartbeat = -math.inf
        self.last_ensure = -math.inf
        self.error = None
        self.children = []

    def __enter__(self):
        try:
            if not self.directory.is_absolute():
                raise ValueError("telemetry state must be an absolute path")
            if self.explicit_state:
                private_directory(self.directory)
                private_directory(self.directory / "clients")
            else:
                private_directory(self.runtime)
                for path in (self.directory.parent, self.directory, self.directory / "clients"):
                    private_directory(path, create=True)
            ticks = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
            self.marker = self.directory / "clients" / f"{os.getpid()}-{ticks}-{secrets.token_hex(8)}.heartbeat"
            fd = os.open(self.marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(fd)
            self.tick()
        except (OSError, ValueError) as error:
            self.error = str(error)
        return self

    def tick(self):
        now = time.monotonic()
        if self.marker is None or self.error:
            return
        if now - self.last_heartbeat >= 5:
            os.utime(self.marker, follow_symlinks=False)
            self.last_heartbeat = now
        self.children = [child for child in self.children if child.poll() is None]
        if now - self.last_ensure >= 10:
            self.last_ensure = now
            environment = {**os.environ, "QWEN_RADIANCE_CACHE_ROOT": self.cache_root}
            # These helpers acquire the same locks used by Pi. Many dashboards
            # and Pi windows still have just one scheduler and one hwmon probe.
            for name, override in (
                ("qwen-radiance-scheduler-status", "QWEN_RADIANCE_SCHEDULER_HELPER"),
                ("qwen-radiance-gpu-temperature", "QWEN_RADIANCE_GPU_TEMPERATURE_HELPER"),
            ):
                helper = os.environ.get(override, str(ROOT / "scripts" / name))
                if not Path(helper).is_file():
                    continue  # A packaged reader can still use an existing feed.
                try:
                    self.children.append(subprocess.Popen(
                        [helper, "ensure", str(self.directory), self.host], env=environment,
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    ))
                except OSError:
                    pass  # Preserve a valid existing feed; retry on the slow timer.

    def read(self):
        try:
            self.tick()
            if self.error:
                return None, self.error
            return read_sample(self.directory / "telemetry-v1.json"), None
        except (OSError, ValueError) as error:
            return None, str(error)

    def __exit__(self, *_):
        if self.marker is not None:
            self.marker.unlink(missing_ok=True)
        for child in self.children:
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                child.terminate()


class Inventory:
    """One bounded background scan, never on the fast display path."""

    def __init__(self, args, collect):
        self.args, self.collect = args, collect
        self.report = None
        self.error = None
        self.completed_at = None
        self.started_at = -math.inf
        self.thread = None

    def refresh(self, *, force=False):
        if self.thread is not None and self.thread.is_alive():
            return
        if not force and time.monotonic() - self.started_at < self.args.inventory_interval:
            return
        self.started_at = time.monotonic()

        def capture():
            try:
                report = self.collect(self.args)
                self.completed_at = time.monotonic()
                self.report = report
                self.error = None
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                self.error = str(error)

        self.thread = threading.Thread(target=capture, name="cache-inventory", daemon=True)
        self.thread.start()


def phase_rows(sample, now_ms):
    phases = sample.get("phases") or {}
    scheduler = sample["scheduler"]
    if (
        phases.get("pid") != scheduler.get("pid") or not finite(phases.get("updated_at"))
        or not -5_000 <= now_ms - phases["updated_at"] * 1000 <= 30_000
    ):
        return {}, {}
    active, recent = {}, {}
    for name, target in (("recent", recent), ("requests", active)):
        for row in phases.get(name, [])[:16]:
            key = identity(row)
            if key:
                target[key] = row
    return active, recent


def cache_breakdown(cache, row, context):
    """The same coverage partition as Pi's cacheBreakdown (disk is backup)."""
    absent = cache.get("complete") is True and cache.get("live") is True
    gpu_prefix = (row.get("gpu_tokens") if row else 0 if absent else None) if cache.get("live") else None
    ram_prefix = (row.get("ram_tokens") if row else 0 if absent else None) if cache.get("live") else None
    disk_prefix = row.get("disk_tokens") if row else 0 if cache.get("complete") else None
    input_tokens = row.get("input_tokens") if row else None
    total = max(
        context if count(context) else 0, input_tokens if count(input_tokens) else 0,
        gpu_prefix or 0 if count(input_tokens) else 0, ram_prefix or 0 if count(input_tokens) else 0,
    ) if count(context) or count(input_tokens) else None
    clamp = lambda v: None if v is None else v if total is None else min(v, total)
    gpu = clamp(gpu_prefix)
    ram = None if gpu is None else 0 if total is not None and gpu >= total else (
        None if ram_prefix is None else max(0, clamp(ram_prefix) - gpu)
    )
    memory = None if gpu is None or ram is None else gpu + ram
    disk = None if memory is None else 0 if total is not None and memory >= total else (
        None if disk_prefix is None else max(0, clamp(disk_prefix) - memory)
    )
    known = max(gpu_prefix or 0, ram_prefix or 0, disk_prefix or 0)
    cold = None if total is None else 0 if known >= total else (
        None if memory is None or disk is None else total - memory - disk
    )
    saved = row.get("disk_saved_tokens") if row else 0 if cache.get("complete") else None
    return {"gpu": gpu, "ram": ram, "disk_saved": saved, "cold": cold, "context": total}


def build_rows(report, sample, *, selector=None, abi=None, now_ms=None):
    from . import radiance_cache_cli as cli

    now_ms = time.time() * 1000 if now_ms is None else now_ms
    report = report or {"chats": [], "unsnapshotted_chats": []}
    records = {}
    for row in report["chats"]:
        if not abi or row["abi"] == abi:
            records[row["id"]] = {
                "id": row["id"], "abi": row["abi"],
                "title": cli.title(row) or row["metadata"].get("title") or row["id"][:12],
                "generation": (row.get("session") or {}).get("generation") or row["metadata"].get("generation"),
                "context": (row.get("session") or {}).get("last_turn_tokens"),
                "published": row["metadata"].get("tokens"),
                "disk_bytes": row["totals"]["file_bytes"],
                "traffic_bytes": row.get("io", {}).get("written_file_bytes") if row.get("io", {}).get("available") else None,
                "state": cli.health(row), "active_processes": row.get("active_processes", []),
                "session": row.get("session") or {}, "cwd": row["metadata"].get("cwd", ""),
            }
    for row in report["unsnapshotted_chats"]:
        records.setdefault(row["id"], {
            "id": row["id"], "abi": None, "title": row["title"], "generation": row["generation"],
            "context": row.get("last_turn_tokens"), "published": 0, "disk_bytes": 0,
            "traffic_bytes": None, "state": "NO SNAPSHOT", "active_processes": row.get("active_processes", []),
            "session": row, "cwd": row.get("cwd", ""),
        })
    current, recent, cache = {}, {}, {}
    if sample:
        current, recent = phase_rows(sample, now_ms)
        cache = sample.get("cache") or {}
        if (
            cache.get("schema") != "urn:qwen-r9700:cache-residency:v2"
            or not count(cache.get("observed_at_ms"))
            or not -1_000 <= now_ms - cache["observed_at_ms"] <= STALE_MS
            or (abi and cache.get("abi") != abi)
        ):
            cache = {}
        for source in (cache.get("chats", []), list(current.values())):
            for row in source:
                key = identity(row)
                if key:
                    # An older probe can still publish purged markers as empty
                    # cache rows. Absence of data/context is not live activity;
                    # retain real inventory rows and active requests separately.
                    if (
                        key not in current and row.get("input_tokens") in (None, 0)
                        and all(count(row.get(field)) and row[field] == 0 for field in (
                            "gpu_tokens", "ram_tokens", "disk_tokens", "disk_saved_tokens",
                        ))
                    ):
                        continue
                    records.setdefault(key[0], {
                        "id": key[0], "abi": cache.get("abi"), "title": key[0][:12],
                        "generation": key[1], "context": row.get("input_tokens"), "published": None,
                        "disk_bytes": None, "traffic_bytes": None, "state": "LIVE",
                        "active_processes": [], "session": {}, "cwd": "",
                    })
    cached = {identity(row): row for row in cache.get("chats", []) if identity(row)}
    for row in records.values():
        # Follow an active new generation immediately after compaction, rather
        # than borrowing an old disk head or a prior request's round statistics.
        active = next((p for key, p in current.items() if key[0] == row["id"]), None)
        if active:
            if row["generation"] != active["generation"] or not count(row["context"]):
                row["context"] = active.get("input_tokens")
            row["generation"] = active["generation"]
        else:
            keys = [key for key in cached if key[0] == row["id"]]
            if len(keys) == 1 and row["generation"] != keys[0][1]:
                row["generation"] = keys[0][1]
                row["context"] = cached[keys[0]].get("input_tokens")
        key = row["id"], row["generation"]
        phase = current.get(key) or recent.get(key)
        cache_row = cached.get(key)
        row.update(cache_breakdown(cache, cache_row, row["context"]))
        if cache_row is not None:
            row["published"] = row["disk_saved"]
        row["round_ms"], row["acceptance"] = None, None
        if phase:
            if finite(phase.get("last_round_ms")) and phase["last_round_ms"] >= 0:
                row["round_ms"] = phase["last_round_ms"]
            if finite(phase.get("acceptance_rate_3s")) and 0 <= phase["acceptance_rate_3s"] <= 1:
                row["acceptance"] = phase["acceptance_rate_3s"]
            if active:
                name = PHASES.get(phase.get("phase"), "Backend preparation")
                blocker = identity(phase.get("blocker"))
                if blocker:
                    name += " · " + ("earlier request" if blocker[0] == row["id"] else blocker[0][:12])
                row["state"] = name
        # The disk byte counts belong to the slower inventory. Restore coverage
        # is never borrowed across a different ABI, including an explicit filter.
        if sample and row["abi"] and row["abi"] != cache.get("abi"):
            row.update(gpu=None, ram=None, disk_saved=None, cold=None, round_ms=None, acceptance=None)
    result = list(records.values())
    if selector:
        result = [row for row in result if row["id"].startswith(selector.casefold()) or any(
            selector.casefold() in str(row.get(field, "")).casefold() for field in ("title", "cwd")
        ) or selector.casefold() in str(row["session"].get("session_id", "")).casefold()]
    return sorted(result, key=lambda row: (row["id"] not in {key[0] for key in current}, row["title"], row["id"]))


def dashboard_lines(report, sample, error, inventory, args, *, issues=False):
    from . import radiance_cache_cli as cli

    now_ms = time.time() * 1000
    live = f"live · sample {(now_ms - sample['observed_at_ms']) / 1000:.1f}s old" if sample else "telemetry unavailable"
    lines = [f"Radiance cache on {cli.clean(args.host)} · {live} · {args.interval * 1000:g} ms refresh"]
    temp = sample.get("temperature") if sample else None
    if (
        isinstance(temp, dict) and count(temp.get("observed_at_ms"))
        and -1_000 <= now_ms - temp["observed_at_ms"] <= 15_000
        and count(temp.get("junction_millicelsius")) and count(temp.get("edge_millicelsius"))
        and count(temp.get("fan_percent"))
    ):
        lines.append(f"{temp['junction_millicelsius'] / 1000:.0f}°C · {temp['edge_millicelsius'] / 1000:.0f}°C · {temp['fan_percent']}%")
    else:
        lines.append("Temperature sample unavailable")
    worker = sample.get("worker") or {} if sample else {}
    worker = worker if sample and worker.get("pid") == sample["scheduler"].get("pid") else {}
    residency = worker.get("residency")
    images = residency.get("images") if isinstance(residency, dict) else None
    parked = (
        f"{len(images)} parked chat(s)"
        if isinstance(images, list) and all(identity(row) for row in images)
        else f"{worker.get('cached_chats', 0)} cached chat(s)"
    )
    lines.append(
        f"Handover RAM: {cli.human(worker['allocated_bytes'])} allocated · {parked}"
        if count(worker.get("allocated_bytes")) else "Handover RAM: telemetry unavailable"
    )
    if report:
        age = time.monotonic() - inventory.completed_at
        traffic = report["io"].get("lifetime", report["io"])
        lines.append(f"Files {cli.human(report['totals']['file_bytes'])} · lifetime disk traffic {cli.human(traffic['written_file_bytes'])}"
                     f" ({cli.human(traffic.get('deleted_written_file_bytes', 0))} from deleted chats)"
                     f" · filesystem free {cli.human(report['filesystem']['available_bytes'])}")
        if traffic.get("complete") is False or traffic.get("untracked_chats"):
            lines.append("Write history incomplete: lifetime total covers recorded completed payload writes.")
        lines.append(f"Disk inventory {age:.1f}s old · disk sizes, traffic and audit refresh every {args.inventory_interval:g}s"
                     + (" · scanning" if inventory.thread and inventory.thread.is_alive() else ""))
    else:
        lines.extend(["Loading disk inventory…", "Live counters do not wait for the disk scan."])
    if error:
        lines.append(f"Telemetry: {cli.clean(error)}")
    if inventory.error:
        lines.append(f"Disk inventory: {cli.clean(inventory.error)}")
    lines.append("")
    headings = (
        f"{'CHAT / ABI':22} {'GPU tok':>9} {'RAM tok':>9} {'Disk tok':>9} {'Cold tok':>9}"
        f" {'Round ms':>9} {'Accept(3s)':>11} {'Disk bytes':>11} {'Traffic':>11} {'PID/PORT':17}  STATE / CHAT"
    )
    lines.append(headings)
    header = len(lines)
    for row in build_rows(report, sample, selector=args.chat, abi=args.abi, now_ms=now_ms):
        size = lambda value: cli.human(value) if value is not None else "—"
        num = lambda value: cli.number(value) if value is not None else "—"
        round_ms = f"{row['round_ms']:.1f}" if row['round_ms'] is not None else "—"
        acceptance = f"{row['acceptance'] * 100:.1f}%" if row['acceptance'] is not None else "—"
        lines.append(
            f"{row['id'][:12]}/{(row['abi'] or '-')[:8]:8} "
            f"{num(row['gpu']):>9} {num(row['ram']):>9} {num(row['disk_saved']):>9} {num(row['cold']):>9}"
            f" {round_ms:>9} {acceptance:>11} {size(row['disk_bytes']):>11} {size(row['traffic_bytes']):>11}"
            f" {cli.active(row['active_processes']):17}  {cli.clean(row['state'])} · {cli.clean(row['title'])}"
        )
    problems = [*(report or {}).get("issues", []), *(
        problem for row in (report or {}).get("chats", []) for problem in row["issues"]
    )]
    if issues:
        lines.append("")
        lines.extend(f"{p['severity'].upper()} {cli.clean(p['code'])}: {cli.clean(p['message'])}" for p in problems)
    footer = f"q/Esc quit · ↑↓/PgUp/PgDn scroll · ←→ columns · r rescan · i issues ({len(problems)}) · Disk tok = saved backup"
    return lines, header, footer


def terminal(screen, args, reader, inventory):
    """Curses updates changed cells; terminal state is restored by wrapper."""
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    screen.timeout(0)
    scroll, horizontal, issues, frames = 0, 0, False, 0
    while True:
        started = time.monotonic()
        inventory.refresh()
        sample, error = reader.read()
        lines, header, footer = dashboard_lines(inventory.report, sample, error, inventory, args, issues=issues)
        height, width = screen.getmaxyx()
        body_height = max(1, height - header - 1)
        scroll = min(scroll, max(0, len(lines) - header - body_height))
        horizontal = min(horizontal, max(0, max(map(len, lines), default=0) - max(1, width - 1)))
        visible = lines[:header] + lines[header + scroll:header + scroll + body_height]
        screen.erase()
        for y, line in enumerate(visible[:max(0, height - 1)]):
            try:
                screen.addnstr(y, 0, line[horizontal:] if y >= header - 1 else line,
                               max(0, width - 1), curses.A_BOLD if y in (0, header - 1) else curses.A_NORMAL)
            except curses.error:
                pass  # A concurrent resize may invalidate the previous bounds.
        try:
            screen.addnstr(max(0, height - 1), 0, footer, max(0, width - 1), curses.A_REVERSE)
        except curses.error:
            pass
        try:
            screen.noutrefresh()
            curses.doupdate()
        except curses.error:
            pass
        frames += 1
        if args.count and frames >= args.count:
            return 0
        # Account for work already done, rather than adding it to every tick.
        screen.timeout(max(1, int((args.interval - (time.monotonic() - started)) * 1000)))
        key = screen.getch()
        if key in (ord("q"), ord("Q"), 27):
            return 0
        if key == curses.KEY_DOWN:
            scroll += 1
        elif key == curses.KEY_UP:
            scroll = max(0, scroll - 1)
        elif key == curses.KEY_NPAGE:
            scroll += body_height
        elif key == curses.KEY_PPAGE:
            scroll = max(0, scroll - body_height)
        elif key == curses.KEY_RIGHT:
            horizontal += 8
        elif key == curses.KEY_LEFT:
            horizontal = max(0, horizontal - 8)
        elif key == ord("r"):
            inventory.refresh(force=True)
        elif key == ord("i"):
            issues = not issues


def watch(args, collect):
    """TTY dashboard or bounded newline-JSON stream; both share the same feed."""
    import sys

    with SharedTelemetry(args) as reader:
        inventory = Inventory(args, collect)
        inventory.refresh()
        if not args.json and sys.stdout.isatty() and sys.stdin.isatty():
            try:
                return curses.wrapper(terminal, args, reader, inventory)
            except curses.error as error:
                raise ValueError(f"cannot initialize live terminal: {error}; use --once") from error
        frames = 0
        while True:
            started = time.monotonic()
            inventory.refresh()
            sample, error = reader.read()
            if args.json:
                print(json.dumps({
                    "schema": "urn:qwen-r9700:cache-live:v1", "host": args.host,
                    "observed_at_ms": int(time.time() * 1000), "telemetry": sample,
                    "telemetry_error": error, "inventory": inventory.report,
                    "inventory_error": inventory.error,
                    "chats": build_rows(inventory.report, sample, selector=args.chat, abi=args.abi),
                }, separators=(",", ":")), flush=True)
            else:
                lines, _, footer = dashboard_lines(inventory.report, sample, error, inventory, args)
                print("\n".join([*lines, footer]), flush=True)
            frames += 1
            if args.count and frames >= args.count:
                return 0
            time.sleep(max(0, args.interval - (time.monotonic() - started)))
