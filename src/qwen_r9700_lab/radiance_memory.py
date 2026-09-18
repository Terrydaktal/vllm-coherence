"""Shared GPU allocation accounting; reads counters and tensor metadata only.

The worker starts this after warmup. No CUDA/HIP operations, allocator resets,
tensor values, garbage collection, or trace-history recording are used.
This module also runs over SSH using only the standard library to read reports.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import stat
import threading
import time
import weakref
from bisect import bisect_left
from pathlib import Path
from uuid import uuid4

DIRECTORY = Path("/dev/shm/qwen-radiance-memory-v1")
SCHEMA = "urn:qwen-r9700:radiance-memory:v1"
MAX_REPORT_BYTES = 65536
INTERVAL = 1.0
STALE_AFTER = 5.0
MAP_SCHEMA = "urn:qwen-r9700:radiance-allocation-map:v1"
MAP_INTERVAL = 10.0
MAP_ROWS = 20
FREE_BINS = (1, 16, 64, 128, 256, 512)
COUNTERS = {
    "allocated_bytes": "allocated_bytes.all.current",
    "allocated_peak_bytes": "allocated_bytes.all.peak",
    "reserved_bytes": "reserved_bytes.all.current",
    "reserved_peak_bytes": "reserved_bytes.all.peak",
    "active_bytes": "active_bytes.all.current",
    "active_peak_bytes": "active_bytes.all.peak",
    "inactive_split_bytes": "inactive_split_bytes.all.current",
    "inactive_split_peak_bytes": "inactive_split_bytes.all.peak",
    "requested_bytes": "requested_bytes.all.current",
    "allocation_retries": "num_alloc_retries",
    "out_of_memory_events": "num_ooms",
}
GROUPS = {
    "kv_cache": "GPU cache pool",
    "model": "Main model weights and buffers",
    "draft_model": "Draft model weights and buffers",
    "input_buffers": "Prompt input buffers",
    "req_states": "Request bookkeeping",
    "model_state": "Model working buffers",
    "sampler": "Sampling buffers",
    "rejection_sampler": "Speculative verification buffers",
    "prompt_logprobs_worker": "Prompt probability buffers",
    "structured_outputs_worker": "Structured output buffers",
    "kv_block_zeroer": "Cache initialization buffers",
    "speculator": "Draft working buffers",
}


def natural(value):
    return type(value) is int and value >= 0


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def safe_directory(directory, *, create=False):
    if create:
        directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("memory report directory must be private and owned by this user")


def read_json(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or info.st_mode & 0o077
            or info.st_size > MAX_REPORT_BYTES
        ):
            raise ValueError("unsafe memory report file")
        data = handle.read(MAX_REPORT_BYTES + 1)
        if len(data) > MAX_REPORT_BYTES:
            raise ValueError("memory report is too large")
    return json.loads(data)


def atomic_json(path, payload):
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("memory report is too large")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        # tmpfs: replace the report, with no fsync or accumulating history.
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def allocator_counts(stats):
    result = {key: stats.get(source) for key, source in COUNTERS.items()}
    result = {key: value if natural(value) else None for key, value in result.items()}
    allocated, active, reserved = (
        result[key] for key in ("allocated_bytes", "active_bytes", "reserved_bytes")
    )
    result["unused_reserved_bytes"] = (
        reserved - allocated
        if natural(reserved) and natural(allocated) and reserved >= allocated
        else None
    )
    result["pending_free_bytes"] = (
        active - allocated
        if natural(active) and natural(allocated) and active >= allocated
        else None
    )
    return result


def buffer_inventory(runner, torch, *, max_nodes=20000, budget_seconds=0.025, storage_index=None):
    """Deduplicate GPU storages in known roots without keeping any tensor alive.

    This is a partial inventory, not a walk of the Python heap. Runtime requests,
    session objects, caches of output objects and tensor contents are never read.
    Storage addresses are used locally to detect aliases and are not reported.
    """
    started = time.monotonic()
    seen_objects, seen_storages = set(), set()
    rows = {key: {"bytes": 0, "storages": 0} for key in GROUPS}
    truncated = False
    nodes = 0

    def visit(value, group, *, modules=False):
        nonlocal nodes, truncated
        if value is None or id(value) in seen_objects:
            return
        if nodes >= max_nodes or time.monotonic() - started > budget_seconds:
            truncated = True
            return
        nodes += 1
        seen_objects.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.device.type != "cuda":
                return
            storage = value.untyped_storage()
            size = storage.nbytes()
            identity = (value.device.index, storage.data_ptr())
            if size and identity not in seen_storages:
                seen_storages.add(identity)
                rows[group]["bytes"] += size
                rows[group]["storages"] += 1
                if storage_index is not None:
                    storage_index[identity] = (group, size)
        elif type(value) in (list, tuple, dict):
            # Snapshot the small fixed-root dictionaries to tolerate a concurrent
            # metadata refresh. Do not traverse module/request objects in them.
            children = tuple(value.values()) if type(value) is dict else value
            for child in children:
                if nodes >= max_nodes or time.monotonic() - started > budget_seconds:
                    truncated = True
                    break
                nodes += 1
                if isinstance(child, torch.Tensor) or type(child) in (list, tuple, dict):
                    visit(child, group, modules=modules)
        elif modules and isinstance(value, torch.nn.Module):
            # Direct tensor attributes include quantization buffers that are not
            # registered parameters. Names and parameter values are not emitted.
            attrs = vars(value)
            for child in tuple(attrs.values()):
                if isinstance(child, torch.Tensor) or type(child) in (list, tuple, dict):
                    visit(child, group)
            for child in tuple(attrs.get("_modules", {}).values()):
                visit(child, group, modules=True)

    def buffers(owner, group):
        if owner is None:
            return
        # Only direct tensors and tensor containers from these fixed roots.
        # Do not follow runner/model/config/request object references.
        for child in tuple(vars(owner).values()):
            if isinstance(child, torch.Tensor) or type(child) in (list, tuple, dict):
                visit(child, group)

    visit(getattr(runner, "kv_caches", None), "kv_cache")
    visit(getattr(runner, "model", None), "model", modules=True)
    speculator = getattr(runner, "speculator", None)
    visit(getattr(speculator, "model", None), "draft_model", modules=True)
    for group in GROUPS:
        if group not in ("kv_cache", "model", "draft_model"):
            buffers(getattr(runner, group, None), group)
    return {
        "collected_at": time.time(),
        "duration_ms": (time.monotonic() - started) * 1000,
        "truncated": truncated,
        "groups": rows,
        "known_storage_bytes": sum(row["bytes"] for row in rows.values()),
    }


MAP_TOTALS = (
    "segments",
    "blocks",
    "reserved_bytes",
    "allocated_bytes",
    "pending_bytes",
    "inactive_bytes",
    "fragmented_bytes",
    "fully_inactive_bytes",
    "largest_free_block_bytes",
)
MAP_POOLS = ("default", "private", "unknown")


def summarize_allocation_map(
    segments,
    storage_index,
    device,
    *,
    max_segments=8192,
    max_blocks=65536,
    budget_seconds=0.1,
):
    """Reduce CPU allocator metadata to sizes and fixed owner groups; discard addresses/traces."""
    started = time.monotonic()
    owners = sorted(
        (address, group, size)
        for (index, address), (group, size) in storage_index.items()
        if index == device
    )
    addresses = [row[0] for row in owners]
    totals = dict.fromkeys(MAP_TOTALS, 0)
    pools = {kind: dict.fromkeys(MAP_TOTALS, 0) for kind in MAP_POOLS}
    histogram = [{"blocks": 0, "bytes": 0} for _ in range(len(FREE_BINS) + 1)]
    rows = []
    truncated = False
    for index, segment in enumerate(segments):
        if index >= max_segments or time.monotonic() - started > budget_seconds:
            truncated = True
            break
        if segment.get("device") != device:
            continue
        blocks = segment["blocks"]
        if len(blocks) + totals["blocks"] > max_blocks:
            truncated = True
            break
        base, size = segment["address"], segment["total_size"]
        if not natural(base) or not natural(size):
            raise ValueError("invalid allocator segment")
        pool_id = segment.get("segment_pool_id", segment.get("owner_private_pool_id"))
        pool = "unknown"
        if type(pool_id) in (tuple, list) and len(pool_id) == 2 and all(map(natural, pool_id)):
            pool = "default" if not any(pool_id) else "private"
        counts = dict.fromkeys(MAP_TOTALS, 0)
        counts.update(segments=1, blocks=len(blocks), reserved_bytes=size)
        groups = dict.fromkeys(GROUPS, 0)
        offset = 0
        free = []
        for block in blocks:
            if time.monotonic() - started > budget_seconds:
                truncated = True
                break
            amount = block["size"]
            address = block.get("address", base + offset)
            if not natural(amount) or address != base + offset:
                raise ValueError("invalid allocator block layout")
            offset += amount
            state = block["state"]
            if state == "inactive":
                counts["inactive_bytes"] += amount
                counts["largest_free_block_bytes"] = max(counts["largest_free_block_bytes"], amount)
                free.append(amount)
            elif state == "active_awaiting_free":
                counts["pending_bytes"] += amount
            elif state == "active_allocated":
                counts["allocated_bytes"] += amount
                cursor = bisect_left(addresses, address)
                matched = 0
                while cursor < len(owners) and owners[cursor][0] < address + amount:
                    owner_address, group, owner_size = owners[cursor]
                    if owner_address + owner_size <= address + amount:
                        groups[group] += owner_size
                        matched += owner_size
                    cursor += 1
                if matched > amount:
                    raise ValueError("overlapping storage ownership")
            else:
                raise ValueError("unknown allocator block state")
        if truncated:
            break
        if offset != size:
            raise ValueError("incomplete allocator segment")
        whole = counts["allocated_bytes"] + counts["pending_bytes"] == 0
        counts["fully_inactive_bytes" if whole else "fragmented_bytes"] = counts["inactive_bytes"]
        for amount in free:
            bucket = next(
                (i for i, bound in enumerate(FREE_BINS) if amount < bound * 1024**2), len(FREE_BINS)
            )
            histogram[bucket]["blocks"] += 1
            histogram[bucket]["bytes"] += amount
        for target in (totals, pools[pool]):
            for key, value in counts.items():
                target[key] = (
                    max(target[key], value)
                    if key == "largest_free_block_bytes"
                    else target[key] + value
                )
        rows.append(
            {
                "segment": index + 1,
                "pool": pool,
                **counts,
                "owners": groups,
                "unattributed_allocated_bytes": counts["allocated_bytes"] - sum(groups.values()),
            }
        )
    rows.sort(key=lambda row: (row["fragmented_bytes"], row["inactive_bytes"]), reverse=True)
    return {
        "totals": totals,
        "pools": pools,
        "free_block_histogram": histogram,
        "segments": rows[:MAP_ROWS],
        "omitted_segments": max(0, len(rows) - MAP_ROWS),
        "truncated": truncated,
        "analysis_ms": (time.monotonic() - started) * 1000,
    }


def allocation_map(runner, torch):
    device = getattr(runner.device, "index", None)
    if not natural(device):
        raise ValueError("allocator map requires an explicit device")
    started = time.monotonic()
    # Qualified runtime supports excluding event history. Never fall back to the
    # trace-enabled default, start history recording, or synchronize the GPU.
    segments = torch.cuda.memory_snapshot(include_traces=False)
    snapshot_ms = (time.monotonic() - started) * 1000
    storage_index = {}
    inventory = buffer_inventory(runner, torch, storage_index=storage_index)
    result = summarize_allocation_map(segments, storage_index, device)
    return {
        **result,
        "collected_at": time.time(),
        "snapshot_ms": snapshot_ms,
        "inventory_ms": inventory["duration_ms"],
        "ownership_truncated": inventory["truncated"],
    }


class MemoryReporter:
    def __init__(self, runner, torch, directory=DIRECTORY):
        self.runner = weakref.ref(runner)
        self.torch = torch
        self.device = runner.device
        self.directory = Path(directory)
        self.instance_id = uuid4().hex
        self.stop_event = threading.Event()
        self.thread = None
        self.lock_fd = None
        self.inventory = None
        self.last_refresh = 0
        self.sequence = 0
        self.last_write_ms = 0.0
        self.max_sample_ms = 0.0
        self.last_map_request = 0
        self.last_map_attempt = -MAP_INTERVAL

    def start(self):
        safe_directory(self.directory, create=True)
        fd = os.open(self.directory / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                raise ValueError("unsafe memory report owner lock")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            os.close(fd)
            raise
        self.lock_fd = fd
        try:
            self.sample()
            self.thread = threading.Thread(target=self.run, name="qwen-memory-report", daemon=True)
            self.thread.start()
        except Exception:
            self.stop()
            raise

    def run(self):
        try:
            while not self.stop_event.wait(INTERVAL):
                if self.runner() is None:
                    break
                try:
                    self.sample()
                except Exception:
                    # No exception text, object reprs or retry spin. A failed
                    # reporter becomes stale and cannot fail model execution.
                    continue
        finally:
            if self.lock_fd is not None:
                os.close(self.lock_fd)
                self.lock_fd = None

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=0.2)
        elif self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None

    def sample(self):
        started = time.monotonic()
        runner = self.runner()
        if runner is None:
            return
        refresh = self.last_refresh
        try:
            request = read_json(self.directory / "refresh.json")
            value = request.get("requested_at_ns")
            if request.get("instance_id") == self.instance_id and natural(value):
                refresh = max(refresh, value)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        inventory_seconds = 0.0
        if self.inventory is None or refresh > self.last_refresh:
            inventory_started = time.monotonic()
            self.inventory = buffer_inventory(runner, self.torch)
            self.last_refresh = refresh
            inventory_seconds = time.monotonic() - inventory_started
        # memory_stats reads the existing allocator's CPU bookkeeping. Pass the
        # already initialized device explicitly; never create a GPU context.
        counts = allocator_counts(self.torch.cuda.memory_stats(self.device))
        self.sequence += 1
        sample_ms = (time.monotonic() - started - inventory_seconds) * 1000
        self.max_sample_ms = max(self.max_sample_ms, sample_ms)
        payload = {
            "schema": SCHEMA,
            "instance_id": self.instance_id,
            "pid": os.getpid(),
            "updated_at": time.time(),
            "sequence": self.sequence,
            "interval_seconds": INTERVAL,
            "allocation_map_version": 1,
            "allocator": counts,
            "inventory": {**self.inventory, "requested_at_ns": self.last_refresh},
            "overhead": {
                "sample_ms": sample_ms,
                "max_sample_ms": self.max_sample_ms,
                "previous_write_ms": self.last_write_ms,
            },
        }
        written = time.monotonic()
        atomic_json(self.directory / "report.json", payload)
        self.last_write_ms = (time.monotonic() - written) * 1000
        self.service_map(runner)

    def service_map(self, runner):
        try:
            request = read_json(self.directory / "map-request.json")
            requested = request.get("requested_at_ns")
            if (
                request.get("instance_id") != self.instance_id
                or not natural(requested)
                or requested <= self.last_map_request
                or time.monotonic() - self.last_map_attempt < MAP_INTERVAL
            ):
                return
        except (OSError, ValueError, TypeError, AttributeError):
            return
        self.last_map_request = requested
        self.last_map_attempt = time.monotonic()
        payload = {
            "schema": MAP_SCHEMA,
            "instance_id": self.instance_id,
            "pid": os.getpid(),
            "requested_at_ns": requested,
        }
        try:
            payload.update(allocation_map(runner, self.torch), available=True, state="current")
        except Exception:
            # A diagnostic failure cannot fail the sampler or model. Do not
            # serialize exception text, which can include object representations.
            payload.update(available=False, state="map_capture_failed", collected_at=time.time())
        with contextlib.suppress(OSError, ValueError):
            atomic_json(self.directory / "map.json", payload)


def start_memory_report(runner):
    """Called once after worker warmup; optional diagnostics cannot fail startup."""
    if getattr(runner, "qwen_memory_report", None) is not None:
        return
    try:
        import torch

        if not torch.cuda.is_initialized():
            return
        report = MemoryReporter(runner, torch)
        report.start()
        runner.qwen_memory_report = report
    except Exception:
        import logging

        logging.getLogger(__name__).warning("GPU memory report unavailable")


def stop_memory_report(runner):
    report = getattr(runner, "qwen_memory_report", None)
    if report is not None:
        report.stop()


def read_report(directory=DIRECTORY):
    """Read a bounded allowlist of metadata. No torch import or session discovery."""
    directory = Path(directory)
    try:
        safe_directory(directory)
        data = read_json(directory / "report.json")
        if data.get("schema") != SCHEMA:
            raise ValueError("unknown schema")
        instance = data.get("instance_id")
        if (
            not isinstance(instance, str)
            or len(instance) != 32
            or any(c not in "0123456789abcdef" for c in instance)
        ):
            raise ValueError("invalid instance")
        if not all(nonnegative(data.get(key)) for key in ("updated_at", "interval_seconds")):
            raise ValueError("invalid timing")
        if not all(natural(data.get(key)) for key in ("pid", "sequence")):
            raise ValueError("invalid process counters")
        counts = {
            key: value if natural(value := data.get("allocator", {}).get(key)) else None
            for key in (*COUNTERS, "unused_reserved_bytes", "pending_free_bytes")
        }
        inventory = data.get("inventory", {})
        if not all(nonnegative(inventory.get(key)) for key in ("collected_at", "duration_ms")):
            raise ValueError("invalid inventory timing")
        if (
            not natural(inventory.get("requested_at_ns"))
            or type(inventory.get("truncated")) is not bool
        ):
            raise ValueError("invalid inventory")
        rows = {}
        for key in GROUPS:
            row = inventory.get("groups", {}).get(key, {})
            if not all(natural(row.get(field)) for field in ("bytes", "storages")):
                raise ValueError("invalid buffer sizes")
            rows[key] = {field: row[field] for field in ("bytes", "storages")}
        age = time.time() - data["updated_at"]
        return {
            "schema": SCHEMA,
            "available": 0 <= age <= STALE_AFTER,
            "state": "current" if 0 <= age <= STALE_AFTER else "stale",
            "age_seconds": max(0, age),
            "allocation_map_version": 1 if data.get("allocation_map_version") == 1 else 0,
            **{
                key: data[key]
                for key in ("instance_id", "pid", "sequence", "updated_at", "interval_seconds")
            },
            "allocator": counts,
            "inventory": {
                **{
                    key: inventory[key]
                    for key in ("collected_at", "duration_ms", "truncated", "requested_at_ns")
                },
                "groups": rows,
                "known_storage_bytes": sum(row["bytes"] for row in rows.values()),
            },
            "overhead": {
                key: value if nonnegative(value := data.get("overhead", {}).get(key)) else None
                for key in ("sample_ms", "max_sample_ms", "previous_write_ms")
            },
        }
    except FileNotFoundError:
        return {"available": False, "state": "not_enabled"}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"available": False, "state": "unavailable"}


def refresh_report(directory=DIRECTORY, timeout=5.0):
    directory = Path(directory)
    previous = read_report(directory)
    if not previous["available"]:
        return previous
    requested = time.time_ns()
    atomic_json(
        directory / "refresh.json",
        {
            "instance_id": previous["instance_id"],
            "requested_at_ns": requested,
        },
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = read_report(directory)
        if not current["available"] or current.get("instance_id") != previous["instance_id"]:
            return {"available": False, "state": "backend_changed"}
        if current["inventory"]["requested_at_ns"] >= requested:
            return current
        time.sleep(0.1)
    return {"available": False, "state": "refresh_timeout"}


def read_allocation_map(directory, instance):
    """Read only the map's numeric/enumerated schema, never addresses or trace fields."""

    def numbers(value, keys):
        if not all(natural(value.get(key)) for key in keys):
            raise ValueError("invalid allocation map counters")
        return {key: value[key] for key in keys}

    try:
        directory = Path(directory)
        safe_directory(directory)
        data = read_json(directory / "map.json")
        if data.get("schema") != MAP_SCHEMA or data.get("instance_id") != instance:
            raise ValueError("allocation map belongs to a different backend")
        age = time.time() - data["collected_at"]
        if not nonnegative(age) or not natural(data.get("pid")):
            raise ValueError("invalid allocation map identity/timing")
        result = {
            "schema": MAP_SCHEMA,
            "instance_id": instance,
            "pid": data["pid"],
            "collected_at": data["collected_at"],
            "age_seconds": age,
            **numbers(data, ("requested_at_ns",)),
        }
        if data.get("available") is False and data.get("state") == "map_capture_failed":
            return {**result, "available": False, "state": "map_capture_failed"}
        if data.get("available") is not True or data.get("state") != "current":
            raise ValueError("invalid allocation map status")
        for key in ("snapshot_ms", "inventory_ms", "analysis_ms"):
            if not nonnegative(data.get(key)):
                raise ValueError("invalid allocation map timing")
            result[key] = data[key]
        for key in ("truncated", "ownership_truncated"):
            if type(data.get(key)) is not bool:
                raise ValueError("invalid allocation map completeness")
            result[key] = data[key]
        segments = data["segments"]
        histogram = data["free_block_histogram"]
        if len(segments) > MAP_ROWS or len(histogram) != len(FREE_BINS) + 1:
            raise ValueError("oversized allocation map")
        clean_segments = []
        for row in segments:
            if row.get("pool") not in MAP_POOLS:
                raise ValueError("unknown allocation map pool")
            clean_segments.append(
                {
                    **numbers(row, (*MAP_TOTALS, "segment", "unattributed_allocated_bytes")),
                    "pool": row["pool"],
                    "owners": numbers(row["owners"], GROUPS),
                }
            )
        return {
            **result,
            "available": True,
            "state": "current",
            "totals": numbers(data["totals"], MAP_TOTALS),
            "pools": {key: numbers(data["pools"][key], MAP_TOTALS) for key in MAP_POOLS},
            "segments": clean_segments,
            "free_block_histogram": [numbers(row, ("blocks", "bytes")) for row in histogram],
            **numbers(data, ("omitted_segments",)),
        }
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return {"schema": MAP_SCHEMA, "available": False, "state": "map_unavailable"}


def request_allocation_map(directory=DIRECTORY, timeout=5.0):
    directory = Path(directory)
    previous = read_report(directory)
    if not previous["available"]:
        return previous
    if previous.get("allocation_map_version") != 1:
        return {"schema": MAP_SCHEMA, "available": False, "state": "map_not_enabled"}
    instance = previous["instance_id"]
    cached = read_allocation_map(directory, instance)
    if cached.get("state") == "map_capture_failed" and cached["age_seconds"] <= MAP_INTERVAL:
        return {**cached, "reused": True}
    if cached["available"] and cached["age_seconds"] <= MAP_INTERVAL:
        return {**cached, "reused": True}
    requested = time.time_ns()
    atomic_json(
        directory / "map-request.json",
        {
            "instance_id": instance,
            "requested_at_ns": requested,
        },
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = read_report(directory)
        if not current["available"] or current.get("instance_id") != instance:
            return {"schema": MAP_SCHEMA, "available": False, "state": "backend_changed"}
        result = read_allocation_map(directory, instance)
        if result["available"] and result["age_seconds"] <= MAP_INTERVAL:
            return {**result, "reused": result["requested_at_ns"] != requested}
        if result.get("state") == "map_capture_failed" and result["requested_at_ns"] >= requested:
            return result
        time.sleep(0.1)
    return {"schema": MAP_SCHEMA, "available": False, "state": "map_timeout"}
