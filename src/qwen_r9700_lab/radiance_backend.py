"""Pinned backend lifecycle control, shared by host Pi and the VM bridge.

Only lifecycle, resource and numeric scheduler metadata is returned. Model inputs,
generated tokens and logs are never read. Detached operations survive a terminal
disconnect, and all callers share a lock on the GPU host. Stopping never escalates
to SIGKILL.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import socket
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

SCHEMA = "urn:coherence:backend-control:v1"
ACTIONS = ("status", "start", "stop")
HEX = re.compile(r"[0-9a-f]{64}\Z")
MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"
SCHEDULER = Path("/dev/shm/qwen-radiance-fair-public-scheduler.json")
TAIL = Path("/dev/shm/qwen-radiance-snapshot-tail.json")
DRM = Path("/sys/class/drm")
MEMINFO = Path("/proc/meminfo")
SCHEDULER_MAX_AGE = 10


def bounded_text(path: Path, maximum=65536):
    """Read small kernel attributes without loading a driver or following pipes."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("non-regular host resource attribute")
        raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError("oversized host resource attribute")
        return raw.decode("ascii").strip()


def resource_number(path: Path, maximum):
    try:
        value = bounded_text(path, 64)
        if not re.fullmatch(r"[0-9]{1,20}", value):
            return None
        number = int(value)
        return number if number <= maximum else None
    except (OSError, ValueError, UnicodeError):
        return None


def host_resources():
    """Command-scoped sysfs/proc sample, independent of backend lifetime."""
    gpu = None
    try:
        devices = sorted(DRM.glob("card[0-9]*/device"))[:16]
    except OSError:
        devices = []
    for device in devices:
        # Select the AMD GPU, just as the shared Pi temperature reader does.
        # A different vendor's resources must not be attributed to this backend.
        try:
            if bounded_text(device / "vendor", 32) != "0x1002":
                continue
        except (OSError, ValueError, UnicodeError):
            continue
        gpu = {
            "edge_millicelsius": None,
            "junction_millicelsius": None,
            "fan_percent": None,
            "utilization_percent": resource_number(device / "gpu_busy_percent", 100),
            "vram_used_bytes": resource_number(
                device / "mem_info_vram_used", 2**53 - 1
            ),
            "vram_total_bytes": resource_number(
                device / "mem_info_vram_total", 2**53 - 1
            ),
        }
        used, total = gpu["vram_used_bytes"], gpu["vram_total_bytes"]
        if total == 0 or (used is not None and total is not None and used > total):
            gpu["vram_used_bytes"] = gpu["vram_total_bytes"] = None
        try:
            hwmons = sorted((device / "hwmon").glob("hwmon*"))[:16]
        except OSError:
            hwmons = []
        for hwmon in hwmons:
            try:
                labels = sorted(hwmon.glob("temp*_label"))[:16]
            except OSError:
                labels = []
            for label in labels:
                try:
                    kind = bounded_text(label, 32)
                except (OSError, ValueError, UnicodeError):
                    continue
                if kind in ("edge", "junction"):
                    gpu[f"{kind}_millicelsius"] = resource_number(
                        label.with_name(label.name.removesuffix("_label") + "_input"),
                        250000,
                    )
            pwm = resource_number(hwmon / "pwm1", 1000000)
            pwm_max = resource_number(hwmon / "pwm1_max", 1000000)
            if pwm is not None and pwm_max and pwm <= pwm_max:
                gpu["fan_percent"] = (pwm * 100 + pwm_max // 2) // pwm_max
        break
    ram = None
    try:
        values = {}
        for line in bounded_text(MEMINFO).splitlines():
            match = re.fullmatch(r"(MemTotal|MemAvailable):\s+([0-9]{1,16}) kB", line)
            if match:
                values[match[1]] = int(match[2]) * 1024
        total, available = values.get("MemTotal"), values.get("MemAvailable")
        if total and available is not None and 0 <= available <= total <= 2**53 - 1:
            ram = {
                "total_bytes": total,
                "available_bytes": available,
                "used_bytes": total - available,
            }
    except (OSError, ValueError, UnicodeError):
        pass
    return {
        "observed_at": time.time(),
        "source": "host_sysfs_proc",
        "gpu": gpu,
        "ram": ram,
    }


def legacy_contract(root: Path, *, container=None, abi=None, cache_root=None) -> dict:
    """Derive the same pinned contract as the existing launcher, without eval."""
    text = (root / "scripts/pi-remote-qwen-radiance").read_text()

    def constant(name):
        found = re.findall(
            rf"^readonly {name}=([A-Za-z0-9_./@-]+)$", text, re.MULTILINE
        )
        if len(found) != 1:
            raise ValueError(f"missing pinned backend setting: {name}")
        return found[0]

    raw = (
        root / "experiments/radiance-public/snapshot-abi-chat-cache-v1.json"
    ).read_bytes()
    runtime_abi = constant("SNAPSHOT_ABI")
    if hashlib.sha256(raw).hexdigest() != runtime_abi:
        raise ValueError("pinned backend manifest hash mismatch")
    manifest = json.loads(raw)
    name, data_abi = constant("CONTAINER"), constant("SNAPSHOT_DATA_ABI")
    if container not in (None, name) or abi not in (None, data_abi):
        raise ValueError("backend lifecycle is not configured for this deployment")
    if manifest["storage"]["data_abi"] != data_abi:
        raise ValueError("pinned snapshot identity mismatch")
    remote = Path(constant("REMOTE_ROOT"))
    patches = remote / "radiance-vllm-mxfp4"
    files = {
        str(patches / name): digest
        for name, digest in manifest["runtime"]["release_files"].items()
    }
    files.update(
        {
            str(patches / name): digest
            for name, digest in manifest["runtime"]["chat_storage"]["modules"].items()
        }
    )
    files[str(patches / "patch_streaming_snapshot.py")] = constant(
        "SNAPSHOT_PATCH_SHA256"
    )
    launcher = str(remote / "launch_public_clean_snapshot_server.sh")
    files[launcher] = constant("REMOTE_LAUNCHER_SHA256")
    cache = cache_root or constant("REMOTE_CACHE")
    files[str(Path(cache) / "snapshots" / runtime_abi / "abi.json")] = runtime_abi
    return {
        "container": name,
        "model": constant("MODEL_ID"),
        "port": int(constant("REMOTE_PORT")),
        "image": constant("IMAGE_ID"),
        "runtime_abi": runtime_abi,
        "data_abi": data_abi,
        "compatible_runtime_abis": manifest["runtime"]["memory_report"].get(
            "compatible_runtime_abis", []
        ),
        "cache_root": cache,
        "launcher": launcher,
        "files": files,
        "cache_module": str(patches / "radiance_cache.py"),
    }


def private_directory(path: Path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError(f"unsafe backend control directory: {path}")


def bounded_json(path: Path, maximum=2 * 1024 * 1024):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_size > maximum
        ):
            raise ValueError("unsafe backend metadata")
        data = stream.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError("oversized backend metadata")
        return json.loads(data)


class Controller:
    def __init__(self, contract):
        self.config = contract
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", contract["container"])
            or contract["model"] != MODEL
            or type(contract["port"]) is not int
            or not 1 <= contract["port"] <= 65535
            or any(
                not HEX.fullmatch(contract[key])
                for key in ("image", "runtime_abi", "data_abi")
            )
        ):
            raise ValueError("invalid pinned backend contract")
        self.directory = (
            Path(contract["cache_root"]).resolve()
            / "backend-control"
            / contract["container"]
        )
        private_directory(self.directory.parent)
        private_directory(self.directory)
        self.record_path = self.directory / "operation.json"
        self.lock_path = self.directory / "operation.lock"

    def lock(self):
        descriptor = os.open(
            self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or info.st_mode & 0o077
        ):
            os.close(descriptor)
            raise ValueError("unsafe backend control lock")
        return descriptor

    def busy(self):
        descriptor = self.lock()
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return False
            except BlockingIOError:
                return True
        finally:
            os.close(descriptor)

    def record(self, **changes):
        previous = bounded_json(self.record_path) or {}
        now = time.time()
        value = {
            **previous,
            **changes,
            "updated_at": now,
        }
        if changes.get("id") and changes["id"] != previous.get("id"):
            value.update(started_at=now, finished_at=None)
        if changes.get("status") in ("complete", "failed"):
            value["finished_at"] = now
        descriptor, name = tempfile.mkstemp(dir=self.directory, prefix=".operation-")
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.record_path)
        finally:
            Path(name).unlink(missing_ok=True)
        return value

    def inspect(self):
        result = subprocess.run(
            ["podman", "container", "inspect", self.config["container"]],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode:
            # A daemon/storage failure must not be mistaken for an absent container.
            exists = subprocess.run(
                ["podman", "container", "exists", self.config["container"]],
                capture_output=True,
                timeout=5,
                check=False,
            )
            if exists.returncode == 1:
                return None
            raise RuntimeError("cannot inspect the backend container")
        if len(result.stdout) > 2 * 1024 * 1024:
            raise ValueError("oversized container inspection")
        rows = json.loads(result.stdout)
        if len(rows) != 1:
            raise ValueError("unexpected container inspection")
        return rows[0]

    def matches(self, row):
        c = self.config
        args = row.get("Args", [])
        if not isinstance(args, list) or any(not isinstance(x, str) for x in args):
            return False
        pairs = [
            ("--served-model-name", c["model"]),
            ("--max-model-len", "253792"),
            ("--kv-cache-dtype", "fp8"),
            ("--port", str(c["port"])),
        ]
        if (
            row.get("Name") != c["container"]
            or row.get("Image") != c["image"]
            or any(
                not any(args[i : i + 2] == list(pair) for i in range(len(args) - 1))
                for pair in pairs
            )
        ):
            return False
        try:
            transfer = json.loads(args[args.index("--kv-transfer-config") + 1])
            extra = transfer["kv_connector_extra_config"]
            accepted = [c["runtime_abi"], *c["compatible_runtime_abis"]]
            tiers = extra["secondary_tiers"]
            return (
                transfer["engine_id"]
                in [f"qwen-radiance-public-clean-{item[:16]}" for item in accepted]
                and any(
                    tier.get("type") == "qwen_chat_fs"
                    and tier.get("root_dir") == f"/cache/snapshots/{c['data_abi']}/data"
                    for tier in tiers
                )
                and "--speculative-config" in args
            )
        except (KeyError, ValueError, IndexError, TypeError):
            return False

    def ready(self):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.config['port']}/health", timeout=1
            ) as stream:
                if stream.status != 200:
                    return False
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.config['port']}/v1/models", timeout=1
            ) as stream:
                value = json.loads(stream.read(65537))
            return (
                isinstance(value, dict)
                and isinstance(value.get("data"), list)
                and len(value["data"]) == 1
                and isinstance(value["data"][0], dict)
                and value["data"][0].get("id") == self.config["model"]
            )
        except (OSError, ValueError, TypeError, urllib.error.URLError):
            return False

    @staticmethod
    def started_at(row):
        try:
            return datetime.fromisoformat(row["State"]["StartedAt"]).timestamp()
        except (KeyError, ValueError, TypeError):
            raise ValueError("backend start time unavailable") from None

    def scheduler(self, row):
        return self.scheduler_counts(row, bounded_json(SCHEDULER))

    def scheduler_counts(self, row, value):
        if (
            not isinstance(value, dict)
            or type(value.get("updated_at")) not in (int, float)
            or not math.isfinite(value["updated_at"])
            or value["updated_at"] < self.started_at(row)
            or not isinstance(value.get("requests"), list)
            or len(value["requests"]) > 4096
            or any(not isinstance(item, dict) for item in value["requests"])
        ):
            return None
        # Return counts only, never request IDs, tokens, or workspace metadata.
        rows = value["requests"]
        return {
            "active_requests": len(rows),
            "queued_requests": sum(r.get("state") == "queued" for r in rows),
            "generating_requests": sum(r.get("state") == "running" for r in rows),
            "paused_requests": sum(r.get("state") == "paused" for r in rows),
        }

    def scheduler_observation(self, row, *, running, now):
        result = {"status": "unavailable", "observed_at": None, "age_seconds": None}
        if running is not True:
            result["status"] = (
                "backend_stopped" if running is False else "backend_unknown"
            )
            return result
        try:
            value = bounded_json(SCHEDULER)
            counts = self.scheduler_counts(row, value)
            if counts is None:
                return result
            updated = value["updated_at"]
            age = now - updated
            result.update(observed_at=updated, age_seconds=max(0, age))
            # Empty idle snapshots do not always change; a fresh tail heartbeat
            # from the same scheduler process can authenticate its idle state.
            if counts["active_requests"] == 0 and age > SCHEDULER_MAX_AGE:
                tail = bounded_json(TAIL)
                if (
                    type(value.get("pid")) is int
                    and value["pid"] > 0
                    and isinstance(tail, dict)
                    and tail.get("pid") == value["pid"]
                    and type(tail.get("updated_at")) in (int, float)
                    and math.isfinite(tail["updated_at"])
                    and updated <= tail["updated_at"]
                    and -5 <= now - tail["updated_at"] <= SCHEDULER_MAX_AGE
                ):
                    age = now - tail["updated_at"]
                    result.update(
                        observed_at=tail["updated_at"], age_seconds=max(0, age)
                    )
            result["status"] = "current" if -5 <= age <= SCHEDULER_MAX_AGE else "stale"
            if result["status"] == "current":
                result.update(counts)
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return result

    def release_identity(self, row, matches):
        expected = {
            key: self.config[key]
            for key in ("container", "model", "image", "runtime_abi", "data_abi")
        }
        observed = None
        if row is not None:

            def label(value):
                return (
                    value
                    if isinstance(value, str)
                    and re.fullmatch(r"[A-Za-z0-9_./:-]{1,256}", value)
                    else None
                )

            observed = {
                "container": label(row.get("Name")),
                "model": None,
                "image": label(row.get("Image")),
                "runtime_abi": None,
                "runtime_abi_prefix": None,
                "data_abi": None,
            }
            args = row.get("Args")
            if isinstance(args, list):
                try:
                    observed["model"] = label(
                        args[args.index("--served-model-name") + 1]
                    )
                except (ValueError, IndexError):
                    pass
                try:
                    transfer = json.loads(args[args.index("--kv-transfer-config") + 1])
                    match = re.fullmatch(
                        r"qwen-radiance-public-clean-([0-9a-f]{16})",
                        transfer.get("engine_id", ""),
                    )
                    if match:
                        observed["runtime_abi_prefix"] = match[1]
                    tiers = transfer["kv_connector_extra_config"]["secondary_tiers"]
                    for tier in tiers:
                        match = re.fullmatch(
                            r"/cache/snapshots/([0-9a-f]{64})/data",
                            tier.get("root_dir", ""),
                        )
                        if tier.get("type") == "qwen_chat_fs" and match:
                            observed["data_abi"] = match[1]
                            break
                except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                    pass
            config = row.get("Config")
            env = config.get("Env", []) if isinstance(config, dict) else []
            if isinstance(env, list):
                for item in env:
                    match = (
                        re.fullmatch(
                            r"TRITON_CACHE_DIR=/cache/runtime/([0-9a-f]{64})/triton",
                            item,
                        )
                        if isinstance(item, str)
                        else None
                    )
                    if match:
                        observed["runtime_abi"] = match[1]
                        break
        return {
            "expected": expected,
            "observed": observed,
            "matches": matches,
            "verification": "container_configuration",
        }

    def status(self, *, busy_override=None):
        now = time.time()
        issues = []
        row, running, matches, ready = None, None, None, None
        try:
            row = self.inspect()
            if row is None:
                running, matches, ready = False, True, False
            else:
                if (
                    not isinstance(row, dict)
                    or type(row["State"]["Running"]) is not bool
                ):
                    raise ValueError("invalid backend container state")
                running = row["State"]["Running"]
                matches = self.matches(row)
                ready = False
        except (
            OSError,
            ValueError,
            RuntimeError,
            TypeError,
            KeyError,
            subprocess.SubprocessError,
        ):
            row, running, matches, ready = None, None, None, None
            issues.append("container_inspection_failed")
        if running and matches:
            try:
                ready = bool(self.ready())
            except (
                OSError,
                ValueError,
                RuntimeError,
                TypeError,
                subprocess.SubprocessError,
            ):
                ready = None
                issues.append("backend_readiness_probe_failed")
        telemetry = self.scheduler_observation(row, running=running, now=now)
        counts = (
            {key: telemetry[key] for key in ("active_requests", "queued_requests")}
            if ready and telemetry["status"] == "current"
            else None
        )
        busy = self.busy() if busy_override is None else busy_override
        operation = bounded_json(self.record_path)
        if operation is not None and not isinstance(operation, dict):
            operation = None
            issues.append("operation_metadata_invalid")
        if not busy and operation and operation.get("status") == "pending":
            operation = {
                **operation,
                "status": "failed",
                "stage": "Operation interrupted",
                "error": "host lifecycle worker ended before completion",
                "finished_at": operation.get("updated_at"),
            }
        state = (
            "unavailable"
            if running is None or ready is None
            else (
                (
                    "generating"
                    if counts and telemetry["generating_requests"]
                    else "idle"
                    if counts and not counts["active_requests"]
                    else "running"
                )
                if ready
                else "starting"
                if running and matches
                else "stopped"
                if not running
                else "unavailable"
            )
        )
        if busy and operation and running is not None:
            state = "starting" if operation.get("action") == "start" else "stopping"
        started = None
        if running:
            try:
                candidate = self.started_at(row)
                if math.isfinite(candidate) and 0 < candidate <= now + 5:
                    started = candidate
            except ValueError:
                issues.append("backend_start_time_unavailable")
        operation_age = operation_elapsed = None
        if operation:
            updated, began, ended = (
                operation.get(key)
                for key in ("updated_at", "started_at", "finished_at")
            )
            if type(updated) in (int, float) and math.isfinite(updated):
                operation_age = max(0, now - updated)
            if type(began) in (int, float) and math.isfinite(began):
                endpoint = (
                    ended
                    if type(ended) in (int, float) and math.isfinite(ended)
                    else now
                )
                operation_elapsed = max(0, endpoint - began)
        return {
            "schema": SCHEMA,
            "observed_at": now,
            "host_control": {"reachable": True, "observed_at": now},
            "state": state,
            "running": running,
            "ready": ready,
            "pinned": matches,
            "container": self.config["container"],
            "model": self.config["model"],
            "busy": busy,
            "operation": operation,
            "operation_age_seconds": operation_age,
            "operation_elapsed_seconds": operation_elapsed,
            "started_at": started,
            "uptime_seconds": max(0, now - started) if started is not None else None,
            "release": self.release_identity(row, matches),
            "resources": host_resources(),
            "telemetry": {"scheduler": telemetry},
            "issues": issues,
            **(counts or {}),
        }

    def verify_release(self):
        for name, expected in self.config["files"].items():
            if not HEX.fullmatch(expected):
                raise ValueError("invalid release file identity")
            with os.fdopen(os.open(name, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_nlink != 1
                    or info.st_mode & 0o022
                    or hashlib.file_digest(stream, "sha256").hexdigest() != expected
                ):
                    raise ValueError(f"pinned release file mismatch: {Path(name).name}")

    def port_occupied(self):
        with socket.socket() as probe:
            probe.settimeout(1)
            return probe.connect_ex(("127.0.0.1", self.config["port"])) == 0

    def start(self, deadline):
        self.record(stage="Verifying pinned release")
        self.verify_release()
        row = self.inspect()
        if row and not self.matches(row):
            raise RuntimeError("existing container differs from the pinned backend")
        if row is None:
            if self.port_occupied():
                raise RuntimeError("backend port is occupied by another process")
            # Older launchers created cache_root/logs with normal 0755 mode.
            # Keep lifecycle logs inside our owner-only control directory;
            # do not impose its private-directory contract on that old folder.
            logdir = self.directory / "logs"
            private_directory(logdir)
            descriptor, name = tempfile.mkstemp(
                dir=logdir, prefix="pi-backend-", suffix=".log"
            )
            with os.fdopen(descriptor, "wb") as log:
                process = subprocess.Popen(
                    [self.config["launcher"]],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            self.record(stage="Loading model and preparing GPU", log_path=name)
        elif not row["State"].get("Running"):
            # The pinned production launcher uses --rm; do not reuse a stale stopped container.
            raise RuntimeError(
                "a stopped container remains; remove it through the host deployment tools"
            )
        else:
            process = None
            self.record(stage="Waiting for backend readiness")
        while time.monotonic() < deadline:
            row = self.inspect()
            if row and not self.matches(row):
                raise RuntimeError("started container differs from the pinned backend")
            if row and row["State"].get("Running") and self.ready():
                return
            if process and process.poll() is not None and not row:
                raise RuntimeError(
                    "backend startup failed; see the recorded startup log on the host"
                )
            time.sleep(1)
        raise TimeoutError("backend startup is still pending; use /backend status")

    def flush_tails(self, row):
        tails = bounded_json(TAIL)
        if (
            not isinstance(tails, dict)
            or tails.get("schema") != "urn:qwen-r9700:radiance-tail-residency:v1"
        ):
            raise RuntimeError("cannot verify snapshot tails; backend left running")
        if tails.get("updated_at", 0) < self.started_at(row) or not isinstance(
            tails.get("chats"), list
        ):
            raise RuntimeError(
                "snapshot tail status belongs to an older backend; backend left running"
            )
        path = self.config["cache_module"]
        expected = self.config["files"][path]
        if (
            Path(path).is_symlink()
            or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("pinned snapshot flush helper mismatch")
        spec = importlib.util.spec_from_file_location("_pinned_cache_flush", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for index, tail in enumerate(tails["chats"]):
            chat = {"id": tail.get("chat_id"), "generation": tail.get("generation")}
            if any(
                not isinstance(v, str) or not HEX.fullmatch(v) for v in chat.values()
            ):
                raise ValueError("invalid snapshot tail identity")
            tokens = tail.get("tokens")
            if type(tokens) is not int or tokens < 0:
                raise ValueError("invalid snapshot tail token count")
            self.record(
                stage=f"Flushing snapshot tails ({index + 1}/{len(tails['chats'])})"
            )
            result = module.request_tail_flush(chat, timeout=120)
            if result.get("status") not in ("flushed", "already_durable"):
                raise RuntimeError("snapshot tail flush failed; backend left running")
            if type(result.get("tokens")) is not int or result["tokens"] < tokens:
                raise RuntimeError(
                    "snapshot flush did not cover the buffered tail; backend left running"
                )
        return len(tails["chats"])

    def stop(self, deadline):
        row = self.inspect()
        if row is None or not row["State"].get("Running"):
            return
        if not self.matches(row):
            raise RuntimeError(
                "refusing to stop a container outside the pinned deployment"
            )
        container_id = row["Id"]
        if self.ready():
            self.record(
                stage="Waiting for active requests to finish before flushing caches"
            )
            while True:
                counts = self.scheduler(row)
                if counts is None:
                    raise RuntimeError(
                        "cannot confirm active requests; backend left running"
                    )
                if not counts["active_requests"]:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "active requests have not finished; backend left running"
                    )
                time.sleep(0.5)
            flushed = self.flush_tails(row)
            self.record(flushed_chats=flushed)
        # SIGTERM invokes the backend's shutdown/drain path, with no force-kill timer.
        # Use the inspected ID so a replacement container cannot be stopped by name.
        current = self.inspect()
        if not current:
            return
        if current["Id"] != container_id:
            raise RuntimeError(
                "backend changed during shutdown; replacement left running"
            )
        self.record(stage="Graceful shutdown; draining cache writes")
        result = subprocess.run(
            ["podman", "kill", "--signal", "TERM", container_id],
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("graceful shutdown signal could not be delivered")
        while time.monotonic() < deadline:
            current = self.inspect()
            if (
                current is None
                or current["Id"] != container_id
                or not current["State"].get("Running")
            ):
                return
            time.sleep(0.5)
        raise TimeoutError("graceful shutdown is still pending; no force-kill was sent")

    def worker(self, action, descriptor):
        try:
            (self.start if action == "start" else self.stop)(time.monotonic() + 900)
            self.record(
                status="complete",
                stage="Backend ready" if action == "start" else "Backend stopped",
            )
        except Exception as error:  # noqa: BLE001 - Persist any detached worker failure for the caller.
            message = (str(error).splitlines() or [type(error).__name__])[0][:512]
            self.record(status="failed", error=message, stage="Operation failed")
        finally:
            os.close(descriptor)

    def detach(self, action, descriptor):
        pid = os.fork()
        if pid == 0:
            try:
                os.setsid()
                if os.fork():
                    os._exit(0)
                with open(os.devnull, "r+b", buffering=0) as null:
                    for number in (0, 1, 2):
                        os.dup2(null.fileno(), number)
                self.worker(action, descriptor)
            finally:
                os._exit(0)
        os.waitpid(pid, 0)

    def execute(self, action):
        if action not in ACTIONS:
            raise ValueError("usage: /backend [status|start|stop]")
        if action == "status":
            return self.status()
        descriptor = self.lock()
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                operation = bounded_json(self.record_path)
                if not operation or operation.get("action") != action:
                    raise RuntimeError(
                        "another backend operation is in progress; use /backend status"
                    )
                return self.status()
            current = self.status(busy_override=False)
            if not current["pinned"]:
                raise RuntimeError("container differs from the pinned backend")
            if (action == "start" and current["ready"]) or (
                action == "stop" and not current["running"]
            ):
                return {**current, "busy": False, "operation": None}
            self.record(
                id=uuid.uuid4().hex,
                action=action,
                status="pending",
                stage="Starting operation",
                error=None,
                log_path=None,
                flushed_chats=None,
            )
            self.detach(action, descriptor)
            # The child owns this open-file-description lock until the operation finishes.
            return self.status()
        finally:
            os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--host", default="ai")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--container")
    parser.add_argument("--abi")
    parser.add_argument("--cache-root")
    parser.add_argument("--contract-json", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    host_reachable = None
    try:
        if args.contract_json is not None:
            host_reachable = True
            report = Controller(json.loads(args.contract_json)).execute(args.action)
        else:
            config = legacy_contract(
                args.root,
                container=args.container,
                abi=args.abi,
                cache_root=args.cache_root,
            )
            if args.host == "local":
                host_reachable = True
                report = Controller(config).execute(args.action)
            else:
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", args.host):
                    raise ValueError("invalid backend host")
                command = shlex.join(
                    [
                        "/usr/bin/python3",
                        "-",
                        args.action,
                        "--contract-json",
                        json.dumps(config),
                    ]
                )
                host_reachable = False
                result = subprocess.run(
                    [
                        "ssh",
                        "-T",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=3",
                        args.host,
                        command,
                    ],
                    input=Path(__file__).read_text(),
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                host_reachable = result.returncode == 0
                if result.returncode or len(result.stdout) > 65536:
                    raise RuntimeError(
                        "backend host control unavailable; check the SSH connection"
                    )
                report = json.loads(result.stdout)
        print(json.dumps(report))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        message = (
            "backend host control unavailable; check the SSH connection"
            if host_reachable is False
            else str(error)[:512]
        )
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "state": "unavailable",
                    "host_control": {
                        "reachable": host_reachable,
                        "observed_at": time.time(),
                    },
                    "issues": [
                        "host_control_unavailable"
                        if host_reachable is False
                        else "host_status_failed"
                    ],
                    "error": message,
                }
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
