"""Opt-in, bounded KFD events for this worker, without SDK/GPU instrumentation."""

import fcntl
import hashlib
import json
import os
import select
import stat
import struct
import threading
import time
from pathlib import Path

# Linux KFD SMI UAPI: _IOWR('K', 0x1f, two u32 values).
SMI_IOCTL = (3 << 30) | (8 << 16) | (ord("K") << 8) | 0x1F
# Migration, page faults, queue eviction/restoration, and GPU unmapping.
# Never request ALL_PROCESS: the kernel filters this descriptor to our process.
EVENT_MASK = sum(1 << (event - 1) for event in range(5, 12))
MAX_SECONDS = 300
MAX_BYTES = 64 * 1024 * 1024


def open_events(gpu_id):
    with open("/dev/kfd", "rb", buffering=0) as device:
        args = bytearray(struct.pack("=II", gpu_id, 0))
        fcntl.ioctl(device.fileno(), SMI_IOCTL, args, True)
    _, fd = struct.unpack("=II", args)
    try:
        os.set_inheritable(fd, False)
        os.set_blocking(fd, False)
        if os.write(fd, struct.pack("=Q", EVENT_MASK)) != 8:
            raise OSError("incomplete KFD event mask write")
    except BaseException:
        os.close(fd)
        raise
    return fd


def clocks():
    return {
        "monotonic_ns": time.monotonic_ns(),
        "boottime_ns": time.clock_gettime_ns(time.CLOCK_BOOTTIME),
        "wall_ns": time.time_ns(),
    }


def gpu_ids():
    return sorted(
        {
            int(p.read_text())
            for p in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/gpu_id")
        }
        - {0}
    )


class Capture:
    def __init__(self, path, trace_id, *, seconds=300, max_bytes=MAX_BYTES):
        if not 0 < seconds <= MAX_SECONDS or not 65536 <= max_bytes <= MAX_BYTES:
            raise ValueError("KFD capture exceeds approved bounds")
        self.path = Path(path)
        self.trace_id, self.seconds, self.max_bytes = trace_id, seconds, max_bytes
        self.pid = os.getpid()
        self.state = "pending"
        self.error_errno = None
        self.bytes = 0
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run, name="kfd-events", daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)

    def health(self):
        return {
            "state": self.state,
            "pid": self.pid,
            "bytes": self.bytes,
            "duration_limit_seconds": self.seconds,
            "byte_limit": self.max_bytes,
            "error_errno": self.error_errno,
        }

    def run(self):
        fds = []
        output = None
        started = clocks()
        info = {
            "schema": "urn:coherence:kfd-process-events:v1",
            "pid": self.pid,
            "trace_id": self.trace_id,
            "event_mask": EVENT_MASK,
            "start_clock": started,
            "duration_limit_seconds": self.seconds,
            "byte_limit": self.max_bytes,
            "all_process_events": False,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "clock_note": "Queue records use driver boottime timestamps; clock pairs bind them to application monotonic time. Other event producers must be checked separately.",
            "coverage_limit": "The driver FIFO may drop events without a loss counter; absence alone is not proof.",
            "scope": "Numeric kernel events only; no chat, token or tensor contents; no GPU API calls or timing markers.",
        }
        try:
            parent = self.path.parent
            # Diagnostic destinations are pre-created by the host operator.
            if parent.is_symlink():
                raise ValueError("unsafe KFD capture directory")
            parent = parent.resolve(strict=True)
            mode = parent.stat()
            if mode.st_uid != os.getuid() or mode.st_mode & 0o077:
                raise ValueError("KFD capture directory must be owner-only")
            self.path = parent / self.path.name
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ValueError("KFD capture must be a regular file")
            output = os.fdopen(descriptor, "wb", buffering=0)
            nodes = gpu_ids()
            if not nodes:
                raise ValueError("no KFD GPU node")
            for gpu_id in nodes:
                fds.append(open_events(gpu_id))
            info["gpu_ids"] = nodes
            self.state = "capturing"
            # Leave a fixed allowance for the small immutable manifest.
            budget = self.max_bytes - 65536
            deadline = started["boottime_ns"] + int(self.seconds * 1e9)
            while not self.stop.is_set():
                remaining = (deadline - clocks()["boottime_ns"]) / 1e9
                if remaining <= 0:
                    self.state = "duration_limit"
                    break
                ready, _, _ = select.select(fds, [], [], min(remaining, 0.1))
                for fd in ready:
                    try:
                        data = os.read(fd, 8192)
                    except BlockingIOError:
                        continue
                    if not data:
                        raise OSError("KFD event stream ended")
                    if len(data) > budget - self.bytes:
                        self.state = "size_limit"
                        self.stop.set()
                        break
                    output.write(data)
                    self.bytes += len(data)
            if self.state == "capturing":
                self.state = "stopped"
        except (OSError, ValueError) as error:
            self.state = "failed"
            self.error_errno = getattr(error, "errno", None)
            info["error_type"] = type(error).__name__
        finally:
            for fd in fds:
                os.close(fd)
            if output is not None:
                output.close()
                self.path.chmod(0o400)
                info["sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
                info.update(self.health(), end_clock=clocks())
                manifest = self.path.with_name(self.path.name + ".json")
                with manifest.open("x") as stream:
                    json.dump(info, stream, indent=2)
                    stream.write("\n")
                manifest.chmod(0o400)


def start_if_approved(trace_id):
    path = os.environ.get("QWEN_KFD_CAPTURE_PATH")
    if not path:
        return None
    seconds = int(os.environ.get("QWEN_KFD_CAPTURE_SECONDS", "300"))
    return Capture(path, trace_id, seconds=seconds).start()
