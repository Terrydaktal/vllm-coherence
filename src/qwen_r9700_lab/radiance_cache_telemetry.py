"""Bounded, content-free cache and round diagnostics. No added GPU waits.

Installed as qwen_radiance_cache_telemetry beside the pinned vLLM package.
The transfer adapter reuses timings which vLLM has already completed/read.
Nothing starts on import; configure() is called by the serving scheduler.
Optional generation timing adds two HIP markers per round, collected by the
writer after completion. It does not enable the stage profiler.
"""

# Diagnostic failures must not change a model/cache operation's result.
# ruff: noqa: BLE001

from __future__ import annotations

import atexit
import contextlib
import functools
import gc
import hashlib
import json
import math
import os
import resource
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

SCHEMA = "urn:coherence:cache-job-timings:v1"
DEFAULT_STATUS = "/dev/shm/qwen-radiance-fair-public"
MAX_BYTES = 16 * 1024 * 1024
MAX_PENDING = 4096
MAX_JOBS = 256
MAX_GPU_ROUNDS = 64
_recorder = None
_local = threading.local()
_round = {}
_gpu_origins = {}
_runtime_hooks_installed = False
_completion_baseline = None

# No paths, exception messages, keys, token values, pointers or tensor contents.
_IDENTITIES = {
    "chat_id",
    "generation",
    "request_id",
    "external_request_id",
    "http_request_id",
}
_COUNTS = {
    "round",
    "computed_tokens",
    "input_tokens",
    "scheduled_tokens",
    "job_id",
    "bytes",
    "block_count",
    "gc_generation",
    "collected",
    "uncollectable",
    "voluntary_switches",
    "involuntary_switches",
    "origin_thread_id",
    "endpoint_tokens",
    "hash_size",
    "cached_tokens",
    "dependency_count",
    "missing_dependencies",
    "pending_dependencies",
    "tier_index",
    "status_code",
}
_LABELS = {"job_kind", "direction", "reason", "cache_source", "outcome"}
_LABEL_VALUES = {
    "gpu",
    "filesystem",
    "store",
    "load",
    "after_cache_prepare",
    "long_response_recovery",
    "tail_flush",
    "shutdown",
    "generation",
    "prefill",
    "execute_return",
    "execute_failed",
    "sample_return",
    "sample_failed",
    "gpu_endpoint",
    "offload_endpoint",
    "gpu_blocks",
    "hit",
    "miss",
    "rejected",
    "loading",
    "no_endpoint",
    "response_end_disabled",
    "prompt_does_not_extend_endpoint",
    "multimodal_request",
    "prompt_embeddings",
    "lora_request",
    "prefix_cache_disabled",
    "prefix_identity_unavailable",
    "cache_salt_changed",
    "prefix_hash_changed",
    "partial_prefix_changed",
    "prefix_identity_changed",
    "endpoint_schema_mismatch",
    "invalid_endpoint_metadata",
    "invalid_endpoint_token_count",
    "invalid_endpoint_block_size",
    "endpoint_not_ahead",
    "aligned_endpoint",
    "unsupported_blocks_per_chunk",
    "hash_block_size_mismatch",
    "cache_group_count_mismatch",
    "cache_group_block_size_mismatch",
    "missing_dependencies",
    "pending_dependencies",
    "no_endpoint_tier",
    "matching_endpoint",
    "normal_prefix_lookup",
    "unspecified",
    "unknown",
}
_METRICS = {
    "gpu_elapsed_ms",
    "queue_ms",
    "lifetime_ms",
    "gpu_inter_round_gap_ms",
    "gpu_inter_prefill_gap_ms",
    "thread_user_ms",
    "thread_system_ms",
    "off_cpu_ms",
}


def fields(values):
    """A whitelist is enforced at the sink, including calls from runtime hooks."""
    result = {}
    for name, value in values.items():
        if name in _IDENTITIES and isinstance(value, str) and len(value) == 64:
            if all(c in "0123456789abcdef" for c in value):
                result[name] = value
        elif (
            name in _COUNTS
            and type(value) is int
            and value >= 0
            or name in _LABELS
            and isinstance(value, str)
            and value in _LABEL_VALUES
        ):
            result[name] = value
        elif name in _METRICS and isinstance(value, (int, float)):
            if math.isfinite(value) and value >= 0:
                result[name] = round(value, 6)
        elif (
            name in {"success", "indices_truncated", "stream_match", "streaming"}
            and type(value) is bool
        ):
            result[name] = value
        elif name in {"logical_ranges", "job_ids"} and isinstance(value, (tuple, list)):
            # Logical indices only, never physical addresses or cache-key bytes.
            if name == "job_ids":
                result[name] = [
                    v for v in value[:MAX_JOBS] if type(v) is int and v >= 0
                ]
            else:
                result[name] = [
                    [v for v in row[:64] if type(v) is int and v >= 0]
                    for row in value[:16]
                    if isinstance(row, (list, tuple))
                ]
    return result


class Recorder:
    def __init__(
        self,
        status_path,
        *,
        capacity=MAX_PENDING,
        max_bytes=MAX_BYTES,
        start=True,
        gc_events=True,
    ):
        self.path = Path(str(status_path) + "-cache-jobs.jsonl")
        self.health_path = Path(str(status_path) + "-cache-jobs-health.json")
        self.capacity, self.max_bytes = capacity, max_bytes
        self.trace_id = uuid.uuid4().hex
        self.pid = os.getpid()
        self.started_ns = time.monotonic_ns()
        try:
            self.clock_id = hashlib.sha256(
                Path("/proc/sys/kernel/random/boot_id").read_bytes()
            ).hexdigest()
        except OSError:
            self.clock_id = None
        self.epoch_offset_ns = time.time_ns() - time.monotonic_ns()
        self.pending = deque()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.sequence = self.written = self.dropped = self.write_errors = 0
        self.context_drops = 0
        self.gpu_hooks = False
        self.worker_first_work_hooks = False
        self.round_hook_attempt_ns = None
        self.round_sampler = None
        self.kfd_capture = None
        self.kfd_attempted = False
        self.main_thread_samples = self.round_sample_errors = 0
        self.source_hashes = {
            "recorder": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        }
        self.last_batch_ms = self.max_batch_ms = 0.0
        self.last_health = 0.0
        self.thread = None
        self.gc_starts = {}
        self.gc_callback = self._gc_callback
        if gc_events:
            gc.callbacks.append(self.gc_callback)
        if start:
            self.thread = threading.Thread(
                target=self._writer, name="cache-timing-writer", daemon=True
            )
            self.thread.start()

    def emit(self, stage, start_ns, end_ns, values=None, *, cpu_ns=None, faults=None):
        # A GC callback can interrupt a producer. Never wait on/reenter its lock.
        if not self.lock.acquire(blocking=False):
            self.dropped += 1
            return
        try:
            self.sequence += 1
            if len(self.pending) >= self.capacity:
                self.dropped += 1
                return
            active = _round
            event = {
                "schema": SCHEMA,
                "trace_id": self.trace_id,
                "pid": self.pid,
                "clock_id": self.clock_id,
                "sequence": self.sequence,
                "thread_id": threading.get_native_id(),
                "stage": stage,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "observed_at_ms": (end_ns + self.epoch_offset_ns) // 1_000_000,
                "duration_ms": round(max(0, end_ns - start_ns) / 1e6, 6),
                "active": fields(active),
                **fields(values or {}),
            }
            if cpu_ns is not None:
                event["thread_cpu_ms"] = round(max(0, cpu_ns) / 1e6, 6)
            if faults is not None:
                event["minor_faults"], event["major_faults"] = faults
            self.pending.append(event)
        finally:
            self.lock.release()

    def _gc_callback(self, phase, info):
        try:
            key = (threading.get_native_id(), info["generation"])
            if phase == "start":
                self.gc_starts[key] = (time.monotonic_ns(), dict(_round))
            else:
                started = self.gc_starts.pop(key, None)
                if started is not None:
                    ns, context = started
                    self.emit(
                        "python_gc",
                        ns,
                        time.monotonic_ns(),
                        {
                            **context,
                            "gc_generation": info["generation"],
                            "collected": info["collected"],
                            "uncollectable": info["uncollectable"],
                        },
                    )
        except Exception:
            self.dropped += 1

    def flush(self):
        # All JSON formatting and file I/O happen on the writer, not producers.
        if self.round_sampler is not None:
            try:
                self.round_sampler.collect()
            except Exception:
                self.round_sample_errors += 1
        with self.lock:
            batch = [self.pending.popleft() for _ in range(min(128, len(self.pending)))]
        started = time.monotonic_ns()
        if batch:
            try:
                payload = b"".join(
                    json.dumps(row, separators=(",", ":")).encode() + b"\n"
                    for row in batch
                )
                if (
                    self.path.exists()
                    and self.path.lstat().st_size + len(payload) > self.max_bytes
                ):
                    self.path.replace(str(self.path) + ".1")
                fd = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
                    0o600,
                )
                try:
                    view = memoryview(payload)
                    while view:
                        count = os.write(fd, view)
                        if count <= 0:
                            raise OSError("short diagnostic write")
                        view = view[count:]
                finally:
                    os.close(fd)
                self.written += len(batch)
            except Exception:
                self.write_errors += 1
                self.dropped += len(batch)
            self.last_batch_ms = (time.monotonic_ns() - started) / 1e6
            self.max_batch_ms = max(self.max_batch_ms, self.last_batch_ms)
            if (
                self.last_batch_ms >= 1
                and not self.stop.is_set()
                and any(row["stage"] != "recorder_batch" for row in batch)
            ):
                self.emit("recorder_batch", started, time.monotonic_ns())
        if time.monotonic() - self.last_health >= 1 or self.stop.is_set():
            self.last_health = time.monotonic()
            self._health()

    def _health(self):
        health = {
            "schema": SCHEMA,
            "trace_id": self.trace_id,
            "pid": self.pid,
            "clock_id": self.clock_id,
            "started_ns": self.started_ns,
            "updated_at_ms": time.time_ns() // 1_000_000,
            "written": self.written,
            "dropped": self.dropped,
            "write_errors": self.write_errors,
            "context_drops": self.context_drops,
            "gpu_hooks": self.gpu_hooks,
            "worker_first_work_hooks": self.worker_first_work_hooks,
            "generation_round_telemetry": generation_timings_enabled(),
            "main_thread_samples": self.main_thread_samples,
            "round_sample_errors": self.round_sample_errors,
            "gpu_rounds": self.round_sampler.health()
            if self.round_sampler is not None
            else None,
            "kfd_capture": self.kfd_capture.health()
            if self.kfd_capture is not None
            else None,
            "source_sha256": self.source_hashes,
            "pending": len(self.pending),
            "capacity": self.capacity,
            "max_file_bytes": self.max_bytes,
            "retained_files": 2,
            "writer_last_batch_ms": self.last_batch_ms,
            "writer_max_batch_ms": self.max_batch_ms,
        }
        temporary = str(self.health_path) + f".{self.pid}.tmp"
        try:
            fd = os.open(
                temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "w") as stream:
                json.dump(health, stream)
            os.replace(temporary, self.health_path)
        except Exception:
            self.write_errors += 1

    def _writer(self):
        while not self.stop.wait(0.1):
            self.flush()
        # Bounded best-effort shutdown; cache durability never depends on this.
        while self.pending:
            self.flush()
        self._health()

    def close(self):
        if self.kfd_capture is not None:
            self.kfd_capture.close()
        if self.gc_callback in gc.callbacks:
            gc.callbacks.remove(self.gc_callback)
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)


def configure(status_path=DEFAULT_STATUS):
    global _recorder
    if (_recorder is None or _recorder.pid != os.getpid()) and os.environ.get(
        "QWEN_CACHE_JOB_TELEMETRY", "1"
    ) != "0":
        _recorder = Recorder(status_path)
        atexit.register(_recorder.close)
    return _recorder


def trace_id():
    return _recorder.trace_id if _recorder is not None else None


def begin_round(context):
    global _round
    _round = fields(context)
    emit("round_start", **_round)


def generation_timings_enabled():
    return (
        _recorder is not None
        and os.environ.get("QWEN_GENERATION_ROUND_TELEMETRY", "1") != "0"
    )


def _thread_snapshot():
    usage = resource.getrusage(resource.RUSAGE_THREAD)
    return (
        time.thread_time_ns(),
        usage.ru_utime,
        usage.ru_stime,
        usage.ru_minflt,
        usage.ru_majflt,
        usage.ru_nvcsw,
        usage.ru_nivcsw,
    )


def _thread_delta(before, after):
    changes = tuple(b - a for a, b in zip(before, after, strict=True))
    if any(v < 0 for v in changes):
        raise ValueError("thread counters moved backwards")
    cpu_ns, user, system, minor, major, voluntary, involuntary = changes
    return (
        cpu_ns,
        (minor, major),
        {
            "thread_user_ms": user * 1000,
            "thread_system_ms": system * 1000,
            "voluntary_switches": voluntary,
            "involuntary_switches": involuntary,
        },
    )


def _consecutive(previous, current):
    return (
        all(
            previous.get(k) == current.get(k) and k in current
            for k in ("request_id", "chat_id", "generation")
        )
        and type(previous.get("round")) is int
        and current.get("round") == previous["round"] + 1
    )


def _consecutive_prefill(previous, current):
    """Only adjacent completed prompt ranges of this exact request may join."""
    return (
        all(
            previous.get(k) == current.get(k) and k in current
            for k in ("request_id", "chat_id", "generation", "input_tokens")
        )
        and previous.get("job_kind") == current.get("job_kind") == "prefill"
        and type(previous.get("computed_tokens")) is int
        and type(previous.get("scheduled_tokens")) is int
        and previous["scheduled_tokens"] > 0
        and current.get("computed_tokens")
        == previous["computed_tokens"] + previous["scheduled_tokens"]
    )


def complete_round(context, end_ns, *, contiguous):
    """RUSAGE_THREAD deltas over the same completed-step interval as Pi.

    This includes scheduler and host work outside the worker's forward call.
    It excludes handover/idle periods and other requests, rather than sampling
    only execute_model and missing a stall after the worker returned.
    """
    global _completion_baseline
    if not generation_timings_enabled():
        _completion_baseline = None
        return
    try:
        current = (
            fields(context),
            end_ns,
            threading.get_native_id(),
            _thread_snapshot(),
        )
        previous, _completion_baseline = _completion_baseline, current
        if (
            contiguous
            and previous is not None
            and previous[2] == current[2]
            and _consecutive(previous[0], current[0])
            and end_ns >= previous[1]
        ):
            cpu_ns, faults, values = _thread_delta(previous[3], current[3])
            values["off_cpu_ms"] = max(0, end_ns - previous[1] - cpu_ns) / 1e6
            _recorder.emit(
                "main_thread_round",
                previous[1],
                end_ns,
                {**current[0], **values},
                cpu_ns=cpu_ns,
                faults=faults,
            )
            _recorder.main_thread_samples += 1
    except Exception:
        _completion_baseline = None
        _recorder.round_sample_errors += 1


class GpuRoundTimings:
    """Two HIP events per round; only the writer queries completed events.

    The writer retains the last completed pair until the next round is read,
    so measuring an inter-round gap cannot read a recycled marker. Deques have
    one producer and one consumer in the pinned CPython/TP1 runtime; no serving
    thread waits for the collector or holds a lock across driver calls.
    """

    def __init__(self, cuda, recorder, *, capacity=MAX_GPU_ROUNDS):
        self.cuda, self.recorder = cuda, recorder
        self.free = deque(
            (cuda.Event(enable_timing=True), cuda.Event(enable_timing=True))
            for _ in range(capacity)
        )
        self.pending = deque()
        self.previous = None
        self.quarantined = []
        self.capacity = capacity
        self.completed = self.dropped = self.errors = 0

    def begin(self, context):
        try:
            pair = self.free.pop()
        except IndexError:
            self.dropped += 1
            return None
        try:
            stream = self.cuda.current_stream()
            item = {
                "pair": pair,
                "context": fields(context),
                "stream": stream,
                "stream_id": stream.cuda_stream,
                "start_ns": time.monotonic_ns(),
                "origin_thread_id": threading.get_native_id(),
                "before": _thread_snapshot(),
            }
            pair[0].record(stream)
            return item
        except Exception:
            self.errors += 1
            self.quarantined.append(pair)
        return None

    def end(self, item, *, success, reason):
        if item is None:
            return
        try:
            if threading.get_native_id() != item["origin_thread_id"]:
                raise ValueError("runner moved between threads")
            # An unexpected stream change is not repaired by inserting a wait.
            # Record on the original stream, but decline to report a round time.
            item["stream_match"] = (
                self.cuda.current_stream().cuda_stream == item["stream_id"]
            )
            item["pair"][1].record(item["stream"])
            item.update(
                end_ns=time.monotonic_ns(),
                after=_thread_snapshot(),
                success=success,
                reason=reason,
            )
            self.pending.append(item)
        except Exception:
            self.errors += 1
            self.quarantined.append(item["pair"])

    def collect(self):
        started = time.monotonic_ns()
        # Keep all event query/elapsed_time calls away from the serving thread.
        for _ in range(self.capacity):
            if not self.pending:
                break
            item = self.pending[0]
            try:
                if not item["pair"][1].query():
                    break
                values = {
                    **item["context"],
                    "origin_thread_id": item["origin_thread_id"],
                    "success": item["success"],
                    "reason": item["reason"],
                    "stream_match": item["stream_match"],
                }
                if item["stream_match"]:
                    values["gpu_elapsed_ms"] = item["pair"][0].elapsed_time(
                        item["pair"][1]
                    )
                    previous = self.previous
                    prefill = item["context"].get("job_kind") == "prefill"
                    if (
                        previous is not None
                        and previous["success"]
                        and item["success"]
                        and previous["stream_match"]
                        and previous["stream_id"] == item["stream_id"]
                        and (
                            _consecutive_prefill(previous["context"], item["context"])
                            if prefill
                            else (
                                previous["context"].get("job_kind") != "prefill"
                                and _consecutive(previous["context"], item["context"])
                            )
                        )
                    ):
                        gap = (
                            "gpu_inter_prefill_gap_ms"
                            if prefill
                            else "gpu_inter_round_gap_ms"
                        )
                        values[gap] = previous["pair"][1].elapsed_time(item["pair"][0])
                cpu_ns, faults, resources = _thread_delta(item["before"], item["after"])
                self.recorder.emit(
                    "gpu_prefill"
                    if item["context"].get("job_kind") == "prefill"
                    else "gpu_round",
                    item["start_ns"],
                    item["end_ns"],
                    {**values, **resources},
                    cpu_ns=cpu_ns,
                    faults=faults,
                )
                self.completed += 1
            except Exception:
                # Completion is uncertain: quarantine rather than reuse/wait.
                self.errors += 1
                self.quarantined.append(item["pair"])
                item = None
            self.pending.popleft()
            if self.previous is not None:
                self.free.append(self.previous["pair"])
            self.previous = item
        end = time.monotonic_ns()
        if end - started >= 1_000_000:
            self.recorder.emit("gpu_observer_poll", started, end)

    def health(self):
        return {
            "capacity": self.capacity,
            "pending": len(self.pending),
            "free": len(self.free),
            "completed": self.completed,
            "dropped": self.dropped,
            "errors": self.errors,
            "quarantined": len(self.quarantined),
        }


def install_round_hooks(owner, cuda):
    if not generation_timings_enabled() or getattr(
        owner, "_qwen_cache_round_hooks", False
    ):
        return
    execute, sample = owner.execute_model, owner.sample_tokens
    sampler = _recorder.round_sampler = GpuRoundTimings(cuda, _recorder)

    @functools.wraps(execute)
    def execute_wrapped(self, *args, **kwargs):
        item = None
        first_work = {}
        try:
            output = args[0] if args else kwargs.get("scheduler_output")
            fair = getattr(output, "qwen_fair", None) or {}
            context = fair.get("cache_timing_context", {})
            count = getattr(output, "total_num_scheduled_tokens", 0)
            dummy = kwargs.get("dummy_run", args[2] if len(args) > 2 else False)
            # Keep host spans for every prefill chunk and the first generation
            # step. Prefill GPU markers use the same bounded asynchronous pool
            # as decode; event queries remain off the serving thread.
            if (
                not dummy
                and not fair.get("barrier")
                and context.get("request_id")
                and context.get("round", 1) <= 1
                and count > 0
                and len(getattr(output, "num_scheduled_tokens", {})) == 1
            ):
                first_work = {**context, "scheduled_tokens": count}
            if (
                generation_timings_enabled()
                and not dummy
                and not fair.get("barrier")
                and context.get("request_id")
                and count > 0
                and (count <= 16 or first_work)
                and len(getattr(output, "num_scheduled_tokens", {})) == 1
            ):
                if count <= 16 and not _recorder.kfd_attempted:
                    _recorder.kfd_attempted = True
                    if os.environ.get("QWEN_KFD_CAPTURE_PATH"):
                        try:
                            from qwen_radiance_kfd_trace import start_if_approved
                        except ImportError:
                            from qwen_r9700_lab.radiance_kfd_trace import (
                                start_if_approved,
                            )
                        _recorder.kfd_capture = start_if_approved(_recorder.trace_id)
                gpu_context = {**context, "scheduled_tokens": count}
                computed = context.get("computed_tokens")
                total = context.get("input_tokens")
                if (
                    type(computed) is int and type(total) is int and computed < total
                ) or count > 16:
                    gpu_context["job_kind"] = "prefill"
                item = sampler.begin(gpu_context)
        except Exception:
            sampler.recorder.round_sample_errors += 1
        self._qwen_cache_round_item = item
        self._qwen_first_work_context = first_work
        try:
            with (
                span("worker_execute", resources=True, **first_work)
                if first_work
                else _NO_SPAN
            ):
                result = execute(self, *args, **kwargs)
        except BaseException:
            sampler.end(item, success=False, reason="execute_failed")
            self._qwen_cache_round_item = None
            self._qwen_first_work_context = {}
            raise
        if getattr(self, "execute_model_state", None) is None:
            sampler.end(item, success=True, reason="execute_return")
            self._qwen_cache_round_item = None
            self._qwen_first_work_context = {}
        _tag_first_work_output(result, first_work)
        return result

    @functools.wraps(sample)
    def sample_wrapped(self, *args, **kwargs):
        success = False
        try:
            context = getattr(self, "_qwen_first_work_context", {})
            with (
                span("worker_sample", resources=True, **context)
                if context
                else _NO_SPAN
            ):
                result = sample(self, *args, **kwargs)
            _tag_first_work_output(result, context)
            success = True
            return result
        finally:
            item = getattr(self, "_qwen_cache_round_item", None)
            self._qwen_cache_round_item = None
            self._qwen_first_work_context = {}
            sampler.end(
                item,
                success=success,
                reason="sample_return" if success else "sample_failed",
            )

    owner.execute_model, owner.sample_tokens = execute_wrapped, sample_wrapped
    owner._qwen_cache_round_hooks = True
    _recorder.worker_first_work_hooks = True


def _tag_first_work_output(result, context):
    # The context belongs to this actual asynchronous result, never to a global
    # "current chat". Class-level wrapping avoids a bound-method reference cycle
    # that could otherwise keep its GPU tensors alive after output collection.
    if context and getattr(type(result), "_qwen_first_work_output_hooks", False):
        try:
            result._qwen_first_work_context = fields(context)
        except Exception:
            _recorder.round_sample_errors += 1


def install_async_output_hooks(owner):
    """Observe the existing completion wait without adding GPU operations."""
    if getattr(owner, "_qwen_first_work_output_hooks", False):
        return
    original = owner.get_output

    @functools.wraps(original)
    def get_output(self, *args, **kwargs):
        context = getattr(self, "_qwen_first_work_context", None)
        with (
            span("worker_get_output", resources=True, **context)
            if context
            else _NO_SPAN
        ):
            return original(self, *args, **kwargs)

    owner.get_output = get_output
    owner._qwen_first_work_output_hooks = True


def emit(stage, **values):
    if _recorder is not None:
        try:
            now = time.monotonic_ns()
            _recorder.emit(stage, now, now, values)
        except Exception:
            _recorder.dropped += 1


def emit_at(stage, start_ns, end_ns, **values):
    """Persist a boundary/span without I/O, waits or exception propagation."""
    if _recorder is not None:
        try:
            _recorder.emit(stage, start_ns, end_ns, values)
        except Exception:
            _recorder.dropped += 1


class Span:
    def __init__(self, stage, values, minimum_ms=0, resources=False):
        self.recorder, self.stage = _recorder, stage
        self.values = {**getattr(_local, "context", {}), **values}
        self.minimum_ms, self.resources = minimum_ms, resources

    def __enter__(self):
        try:
            self.start = time.monotonic_ns()
            self.cpu = time.thread_time_ns()
            self.faults = (
                resource.getrusage(resource.RUSAGE_THREAD) if self.resources else None
            )
        except Exception:
            self.recorder.dropped += 1
            self.start = None
        return self

    def __exit__(self, kind, _error, _tb):
        if self.start is None:
            return False
        try:
            end = time.monotonic_ns()
            if (end - self.start) / 1e6 >= self.minimum_ms:
                faults = None
                if self.faults is not None:
                    after = resource.getrusage(resource.RUSAGE_THREAD)
                    faults = (
                        after.ru_minflt - self.faults.ru_minflt,
                        after.ru_majflt - self.faults.ru_majflt,
                    )
                self.recorder.emit(
                    self.stage,
                    self.start,
                    end,
                    {**self.values, "success": kind is None},
                    cpu_ns=time.thread_time_ns() - self.cpu,
                    faults=faults,
                )
        except Exception:
            self.recorder.dropped += 1
        return False


_NO_SPAN = contextlib.nullcontext()


def span(stage, *, minimum_ms=0, resources=False, **values):
    return (
        Span(stage, values, minimum_ms, resources)
        if _recorder is not None
        else _NO_SPAN
    )


@contextlib.contextmanager
def _timed_lock(mutex, stage):
    with span(stage, minimum_ms=0.05):
        mutex.acquire()
    try:
        yield
    finally:
        mutex.release()


def lock(mutex, stage="tier_lock"):
    return _timed_lock(mutex, stage) if _recorder is not None else mutex


def measured(stage, *, minimum_ms=0, **values):
    def decorate(operation):
        @functools.wraps(operation)
        def wrapped(*args, **kwargs):
            with span(stage, minimum_ms=minimum_ms, **values):
                return operation(*args, **kwargs)

        return wrapped

    return decorate


def request_context(req_id, chat, job_id, *, is_store, size, block_count):
    request_id = hashlib.sha256(str(req_id).encode()).hexdigest()
    origin = _round if _round.get("request_id") == request_id else {}
    return {
        **origin,
        "request_id": request_id,
        "chat_id": chat["id"],
        "generation": chat["generation"],
        "job_id": int(job_id),
        "job_kind": "filesystem",
        "direction": "store" if is_store else "load",
        "bytes": size,
        "block_count": block_count,
    }


def task(operation, context, queued_ns):
    """Capture the submitter context; do not attribute a parked chat to the active one."""
    if _recorder is None:
        return operation()
    prior = getattr(_local, "context", {})
    _local.context = context
    try:
        now = time.monotonic_ns()
        try:
            _recorder.emit("cpu_queue", queued_ns, now, context)
        except Exception:
            _recorder.dropped += 1
        with span("cpu_job", resources=True):
            return operation()
    finally:
        _local.context = prior


def install_transfer_hooks(handler):
    """Wrap the pinned handler without adding HIP operations or consuming results.

    get_finished() already queries readiness and reads start/end elapsed time.
    The observed lifetime also includes queue/dependency waits and polling delay;
    it must not be reported as the duration of DMA alone.
    """
    if getattr(handler, "_qwen_cache_timing_hooks", False):
        return
    submit, finished, wait = handler.transfer_async, handler.get_finished, handler.wait

    @functools.wraps(submit)
    def traced_submit(self, job_id, src_spec, dst_spec):
        if _recorder is None:
            return submit(self, job_id, src_spec, dst_spec)
        context = {
            **_gpu_origins.pop(job_id, _round),
            "job_kind": "gpu",
            "job_id": int(job_id),
            "direction": "store" if self.gpu_to_cpu else "load",
        }
        try:
            spec = src_spec if self.gpu_to_cpu else dst_spec
            indices, sizes = spec.block_indices, spec.group_sizes
            context["logical_ranges"] = [
                [int(first), int(count)]
                for first, count in zip(indices[:16], sizes[:16], strict=True)
            ]
            context["indices_truncated"] = len(indices) > 16
            context["block_count"] = sum(int(count) for count in sizes)
        except Exception:
            _recorder.context_drops += 1
        started = time.monotonic_ns()
        with span("gpu_submit", resources=True, **context) as measurement:
            result = submit(self, job_id, src_spec, dst_spec)
            if result:
                try:
                    transfer = self._transfers[-1]
                    context["bytes"] = int(transfer.num_bytes)
                    measurement.values.update(context)
                    jobs = getattr(self, "_qwen_cache_timing_jobs", None)
                    if jobs is None:
                        jobs = self._qwen_cache_timing_jobs = {}
                    if len(jobs) >= MAX_JOBS:
                        jobs.pop(next(iter(jobs)))
                        _recorder.context_drops += 1
                    jobs[job_id] = (started, context)
                except Exception:
                    _recorder.context_drops += 1
            return result

    @functools.wraps(finished)
    def traced_finished(self):
        if _recorder is None:
            return finished(self)
        with span("gpu_poll", minimum_ms=1):
            results = finished(self)
        now = time.monotonic_ns()
        for result in results:
            try:
                origin = getattr(self, "_qwen_cache_timing_jobs", {}).pop(
                    result.job_id, None
                )
                if origin is not None:
                    started, context = origin
                    values = {
                        **context,
                        "success": bool(result.success),
                        "lifetime_ms": (now - started) / 1e6,
                    }
                    if result.transfer_time is not None:
                        values["gpu_elapsed_ms"] = result.transfer_time * 1000
                    _recorder.emit("gpu_complete", started, now, values)
                else:
                    _recorder.context_drops += 1
            except Exception:
                _recorder.dropped += 1
        return results

    @functools.wraps(wait)
    def traced_wait(self, job_ids):
        if _recorder is None:
            return wait(self, job_ids)
        with span(
            "gpu_wait",
            job_kind="gpu",
            **_round,
            job_ids=sorted(job_ids),
            direction="store" if self.gpu_to_cpu else "load",
        ):
            return wait(self, job_ids)

    handler.transfer_async, handler.get_finished, handler.wait = (
        traced_submit,
        traced_finished,
        traced_wait,
    )
    handler._qwen_cache_timing_hooks = True


def install_connector_hooks(connector):
    if getattr(connector, "_qwen_cache_timing_hooks", False):
        return

    def remember(jobs):
        if _recorder is None:
            return
        for job_id, entry in jobs.items():
            request_id = hashlib.sha256(str(entry.req_id).encode()).hexdigest()
            origin = _round if _round.get("request_id") == request_id else {}
            if job_id not in _gpu_origins and len(_gpu_origins) >= MAX_JOBS:
                _gpu_origins.pop(next(iter(_gpu_origins)))
                _recorder.context_drops += 1
            _gpu_origins[job_id] = {**origin, "request_id": request_id}

    for name, job_field in (
        ("prepare_store_kv", "store_jobs"),
        ("start_kv_transfers", "load_jobs"),
        ("handle_preemptions", "store_jobs"),
    ):
        original = getattr(connector, name)

        def wrap(original, job_field):
            @functools.wraps(original)
            def method(self, metadata):
                try:
                    remember(getattr(metadata, job_field))
                except Exception:
                    if _recorder is not None:
                        _recorder.context_drops += 1
                return original(self, metadata)

            return method

        setattr(connector, name, wrap(original, job_field))
    connector._qwen_cache_timing_hooks = True


def install_runtime_hooks():
    global _runtime_hooks_installed
    if _recorder is None:
        return
    if not _runtime_hooks_installed:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
            OffloadingConnectorWorker,
        )
        from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

        install_transfer_hooks(SingleDirectionOffloadingHandler)
        install_connector_hooks(OffloadingConnectorWorker)
        for name, cls in (
            ("transfer_handler", SingleDirectionOffloadingHandler),
            ("connector", OffloadingConnectorWorker),
        ):
            _recorder.source_hashes[name] = hashlib.sha256(
                Path(sys.modules[cls.__module__].__file__).read_bytes()
            ).hexdigest()
        _runtime_hooks_installed = True
        _recorder.gpu_hooks = True
    if generation_timings_enabled() and not _recorder.worker_first_work_hooks:
        # A failed early import must not permanently remove cold-prefill timing.
        # Retry at most once per second, never rewrap already installed methods.
        now = time.monotonic_ns()
        if (
            _recorder.round_hook_attempt_ns is not None
            and now - _recorder.round_hook_attempt_ns < 1_000_000_000
        ):
            return
        _recorder.round_hook_attempt_ns = now
        try:
            import torch
            from vllm.v1.worker.gpu.async_utils import AsyncOutput
            from vllm.v1.worker.gpu.model_runner import GPUModelRunner

            install_async_output_hooks(AsyncOutput)
            install_round_hooks(GPUModelRunner, torch.cuda)
            _recorder.worker_first_work_hooks = bool(
                getattr(GPUModelRunner, "_qwen_cache_round_hooks", False)
            )
            _recorder.source_hashes["round_runner"] = hashlib.sha256(
                Path(sys.modules[GPUModelRunner.__module__].__file__).read_bytes()
            ).hexdigest()
        except Exception:
            _recorder.round_sample_errors += 1
