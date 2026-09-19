# QWEN_ASSURANCE_ONLY_BEGIN: runtime-stage-hooks
"""Opt-in GPU-event accounting for the pinned vLLM runtime.

The kernel adapters already report their own intervals.  This module adds the
missing outer boundaries around a vLLM worker forward, draft proposal, and
sampler call.  It is deliberately opt-in: importing it does nothing unless
``QWEN_STAGE_TIMING=1`` or ``QWEN_OUTER_STAGE_TIMING=1`` is set, and the
wrappers are installed lazily after the vLLM worker classes have been imported.

``QWEN_STAGE_TIMING=1`` preserves the original component-heavy diagnostic.
``QWEN_OUTER_STAGE_TIMING=1`` installs only the six low-observer-overhead M8
boundaries used by the current decode decomposition.  The latter reuses a
fixed ring of HIP events instead of constructing event objects in every call.
PyTorch exposes these ROCm events through its ``torch.cuda`` compatibility
namespace; this does not require NVIDIA CUDA.

When ``QWEN_ROUND_EVENT_TELEMETRY=1`` is set alongside the outer mode, one
bounded, content-free JSONL record is written for every measured round.  It
contains the scheduled shape, stream identity, ordered HIP GPU-event markers,
event durations, and the gaps between adjacent markers.  Set
``QWEN_ROUND_EVENT_TELEMETRY_SYNC=1`` for a complete record immediately after
each round; this waits for the round-end event and reports the wait separately.
The default is asynchronous collection, which leaves unavailable event
values explicitly null rather than adding an unreported synchronization to a
serving run.

The hook is diagnostic only.  It never changes tensors or scheduling decisions
and it is safe to leave installed for a short warm profiling run.  The process
exit line is consumed by ``scripts/summarize_stage_timing.py``.
"""

from __future__ import annotations

import atexit
import functools
import json
import math
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

_LEGACY_ENABLED = os.environ.get("QWEN_STAGE_TIMING", "0") == "1"
_OUTER_ONLY = os.environ.get("QWEN_OUTER_STAGE_TIMING", "0") == "1"
_ENABLED = _LEGACY_ENABLED or _OUTER_ONLY
_MODE_CONFLICT = _LEGACY_ENABLED and _OUTER_ONLY
_MODEL_HOOKS_INSTALLED = False
_COMPONENT_HOOKS_INSTALLED = False
_DRAFT_HOOK_INSTALLED = False
_SCHEDULER_HOOK_INSTALLED = False
_SAMPLE_CORE_HOOK_INSTALLED = False
_TARGET_LOGITS_HOOK_INSTALLED = False
_REJECTION_HOOK_INSTALLED = False
_ROUND_HOOK_INSTALLED = False
_STATE_COMMIT_HOOK_INSTALLED = False
_PENDING_LOGGED = False
_READY_LOGGED = False
_LAST_CRITICAL_MISSING: tuple[str, ...] | None = None
_LOCK = threading.Lock()
_TOTALS: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])


_PENDING_EVENTS: list[tuple[str, Any, Any]] = []
_EVENT_RING_SIZE = 1_024
_EVENT_PAIRS_PER_ROUND_RESERVE = 16
_EVENT_RING: list[tuple[Any, Any]] | None = None
_EVENT_RING_CURSOR = 0
_EVENT_RING_CAPPED_LOGGED = False
_OUTER_ROUND_ACTIVE = False
_OUTER_COMMIT_PENDING = False
_OUTER_COMMIT_LABEL = "scheduler.commit"
_OUTER_REJECTION_SEEN = False
_OUTER_CURRENT_EVENTS: list[tuple[str, Any, Any]] = []
_OUTER_EVENT_ORDER: list[tuple[str, Any, str]] = []
_PREVIOUS_ROUND_END: Any | None = None
_PREVIOUS_ROUND_STREAM: str | None = None
_INTER_ROUND_MARKERS: list[Any] | None = None
_INTER_ROUND_MARKER_CURSOR = 0
_ROUND_EVENT_DROPS = 0
_ROUND_EVENT_TELEMETRY = os.environ.get("QWEN_ROUND_EVENT_TELEMETRY", "0") == "1"
_ROUND_EVENT_TELEMETRY_SYNC = (
    os.environ.get("QWEN_ROUND_EVENT_TELEMETRY_SYNC", "0") == "1"
)
# v2 names the event section by its hardware-neutral role and declares that
# this deployment records ROCm/HIP events rather than NVIDIA CUDA events.
ROUND_EVENT_LOG_SCHEMA = "urn:qwen-r9700:decode-round-gpu:v2"
ROUND_EVENT_LOG_MAX_BYTES = 8 * 1024 * 1024


def _stream_id(stream: Any) -> str:
    """Return an opaque, stable-for-process stream label without payload data."""
    value = getattr(stream, "cuda_stream", stream)
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _scheduled_shape(scheduler_output: Any) -> dict[str, Any]:
    """Extract only bounded scheduler shape metadata; never retain request text."""
    total = _scheduled_tokens(scheduler_output)
    raw = getattr(scheduler_output, "num_scheduled_tokens", None)
    counts: list[int] = []
    if hasattr(raw, "items"):
        for value in raw.values():
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                counts.append(value)
    elif total is not None and total > 0:
        counts = [total]
    counts.sort()
    drafts = getattr(scheduler_output, "scheduled_spec_decode_tokens", None)
    draft_widths: list[int] = []
    if hasattr(drafts, "items"):
        for value in drafts.values():
            try:
                width = len(value)
            except TypeError:
                continue
            if width > 0:
                draft_widths.append(width)
    draft_widths.sort()
    return {
        "mode": (
            "decode"
            if total is not None and 1 <= total <= 16
            else "prefill_or_other"
            if total is not None
            else "unknown"
        ),
        "scheduled_tokens": total,
        "request_count": len(counts),
        "tokens_per_request": counts,
        "draft_widths": draft_widths,
    }


def _prepare_event_ring() -> None:
    global _EVENT_RING, _INTER_ROUND_MARKERS
    if not _ENABLED or not _OUTER_ONLY or _EVENT_RING is not None:
        return
    import torch

    _EVENT_RING = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(_EVENT_RING_SIZE)
    ]
    if _ROUND_EVENT_TELEMETRY:
        # Two dedicated end markers ping-pong across rounds. They keep the
        # previous round's HIP event alive while the next round's fixed event
        # ring is reused, which makes the inter-round queue gap measurable.
        _INTER_ROUND_MARKERS = [
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        ]


def _event_pair(event_label: str | None = None):
    if not _ENABLED:
        return None
    import torch

    stream = torch.cuda.current_stream()
    if _OUTER_ONLY:
        global _EVENT_RING_CURSOR
        _prepare_event_ring()
        assert _EVENT_RING is not None
        if len(_EVENT_RING) <= _EVENT_RING_CURSOR:
            raise RuntimeError(
                "outer stage-timing event ring exhausted; refuse to reuse pending events"
            )
        start, end = _EVENT_RING[_EVENT_RING_CURSOR]
        _EVENT_RING_CURSOR += 1
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    if (
        event_label is not None
        and _OUTER_ONLY
        and _ROUND_EVENT_TELEMETRY
        and _OUTER_ROUND_ACTIVE
    ):
        _OUTER_EVENT_ORDER.append(
            (event_label + ".start", start, _stream_id(stream))
        )
    return start, end


def _finish(label: str, interval) -> None:
    if interval is None:
        return
    import torch

    start, end = interval
    stream = torch.cuda.current_stream()
    end.record(stream)
    with _LOCK:
        if _OUTER_ONLY and _OUTER_ROUND_ACTIVE:
            _OUTER_CURRENT_EVENTS.append((label, start, end))
            if _ROUND_EVENT_TELEMETRY:
                _OUTER_EVENT_ORDER.append(
                    (label + ".end", end, _stream_id(stream))
                )
        else:
            _PENDING_EVENTS.append((label, start, end))


def _record_event_marker(label: str):
    """Record one ordered marker using the preallocated event ring."""
    interval = _event_pair(label)
    return None if interval is None else interval[0]


def _record_inter_round_end() -> tuple[Any | None, str | None]:
    global _INTER_ROUND_MARKER_CURSOR
    if not _ROUND_EVENT_TELEMETRY or _INTER_ROUND_MARKERS is None:
        return None, None
    import torch

    stream = torch.cuda.current_stream()
    marker = _INTER_ROUND_MARKERS[_INTER_ROUND_MARKER_CURSOR]
    _INTER_ROUND_MARKER_CURSOR = (_INTER_ROUND_MARKER_CURSOR + 1) % len(_INTER_ROUND_MARKERS)
    marker.record(stream)
    return marker, _stream_id(stream)


def _append_round_event_log(path: str, record: dict[str, Any]) -> None:
    """Append a bounded diagnostic record to the backend's tmpfs feed."""
    payload = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
    destination = Path(path)
    if destination.exists() and destination.stat().st_size + len(payload) > ROUND_EVENT_LOG_MAX_BYTES:
        rotated = destination.with_name(destination.name + ".1")
        try:
            rotated.unlink()
        except FileNotFoundError:
            pass
        destination.replace(rotated)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short GPU round telemetry write")
            view = view[written:]
    finally:
        os.close(fd)


def _event_elapsed(start: Any, end: Any) -> float | None:
    try:
        value = float(start.elapsed_time(end))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return value if value >= 0 and math.isfinite(value) else None


def _round_event_record(context: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Build a content-free per-round record from ordered HIP GPU events."""
    durations = []
    for label, start, end in _OUTER_CURRENT_EVENTS:
        durations.append({"label": label, "ms": _event_elapsed(start, end)})
    gaps = []
    for previous, current in pairwise(_OUTER_EVENT_ORDER):
        previous_label, previous_event, previous_stream = previous
        current_label, current_event, current_stream = current
        same_stream = previous_stream == current_stream
        gaps.append(
            {
                "from": previous_label,
                "to": current_label,
                "same_stream": same_stream,
                "ms": _event_elapsed(previous_event, current_event)
                if same_stream
                else None,
            }
        )
    previous_end = context.get("previous_round_end_event")
    previous_stream = context.get("previous_round_stream")
    current_start = context.get("round_start_event")
    current_stream = context.get("stream")
    inter_round_gap = (
        _event_elapsed(previous_end, current_start)
        if previous_end is not None
        and current_start is not None
        and previous_stream == current_stream
        else None
    )
    sync_count = context.get("runner_sync_count")
    sync_after = context.get("runner_sync_count_after")
    return {
        "schema": ROUND_EVENT_LOG_SCHEMA,
        "pid": os.getpid(),
        "observed_at_ms": int(time.time() * 1000),
        "round": context["round"],
        "phase": prefix[:-1] if prefix.endswith(".") else (prefix or "verification"),
        "scheduled_shape": context["scheduled_shape"],
        "gpu_events": {
            "backend": "rocm-hip",
            "device": context.get("device"),
            "stream": context.get("stream"),
            "event_status": context.get("event_status", "unknown"),
            "event_order": [label for label, _, _ in _OUTER_EVENT_ORDER],
            "durations_ms": durations,
            "gaps_ms": gaps,
        },
        "dispatch": {
            "host_ms": context.get("host_ms"),
            "telemetry_wait_ms": context.get("telemetry_wait_ms"),
            "previous_round_end_to_start_ms": inter_round_gap,
        },
        "queue_sync": {
            "count_at_start": sync_count,
            "count_at_end": sync_after,
            "count_delta": (
                sync_after - sync_count
                if isinstance(sync_count, int) and isinstance(sync_after, int)
                else None
            ),
            "mode": context.get("runner_sync_mode"),
            "reason": context.get("runner_sync_reason"),
            "last_elapsed_ms": context.get("runner_sync_elapsed_ms"),
        },
    }


def _write_round_event_telemetry(context: dict[str, Any], prefix: str) -> str | None:
    global _ROUND_EVENT_DROPS
    if not _ROUND_EVENT_TELEMETRY or not context.get("status_path"):
        return None
    event_status = "complete"
    waited = 0.0
    round_end = context.get("round_end_event")
    if _ROUND_EVENT_TELEMETRY_SYNC and round_end is not None:
        started = time.perf_counter()
        try:
            round_end.synchronize()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            event_status = "unavailable"
        waited = (time.perf_counter() - started) * 1000.0
    elif round_end is not None:
        query = getattr(round_end, "query", None)
        if callable(query):
            try:
                event_status = "complete" if query() else "pending"
            except (AttributeError, RuntimeError, TypeError, ValueError):
                event_status = "unavailable"
        else:
            event_status = "pending"
    context["event_status"] = event_status
    context["telemetry_wait_ms"] = round(waited, 3)
    record = _round_event_record(context, prefix)
    record["dropped_records_before"] = _ROUND_EVENT_DROPS
    try:
        _append_round_event_log(
            str(context["status_path"]) + "-gpu-rounds.jsonl", record
        )
        _ROUND_EVENT_DROPS = 0
    except (OSError, ValueError) as exc:
        # Diagnostics must not abort inference. The next round remains useful;
        # the missing record is explicit in the process log.
        print(
            f"[qwen-runtime] GPU round telemetry write failed: {type(exc).__name__}",
            flush=True,
        )
        _ROUND_EVENT_DROPS += 1
    return event_status


def _account_closed_round_events(prefix: str) -> None:
    """Account a synchronized round before its fixed event slots are reused."""
    for label, start, end in _OUTER_CURRENT_EVENTS:
        elapsed = _event_elapsed(start, end)
        if elapsed is None:
            continue
        total, count = _TOTALS[prefix + label]
        _TOTALS[prefix + label] = [total + elapsed, count + 1.0]


def _publish_outer_round(*, prefix: str = "", context: dict[str, Any] | None = None) -> None:
    """Publish one closed outer round, classifying non-verification calls."""
    event_status = (
        _write_round_event_telemetry(context, prefix) if context is not None else None
    )
    global _PREVIOUS_ROUND_END, _PREVIOUS_ROUND_STREAM
    with _LOCK:
        # A synchronized diagnostic round has complete event values. Account it
        # now and recycle the fixed event slots so a long capture does not stop
        # after the first 64 rounds. Without the explicit sync, retain events
        # until the normal bounded process-exit flush instead of guessing.
        if _ROUND_EVENT_TELEMETRY_SYNC and event_status == "complete":
            _account_closed_round_events(prefix)
            global _EVENT_RING_CURSOR
            _EVENT_RING_CURSOR = 0
        else:
            _PENDING_EVENTS.extend(
                (prefix + label, start, end)
                for label, start, end in _OUTER_CURRENT_EVENTS
            )
        _OUTER_CURRENT_EVENTS.clear()
        _OUTER_EVENT_ORDER.clear()
        if context is not None:
            _PREVIOUS_ROUND_END = context.get("round_end_marker")
            _PREVIOUS_ROUND_STREAM = context.get("round_end_marker_stream")


def _shape_label(scheduler_output: Any) -> str:
    """Classify a worker call without assuming a particular vLLM dataclass."""
    value = _scheduled_tokens(scheduler_output)
    return f"M={value}" if value is not None else "M=unknown"


def _scheduled_tokens(scheduler_output: Any) -> int | None:
    """Return the flattened scheduled-row count when the ABI exposes it."""
    for name in ("total_num_scheduled_tokens", "num_scheduled_tokens"):
        value = getattr(scheduler_output, name, None)
        if value is not None:
            if hasattr(value, "values"):
                total = 0
                valid = False
                for item in value.values():
                    try:
                        item = int(item)
                    except (TypeError, ValueError):
                        continue
                    if item > 0:
                        total += item
                        valid = True
                if valid:
                    return total
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _wrap_method(owner: type, name: str, label: Callable[..., str]) -> bool:
    original = getattr(owner, name, None)
    if original is None:
        return False
    if getattr(original, "_qwen_stage_hook", False):
        return True

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        global _OUTER_REJECTION_SEEN
        try:
            stage = label(self, args, kwargs)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            stage = name
        if _OUTER_ONLY and not _OUTER_ROUND_ACTIVE:
            return original(self, *args, **kwargs)
        interval = _event_pair(stage if _ROUND_EVENT_TELEMETRY else None)
        try:
            return original(self, *args, **kwargs)
        finally:
            _finish(stage, interval)
            if _OUTER_ONLY and stage == "rejection.verify":
                _OUTER_REJECTION_SEEN = True

    wrapped._qwen_stage_hook = True  # type: ignore[attr-defined]
    setattr(owner, name, wrapped)
    return True


def _wrap_host_method(owner: type, name: str, label: str) -> bool:
    """Wrap a CPU scheduler boundary for commit/queue accounting.

    Scheduler state mutation is host work rather than a CUDA interval.  Keep it
    separate from ``_wrap_method`` so GPU-event totals are never mislabeled as a
    device measurement, while still reporting the commit cost in the same
    process-exit summary.
    """
    original = getattr(owner, name, None)
    if original is None:
        return False
    if getattr(original, "_qwen_stage_hook", False):
        return True

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        global _OUTER_COMMIT_LABEL, _OUTER_COMMIT_PENDING
        if _OUTER_ONLY and label == "scheduler.commit" and not _OUTER_COMMIT_PENDING:
            return original(self, *args, **kwargs)
        observed_label = _OUTER_COMMIT_LABEL if _OUTER_ONLY else label
        started = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - started) * 1_000.0
            total, count = _TOTALS[observed_label]
            _TOTALS[observed_label] = [total + elapsed, count + 1.0]
            if not _OUTER_ONLY:
                print(
                    f"[qwen-runtime] stage timing: {label} ms={elapsed:.3f}",
                    flush=True,
                )
            elif label == "scheduler.commit":
                _OUTER_COMMIT_PENDING = False

    wrapped._qwen_stage_hook = True  # type: ignore[attr-defined]
    setattr(owner, name, wrapped)
    return True


def _record_host_total(label: str, elapsed_ms: float) -> None:
    with _LOCK:
        total, count = _TOTALS[label]
        _TOTALS[label] = [total + elapsed_ms, count + 1.0]


def _wrap_outer_target_and_round(owner: type) -> bool:
    """Open one whole-round span and time the outer target forward."""

    original = getattr(owner, "execute_model", None)
    if original is None:
        return False
    if getattr(original, "_qwen_outer_stage_hook", False):
        return True

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        global _EVENT_RING_CAPPED_LOGGED, _OUTER_REJECTION_SEEN, _OUTER_ROUND_ACTIVE
        scheduler_output = args[0] if args else kwargs.get("scheduler_output")
        scheduled_tokens = _scheduled_tokens(scheduler_output)
        # vLLM startup and prompt-prefill forwards do not call sample_tokens.
        # Only a decode-sized target batch constitutes the round measured here.
        if scheduled_tokens is None or not 1 <= scheduled_tokens <= 16:
            return original(self, *args, **kwargs)
        # Keep observer state strictly bounded during long qualification
        # responses. Stop measuring only at a round boundary; serving itself
        # must continue after the fixed event pool has enough complete rounds.
        if _EVENT_RING_CURSOR + _EVENT_PAIRS_PER_ROUND_RESERVE > _EVENT_RING_SIZE:
            if not _EVENT_RING_CAPPED_LOGGED:
                print(
                    "[qwen-runtime] outer GPU-event measurement capped at complete rounds "
                    f"ring_pairs={_EVENT_RING_SIZE}",
                    flush=True,
                )
                _EVENT_RING_CAPPED_LOGGED = True
            return original(self, *args, **kwargs)
        previous = getattr(self, "_qwen_outer_round_interval", None)
        if previous is not None or _OUTER_ROUND_ACTIVE:
            raise RuntimeError("outer stage-timing round was not closed by sample_tokens")
        if _OUTER_CURRENT_EVENTS:
            raise RuntimeError("outer stage-timing found unpublished events before a new round")
        if _OUTER_EVENT_ORDER:
            raise RuntimeError("outer stage-timing found unpublished event markers before a new round")
        _OUTER_ROUND_ACTIVE = True
        round_label = "round.gpu"
        target_label = "target.forward " + _shape_label(scheduler_output)
        round_interval = _event_pair(round_label if _ROUND_EVENT_TELEMETRY else None)
        target_interval = _event_pair(target_label if _ROUND_EVENT_TELEMETRY else None)
        self._qwen_outer_round_interval = round_interval
        self._qwen_outer_round_started = time.perf_counter()
        fair = getattr(scheduler_output, "qwen_fair", None) or {}
        stream = None
        device = None
        try:
            import torch

            stream_object = torch.cuda.current_stream()
            stream = _stream_id(stream_object)
            device = int(torch.cuda.current_device())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            stream = None
            device = None
        self._qwen_outer_round_context = {
            "round": int(getattr(self, "_qwen_outer_round_count", 0)) + 1,
            "status_path": fair.get("status_path"),
            "scheduled_shape": _scheduled_shape(scheduler_output),
            "stream": stream,
            "device": device,
            "round_start_event": round_interval[0] if round_interval is not None else None,
            "previous_round_end_event": _PREVIOUS_ROUND_END,
            "previous_round_stream": _PREVIOUS_ROUND_STREAM,
            "runner_sync_count": getattr(self, "_qwen_decode_sync_count", None),
            "runner_sync_mode": getattr(self, "_qwen_decode_last_sync_mode", None),
            "runner_sync_reason": getattr(self, "_qwen_decode_last_sync_reason", None),
            "runner_sync_elapsed_ms": getattr(
                self, "_qwen_decode_last_sync_elapsed_ms", None
            ),
            "round_end_event": round_interval[1] if round_interval is not None else None,
        }
        self._qwen_outer_round_count = self._qwen_outer_round_context["round"]
        _OUTER_REJECTION_SEEN = False
        failed = True
        try:
            result = original(self, *args, **kwargs)
            failed = False
            return result
        finally:
            _finish(target_label, target_interval)
            if failed:
                _finish("round.gpu", round_interval)
                context = getattr(self, "_qwen_outer_round_context", None)
                if context is not None:
                    context["host_ms"] = (time.perf_counter() - self._qwen_outer_round_started) * 1000.0
                    marker, marker_stream = _record_inter_round_end()
                    context["round_end_marker"] = marker
                    context["round_end_marker_stream"] = marker_stream
                _publish_outer_round(prefix="failed.", context=context)
                self._qwen_outer_round_interval = None
                self._qwen_outer_round_started = None
                self._qwen_outer_round_context = None
                _OUTER_ROUND_ACTIVE = False

    wrapped._qwen_stage_hook = True  # type: ignore[attr-defined]
    wrapped._qwen_outer_stage_hook = True  # type: ignore[attr-defined]
    owner.execute_model = wrapped
    return True


def _wrap_outer_round_close(owner: type) -> bool:
    """Close the whole-round span after sample/commit/draft proposal."""

    original = getattr(owner, "sample_tokens", None)
    if original is None:
        return False
    if getattr(original, "_qwen_outer_stage_hook", False):
        return True

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        global _OUTER_COMMIT_LABEL, _OUTER_COMMIT_PENDING
        global _OUTER_REJECTION_SEEN, _OUTER_ROUND_ACTIVE
        if _OUTER_ONLY and _OUTER_ROUND_ACTIVE and _ROUND_EVENT_TELEMETRY:
            _record_event_marker("sample_tokens.start")
        try:
            return original(self, *args, **kwargs)
        finally:
            interval = getattr(self, "_qwen_outer_round_interval", None)
            started = getattr(self, "_qwen_outer_round_started", None)
            context = getattr(self, "_qwen_outer_round_context", None)
            self._qwen_outer_round_interval = None
            self._qwen_outer_round_started = None
            if interval is not None:
                _finish("round.gpu", interval)
                verified = _OUTER_REJECTION_SEEN
                prefix = "" if verified else "nonverification."
                if context is not None and started is not None:
                    context["host_ms"] = (time.perf_counter() - float(started)) * 1000.0
                    context["runner_sync_count_after"] = getattr(
                        self, "_qwen_decode_sync_count", None
                    )
                    marker, marker_stream = _record_inter_round_end()
                    context["round_end_marker"] = marker
                    context["round_end_marker_stream"] = marker_stream
                _publish_outer_round(prefix=prefix, context=context)
                _OUTER_COMMIT_PENDING = True
                _OUTER_COMMIT_LABEL = prefix + "scheduler.commit"
                wall_label = prefix + "round.dispatch-wall"
            else:
                wall_label = "nonverification.round.dispatch-wall"
            self._qwen_outer_round_context = None
            _OUTER_ROUND_ACTIVE = False
            _OUTER_REJECTION_SEEN = False
            if started is not None:
                _record_host_total(
                    wall_label,
                    (time.perf_counter() - float(started)) * 1_000.0,
                )

    wrapped._qwen_stage_hook = True  # type: ignore[attr-defined]
    wrapped._qwen_outer_stage_hook = True  # type: ignore[attr-defined]
    owner.sample_tokens = wrapped
    return True


def install_runtime_stage_hooks() -> bool:
    """Install worker-level event hooks when vLLM classes are available.

    vLLM imports the worker classes lazily and some DFlash startup paths load
    the Quest/GDN adapters before ``GPUModelRunner`` is importable.  A failed
    early attempt must therefore remain retryable; otherwise the process can
    run an entire qualification request with only the scheduler hook (or no
    outer GPU totals at all).
    """
    global _MODEL_HOOKS_INSTALLED, _COMPONENT_HOOKS_INSTALLED
    global _DRAFT_HOOK_INSTALLED, _SCHEDULER_HOOK_INSTALLED
    global _SAMPLE_CORE_HOOK_INSTALLED, _TARGET_LOGITS_HOOK_INSTALLED
    global _REJECTION_HOOK_INSTALLED, _ROUND_HOOK_INSTALLED
    global _STATE_COMMIT_HOOK_INSTALLED
    global _PENDING_LOGGED, _READY_LOGGED, _LAST_CRITICAL_MISSING
    if not _ENABLED:
        return False
    if _MODE_CONFLICT:
        raise RuntimeError("QWEN_STAGE_TIMING and QWEN_OUTER_STAGE_TIMING are mutually exclusive")
    with _LOCK:
        runner_classes: list[type] = []
        import_errors: list[str] = []
        for module_name, class_name in (
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
            # DFlash and the current gfx1201 qualification lane use V2.
            ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
        ):
            try:
                module = __import__(module_name, fromlist=[class_name])
                owner = getattr(module, class_name)
            except Exception as exc:  # pragma: no cover - runtime-specific
                import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            else:
                if owner not in runner_classes:
                    runner_classes.append(owner)

        target_ready = False
        sampler_ready = False
        sample_core_ready = False
        round_ready = False
        state_commit_ready = False
        draft_ready = False
        for owner in runner_classes:
            if _OUTER_ONLY:
                target_ready |= _wrap_outer_target_and_round(owner)
                round_ready |= _wrap_outer_round_close(owner)
                state_commit_ready |= _wrap_method(
                    owner, "postprocess_sampled", lambda *_: "state.commit"
                )
                sampler_ready = round_ready
                sample_core_ready = True
            else:
                target_ready |= _wrap_method(
                    owner,
                    "execute_model",
                    lambda self, args, kwargs: (
                        "target.execute_model "
                        + _shape_label(args[0] if args else kwargs.get("scheduler_output"))
                    ),
                )
                sampler_ready |= _wrap_method(owner, "sample_tokens", lambda *_: "sampler")
                # ``sample_tokens`` also performs post-processing and the next draft
                # proposal. Keep the core target-logits + verification interval
                # separate so the outer sampler total is not misattributed to the
                # rejection sampler.
                sample_core_ready |= _wrap_method(owner, "sample", lambda *_: "sample.core")
            draft_ready |= _wrap_method(
                owner, "propose_draft_token_ids", lambda *_: "draft.propose"
            )

        # Account for the model components which are not represented by the
        # kernel-adapter totals.  In particular, Qwen3-Next routes a number of
        # layers through MoE expert GEMMs; those calls are the most likely
        # source of the remaining M=16 target latency after Native-B has taken
        # every dense W4A16 projection.  These wrappers are diagnostic only and
        # are installed on the concrete module classes after lazy model import.
        component_ready = False
        target_logits_ready = False
        for mod_name in (
            "vllm.model_executor.models.qwen3_5",
            "vllm.model_executor.models.qwen3_next",
        ):
            try:
                module = __import__(
                    mod_name,
                    fromlist=[
                        "Qwen3NextSparseMoeBlock",
                        "Qwen3NextMLP",
                        "Qwen3NextAttention",
                        "Qwen3_5DecoderLayer",
                        "Qwen3_5Model",
                    ],
                )
                if not _OUTER_ONLY and hasattr(module, "Qwen3NextSparseMoeBlock"):
                    component_ready |= _wrap_method(
                        module.Qwen3NextSparseMoeBlock,
                        "forward",
                        lambda *_: "target.moe",
                    )
                if not _OUTER_ONLY and hasattr(module, "Qwen3NextMLP"):
                    component_ready |= _wrap_method(
                        module.Qwen3NextMLP,
                        "forward",
                        lambda *_: "target.dense-mlp",
                    )
                if not _OUTER_ONLY and hasattr(module, "Qwen3NextAttention"):
                    component_ready |= _wrap_method(
                        module.Qwen3NextAttention,
                        "forward",
                        lambda *_: "target.full-attention-layer",
                    )
                if not _OUTER_ONLY and hasattr(module, "Qwen3_5DecoderLayer"):
                    component_ready |= _wrap_method(
                        module.Qwen3_5DecoderLayer,
                        "forward",
                        lambda *_: "target.decoder-layer",
                    )
                for causal_lm_name in (
                    "Qwen3_5ForCausalLMBase",
                    "Qwen3NextForCausalLM",
                ):
                    if hasattr(module, causal_lm_name):
                        target_logits_ready |= _wrap_method(
                            getattr(module, causal_lm_name),
                            "compute_logits",
                            lambda *_: "target.logits",
                        )
            except Exception as exc:  # pragma: no cover - runtime-specific
                import_errors.append(f"{mod_name} components: {type(exc).__name__}: {exc}")

        # V2 has a separate rejection sampler from the legacy
        # ``vllm.v1.sample`` implementation. Time the class actually called by
        # ``vllm.v1.worker.gpu.model_runner`` so a legacy microbenchmark cannot
        # be mistaken for evidence about the live verifier.
        rejection_ready = False
        try:
            module = __import__(
                "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
                fromlist=["RejectionSampler"],
            )
            rejection_ready |= _wrap_method(
                module.RejectionSampler,
                "__call__",
                lambda *_: "rejection.verify",
            )
        except Exception as exc:  # pragma: no cover - runtime-specific
            import_errors.append(f"V2 rejection sampler: {type(exc).__name__}: {exc}")

        # The GDN class lives in the pinned overlay rather than in the model
        # module.  Its fused kernel reports its own narrower interval; this
        # outer interval captures projections, state updates, and epilogue too.
        if not _OUTER_ONLY:
            try:
                module = __import__(
                    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
                    fromlist=["QwenGatedDeltaNetAttention"],
                )
                component_ready |= _wrap_method(
                    module.QwenGatedDeltaNetAttention,
                    "forward",
                    lambda *_: "target.gdn-layer",
                )
            except Exception as exc:  # pragma: no cover - runtime-specific
                import_errors.append(f"qwen_gdn component: {type(exc).__name__}: {exc}")

        _COMPONENT_HOOKS_INSTALLED = component_ready
        _SAMPLE_CORE_HOOK_INSTALLED = sample_core_ready
        _TARGET_LOGITS_HOOK_INSTALLED = target_logits_ready
        _REJECTION_HOOK_INSTALLED = rejection_ready
        _ROUND_HOOK_INSTALLED = round_ready
        _STATE_COMMIT_HOOK_INSTALLED = state_commit_ready

        # V2 performs drafting inside a speculator rather than exposing the V1
        # propose_draft_token_ids method.  Wrap the concrete classes when they
        # are present; failed imports remain retryable during lazy startup.
        for module_name, class_name in (
            (
                "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
                "DFlashSpeculator",
            ),
            (
                "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator",
                "MultiModuleMTPSpeculator",
            ),
        ):
            try:
                module = __import__(module_name, fromlist=[class_name])
                owner = getattr(module, class_name)
            except Exception as exc:  # pragma: no cover - runtime-specific
                import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            else:
                draft_ready |= _wrap_method(owner, "propose", lambda *_: "draft.propose")

        _MODEL_HOOKS_INSTALLED = target_ready and sampler_ready
        _DRAFT_HOOK_INSTALLED = draft_ready
        if not runner_classes and not _PENDING_LOGGED:
            detail = "; ".join(import_errors[:2]) or "runner classes unavailable"
            print(f"[qwen-runtime] stage hooks pending: {detail}", flush=True)
            _PENDING_LOGGED = True

        try:
            from vllm.v1.core.sched.scheduler import Scheduler

            _SCHEDULER_HOOK_INSTALLED = _wrap_host_method(
                Scheduler, "update_from_output", "scheduler.commit"
            )
        except Exception as exc:  # pragma: no cover - runtime-specific
            if not _PENDING_LOGGED:
                print(
                    f"[qwen-runtime] scheduler hook pending: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                _PENDING_LOGGED = True

        if _OUTER_ONLY:
            critical = {
                "round.gpu": _ROUND_HOOK_INSTALLED,
                "draft.propose": _DRAFT_HOOK_INSTALLED,
                "target.logits": _TARGET_LOGITS_HOOK_INSTALLED,
                "rejection.verify": _REJECTION_HOOK_INSTALLED,
                "state.commit": _STATE_COMMIT_HOOK_INSTALLED,
            }
        else:
            critical = {
                "sample.core": _SAMPLE_CORE_HOOK_INSTALLED,
                "target.logits": _TARGET_LOGITS_HOOK_INSTALLED,
                "rejection.verify": _REJECTION_HOOK_INSTALLED,
            }
        critical_missing = tuple(label for label, installed in critical.items() if not installed)
        if critical_missing != _LAST_CRITICAL_MISSING:
            if critical_missing:
                print(
                    "[qwen-runtime] critical stage hooks pending: " + ", ".join(critical_missing),
                    flush=True,
                )
            _LAST_CRITICAL_MISSING = critical_missing
        ready = _MODEL_HOOKS_INSTALLED and _SCHEDULER_HOOK_INSTALLED and not critical_missing
        if ready and not _READY_LOGGED:
            if _OUTER_ONLY:
                _prepare_event_ring()
                print(
                    "[qwen-runtime] outer GPU-event stage hooks installed "
                    f"ring_pairs={_EVENT_RING_SIZE}",
                    flush=True,
                )
            else:
                print("[qwen-runtime] GPU-event stage hooks installed", flush=True)
            _READY_LOGGED = True
        elif _MODEL_HOOKS_INSTALLED and not _SCHEDULER_HOOK_INSTALLED and not _PENDING_LOGGED:
            print(
                "[qwen-runtime] worker GPU-event hooks installed; scheduler hook pending",
                flush=True,
            )
            _PENDING_LOGGED = True
        return ready


def _flush_events() -> None:
    global _EVENT_RING_CURSOR
    with _LOCK:
        if not _PENDING_EVENTS:
            return
        pending = list(_PENDING_EVENTS)
        _PENDING_EVENTS.clear()

    import torch

    torch.cuda.synchronize()
    for label, start, end in pending:
        try:
            elapsed = float(start.elapsed_time(end))
            total, count = _TOTALS[label]
            _TOTALS[label] = [total + elapsed, count + 1.0]
        except Exception:
            pass
    if _OUTER_ONLY:
        _EVENT_RING_CURSOR = 0


def _dump_totals() -> None:
    _flush_events()
    if not _TOTALS:
        return
    fields = []
    for label, (total, count) in sorted(_TOTALS.items()):
        fields.append(f"{label}:count={int(count)} total_ms={total:.3f}")
    print("[qwen-runtime] stage totals: " + " | ".join(fields), flush=True)


atexit.register(_dump_totals)


def _signal_handler(signum, frame):
    _dump_totals()
    try:
        from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
            _dump_stage_totals as d1,
        )

        d1()
    except Exception:
        pass
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _dump_gdn_stage_totals as d2,
        )

        d2()
    except Exception:
        pass
    try:
        from vllm.v1.attention.backends.quest_vllm_attention import _dump_quest_stage_totals as d3

        d3()
    except Exception:
        pass
    sys.exit(0)


try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except Exception:
    pass


__all__ = ["install_runtime_stage_hooks"]
# QWEN_ASSURANCE_ONLY_END: runtime-stage-hooks
