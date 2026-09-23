"""Low-observer-overhead GPU stage timing for the compiled profile.

The old stage profile used ``torch.profiler`` with CPU and GPU activities.  It
is useful for attribution, but it changes host dispatch and queue behaviour.
This module records only preallocated device events around the stage scopes and
resolves them after the round has been queued.  It never synchronizes during a
measured round.  A synchronization is used only at the end of the profile
window to drain the evidence, and that wait is reported separately.

The result is intentionally an interval profile, not a claim that every GPU
kernel was independently observed.  Container scopes are ignored so nested
module wrappers cannot double count their children; any work outside the
recorded leaf scopes remains an explicit un-attributed interval.  On ROCm,
PyTorch exposes HIP events through its compatibility ``torch.cuda.Event``
namespace; no CUDA runtime is required.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

LOW_OVERHEAD_PROFILE_SCHEMA = "urn:coherence:low-overhead-stage-events-v1"
DEFAULT_EVENT_PAIRS = 8_192

# These scopes contain other semantic scopes.  Timing both the container and
# its children would double count GPU work.  Their uncovered work is reported
# separately rather than silently assigned to a child stage.
DEFAULT_IGNORED_SCOPES = frozenset(
    {
        "target_other",
        "layer_other",
        "gdn_other",
        "attention_gating_and_other",
        "input_preparation",
        "forced_replay_control",
        "target_sampling",
    }
)


@dataclass
class _EventPair:
    start: Any
    end: Any
    slot: int


@dataclass
class _ScopeRecord:
    name: str
    pair: _EventPair
    stream: str
    host_start: float
    host_end: float | None = None


@dataclass
class _RoundRecord:
    scopes: list[_ScopeRecord]
    markers: list[_EventPair]
    span: _EventPair
    span_stream: str
    round_number: int
    host_start: float
    host_end: float
    host_span_ms: float
    metadata: dict[str, Any]


class LowOverheadStageTimer:
    """Record stage intervals with a bounded reusable HIP event pool.

    ``torch`` is imported only when the timer is started so this module can be
    tested on a CPU-only host.  The timer is deliberately fail-closed: pool
    exhaustion or an unresolved event is an error, not a partial timing result.
    The default pool is large enough for the mapped Qwen stage scopes while
    remaining bounded; ``QWEN_LOW_OVERHEAD_EVENT_PAIRS`` can raise it for a
    larger model or a denser wrapper set.
    """

    def __init__(
        self,
        *,
        max_event_pairs: int = DEFAULT_EVENT_PAIRS,
        ignored_scopes: frozenset[str] = DEFAULT_IGNORED_SCOPES,
    ) -> None:
        if max_event_pairs < 2:
            raise ValueError("at least two event pairs are required")
        self.max_event_pairs = int(max_event_pairs)
        self.ignored_scopes = frozenset(ignored_scopes)
        self._torch: Any | None = None
        self._pairs: list[_EventPair] = []
        self._free_slots: list[int] = []
        self._pending: deque[_RoundRecord] = deque()
        self._current_scopes: list[_ScopeRecord] = []
        self._current_streams: dict[str, Any] = {}
        self._current_span: _EventPair | None = None
        self._current_span_stream: str | None = None
        self._current_span_stream_object: Any | None = None
        self._current_host_start = 0.0
        self._current_metadata: dict[str, Any] = {}
        self._round_active = False
        self._active = False
        self._started = False
        self._round_number = 0
        self._completed_rounds = 0
        self._stage_totals_us: dict[str, float] = defaultdict(float)
        self._stage_intervals: dict[str, int] = defaultdict(int)
        self._stage_scope_calls: dict[str, int] = defaultdict(int)
        self._rounds: list[dict[str, Any]] = []
        self._round_span_total_us = 0.0
        self._unattributed_gap_total_us = 0.0
        self._overlap_total_us = 0.0
        self._host_round_total_us = 0.0
        self._host_observed_wait_total_us = 0.0
        self._event_pairs_recorded = 0
        self._marker_pairs_recorded = 0
        self._max_pending_rounds = 0
        self._pool_exhaustions = 0
        self._start_sync_ms = 0.0
        self._finish_sync_ms = 0.0
        self._start_time = 0.0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def round_active(self) -> bool:
        return self._round_active

    def _new_event_pair(self, slot: int) -> _EventPair:
        assert self._torch is not None
        event_type = self._torch.cuda.Event
        return _EventPair(
            start=event_type(enable_timing=True),
            end=event_type(enable_timing=True),
            slot=slot,
        )

    def start(self) -> None:
        if self._started:
            raise RuntimeError("low-overhead stage timer already started")
        import torch

        self._torch = torch
        sync_started = time.perf_counter()
        torch.cuda.synchronize()
        self._start_sync_ms = (time.perf_counter() - sync_started) * 1_000.0
        self._pairs = [
            self._new_event_pair(index) for index in range(self.max_event_pairs)
        ]
        self._free_slots = list(range(self.max_event_pairs - 1, -1, -1))
        self._pending.clear()
        self._active = True
        self._started = True
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        if not self._started:
            return
        self._active = False
        if self._round_active:
            raise RuntimeError("low-overhead stage timer stopped with an open round")

    def _acquire(self) -> _EventPair:
        if not self._free_slots:
            # Normally the previous round was drained at begin/finish.  Only
            # poll here when the bounded pool is genuinely full; querying on
            # every stage boundary would add avoidable host overhead.
            self._drain_completed()
        if not self._free_slots:
            self._pool_exhaustions += 1
            raise RuntimeError(
                "low-overhead stage event pool exhausted; increase "
                f"max_event_pairs above {self.max_event_pairs}"
            )
        return self._pairs[self._free_slots.pop()]

    @staticmethod
    def _stream_label(stream: Any) -> str:
        value = getattr(stream, "cuda_stream", stream)
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)

    def begin_round(self) -> None:
        if not self._active:
            raise RuntimeError("cannot begin a round before starting the timer")
        if self._round_active or self._current_scopes:
            raise RuntimeError("previous low-overhead stage round is still open")
        self._drain_completed()
        self._current_scopes = []
        self._current_streams = {}
        stream = self._torch.cuda.current_stream()
        self._current_span = self._acquire()
        self._current_span.start.record(stream)
        self._current_span_stream = self._stream_label(stream)
        self._current_span_stream_object = stream
        self._current_host_start = time.perf_counter()
        self._current_metadata = {}
        self._round_active = True
        self._round_number += 1

    @contextlib.contextmanager
    def scope(self, name: str):
        if not self._active or not self._round_active or name in self.ignored_scopes:
            yield
            return
        assert self._torch is not None
        pair = self._acquire()
        stream = self._torch.cuda.current_stream()
        stream_label = self._stream_label(stream)
        host_start = time.perf_counter()
        pair.start.record(stream)
        record = _ScopeRecord(
            name=name,
            pair=pair,
            stream=stream_label,
            host_start=host_start,
        )
        self._current_scopes.append(record)
        self._current_streams[stream_label] = stream
        try:
            yield
        finally:
            pair.end.record(stream)
            record.host_end = time.perf_counter()

    def annotate_round(self, **metadata: Any) -> None:
        """Attach discrete scheduler/shape facts to the open round.

        This records no tensor data and performs no device synchronization. It
        exists so a long/short GPU interval can be correlated with accepted
        speculative width and batch shape instead of being guessed from wall
        time alone.
        """
        if not self._round_active:
            raise RuntimeError("cannot annotate a stage round that is not open")
        self._current_metadata.update(metadata)

    def finish_round(self) -> None:
        if not self._round_active:
            raise RuntimeError("cannot finish a stage round that was not started")
        if self._current_span is None or self._current_span_stream is None:
            raise RuntimeError("low-overhead stage round has no GPU span")
        span = self._current_span
        span_stream = self._current_span_stream
        assert self._current_span_stream_object is not None
        span.end.record(self._current_span_stream_object)
        markers: list[_EventPair] = []
        for stream_label, stream in self._current_streams.items():
            if stream_label == span_stream:
                # The round-span end marker is also the completion marker for
                # the primary stream, so do not allocate a duplicate event.
                continue
            marker = self._acquire()
            marker.end.record(stream)
            markers.append(marker)
            self._marker_pairs_recorded += 1
        host_end = time.perf_counter()
        record = _RoundRecord(
            scopes=self._current_scopes,
            markers=markers,
            span=span,
            span_stream=span_stream,
            round_number=self._round_number,
            host_start=self._current_host_start,
            host_end=host_end,
            host_span_ms=(host_end - self._current_host_start) * 1_000.0,
            metadata=dict(self._current_metadata),
        )
        self._pending.append(record)
        self._event_pairs_recorded += len(record.scopes)
        self._max_pending_rounds = max(self._max_pending_rounds, len(self._pending))
        self._current_scopes = []
        self._current_streams = {}
        self._current_span = None
        self._current_span_stream = None
        self._current_span_stream_object = None
        self._round_active = False
        self._completed_rounds += 1
        self._drain_completed()

    @staticmethod
    def _query(event: Any) -> bool:
        query = getattr(event, "query", None)
        if query is None:
            return False
        try:
            return bool(query())
        except Exception as exc:  # pragma: no cover - runtime-specific
            raise RuntimeError("HIP event query failed") from exc

    def _release(self, pair: _EventPair) -> None:
        self._free_slots.append(pair.slot)

    def _resolve_round(self, record: _RoundRecord) -> None:
        try:
            span_ms = float(record.span.start.elapsed_time(record.span.end))
        except Exception as exc:  # pragma: no cover - runtime-specific
            raise RuntimeError("HIP event round-span query failed") from exc
        if not math.isfinite(span_ms) or span_ms < 0:
            raise RuntimeError("HIP event returned an invalid round span")
        by_stage: dict[str, float] = defaultdict(float)
        by_stage_count: dict[str, int] = defaultdict(int)
        intervals: list[dict[str, Any]] = []
        for scope in record.scopes:
            try:
                elapsed_ms = float(scope.pair.start.elapsed_time(scope.pair.end))
                start_ms = float(record.span.start.elapsed_time(scope.pair.start))
                end_ms = float(record.span.start.elapsed_time(scope.pair.end))
            except Exception as exc:  # pragma: no cover - runtime-specific
                raise RuntimeError("HIP event elapsed-time query failed") from exc
            if (
                not math.isfinite(elapsed_ms)
                or elapsed_ms < 0
                or not math.isfinite(start_ms)
                or not math.isfinite(end_ms)
                or start_ms < 0
                or end_ms < start_ms
                or end_ms > span_ms
            ):
                raise RuntimeError("HIP event returned an invalid stage duration")
            by_stage[scope.name] += elapsed_ms
            by_stage_count[scope.name] += 1
            intervals.append(
                {
                    "stage": scope.name,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": elapsed_ms,
                    "stream": scope.stream,
                    "host_start_ms": (scope.host_start - record.host_start) * 1_000.0,
                    "host_end_ms": (
                        None
                        if scope.host_end is None
                        else (scope.host_end - record.host_start) * 1_000.0
                    ),
                }
            )
            self._release(scope.pair)

        for marker in record.markers:
            self._release(marker)
        self._release(record.span)
        stage_sum_ms = sum(by_stage.values())
        # Stage scopes can overlap (for example, work launched on a secondary
        # stream while the primary stream advances).  Adding durations would
        # count that work twice.  Merge event intervals first, then account for
        # only the portions of the round not covered by any named scope.
        ordered = sorted(intervals, key=lambda item: (item["start_ms"], item["end_ms"]))
        union: list[dict[str, Any]] = []
        for item in ordered:
            start_ms = float(item["start_ms"])
            end_ms = float(item["end_ms"])
            if not union or start_ms > union[-1]["end_ms"]:
                union.append(
                    {
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "stages": [item["stage"]],
                    }
                )
            else:
                union[-1]["end_ms"] = max(union[-1]["end_ms"], end_ms)
                if item["stage"] not in union[-1]["stages"]:
                    union[-1]["stages"].append(item["stage"])
        union_ms = sum(item["end_ms"] - item["start_ms"] for item in union)
        gap_segments: list[dict[str, Any]] = []
        cursor_ms = 0.0
        previous_stages: list[str] = ["round_start"]
        for index, item in enumerate(union):
            start_ms = float(item["start_ms"])
            end_ms = float(item["end_ms"])
            if start_ms > cursor_ms:
                gap_segments.append(
                    {
                        "kind": "unattributed",
                        "from_ms": cursor_ms,
                        "to_ms": start_ms,
                        "gap_ms": start_ms - cursor_ms,
                        "before_stages": previous_stages,
                        "after_stages": item["stages"],
                    }
                )
            cursor_ms = max(cursor_ms, end_ms)
            previous_stages = item["stages"]
        if cursor_ms < span_ms:
            gap_segments.append(
                {
                    "kind": "unattributed",
                    "from_ms": cursor_ms,
                    "to_ms": span_ms,
                    "gap_ms": span_ms - cursor_ms,
                    "before_stages": previous_stages,
                    "after_stages": ["round_end"],
                }
            )
        gap_ms = sum(segment["gap_ms"] for segment in gap_segments)
        overlap_ms = max(0.0, stage_sum_ms - union_ms)

        # Preserve the recorded scope order as a second, causal view.  This is
        # useful for identifying a launch/queue hole between two semantic
        # stages, but it is deliberately not used for accounting when scopes
        # overlap or are nested.
        boundary_gaps: list[dict[str, Any]] = []
        previous_name = "round_start"
        previous_end_ms = 0.0
        for item in intervals:
            delta_ms = float(item["start_ms"]) - previous_end_ms
            boundary_gaps.append(
                {
                    "from": previous_name,
                    "to": item["stage"],
                    "delta_ms": delta_ms,
                    "gap_ms": max(0.0, delta_ms),
                    "overlap_ms": max(0.0, -delta_ms),
                }
            )
            previous_name = item["stage"]
            previous_end_ms = max(previous_end_ms, float(item["end_ms"]))
        final_delta_ms = span_ms - previous_end_ms
        boundary_gaps.append(
            {
                "from": previous_name,
                "to": "round_end",
                "delta_ms": final_delta_ms,
                "gap_ms": max(0.0, final_delta_ms),
                "overlap_ms": max(0.0, -final_delta_ms),
            }
        )
        host_boundary_gaps: list[dict[str, Any]] = []
        previous_host_end_ms = 0.0
        previous_host_name = "round_start"
        for item in intervals:
            host_start_ms = float(item["host_start_ms"])
            host_end_ms = item["host_end_ms"]
            if host_end_ms is None:
                continue
            host_delta_ms = host_start_ms - previous_host_end_ms
            host_boundary_gaps.append(
                {
                    "from": previous_host_name,
                    "to": item["stage"],
                    "gap_ms": max(0.0, host_delta_ms),
                    "overlap_ms": max(0.0, -host_delta_ms),
                }
            )
            previous_host_name = item["stage"]
            previous_host_end_ms = max(previous_host_end_ms, float(host_end_ms))
        host_boundary_gaps.append(
            {
                "from": previous_host_name,
                "to": "round_end",
                "gap_ms": max(0.0, record.host_span_ms - previous_host_end_ms),
                "overlap_ms": max(0.0, previous_host_end_ms - record.host_span_ms),
            }
        )
        host_wait_ms = max(0.0, record.host_span_ms - span_ms)
        self._round_span_total_us += span_ms * 1_000.0
        self._unattributed_gap_total_us += gap_ms * 1_000.0
        self._overlap_total_us += overlap_ms * 1_000.0
        self._host_round_total_us += record.host_span_ms * 1_000.0
        self._host_observed_wait_total_us += host_wait_ms * 1_000.0
        for stage, elapsed_ms in by_stage.items():
            self._stage_totals_us[stage] += elapsed_ms * 1_000.0
            self._stage_intervals[stage] += by_stage_count[stage]
            self._stage_scope_calls[stage] += by_stage_count[stage]
        self._rounds.append(
            {
                "round": record.round_number,
                "stages_ms": {stage: value for stage, value in by_stage.items()},
                "round_span_ms": span_ms,
                "host_round_ms": record.host_span_ms,
                "host_observed_wait_ms": host_wait_ms,
                "metadata": record.metadata,
                "named_stage_sum_ms": stage_sum_ms,
                "named_stage_union_ms": union_ms,
                "unattributed_gap_ms": gap_ms,
                "overlap_ms": overlap_ms,
                "gap_segments": gap_segments,
                "boundary_gaps": boundary_gaps,
                "host_boundary_gaps": host_boundary_gaps,
                "intervals_detail": intervals,
                "intervals": sum(by_stage_count.values()),
                "streams": sorted({scope.stream for scope in record.scopes}),
            }
        )

    def _drain_completed(self, *, force: bool = False) -> None:
        while self._pending:
            record = self._pending[0]
            completion_events = [record.span.end] + [
                marker.end for marker in record.markers
            ]
            if not force and not all(self._query(event) for event in completion_events):
                break
            self._pending.popleft()
            self._resolve_round(record)

    def finish(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("low-overhead stage timer was never started")
        self.stop()
        assert self._torch is not None
        sync_started = time.perf_counter()
        self._torch.cuda.synchronize()
        self._finish_sync_ms = (time.perf_counter() - sync_started) * 1_000.0
        self._drain_completed(force=True)
        if self._pending:
            raise RuntimeError("low-overhead stage events remained unresolved")
        if self._round_active:
            raise RuntimeError("low-overhead stage timer finished with an open round")
        steps = self._completed_rounds
        if steps <= 0:
            raise RuntimeError("low-overhead stage timer captured no rounds")
        stages = {}
        for stage in sorted(self._stage_totals_us):
            stages[stage] = {
                "gpu_us": self._stage_totals_us[stage],
                "intervals": self._stage_intervals[stage],
                "scope_calls": self._stage_scope_calls[stage],
                "ms_per_step": self._stage_totals_us[stage] / (1_000.0 * steps),
            }
        return {
            "schema": LOW_OVERHEAD_PROFILE_SCHEMA,
            "observer": "preallocated HIP events; asynchronous resolution",
            "production_timing_eligible": False,
            "timing_contract": {
                "stage_metric": "hip_scope_interval",
                "per_stage_event_probes": True,
                "added_synchronization": False,
                "zero_observer_effect_proven": False,
                "limitation": (
                    "Scope intervals can include host dispatch gaps and event-recording effects. "
                    "They are diagnostic intervals, not kernel-only durations or profiler-free gaps."
                ),
            },
            "steps": steps,
            "stages": stages,
            "rounds": self._rounds,
            "round_span_us": self._round_span_total_us,
            "unattributed_gap_us": self._unattributed_gap_total_us,
            "overlap_us": self._overlap_total_us,
            "gap_accounting": "union-of-named-GPU-intervals-v1",
            "host_round_us": self._host_round_total_us,
            "host_observed_wait_us": self._host_observed_wait_total_us,
            "ignored_scopes": sorted(self.ignored_scopes),
            "event_pool_pairs": self.max_event_pairs,
            "event_pairs_recorded": self._event_pairs_recorded,
            "marker_pairs_recorded": self._marker_pairs_recorded,
            "max_pending_rounds": self._max_pending_rounds,
            "pool_exhaustions": self._pool_exhaustions,
            "setup_sync_ms": self._start_sync_ms,
            "finish_sync_ms": self._finish_sync_ms,
            # This is diagnostic wall time for the profile window, not a
            # claimed observer cost.  The only waits introduced by this
            # timer are setup_sync_ms and finish_sync_ms above.
            "capture_elapsed_wall_ms": (time.perf_counter() - self._start_time)
            * 1_000.0,
            "scope_accounting": (
                "Direct leaf-scope GPU event durations. Container scopes are ignored. "
                "Named intervals are merged before accounting so overlapping streams "
                "are counted once; the remaining span is emitted as explicit gap "
                "segments and recorded again as ordered boundary gaps. The host round "
                "and host-observed wait are diagnostic comparisons against the GPU span; "
                "they include dispatch and stream handoff effects and are not added to "
                "GPU stage time."
            ),
        }


__all__ = [
    "DEFAULT_EVENT_PAIRS",
    "LOW_OVERHEAD_PROFILE_SCHEMA",
    "LowOverheadStageTimer",
]
