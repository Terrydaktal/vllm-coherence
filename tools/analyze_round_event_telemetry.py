#!/usr/bin/env python3
"""Summarize content-free ROCm/HIP per-round telemetry.

The input is the JSONL feed written by ``runtime_stage_hooks``.  This tool is
deliberately strict: a missing round span, a dropped row, or an incomplete
event record is reported as an invalid capture instead of being silently
treated as zero.  It emits timings, shapes and hashes only; it never reads
prompt text, token IDs, tool arguments or transcript files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "urn:qwen-r9700:decode-round-gpu:v2"


def _number(value: Any, *, name: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} is invalid")
    return result


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _stats(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": _quantile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _histogram(values: list[float], width: float) -> list[dict[str, float | int]]:
    if not values:
        return []
    if width <= 0 or not math.isfinite(width):
        raise ValueError("histogram width must be positive")
    first = math.floor(min(values) / width) * width
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        counts[int(math.floor((value - first) / width))] += 1
    return [
        {
            "from_ms": first + index * width,
            "to_ms": first + (index + 1) * width,
            "count": counts[index],
        }
        for index in range(max(counts) + 1)
    ]


def _shape_key(row: dict[str, Any]) -> str:
    return json.dumps(row.get("scheduled_shape", {}), sort_keys=True, separators=(",", ":"))


def _runtime_shape_key(row: dict[str, Any]) -> str:
    """Return the content-free graph/padding shape, if this runtime recorded it."""
    return json.dumps(row.get("runtime_shape") or {}, sort_keys=True, separators=(",", ":"))


def _derived_round_span(row: dict[str, Any]) -> float | None:
    """Recover the span from the named outer event in pre-v2 captures."""
    for item in row.get("gpu_events", {}).get("durations_ms", []):
        if item.get("label") == "round.gpu":
            value = item.get("ms")
            if value is not None:
                return _number(value, name="derived round span")
    return None


def _slow_episodes(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "baseline_ms": None,
            "threshold_ms": None,
            "slow_rounds": 0,
            "episodes": [],
        }
    baseline = statistics.median(values)
    threshold = max(4.0, baseline * 0.08)
    episodes: list[dict[str, Any]] = []
    start = None
    for index, value in enumerate(values):
        slow = value > baseline + threshold
        if slow and start is None:
            start = index
        if not slow and start is not None:
            episode_values = values[start:index]
            episodes.append(
                {
                    "first_round_index": start,
                    "last_round_index": index - 1,
                    "rounds": len(episode_values),
                    "mean_ms": statistics.fmean(episode_values),
                    "max_ms": max(episode_values),
                }
            )
            start = None
    if start is not None:
        episode_values = values[start:]
        episodes.append(
            {
                "first_round_index": start,
                "last_round_index": len(values) - 1,
                "rounds": len(episode_values),
                "mean_ms": statistics.fmean(episode_values),
                "max_ms": max(episode_values),
            }
        )
    return {
        "baseline_ms": baseline,
        "threshold_ms": threshold,
        "slow_rounds": sum(item["rounds"] for item in episodes),
        "episodes": episodes,
    }


def _read_records(
    paths: Iterable[Path],
    *,
    require_complete: bool,
    derive_round_span: bool = False,
) -> tuple[list[dict[str, Any]], list[str], str]:
    records: list[dict[str, Any]] = []
    source_hashes: list[str] = []
    span_source = "round_start_end_events"
    for path in paths:
        raw = path.read_bytes()
        source_hashes.append(hashlib.sha256(raw).hexdigest())
        for line_number, line in enumerate(raw.splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if row.get("schema") != SCHEMA:
                raise ValueError(f"{path}:{line_number}: unexpected telemetry schema")
            if row.get("dropped_records_before", 0):
                raise ValueError(f"{path}:{line_number}: telemetry rows were dropped")
            status = row.get("gpu_events", {}).get("event_status")
            if require_complete and status != "complete":
                raise ValueError(f"{path}:{line_number}: event record is {status!r}")
            span = row.get("gpu_events", {}).get("round_span_ms")
            if span is None:
                if not derive_round_span:
                    raise ValueError(f"{path}:{line_number}: round span is missing")
                span = _derived_round_span(row)
                if span is None:
                    raise ValueError(
                        f"{path}:{line_number}: round span is missing and cannot be derived"
                    )
                span_source = "round.gpu_duration_legacy"
                row.setdefault("gpu_events", {})["round_span_ms"] = span
            _number(span, name="round span")
            records.append(row)
    records.sort(key=lambda row: (int(row.get("round", 0)), int(row.get("observed_at_ms", 0))))
    if not records:
        raise ValueError("telemetry input contains no records")
    return records, source_hashes, span_source


def analyze(
    paths: Iterable[Path],
    *,
    histogram_width_ms: float = 0.5,
    require_complete: bool = True,
    derive_round_span: bool = False,
) -> dict[str, Any]:
    records, source_hashes, span_source = _read_records(
        paths,
        require_complete=require_complete,
        derive_round_span=derive_round_span,
    )
    round_ms: list[float] = []
    gpu_ms: list[float] = []
    host_wait_ms: list[float] = []
    gap_ms: list[float] = []
    telemetry_wait_ms: list[float] = []
    boundary_gap_ms: dict[str, float] = defaultdict(float)
    boundary_observations: dict[str, int] = defaultdict(int)
    shapes: dict[str, list[float]] = defaultdict(list)
    runtime_shapes: dict[str, list[float]] = defaultdict(list)
    sync_reasons: dict[str, int] = defaultdict(int)
    unavailable_cross_stream_gaps = 0
    for row in records:
        gpu = row["gpu_events"]
        dispatch = row.get("dispatch", {})
        value = _number(gpu["round_span_ms"], name="round span")
        assert value is not None
        round_ms.append(value)
        gpu_ms.append(value)
        host = _number(dispatch.get("host_ms"), name="host round", allow_none=True)
        if host is not None:
            host_wait_ms.append(max(0.0, host - value))
        telemetry = _number(dispatch.get("telemetry_wait_ms"), name="telemetry wait", allow_none=True)
        if telemetry is not None:
            telemetry_wait_ms.append(telemetry)
        for gap in gpu.get("gaps_ms", []):
            amount = _number(gap.get("ms"), name="event gap", allow_none=True)
            if amount is None:
                if not gap.get("same_stream", False):
                    unavailable_cross_stream_gaps += 1
                continue
            gap_ms.append(amount)
            key = f"{gap.get('from', '?')} -> {gap.get('to', '?')}"
            boundary_gap_ms[key] += amount
            boundary_observations[key] += 1
        reason = row.get("queue_sync", {}).get("reason")
        if reason:
            sync_reasons[str(reason)] += 1
        shapes[_shape_key(row)].append(value)
        runtime_shapes[_runtime_shape_key(row)].append(value)
    shape_summary = {
        key: {"rounds": len(values), "round_ms": _stats(values)}
        for key, values in sorted(shapes.items())
    }
    runtime_shape_summary = {
        key: {"rounds": len(values), "round_ms": _stats(values)}
        for key, values in sorted(runtime_shapes.items())
    }
    return {
        "schema": "urn:qwen-r9700:decode-round-analysis:v1",
        "status": "COMPLETE",
        "private_chat_text_read": False,
        "source_sha256": source_hashes,
        "round_span_source": span_source,
        "records": len(records),
        "round_ms": _stats(round_ms),
        "gpu_round_ms": _stats(gpu_ms),
        "host_wait_ms": _stats(host_wait_ms),
        "telemetry_wait_ms": _stats(telemetry_wait_ms),
        "gpu_gap_ms": _stats(gap_ms),
        "gpu_gap_total_ms": sum(gap_ms),
        "gpu_gap_by_boundary_ms": dict(sorted(boundary_gap_ms.items())),
        "gpu_gap_boundary_observations": dict(sorted(boundary_observations.items())),
        "unavailable_cross_stream_gaps": unavailable_cross_stream_gaps,
        "queue_sync_reasons": dict(sorted(sync_reasons.items())),
        "histogram_ms": _histogram(round_ms, histogram_width_ms),
        "slow_episode_diagnostics": _slow_episodes(round_ms),
        "by_shape": shape_summary,
        "by_runtime_shape": runtime_shape_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bin-ms", type=float, default=0.5)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--derive-round-span",
        action="store_true",
        help="derive round span from the legacy round.gpu duration when the v2 field is absent",
    )
    args = parser.parse_args()
    result = analyze(
        args.paths,
        histogram_width_ms=args.bin_ms,
        require_complete=not args.allow_incomplete,
        derive_round_span=args.derive_round_span,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
