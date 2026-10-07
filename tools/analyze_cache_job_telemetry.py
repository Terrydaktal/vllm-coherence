#!/usr/bin/env python3
"""Correlate content-free round or first-output timing; overlap is not causation."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SCHEMA = "urn:coherence:cache-job-timings:v1"
ROUND_SCHEMA = "urn:qwen-r9700:decode-rounds:v1"
IDENTITIES = ("request_id", "external_request_id", "http_request_id")
HEX_ID = re.compile(r"[0-9a-f]{64}\Z")

# Envelopes, observer work and point markers do not explain what happens
# inside a request. Preserve them in the timeline without claiming coverage.
REQUEST_ENVELOPES = {"http_request", "http_end", "recorder_batch"}


def read_records(paths, schema):
    records, invalid = [], 0
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                if isinstance(row, dict) and row.get("schema") == schema:
                    records.append(row)
                else:
                    invalid += 1
    return records, invalid


def union_ms(spans):
    total, right = 0, None
    for start, end in sorted(spans):
        if end <= start:
            continue
        total += max(0, end - max(start, right if right is not None else start))
        right = max(end, right if right is not None else end)
    return total / 1e6


def _identities(event):
    return {
        (name, event[name])
        for name in IDENTITIES
        if isinstance(event.get(name), str) and HEX_ID.fullmatch(event[name])
    }


def _valid_event(event):
    return (
        isinstance(event, dict)
        and type(event.get("pid")) is int
        and isinstance(event.get("trace_id"), str)
        and bool(event["trace_id"])
        and (event.get("clock_id") is None or isinstance(event["clock_id"], str))
        and type(event.get("sequence")) is int
        and event["sequence"] >= 0
        and isinstance(event.get("stage"), str)
        and type(event.get("start_ns")) is int
        and type(event.get("end_ns")) is int
        and 0 <= event["start_ns"] <= event["end_ns"]
    )


def analyze_request(events, request_id, health=None, *, invalid_records=0):
    """Join explicit hashed-ID bridges and account for a same-host latency window.

    Numeric monotonic timestamps from processes on the same boot can be compared.
    A request identity must travel through an explicit bridge to a new identity;
    proximity, active chat identity or matching PID is never a substitute.
    """
    if not isinstance(request_id, str) or not HEX_ID.fullmatch(request_id):
        raise ValueError("request ID must be a lowercase SHA-256 digest")
    valid = [event for event in events if _valid_event(event)]
    invalid_records += len(events) - len(valid)
    linked = {(name, request_id) for name in IDENTITIES}
    # Only bridges can connect namespace-specific identifiers. A background
    # record's active-chat snapshot must never bring another request into scope.
    while True:
        expanded = set(linked)
        for event in valid:
            ids = _identities(event)
            if event["stage"] == "internal_id_bridge" and linked & ids:
                expanded.update(ids)
        if expanded == linked:
            break
        linked = expanded
    matches = [event for event in valid if _identities(event) & linked]
    # Duplicate inputs (e.g. the same file twice) do not create extra spans.
    unique = {}
    conflicts = 0
    for event in matches:
        key = (event["pid"], event["trace_id"], event["sequence"])
        if key in unique and unique[key] != event:
            conflicts += 1
        else:
            unique[key] = event
    matches = sorted(
        unique.values(), key=lambda row: (row["start_ns"], row["sequence"])
    )
    by_stage = defaultdict(list)
    for event in matches:
        by_stage[event["stage"]].append(event)
    reasons = []
    clocks = {row.get("clock_id") for row in matches}
    clock_verified = (
        len(clocks) == 1
        and None not in clocks
        and all(isinstance(clock, str) and HEX_ID.fullmatch(clock) for clock in clocks)
    )
    if not clock_verified:
        reasons.append(
            "clock_domain_mismatch"
            if len(clocks - {None}) > 1
            else "clock_domain_unverified"
        )
    starts = by_stage["http_request"]
    if len(starts) != 1:
        reasons.append("missing_http_start" if not starts else "ambiguous_http_start")
    start = starts[0]["start_ns"] if len(starts) == 1 else None
    # A protocol-first-body marker is deliberately not treated as a token.
    checkpoints = {
        "engine_first_output": "first_engine_output",
        "api_first_content": "first_api_content",
        "http_first_body": "http_first_body",
        "http_headers": "http_headers",
        "http_end": "http_end",
        "scheduler_first_output": "scheduler_first_output",
    }
    points = {}
    for label, stage in checkpoints.items():
        values = by_stage[stage]
        points[label] = min((row["end_ns"] for row in values), default=None)
    if not by_stage["internal_id_bridge"]:
        reasons.append("missing_internal_id_bridge")
    if points["engine_first_output"] is None:
        reasons.append("missing_engine_first_output")
    if len(by_stage["http_end"]) != 1:
        reasons.append(
            "missing_http_end" if not by_stage["http_end"] else "ambiguous_http_end"
        )
    elif by_stage["http_end"][0].get("success") is not True:
        reasons.append("http_request_failed_or_cancelled")
    if any(row.get("success") is False for row in matches):
        reasons.append("failed_phase")
    if invalid_records:
        reasons.append("invalid_records")
    if conflicts:
        reasons.append("conflicting_duplicate_events")
    if start is not None and any(
        value is not None and value < start for value in points.values()
    ):
        reasons.append("noncausal_timestamps")
    sequences = defaultdict(set)
    for row in valid:
        sequences[(row["pid"], row["trace_id"])].add(row["sequence"])
    lifecycles = {(row["pid"], row["trace_id"]) for row in matches}
    gaps = sum(
        max(sequences[key]) - min(sequences[key]) + 1 - len(sequences[key])
        for key in lifecycles
        if sequences[key]
    )
    if gaps:
        reasons.append("sequence_gaps")
    # One health file covers one producer; cross-process captures should include
    # all health snapshots. Missing health is unknown, never a clean zero.
    health_rows = [health] if isinstance(health, dict) else (health or [])
    health_by_key = {
        (row.get("pid"), row.get("trace_id")): row
        for row in health_rows
        if isinstance(row, dict)
    }
    unmatched_health = sorted(lifecycles - set(health_by_key))
    if unmatched_health:
        reasons.append("missing_lifecycle_health")
    for key in lifecycles & set(health_by_key):
        row = health_by_key[key]
        if (
            not clock_verified
            or not isinstance(row.get("clock_id"), str)
            or row["clock_id"] not in clocks
        ):
            reasons.append("lifecycle_clock_unverified")
            clock_verified = False
        if type(row.get("started_ns")) is not int or row["started_ns"] < 0:
            reasons.append("missing_lifecycle_start")
        elif any(
            event["start_ns"] < row["started_ns"]
            for event in matches
            if (event["pid"], event["trace_id"]) == key
        ):
            reasons.append("event_precedes_lifecycle")
        if (
            any(
                event["stage"] == "scheduler_admitted"
                and (event["pid"], event["trace_id"]) == key
                for event in matches
            )
            and row.get("worker_first_work_hooks") is not True
        ):
            reasons.append("worker_first_work_hooks_unconfirmed")
        if any(
            type(row.get(name)) is not int or row[name] < 0
            for name in ("dropped", "write_errors")
        ):
            reasons.append("invalid_lifecycle_health")
        elif row["dropped"] or row["write_errors"] or row.get("context_drops", 0):
            reasons.append("recorder_loss")
    selected_label = (
        "api_first_content"
        if points["api_first_content"] is not None
        else "engine_first_output"
    )
    end = points[selected_label]
    window_valid = (
        clock_verified and start is not None and end is not None and end >= start
    )
    unterminated = []
    if window_valid:
        for row in matches:
            if not row["stage"].startswith("phase_enter_") or row["start_ns"] >= end:
                continue
            stage = "phase_" + row["stage"].removeprefix("phase_enter_")
            if not any(
                close["pid"] == row["pid"]
                and close["trace_id"] == row["trace_id"]
                and close["start_ns"] <= row["start_ns"] <= close["end_ns"]
                for close in by_stage[stage]
            ):
                unterminated.append(stage)
        if unterminated:
            reasons.append("unterminated_phase")
    intervals = defaultdict(list)
    timeline = []
    for row in matches:
        timeline.append(
            {
                "stage": row["stage"],
                "pid": row["pid"],
                "trace_id": row["trace_id"],
                "sequence": row["sequence"],
                "start_offset_ms": (row["start_ns"] - start) / 1e6
                if clock_verified and start is not None
                else None,
                "end_offset_ms": (row["end_ns"] - start) / 1e6
                if clock_verified and start is not None
                else None,
                **{
                    name: row[name]
                    for name in (
                        *IDENTITIES,
                        "success",
                        "status_code",
                        "bytes",
                        "input_tokens",
                        "computed_tokens",
                        "scheduled_tokens",
                    )
                    if name in row
                },
            }
        )
        if window_valid and row["stage"] not in REQUEST_ENVELOPES:
            first, last = max(start, row["start_ns"]), min(end, row["end_ns"])
            if last > first:
                intervals[row["stage"]].append((first, last))
    covered = (
        union_ms([span for spans in intervals.values() for span in spans])
        if window_valid
        else None
    )
    elapsed = (end - start) / 1e6 if window_valid else None
    timings = {
        label + "_ms": (value - start) / 1e6
        if clock_verified and start is not None and value is not None and value >= start
        else None
        for label, value in points.items()
    }
    return {
        "schema": "urn:coherence:request-timeline:v1",
        "request_id": request_id,
        "status": "INCOMPLETE" if reasons else "COMPLETE",
        "incomplete_reasons": sorted(set(reasons)),
        "latencies": timings,
        "window": {
            "start": "http_request",
            "end": selected_label,
            "elapsed_ms": elapsed,
        },
        "covered_union_ms": covered,
        "unattributed_ms": max(0.0, elapsed - covered) if window_valid else None,
        "overlap_by_stage_ms": {
            stage: union_ms(spans) for stage, spans in intervals.items()
        },
        "timeline": timeline,
        "coverage": {
            "invalid_records": invalid_records,
            "sequence_gaps": gaps,
            "conflicting_duplicates": conflicts,
            "clock_domain_verified": clock_verified,
            "unterminated_phases": sorted(set(unterminated)),
            "missing_health_lifecycles": [
                {"pid": pid, "trace_id": trace} for pid, trace in unmatched_health
            ],
            "recorder_health": health_rows,
        },
        "interpretation": (
            "Engine TTFT, serialized API content, first HTTP body and completed HTTP wall time are separate boundaries. "
            "An HTTP body can be a role-only SSE event, not a generated token. Cross-process joins require explicit "
            "hashed-ID bridges on the same host/boot. Phase spans are clipped to the selected window and covered "
            "time is their union; per-stage overlaps must not be summed. Unattributed time is an observed gap, "
            "not an identified cause. COMPLETE means captured boundaries/health are present, not proof of a "
            "blocking mechanism or guaranteed retention; missing/drop/corrupt/failed evidence is INCOMPLETE."
        ),
    }


def analyze(rounds, events, health=None, *, threshold_ms=65):
    # Joining on lifecycle identity prevents attributing old-process events.
    indexed = defaultdict(list)
    sequences = defaultdict(set)
    for row in events:
        key = (row["trace_id"], row["pid"])
        indexed[key].append(row)
        sequences[key].add(row["sequence"])
    outliers = []
    timed = [r for r in rounds if isinstance(r.get("round_ms"), (int, float))]
    for row in timed:
        if row["round_ms"] < threshold_ms:
            continue
        end = row.get("monotonic_ns")
        key = (row.get("cache_trace_id"), row["pid"])
        matches = []
        if end is not None and key in indexed:
            start = end - int(row["round_ms"] * 1e6)
            matches = [
                e for e in indexed[key] if e["end_ns"] > start and e["start_ns"] < end
            ]
        by_stage = defaultdict(list)
        for event in matches:
            by_stage[event["stage"]].append(
                (max(start, event["start_ns"]), min(end, event["end_ns"]))
            )
        outliers.append(
            {
                "request_id": row["request_id"],
                "round": row["round"],
                "round_ms": row["round_ms"],
                "computed_tokens": row.get("computed_tokens"),
                "matched_lifecycle": end is not None and key in indexed,
                "overlap_by_stage_ms": {
                    name: round(union_ms(spans), 6) for name, spans in by_stage.items()
                },
                "events": [
                    {
                        "start_offset_ms": round((e["start_ns"] - start) / 1e6, 6),
                        "end_offset_ms": round((e["end_ns"] - start) / 1e6, 6),
                        **{
                            name: e[name]
                            for name in (
                                "stage",
                                "duration_ms",
                                "thread_cpu_ms",
                                "minor_faults",
                                "major_faults",
                                "job_kind",
                                "job_id",
                                "direction",
                                "request_id",
                                "round",
                                "bytes",
                                "logical_ranges",
                                "gpu_elapsed_ms",
                                "gpu_inter_round_gap_ms",
                                "thread_user_ms",
                                "thread_system_ms",
                                "off_cpu_ms",
                                "voluntary_switches",
                                "involuntary_switches",
                                "origin_thread_id",
                                "stream_match",
                                "lifetime_ms",
                                "success",
                                "thread_id",
                            )
                            if name in e
                        },
                    }
                    for e in matches
                ],
            }
        )
    gaps = sum(
        max(values) - min(values) + 1 - len(values)
        for values in sequences.values()
        if values
    )
    return {
        "schema": "urn:coherence:cache-round-correlation:v1",
        "timed_rounds": len(timed),
        "threshold_ms": threshold_ms,
        "outlier_count": len(outliers),
        "outliers": outliers,
        "coverage": {
            "sequence_gaps": gaps,
            "recorder_health": health,
            "unmatched_outliers": sum(not r["matched_lifecycle"] for r in outliers),
        },
        "interpretation": (
            "Intervals overlap; this does not establish the blocking cause. gpu_elapsed_ms uses "
            "the handler's existing completed HIP events. gpu_complete lifetime includes host "
            "submission, queue/dependency waits and completion polling delay. CPU spans can nest "
            "and run concurrently; never add their durations to infer round cost. Missing or "
            "dropped events, rotation and incomplete boundary jobs limit attribution. "
            "main_thread_round measures thread counters over the completed-step interval, "
            "including work outside execute_model. off_cpu_ms is wall minus thread CPU, not "
            "an identified wait reason. gpu_round is the current-stream marker span from "
            "execute_model entry to sample_tokens return; it includes GPU idle/dependency "
            "gaps inside that span, not just active kernels. Its inter-round gap is reported "
            "only for consecutive successful rounds of the same request and stream."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-log", type=Path, action="append")
    parser.add_argument("--cache-log", type=Path, action="append", required=True)
    parser.add_argument("--health", type=Path, action="append")
    parser.add_argument(
        "--request-id",
        help="Hashed HTTP, external or internal request ID for first-output analysis",
    )
    parser.add_argument("--threshold-ms", type=float, default=65)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.threshold_ms <= 0:
        parser.error("threshold must be positive")
    events, bad_events = read_records(args.cache_log, SCHEMA)
    health_rows = [json.loads(path.read_text()) for path in args.health or []]
    if args.request_id:
        if not HEX_ID.fullmatch(args.request_id):
            parser.error("request ID must be a lowercase SHA-256 digest")
        result = analyze_request(
            events, args.request_id, health_rows, invalid_records=bad_events
        )
    else:
        if not args.round_log:
            parser.error("--round-log is required unless --request-id is supplied")
        rounds, bad_rounds = read_records(args.round_log, ROUND_SCHEMA)
        health = health_rows[0] if len(health_rows) == 1 else health_rows or None
        result = analyze(rounds, events, health, threshold_ms=args.threshold_ms)
        result["coverage"]["invalid_records"] = bad_rounds + bad_events
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
