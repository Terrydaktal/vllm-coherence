#!/usr/bin/env python3
"""Join content-free cache/GC spans to slow round intervals; overlap is not causation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SCHEMA = "urn:coherence:cache-job-timings:v1"
ROUND_SCHEMA = "urn:qwen-r9700:decode-rounds:v1"


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
    parser.add_argument("--round-log", type=Path, action="append", required=True)
    parser.add_argument("--cache-log", type=Path, action="append", required=True)
    parser.add_argument("--health", type=Path)
    parser.add_argument("--threshold-ms", type=float, default=65)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.threshold_ms <= 0:
        parser.error("threshold must be positive")
    rounds, bad_rounds = read_records(args.round_log, ROUND_SCHEMA)
    events, bad_events = read_records(args.cache_log, SCHEMA)
    health = json.loads(args.health.read_text()) if args.health else None
    result = analyze(rounds, events, health, threshold_ms=args.threshold_ms)
    result["coverage"]["invalid_records"] = bad_rounds + bad_events
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
