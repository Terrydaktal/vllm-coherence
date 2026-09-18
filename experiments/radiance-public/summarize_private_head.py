"""Summarize public timing/count metadata without loading private Pi captures."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def estimate(record, calls, stage, alternative, *, decode_rows=8):
    """Change head cost only, holding generated tokens/acceptance/schedule fixed."""
    counts = dict(calls["shapes"])
    # This pinned D7 run has M1 prefill heads followed by M8 verification
    # heads. Consecutive capture confirms that order. Prefill belongs to TTFT.
    if set(counts) - {"1", str(decode_rows)} or calls["first_rows"] != 1:
        return None
    counts.pop("1", None)
    delta_ms = 0.0
    for shape, count in counts.items():
        if not count:
            continue
        times = stage["shape_times"].get(shape)
        if times is None:
            return None  # Never substitute a different batch shape silently.
        delta_ms += count * (times[alternative]["mean_ms"] - times["full"]["mean_ms"])
    seconds = record["post_first_seconds"] + delta_ms / 1000
    if seconds <= 0:
        raise ValueError("head-only estimate implies a nonpositive decoding duration")
    return {
        "seed": record["seed"],
        "post_first_seconds": seconds,
        "post_first_tps": (record["output_tokens"] - record["first_chunk_tokens"]) / seconds,
        "head_time_change_seconds": delta_ms / 1000,
    }


def summarize(root):
    records = json.loads((root / "generation.json").read_text())
    stage = json.loads((root / "head-stage.json").read_text())
    trials = {row["trial"]: row for row in json.loads((root / "head-calls.json").read_text())}
    result = {
        "fixture": json.loads((root / "fixture.json").read_text()),
        "head_stage": stage,
        "cold_warmup": json.loads((root / "warmup.json").read_text()),
        "generation_runs": records,
        "estimated_tps_assumption": (
            "Replace only measured full-head cost in observed M8 target calls; exclude M1 prefill. "
            "The captured call order is two M1 prefill calls followed by 81 M8 calls; "
            "assume the same phase mapping for matching shapes in these fixed-prompt D7 trials. "
            "Hold output tokens, speculative acceptance and all other execution time fixed. "
            "This is an estimate, not a measured alternative continuation."
        ),
        "variants": {},
    }
    baseline = [r for r in records if r["mode"] == "full"]
    for mode in stage["recall"]:
        actual = [r for r in records if r["mode"] == mode]
        estimates = [estimate(r, trials[r["trial"]], stage, mode) for r in baseline]
        valid = [r for r in estimates if r is not None]
        tokens = sum(r["output_tokens"] - r["first_chunk_tokens"] for r in actual)
        seconds = sum(r["post_first_seconds"] for r in actual)
        result["variants"][mode] = {
            "head_median_ms": stage["gpu_event_times"][mode]["median_ms"],
            "head_mean_ms": stage["gpu_event_times"][mode]["mean_ms"],
            "head_p95_ms": stage["gpu_event_times"][mode]["p95_ms"],
            "measured_post_first_tps_pooled": tokens / seconds,
            "measured_post_first_tps_median": statistics.median(
                r["post_first_tps"] for r in actual
            ),
            "measured_output_tokens": sum(r["output_tokens"] for r in actual),
            "measured_wall_seconds": sum(r["wall_seconds"] for r in actual),
            "measured_post_first_seconds": seconds,
            "fixed_acceptance_estimates": estimates,
            "fixed_acceptance_estimated_tps_median": (
                statistics.median(r["post_first_tps"] for r in valid) if valid else None
            ),
            "fixed_acceptance_estimated_tps_pooled": (
                sum(r["output_tokens"] - r["first_chunk_tokens"] for r in baseline)
                / sum(r["post_first_seconds"] for r in valid)
                if len(valid) == len(baseline) and valid
                else None
            ),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_root", type=Path)
    arguments = parser.parse_args()
    result = summarize(arguments.report_root)
    (arguments.report_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fixture": result["fixture"], "variants": result["variants"]}, indent=2))
