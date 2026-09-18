"""Summarize the completed natural-output benchmark using public metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_private_head import write
from benchmark_private_head_long import MODES, complete, totals
from summarize_private_head import estimate


def validate_replay(stage, expected_captures):
    if (
        stage["status"] != "NATIVE_HEAD_REPLAY_MEASURED"
        or stage["capture_calls"] != expected_captures
        or stage["reference_matches_in_model"] is not True
    ):
        raise ValueError("head replay did not verify every captured reference invocation")
    shapes = stage["row_shape_histogram"]
    rows = sum(int(shape) * count for shape, count in shapes.items())
    if not rows or sum(shapes.values()) != expected_captures or rows != stage["capture_rows"]:
        raise ValueError("head replay capture rows do not cover the recorded calls")
    for mode in MODES:
        recall = stage["recall"][mode]
        if recall["rows"] != rows or any(
            not 0 <= recall[field] <= rows
            for field in (
                "argmax_retained",
                "any_maximum_retained",
                "top20_complete",
                "argmax_equal",
            )
        ):
            raise ValueError("head replay recall denominator or numerator is inconsistent")
    full = stage["recall"]["full"]
    if (
        full["argmax_equal"] != rows
        or full["top20_complete"] != rows
        or full["retained_logit_mismatches"] != 0
    ):
        raise ValueError("full-head reference failed its own exact comparison")


def summarize(root):
    def read(name):
        return json.loads((root / name).read_text())

    method = read("methodology.json")
    if read("completed.json")["status"] != "MEASURED":
        raise ValueError("native benchmark has not completed")
    target = method["minimum_output_tokens_per_variant"]
    records, captured = read("generation.json"), read("capture-generation.json")
    if target < 60000 or not complete(records, target) or not complete(captured, target, ("full",)):
        raise ValueError("refuse an incomplete per-variant 60K output benchmark")
    stage = read("head-stage.json")
    calls = {r["trial"]: r for r in read("head-calls.json")}
    expected_captures = sum(calls[r["trial"]]["calls"] for r in captured)
    validate_replay(stage, expected_captures)
    baseline = [r for r in records if r["mode"] == "full"]
    pairs = sorted({r["pair"] for r in records})
    for pair in pairs:
        rows = [r for r in records if r["pair"] == pair]
        if len(rows) != 4 or {r["mode"] for r in rows} != set(MODES):
            raise ValueError("incomplete variant pairing")
        if len({(r["fixture"], r["seed"], r["input_tokens"]) for r in rows}) != 1:
            raise ValueError("variants did not receive matching prompt/seed pairs")
    variants = {}
    for mode in MODES:
        rows = [r for r in records if r["mode"] == mode]
        seconds = sum(r["post_first_seconds"] for r in rows)
        output = sum(r["output_tokens"] for r in rows)
        decode = sum(r["output_tokens"] - r["first_chunk_tokens"] for r in rows)
        estimates = [estimate(r, calls[r["trial"]], stage, mode) for r in baseline]
        known = bool(estimates) and all(r is not None for r in estimates)
        shape = stage["shape_times"].get("8", {}).get(mode)
        if shape is None:
            raise ValueError("missing native M8 head timing")
        variants[mode] = {
            "requests": len(rows),
            "output_tokens": output,
            "natural_stops": sum(r["finish_reason"] == "stop" for r in rows),
            "length_stops": sum(r["finish_reason"] == "length" for r in rows),
            "request_seconds": sum(r["wall_seconds"] for r in rows),
            "post_first_seconds": seconds,
            "measured_post_first_tps": decode / seconds,
            "head_m8": shape,
            "recall": stage["recall"][mode],
            "verification_calls": sum(calls[r["trial"]]["shapes"].get("8", 0) for r in rows),
            "estimated_fixed_acceptance_tps": (
                sum(r["output_tokens"] - r["first_chunk_tokens"] for r in baseline)
                / sum(r["post_first_seconds"] for r in estimates)
                if known
                else None
            ),
        }
        steps = variants[mode]["verification_calls"]
        compatible = all(
            not (set(calls[r["trial"]]["shapes"]) - {"1", "8"})
            and calls[r["trial"]]["first_rows"] == 1
            for r in rows
        )
        variants[mode]["emitted_tokens_per_verification_approx"] = (
            decode / steps if steps and compatible else None
        )
        variants[mode]["wall_ms_per_verification_approx"] = (
            seconds * 1000 / steps if steps and compatible else None
        )
    return {
        "methodology": method,
        "fixtures": read("fixtures.json"),
        "variants": variants,
        "generation_runs": records,
        "capture_runs": captured,
        "head_stage": stage,
        "cold_warmup": read("warmup.json"),
        "timed_output_tokens": totals(records),
        "captured_output_tokens": totals(captured, ("full",))["full"],
        "capture_matches_timed_reference": sum(r["matches_timed_reference"] for r in captured),
        "paired_output_matches": {
            mode: sum(
                next(r for r in records if r["pair"] == base["pair"] and r["mode"] == mode)[
                    "output_sha256"
                ]
                == base["output_sha256"]
                for base in baseline
            )
            for mode in MODES
        },
        "pair_count": len(pairs),
        "limitations": [
            "Private prefixes around 60K context; natural completions over repeated seeds.",
            "Head recall rows include rejected proposals, not just committed output tokens.",
            "Full-reference replay is digest-checked against every saved in-model head invocation.",
            "Observed top-k retention and argmax agreement are empirical, not completeness proofs.",
            "Numerical head prototype timings exclude public integration eligibility checks.",
            "The matrix was not interrupted; GPU use was serialized by the shared lease.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    write(args.root / "summary.json", result)
    print(json.dumps({"variants": result["variants"], "pairs": result["pair_count"]}, indent=2))
