"""Export only aggregate numbers from a completed private head benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmark_private_head_long import MODES
from summarize_private_head_long import summarize

LABELS = {
    "full": "Full BF16 fallback",
    "block80": "Original block-8/64 + rerank-80",
    "global128": "Global INT2 top-128 + BF16 rerank",
    "global256": "Global INT2 top-256 + BF16 rerank (default)",
}
TIMING_KEYS = ("samples", "median_ms", "mean_ms", "p95_ms", "min_ms", "max_ms")
RECALL_KEYS = (
    "rows",
    "argmax_retained",
    "any_maximum_retained",
    "top20_complete",
    "argmax_equal",
    "retained_logit_mismatches",
    "max_retained_logit_difference",
)
VARIANT_KEYS = (
    "requests",
    "output_tokens",
    "natural_stops",
    "length_stops",
    "request_seconds",
    "post_first_seconds",
    "measured_post_first_tps",
    "verification_calls",
    "estimated_fixed_acceptance_tps",
    "emitted_tokens_per_verification_approx",
    "wall_ms_per_verification_approx",
)


def numbers(source, keys):
    result = {key: source[key] for key in keys}
    if any(
        value is not None and (type(value) not in (int, float) or not math.isfinite(value))
        for value in result.values()
    ):
        raise ValueError("public aggregate contains a nonnumeric value")
    return result


def public_summary(summary):
    if summary["captured_output_tokens"] < 60000:
        raise ValueError("reference output budget is incomplete")
    stage = summary["head_stage"]
    fixture_sizes = [row["input_tokens"] for row in summary["fixtures"]]
    if not fixture_sizes or any(type(size) is not int or size < 1 for size in fixture_sizes):
        raise ValueError("invalid numeric fixture sizes")
    shape = stage["head_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(size) is not int or size < 1 for size in shape)
    ):
        raise ValueError("invalid numeric head shape")
    metadata = numbers(
        summary, ("captured_output_tokens", "pair_count", "capture_matches_timed_reference")
    )
    capture = numbers(stage, ("capture_calls", "capture_rows"))
    matches = numbers(summary["paired_output_matches"], MODES)
    if stage["reference_matches_in_model"] is not True:
        raise ValueError("full reference has not matched its in-model digests")
    variants = {}
    for mode in MODES:
        source = summary["variants"][mode]
        if source["output_tokens"] < 60000 or source["natural_stops"] != source["requests"]:
            raise ValueError("measured natural-completion budget is incomplete")
        variants[mode] = numbers(source, VARIANT_KEYS)
        variants[mode]["identical_output_pairs_to_timed_full_head"] = matches[mode]
        variants[mode]["head_m8_timing"] = numbers(source["head_m8"], TIMING_KEYS)
        variants[mode]["recall"] = numbers(source["recall"], RECALL_KEYS)
    if any(row["finish_reason"] != "stop" for row in summary["capture_runs"]):
        raise ValueError("reference capture includes a non-natural completion")
    return {
        "schema": "radiance-private-head-natural-output-v1",
        "workload": {
            "source": "Intact private Pi coding request boundaries; no tools executed.",
            "request_boundaries": len(fixture_sizes),
            "input_tokens_min": min(fixture_sizes),
            "input_tokens_max": max(fixture_sizes),
            "paired_comparisons": metadata["pair_count"],
            "minimum_measured_output_tokens_per_method": 60000,
            "natural_eos": True,
            "sampling": {"temperature": 1, "top_p": 0.95, "top_k": 20},
            "speculation": "Fixed D7, TP1",
            "warmup": "Entire first four-method comparison excluded; extension warmed all methods.",
        },
        "capture": {
            "output_tokens": metadata["captured_output_tokens"],
            "requests": len(summary["capture_runs"]),
            "identical_output_pairs_to_timed_reference": metadata[
                "capture_matches_timed_reference"
            ],
            "head_invocations": capture["capture_calls"],
            "prediction_rows": capture["capture_rows"],
            "reference_matches_saved_in_model_digests": True,
            "scope": (
                "Every saved consecutive call, including prefill and rejected speculative rows."
            ),
        },
        "head_shape": shape,
        "variants": variants,
        "total_timed_response_tokens": sum(v["output_tokens"] for v in variants.values()),
        "total_timed_request_seconds": sum(v["request_seconds"] for v in variants.values()),
        "total_timed_post_first_seconds": sum(v["post_first_seconds"] for v in variants.values()),
        "limitations": [
            "Empirical measurements on one private coding workload; no universal recall guarantee.",
            "Top-20 retention includes ties and does not imply equal logits or probabilities.",
            "Global shortlist selection and selective BF16 reranking remain approximate.",
            "Measured throughput pools post-first-output time; warmup and prefill are excluded.",
            "Approximate heads may generate different continuations and speculative acceptance.",
            "Paired output identity is reported separately from head recall on identical inputs.",
            "Fixed-acceptance estimates replace head time only in the full-head measurements.",
            "Head timing is per M8 invocation and excludes public eligibility checks.",
            "This head replay does not certify upstream hidden-state computation.",
        ],
    }


def fraction(matched, rows):
    return f"{matched:,}/{rows:,} ({100 * matched / rows:.4f}%)"


def table(result):
    lines = [
        "| Target path | Median M8 head time | Top-1 match | Complete reference top-20 retained "
        "| Measured tok/s | Estimated tok/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        value = result["variants"][mode]
        recall = value["recall"]
        estimate = value["estimated_fixed_acceptance_tps"]
        cells = [
            LABELS[mode],
            f"{value['head_m8_timing']['median_ms']:.3f} ms",
            fraction(recall["argmax_equal"], recall["rows"]),
            fraction(recall["top20_complete"], recall["rows"]),
            f"{value['measured_post_first_tps']:.1f}",
            "unavailable" if estimate is None else f"{estimate:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = public_summary(summarize(args.root))
    result["exporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "table.md").write_text(table(result))
    print(table(result))
