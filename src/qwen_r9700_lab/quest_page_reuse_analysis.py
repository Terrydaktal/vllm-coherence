"""Analyze exact-selector page-ID traces for safe temporal reuse cadences."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any


class PageReuseAnalysisError(ValueError):
    """Raised when a trace cannot support an exact-selector reuse simulation."""


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p1": None,
            "minimum": None,
            "maximum": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": float(sum(ordered) / len(ordered)),
        "median": _quantile(ordered, 0.5),
        "p10": _quantile(ordered, 0.1),
        "p1": _quantile(ordered, 0.01),
        "minimum": float(ordered[0]),
        "maximum": float(ordered[-1]),
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def load_page_set_trace(path: Path) -> list[dict[str, Any]]:
    """Load and validate qualification JSONL records."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise PageReuseAnalysisError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            _validate_record(record, path=path, line_number=line_number)
            records.append(record)
    if not records:
        raise PageReuseAnalysisError(f"{path}: trace contains no records")
    return records


def _validate_record(record: Any, *, path: Path, line_number: int) -> None:
    if not isinstance(record, dict):
        raise PageReuseAnalysisError(f"{path}:{line_number}: record must be an object")
    required = {
        "format",
        "pid",
        "request_id",
        "layer_id",
        "round_index",
        "sequence_length",
        "budget_pages",
        "recent_pages",
        "historical_pages",
        "selection_source",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise PageReuseAnalysisError(f"{path}:{line_number}: missing fields: {', '.join(missing)}")
    if record["format"] != 1:
        raise PageReuseAnalysisError(f"{path}:{line_number}: unsupported format")
    integer_fields = required - {"historical_pages", "selection_source"}
    if any(not isinstance(record[field], int) for field in integer_fields):
        raise PageReuseAnalysisError(f"{path}:{line_number}: integer field has invalid type")
    if record["selection_source"] not in {"fresh", "reuse"}:
        raise PageReuseAnalysisError(f"{path}:{line_number}: invalid selection_source")
    pages = record["historical_pages"]
    if not isinstance(pages, list) or any(
        not isinstance(page, int) or isinstance(page, bool) for page in pages
    ):
        raise PageReuseAnalysisError(f"{path}:{line_number}: historical_pages must be integers")
    if pages != sorted(set(pages)):
        raise PageReuseAnalysisError(
            f"{path}:{line_number}: historical_pages must be unique and ascending"
        )
    historical_count = record["budget_pages"] - record["recent_pages"]
    if historical_count <= 0 or len(pages) != historical_count:
        raise PageReuseAnalysisError(
            f"{path}:{line_number}: historical page count does not match policy"
        )
    total_pages = (record["sequence_length"] + 63) // 64
    recent_start = total_pages - record["recent_pages"]
    if pages and (pages[0] < 0 or pages[-1] >= recent_start):
        raise PageReuseAnalysisError(
            f"{path}:{line_number}: historical page overlaps the mandatory recent tail"
        )


def _record_group(record: dict[str, Any]) -> tuple[int, int, int]:
    return int(record["pid"]), int(record["request_id"]), int(record["layer_id"])


def _comparison(
    base: dict[str, Any], current: dict[str, Any], *, refresh_interval: int
) -> dict[str, Any]:
    base_historical = set(base["historical_pages"])
    current_historical = set(current["historical_pages"])
    current_total_pages = (int(current["sequence_length"]) + 63) // 64
    current_recent = set(
        range(current_total_pages - int(current["recent_pages"]), current_total_pages)
    )
    reuse_selection = base_historical | current_recent
    fresh_selection = current_historical | current_recent
    intersection = base_historical & current_historical
    return {
        "pid": int(current["pid"]),
        "request_id": int(current["request_id"]),
        "layer_id": int(current["layer_id"]),
        "refresh_interval": refresh_interval,
        "refresh_round": int(base["round_index"]),
        "round_index": int(current["round_index"]),
        "rounds_since_refresh": int(current["round_index"]) - int(base["round_index"]),
        "sequence_length": int(current["sequence_length"]),
        "historical_jaccard": _jaccard(base_historical, current_historical),
        "effective_selected_jaccard": _jaccard(reuse_selection, fresh_selection),
        "historical_retention": len(intersection) / len(current_historical),
        "removed_historical_pages": len(base_historical - current_historical),
        "added_historical_pages": len(current_historical - base_historical),
    }


def _summarize_comparisons(comparisons: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "observations": len(comparisons),
        "historical_jaccard": _distribution(
            [float(record["historical_jaccard"]) for record in comparisons]
        ),
        "effective_selected_jaccard": _distribution(
            [float(record["effective_selected_jaccard"]) for record in comparisons]
        ),
        "historical_retention": _distribution(
            [float(record["historical_retention"]) for record in comparisons]
        ),
        "removed_historical_pages": _distribution(
            [float(record["removed_historical_pages"]) for record in comparisons]
        ),
        "exact_historical_match_fraction": (
            sum(record["historical_jaccard"] == 1.0 for record in comparisons) / len(comparisons)
            if comparisons
            else None
        ),
    }


def analyze_page_set_records(
    records: Iterable[dict[str, Any]],
    *,
    refresh_intervals: Sequence[int] = (2, 4, 8),
    selector_ms_per_layer: float | None = None,
    full_attention_layers: int = 16,
) -> dict[str, Any]:
    """Simulate reuse against fresh selector output for each layer and request."""

    records = list(records)
    if not records:
        raise PageReuseAnalysisError("trace contains no records")
    intervals = tuple(dict.fromkeys(int(interval) for interval in refresh_intervals))
    if not intervals or any(interval < 2 for interval in intervals):
        raise PageReuseAnalysisError("refresh intervals must all be at least two")
    if any(record["selection_source"] != "fresh" for record in records):
        raise PageReuseAnalysisError(
            "reuse simulation requires a trace captured with QWEN_QUEST_PAGE_REUSE=0"
        )
    if selector_ms_per_layer is not None and selector_ms_per_layer < 0:
        raise PageReuseAnalysisError("selector_ms_per_layer must be non-negative")
    if full_attention_layers < 1:
        raise PageReuseAnalysisError("full_attention_layers must be positive")

    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_record_group(record)].append(record)
    for key, group in grouped.items():
        group.sort(key=lambda record: int(record["round_index"]))
        rounds = [int(record["round_index"]) for record in group]
        if len(rounds) != len(set(rounds)):
            raise PageReuseAnalysisError(f"duplicate round in group {key}")
        if rounds != list(range(rounds[0], rounds[0] + len(rounds))):
            raise PageReuseAnalysisError(f"non-consecutive rounds in group {key}")

    consecutive: list[dict[str, Any]] = []
    interval_comparisons: dict[int, list[dict[str, Any]]] = {interval: [] for interval in intervals}
    for group in grouped.values():
        for previous, current in pairwise(group):
            consecutive.append(_comparison(previous, current, refresh_interval=1))
        for interval in intervals:
            for offset in range(0, len(group), interval):
                base = group[offset]
                for current in group[offset + 1 : offset + interval]:
                    interval_comparisons[interval].append(
                        _comparison(base, current, refresh_interval=interval)
                    )

    interval_results: dict[str, Any] = {}
    for interval, comparisons in interval_comparisons.items():
        per_layer: dict[str, Any] = {}
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for comparison in comparisons:
            by_layer[int(comparison["layer_id"])].append(comparison)
        for layer_id, layer_comparisons in sorted(by_layer.items()):
            per_layer[str(layer_id)] = _summarize_comparisons(layer_comparisons)
        result = {
            "selector_fraction_skipped": (interval - 1) / interval,
            "overall": _summarize_comparisons(comparisons),
            "per_layer": per_layer,
        }
        if selector_ms_per_layer is not None:
            fresh_round_ms = selector_ms_per_layer * full_attention_layers
            result["projected_timing"] = {
                "source_selector_ms_per_layer": selector_ms_per_layer,
                "full_attention_layers": full_attention_layers,
                "fresh_selector_ms_per_round": fresh_round_ms,
                "amortized_selector_ms_per_round": fresh_round_ms / interval,
                "saved_ms_per_round": fresh_round_ms * (interval - 1) / interval,
            }
        interval_results[str(interval)] = result

    return {
        "format": 1,
        "fresh_selector_records": len(records),
        "request_layer_groups": len(grouped),
        "requests": len({(record["pid"], record["request_id"]) for record in records}),
        "layers": sorted({int(record["layer_id"]) for record in records}),
        "rounds_per_group": _distribution([float(len(group)) for group in grouped.values()]),
        "consecutive": _summarize_comparisons(consecutive),
        "refresh_intervals": interval_results,
    }


def _parse_intervals(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("intervals must be comma-separated integers") from error
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("intervals must all be at least two")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze fresh Quest historical page-ID JSONL at reuse intervals."
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refresh-intervals", type=_parse_intervals, default=(2, 4, 8))
    parser.add_argument(
        "--selector-ms-per-layer",
        type=float,
        help="optional measured selector time used only for projected savings",
    )
    parser.add_argument("--full-attention-layers", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        report = analyze_page_set_records(
            load_page_set_trace(args.trace),
            refresh_intervals=args.refresh_intervals,
            selector_ms_per_layer=args.selector_ms_per_layer,
            full_attention_layers=args.full_attention_layers,
        )
    except (OSError, PageReuseAnalysisError) as error:
        parser.error(str(error))
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
