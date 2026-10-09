"""Keep finer README display bins faithful to every recorded round."""

import copy
import json
import math
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import render_current_tables as renderer


def captured_report(values):
    """Use coarse capture bins so the display must rebin original records."""
    bounds = [("<35", -math.inf, 35), ("35–50", 35, 50),
              ("50–500", 50, 500), ("≥500", 500, math.inf)]
    records = [{"round": 1, "round_ms": None}]
    records.extend({"round": index, "round_ms": value}
                   for index, value in enumerate(values, 2))
    capture = {
        "status": "captured",
        "record_count": len(records),
        "measured_round_count": len(values),
        "unmeasured_round_count": 1,
        "records": records,
        "histogram": {
            "measured_round_count": len(values), "unmeasured_round_count": 1,
            "bins": [{"label": label, "count": sum(lo <= v < hi for v in values)}
                     for label, lo, hi in bounds],
            "mean_ms": statistics.mean(values) if values else None,
            "median_ms": statistics.median(values) if values else None,
        },
    }
    return {"suite_capture_id": "test", "measurement_mode": "histogram_only_after_profile",
            "contexts": {
        context: {"round_capture": copy.deepcopy(capture)}
        for context in ("0K", "60K", "200K")
    }}


def render_report(report, tmp_path, monkeypatch):
    path = tmp_path / "rounds.json"
    path.write_text(json.dumps(report))
    original = path.read_bytes()
    monkeypatch.setattr(renderer, "ROUND_HISTOGRAM_RESULTS", path)
    lines = renderer.render_round_histogram()
    assert path.read_bytes() == original
    return lines


def rendered_bins(lines):
    result = {}
    for line in lines:
        if line.startswith("| `"):
            label, *cells = [cell.strip() for cell in line.split("|")[1:-1]]
            result[label.strip("`")] = [int(cell.split()[0].replace(",", ""))
                                          for cell in cells]
    return result


def test_display_rebins_boundaries_and_includes_all_outliers(tmp_path, monkeypatch):
    values = [0, 34.999, 35, 35.499, 35.5, 41.999, 42, 48.999,
              49, 49.499, 49.5, 50, 52.999, 53, 100, 499.999, 500, 9999]
    lines = render_report(captured_report(values), tmp_path, monkeypatch)
    bins = rendered_bins(lines)
    expected = {"<35": 2, "35–35.5": 2, "35.5–36": 1, "41.5–42": 1,
                "42–43": 1, "48.5–49": 1, "49–49.5": 2,
                "49.5–50": 1, "50–50.5": 1, "52.5–53": 1,
                "53–53.5": 1, "100–250": 1, "250–500": 1, "≥500": 2}
    assert {label: counts[0] for label, counts in bins.items() if counts[0]} == expected
    assert all(sum(counts[arm] for counts in bins.values()) == len(values)
               for arm in range(3))
    assert "| **Untimed events** | **1** | **1** | **1** |" in lines
    assert any("measurement identity are unchanged" in line for line in lines)


def test_display_current_clusters_and_complete_counts():
    lines = renderer.render_round_histogram()
    bins = rendered_bins(lines)
    report = json.loads(renderer.round_histogram_result_path().read_text())
    for arm, context in enumerate(("0K", "60K", "200K")):
        capture = report["contexts"][context]["round_capture"]
        values = [row["round_ms"] for row in capture["records"]
                  if row["round_ms"] is not None]
        for lower, upper in ((36.5, 37), (37, 37.5), (40.5, 41),
                             (50, 50.5), (50.5, 51)):
            assert bins[f"{lower:g}–{upper:g}"][arm] == sum(
                lower <= ms < upper for ms in values
            )
        assert sum(counts[arm] for counts in bins.values()) == len(values)
        assert bins["≥500"][arm] == sum(ms >= 500 for ms in values)


def test_histogram_only_rerun_does_not_replace_workload_tables(tmp_path, monkeypatch):
    coding_before = renderer.render_coding_context_benchmark()
    coverage_before = renderer.render_round_capture_summary()
    chained_before = renderer.render_chained_workload_results()
    original_coding = renderer.CODING_CONTEXT_RESULTS.read_bytes()
    report = captured_report([36.75, 41.0, 501])
    lines = render_report(report, tmp_path, monkeypatch)

    assert any("histogram-only rerun" in line for line in lines)
    assert any("separate unprofiled capture" in line for line in lines)
    assert rendered_bins(lines)["≥500"] == [1, 1, 1]
    assert renderer.render_coding_context_benchmark() == coding_before
    assert renderer.render_round_capture_summary() == coverage_before
    assert renderer.render_chained_workload_results() == chained_before
    assert renderer.CODING_CONTEXT_RESULTS.read_bytes() == original_coding


def test_histogram_uses_previous_complete_capture_until_rerun_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "ROUND_HISTOGRAM_RESULTS", tmp_path / "not-yet-captured.json")
    assert renderer.round_histogram_result_path() == renderer.CODING_CONTEXT_RESULTS
    assert any("coding-context benchmark" in line for line in renderer.render_round_histogram())


def test_regular_suite_histogram_reports_shared_capture_provenance(tmp_path, monkeypatch):
    report = captured_report([37, 751])
    report["measurement_mode"] = "shared_suite_control_after"
    report["coding_context_result"] = "pi-coding-contexts.json"
    lines = render_report(report, tmp_path, monkeypatch)
    text = "\n".join(lines)
    assert "shared-suite coding controls" in text
    assert "same predetermined clean control for the coding row and histogram" in text
    assert "histogram-only rerun" not in text
    assert "separate unprofiled capture" not in text
    assert "retain their original measurement captures" not in text
    symptoms = "\n".join(renderer.render_known_remaining_symptoms())
    assert "Isolated round stalls remain" in symptoms
    assert "current shared-suite coding controls" in symptoms
    assert "did not reproduce the large stalls" not in symptoms
    assert "591.203" not in symptoms


def test_capture_provenance_remains_true_after_the_repair_is_committed():
    lines = renderer.render_round_histogram()
    text = "\n".join(lines)
    report = json.loads(renderer.ROUND_HISTOGRAM_RESULTS.read_text())
    assert f"Captured source: base `{report['source_base_commit'][:7]}`" in text
    assert "exact file hashes" in text
    assert "uncommitted" not in text
    assert "filename-prefix repair was added after that source was frozen" in text
    assert "experiments/radiance-public/matched_stage_profile_worker.py" in text
    assert "0K stopped naturally below the requested 5,000-token minimum" in text
    assert "CPU and HIP diagnostics match all 4,188 selected timed rounds" in text
    assert "generic drop is separate from that complete round coverage" in text


@pytest.mark.parametrize("damaged", ["invalid_json", "missing_context", "partial_capture"])
def test_existing_rerun_never_silently_falls_back_to_old_measurements(
    damaged, tmp_path, monkeypatch
):
    report = captured_report([37])
    if damaged == "missing_context":
        del report["contexts"]["200K"]
    elif damaged == "partial_capture":
        report["contexts"]["60K"]["round_capture"]["status"] = "partial"
    path = tmp_path / "histogram.json"
    path.write_text("{" if damaged == "invalid_json" else json.dumps(report))
    monkeypatch.setattr(renderer, "ROUND_HISTOGRAM_RESULTS", path)
    with pytest.raises(ValueError, match="complete 0K/60K/200K captures"):
        renderer.render_round_histogram()


@pytest.mark.parametrize("value", [True, "37", -0.001, math.nan, math.inf])
def test_display_rejects_invalid_round_durations(value, tmp_path, monkeypatch):
    report = captured_report([37])
    report["contexts"]["0K"]["round_capture"]["records"][1]["round_ms"] = value
    with pytest.raises(ValueError, match="invalid round duration"):
        render_report(report, tmp_path, monkeypatch)


def test_display_rejects_dropped_record(tmp_path, monkeypatch):
    report = captured_report([37, 38])
    report["contexts"]["60K"]["round_capture"]["records"].pop()
    with pytest.raises(ValueError, match="every captured record"):
        render_report(report, tmp_path, monkeypatch)


def test_display_rejects_missing_or_duplicate_round(tmp_path, monkeypatch):
    report = captured_report([37, 38])
    report["contexts"]["60K"]["round_capture"]["records"][2]["round"] = 2
    with pytest.raises(ValueError, match="missing or duplicate round"):
        render_report(report, tmp_path, monkeypatch)


def test_display_rejects_missing_duration_field(tmp_path, monkeypatch):
    report = captured_report([37])
    del report["contexts"]["0K"]["round_capture"]["records"][1]["round_ms"]
    with pytest.raises(ValueError, match="invalid round duration"):
        render_report(report, tmp_path, monkeypatch)


@pytest.mark.parametrize("corruption,message", [
    ("timed_count", "count disagrees"),
    ("bin_count", "bin disagrees"),
    ("overlapping_bins", "overlap or have a gap"),
    ("dropped_bin", "does not cover every"),
    ("mean", "mean_ms disagrees"),
    ("median", "median_ms disagrees"),
])
def test_display_rejects_stored_capture_disagreement(
    corruption, message, tmp_path, monkeypatch
):
    report = captured_report([37, 501])
    capture = report["contexts"]["0K"]["round_capture"]
    histogram = capture["histogram"]
    if corruption == "timed_count":
        capture["measured_round_count"] += 1
    elif corruption == "bin_count":
        histogram["bins"][1]["count"] += 1
    elif corruption == "overlapping_bins":
        histogram["bins"][1]["label"] = "34–50"
    elif corruption == "dropped_bin":
        histogram["bins"].pop()
    elif corruption == "mean":
        histogram["mean_ms"] += 1
    elif corruption == "median":
        histogram["median_ms"] += 1
    with pytest.raises(ValueError, match=message):
        render_report(report, tmp_path, monkeypatch)
