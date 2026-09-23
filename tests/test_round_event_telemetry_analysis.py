from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "tools/analyze_round_event_telemetry.py"


def load_module():
    spec = importlib.util.spec_from_file_location("round_event_analysis_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(index: int, span: float, *, dropped: int = 0, status: str = "complete"):
    return {
        "schema": "urn:qwen-r9700:decode-round-gpu:v2",
        "round": index,
        "observed_at_ms": index,
        "dropped_records_before": dropped,
        "scheduled_shape": {"mode": "decode", "scheduled_tokens": 8},
        "gpu_events": {
            "event_status": status,
            "round_span_ms": span,
            "gaps_ms": [
                {"from": "target.start", "to": "target.end", "same_stream": True, "ms": 1.5},
                {"from": "target.end", "to": "draft.start", "same_stream": False, "ms": None},
            ],
        },
        "dispatch": {"host_ms": span + 0.75, "telemetry_wait_ms": 0.0},
        "queue_sync": {"reason": None},
    }


def test_analyzer_reports_rounds_gaps_histogram_and_slow_episode(tmp_path):
    module = load_module()
    path = tmp_path / "rounds.jsonl"
    path.write_text("\n".join(json.dumps(row(i, 44.0 if i != 5 else 53.5)) for i in range(8)) + "\n")

    result = module.analyze([path], histogram_width_ms=0.5)

    assert result["status"] == "COMPLETE"
    assert result["private_chat_text_read"] is False
    assert result["records"] == 8
    assert result["round_ms"]["median_ms"] == 44.0
    assert result["gpu_gap_total_ms"] == pytest.approx(12.0)
    assert result["unavailable_cross_stream_gaps"] == 8
    assert result["host_wait_ms"]["mean_ms"] == pytest.approx(0.75)
    assert result["slow_episode_diagnostics"]["slow_rounds"] == 1
    assert result["slow_episode_diagnostics"]["episodes"][0]["first_round_index"] == 5
    assert result["by_shape"]


def test_analyzer_rejects_dropped_or_incomplete_rows(tmp_path):
    module = load_module()
    dropped = tmp_path / "dropped.jsonl"
    dropped.write_text(json.dumps(row(1, 44.0, dropped=1)) + "\n")
    with pytest.raises(ValueError, match="dropped"):
        module.analyze([dropped])

    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text(json.dumps(row(1, 44.0, status="pending")) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        module.analyze([incomplete])


def test_analyzer_can_explicitly_derive_legacy_round_span_and_group_runtime_shape(tmp_path):
    module = load_module()
    path = tmp_path / "legacy.jsonl"
    legacy = row(1, 44.0)
    legacy["gpu_events"].pop("round_span_ms")
    legacy["gpu_events"]["durations_ms"] = [{"label": "round.gpu", "ms": 44.0}]
    legacy["runtime_shape"] = {"cudagraph_mode": "PIECEWISE", "padded_tokens": 8}
    path.write_text(json.dumps(legacy) + "\n")

    with pytest.raises(ValueError, match="round span is missing"):
        module.analyze([path])
    result = module.analyze([path], derive_round_span=True)
    assert result["round_span_source"] == "round.gpu_duration_legacy"
    assert result["by_runtime_shape"]
