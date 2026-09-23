from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load_timer(monkeypatch):
    class Event:
        sequence = 0

        def __init__(self, *, enable_timing):
            assert enable_timing is True
            self.order = None

        def record(self, stream):
            Event.sequence += 1
            self.order = Event.sequence

        def query(self):
            return self.order is not None

        def elapsed_time(self, other):
            assert self.order is not None and other.order is not None
            return float(other.order - self.order)

    class Stream:
        cuda_stream = 17

    sync_calls = []
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        Event=Event,
        current_stream=lambda: Stream(),
        synchronize=lambda: sync_calls.append(True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    spec = importlib.util.spec_from_file_location(
        "low_overhead_stage_events_test_instance",
        ROOT / "experiments/radiance-public/low_overhead_stage_events.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "low_overhead_stage_events_test_instance", module)
    spec.loader.exec_module(module)
    return module, sync_calls


def test_event_timer_resolves_asynchronously_and_reports_setup_wait_separately(
    monkeypatch,
):
    module, sync_calls = load_timer(monkeypatch)
    timer = module.LowOverheadStageTimer(max_event_pairs=8)
    timer.start()
    timer.begin_round()
    timer.annotate_round(accepted_tokens=8, rejected_tokens=0)
    with timer.scope("gdn_convolution"):
        pass
    with timer.scope("target_other"):
        pass
    timer.finish_round()

    # A completed round is drained without synchronizing.  Only start and
    # final evidence flush may wait for the device.
    assert len(sync_calls) == 1
    timer.stop()
    evidence = timer.finish()
    assert len(sync_calls) == 2
    assert evidence["schema"] == module.LOW_OVERHEAD_PROFILE_SCHEMA
    assert evidence["production_timing_eligible"] is False
    assert evidence["timing_contract"]["stage_metric"] == "hip_scope_interval"
    assert evidence["timing_contract"]["per_stage_event_probes"] is True
    assert evidence["timing_contract"]["zero_observer_effect_proven"] is False
    assert evidence["steps"] == 1
    assert evidence["stages"]["gdn_convolution"]["intervals"] == 1
    assert "target_other" not in evidence["stages"]
    assert evidence["rounds"][0]["round_span_ms"] == 3.0
    assert evidence["rounds"][0]["unattributed_gap_ms"] == 2.0
    assert evidence["rounds"][0]["named_stage_union_ms"] == 1.0
    assert [segment["gap_ms"] for segment in evidence["rounds"][0]["gap_segments"]] == [
        1.0,
        1.0,
    ]
    assert evidence["rounds"][0]["gap_segments"][0]["after_stages"] == [
        "gdn_convolution"
    ]
    assert [boundary["gap_ms"] for boundary in evidence["rounds"][0]["boundary_gaps"]] == [
        1.0,
        1.0,
    ]
    assert evidence["rounds"][0]["intervals_detail"][0]["stage"] == "gdn_convolution"
    assert evidence["rounds"][0]["host_round_ms"] >= 0
    assert evidence["rounds"][0]["host_observed_wait_ms"] >= 0
    assert evidence["rounds"][0]["metadata"] == {
        "accepted_tokens": 8,
        "rejected_tokens": 0,
    }
    assert evidence["host_round_us"] >= evidence["host_observed_wait_us"]
    assert evidence["overlap_us"] == 0.0
    assert evidence["pool_exhaustions"] == 0
    assert evidence["finish_sync_ms"] >= 0


def test_event_timer_reports_each_unattributed_boundary(monkeypatch):
    module, _ = load_timer(monkeypatch)
    timer = module.LowOverheadStageTimer(max_event_pairs=8)
    timer.start()
    timer.begin_round()
    with timer.scope("first"):
        pass
    with timer.scope("second"):
        pass
    timer.finish_round()
    timer.stop()
    evidence = timer.finish()
    round_evidence = evidence["rounds"][0]

    assert round_evidence["round_span_ms"] == 5.0
    assert round_evidence["named_stage_union_ms"] == 2.0
    assert round_evidence["unattributed_gap_ms"] == 3.0
    assert [segment["gap_ms"] for segment in round_evidence["gap_segments"]] == [
        1.0,
        1.0,
        1.0,
    ]
    assert [boundary["from"] for boundary in round_evidence["boundary_gaps"]] == [
        "round_start",
        "first",
        "second",
    ]
    assert [boundary["to"] for boundary in round_evidence["boundary_gaps"]] == [
        "first",
        "second",
        "round_end",
    ]
    assert len(round_evidence["host_boundary_gaps"]) == 3


def test_event_timer_fails_closed_when_the_bounded_pool_is_too_small(monkeypatch):
    module, _ = load_timer(monkeypatch)
    timer = module.LowOverheadStageTimer(max_event_pairs=3)
    timer.start()
    timer.begin_round()
    with timer.scope("first"):
        pass
    with timer.scope("second"):
        pass
    with (
        pytest.raises(RuntimeError, match="event pool exhausted"),
        timer.scope("third"),
    ):
        pass
