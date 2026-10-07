"""CPU coverage for binding worker/job telemetry to admitted scheduler rows."""

import hashlib
from types import SimpleNamespace

import pytest
from test_radiance_fair_scheduler import (
    BANK_A,
    CHAT_A,
    GEN_A,
    load_module,
    new_scheduler,
)


def request(request_id, *, computed=14_096):
    return SimpleNamespace(
        request_id=request_id,
        kv_transfer_params={"qwen_chat": {"id": CHAT_A, "generation": GEN_A}},
        num_prompt_tokens=60_000,
        num_computed_tokens=computed,
        is_finished=lambda: False,
    )


def prepare(monkeypatch):
    module = load_module(monkeypatch)
    scheduler = new_scheduler(module)
    scheduler.request_phases = module.RequestPhases()
    actual, previous = request("actually-admitted"), request("blocked-or-previous")
    scheduler.requests = {r.request_id: r for r in (actual, previous)}
    for item in scheduler.requests.values():
        scheduler.request_phases.set(item, "admission")
    old_row = scheduler.request_phases.live[previous.request_id]
    old_row.update(generation_rounds=50, last_round_ms=99.0)
    metadata = {
        "bank": BANK_A,
        "barrier": False,
        "cache_timing_context": {"request_id": old_row["request_id"], "round": 51},
        "last_round_ms": 99.0,
        "drop_banks": ["retired"],
        "response_end_copies": ["copy"],
    }
    return scheduler, actual, previous, metadata


@pytest.mark.parametrize(
    "boundary", ["fresh_prefill", "skipped_admission", "request_rollover"]
)
def test_schedule_and_early_connector_use_actual_admitted_request(
    monkeypatch, boundary
):
    scheduler, actual, previous, metadata = prepare(monkeypatch)
    # Ownership is selected before parent admission and can name a blocked or
    # finished predecessor. It must never decide a worker span's request identity.
    scheduler.response_request = None if boundary == "fresh_prefill" else previous
    scheduler.running = []
    scheduler.pending_switch = None
    scheduler.banks = SimpleNamespace(active=BANK_A)
    scheduler.last_served = {}
    scheduler._choose = lambda: BANK_A
    scheduler._refresh_request_phases = lambda: None
    scheduler._publish_status = lambda **_: None
    scheduler._attach_decode_sync_key = lambda *_: None
    metadata_calls, connector_views = [], []

    def worker_metadata(*args, **kwargs):
        metadata_calls.append((args, kwargs))
        return metadata

    scheduler._worker_metadata = worker_metadata

    def parent_step(*_):
        output = SimpleNamespace(
            qwen_fair=None,
            num_scheduled_tokens={actual.request_id: 4096, previous.request_id: 0},
            total_num_scheduled_tokens=4096,
        )
        scheduler.running = [actual]
        scheduler._build_kv_connector_meta(
            lambda value: connector_views.append(
                dict(value.qwen_fair["cache_timing_context"])
            ),
            output,
        )
        return output

    scheduler._parent_step = parent_step
    output = scheduler.schedule()
    expected = {
        "request_id": hashlib.sha256(actual.request_id.encode()).hexdigest(),
        "chat_id": CHAT_A,
        "generation": GEN_A,
        "round": 1,
        "input_tokens": 60_000,
        "computed_tokens": 10_000,
    }
    assert connector_views == [expected], (
        "connector jobs bind before parent schedule returns"
    )
    assert output.qwen_fair["cache_timing_context"] == expected
    assert output.qwen_fair["last_round_ms"] is None, (
        "a new answer cannot inherit old latency"
    )
    assert scheduler.request_phases.computed_tokens(actual) == 10_000
    assert len(metadata_calls) == 1, (
        "rebinding must not consume copy/drop metadata twice"
    )
    assert metadata["drop_banks"] == ["retired"]
    assert metadata["response_end_copies"] == ["copy"]
    assert scheduler.response_request is actual


def test_contiguous_generation_uses_actual_requests_last_completed_latency(monkeypatch):
    scheduler, actual, previous, metadata = prepare(monkeypatch)
    scheduler.running = [previous]
    scheduler.response_request = previous
    scheduler.request_phases.live[actual.request_id].update(
        generation_rounds=7, last_round_ms=44.5
    )
    output = SimpleNamespace(num_scheduled_tokens={actual.request_id: 8})
    scheduler._bind_timing_context(output, metadata)
    assert metadata["last_round_ms"] == 44.5
    assert metadata["cache_timing_context"]["round"] == 8
    assert (
        metadata["cache_timing_context"]["computed_tokens"]
        == actual.num_computed_tokens - 8
    )


@pytest.mark.parametrize(
    "shape", ["multi", "zero", "missing", "barrier", "unknown", "untracked"]
)
def test_unattributable_frame_discards_stale_request_and_latency(monkeypatch, shape):
    scheduler, actual, previous, metadata = prepare(monkeypatch)
    counts = {actual.request_id: 8}
    if shape == "multi":
        counts[previous.request_id] = 8
    elif shape == "zero":
        counts = {actual.request_id: 0}
    elif shape == "unknown":
        counts = {"not-owned": 8}
    elif shape == "untracked":
        scheduler.request_phases.live.pop(actual.request_id)
    elif shape == "barrier":
        metadata["barrier"] = True
    output = (
        SimpleNamespace()
        if shape == "missing"
        else SimpleNamespace(num_scheduled_tokens=counts)
    )
    scheduler._step_worker_metadata = metadata
    observed = scheduler._build_kv_connector_meta(lambda value: value.qwen_fair, output)
    assert observed is metadata
    assert "cache_timing_context" not in observed
    assert "last_round_ms" not in observed
    assert observed["drop_banks"] == ["retired"]


def test_stock_output_has_no_new_scheduler_metadata(monkeypatch):
    scheduler, actual, _, _ = prepare(monkeypatch)
    assert (
        scheduler._bind_timing_context(
            SimpleNamespace(num_scheduled_tokens={actual.request_id: 8}), None
        )
        is None
    )
