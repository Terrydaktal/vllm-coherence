"""Exercise the production sink rather than an unfiltered capture callback."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_prefix_lineage_runtime import (
    Tokenizer,
    first_response,
    record_render,
    render,
    request,
    state,
)

from qwen_r9700_lab import radiance_cache_telemetry as telemetry
from qwen_r9700_lab import radiance_prefix_runtime as runtime


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    value = telemetry.Recorder(tmp_path / "status", start=False, gc_events=False)
    monkeypatch.setattr(telemetry, "_recorder", value)
    monkeypatch.setattr(telemetry, "_round", {})
    monkeypatch.setattr(runtime, "_observer", None)
    monkeypatch.setattr(runtime, "_observer_pid", None)
    yield value
    value.close()


def events(recorder):
    recorder.flush()
    while recorder.pending:
        recorder.flush()
    return [json.loads(line) for line in recorder.path.read_text().splitlines()]


def test_actual_template_and_token_differences_survive_the_production_sink(recorder):
    tokenizer = Tokenizer()
    messages = first_response(
        tokenizer, reasoning=" r ", content=" answer ", noncanonical=True
    )
    current = state("b")
    runtime.begin(current, request(messages), tokenizer, {})
    text = render(messages, st=current, tokenizer=tokenizer)
    record_render(current, tokenizer, text)
    rows = [
        item
        for item in events(recorder)
        if item["stage"] == "prefix_lineage" and item.get("http_request_id") == "b" * 64
    ]
    roundtrip = next(row for row in rows if row["comparison"] == "raw_to_reencoded")
    assert roundtrip["diagnostic_status"] == "complete"
    assert roundtrip["equal"] is False
    assert roundtrip["first_difference"] < roundtrip["previous_tokens"]
    template = next(
        row
        for row in rows
        if row["comparison"] == "template_normalization"
        and row.get("comparison_complete") is True
    )
    assert template["equal"] is False and template["trim_changed"] is True
    assert template["unit"] == "characters"
    assert template["previous_characters"] > template["current_characters"]
    delimiters = next(
        row for row in rows if row.get("operation") == "assistant_delimiters"
    )
    assert delimiters["scope"] == "assistant_message_without_terminator"
    assert delimiters["recognized_stop_marker_removed"] is True
    assert {row["comparison"] for row in rows} == {
        "raw_to_reencoded",
        "output_to_message",
        "template_normalization",
        "prompt_prefix",
        "input_processor",
    }
    # None of the captured raw IDs/text or ephemeral content fingerprints are logs.
    serialized = recorder.path.read_text()
    assert '"token_ids"' not in serialized
    assert '"messages"' not in serialized
    assert '"fingerprint"' not in serialized
    assert "synthetic system" not in serialized
    assert " answer " not in serialized
    assert recorder.prefix_lineage_hooks["engine_tokens"] is True
    assert set(recorder.source_hashes) >= {
        "recorder",
        "prefix_lineage",
        "prefix_runtime",
    }


@pytest.mark.parametrize(
    "reason",
    ["restart", "evicted", "unsupported", "partial_output", "overlapping_requests"],
)
def test_observer_gap_and_loss_counts_survive_the_production_sink(recorder, reason):
    runtime._emit(
        {
            "stage": "prefix_lineage",
            "http_request_id": "c" * 64,
            "comparison": "prompt_prefix",
            "diagnostic_status": "unavailable",
            "reason": reason,
            "dropped_records": 3,
            "secret_text": "must not persist",
            "token_ids": [999],
        }
    )
    (row,) = events(recorder)
    assert row["reason"] == reason and row["dropped_records"] == 3
    assert row["http_request_id"] == "c" * 64
    assert row["diagnostic_status"] == "unavailable"
    assert "secret_text" not in row and "token_ids" not in row


def test_missing_http_state_is_a_noop():
    runtime.finish(None, True)
    runtime.output(None, SimpleNamespace(outputs=[]))
