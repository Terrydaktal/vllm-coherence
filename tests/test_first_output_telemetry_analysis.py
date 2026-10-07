from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HTTP, EXTERNAL, INTERNAL, OTHER = (character * 64 for character in "abcd")
CLOCK = "e" * 64


@pytest.fixture
def analyzer():
    spec = importlib.util.spec_from_file_location(
        "request_timeline_analyzer", ROOT / "tools/analyze_cache_job_telemetry.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event(stage, start, end=None, *, pid=10, trace="api", sequence=1, **values):
    return {
        "schema": "urn:coherence:cache-job-timings:v1",
        "pid": pid,
        "trace_id": trace,
        "clock_id": CLOCK,
        "sequence": sequence,
        "stage": stage,
        "start_ns": start * 1_000_000,
        "end_ns": (start if end is None else end) * 1_000_000,
        **values,
    }


def request_records():
    # An early protocol event is deliberately before any generated token.
    return [
        event("http_request", 100, http_request_id=HTTP),
        event("http_body_receive", 101, 110, sequence=2, http_request_id=HTTP),
        event(
            "api_render",
            110,
            130,
            sequence=3,
            http_request_id=HTTP,
            external_request_id=EXTERNAL,
            success=True,
        ),
        event("http_first_body", 140, sequence=4, http_request_id=HTTP),
        event(
            "internal_id_bridge",
            150,
            sequence=5,
            http_request_id=HTTP,
            external_request_id=EXTERNAL,
            request_id=INTERNAL,
        ),
        event(
            "async_input_process",
            130,
            170,
            sequence=6,
            request_id=INTERNAL,
            success=True,
        ),
        event("engine_submit", 165, 190, sequence=7, request_id=INTERNAL, success=True),
        event(
            "phase_prefill",
            200,
            290,
            pid=20,
            trace="engine",
            request_id=INTERNAL,
            success=True,
        ),
        event(
            "worker_execute",
            215,
            270,
            pid=20,
            trace="engine",
            sequence=2,
            request_id=INTERNAL,
            success=True,
        ),
        event(
            "scheduler_first_output",
            295,
            pid=20,
            trace="engine",
            sequence=3,
            request_id=INTERNAL,
        ),
        event("first_engine_output", 300, sequence=8, request_id=INTERNAL),
        event(
            "first_api_content",
            320,
            sequence=9,
            http_request_id=HTTP,
            external_request_id=EXTERNAL,
        ),
        event("http_end", 350, sequence=10, http_request_id=HTTP, success=True),
    ]


def healthy():
    return [
        {
            "pid": 10,
            "trace_id": "api",
            "clock_id": CLOCK,
            "started_ns": 0,
            "dropped": 0,
            "write_errors": 0,
        },
        {
            "pid": 20,
            "trace_id": "engine",
            "clock_id": CLOCK,
            "started_ns": 0,
            "dropped": 0,
            "write_errors": 0,
        },
    ]


def test_cross_process_timeline_preserves_wall_time_without_double_count(analyzer):
    result = analyzer.analyze_request(request_records(), HTTP, healthy())
    assert result["status"] == "COMPLETE"
    assert result["latencies"]["http_first_body_ms"] == 40
    assert result["latencies"]["engine_first_output_ms"] == 200
    assert result["latencies"]["api_first_content_ms"] == 220
    assert result["latencies"]["http_end_ms"] == 250
    assert result["covered_union_ms"] == 179
    assert result["unattributed_ms"] == 41
    assert sum(result["overlap_by_stage_ms"].values()) == 239
    assert (
        result["covered_union_ms"] + result["unattributed_ms"]
        == result["window"]["elapsed_ms"]
    )


@pytest.mark.parametrize("identifier", [HTTP, EXTERNAL, INTERNAL])
def test_all_explicit_identity_namespaces_resolve_same_request(analyzer, identifier):
    result = analyzer.analyze_request(request_records(), identifier, healthy())
    assert result["status"] == "COMPLETE"
    assert result["window"]["elapsed_ms"] == 220


def test_unrelated_requests_and_prior_process_are_not_joined(analyzer):
    records = request_records() + [
        event("phase_prefill", 100, 320, pid=20, trace="old-engine", request_id=OTHER),
        event(
            "worker_execute",
            100,
            320,
            pid=20,
            trace="engine",
            sequence=4,
            request_id=OTHER,
            active={"request_id": INTERNAL},
        ),
    ]
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["covered_union_ms"] == 179
    assert len(result["timeline"]) == len(request_records())


def test_clips_boundary_spans_and_does_not_use_envelope_as_attribution(analyzer):
    records = request_records()
    records += [
        event(
            "worker_sample",
            80,
            400,
            pid=20,
            trace="engine",
            sequence=4,
            request_id=INTERNAL,
        ),
        event("http_request", 100, 350, pid=30, trace="unrelated", request_id=OTHER),
    ]
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["covered_union_ms"] == 220
    assert result["unattributed_ms"] == 0
    assert result["overlap_by_stage_ms"]["worker_sample"] == 220


@pytest.mark.parametrize(
    ("removed", "reason"),
    [
        ("http_request", "missing_http_start"),
        ("internal_id_bridge", "missing_internal_id_bridge"),
        ("first_engine_output", "missing_engine_first_output"),
        ("http_end", "missing_http_end"),
    ],
)
def test_missing_boundary_cannot_claim_complete(analyzer, removed, reason):
    records = [row for row in request_records() if row["stage"] != removed]
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["status"] == "INCOMPLETE"
    assert reason in result["incomplete_reasons"]


def test_protocol_only_stream_does_not_become_engine_ttft(analyzer):
    records = [
        row
        for row in request_records()
        if row["stage"] not in {"first_engine_output", "first_api_content"}
    ]
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["status"] == "INCOMPLETE"
    assert result["latencies"]["http_first_body_ms"] == 40
    assert result["latencies"]["engine_first_output_ms"] is None
    assert result["window"]["elapsed_ms"] is None


def test_nonstream_response_uses_engine_boundary_without_inventing_content_ttft(
    analyzer,
):
    records = [row for row in request_records() if row["stage"] != "first_api_content"]
    records[-1]["sequence"] = 9
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["status"] == "COMPLETE"
    assert result["window"]["end"] == "engine_first_output"
    assert result["window"]["elapsed_ms"] == 200


@pytest.mark.parametrize(
    "damage",
    [
        "drop",
        "write_error",
        "missing_health",
        "sequence_gap",
        "invalid",
        "cancelled",
        "phase_failed",
        "conflict",
    ],
)
def test_incomplete_capture_is_explicit(analyzer, damage):
    records, health, invalid = request_records(), healthy(), 0
    if damage == "drop":
        health[0]["dropped"] = 1
    elif damage == "write_error":
        health[1]["write_errors"] = 1
    elif damage == "missing_health":
        health.pop()
    elif damage == "sequence_gap":
        records = [row for row in records if row["stage"] != "engine_submit"]
    elif damage == "invalid":
        invalid = 1
    elif damage == "cancelled":
        records[-1]["success"] = False
    elif damage == "phase_failed":
        records[2]["success"] = False
    else:
        records.append({**records[2], "end_ns": 131_000_000})
    result = analyzer.analyze_request(records, HTTP, health, invalid_records=invalid)
    assert result["status"] == "INCOMPLETE"
    assert result["incomplete_reasons"]


def test_startup_work_outside_request_not_charged_and_duplicate_files_are_safe(
    analyzer,
):
    records = request_records()
    startup = event("model_warmup", 0, 90, pid=20, trace="engine", sequence=4)
    result = analyzer.analyze_request(records + records + [startup], HTTP, healthy())
    assert result["status"] == "COMPLETE"
    assert result["covered_union_ms"] == 179
    assert "model_warmup" not in result["overlap_by_stage_ms"]


def test_corrupt_numeric_record_and_reversed_clock_are_not_accepted(analyzer):
    records = request_records()
    records[2]["end_ns"] = -1
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["status"] == "INCOMPLETE"
    assert result["coverage"]["invalid_records"] == 1
    assert "invalid_records" in result["incomplete_reasons"]


def test_ambiguous_reused_request_identity_is_not_merged_as_complete(analyzer):
    records = request_records() + [
        event("http_request", 101, pid=11, trace="another", http_request_id=HTTP)
    ]
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["status"] == "INCOMPLETE"
    assert result["window"]["elapsed_ms"] is None


def test_request_cli_needs_no_round_log_and_keeps_invalid_line_visible(
    analyzer, tmp_path
):
    capture, health, output = (
        tmp_path / name for name in ("cache.jsonl", "health.json", "output.json")
    )
    capture.write_text(
        "".join(json.dumps(row) + "\n" for row in request_records()) + "{broken\n"
    )
    health.write_text(json.dumps(healthy()))
    # CLI accepts separate per-process snapshots as produced by the recorders.
    health_paths = []
    for index, row in enumerate(healthy()):
        path = tmp_path / f"health-{index}.json"
        path.write_text(json.dumps(row))
        health_paths += ["--health", str(path)]
    analyzer.main(
        [
            "--request-id",
            HTTP,
            "--cache-log",
            str(capture),
            *health_paths,
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text())
    assert result["status"] == "INCOMPLETE"
    assert result["coverage"]["invalid_records"] == 1


def test_request_analysis_outputs_no_freeform_input_values(analyzer):
    records = request_records()
    records[2].update(
        prompt="SECRET CONTENT",
        token_ids=[654321],
        path="SECRET PATH",
        exception="SECRET ERROR",
    )
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert "SECRET" not in json.dumps(result)
    assert "654321" not in json.dumps(result)


@pytest.mark.parametrize("clock", [None, "f" * 64])
def test_missing_or_different_boot_clocks_never_subtract_cross_process_time(
    analyzer, clock
):
    records = request_records()
    records[7]["clock_id"] = clock
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["status"] == "INCOMPLETE"
    assert not result["coverage"]["clock_domain_verified"]
    assert result["window"]["elapsed_ms"] is None
    assert result["latencies"]["engine_first_output_ms"] is None


def test_lifecycle_start_cannot_follow_an_event(analyzer):
    health = healthy()
    health[0]["started_ns"] = 101_000_000
    result = analyzer.analyze_request(request_records(), HTTP, health)
    assert result["status"] == "INCOMPLETE"
    assert "event_precedes_lifecycle" in result["incomplete_reasons"]


def test_unclosed_phase_is_explicit_and_never_fabricated_as_observed_time(analyzer):
    records = request_records()
    records += [
        event(
            "phase_enter_cache_lookup",
            190,
            pid=20,
            trace="engine",
            sequence=4,
            request_id=INTERNAL,
        )
    ]
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["status"] == "INCOMPLETE"
    assert result["coverage"]["unterminated_phases"] == ["phase_cache_lookup"]
    assert "phase_cache_lookup" not in result["overlap_by_stage_ms"]
    assert result["covered_union_ms"] == 179


def test_phase_closing_after_first_content_is_clipped_and_complete(analyzer):
    records = request_records()
    records += [
        event(
            "phase_enter_generate",
            295,
            pid=20,
            trace="engine",
            sequence=4,
            request_id=INTERNAL,
        ),
        event(
            "phase_generate",
            295,
            340,
            pid=20,
            trace="engine",
            sequence=5,
            request_id=INTERNAL,
        ),
    ]
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["status"] == "COMPLETE"
    assert result["overlap_by_stage_ms"]["phase_generate"] == 25
    assert result["covered_union_ms"] == 204


def test_scheduler_first_work_hook_coverage_is_required_when_admitted(analyzer):
    records = request_records() + [
        event(
            "scheduler_admitted",
            190,
            pid=20,
            trace="engine",
            sequence=4,
            request_id=INTERNAL,
        ),
    ]
    health = healthy()
    result = analyzer.analyze_request(records, HTTP, health)
    assert result["status"] == "INCOMPLETE"
    assert "worker_first_work_hooks_unconfirmed" in result["incomplete_reasons"]
    health[1]["worker_first_work_hooks"] = True
    assert analyzer.analyze_request(records, HTTP, health)["status"] == "COMPLETE"


def test_invalid_clock_types_fail_closed_instead_of_breaking_analysis(analyzer):
    records, health = request_records(), healthy()
    records[7]["clock_id"] = {"untrusted": "clock"}
    health[1]["clock_id"] = ["untrusted"]
    result = analyzer.analyze_request(records, HTTP, health)
    assert result["status"] == "INCOMPLETE"
    assert result["window"]["elapsed_ms"] is None
    assert result["coverage"]["invalid_records"] == 1
