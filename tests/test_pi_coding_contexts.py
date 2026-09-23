"""CPU-only contract checks for the coding context-length benchmark."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/radiance-public/benchmark_pi_coding_contexts.py"
SPEC = importlib.util.spec_from_file_location("pi_coding_contexts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_context_table_order_is_stable():
    assert MODULE.CONTEXTS == ("0K", "60K", "200K")
    assert [MODULE.CONTEXT_TOKEN_COUNTS[name] for name in MODULE.CONTEXTS] == [
        0,
        60_000,
        200_000,
    ]


def test_empty_arm_does_not_accept_a_fixture(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"prefix": []}))
    with pytest.raises(ValueError, match="0K arm"):
        MODULE._load_prefix(fixture, 0)
    assert MODULE._load_prefix(None, 0) == ([], None)


def test_nonempty_fixture_requires_exact_integer_prefix(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"prefix": [1, 2]}))
    with pytest.raises(ValueError, match="exactly 3"):
        MODULE._load_prefix(fixture, 3)

    fixture.write_text(json.dumps({"prefix": [1, True, 3]}))
    with pytest.raises(ValueError, match="exactly 3"):
        MODULE._load_prefix(fixture, 3)


def test_valid_fixture_returns_only_digest_metadata(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"prefix": [11, 12, 13]}))
    prefix, digest = MODULE._load_prefix(fixture, 3)
    assert prefix == [11, 12, 13]
    assert digest == MODULE._sha256_bytes(fixture.read_bytes())


def _round_event(identity, round_number, observed_at_ms, round_ms):
    return {
        "schema": MODULE.ROUND_LOG_SCHEMA,
        "pid": 123,
        "observed_at_ms": observed_at_ms,
        "chat_id": identity["id"],
        "generation": identity["generation"],
        "request_id": "request",
        "round": round_number,
        "round_ms": round_ms,
        "draft_tokens": 7,
        "accepted_tokens": 4,
        "acceptance_rate": 4 / 7,
    }


def test_round_capture_retains_every_new_event_including_unmeasured_first_round(
    tmp_path,
):
    path = tmp_path / "rounds.jsonl"
    identity = {"id": "a" * 64, "generation": "b" * 64}
    old = _round_event(identity, 99, 900, 40.0)
    first = _round_event(identity, 1, 1_000, None)
    first["draft_tokens"] = 0
    first["accepted_tokens"] = 0
    second = _round_event(identity, 2, 1_050, 41.25)
    third = _round_event(identity, 3, 1_100, 41.5)
    path.with_name("rounds.jsonl.1").write_text(json.dumps(old) + "\n")
    path.write_text("\n".join(json.dumps(row) for row in (first, second, third)) + "\n")

    capture = MODULE._round_capture(
        path=path,
        identity=identity,
        baseline_keys={MODULE._round_event_key(old)},
        started_at_ms=1_000,
        ended_at_ms=1_200,
        expected_rounds=2,
    )

    assert capture["status"] == "captured"
    assert capture["record_count"] == 3
    assert capture["speculative_round_count"] == 2
    assert capture["expected_rounds_metric"] == "vllm:spec_decode_num_drafts_total"
    assert capture["measured_round_count"] == 2
    assert capture["unmeasured_round_count"] == 1
    assert capture["round_numbers"] == [1, 2, 3]
    assert capture["missing_round_numbers"] == []
    assert [row["round"] for row in capture["records"]] == [1, 2, 3]
    assert capture["records"][0]["round_ms"] is None


def test_round_capture_reports_missing_rows_instead_of_hiding_them(tmp_path):
    path = tmp_path / "rounds.jsonl"
    identity = {"id": "a" * 64, "generation": "b" * 64}
    rows = [
        _round_event(identity, 1, 1_000, 40.0),
        _round_event(identity, 3, 1_100, 42.0),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    capture = MODULE._round_capture(
        path=path,
        identity=identity,
        baseline_keys=set(),
        started_at_ms=1_000,
        ended_at_ms=1_200,
        expected_rounds=3,
    )

    assert capture["status"] == "incomplete"
    assert capture["record_count"] == 2
    assert capture["missing_round_numbers"] == [2]


def test_round_capture_marks_missing_feed_as_unavailable(tmp_path):
    identity = {"id": "a" * 64, "generation": "b" * 64}
    capture = MODULE._round_capture(
        path=tmp_path / "missing.jsonl",
        identity=identity,
        baseline_keys=set(),
        started_at_ms=1_000,
        ended_at_ms=1_200,
        expected_rounds=1,
    )
    assert capture["status"] == "unavailable"
    assert capture["read_error"] == "round log is not present"


def test_missing_first_event_fails_even_when_all_speculative_events_are_present(
    tmp_path,
):
    path = tmp_path / "rounds.jsonl"
    identity = {"id": "a" * 64, "generation": "b" * 64}
    rows = [
        _round_event(identity, 2, 1_050, 41.0),
        _round_event(identity, 3, 1_100, 41.5),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    capture = MODULE._round_capture(
        path=path,
        identity=identity,
        baseline_keys=set(),
        started_at_ms=1_000,
        ended_at_ms=1_200,
        expected_rounds=2,
    )
    assert capture["speculative_round_count"] == 2
    assert capture["status"] == "incomplete"
    assert capture["missing_round_numbers"] == [1]


def test_round_histogram_accounts_for_every_measured_and_unmeasured_record():
    records = [
        {"round_ms": None},
        {"round_ms": 34.99},
        {"round_ms": 40.25},
        {"round_ms": 500.0},
    ]
    histogram = MODULE._round_histogram(records)

    assert histogram["measured_round_count"] == 3
    assert histogram["unmeasured_round_count"] == 1
    assert sum(row["count"] for row in histogram["bins"]) == 3
    assert histogram["bins"][0]["count"] == 1
    assert histogram["bins"][4]["count"] == 1
    assert histogram["bins"][-1]["count"] == 1
    assert histogram["mean_ms"] == pytest.approx((34.99 + 40.25 + 500.0) / 3)
    assert histogram["median_ms"] == pytest.approx(40.25)
