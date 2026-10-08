"""Causal prefix coverage must not borrow success from a complete timing trace."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import test_first_output_telemetry_analysis as timeline_tests
from test_first_output_telemetry_analysis import (
    EXTERNAL,
    HTTP,
    INTERNAL,
    OTHER,
    event,
    request_records,
)
from test_first_output_telemetry_analysis import (
    healthy as timeline_health,
)


@pytest.fixture
def analyzer():
    return timeline_tests.analyzer.__wrapped__()


def healthy():
    rows = timeline_health()
    rows[0].update(
        prefix_lineage_hooks={
            "api_request": True,
            "engine_tokens": True,
            "serialized_delta": True,
            "input_processor": True,
        },
        source_sha256={"prefix_lineage": "f" * 64, "prefix_runtime": "f" * 64},
    )
    return rows


COMPARISONS = (
    "raw_to_reencoded",
    "output_to_message",
    "template_normalization",
    "prompt_prefix",
    "input_processor",
)


def prefix_request(*, reason="prefix_hash_changed", outcome="rejected"):
    return request_records() + [
        event(
            "response_end_lookup",
            190,
            sequence=11,
            request_id=INTERNAL,
            cache_source="gpu_endpoint",
            reason=reason,
            outcome=outcome,
            endpoint_tokens=97_137,
        )
    ]


def comparison_records(*, unequal="prompt_prefix"):
    return [
        event(
            "prefix_lineage",
            180,
            sequence=12 + index,
            request_id=INTERNAL,
            comparison=name,
            diagnostic_status="complete",
            equal=name not in {unequal, "prompt_prefix"},
            compared_tokens=97_137,
            previous_tokens=97_137,
            current_tokens=97_137,
            **(
                {"first_difference": 87_344}
                if name in {unequal, "prompt_prefix"}
                else {}
            ),
        )
        for index, name in enumerate(COMPARISONS)
    ]


def test_real_rejection_can_have_complete_timing_but_missing_causal_evidence(analyzer):
    result = analyzer.analyze_request(prefix_request(), HTTP, healthy())
    assert result["status"] == "COMPLETE"
    assert result["timing_complete"] is True
    assert result["prefix_diagnosis_complete"] is False
    diagnosis = result["prefix_diagnosis"]
    assert diagnosis["status"] == "INCOMPLETE"
    assert diagnosis["rejection_reasons"] == ["prefix_hash_changed"]
    assert diagnosis["required_comparisons"] == list(COMPARISONS)
    assert diagnosis["missing_evidence"] == [
        f"missing:{name}" for name in sorted(COMPARISONS)
    ]
    assert diagnosis["observed_causes"] == []


@pytest.mark.parametrize(
    ("comparison", "cause"),
    [
        ("raw_to_reencoded", "token_roundtrip_changed"),
        ("output_to_message", "assistant_message_changed"),
        ("template_normalization", "template_normalization_changed"),
        ("prompt_prefix", "rendered_prompt_prefix_changed"),
    ],
)
def test_only_observed_failed_transformation_is_identified(analyzer, comparison, cause):
    result = analyzer.analyze_request(
        prefix_request() + comparison_records(unequal=comparison), HTTP, healthy()
    )
    assert result["timing_complete"] is True
    assert result["prefix_diagnosis_complete"] is True
    assert result["prefix_diagnosis"]["status"] == "COMPLETE"
    assert result["prefix_diagnosis"]["observed_causes"] == sorted(
        {cause, "rendered_prompt_prefix_changed"}
    )
    assert result["prefix_diagnosis"]["missing_evidence"] == []


@pytest.mark.parametrize(
    "reason", ["partial_prefix_changed", "prefix_identity_changed"]
)
def test_all_identity_rejections_require_lineage(analyzer, reason):
    result = analyzer.analyze_request(prefix_request(reason=reason), HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is False
    assert result["prefix_diagnosis"]["rejection_reasons"] == [reason]


def test_observed_cache_hit_does_not_require_rejected_prefix_comparisons(analyzer):
    result = analyzer.analyze_request(
        prefix_request(reason="matching_endpoint", outcome="hit"), HTTP, healthy()
    )
    assert result["prefix_diagnosis_complete"] is True
    assert result["prefix_diagnosis"]["status"] == "NOT_APPLICABLE"
    assert result["prefix_diagnosis"]["required_comparisons"] == []


def test_no_lookup_record_is_unavailable_not_an_invented_hit(analyzer):
    result = analyzer.analyze_request(request_records(), HTTP, healthy())
    assert result["status"] == "COMPLETE"
    assert result["prefix_diagnosis_complete"] is False
    assert result["prefix_diagnosis"]["status"] == "UNAVAILABLE"
    assert result["prefix_diagnosis"]["missing_evidence"] == [
        "missing_response_end_lookup"
    ]


@pytest.mark.parametrize("missing", COMPARISONS)
def test_each_required_comparison_is_enforced(analyzer, missing):
    records = prefix_request() + [
        row for row in comparison_records() if row["comparison"] != missing
    ]
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is False
    assert f"missing:{missing}" in result["prefix_diagnosis"]["missing_evidence"]


@pytest.mark.parametrize(
    "reason", ["unsupported", "evicted", "restart", "dropped", "failed"]
)
def test_source_failure_is_explicit_and_does_not_become_no_change(analyzer, reason):
    rows = comparison_records()
    rows[0] = {
        **rows[0],
        "diagnostic_status": "unavailable",
        "reason": reason,
        "equal": False,
        "first_difference": 0,
    }
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is False
    assert (
        f"{reason}:raw_to_reencoded" in result["prefix_diagnosis"]["missing_evidence"]
    )
    assert (
        "token_roundtrip_changed" not in result["prefix_diagnosis"]["observed_causes"]
    )


@pytest.mark.parametrize(
    "damage",
    [
        {"comparison": ["prompt_prefix"]},
        {"diagnostic_status": ["complete"]},
        {"compared_tokens": True},
        {"previous_tokens": -1},
        {"current_tokens": None},
        {"equal": 1},
        {"first_difference": -1},
        {"first_difference": 97_138},
    ],
)
def test_malformed_lineage_fails_closed_without_breaking_analysis(analyzer, damage):
    rows = comparison_records()
    rows[3] = {**rows[3], **damage}
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is False
    assert result["prefix_diagnosis"]["missing_evidence"]


def test_conflicting_producer_observations_cannot_certify_the_comparison(analyzer):
    rows = comparison_records()
    conflicting = {**rows[3], "sequence": 17, "equal": True}
    conflicting.pop("first_difference")
    result = analyzer.analyze_request(
        prefix_request() + rows + [conflicting], HTTP, healthy()
    )
    assert result["prefix_diagnosis_complete"] is False
    assert (
        "conflicting_comparison:prompt_prefix"
        in result["prefix_diagnosis"]["missing_evidence"]
    )
    assert result["prefix_diagnosis"]["observed_causes"] == []


@pytest.mark.parametrize("outcome", [None, [], "hit", "unknown"])
def test_bad_lookup_does_not_hide_a_recorded_identity_rejection(analyzer, outcome):
    result = analyzer.analyze_request(prefix_request(outcome=outcome), HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is False
    assert result["prefix_diagnosis"]["rejection_reasons"] == ["prefix_hash_changed"]
    assert "missing:prompt_prefix" in result["prefix_diagnosis"]["missing_evidence"]


@pytest.mark.parametrize("damage", ["drop", "missing_health", "sequence_gap"])
def test_recorder_evidence_loss_invalidates_prefix_completeness(analyzer, damage):
    records, health = prefix_request() + comparison_records(), healthy()
    expected = "recorder_loss"
    if damage == "drop":
        health[0]["dropped"] = 1
    elif damage == "missing_health":
        health.pop(0)
        expected = "missing_lifecycle_health"
    else:
        records = [row for row in records if row["stage"] != "engine_submit"]
        expected = "sequence_gaps"
    result = analyzer.analyze_request(records, HTTP, health)
    assert result["prefix_diagnosis_complete"] is False
    assert expected in result["prefix_diagnosis"]["missing_evidence"]


def test_another_requests_good_lineage_cannot_fill_missing_evidence(analyzer):
    other = [{**row, "request_id": OTHER} for row in comparison_records()]
    result = analyzer.analyze_request(prefix_request() + other, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is False
    assert result["prefix_diagnosis"]["comparisons"] == []


def test_observer_spans_do_not_claim_to_explain_prefill_time(analyzer):
    records = prefix_request() + comparison_records()
    for row in records:
        if row["stage"] == "prefix_lineage":
            row["end_ns"] = 320_000_000
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert "prefix_lineage" not in result["overlap_by_stage_ms"]
    assert result["covered_union_ms"] == 179


def test_lineage_report_never_copies_payload_or_arbitrary_reason(analyzer):
    rows = comparison_records()
    rows[0] = {
        **rows[0],
        "diagnostic_status": "unavailable",
        "reason": "SECRET TEXT",
        "token_ids": [654321],
        "text": "SECRET TEXT",
        "prefix_hash": "SECRET HASH",
    }
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    serialized = json.dumps(result)
    assert "SECRET" not in serialized
    assert "654321" not in serialized


def test_input_processor_change_is_separately_visible(analyzer):
    rows = comparison_records()
    rows[-1] = {**rows[-1], "equal": False, "first_difference": 9}
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is True
    assert result["prefix_diagnosis"]["observed_causes"] == [
        "input_processor_changed",
        "rendered_prompt_prefix_changed",
    ]


def test_template_operation_observations_do_not_replace_the_aggregate(analyzer):
    rows = comparison_records()
    rows[2] = {
        **rows[2],
        "unit": "characters",
        "comparison_complete": True,
        "equal": False,
        "difference_position": 30,
        "previous_characters": 100,
        "current_characters": 98,
        "compared_characters": 100,
    }
    for name in ("compared_tokens", "previous_tokens", "current_tokens"):
        rows[2].pop(name)
    rows.append(
        {
            **rows[2],
            "sequence": 17,
            "comparison_complete": False,
            "operation": "reasoning_trim",
            "equal": True,
        }
    )
    rows[-1].pop("difference_position")
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is True
    normalized = next(
        row
        for row in result["prefix_diagnosis"]["comparisons"]
        if row["comparison"] == "template_normalization"
    )
    assert normalized["unit"] == "characters"
    assert normalized["first_difference"] == 30
    assert "previous_tokens" not in normalized
    assert result["prefix_diagnosis"]["template_operations"][0]["equal"] is True


@pytest.mark.parametrize("removed", [False, True])
def test_template_operation_comparison_scope_survives_analysis(analyzer, removed):
    rows = comparison_records()
    rows.append(
        event(
            "prefix_lineage",
            180,
            sequence=17,
            request_id=INTERNAL,
            comparison="template_normalization",
            diagnostic_status="complete",
            operation="assistant_delimiters",
            scope="assistant_message_without_terminator",
            recognized_stop_marker_removed=removed,
            unit="characters",
            equal=False,
            previous_characters=101,
            current_characters=100,
            compared_characters=100,
            difference_position=100,
            text="PRIVATE CONTENT",
            token_ids=[99999],
        )
    )
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    diagnosis = result["prefix_diagnosis"]
    assert result["prefix_diagnosis_complete"] is True
    assert diagnosis["template_operations"] == [
        {
            "comparison": "template_normalization",
            "status": "complete",
            "equal": False,
            "unit": "characters",
            "previous_characters": 101,
            "current_characters": 100,
            "compared_characters": 100,
            "first_difference": 100,
            "operation": "assistant_delimiters",
            "scope": "assistant_message_without_terminator",
            "recognized_stop_marker_removed": removed,
        }
    ]
    assert diagnosis["observed_causes"] == ["rendered_prompt_prefix_changed"]
    assert "PRIVATE CONTENT" not in json.dumps(result)
    assert "token_ids" not in json.dumps(result)


@pytest.mark.parametrize(
    ("field", "value", "gap"),
    [
        ("scope", "PRIVATE CONTENT", "invalid_template_scope"),
        ("scope", 1, "invalid_template_scope"),
        ("scope", ["PRIVATE CONTENT"], "invalid_template_scope"),
        (
            "recognized_stop_marker_removed",
            "PRIVATE CONTENT",
            "invalid_template_stop_marker",
        ),
        ("recognized_stop_marker_removed", 0, "invalid_template_stop_marker"),
        ("recognized_stop_marker_removed", None, "invalid_template_stop_marker"),
    ],
)
def test_invalid_template_operation_metadata_is_not_disclosed_or_certified(
    analyzer, field, value, gap
):
    row = {
        **comparison_records()[2],
        "sequence": 17,
        "operation": "assistant_delimiters",
        "unit": "characters",
        "equal": True,
        "previous_characters": 100,
        "current_characters": 100,
        "compared_characters": 100,
        field: value,
    }
    result = analyzer.analyze_request(
        prefix_request() + comparison_records() + [row], HTTP, healthy()
    )
    diagnosis = result["prefix_diagnosis"]
    assert result["timing_complete"] is True
    assert result["prefix_diagnosis_complete"] is False
    assert gap in diagnosis["missing_evidence"]
    assert diagnosis["template_operations"][0]["status"] == "invalid"
    assert field not in diagnosis["template_operations"][0]
    assert "PRIVATE CONTENT" not in json.dumps(result)


def test_template_subevents_alone_do_not_certify_the_final_render(analyzer):
    rows = comparison_records()
    rows[2] = {**rows[2], "operation": "reasoning_trim", "comparison_complete": False}
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is False
    assert (
        "missing:template_normalization"
        in result["prefix_diagnosis"]["missing_evidence"]
    )


def test_message_field_identity_difference_does_not_invent_a_token_offset(analyzer):
    rows = comparison_records(unequal="output_to_message")
    rows[1] = {
        **rows[1],
        "unit": "message_fields",
        "difference_position_available": False,
        "previous_fields": 3,
        "current_fields": 3,
        "compared_fields": 3,
    }
    for name in (
        "compared_tokens",
        "previous_tokens",
        "current_tokens",
        "first_difference",
    ):
        rows[1].pop(name)
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is True
    compared = next(
        row
        for row in result["prefix_diagnosis"]["comparisons"]
        if row["comparison"] == "output_to_message"
    )
    assert compared["unit"] == "message_fields"
    assert compared["difference_position_available"] is False
    assert "first_difference" not in compared


def test_emitted_pending_suffix_does_not_explain_processed_endpoint_rejection(analyzer):
    rows = comparison_records()
    rows[3]["previous_tokens"] = rows[3]["compared_tokens"] = 97_138
    rows[3]["current_tokens"] = 97_158
    rows[3]["first_difference"] = 97_137
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    diagnosis = result["prefix_diagnosis"]
    assert result["prefix_diagnosis_complete"] is False
    assert "emitted_unprocessed_boundary" in diagnosis["missing_evidence"]
    assert diagnosis["endpoint_scope"][0]["covered"] is True
    assert diagnosis["endpoint_scope"][0]["difference_within_endpoint"] is False
    assert diagnosis["observed_causes"] == []


@pytest.mark.parametrize("damage", ["missing", "insufficient"])
def test_missing_actual_endpoint_scope_is_explicit(analyzer, damage):
    records = prefix_request()
    rows = comparison_records()
    if damage == "missing":
        records[-1].pop("endpoint_tokens")
    else:
        rows[3]["previous_tokens"] = rows[3]["compared_tokens"] = 97_000
    result = analyzer.analyze_request(records + rows, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is False
    assert "exact_endpoint_missing" in result["prefix_diagnosis"]["missing_evidence"]
    assert result["prefix_diagnosis"]["observed_causes"] == []


def test_actual_input_change_can_explain_rejection_when_pending_suffix_does_not(
    analyzer,
):
    rows = comparison_records()
    rows[3]["previous_tokens"] = rows[3]["compared_tokens"] = 97_138
    rows[3]["current_tokens"] = 97_158
    rows[3]["first_difference"] = 97_137
    rows[4].update(equal=False, first_difference=10)
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is True
    assert result["prefix_diagnosis"]["observed_causes"] == ["input_processor_changed"]
    assert result["prefix_diagnosis"]["endpoint_scope"][0]["first_difference"] == 10


def test_equal_token_prefixes_do_not_explain_a_changed_hash(analyzer):
    rows = comparison_records()
    rows[3]["equal"] = True
    rows[3].pop("first_difference")
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is False
    assert (
        "prefix_identity_unexplained" in result["prefix_diagnosis"]["missing_evidence"]
    )
    assert result["prefix_diagnosis"]["observed_causes"] == []


def test_requirements_inventory_points_to_implemented_negative_controls(analyzer):
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads(
        (root / "tests/prefix_lineage_requirements.json").read_text()
    )
    assert inventory["schema"] == "urn:coherence:focused-verification:v1"
    assert inventory["required_rejection_comparisons"] == list(COMPARISONS)
    assert len({row["id"] for row in inventory["requirements"]}) == len(
        inventory["requirements"]
    )
    for requirement in inventory["requirements"]:
        assert requirement["boundary"] and requirement["contract"]
        assert requirement["checks"] and requirement["negative_controls"]
        for check in requirement["checks"]:
            path = root / check["file"]
            source = path.read_text()
            if path.suffix == ".py":
                names = {
                    node.name
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef)
                }
                assert check["case"] in names
            else:
                assert f"test({json.dumps(check['case'])}," in source
    assert inventory["required_rejection_comparisons"] == list(
        analyzer.PREFIX_COMPARISONS
    )


def test_actual_core_records_keep_roundtrip_and_template_changes_distinct(analyzer):
    from test_radiance_prefix_lineage import Tokenizer, completed, next_request

    from qwen_r9700_lab.radiance_prefix_lineage import PrefixLineageObserver

    observed, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(observed.append)
    completed(observer, tokenizer, output=[999, 32], content="a ")
    observed.clear()
    handle = observer.begin(next_request(content="a "), tokenizer, {})
    observer.template_operation(
        handle, "a ", "a", "content_trim", observer.assistant_index(handle)
    )
    observer.template_complete(handle, True)
    observer.rendered(handle, [80, 97, 78])
    observer.input_processor(handle, [80, 97, 78])
    records = prefix_request()
    records[-1]["endpoint_tokens"] = 3
    for sequence, row in enumerate(observed, start=12):
        records.append(
            event(
                "prefix_lineage",
                180,
                sequence=sequence,
                request_id=INTERNAL,
                **{name: value for name, value in row.items() if name != "stage"},
            )
        )
    result = analyzer.analyze_request(records, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is True
    assert result["prefix_diagnosis"]["observed_causes"] == [
        "rendered_prompt_prefix_changed",
        "template_normalization_changed",
        "token_roundtrip_changed",
    ]
    assert result["prefix_diagnosis"]["endpoint_scope"][0]["first_difference"] == 1


def pi_records(*, response_id=EXTERNAL):
    stages = (
        "context_before_conversion",
        "provider_converted",
        "provider_sdk_input",
        "provider_wire",
        "provider_response_identity",
        "provider_response_assembled",
    )
    result = []
    for index, stage in enumerate(stages):
        row = {
            "schema": "urn:coherence:pi-prefix-lineage:v1",
            "stage": stage,
            "timestamp_ms": 9_999_999_000 + index,
            "pid": 700,
            "trace_id": "pi-lifecycle",
            "ordinal": 4,
            "source_id": "f" * 64,
            "request_id_sha256": response_id if index >= 4 else None,
            "hook_mask": 63,
            "recorder": {"dropped": 0, "errors": 0},
        }
        if stage in {"provider_converted", "provider_sdk_input", "provider_wire"}:
            row.update(equal=True, coverage_complete=True)
        if stage == "provider_sdk_input":
            row.update(
                previous_available=True, previous_output_equal=True, history_equal=True
            )
        if stage == "provider_response_assembled":
            row.update(success=True, coverage_complete=True)
        result.append(row)
    return result


def test_backend_diagnosis_does_not_claim_the_missing_client_route_is_complete(
    analyzer,
):
    result = analyzer.analyze_request(
        prefix_request() + comparison_records(), HTTP, healthy()
    )
    assert result["prefix_diagnosis_complete"] is True
    assert result["pi_prefix_diagnosis"]["status"] == "UNAVAILABLE"
    assert result["production_path_diagnosis_complete"] is False


def test_pi_early_boundaries_join_through_observed_response_identity_not_clocks(
    analyzer,
):
    result = analyzer.analyze_request(
        prefix_request() + comparison_records(),
        HTTP,
        healthy(),
        pi_lineage=pi_records(),
    )
    assert result["production_path_diagnosis_complete"] is True
    diagnosis = result["pi_prefix_diagnosis"]
    assert diagnosis["join"] == "explicit_sse_response_id_sha256"
    assert len(diagnosis["observations"]) == 6
    assert "context_before_conversion" in {
        row["stage"] for row in diagnosis["observations"]
    }
    assert "timestamp_ms" not in json.dumps(diagnosis)
    assert result["latencies"]["api_first_content_ms"] == 220


def test_chat_identity_and_matching_timestamp_cannot_replace_pi_response_bridge(
    analyzer,
):
    rows = pi_records(response_id=OTHER)
    for row in rows:
        row.update(chat_id=INTERNAL, timestamp_ms=180)
    result = analyzer.analyze_request(
        prefix_request() + comparison_records(), HTTP, healthy(), pi_lineage=rows
    )
    assert result["production_path_diagnosis_complete"] is False
    assert (
        "missing_pi_request_bridge" in result["pi_prefix_diagnosis"]["missing_evidence"]
    )


@pytest.mark.parametrize(
    "missing",
    [
        "context_before_conversion",
        "provider_converted",
        "provider_sdk_input",
        "provider_wire",
        "provider_response_identity",
        "provider_response_assembled",
    ],
)
def test_every_client_boundary_is_required_in_the_report(analyzer, missing):
    rows = [row for row in pi_records() if row["stage"] != missing]
    result = analyzer.analyze_request(
        prefix_request() + comparison_records(), HTTP, healthy(), pi_lineage=rows
    )
    assert result["production_path_diagnosis_complete"] is False
    assert (
        f"missing_pi_stage:{missing}"
        in result["pi_prefix_diagnosis"]["missing_evidence"]
    )


@pytest.mark.parametrize(
    "damage, gap",
    [
        ({"unsupported": True}, "pi_unsupported"),
        ({"truncated": True}, "pi_truncated"),
        ({"observer_failed": True}, "pi_observer_failed"),
        ({"recorder": {"dropped": 1, "errors": 0}}, "pi_recorder_loss"),
        ({"recorder": None}, "missing_pi_recorder_health"),
        ({"source_id": None}, "missing_pi_source_identity"),
        ({"previous_available": False}, "previous_client_output_unavailable"),
    ],
)
def test_pi_unsupported_lost_or_missing_evidence_is_explicit(analyzer, damage, gap):
    rows = pi_records()
    rows[2].update(damage)
    result = analyzer.analyze_request(
        prefix_request() + comparison_records(), HTTP, healthy(), pi_lineage=rows
    )
    assert result["production_path_diagnosis_complete"] is False
    assert gap in result["pi_prefix_diagnosis"]["missing_evidence"]


def test_client_modifications_are_observed_separately_from_backend_changes(analyzer):
    rows = pi_records()
    rows[1].update(
        equal=False, first_changed_message=2, first_change_reasoning_equal=False
    )
    rows[2].update(config_changed=True, previous_output_equal=False)
    result = analyzer.analyze_request(
        prefix_request() + comparison_records(), HTTP, healthy(), pi_lineage=rows
    )
    assert result["production_path_diagnosis_complete"] is True
    assert result["pi_prefix_diagnosis"]["observed_changes"] == [
        "client_previous_output_changed",
        "config_changed",
        "provider_conversion_changed",
    ]


def test_multiple_client_lifecycles_cannot_claim_the_same_request(analyzer):
    rows = pi_records() + [{**row, "pid": 701} for row in pi_records()]
    result = analyzer.analyze_request(
        prefix_request() + comparison_records(), HTTP, healthy(), pi_lineage=rows
    )
    assert result["production_path_diagnosis_complete"] is False
    assert (
        "ambiguous_pi_request_bridge"
        in result["pi_prefix_diagnosis"]["missing_evidence"]
    )


def test_pi_report_drops_payloads_and_free_form_labels(analyzer):
    rows = pi_records()
    rows[2].update(
        text="SECRET TEXT",
        token_ids=[654321],
        error="SECRET ERROR",
        reason="SECRET WHY",
    )
    result = analyzer.analyze_request(
        prefix_request() + comparison_records(), HTTP, healthy(), pi_lineage=rows
    )
    assert "SECRET" not in json.dumps(result)
    assert "654321" not in json.dumps(result)


def test_lazy_roundtrip_can_be_inapplicable_when_input_processing_is_the_change(
    analyzer,
):
    rows = comparison_records()
    rows[0] = {
        **rows[0],
        "diagnostic_status": "not_applicable",
        "reason": "matching_prefix",
    }
    rows[3]["equal"] = True
    rows[3].pop("first_difference")
    rows[4].update(equal=False, first_difference=10)
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    assert result["prefix_diagnosis_complete"] is True
    assert result["prefix_diagnosis"]["observed_causes"] == ["input_processor_changed"]
    assert (
        result["prefix_diagnosis"]["comparisons"][3]["not_applicable_justified"] is True
    )


def test_real_pi_module_records_can_be_joined_to_backend_without_http_requests(
    analyzer,
):
    module = (
        Path(__file__).resolve().parents[1] / "integrations/pi/qwen-prefix-lineage.mjs"
    ).as_uri()
    script = f"""
import {{ createPrefixLineage }} from {json.dumps(module)};
const user = content => ({{role: 'user', content}});
const answer = text => ({{role:'assistant', content:[{{type:'text',text}}]}});
const chat = {{id:'a'.repeat(64),generation:'b'.repeat(64)}};
const rows=[];
function call(messages, wireMessages, output, id) {{
 const o=createPrefixLineage({{context:{{messages}},sessionId:'synthetic-session',fetch:()=>null,emit:r=>rows.push(r)}});
 const wire={{model:'qwen',messages:wireMessages,kv_transfer_params:{{qwen_chat:chat}}}};
 o.converted(wire);o.wire(wire);o.fetch('http://127.0.0.1/v1/chat/completions',{{body:JSON.stringify(wire)}});
 o.responseId(id);o.assembled(output);
}}
call([user('fixture')],[user('fixture')],answer('previous'),'chatcmpl-old');
rows.length=0;
call([user('fixture'),answer('previous'),user('next')],
 [user('fixture'),{{role:'assistant',content:'previous'}},user('next')],answer('current'),'chatcmpl-current');
console.log(JSON.stringify(rows));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    records = prefix_request() + comparison_records()
    external = hashlib.sha256(b"chatcmpl-current").hexdigest()
    for row in records:
        if "external_request_id" in row:
            row["external_request_id"] = external
    result = analyzer.analyze_request(
        records, HTTP, healthy(), pi_lineage=json.loads(completed.stdout)
    )
    assert result["production_path_diagnosis_complete"] is True
    assert len(result["pi_prefix_diagnosis"]["observations"]) == 6


def test_core_observer_drop_is_not_hidden_by_clean_background_writer_health(analyzer):
    rows = comparison_records()
    rows[0]["dropped_records"] = 1
    result = analyzer.analyze_request(prefix_request() + rows, HTTP, healthy())
    assert result["timing_complete"] is True
    assert result["prefix_diagnosis_complete"] is False
    assert (
        "prefix_source_recorder_loss" in result["prefix_diagnosis"]["missing_evidence"]
    )


@pytest.mark.parametrize("missing", ["hooks", "source"])
def test_prefix_producer_activation_is_required_in_addition_to_healthy_writes(
    analyzer, missing
):
    health = healthy()
    health[0].pop("prefix_lineage_hooks" if missing == "hooks" else "source_sha256")
    result = analyzer.analyze_request(
        prefix_request() + comparison_records(), HTTP, health
    )
    assert result["timing_complete"] is True
    assert result["prefix_diagnosis_complete"] is False
    expected = (
        "prefix_hooks_unconfirmed"
        if missing == "hooks"
        else "prefix_source_identity_unconfirmed"
    )
    assert expected in result["prefix_diagnosis"]["missing_evidence"]


def test_request_cli_accepts_pi_lineage_logs_and_reports_corrupt_client_records(
    analyzer, tmp_path
):
    backend, pi_log, health, output = (
        tmp_path / name
        for name in ("backend.jsonl", "pi.jsonl", "health.json", "report.json")
    )
    backend.write_text(
        "".join(
            json.dumps(row) + "\n" for row in prefix_request() + comparison_records()
        )
    )
    pi_log.write_text(
        "".join(json.dumps(row) + "\n" for row in pi_records()) + "{broken\n"
    )
    health.write_text(json.dumps(healthy()[0]))
    engine_health = tmp_path / "engine-health.json"
    engine_health.write_text(json.dumps(healthy()[1]))
    analyzer.main(
        [
            "--request-id",
            HTTP,
            "--cache-log",
            str(backend),
            "--health",
            str(health),
            "--health",
            str(engine_health),
            "--pi-lineage-log",
            str(pi_log),
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text())
    assert result["prefix_diagnosis_complete"] is True
    assert result["production_path_diagnosis_complete"] is False
    assert result["pi_prefix_diagnosis"]["invalid_records"] == 1
    assert "invalid_pi_records" in result["pi_prefix_diagnosis"]["missing_evidence"]
