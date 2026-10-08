#!/usr/bin/env python3
"""Correlate content-free round or first-output timing; overlap is not causation."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SCHEMA = "urn:coherence:cache-job-timings:v1"
ROUND_SCHEMA = "urn:qwen-r9700:decode-rounds:v1"
PI_LINEAGE_SCHEMA = "urn:coherence:pi-prefix-lineage:v1"
IDENTITIES = ("request_id", "external_request_id", "http_request_id")
HEX_ID = re.compile(r"[0-9a-f]{64}\Z")

# Envelopes, observer work and point markers do not explain what happens
# inside a request. Preserve them in the timeline without claiming coverage.
REQUEST_ENVELOPES = {
    "http_request",
    "http_end",
    "recorder_batch",
    "prefix_lineage",
    "prefix_lineage_observer",
}
PREFIX_COMPARISONS = (
    "raw_to_reencoded",
    "output_to_message",
    "template_normalization",
    "prompt_prefix",
    "input_processor",
)
PREFIX_REJECTION_REASONS = {
    "prefix_hash_changed",
    "partial_prefix_changed",
    "prefix_identity_changed",
    "prefix_identity_unavailable",
}
PREFIX_DIAGNOSTIC_STATUSES = {"complete", "incomplete", "unavailable", "not_applicable"}
PREFIX_GAP_REASONS = {
    "missing",
    "unsupported",
    "evicted",
    "restart",
    "dropped",
    "failed",
    "missing_source",
    "unsupported_tokenizer",
    "source_evicted",
    "source_restart",
    "source_unavailable",
    "comparison_failed",
    "invalid_evidence",
    "not_applicable",
    "missing_cache_salt",
    "partial_output",
    "configuration_changed",
    "matching_prefix",
    "missing_previous_output",
    "overlapping_requests",
}
PREFIX_UNITS = {
    "tokens": ("compared_tokens", "previous_tokens", "current_tokens"),
    "characters": ("compared_characters", "previous_characters", "current_characters"),
    "message_fields": ("compared_fields", "previous_fields", "current_fields"),
}
PREFIX_TEMPLATE_OPERATIONS = {
    "content_trim",
    "reasoning_trim",
    "leading_delimiter",
    "inline_reasoning_split",
    "assistant_delimiters",
    "trim",
    "delimiter",
}
PREFIX_TEMPLATE_SCOPES = {"assistant_message_without_terminator"}
PREFIX_CAUSES = {
    "raw_to_reencoded": "token_roundtrip_changed",
    "output_to_message": "assistant_message_changed",
    "template_normalization": "template_normalization_changed",
    "prompt_prefix": "rendered_prompt_prefix_changed",
    "input_processor": "input_processor_changed",
}


def read_records(paths, schema):
    records, invalid = [], 0
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                if isinstance(row, dict) and row.get("schema") == schema:
                    records.append(row)
                else:
                    invalid += 1
    return records, invalid


def union_ms(spans):
    total, right = 0, None
    for start, end in sorted(spans):
        if end <= start:
            continue
        total += max(0, end - max(start, right if right is not None else start))
        right = max(end, right if right is not None else end)
    return total / 1e6


def _identities(event):
    return {
        (name, event[name])
        for name in IDENTITIES
        if isinstance(event.get(name), str) and HEX_ID.fullmatch(event[name])
    }


def _valid_event(event):
    return (
        isinstance(event, dict)
        and type(event.get("pid")) is int
        and isinstance(event.get("trace_id"), str)
        and bool(event["trace_id"])
        and (event.get("clock_id") is None or isinstance(event["clock_id"], str))
        and type(event.get("sequence")) is int
        and event["sequence"] >= 0
        and isinstance(event.get("stage"), str)
        and type(event.get("start_ns")) is int
        and type(event.get("end_ns")) is int
        and 0 <= event["start_ns"] <= event["end_ns"]
    )


def analyze_prefix_lineage(matches, *, evidence_gaps=()):
    """Keep causal prefix evidence separate from complete timing boundaries.

    Comparisons are supplied by their actual producers. Counts and differences
    are observations, not an inference that a particular transformation caused
    the cache rejection. No token values, text or arbitrary producer messages
    are copied into this report.
    """
    lookups = [row for row in matches if row["stage"] == "response_end_lookup"]
    rejected = [
        row
        for row in lookups
        if isinstance(row.get("reason"), str)
        and row["reason"] in PREFIX_REJECTION_REASONS
    ]
    gaps = set(evidence_gaps)
    for row in lookups:
        outcome, reason = row.get("outcome"), row.get("reason")
        if (
            not isinstance(outcome, str)
            or outcome not in {"hit", "miss", "rejected", "loading"}
            or not isinstance(reason, str)
            or not reason
        ):
            gaps.add("invalid_response_end_lookup")
        elif reason in PREFIX_REJECTION_REASONS and outcome not in {"miss", "rejected"}:
            gaps.add("inconsistent_response_end_lookup")
    comparisons = {}
    template_operations = []
    lineage = [row for row in matches if row["stage"] == "prefix_lineage"]
    for row in lineage:
        if "dropped_records" in row:
            if type(row["dropped_records"]) is not int or row["dropped_records"] < 0:
                gaps.add("invalid_prefix_source_health")
            elif row["dropped_records"]:
                gaps.add("prefix_source_recorder_loss")
        comparison = row.get("comparison")
        if not isinstance(comparison, str) or comparison not in PREFIX_CAUSES:
            gaps.add("invalid_comparison")
            continue
        operation = row.get("operation")
        is_template_operation = (
            operation is not None and row.get("comparison_complete") is not True
        )
        if is_template_operation and not (
            comparison == "template_normalization"
            and isinstance(operation, str)
            and operation in PREFIX_TEMPLATE_OPERATIONS
        ):
            gaps.add("invalid_template_subevent")
            continue
        status = row.get("diagnostic_status")
        observed = {"comparison": comparison}
        if not isinstance(status, str) or status not in PREFIX_DIAGNOSTIC_STATUSES:
            observed["status"] = "invalid"
            gaps.add(f"invalid_status:{comparison}")
        elif status != "complete":
            observed["status"] = status
            reason = row.get("reason")
            observed["reason"] = (
                reason
                if isinstance(reason, str) and reason in PREFIX_GAP_REASONS
                else "missing"
            )
            if rejected:
                gaps.add(f"{observed['reason']}:{comparison}")
        else:
            unit = row.get("unit", "tokens")
            counts = PREFIX_UNITS.get(unit, ()) if isinstance(unit, str) else ()
            valid = (
                bool(counts)
                and type(row.get("equal")) is bool
                and all(
                    type(row.get(name)) is int and row[name] >= 0 for name in counts
                )
            )
            different = row.get("first_difference", row.get("difference_position"))
            valid = valid and (
                row.get("equal") is True
                and different is None
                or row.get("equal") is False
                and type(different) is int
                and 0 <= different <= row[counts[0]]
                or row.get("equal") is False
                and unit in {"message_fields", "characters"}
                and different is None
                and row.get("difference_position_available") is False
            )
            if not valid:
                observed["status"] = "invalid"
                gaps.add(f"invalid_evidence:{comparison}")
            else:
                observed.update(status="complete", equal=row["equal"], unit=unit)
                observed.update({name: row[name] for name in counts})
                if different is not None:
                    observed["first_difference"] = different
                elif row["equal"] is False:
                    observed["difference_position_available"] = False
        if "scope" in row:
            scope = row["scope"]
            if (
                comparison == "template_normalization"
                and row.get("unit") == "characters"
                and isinstance(scope, str)
                and scope in PREFIX_TEMPLATE_SCOPES
            ):
                observed["scope"] = scope
            else:
                observed["status"] = "invalid"
                gaps.add("invalid_template_scope")
        if "recognized_stop_marker_removed" in row:
            removed = row["recognized_stop_marker_removed"]
            if (
                comparison == "template_normalization"
                and row.get("unit") == "characters"
                and type(removed) is bool
            ):
                observed["recognized_stop_marker_removed"] = removed
            else:
                observed["status"] = "invalid"
                gaps.add("invalid_template_stop_marker")
        if is_template_operation:
            # Keep the admitted operation scope visible without allowing an
            # individual transformation to certify the final normalization.
            observed["operation"] = operation
            if observed not in template_operations:
                template_operations.append(observed)
            continue
        previous = comparisons.get(comparison)
        if previous is not None and previous != observed:
            gaps.add(f"conflicting_comparison:{comparison}")
        else:
            comparisons[comparison] = observed
    if not lookups:
        gaps.add("missing_response_end_lookup")
    if rejected:
        for name in PREFIX_COMPARISONS:
            if name not in comparisons:
                gaps.add(f"missing:{name}")
            elif comparisons[name]["status"] != "complete":
                gaps.add(f"incomplete:{name}")
    endpoint_scope = []
    relevant_prefix_change = False
    for lookup in rejected:
        endpoint = lookup.get("endpoint_tokens")
        source = lookup.get("cache_source")
        inspected = {
            "cache_source": source
            if isinstance(source, str)
            and source in {"gpu_endpoint", "offload_endpoint"}
            else "unknown",
            "covered": False,
        }
        if type(endpoint) is not int or endpoint <= 0:
            gaps.add("exact_endpoint_missing")
        else:
            inspected["endpoint_tokens"] = endpoint
            prefix = comparisons.get("prompt_prefix", {})
            processor = comparisons.get("input_processor", {})
            complete = prefix.get("status") == processor.get("status") == "complete"
            if complete:
                covered = (
                    prefix.get("unit") == processor.get("unit") == "tokens"
                    and prefix["previous_tokens"] >= endpoint
                    and prefix["current_tokens"] >= endpoint
                    and processor["previous_tokens"] >= endpoint
                    and processor["current_tokens"] >= endpoint
                )
                inspected["covered"] = covered
                if not covered:
                    gaps.add("exact_endpoint_missing")
                else:
                    differences = [
                        row["first_difference"]
                        for row in (prefix, processor)
                        if row.get("equal") is False and "first_difference" in row
                    ]
                    if differences:
                        first = min(differences)
                        inspected["first_difference"] = first
                        inspected["difference_within_endpoint"] = first < endpoint
                        if first < endpoint:
                            relevant_prefix_change = True
                        else:
                            gaps.add("emitted_unprocessed_boundary")
                    else:
                        inspected["difference_within_endpoint"] = False
                        if lookup["reason"] != "prefix_identity_unavailable":
                            gaps.add("prefix_identity_unexplained")
        endpoint_scope.append(inspected)
    known_endpoints = [
        row["endpoint_tokens"] for row in endpoint_scope if "endpoint_tokens" in row
    ]
    roundtrip = comparisons.get("raw_to_reencoded", {})
    prefix = comparisons.get("prompt_prefix", {})
    processor = comparisons.get("input_processor", {})
    if (
        roundtrip.get("status") == "not_applicable"
        and roundtrip.get("reason") == "matching_prefix"
        and prefix.get("status") == processor.get("status") == "complete"
        and prefix.get("equal") is True
        and processor.get("equal") is False
        and known_endpoints
        and all(row.get("covered") for row in endpoint_scope)
        and processor.get("first_difference", max(known_endpoints))
        < min(known_endpoints)
    ):
        roundtrip["not_applicable_justified"] = True
        gaps.discard("matching_prefix:raw_to_reencoded")
        gaps.discard("incomplete:raw_to_reencoded")
    observed_causes = sorted(
        PREFIX_CAUSES[name]
        for name, row in comparisons.items()
        if row.get("status") == "complete"
        and row.get("equal") is False
        and f"conflicting_comparison:{name}" not in gaps
        and relevant_prefix_change
        and (
            row.get("unit") != "tokens"
            or known_endpoints
            and row.get("first_difference", max(known_endpoints)) < max(known_endpoints)
        )
    )
    status = (
        "UNAVAILABLE"
        if not lookups
        else "INCOMPLETE"
        if gaps
        else "COMPLETE"
        if rejected
        else "NOT_APPLICABLE"
    )
    return {
        "status": status,
        "complete": status in {"COMPLETE", "NOT_APPLICABLE"},
        "required_comparisons": list(PREFIX_COMPARISONS) if rejected else [],
        "comparisons": [comparisons[name] for name in sorted(comparisons)],
        "template_operations": template_operations,
        "rejection_reasons": sorted({row["reason"] for row in rejected}),
        "endpoint_scope": endpoint_scope,
        "observed_causes": observed_causes,
        "missing_evidence": sorted(gaps),
        "interpretation": (
            "Observed causes identify failed boundary comparisons, not a speculative root cause. "
            "COMPLETE requires the declared comparisons and retained producer evidence; "
            "NOT_APPLICABLE means no recorded prefix-identity rejection required them."
        ),
    }


PI_STAGES = {
    "context_before_conversion",
    "provider_converted",
    "provider_sdk_input",
    "provider_wire",
    "provider_response_identity",
    "provider_response_assembled",
}
PI_COMPARISON_STAGES = {"provider_converted", "provider_sdk_input", "provider_wire"}
PI_CHANGE_FLAGS = {
    "provider_converted": "provider_conversion_changed",
    "provider_sdk_input": "payload_hooks_changed",
    "provider_wire": "wire_serialization_changed",
}


def analyze_pi_prefix_lineage(
    records, external_ids, *, invalid_records=0, require_predecessor=False
):
    """Join client evidence only through its explicitly observed SSE response ID.

    Client wall clocks may be on another machine. Neither timestamps nor chat
    identities are used to bridge a request, and no client/server duration is
    subtracted. Early events share the anchor event's producer/ordinal identity.
    """
    valid = []
    for row in records or []:
        if (
            isinstance(row, dict)
            and row.get("schema") == PI_LINEAGE_SCHEMA
            and type(row.get("pid")) is int
            and row["pid"] > 0
            and isinstance(row.get("trace_id"), str)
            and bool(row["trace_id"])
            and type(row.get("ordinal")) is int
            and row["ordinal"] > 0
            and isinstance(row.get("stage"), str)
        ):
            valid.append(row)
        else:
            invalid_records += 1
    groups = defaultdict(list)
    for row in valid:
        groups[(row["pid"], row["trace_id"], row["ordinal"])].append(row)
    matches = [
        (key, rows)
        for key, rows in groups.items()
        if any(
            isinstance(row.get("request_id_sha256"), str)
            and row["request_id_sha256"] in external_ids
            for row in rows
        )
    ]
    gaps = set()
    if not records:
        gaps.add("missing_pi_lineage")
    if not external_ids:
        gaps.add("missing_backend_external_id")
    if not matches:
        gaps.add("missing_pi_request_bridge")
    elif len(matches) != 1:
        gaps.add("ambiguous_pi_request_bridge")
    if invalid_records:
        gaps.add("invalid_pi_records")
    selected = matches[0][1] if len(matches) == 1 else []
    by_stage = defaultdict(list)
    sources = set()
    observed_changes = set()
    safe_rows = []
    count_fields = (
        "hook_mask",
        "messages",
        "common_messages",
        "first_changed_message",
        "before_messages",
        "after_messages",
        "history_first_changed_message",
        "text_bytes",
        "reasoning_bytes",
        "thinking_blocks",
        "whitespace_only_blocks",
    )
    bool_fields = (
        "equal",
        "coverage_complete",
        "truncated",
        "unsupported",
        "observer_failed",
        "body_unavailable",
        "config_changed",
        "serialization_config_changed",
        "previous_available",
        "history_equal",
        "previous_output_equal",
        "previous_output_text_equal",
        "previous_output_reasoning_equal",
        "success",
        "response_id_changed",
        "first_change_text_equal",
        "first_change_reasoning_equal",
        "first_change_toolcalls_equal",
        "first_change_role_equal",
    )
    for row in selected:
        stage = row["stage"]
        if stage not in PI_STAGES | {"provider_response_failed"}:
            gaps.add("unsupported_pi_stage")
            continue
        by_stage[stage].append(row)
        source = row.get("source_id")
        if isinstance(source, str) and HEX_ID.fullmatch(source):
            sources.add(source)
        else:
            gaps.add("missing_pi_source_identity")
        recorder = row.get("recorder")
        if not isinstance(recorder, dict) or any(
            type(recorder.get(name)) is not int or recorder[name] < 0
            for name in ("dropped", "errors")
        ):
            gaps.add("missing_pi_recorder_health")
        elif recorder["dropped"] or recorder["errors"]:
            gaps.add("pi_recorder_loss")
        safe = {"stage": stage}
        for name in count_fields:
            if name in row and row[name] is not None:
                if type(row[name]) is int and row[name] >= 0:
                    safe[name] = row[name]
                else:
                    gaps.add("invalid_pi_evidence")
        for name in bool_fields:
            if name in row and row[name] is not None:
                if type(row[name]) is bool:
                    safe[name] = row[name]
                else:
                    gaps.add("invalid_pi_evidence")
        if safe not in safe_rows:
            safe_rows.append(safe)
        for name in ("truncated", "unsupported", "observer_failed", "body_unavailable"):
            if row.get(name) is True:
                gaps.add(f"pi_{name}")
        if row.get("response_id_changed") is True:
            gaps.add("pi_response_identity_changed")
        if stage in PI_COMPARISON_STAGES:
            if (
                row.get("coverage_complete") is not True
                or type(row.get("equal")) is not bool
            ):
                gaps.add(f"incomplete_pi_comparison:{stage}")
            elif row["equal"] is False:
                observed_changes.add(PI_CHANGE_FLAGS[stage])
        for name in ("config_changed", "serialization_config_changed"):
            if row.get(name) is True:
                observed_changes.add(name)
        if row.get("history_equal") is False:
            observed_changes.add("client_history_changed")
        if row.get("previous_output_equal") is False:
            observed_changes.add("client_previous_output_changed")
    if len(sources) > 1:
        gaps.add("pi_source_identity_changed")
    if selected:
        for stage in PI_STAGES:
            if not by_stage[stage]:
                gaps.add(f"missing_pi_stage:{stage}")
        terminal = by_stage["provider_response_assembled"]
        if not terminal or any(
            row.get("success") is not True
            or row.get("coverage_complete") is not True
            or type(row.get("hook_mask")) is not int
            or row["hook_mask"] != 63
            for row in terminal
        ):
            gaps.add("incomplete_pi_response")
        if by_stage["provider_response_failed"]:
            gaps.add("pi_response_failed")
        if require_predecessor and not any(
            row.get("previous_available") is True
            and type(row.get("previous_output_equal")) is bool
            and type(row.get("history_equal")) is bool
            for row in by_stage["provider_sdk_input"]
        ):
            gaps.add("previous_client_output_unavailable")
    return {
        "status": "UNAVAILABLE"
        if not selected
        else "INCOMPLETE"
        if gaps
        else "COMPLETE",
        "complete": bool(selected) and not gaps,
        "join": "explicit_sse_response_id_sha256",
        "source_ids": sorted(sources),
        "observations": safe_rows,
        "observed_changes": sorted(observed_changes),
        "missing_evidence": sorted(gaps),
        "invalid_records": invalid_records,
        "interpretation": (
            "Only an explicit hashed streamed response ID joins Pi to backend evidence. "
            "Client timestamps do not contribute cross-host durations. COMPLETE describes "
            "the admitted request-construction route, not all client internals or hardware."
        ),
    }


def analyze_request(
    events,
    request_id,
    health=None,
    *,
    invalid_records=0,
    pi_lineage=None,
    invalid_pi_records=0,
):
    """Join explicit hashed-ID bridges and account for a same-host latency window.

    Numeric monotonic timestamps from processes on the same boot can be compared.
    A request identity must travel through an explicit bridge to a new identity;
    proximity, active chat identity or matching PID is never a substitute.
    """
    if not isinstance(request_id, str) or not HEX_ID.fullmatch(request_id):
        raise ValueError("request ID must be a lowercase SHA-256 digest")
    valid = [event for event in events if _valid_event(event)]
    invalid_records += len(events) - len(valid)
    linked = {(name, request_id) for name in IDENTITIES}
    # Only bridges can connect namespace-specific identifiers. A background
    # record's active-chat snapshot must never bring another request into scope.
    while True:
        expanded = set(linked)
        for event in valid:
            ids = _identities(event)
            if event["stage"] == "internal_id_bridge" and linked & ids:
                expanded.update(ids)
        if expanded == linked:
            break
        linked = expanded
    matches = [event for event in valid if _identities(event) & linked]
    # Duplicate inputs (e.g. the same file twice) do not create extra spans.
    unique = {}
    conflicts = 0
    for event in matches:
        key = (event["pid"], event["trace_id"], event["sequence"])
        if key in unique and unique[key] != event:
            conflicts += 1
        else:
            unique[key] = event
    matches = sorted(
        unique.values(), key=lambda row: (row["start_ns"], row["sequence"])
    )
    by_stage = defaultdict(list)
    for event in matches:
        by_stage[event["stage"]].append(event)
    reasons = []
    clocks = {row.get("clock_id") for row in matches}
    clock_verified = (
        len(clocks) == 1
        and None not in clocks
        and all(isinstance(clock, str) and HEX_ID.fullmatch(clock) for clock in clocks)
    )
    if not clock_verified:
        reasons.append(
            "clock_domain_mismatch"
            if len(clocks - {None}) > 1
            else "clock_domain_unverified"
        )
    starts = by_stage["http_request"]
    if len(starts) != 1:
        reasons.append("missing_http_start" if not starts else "ambiguous_http_start")
    start = starts[0]["start_ns"] if len(starts) == 1 else None
    # A protocol-first-body marker is deliberately not treated as a token.
    checkpoints = {
        "engine_first_output": "first_engine_output",
        "api_first_content": "first_api_content",
        "http_first_body": "http_first_body",
        "http_headers": "http_headers",
        "http_end": "http_end",
        "scheduler_first_output": "scheduler_first_output",
    }
    points = {}
    for label, stage in checkpoints.items():
        values = by_stage[stage]
        points[label] = min((row["end_ns"] for row in values), default=None)
    if not by_stage["internal_id_bridge"]:
        reasons.append("missing_internal_id_bridge")
    if points["engine_first_output"] is None:
        reasons.append("missing_engine_first_output")
    if len(by_stage["http_end"]) != 1:
        reasons.append(
            "missing_http_end" if not by_stage["http_end"] else "ambiguous_http_end"
        )
    elif by_stage["http_end"][0].get("success") is not True:
        reasons.append("http_request_failed_or_cancelled")
    if any(row.get("success") is False for row in matches):
        reasons.append("failed_phase")
    if invalid_records:
        reasons.append("invalid_records")
    if conflicts:
        reasons.append("conflicting_duplicate_events")
    if start is not None and any(
        value is not None and value < start for value in points.values()
    ):
        reasons.append("noncausal_timestamps")
    sequences = defaultdict(set)
    for row in valid:
        sequences[(row["pid"], row["trace_id"])].add(row["sequence"])
    lifecycles = {(row["pid"], row["trace_id"]) for row in matches}
    gaps = sum(
        max(sequences[key]) - min(sequences[key]) + 1 - len(sequences[key])
        for key in lifecycles
        if sequences[key]
    )
    if gaps:
        reasons.append("sequence_gaps")
    # One health file covers one producer; cross-process captures should include
    # all health snapshots. Missing health is unknown, never a clean zero.
    health_rows = [health] if isinstance(health, dict) else (health or [])
    health_by_key = {
        (row.get("pid"), row.get("trace_id")): row
        for row in health_rows
        if isinstance(row, dict)
    }
    unmatched_health = sorted(lifecycles - set(health_by_key))
    if unmatched_health:
        reasons.append("missing_lifecycle_health")
    for key in lifecycles & set(health_by_key):
        row = health_by_key[key]
        if (
            not clock_verified
            or not isinstance(row.get("clock_id"), str)
            or row["clock_id"] not in clocks
        ):
            reasons.append("lifecycle_clock_unverified")
            clock_verified = False
        if type(row.get("started_ns")) is not int or row["started_ns"] < 0:
            reasons.append("missing_lifecycle_start")
        elif any(
            event["start_ns"] < row["started_ns"]
            for event in matches
            if (event["pid"], event["trace_id"]) == key
        ):
            reasons.append("event_precedes_lifecycle")
        if (
            any(
                event["stage"] == "scheduler_admitted"
                and (event["pid"], event["trace_id"]) == key
                for event in matches
            )
            and row.get("worker_first_work_hooks") is not True
        ):
            reasons.append("worker_first_work_hooks_unconfirmed")
        if any(
            type(row.get(name)) is not int or row[name] < 0
            for name in ("dropped", "write_errors")
        ):
            reasons.append("invalid_lifecycle_health")
        elif row["dropped"] or row["write_errors"] or row.get("context_drops", 0):
            reasons.append("recorder_loss")
    selected_label = (
        "api_first_content"
        if points["api_first_content"] is not None
        else "engine_first_output"
    )
    end = points[selected_label]
    window_valid = (
        clock_verified and start is not None and end is not None and end >= start
    )
    unterminated = []
    if window_valid:
        for row in matches:
            if not row["stage"].startswith("phase_enter_") or row["start_ns"] >= end:
                continue
            stage = "phase_" + row["stage"].removeprefix("phase_enter_")
            if not any(
                close["pid"] == row["pid"]
                and close["trace_id"] == row["trace_id"]
                and close["start_ns"] <= row["start_ns"] <= close["end_ns"]
                for close in by_stage[stage]
            ):
                unterminated.append(stage)
        if unterminated:
            reasons.append("unterminated_phase")
    intervals = defaultdict(list)
    timeline = []
    for row in matches:
        timeline.append(
            {
                "stage": row["stage"],
                "pid": row["pid"],
                "trace_id": row["trace_id"],
                "sequence": row["sequence"],
                "start_offset_ms": (row["start_ns"] - start) / 1e6
                if clock_verified and start is not None
                else None,
                "end_offset_ms": (row["end_ns"] - start) / 1e6
                if clock_verified and start is not None
                else None,
                **{
                    name: row[name]
                    for name in (
                        *IDENTITIES,
                        "success",
                        "status_code",
                        "bytes",
                        "input_tokens",
                        "computed_tokens",
                        "scheduled_tokens",
                    )
                    if name in row
                },
            }
        )
        if window_valid and row["stage"] not in REQUEST_ENVELOPES:
            first, last = max(start, row["start_ns"]), min(end, row["end_ns"])
            if last > first:
                intervals[row["stage"]].append((first, last))
    covered = (
        union_ms([span for spans in intervals.values() for span in spans])
        if window_valid
        else None
    )
    elapsed = (end - start) / 1e6 if window_valid else None
    timings = {
        label + "_ms": (value - start) / 1e6
        if clock_verified and start is not None and value is not None and value >= start
        else None
        for label, value in points.items()
    }
    prefix_integrity_reasons = {
        "clock_domain_mismatch",
        "clock_domain_unverified",
        "invalid_records",
        "conflicting_duplicate_events",
        "sequence_gaps",
        "missing_lifecycle_health",
        "lifecycle_clock_unverified",
        "missing_lifecycle_start",
        "event_precedes_lifecycle",
        "invalid_lifecycle_health",
        "recorder_loss",
    }
    prefix_gaps = set(reasons) & prefix_integrity_reasons
    prefix_producers = {
        (row["pid"], row["trace_id"])
        for row in matches
        if row["stage"] == "prefix_lineage"
    }
    for key in prefix_producers:
        producer = health_by_key.get(key, {})
        hooks = producer.get("prefix_lineage_hooks")
        if not isinstance(hooks, dict) or any(
            hooks.get(name) is not True
            for name in (
                "api_request",
                "engine_tokens",
                "serialized_delta",
                "input_processor",
            )
        ):
            prefix_gaps.add("prefix_hooks_unconfirmed")
        sources = producer.get("source_sha256")
        if not isinstance(sources, dict) or any(
            not isinstance(sources.get(name), str)
            or not HEX_ID.fullmatch(sources[name])
            for name in ("prefix_lineage", "prefix_runtime")
        ):
            prefix_gaps.add("prefix_source_identity_unconfirmed")
    prefix_diagnosis = analyze_prefix_lineage(matches, evidence_gaps=prefix_gaps)
    external_ids = {
        row["external_request_id"]
        for row in matches
        if isinstance(row.get("external_request_id"), str)
        and HEX_ID.fullmatch(row["external_request_id"])
    }
    pi_diagnosis = analyze_pi_prefix_lineage(
        pi_lineage,
        external_ids,
        invalid_records=invalid_pi_records,
        require_predecessor=bool(prefix_diagnosis["rejection_reasons"]),
    )
    return {
        "schema": "urn:coherence:request-timeline:v1",
        "request_id": request_id,
        "status": "INCOMPLETE" if reasons else "COMPLETE",
        "timing_complete": not reasons,
        "prefix_diagnosis_complete": prefix_diagnosis["complete"],
        "prefix_diagnosis": prefix_diagnosis,
        "pi_prefix_diagnosis": pi_diagnosis,
        "production_path_diagnosis_complete": not reasons
        and prefix_diagnosis["complete"]
        and pi_diagnosis["complete"],
        "incomplete_reasons": sorted(set(reasons)),
        "latencies": timings,
        "window": {
            "start": "http_request",
            "end": selected_label,
            "elapsed_ms": elapsed,
        },
        "covered_union_ms": covered,
        "unattributed_ms": max(0.0, elapsed - covered) if window_valid else None,
        "overlap_by_stage_ms": {
            stage: union_ms(spans) for stage, spans in intervals.items()
        },
        "timeline": timeline,
        "coverage": {
            "invalid_records": invalid_records,
            "sequence_gaps": gaps,
            "conflicting_duplicates": conflicts,
            "clock_domain_verified": clock_verified,
            "unterminated_phases": sorted(set(unterminated)),
            "missing_health_lifecycles": [
                {"pid": pid, "trace_id": trace} for pid, trace in unmatched_health
            ],
            "recorder_health": health_rows,
        },
        "interpretation": (
            "Engine TTFT, serialized API content, first HTTP body and completed HTTP wall time are separate boundaries. "
            "An HTTP body can be a role-only SSE event, not a generated token. Cross-process joins require explicit "
            "hashed-ID bridges on the same host/boot. Phase spans are clipped to the selected window and covered "
            "time is their union; per-stage overlaps must not be summed. Unattributed time is an observed gap, "
            "not an identified cause. COMPLETE means captured boundaries/health are present, not proof of a "
            "blocking mechanism or guaranteed retention; missing/drop/corrupt/failed evidence is INCOMPLETE."
        ),
    }


def analyze(rounds, events, health=None, *, threshold_ms=65):
    # Joining on lifecycle identity prevents attributing old-process events.
    indexed = defaultdict(list)
    sequences = defaultdict(set)
    for row in events:
        key = (row["trace_id"], row["pid"])
        indexed[key].append(row)
        sequences[key].add(row["sequence"])
    outliers = []
    timed = [r for r in rounds if isinstance(r.get("round_ms"), (int, float))]
    for row in timed:
        if row["round_ms"] < threshold_ms:
            continue
        end = row.get("monotonic_ns")
        key = (row.get("cache_trace_id"), row["pid"])
        matches = []
        if end is not None and key in indexed:
            start = end - int(row["round_ms"] * 1e6)
            matches = [
                e for e in indexed[key] if e["end_ns"] > start and e["start_ns"] < end
            ]
        by_stage = defaultdict(list)
        for event in matches:
            by_stage[event["stage"]].append(
                (max(start, event["start_ns"]), min(end, event["end_ns"]))
            )
        outliers.append(
            {
                "request_id": row["request_id"],
                "round": row["round"],
                "round_ms": row["round_ms"],
                "computed_tokens": row.get("computed_tokens"),
                "matched_lifecycle": end is not None and key in indexed,
                "overlap_by_stage_ms": {
                    name: round(union_ms(spans), 6) for name, spans in by_stage.items()
                },
                "events": [
                    {
                        "start_offset_ms": round((e["start_ns"] - start) / 1e6, 6),
                        "end_offset_ms": round((e["end_ns"] - start) / 1e6, 6),
                        **{
                            name: e[name]
                            for name in (
                                "stage",
                                "duration_ms",
                                "thread_cpu_ms",
                                "minor_faults",
                                "major_faults",
                                "job_kind",
                                "job_id",
                                "direction",
                                "request_id",
                                "round",
                                "bytes",
                                "logical_ranges",
                                "gpu_elapsed_ms",
                                "gpu_inter_round_gap_ms",
                                "thread_user_ms",
                                "thread_system_ms",
                                "off_cpu_ms",
                                "voluntary_switches",
                                "involuntary_switches",
                                "origin_thread_id",
                                "stream_match",
                                "lifetime_ms",
                                "success",
                                "thread_id",
                            )
                            if name in e
                        },
                    }
                    for e in matches
                ],
            }
        )
    gaps = sum(
        max(values) - min(values) + 1 - len(values)
        for values in sequences.values()
        if values
    )
    return {
        "schema": "urn:coherence:cache-round-correlation:v1",
        "timed_rounds": len(timed),
        "threshold_ms": threshold_ms,
        "outlier_count": len(outliers),
        "outliers": outliers,
        "coverage": {
            "sequence_gaps": gaps,
            "recorder_health": health,
            "unmatched_outliers": sum(not r["matched_lifecycle"] for r in outliers),
        },
        "interpretation": (
            "Intervals overlap; this does not establish the blocking cause. gpu_elapsed_ms uses "
            "the handler's existing completed HIP events. gpu_complete lifetime includes host "
            "submission, queue/dependency waits and completion polling delay. CPU spans can nest "
            "and run concurrently; never add their durations to infer round cost. Missing or "
            "dropped events, rotation and incomplete boundary jobs limit attribution. "
            "main_thread_round measures thread counters over the completed-step interval, "
            "including work outside execute_model. off_cpu_ms is wall minus thread CPU, not "
            "an identified wait reason. gpu_round is the current-stream marker span from "
            "execute_model entry to sample_tokens return; it includes GPU idle/dependency "
            "gaps inside that span, not just active kernels. Its inter-round gap is reported "
            "only for consecutive successful rounds of the same request and stream."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-log", type=Path, action="append")
    parser.add_argument("--cache-log", type=Path, action="append", required=True)
    parser.add_argument(
        "--pi-lineage-log",
        type=Path,
        action="append",
        help="optional Pi prefix-lineage JSONL files, joined by streamed response identity",
    )
    parser.add_argument("--health", type=Path, action="append")
    parser.add_argument(
        "--request-id",
        help="Hashed HTTP, external or internal request ID for first-output analysis",
    )
    parser.add_argument("--threshold-ms", type=float, default=65)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.threshold_ms <= 0:
        parser.error("threshold must be positive")
    events, bad_events = read_records(args.cache_log, SCHEMA)
    health_rows = [json.loads(path.read_text()) for path in args.health or []]
    if args.request_id:
        if not HEX_ID.fullmatch(args.request_id):
            parser.error("request ID must be a lowercase SHA-256 digest")
        pi_rows, bad_pi = read_records(args.pi_lineage_log or [], PI_LINEAGE_SCHEMA)
        result = analyze_request(
            events,
            args.request_id,
            health_rows,
            invalid_records=bad_events,
            pi_lineage=pi_rows,
            invalid_pi_records=bad_pi,
        )
    else:
        if args.pi_lineage_log:
            parser.error("--pi-lineage-log requires --request-id")
        if not args.round_log:
            parser.error("--round-log is required unless --request-id is supplied")
        rounds, bad_rounds = read_records(args.round_log, ROUND_SCHEMA)
        health = health_rows[0] if len(health_rows) == 1 else health_rows or None
        result = analyze(rounds, events, health, threshold_ms=args.threshold_ms)
        result["coverage"]["invalid_records"] = bad_rounds + bad_events
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
