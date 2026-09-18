# QWEN_ASSURANCE_ONLY_BEGIN: assurance-layer-diagnosis
"""Locate the first serial/M8 semantic divergence in assurance layer streams.

The long-form oracle proves post-commit equality.  This reducer is deliberately
narrower: it compares one absolute target position in the assurance-only
``layers.jsonl`` streams and reports the first unequal decoder or Quest object.
It never treats a later equality as repairing an earlier divergence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

LAYER_SCHEMA = "qwen-r9700.dflash-layer-boundary.v1"
QUEST_SCHEMA = "qwen-r9700.quest-m8-q1-micro.v3"
LEGACY_QUEST_SCHEMAS = {
    "qwen-r9700.quest-m8-q1-micro.v2",
    "qwen-r9700.quest-m8-q1-micro.v1",
}
GDN_SERIAL_SCHEMA = "qwen-r9700.dflash-gdn-state-boundary.v3"
GDN_M8_SCHEMA = "qwen-r9700.dflash-gdn-spec-boundary.v2"
LEGACY_GDN_SERIAL_SCHEMAS = {"qwen-r9700.dflash-gdn-state-boundary.v2"}
LEGACY_GDN_M8_SCHEMAS = {"qwen-r9700.dflash-gdn-spec-boundary.v1"}
GDN_SCHEMAS = {
    GDN_SERIAL_SCHEMA,
    GDN_M8_SCHEMA,
    *LEGACY_GDN_SERIAL_SCHEMAS,
    *LEGACY_GDN_M8_SCHEMAS,
}
LAYER_COUNT = 64
LAYER_DETAIL_FIELDS = (
    ("entry_hidden_sha256", "decoder_layer_input"),
    ("entry_residual_sha256", "decoder_layer_input"),
    ("input_norm_sha256", "input_normalization"),
    ("attention_qkv_projection_sha256", "w4a16_projections"),
    ("attention_aux_projection_sha256", "w4a16_projections"),
    ("attention_output_sha256", "attention_output"),
    ("post_attention_norm_sha256", "post_attention_normalization"),
    ("mlp_gate_up_projection_sha256", "w4a16_projections"),
    ("mlp_activation_sha256", "swiglu_activation"),
    ("mlp_down_projection_sha256", "w4a16_projections"),
    ("mlp_output_sha256", "mlp_output"),
)
SECONDARY_DETAIL_MODULES = {
    "input_norm_secondary_sha256": "residual_stream",
    "post_attention_norm_secondary_sha256": "residual_stream",
}


class LayerDiagnosisError(RuntimeError):
    """Raised when layer evidence is unsafe, malformed, or incomparable."""


def _private_regular(path: Path, label: str) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise LayerDiagnosisError(f"{label} does not exist") from error
    if not stat.S_ISREG(status.st_mode) or path.is_symlink():
        raise LayerDiagnosisError(f"{label} must be a regular non-symlink file")
    if status.st_uid != os.getuid():
        raise LayerDiagnosisError(f"{label} is not owned by the current user")
    if stat.S_IMODE(status.st_mode) & 0o077:
        raise LayerDiagnosisError(f"{label} must be owner-only")


def _load_stream(path: Path, label: str) -> list[dict[str, Any]]:
    path = path.absolute()
    _private_regular(path, label)
    records: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, payload in enumerate(stream, 1):
            if len(payload) > 16 * 1024 * 1024:
                raise LayerDiagnosisError(f"{label} line {line_number} exceeds 16 MiB")
            try:
                value = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise LayerDiagnosisError(f"{label} line {line_number} is invalid JSON") from error
            if not isinstance(value, dict):
                raise LayerDiagnosisError(f"{label} line {line_number} is not an object")
            if set(value) == {"event"}:
                event = value["event"]
                if not isinstance(event, dict):
                    raise LayerDiagnosisError(f"{label} line {line_number} event is not an object")
                records.append(event)
            else:
                records.append(value)
    if not records:
        raise LayerDiagnosisError(f"{label} is empty")
    return records


def _position_row(record: dict[str, Any], target_position: int) -> int | None:
    positions = record.get("positions")
    rows = record.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= 8:
        raise LayerDiagnosisError("layer record rows are invalid")
    if not isinstance(positions, list) or len(positions) != rows:
        raise LayerDiagnosisError("layer record positions do not match rows")
    matches: list[int] = []
    for row, axes in enumerate(positions):
        if (
            not isinstance(axes, list)
            or not axes
            or any(isinstance(item, bool) or not isinstance(item, int) for item in axes)
        ):
            raise LayerDiagnosisError("layer record position axes are invalid")
        if target_position in axes:
            matches.append(row)
    if len(matches) > 1:
        raise LayerDiagnosisError("target position occurs in more than one layer row")
    return matches[0] if matches else None


def _validated_layer_indices(layer_indices: tuple[int, ...] | None) -> tuple[int, ...]:
    selected = tuple(range(LAYER_COUNT)) if layer_indices is None else layer_indices
    if (
        not selected
        or tuple(sorted(set(selected))) != selected
        or any(isinstance(layer, bool) or not 0 <= layer < LAYER_COUNT for layer in selected)
    ):
        raise LayerDiagnosisError("layer indices must be unique ordered values in 0..63")
    return selected


def _layer_passes(
    records: list[dict[str, Any]],
    target_position: int,
    layer_indices: tuple[int, ...],
) -> list[list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        if record.get("schema") != LAYER_SCHEMA:
            continue
        row = _position_row(record, target_position)
        if row is None:
            continue
        layer = record.get("layer_index")
        if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < LAYER_COUNT:
            raise LayerDiagnosisError("layer index is invalid")
        if layer not in layer_indices:
            continue
        selected.append({**record, "_comparison_row": row})
    if not selected:
        raise LayerDiagnosisError("stream has no decoder layers at the target position")
    pass_width = len(layer_indices)
    if len(selected) % pass_width:
        raise LayerDiagnosisError(
            "target-position decoder records do not form complete selected-layer passes"
        )
    passes = [selected[index : index + pass_width] for index in range(0, len(selected), pass_width)]
    for pass_index, layer_pass in enumerate(passes):
        indices = [record["layer_index"] for record in layer_pass]
        if indices != list(layer_indices):
            raise LayerDiagnosisError(
                f"decoder pass {pass_index} does not match the selected ordered layers"
            )
    return passes


def _quest_passes(
    records: list[dict[str, Any]],
    target_position: int,
    quest_layer_indices: tuple[int, ...],
) -> list[list[dict[str, Any]]]:
    pass_width = len(quest_layer_indices)
    if not pass_width:
        return []
    selected: list[dict[str, Any]] = []
    for record in records:
        schema = record.get("schema")
        if schema not in {QUEST_SCHEMA, *LEGACY_QUEST_SCHEMAS}:
            continue
        position = (
            record.get("logical_position")
            if schema in {QUEST_SCHEMA, "qwen-r9700.quest-m8-q1-micro.v2"}
            else record.get("committed_prefix_length")
        )
        if position != target_position:
            continue
        name = record.get("layer_name")
        if not isinstance(name, str) or not name:
            raise LayerDiagnosisError("Quest record has an invalid layer name")
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match is None:
            raise LayerDiagnosisError("Quest record layer name has no numeric model layer")
        if int(match.group(1)) in quest_layer_indices:
            selected.append(record)
    if not selected:
        raise LayerDiagnosisError("stream has no Quest micro records at the target position")
    if len(selected) % pass_width:
        raise LayerDiagnosisError("Quest micro records do not form complete selected-layer passes")
    passes = [selected[index : index + pass_width] for index in range(0, len(selected), pass_width)]
    for pass_index, quest_pass in enumerate(passes):
        names = [record.get("layer_name") for record in quest_pass]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
            names
        ):
            raise LayerDiagnosisError(
                f"Quest pass {pass_index} has invalid or duplicate layer names"
            )
        observed_indices = tuple(
            int(match.group(1))
            for name in names
            if (match := re.search(r"\.layers\.(\d+)\.", name)) is not None
        )
        if observed_indices != quest_layer_indices:
            raise LayerDiagnosisError(
                f"Quest pass {pass_index} does not match the selected attention layers"
            )
    return passes


def _layer_index_from_prefix(value: object, label: str) -> int:
    if not isinstance(value, str) or not value:
        raise LayerDiagnosisError(f"{label} has an invalid layer prefix")
    match = re.search(r"\.layers\.(\d+)\.", value)
    if match is None:
        raise LayerDiagnosisError(f"{label} prefix has no numeric model layer")
    layer = int(match.group(1))
    if not 0 <= layer < LAYER_COUNT:
        raise LayerDiagnosisError(f"{label} layer index is outside 0..63")
    return layer


def _gdn_passes(
    records: list[dict[str, Any]],
    target_position: int,
    gdn_layer_indices: tuple[int, ...],
    *,
    required: bool,
) -> list[list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        if record.get("schema") not in GDN_SCHEMAS:
            continue
        row = _position_row(record, target_position)
        if row is None:
            continue
        layer = _layer_index_from_prefix(record.get("prefix"), "GDN record")
        if layer in gdn_layer_indices:
            selected.append({**record, "_comparison_row": row, "_layer_index": layer})
    if not selected:
        if required:
            raise LayerDiagnosisError(
                "stream has no GDN sub-operation records at the target position"
            )
        return []
    pass_width = len(gdn_layer_indices)
    if len(selected) % pass_width:
        raise LayerDiagnosisError("GDN records do not form complete selected-layer passes")
    passes = [selected[index : index + pass_width] for index in range(0, len(selected), pass_width)]
    for pass_index, gdn_pass in enumerate(passes):
        indices = [record["_layer_index"] for record in gdn_pass]
        if indices != list(gdn_layer_indices):
            raise LayerDiagnosisError(
                f"GDN pass {pass_index} does not match the selected recurrent layers"
            )
    return passes


def _row_value(record: dict[str, Any], field: str) -> Any:
    value = record.get(field)
    rows = record["rows"]
    row = record["_comparison_row"]
    if value is None and field == "input_residual_sha256":
        return None
    if not isinstance(value, list) or len(value) != rows:
        raise LayerDiagnosisError(f"layer field {field} does not match its row count")
    return value[row]


def _gdn_row_value(record: dict[str, Any], field: str) -> Any:
    value = record.get(field)
    rows = record["rows"]
    row = record["_comparison_row"]
    if not isinstance(value, list) or len(value) != rows:
        raise LayerDiagnosisError(f"GDN field {field} does not match its row count")
    return value[row]


def _logical_gdn_call(record: dict[str, Any], field: str) -> tuple[dict[str, Any], int]:
    calls = record.get(field)
    rows = record["rows"]
    row = record["_comparison_row"]
    if not isinstance(calls, list) or not calls:
        raise LayerDiagnosisError(f"GDN record has no {field}")
    if len(calls) == 1:
        call = calls[0]
        call_row = row
    elif len(calls) == rows:
        call = calls[row]
        call_row = 0
    else:
        raise LayerDiagnosisError(
            f"GDN {field} count must be one batched call or one call per logical row"
        )
    if not isinstance(call, dict):
        raise LayerDiagnosisError(f"GDN {field} entry is not an object")
    return call, call_row


def _gdn_call_row_value(record: dict[str, Any], call_field: str, value_field: str) -> Any:
    call, call_row = _logical_gdn_call(record, call_field)
    value = call.get(value_field)
    if not isinstance(value, list) or not 0 <= call_row < len(value):
        raise LayerDiagnosisError(f"GDN {call_field}.{value_field} has no comparison row")
    return value[call_row]


def _flatten_state_indices(value: object) -> list[int]:
    if isinstance(value, list):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_flatten_state_indices(item))
        return flattened
    if isinstance(value, bool) or not isinstance(value, int):
        raise LayerDiagnosisError("GDN state indices are not integers")
    return [value]


def _gdn_call_state_value(record: dict[str, Any], call_field: str, state_field: str) -> str:
    call, call_row = _logical_gdn_call(record, call_field)
    state = call.get(state_field)
    if not isinstance(state, dict):
        raise LayerDiagnosisError(f"GDN {call_field}.{state_field} is not a state map")
    indices = _flatten_state_indices(call.get("state_indices"))
    if len(indices) == 1:
        physical_index = indices[0]
    elif len(indices) == record["rows"]:
        physical_index = indices[record["_comparison_row"]]
    elif 0 <= call_row < len(indices):
        physical_index = indices[call_row]
    else:
        raise LayerDiagnosisError(f"GDN {call_field} state indices have no comparison row")
    digest = state.get(str(physical_index))
    if not isinstance(digest, str) or len(digest) != 64:
        raise LayerDiagnosisError(
            f"GDN {call_field}.{state_field} lacks physical state {physical_index}"
        )
    return digest


def _first_gdn_difference(
    serial: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    allow_external_recurrence_state: bool,
) -> dict[str, Any] | None:
    if len(serial) != len(candidate):
        raise LayerDiagnosisError("serial and M8 GDN layer sets differ")
    call_fields = (
        ("conv_calls", "input_sha256", "gdn_convolution_input"),
        ("conv_calls", "state_before", "gdn_convolution_state"),
        ("conv_calls", "output_sha256", "gdn_convolution_output"),
        ("conv_calls", "state_after", "gdn_convolution_state"),
        ("recurrent_calls", "q_sha256", "gdn_recurrence_input"),
        ("recurrent_calls", "k_sha256", "gdn_recurrence_input"),
        ("recurrent_calls", "v_sha256", "gdn_recurrence_input"),
        ("recurrent_calls", "state_before", "gdn_recurrence_state"),
        ("recurrent_calls", "output_sha256", "gdn_recurrence_output"),
        ("recurrent_calls", "state_after", "gdn_recurrence_state"),
        ("norm_calls", "input_sha256", "gdn_gated_rmsnorm_input"),
        ("norm_calls", "output_gate_sha256", "gdn_gated_rmsnorm_gate"),
        ("norm_calls", "output_sha256", "gdn_gated_rmsnorm_output"),
    )
    for left, right in zip(serial, candidate, strict=True):
        if left["_layer_index"] != right["_layer_index"]:
            raise LayerDiagnosisError("serial and M8 GDN layer identities differ")
        layer = left["_layer_index"]
        if left.get("schema") not in {GDN_SERIAL_SCHEMA, *LEGACY_GDN_SERIAL_SCHEMAS}:
            raise LayerDiagnosisError("serial GDN evidence is not a serial state-boundary record")
        if right.get("schema") not in GDN_SCHEMAS:
            raise LayerDiagnosisError("candidate GDN evidence has an unsupported schema")
        left_has_norm = left.get("schema") == GDN_SERIAL_SCHEMA
        right_has_norm = right.get("schema") in {GDN_SERIAL_SCHEMA, GDN_M8_SCHEMA}
        if left_has_norm != right_has_norm:
            raise LayerDiagnosisError("serial and M8 GDN norm evidence versions differ")
        right_mixed_field = (
            "mixed_qkv_preconv_sha256"
            if right.get("schema") == GDN_M8_SCHEMA
            else "mixed_qkv_sha256"
        )
        top_level_fields = (
            ("mixed_qkv_sha256", right_mixed_field, "gdn_projected_qkv"),
            ("a_sha256", "a_sha256", "gdn_gate_parameters"),
            ("b_sha256", "b_sha256", "gdn_gate_parameters"),
        )
        for left_field, right_field, module in top_level_fields:
            left_value = _gdn_row_value(left, left_field)
            right_value = _gdn_row_value(right, right_field)
            if left_value != right_value:
                return {
                    "boundary": "gdn_suboperation",
                    "field": left_field,
                    "layer_index": layer,
                    "module": module,
                    "serial_sha256": left_value,
                    "m8_sha256": right_value,
                }
        for call_field, value_field, module in call_fields:
            if call_field == "norm_calls" and not left_has_norm:
                continue
            if call_field == "recurrent_calls" and (
                left.get(call_field) is None or right.get(call_field) is None
            ):
                if not allow_external_recurrence_state:
                    raise LayerDiagnosisError("GDN record has no recurrent_calls")
                if left.get(call_field) is not None or right.get(call_field) is not None:
                    raise LayerDiagnosisError(
                        "serial and M8 recurrent sub-operation evidence availability differs"
                    )
                # The qualified packed recurrence kernel does not expose an
                # intermediate Python call to capture.  A caller may omit this
                # inner boundary only when it separately binds the canonical
                # post-transition state for every accepted width.  Projected
                # operands, convolution, norm, decoder output and all other
                # boundaries remain mandatory here.
                continue
            if value_field in {"state_before", "state_after"}:
                left_value = _gdn_call_state_value(left, call_field, value_field)
                right_value = _gdn_call_state_value(right, call_field, value_field)
            else:
                left_value = _gdn_call_row_value(left, call_field, value_field)
                right_value = _gdn_call_row_value(right, call_field, value_field)
            if left_value != right_value:
                return {
                    "boundary": "gdn_suboperation",
                    "field": f"{call_field}.{value_field}",
                    "layer_index": layer,
                    "module": module,
                    "serial_sha256": left_value,
                    "m8_sha256": right_value,
                }
    return None


def _first_layer_difference(
    serial: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for left, right in zip(serial, candidate, strict=True):
        if left["layer_index"] != right["layer_index"] or left.get("layer_type") != right.get(
            "layer_type"
        ):
            raise LayerDiagnosisError("serial and M8 decoder layer identities differ")
        for field in ("input_hidden_sha256", "input_residual_sha256"):
            left_value = _row_value(left, field)
            right_value = _row_value(right, field)
            if left_value != right_value:
                return {
                    "boundary": "decoder_layer",
                    "field": field,
                    "layer_index": left["layer_index"],
                    "layer_type": left["layer_type"],
                    "module": "decoder_layer_input",
                    "serial_sha256": left_value,
                    "m8_sha256": right_value,
                }
        left_detail = left.get("detail")
        right_detail = right.get("detail")
        if left_detail is not None or right_detail is not None:
            if not isinstance(left_detail, dict) or not isinstance(right_detail, dict):
                raise LayerDiagnosisError(
                    "serial and M8 decoder detail captures are not both present"
                )
            if set(left_detail) != set(right_detail):
                raise LayerDiagnosisError("serial and M8 decoder detail fields differ")
            expected = {field for field, _module in LAYER_DETAIL_FIELDS}
            if not expected.issubset(left_detail):
                missing = sorted(expected - set(left_detail))
                raise LayerDiagnosisError(
                    f"decoder detail capture lacks required semantic fields: {missing}"
                )
            ordered_fields = [*LAYER_DETAIL_FIELDS]
            ordered_fields.extend(
                (field, SECONDARY_DETAIL_MODULES.get(field, "w4a16_projections"))
                for field in sorted(set(left_detail) - expected)
                if field.endswith("_secondary_sha256")
            )
            for field, module in ordered_fields:
                left_value = left_detail[field]
                right_value = right_detail[field]
                if left_value is None and right_value is None:
                    # Some decoder inputs are intentionally absent.  In particular,
                    # the first layer has no incoming residual tensor.  The capture
                    # contract records that as JSON null in both serial and M8
                    # streams, so it is an equal semantic value rather than a
                    # malformed row vector.
                    continue
                if left_value is None or right_value is None:
                    return {
                        "boundary": "decoder_layer_detail",
                        "field": field,
                        "layer_index": left["layer_index"],
                        "layer_type": left["layer_type"],
                        "module": module,
                        "serial_sha256": left_value,
                        "m8_sha256": right_value,
                    }
                rows = left["rows"]
                row = left["_comparison_row"]
                if (
                    not isinstance(left_value, list)
                    or not isinstance(right_value, list)
                    or len(left_value) != rows
                    or len(right_value) != right["rows"]
                ):
                    raise LayerDiagnosisError(
                        f"decoder detail field {field} does not match its row count"
                    )
                if left_value[row] != right_value[right["_comparison_row"]]:
                    return {
                        "boundary": "decoder_layer_detail",
                        "field": field,
                        "layer_index": left["layer_index"],
                        "layer_type": left["layer_type"],
                        "module": module,
                        "serial_sha256": left_value[row],
                        "m8_sha256": right_value[right["_comparison_row"]],
                    }
        for field in ("output_hidden_sha256", "output_residual_sha256"):
            left_value = _row_value(left, field)
            right_value = _row_value(right, field)
            if left_value != right_value:
                return {
                    "boundary": "decoder_layer",
                    "field": field,
                    "layer_index": left["layer_index"],
                    "layer_type": left["layer_type"],
                    "module": (
                        "residual_stream"
                        if field == "output_residual_sha256"
                        else "decoder_layer_output"
                    ),
                    "serial_sha256": left_value,
                    "m8_sha256": right_value,
                }
    return None


def _detailed_layer_count(records: list[dict[str, Any]]) -> int:
    required = {field for field, _module in LAYER_DETAIL_FIELDS}
    return sum(
        isinstance(record.get("detail"), dict) and required.issubset(record["detail"])
        for record in records
    )


def _scored_quest_layer_count(records: list[dict[str, Any]]) -> int:
    return sum(
        record.get("schema") == QUEST_SCHEMA
        and isinstance(record.get("selector_score_count"), int)
        and not isinstance(record.get("selector_score_count"), bool)
        and record["selector_score_count"] > 0
        and isinstance(record.get("selector_scores_sha256"), str)
        and len(record["selector_scores_sha256"]) == 64
        for record in records
    )


def _first_quest_difference(
    serial: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any] | None:
    candidate_by_name = {record["layer_name"]: record for record in candidate}
    candidate_modes = {record.get("execution_mode") for record in candidate}
    if len(candidate_modes) != 1 or next(iter(candidate_modes)) not in {"common_q1", "m8"}:
        raise LayerDiagnosisError("candidate Quest pass has inconsistent execution modes")
    fields = (
        ("query_sha256", "query_row0_sha256", "quest_query"),
        ("current_key_sha256", "current_key_row0_sha256", "quest_current_kv"),
        ("current_value_sha256", "current_value_row0_sha256", "quest_current_kv"),
        ("selector_score_count", None, "quest_scoring"),
        ("selector_scores_sha256", None, "quest_scoring"),
        ("selected_pages", "selected_pages_row0", "quest_top96_selection"),
        ("selected_pages_sha256", "selected_pages_row0_sha256", "quest_top96_selection"),
        ("visible_union_pages_by_split", None, "quest_row_visibility"),
        ("historical_output_sha256", "historical_output_row0_sha256", "quest_historical_attention"),
        ("final_output_sha256", "final_output_row0_sha256", "quest_causal_tail_reduce"),
    )
    for left in serial:
        name = left["layer_name"]
        right = candidate_by_name.get(name)
        if right is None:
            raise LayerDiagnosisError(f"M8 Quest pass is missing layer {name}")
        if left.get("execution_mode") != "common_q1":
            raise LayerDiagnosisError("serial Quest execution mode is not common_q1")
        for field, legacy_field, module in fields:
            if legacy_field is None and (
                left.get("schema") != QUEST_SCHEMA or right.get("schema") != QUEST_SCHEMA
            ):
                continue
            left_field = field if left.get("schema") == QUEST_SCHEMA else legacy_field
            right_field = field if right.get("schema") == QUEST_SCHEMA else legacy_field
            if left_field not in left or right_field not in right:
                raise LayerDiagnosisError(f"Quest record is missing {field}")
            if left[left_field] != right[right_field]:
                difference = {
                    "boundary": "quest_attention",
                    "field": field,
                    "layer_name": name,
                    "module": module,
                    "serial_value": left[left_field],
                    "m8_value": right[right_field],
                }
                if field == "selected_pages":
                    serial_pages = left[left_field]
                    m8_pages = right[right_field]
                    if not isinstance(serial_pages, list) or not isinstance(m8_pages, list):
                        raise LayerDiagnosisError("Quest selected pages are not lists")
                    difference["serial_only_pages"] = sorted(set(serial_pages) - set(m8_pages))
                    difference["m8_only_pages"] = sorted(set(m8_pages) - set(serial_pages))
                    difference["first_order_mismatch"] = next(
                        (
                            index
                            for index, pair in enumerate(zip(serial_pages, m8_pages, strict=False))
                            if pair[0] != pair[1]
                        ),
                        min(len(serial_pages), len(m8_pages)),
                    )
                return difference
    if len(candidate_by_name) != len(serial):
        raise LayerDiagnosisError("serial and M8 Quest layer sets differ")
    return None


def _quest_layer_index(difference: dict[str, Any]) -> int:
    name = difference.get("layer_name")
    if not isinstance(name, str):
        raise LayerDiagnosisError("Quest difference has no layer name")
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match is None:
        raise LayerDiagnosisError("Quest difference layer name has no numeric model layer")
    return int(match.group(1))


def _embedded_difference_precedes_layer(
    embedded: dict[str, Any], layer: dict[str, Any] | None
) -> bool:
    if layer is None:
        return True
    embedded_layer = (
        int(embedded["layer_index"]) if "layer_index" in embedded else _quest_layer_index(embedded)
    )
    decoder_layer = int(layer["layer_index"])
    if embedded_layer != decoder_layer:
        return embedded_layer < decoder_layer
    before_core_fields = {
        "input_hidden_sha256",
        "input_residual_sha256",
        "entry_hidden_sha256",
        "entry_residual_sha256",
        "input_norm_sha256",
        "attention_qkv_projection_sha256",
        "attention_aux_projection_sha256",
    }
    return layer.get("field") not in before_core_fields


def diagnose(
    serial_path: Path,
    candidate_path: Path,
    *,
    target_position: int,
    serial_pass: int = 0,
    candidate_pass: int = 0,
    serial_quest_pass: int | None = None,
    candidate_quest_pass: int | None = None,
    serial_gdn_pass: int | None = None,
    candidate_gdn_pass: int | None = None,
    layer_indices: tuple[int, ...] | None = None,
    comparison_scope: str = "full",
    require_gdn: bool = False,
    allow_external_recurrence_state: bool = False,
) -> dict[str, Any]:
    """Compare one target position and return its earliest unequal semantic object."""

    if serial_quest_pass is None:
        serial_quest_pass = serial_pass
    if candidate_quest_pass is None:
        candidate_quest_pass = candidate_pass
    if serial_gdn_pass is None:
        serial_gdn_pass = serial_pass
    if candidate_gdn_pass is None:
        candidate_gdn_pass = candidate_pass
    if (
        target_position <= 0
        or serial_pass < 0
        or candidate_pass < 0
        or serial_quest_pass < 0
        or candidate_quest_pass < 0
        or serial_gdn_pass < 0
        or candidate_gdn_pass < 0
    ):
        raise LayerDiagnosisError(
            "position and pass indices must be nonnegative, with position positive"
        )
    selected_layer_indices = _validated_layer_indices(layer_indices)
    quest_layer_indices = tuple(layer for layer in selected_layer_indices if layer % 4 == 3)
    gdn_layer_indices = tuple(layer for layer in selected_layer_indices if layer % 4 != 3)
    if comparison_scope not in {"full", "quest"}:
        raise LayerDiagnosisError("comparison scope must be full or quest")
    if comparison_scope == "quest" and quest_layer_indices != selected_layer_indices:
        raise LayerDiagnosisError("Quest-only scope requires attention-layer indices")
    serial_records = _load_stream(serial_path, "serial layer stream")
    candidate_records = _load_stream(candidate_path, "M8 layer stream")
    serial_layers = (
        _layer_passes(serial_records, target_position, selected_layer_indices)
        if comparison_scope == "full"
        else []
    )
    candidate_layers = (
        _layer_passes(candidate_records, target_position, selected_layer_indices)
        if comparison_scope == "full"
        else []
    )
    serial_quest = _quest_passes(serial_records, target_position, quest_layer_indices)
    candidate_quest = _quest_passes(candidate_records, target_position, quest_layer_indices)
    serial_gdn = (
        _gdn_passes(
            serial_records,
            target_position,
            gdn_layer_indices,
            required=require_gdn,
        )
        if comparison_scope == "full" and gdn_layer_indices
        else []
    )
    candidate_gdn = (
        _gdn_passes(
            candidate_records,
            target_position,
            gdn_layer_indices,
            required=require_gdn,
        )
        if comparison_scope == "full" and gdn_layer_indices
        else []
    )
    try:
        left_layers = serial_layers[serial_pass] if comparison_scope == "full" else []
        right_layers = candidate_layers[candidate_pass] if comparison_scope == "full" else []
        left_quest = serial_quest[serial_quest_pass] if quest_layer_indices else []
        right_quest = candidate_quest[candidate_quest_pass] if quest_layer_indices else []
        left_gdn = serial_gdn[serial_gdn_pass] if serial_gdn else []
        right_gdn = candidate_gdn[candidate_gdn_pass] if candidate_gdn else []
    except IndexError as error:
        raise LayerDiagnosisError(
            "requested serial/M8 pass is absent from the target-position evidence"
        ) from error
    layer_difference = (
        _first_layer_difference(left_layers, right_layers) if comparison_scope == "full" else None
    )
    quest_difference = (
        _first_quest_difference(left_quest, right_quest) if quest_layer_indices else None
    )
    gdn_difference = (
        _first_gdn_difference(
            left_gdn,
            right_gdn,
            allow_external_recurrence_state=allow_external_recurrence_state,
        )
        if left_gdn or right_gdn
        else None
    )
    first_difference = layer_difference
    for embedded_difference in (gdn_difference, quest_difference):
        if embedded_difference is not None and _embedded_difference_precedes_layer(
            embedded_difference, first_difference
        ):
            first_difference = embedded_difference
    return {
        "candidate_gdn_mode": (
            (
                "m8"
                if right_gdn[0].get("schema") in {GDN_M8_SCHEMA, *LEGACY_GDN_M8_SCHEMAS}
                else "common_q1"
            )
            if right_gdn
            else None
        ),
        "candidate_gdn_pass": candidate_gdn_pass,
        "candidate_gdn_pass_count": len(candidate_gdn),
        "candidate_pass": candidate_pass,
        "candidate_pass_count": len(candidate_layers) if comparison_scope == "full" else None,
        "candidate_detailed_layer_count": (
            _detailed_layer_count(right_layers) if comparison_scope == "full" else None
        ),
        "candidate_quest_pass": candidate_quest_pass,
        "candidate_quest_pass_count": len(candidate_quest),
        "candidate_scored_quest_layer_count": _scored_quest_layer_count(right_quest),
        "candidate_quest_mode": right_quest[0]["execution_mode"] if right_quest else None,
        "comparison_scope": comparison_scope,
        "gdn_recurrence_evidence": (
            "external_state_campaign"
            if allow_external_recurrence_state
            and left_gdn
            and all(record.get("recurrent_calls") is None for record in left_gdn)
            and all(record.get("recurrent_calls") is None for record in right_gdn)
            else "captured_suboperations"
            if left_gdn
            else None
        ),
        "first_difference": first_difference,
        "passed": first_difference is None,
        "serial_pass": serial_pass,
        "serial_pass_count": len(serial_layers) if comparison_scope == "full" else None,
        "serial_detailed_layer_count": (
            _detailed_layer_count(left_layers) if comparison_scope == "full" else None
        ),
        "serial_quest_pass": serial_quest_pass,
        "serial_quest_pass_count": len(serial_quest),
        "serial_scored_quest_layer_count": _scored_quest_layer_count(left_quest),
        "serial_gdn_pass": serial_gdn_pass,
        "serial_gdn_pass_count": len(serial_gdn),
        "selected_layer_indices": list(selected_layer_indices),
        "target_position": target_position,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locate the first serial/M8 layer divergence at one target position."
    )
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--m8", type=Path, required=True)
    parser.add_argument("--target-position", type=int, required=True)
    parser.add_argument("--serial-pass", type=int, default=0)
    parser.add_argument("--m8-pass", type=int, default=0)
    parser.add_argument("--serial-quest-pass", type=int)
    parser.add_argument("--m8-quest-pass", type=int)
    parser.add_argument("--serial-gdn-pass", type=int)
    parser.add_argument("--m8-gdn-pass", type=int)
    parser.add_argument(
        "--require-gdn",
        action="store_true",
        help="fail closed unless all selected recurrent layers have GDN sub-operation records",
    )
    parser.add_argument(
        "--layer-indices",
        help="comma-separated ordered decoder layers captured in each pass (default: 0..63)",
    )
    parser.add_argument(
        "--scope",
        choices=("full", "quest"),
        default="full",
        help="compare full decoder boundaries or only the selected Quest module boundaries",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        layer_indices = (
            tuple(int(value) for value in args.layer_indices.split(","))
            if args.layer_indices
            else None
        )
        result = diagnose(
            args.serial,
            args.m8,
            target_position=args.target_position,
            serial_pass=args.serial_pass,
            candidate_pass=args.m8_pass,
            serial_quest_pass=args.serial_quest_pass,
            candidate_quest_pass=args.m8_quest_pass,
            serial_gdn_pass=args.serial_gdn_pass,
            candidate_gdn_pass=args.m8_gdn_pass,
            layer_indices=layer_indices,
            comparison_scope=args.scope,
            require_gdn=args.require_gdn,
        )
    except (LayerDiagnosisError, OSError, ValueError) as error:
        print(f"qwen-assurance-layer-diagnosis: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
# QWEN_ASSURANCE_ONLY_END: assurance-layer-diagnosis
