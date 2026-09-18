"""Seal narrowly scoped M1/M8 semantic regression counterexamples.

This module deliberately produces non-promotable evidence.  It authenticates
complete, private layer-diagnostic streams and the command/manifest files for
each arm, selects one exact record from each stream, evaluates one of two
explicit regression contracts, and writes a create-only capsule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

HEADER_SCHEMA = "qwen-r9700.m1-m8-layer-diagnostic-header.v1"
MODEL_INPUT_SCHEMA = "qwen-r9700.m1-m8-model-input.v1"
LAYER_SCHEMA = "qwen-r9700.m1-m8-layer-boundary.v1"
RECORDS_SCHEMA = "urn:qwen-r9700:m1-m8-semantic-counterexample-records:v1"
METADATA_SCHEMA = "urn:qwen-r9700:m1-m8-semantic-counterexample-metadata:v1"
COMPLETE_SCHEMA = "urn:qwen-r9700:m1-m8-semantic-counterexample-complete:v1"
CLASSIFICATION = "non_promotable_regression_counterexample"
CASE_CONTRACTS = ("full_attention_batched_k", "gdn_recurrence_source")
MAX_SOURCE_BYTES = 1 << 30

_CLAIM_SCOPE = {
    "full_attention_batched_k": {
        "causal_claim": False,
        "exhaustive_earliest_difference_claim": False,
        "interpretation": (
            "This is an observed batched-K semantic counterexample at the selected captured "
            "boundary. It does not prove that no earlier uncaptured boundary differs."
        ),
    },
    "gdn_recurrence_source": {
        "causal_claim": False,
        "independent_root_cause_claim": False,
        "interpretation": (
            "This is an observed GDN recurrence-source persistence counterexample at the "
            "selected boundary. It may be downstream of an earlier divergence and does not "
            "establish an independent root cause."
        ),
    },
}

_HEADER_KEYS = {"schema", "pid", "positions", "module_sha256"}
_MODEL_INPUT_KEYS = {
    "schema",
    "pass_index",
    "rows",
    "row",
    "position",
    "token_id",
    "inputs_embeds",
    "batched_embedding_sha256",
    "serial_embedding_sha256",
}
_LAYER_KEYS = {
    "schema",
    "pass_index",
    "layer_index",
    "layer_type",
    "rows",
    "row",
    "position",
    "input_hidden_sha256",
    "input_residual_sha256",
    "output_hidden_sha256",
    "output_residual_sha256",
    "detail",
}
_SELECTED_ARM_KEYS = {"selector", "model_input", "layer"}
_SELECTOR_KEYS = {"pass_index", "position", "layer_index"}
_SOURCE_DESCRIPTOR_KEYS = {"stream", "command", "manifest", "stream_header"}
_FILE_DESCRIPTOR_KEYS = {"path", "bytes", "sha256"}
_METADATA_KEYS = {
    "schema",
    "classification",
    "promotable",
    "case_contract",
    "claim_scope",
    "sources",
    "contract_comparisons",
    "selected_records_sha256",
    "metadata_sha256",
}
_COMPLETE_KEYS = {
    "schema",
    "classification",
    "promotable",
    "selected_records_file_sha256",
    "metadata_file_sha256",
    "metadata_sha256",
    "complete_sha256",
}


class CounterexampleError(RuntimeError):
    """Raised when evidence cannot satisfy the fail-closed contract."""


@dataclass(frozen=True)
class Selector:
    pass_index: int
    position: int
    layer_index: int


@dataclass(frozen=True)
class SourceFiles:
    stream: Path
    stream_sha256: str
    command: Path
    command_sha256: str
    manifest: Path
    manifest_sha256: str


@dataclass(frozen=True)
class _AuthenticatedFile:
    path: str
    size: int
    sha256: str
    payload: bytes


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    except (TypeError, ValueError) as error:
        raise CounterexampleError(f"value is not canonical JSON: {error}") from error


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CounterexampleError(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise CounterexampleError(f"{label} is outside its integer contract")
    return value


def _stable_private_file(path: Path, expected_sha256: str, label: str) -> _AuthenticatedFile:
    expected = _require_digest(expected_sha256, f"{label} expected digest")
    if not path.is_absolute() or path != path.resolve(strict=True) or path.is_symlink():
        raise CounterexampleError(f"{label} path must be absolute and must not traverse symlinks")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
            raise CounterexampleError(f"{label} must be an owned regular file")
        if stat.S_IMODE(before.st_mode) & 0o077:
            raise CounterexampleError(f"{label} must not grant group or other permissions")
        if before.st_size < 1 or before.st_size > MAX_SOURCE_BYTES:
            raise CounterexampleError(f"{label} size is outside the evidence contract")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise CounterexampleError(f"{label} ended before its stat-authenticated size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CounterexampleError(f"{label} grew during authentication")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise CounterexampleError(f"{label} changed during authentication")
    payload = b"".join(chunks)
    actual = _sha_bytes(payload)
    if actual != expected:
        raise CounterexampleError(f"{label} SHA-256 differs from the required digest")
    return _AuthenticatedFile(path=str(path), size=len(payload), sha256=actual, payload=payload)


def _digest_field(value: object, label: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    return _require_digest(value, label)


def _validate_header(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _HEADER_KEYS:
        raise CounterexampleError("layer stream header keys differ from the exact contract")
    if value.get("schema") != HEADER_SCHEMA:
        raise CounterexampleError("layer stream header schema differs")
    _require_int(value.get("pid"), "layer stream pid", minimum=1, maximum=(1 << 31) - 1)
    positions = value.get("positions")
    if (
        not isinstance(positions, list)
        or not positions
        or len(positions) > 64
        or any(
            isinstance(position, bool) or not isinstance(position, int) or position < 1
            for position in positions
        )
        or positions != sorted(set(positions))
    ):
        raise CounterexampleError("layer stream header positions are not canonical")
    _require_digest(value.get("module_sha256"), "layer stream module digest")
    return value


def _validate_model_input(value: object, positions: set[int]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _MODEL_INPUT_KEYS:
        raise CounterexampleError("model-input record keys differ from the exact contract")
    if value.get("schema") != MODEL_INPUT_SCHEMA:
        raise CounterexampleError("model-input record schema differs")
    _validate_coordinates(value, positions, has_layer=False)
    token_id = value.get("token_id")
    if token_id is not None:
        _require_int(token_id, "model-input token id", minimum=0, maximum=(1 << 31) - 1)
    if not isinstance(value.get("inputs_embeds"), bool):
        raise CounterexampleError("model-input inputs_embeds is not boolean")
    _digest_field(value.get("batched_embedding_sha256"), "batched embedding")
    _digest_field(value.get("serial_embedding_sha256"), "serial embedding")
    if value["batched_embedding_sha256"] != value["serial_embedding_sha256"]:
        raise CounterexampleError("model-input batched and serial embeddings differ")
    return value


def _validate_coordinates(value: dict[str, Any], positions: set[int], *, has_layer: bool) -> None:
    _require_int(value.get("pass_index"), "pass index", minimum=0, maximum=1_000_000)
    rows = _require_int(value.get("rows"), "row count", minimum=1, maximum=8)
    _require_int(value.get("row"), "row", minimum=0, maximum=rows - 1)
    position = _require_int(value.get("position"), "position", minimum=1, maximum=10_000_000)
    if position not in positions:
        raise CounterexampleError("record position is absent from the stream header")
    if has_layer:
        _require_int(value.get("layer_index"), "layer index", minimum=0, maximum=63)


def _validate_layer(value: object, positions: set[int]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _LAYER_KEYS:
        raise CounterexampleError("layer record keys differ from the exact contract")
    if value.get("schema") != LAYER_SCHEMA:
        raise CounterexampleError("layer record schema differs")
    _validate_coordinates(value, positions, has_layer=True)
    expected_type = "full_attention" if value["layer_index"] % 4 == 3 else "linear_attention"
    if value.get("layer_type") != expected_type:
        raise CounterexampleError("layer type differs from the Qwen3.8 decoder inventory")
    for field in ("input_hidden_sha256", "output_hidden_sha256", "output_residual_sha256"):
        _digest_field(value.get(field), f"layer {field}")
    _digest_field(value.get("input_residual_sha256"), "layer input residual", nullable=True)
    if not isinstance(value.get("detail"), dict) or any(
        not isinstance(key, str) or not key for key in value["detail"]
    ):
        raise CounterexampleError("layer detail is not a string-keyed object")
    _canonical(value["detail"])
    return value


def _parse_layer_stream(payload: bytes, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not payload.endswith(b"\n"):
        raise CounterexampleError(f"{label} has no durable final newline")
    raw_lines = payload.splitlines()
    if not raw_lines or any(not line for line in raw_lines):
        raise CounterexampleError(f"{label} contains a missing or blank record")
    decoded: list[object] = []
    for line_number, raw in enumerate(raw_lines, start=1):
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CounterexampleError(
                f"{label} line {line_number} is invalid JSON: {error}"
            ) from error
        if raw != _canonical(value):
            raise CounterexampleError(f"{label} line {line_number} is not canonical JSON")
        decoded.append(value)
    header = _validate_header(decoded[0])
    positions = set(header["positions"])
    records: list[dict[str, Any]] = []
    for value in decoded[1:]:
        if not isinstance(value, dict):
            raise CounterexampleError(f"{label} contains a non-object record")
        schema = value.get("schema")
        if schema == MODEL_INPUT_SCHEMA:
            records.append(_validate_model_input(value, positions))
        elif schema == LAYER_SCHEMA:
            records.append(_validate_layer(value, positions))
        else:
            raise CounterexampleError(f"{label} contains an unknown record schema")
    _validate_complete_groups(records, label)
    return header, records


def _validate_complete_groups(records: list[dict[str, Any]], label: str) -> None:
    if not records:
        raise CounterexampleError(f"{label} contains no records")
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["pass_index"], record["position"]), []).append(record)
    passes = sorted({key[0] for key in groups})
    if passes != list(range(passes[-1] + 1)):
        raise CounterexampleError(f"{label} pass indexes are not contiguous from zero")
    for (pass_index, position), group in groups.items():
        model_inputs = [record for record in group if record["schema"] == MODEL_INPUT_SCHEMA]
        layers = [record for record in group if record["schema"] == LAYER_SCHEMA]
        if len(model_inputs) != 1 or len(layers) != 64:
            raise CounterexampleError(
                f"{label} pass {pass_index} position {position} is not a complete 65-record pass"
            )
        if sorted(record["layer_index"] for record in layers) != list(range(64)):
            raise CounterexampleError(f"{label} layer inventory is missing or duplicated")
        model_input = model_inputs[0]
        identity = (model_input["rows"], model_input["row"])
        ordered = sorted(layers, key=lambda record: record["layer_index"])
        if any((record["rows"], record["row"]) != identity for record in ordered):
            raise CounterexampleError(f"{label} row identity changes within one decoder pass")
        if ordered[0]["input_hidden_sha256"] != model_input["batched_embedding_sha256"]:
            raise CounterexampleError(f"{label} model input does not bind layer zero")
        for previous, current in pairwise(ordered):
            if (
                previous["output_hidden_sha256"] != current["input_hidden_sha256"]
                or previous["output_residual_sha256"] != current["input_residual_sha256"]
            ):
                raise CounterexampleError(f"{label} layer boundary chain is discontinuous")


def _select(
    records: list[dict[str, Any]], selector: Selector, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = [
        record
        for record in records
        if record["schema"] == LAYER_SCHEMA
        and record["pass_index"] == selector.pass_index
        and record["position"] == selector.position
        and record["layer_index"] == selector.layer_index
    ]
    if len(matches) != 1:
        raise CounterexampleError(
            f"{label} selector matched {len(matches)} layer records, expected one"
        )
    model_inputs = [
        record
        for record in records
        if record["schema"] == MODEL_INPUT_SCHEMA
        and record["pass_index"] == selector.pass_index
        and record["position"] == selector.position
    ]
    if len(model_inputs) != 1:
        raise CounterexampleError(f"{label} selector has no unique model-input record")
    return model_inputs[0], matches[0]


def _detail_digest(record: dict[str, Any], field: str, label: str) -> str:
    if field not in record["detail"]:
        raise CounterexampleError(f"{label} is missing required detail field {field}")
    return _require_digest(record["detail"][field], f"{label} {field}")


def _compare(
    semantic_field: str,
    target: str | None,
    m8: str | None,
    *,
    equal: bool,
) -> dict[str, Any]:
    if (target == m8) != equal:
        relation = "equal" if equal else "different"
        raise CounterexampleError(f"contract requires {semantic_field} to be {relation}")
    return {
        "field": semantic_field,
        "relation": "equal" if equal else "different",
        "target_m1": target,
        "production_m8": m8,
    }


def _full_attention_contract(
    target: dict[str, Any], m8: dict[str, Any], qk_ref: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if target["layer_type"] != "full_attention" or m8["layer_type"] != "full_attention":
        raise CounterexampleError("full-attention contract selected a non-full-attention layer")
    if (target["rows"], target["row"]) != (1, 0) or m8["rows"] != 8:
        raise CounterexampleError("full-attention contract requires target M1 and production M8")
    if qk_ref is not None and qk_ref["rows"] != 8:
        raise CounterexampleError("full-attention auxiliary QK reference is not an M8 record")
    pairs: list[dict[str, Any]] = [
        _compare(field, target[field], m8[field], equal=True)
        for field in ("input_hidden_sha256", "input_residual_sha256")
    ]
    for semantic, detail_field in (
        ("qkv", "full_attention_qkv_projection_sha256"),
        ("projected_q", "full_attention_projected_q_sha256"),
        ("projected_v", "full_attention_projected_v_sha256"),
        ("projected_gate", "full_attention_projected_gate_sha256"),
    ):
        pairs.append(
            _compare(
                semantic,
                _detail_digest(target, detail_field, "target M1"),
                _detail_digest(m8, detail_field, "production M8"),
                equal=True,
            )
        )
    target_k = _detail_digest(target, "full_attention_projected_k_sha256", "target M1")
    m8_k = _detail_digest(m8, "full_attention_projected_k_sha256", "production M8")
    pairs.append(_compare("projected_k_batched", target_k, m8_k, equal=False))
    serial_record = m8 if qk_ref is None else qk_ref
    if qk_ref is not None:
        pairs.extend(
            _compare(f"qk_ref_{field}", m8[field], qk_ref[field], equal=True)
            for field in ("input_hidden_sha256", "input_residual_sha256")
        )
        for semantic, detail_field in (
            ("qkv", "full_attention_qkv_projection_sha256"),
            ("projected_q", "full_attention_projected_q_sha256"),
            ("projected_k", "full_attention_projected_k_sha256"),
            ("projected_v", "full_attention_projected_v_sha256"),
            ("projected_gate", "full_attention_projected_gate_sha256"),
        ):
            pairs.append(
                _compare(
                    f"qk_ref_{semantic}",
                    _detail_digest(m8, detail_field, "production M8"),
                    _detail_digest(qk_ref, detail_field, "auxiliary QK reference"),
                    equal=True,
                )
            )
    serial_k = _detail_digest(
        serial_record, "full_attention_serial_k_sha256", "M8 serial QK reference"
    )
    pairs.append(_compare("projected_k_serial", target_k, serial_k, equal=True))
    return pairs


def _gdn_contract(target: dict[str, Any], m8: dict[str, Any]) -> list[dict[str, Any]]:
    if target["layer_type"] != "linear_attention" or m8["layer_type"] != "linear_attention":
        raise CounterexampleError("GDN contract selected a non-GDN layer")
    if (target["rows"], target["row"]) != (1, 0) or m8["rows"] != 8:
        raise CounterexampleError("GDN contract requires target M1 and production M8")
    pairs: list[dict[str, Any]] = [
        _compare(field, target[field], m8[field], equal=True)
        for field in ("input_hidden_sha256", "input_residual_sha256")
    ]
    for semantic, target_field, m8_field in (
        ("qkv", "gdn_non_spec_mixed_qkv_sha256", "gdn_spec_entry_mixed_qkv_sha256"),
        ("a", "gdn_non_spec_a_sha256", "gdn_spec_entry_a_sha256"),
        ("b", "gdn_non_spec_b_sha256", "gdn_spec_entry_b_sha256"),
        (
            "postconv",
            "gdn_reference_postconv_mixed_qkv_sha256",
            "gdn_postconv_entry_mixed_qkv_sha256",
        ),
    ):
        pairs.append(
            _compare(
                semantic,
                _detail_digest(target, target_field, "target M1"),
                _detail_digest(m8, m8_field, "production M8"),
                equal=True,
            )
        )
    for semantic, suffix in (
        ("recurrence_source_state_index", "state_index"),
        ("recurrence_source_conv_state", "conv_state_sha256"),
        ("recurrence_source_ssm_state", "ssm_state_sha256"),
    ):
        target_field = f"gdn_non_spec_source_{suffix}"
        m8_field = f"gdn_spec_recurrence_source_{suffix}"
        target_value = target["detail"].get(target_field)
        m8_value = m8["detail"].get(m8_field)
        if suffix != "state_index":
            target_value = _detail_digest(target, target_field, "target M1")
            m8_value = _detail_digest(m8, m8_field, "production M8")
        elif (
            not isinstance(target_value, str)
            or not target_value.isdecimal()
            or not isinstance(m8_value, str)
            or not m8_value.isdecimal()
        ):
            raise CounterexampleError("GDN recurrence source index is absent or non-canonical")
        pairs.append(_compare(semantic, target_value, m8_value, equal=False))
    pairs.append(
        _compare(
            "recurrent_output",
            _detail_digest(target, "gdn_non_spec_recurrent_output_sha256", "target M1"),
            _detail_digest(m8, "gdn_spec_normalized_output_sha256", "production M8"),
            equal=False,
        )
    )
    pairs.append(
        _compare(
            "final_normalized_output",
            _detail_digest(target, "linear_attention_output_projection_input_sha256", "target M1"),
            _detail_digest(m8, "gdn_postconv_exit_normalized_output_sha256", "production M8"),
            equal=False,
        )
    )
    return pairs


def _authenticate_source(
    source: SourceFiles, label: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    stream = _stable_private_file(source.stream, source.stream_sha256, f"{label} stream")
    command = _stable_private_file(source.command, source.command_sha256, f"{label} command")
    manifest = _stable_private_file(source.manifest, source.manifest_sha256, f"{label} manifest")
    header, records = _parse_layer_stream(stream.payload, f"{label} stream")
    descriptor = {
        "stream": {"path": stream.path, "bytes": stream.size, "sha256": stream.sha256},
        "command": {"path": command.path, "bytes": command.size, "sha256": command.sha256},
        "manifest": {"path": manifest.path, "bytes": manifest.size, "sha256": manifest.sha256},
        "stream_header": header,
    }
    return header, records, descriptor


def _private_new_root(root: Path) -> Path:
    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or root.exists()
        or root.is_symlink()
    ):
        raise CounterexampleError("capsule root must be a new absolute path without symlinks")
    parent = root.parent.resolve(strict=True)
    if parent != root.parent or not parent.is_dir() or parent.is_symlink():
        raise CounterexampleError("capsule parent path is unsafe")
    status = parent.stat()
    if status.st_uid != os.getuid() or stat.S_IMODE(status.st_mode) != 0o700:
        raise CounterexampleError("capsule parent must be owned mode 0700")
    return parent


def _write_create_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _read_sealed_file(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_size < 1
            or before.st_size > MAX_SOURCE_BYTES
        ):
            raise CounterexampleError(f"{label} is not a sealed owned mode-0400 file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise CounterexampleError(f"{label} ended during authentication")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CounterexampleError(f"{label} grew during authentication")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_mode,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_mode,
    )
    if before_identity != after_identity:
        raise CounterexampleError(f"{label} changed during authentication")
    return b"".join(chunks)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal_counterexample(
    root: Path,
    *,
    case_contract: str,
    target_source: SourceFiles,
    target_selector: Selector,
    m8_source: SourceFiles,
    m8_selector: Selector,
    qk_ref_source: SourceFiles | None = None,
    qk_ref_selector: Selector | None = None,
) -> dict[str, Any]:
    """Validate evidence and durably write one non-promotable capsule."""

    if case_contract not in CASE_CONTRACTS:
        raise CounterexampleError("unknown M1/M8 semantic counterexample contract")
    if (qk_ref_source is None) != (qk_ref_selector is None):
        raise CounterexampleError(
            "auxiliary QK reference source and selector must be supplied together"
        )
    if case_contract != "full_attention_batched_k" and qk_ref_source is not None:
        raise CounterexampleError("auxiliary QK reference is valid only for full attention")
    parent = _private_new_root(root)
    _target_header, target_records, target_descriptor = _authenticate_source(
        target_source, "target M1"
    )
    _m8_header, m8_records, m8_descriptor = _authenticate_source(m8_source, "production M8")
    target_input, target = _select(target_records, target_selector, "target M1")
    m8_input, m8 = _select(m8_records, m8_selector, "production M8")
    if (target_selector.position, target_selector.layer_index) != (
        m8_selector.position,
        m8_selector.layer_index,
    ):
        raise CounterexampleError("target and M8 selectors do not name the same absolute boundary")
    qk_ref_input: dict[str, Any] | None = None
    qk_ref: dict[str, Any] | None = None
    qk_ref_descriptor: dict[str, Any] | None = None
    if qk_ref_source is not None and qk_ref_selector is not None:
        _qk_header, qk_records, qk_ref_descriptor = _authenticate_source(
            qk_ref_source, "auxiliary QK reference"
        )
        qk_ref_input, qk_ref = _select(qk_records, qk_ref_selector, "auxiliary QK reference")
        if (qk_ref_selector.position, qk_ref_selector.layer_index) != (
            m8_selector.position,
            m8_selector.layer_index,
        ):
            raise CounterexampleError("auxiliary QK selector names a different absolute boundary")
    comparisons = (
        _full_attention_contract(target, m8, qk_ref)
        if case_contract == "full_attention_batched_k"
        else _gdn_contract(target, m8)
    )
    selected_records: dict[str, Any] = {
        "schema": RECORDS_SCHEMA,
        "case_contract": case_contract,
        "target_m1": {
            "selector": target_selector.__dict__,
            "model_input": target_input,
            "layer": target,
        },
        "production_m8": {
            "selector": m8_selector.__dict__,
            "model_input": m8_input,
            "layer": m8,
        },
    }
    sources: dict[str, Any] = {
        "target_m1": target_descriptor,
        "production_m8": m8_descriptor,
    }
    if qk_ref is not None and qk_ref_input is not None and qk_ref_descriptor is not None:
        selected_records["auxiliary_qk_reference"] = {
            "selector": qk_ref_selector.__dict__,
            "model_input": qk_ref_input,
            "layer": qk_ref,
        }
        sources["auxiliary_qk_reference"] = qk_ref_descriptor
    records_payload = _canonical(selected_records) + b"\n"
    metadata: dict[str, Any] = {
        "schema": METADATA_SCHEMA,
        "classification": CLASSIFICATION,
        "promotable": False,
        "case_contract": case_contract,
        "claim_scope": _CLAIM_SCOPE[case_contract],
        "sources": sources,
        "contract_comparisons": comparisons,
        "selected_records_sha256": _sha_bytes(records_payload),
    }
    metadata["metadata_sha256"] = _sha_bytes(_canonical(metadata))
    metadata_payload = _canonical(metadata) + b"\n"
    root.mkdir(mode=0o700)
    try:
        _write_create_only(root / "selected-records.json", records_payload)
        _write_create_only(root / "metadata.json", metadata_payload)
        complete: dict[str, Any] = {
            "schema": COMPLETE_SCHEMA,
            "classification": CLASSIFICATION,
            "promotable": False,
            "selected_records_file_sha256": _sha_bytes(records_payload),
            "metadata_file_sha256": _sha_bytes(metadata_payload),
            "metadata_sha256": metadata["metadata_sha256"],
        }
        complete["complete_sha256"] = _sha_bytes(_canonical(complete))
        _write_create_only(root / "complete.json", _canonical(complete) + b"\n")
        _fsync_directory(root)
        _fsync_directory(parent)
    except BaseException:
        # A partial create-only directory remains visibly incomplete for forensic use.
        raise
    return complete


def _loaded_selector(value: object, label: str) -> Selector:
    if not isinstance(value, dict) or set(value) != _SELECTOR_KEYS:
        raise CounterexampleError(f"{label} selector keys differ")
    return Selector(
        pass_index=_require_int(
            value["pass_index"], f"{label} pass index", minimum=0, maximum=1_000_000
        ),
        position=_require_int(
            value["position"], f"{label} position", minimum=1, maximum=10_000_000
        ),
        layer_index=_require_int(value["layer_index"], f"{label} layer", minimum=0, maximum=63),
    )


def _loaded_arm(value: object, label: str) -> tuple[Selector, dict[str, Any], dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != _SELECTED_ARM_KEYS:
        raise CounterexampleError(f"{label} selected-record keys differ")
    selector = _loaded_selector(value["selector"], label)
    model_input = _validate_model_input(value["model_input"], {selector.position})
    layer = _validate_layer(value["layer"], {selector.position})
    if (
        model_input["pass_index"] != selector.pass_index
        or layer["pass_index"] != selector.pass_index
        or layer["layer_index"] != selector.layer_index
        or model_input["position"] != selector.position
        or layer["position"] != selector.position
        or (model_input["rows"], model_input["row"]) != (layer["rows"], layer["row"])
    ):
        raise CounterexampleError(f"{label} selected records do not bind their selector")
    return selector, model_input, layer


def _validate_loaded_source(value: object, label: str) -> None:
    if not isinstance(value, dict) or set(value) != _SOURCE_DESCRIPTOR_KEYS:
        raise CounterexampleError(f"{label} source descriptor keys differ")
    _validate_header(value["stream_header"])
    for kind in ("stream", "command", "manifest"):
        descriptor = value[kind]
        if not isinstance(descriptor, dict) or set(descriptor) != _FILE_DESCRIPTOR_KEYS:
            raise CounterexampleError(f"{label} {kind} descriptor keys differ")
        if not isinstance(descriptor["path"], str) or not Path(descriptor["path"]).is_absolute():
            raise CounterexampleError(f"{label} {kind} path is not absolute")
        _require_int(
            descriptor["bytes"], f"{label} {kind} bytes", minimum=1, maximum=MAX_SOURCE_BYTES
        )
        _require_digest(descriptor["sha256"], f"{label} {kind} digest")


def _validate_loaded_semantics(records: dict[str, Any], metadata: dict[str, Any]) -> None:
    case_contract = records.get("case_contract")
    if case_contract not in CASE_CONTRACTS or metadata.get("case_contract") != case_contract:
        raise CounterexampleError("counterexample case contract differs")
    expected_record_keys = {"schema", "case_contract", "target_m1", "production_m8"}
    has_qk_ref = "auxiliary_qk_reference" in records
    if has_qk_ref:
        expected_record_keys.add("auxiliary_qk_reference")
    if set(records) != expected_record_keys:
        raise CounterexampleError("counterexample selected-record keys differ")
    target_selector, _target_input, target = _loaded_arm(records["target_m1"], "target M1")
    m8_selector, _m8_input, m8 = _loaded_arm(records["production_m8"], "production M8")
    if (target_selector.position, target_selector.layer_index) != (
        m8_selector.position,
        m8_selector.layer_index,
    ):
        raise CounterexampleError("loaded target and M8 selectors name different boundaries")
    qk_ref: dict[str, Any] | None = None
    if has_qk_ref:
        qk_selector, _qk_input, qk_ref = _loaded_arm(
            records["auxiliary_qk_reference"], "auxiliary QK reference"
        )
        if (qk_selector.position, qk_selector.layer_index) != (
            m8_selector.position,
            m8_selector.layer_index,
        ):
            raise CounterexampleError("loaded auxiliary QK selector names a different boundary")
    comparisons = (
        _full_attention_contract(target, m8, qk_ref)
        if case_contract == "full_attention_batched_k"
        else _gdn_contract(target, m8)
    )
    if metadata.get("contract_comparisons") != comparisons:
        raise CounterexampleError("counterexample comparisons do not bind the selected records")
    sources = metadata.get("sources")
    expected_source_roles = {"target_m1", "production_m8"}
    if has_qk_ref:
        expected_source_roles.add("auxiliary_qk_reference")
    if not isinstance(sources, dict) or set(sources) != expected_source_roles:
        raise CounterexampleError("counterexample source roles differ")
    for role, descriptor in sources.items():
        _validate_loaded_source(descriptor, role)


def load_counterexample(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate a completed capsule and reject every unknown artifact."""

    if not root.is_absolute() or root != root.resolve(strict=True) or root.is_symlink():
        raise CounterexampleError("counterexample root is unsafe or absent")
    status = root.stat()
    if not root.is_dir() or status.st_uid != os.getuid() or stat.S_IMODE(status.st_mode) != 0o700:
        raise CounterexampleError("counterexample root is not an owned mode-0700 directory")
    root_identity = (
        status.st_dev,
        status.st_ino,
        status.st_mtime_ns,
        status.st_ctime_ns,
        status.st_mode,
    )
    expected_names = {"selected-records.json", "metadata.json", "complete.json"}
    if {path.name for path in root.iterdir()} != expected_names:
        raise CounterexampleError("counterexample capsule contains missing or unknown artifacts")
    payloads: dict[str, bytes] = {}
    for name in expected_names:
        payloads[name] = _read_sealed_file(root / name, f"counterexample artifact {name}")
    final_status = root.stat()
    final_identity = (
        final_status.st_dev,
        final_status.st_ino,
        final_status.st_mtime_ns,
        final_status.st_ctime_ns,
        final_status.st_mode,
    )
    if root_identity != final_identity or {path.name for path in root.iterdir()} != expected_names:
        raise CounterexampleError("counterexample capsule changed during authentication")
    decoded: dict[str, dict[str, Any]] = {}
    for name, payload in payloads.items():
        if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
            raise CounterexampleError(f"counterexample artifact {name} is not one canonical record")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CounterexampleError(
                f"counterexample artifact {name} is invalid: {error}"
            ) from error
        if not isinstance(value, dict) or payload != _canonical(value) + b"\n":
            raise CounterexampleError(f"counterexample artifact {name} is not canonical")
        decoded[name] = value
    records = decoded["selected-records.json"]
    metadata = decoded["metadata.json"]
    complete = decoded["complete.json"]
    if records.get("schema") != RECORDS_SCHEMA:
        raise CounterexampleError("counterexample selected-record schema differs")
    if (
        set(metadata) != _METADATA_KEYS
        or set(complete) != _COMPLETE_KEYS
        or metadata.get("schema") != METADATA_SCHEMA
        or complete.get("schema") != COMPLETE_SCHEMA
    ):
        raise CounterexampleError("counterexample seal schema differs")
    unsigned_metadata = dict(metadata)
    metadata_sha256 = unsigned_metadata.pop("metadata_sha256", None)
    if metadata_sha256 != _sha_bytes(_canonical(unsigned_metadata)):
        raise CounterexampleError("counterexample metadata self-hash differs")
    unsigned_complete = dict(complete)
    complete_sha256 = unsigned_complete.pop("complete_sha256", None)
    if complete_sha256 != _sha_bytes(_canonical(unsigned_complete)):
        raise CounterexampleError("counterexample completion self-hash differs")
    if (
        metadata.get("classification") != CLASSIFICATION
        or metadata.get("promotable") is not False
        or complete.get("classification") != CLASSIFICATION
        or complete.get("promotable") is not False
        or metadata.get("selected_records_sha256") != _sha_bytes(payloads["selected-records.json"])
        or complete.get("selected_records_file_sha256")
        != _sha_bytes(payloads["selected-records.json"])
        or complete.get("metadata_file_sha256") != _sha_bytes(payloads["metadata.json"])
        or complete.get("metadata_sha256") != metadata_sha256
    ):
        raise CounterexampleError("counterexample completion bindings differ")
    if metadata.get("claim_scope") != _CLAIM_SCOPE.get(metadata.get("case_contract")):
        raise CounterexampleError("counterexample claim scope differs")
    _validate_loaded_semantics(records, metadata)
    return records, metadata, complete


def _parse_selector(raw: str) -> Selector:
    try:
        values = tuple(int(item) for item in raw.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("selector must be PASS:POSITION:LAYER") from error
    if len(values) != 3 or values[0] < 0 or values[1] < 1 or not 0 <= values[2] < 64:
        raise argparse.ArgumentTypeError("selector is outside PASS:POSITION:LAYER bounds")
    return Selector(*values)


def _add_source_arguments(parser: argparse.ArgumentParser, prefix: str, *, required: bool) -> None:
    option_prefix = prefix.replace("_", "-")
    for kind in ("stream", "command", "manifest"):
        parser.add_argument(f"--{option_prefix}-{kind}", type=Path, required=required)
        parser.add_argument(f"--{option_prefix}-{kind}-sha256", required=required)
    parser.add_argument(f"--{option_prefix}-selector", type=_parse_selector, required=required)


def _source_from_args(args: argparse.Namespace, prefix: str) -> SourceFiles | None:
    values = [
        getattr(args, f"{prefix}_{kind}")
        for kind in (
            "stream",
            "stream_sha256",
            "command",
            "command_sha256",
            "manifest",
            "manifest_sha256",
        )
    ]
    if not any(value is not None for value in values):
        return None
    if any(value is None for value in values):
        raise CounterexampleError(f"{prefix} source arguments must be supplied as one complete set")
    return SourceFiles(*values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seal one explicit, permanently non-promotable M1/M8 regression counterexample."
        ),
        epilog=(
            "Selectors are PASS:POSITION:LAYER. POSITION is the absolute decoder position stored "
            "in the layer stream, not a response-relative output index."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-contract", choices=CASE_CONTRACTS, required=True)
    _add_source_arguments(parser, "target", required=True)
    _add_source_arguments(parser, "m8", required=True)
    _add_source_arguments(parser, "qk_ref", required=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        target = _source_from_args(args, "target")
        m8 = _source_from_args(args, "m8")
        qk_ref = _source_from_args(args, "qk_ref")
        assert target is not None and m8 is not None
        complete = seal_counterexample(
            args.output,
            case_contract=args.case_contract,
            target_source=target,
            target_selector=args.target_selector,
            m8_source=m8,
            m8_selector=args.m8_selector,
            qk_ref_source=qk_ref,
            qk_ref_selector=args.qk_ref_selector,
        )
    except (CounterexampleError, OSError) as error:
        print(f"qwen-m1-m8-semantic-counterexample: {error}", file=sys.stderr)
        return 2
    print(_canonical(complete).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
