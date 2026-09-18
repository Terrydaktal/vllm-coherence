"""Fail-closed comparison for post-proposal serial-DFlash and M8 state streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

STREAM_SCHEMA = "urn:qwen-r9700:full-state-diagnostic-stream:v1"
EVENT_SCHEMA = "urn:qwen-r9700:full-state-diagnostic-event:v1"
STATE_SCHEMA = "urn:qwen-r9700:coding-turbo-authoritative-state-export:v1"
COMPARISON_SCHEMA = "urn:qwen-r9700:full-state-diagnostic-comparison:v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_FINAL_COUNTS = (8, 322, 512)
STATE_LIST_LENGTHS = {
    "canonical_gdn_layers": 48,
    "convolution_layers": 48,
    "device_error_words": 48,
    "draft_kv_pages": 5,
    "draft_kv_scales": 5,
    "target_kv_pages": 16,
    "target_kv_scales": 16,
}


class StateComparisonError(RuntimeError):
    """A capture or identity violated the comparison contract."""


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_private(path: Path, label: str) -> bytes:
    path = path.expanduser().absolute()
    before = path.lstat()
    payload = path.read_bytes()
    after = path.lstat()
    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    if (
        identity(before) != identity(after)
        or not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o077
    ):
        raise StateComparisonError(f"{label} must be one stable private owned regular file")
    return payload


def _documents(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StateComparisonError(f"{label} is not UTF-8: {error}") from error
    decoder = json.JSONDecoder()
    documents: list[dict[str, Any]] = []
    offset = 0
    while True:
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        try:
            document, offset = decoder.raw_decode(text, offset)
        except json.JSONDecodeError as error:
            raise StateComparisonError(f"{label} contains malformed JSON: {error}") from error
        if not isinstance(document, dict):
            raise StateComparisonError(f"{label} contains a non-object document")
        documents.append(document)
    if not documents:
        raise StateComparisonError(f"{label} is empty")
    return documents


def _descriptor(value: object, label: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"name", "nonfinite_count", "tensor_sha256"}
        or not isinstance(value["name"], str)
        or not value["name"]
        or type(value["nonfinite_count"]) is not int
        or value["nonfinite_count"] != 0
        or SHA256_RE.fullmatch(value["tensor_sha256"]) is None
    ):
        raise StateComparisonError(f"{label} tensor descriptor is invalid or nonfinite")


def _validate_state(state: object, label: str) -> dict[str, Any]:
    required = {"cache_mapping", "rollback_generation", "schema", *STATE_LIST_LENGTHS}
    if not isinstance(state, dict) or set(state) != required or state.get("schema") != STATE_SCHEMA:
        raise StateComparisonError(f"{label} state schema is invalid")
    _descriptor(state["cache_mapping"], f"{label} cache mapping")
    if type(state["rollback_generation"]) is not int or state["rollback_generation"] < 0:
        raise StateComparisonError(f"{label} rollback generation is invalid")
    for name, expected_length in STATE_LIST_LENGTHS.items():
        value = state[name]
        if not isinstance(value, list) or len(value) != expected_length:
            raise StateComparisonError(f"{label} {name} length is invalid")
        if name == "device_error_words":
            if any(type(word) is not int or word != 0 for word in value):
                raise StateComparisonError(f"{label} contains a device error")
        else:
            for index, descriptor in enumerate(value):
                _descriptor(descriptor, f"{label} {name}[{index}]")
    return state


def _load_stream(
    path: Path, label: str
) -> tuple[bytes, tuple[int, ...], dict[tuple[int, int], dict[str, Any]]]:
    payload = _stable_private(path, label)
    documents = _documents(payload, label)
    if set(documents[0]) != {"header"} or not isinstance(documents[0]["header"], dict):
        raise StateComparisonError(f"{label} lacks one leading header")
    header = documents[0]["header"]
    counts_value = header.get("counts")
    if (
        header.get("schema") != STREAM_SCHEMA
        or not isinstance(counts_value, list)
        or not counts_value
        or any(type(count) is not int or count <= 0 for count in counts_value)
        or counts_value != sorted(set(counts_value))
    ):
        raise StateComparisonError(f"{label} header contract is invalid")
    final_counts = tuple(counts_value)

    request_ordinals: dict[str, int] = {}
    events: dict[tuple[int, int], dict[str, Any]] = {}
    maxima: dict[int, int] = {}
    for index, document in enumerate(documents[1:], start=1):
        if set(document) != {"event"} or not isinstance(document["event"], dict):
            raise StateComparisonError(f"{label} document {index} is not one event")
        event = document["event"]
        event_without_digest = dict(event)
        event_digest = event_without_digest.pop("event_sha256", None)
        if event_digest != _digest(_canonical(event_without_digest)):
            raise StateComparisonError(f"{label} event {index} digest differs")
        required = {
            "committed_token_count",
            "committed_width",
            "event_sha256",
            "request_id",
            "schema",
            "speculative_step",
            "state",
        }
        if set(event) != required or event.get("schema") != EVENT_SCHEMA:
            raise StateComparisonError(f"{label} event {index} schema is invalid")
        request_id = event["request_id"]
        count = event["committed_token_count"]
        width = event["committed_width"]
        if (
            not isinstance(request_id, str)
            or not request_id
            or type(count) is not int
            or count not in final_counts
            or type(width) is not int
            or not 1 <= width <= 8
            or type(event["speculative_step"]) is not bool
        ):
            raise StateComparisonError(f"{label} event {index} transition metadata is invalid")
        ordinal = request_ordinals.setdefault(request_id, len(request_ordinals))
        coordinate = (ordinal, count)
        if coordinate in events:
            raise StateComparisonError(f"{label} duplicates event coordinate {coordinate}")
        events[coordinate] = _validate_state(event["state"], f"{label} event {index}")
        maxima[ordinal] = max(maxima.get(ordinal, 0), count)

    # Two authenticated capture shapes are supported:
    #
    # * a campaign containing one request per declared final count; and
    # * one request observed at every declared intermediate boundary.
    #
    # The runtime capture has always been able to emit the latter, but the
    # comparator previously interpreted every header count as a separate
    # request and rejected a valid single-request first-divergence trace.
    if len(request_ordinals) == 1:
        expected_coordinates = {(0, count) for count in final_counts}
        if set(events) != expected_coordinates:
            raise StateComparisonError(
                f"{label} single-request boundary coverage is incomplete or out of order"
            )
        final_events = dict(events)
    else:
        if set(maxima) != set(range(len(final_counts))) or tuple(
            maxima[ordinal] for ordinal in range(len(final_counts))
        ) != final_counts:
            raise StateComparisonError(f"{label} event coverage is incomplete or out of order")
        final_events = {
            (ordinal, final_count): events[(ordinal, final_count)]
            for ordinal, final_count in enumerate(final_counts)
        }
        if any(count > final_counts[ordinal] for ordinal, count in events):
            raise StateComparisonError(f"{label} contains an event beyond its request boundary")
    return payload, final_counts, final_events


def _first_difference(left: object, right: object, path: str = "state") -> dict[str, object] | None:
    if type(left) is not type(right):
        return {"path": path, "serial": type(left).__name__, "m8": type(right).__name__}
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return {"path": f"{path}.{key}", "serial": left.get(key), "m8": right.get(key)}
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": f"{path}.length", "serial": len(left), "m8": len(right)}
        for index, (serial, m8) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(serial, m8, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return {"path": path, "serial": left, "m8": right}
    return None


def compare(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    serial_payload, serial_counts, serial_events = _load_stream(args.serial, "serial stream")
    m8_payload, m8_counts, m8_events = _load_stream(args.m8, "M8 stream")
    if serial_counts != m8_counts:
        raise StateComparisonError("serial and M8 stream boundary declarations differ")
    if set(serial_events) != set(m8_events):
        raise StateComparisonError("serial and M8 event coverage differs")
    identities: dict[str, dict[str, str]] = {}
    for name, path, expected in (
        ("serial_command", args.serial_command, args.serial_command_sha256),
        ("serial_manifest", args.serial_manifest, args.serial_manifest_sha256),
        ("m8_command", args.m8_command, args.m8_command_sha256),
        ("m8_manifest", args.m8_manifest, args.m8_manifest_sha256),
    ):
        if SHA256_RE.fullmatch(expected) is None:
            raise StateComparisonError(f"{name} expected digest is malformed")
        payload = _stable_private(path, name)
        if _digest(payload) != expected:
            raise StateComparisonError(f"{name} digest differs")
        identities[name] = {"path": str(path.expanduser().absolute()), "sha256": expected}

    comparisons: list[dict[str, Any]] = []
    first_difference: dict[str, object] | None = None
    for ordinal, count in sorted(serial_events):
        serial_state = serial_events[(ordinal, count)]
        m8_state = m8_events[(ordinal, count)]
        difference = _first_difference(serial_state, m8_state)
        record = {
            "committed_token_count": count,
            "equal": difference is None,
            "first_difference": difference,
            "request_ordinal": ordinal,
            "serial_state_sha256": _digest(_canonical(serial_state)),
            "m8_state_sha256": _digest(_canonical(m8_state)),
        }
        comparisons.append(record)
        if first_difference is None and difference is not None:
            first_difference = {
                **difference,
                "request_ordinal": ordinal,
                "committed_token_count": count,
            }
    equal = first_difference is None
    result = {
        "schema": COMPARISON_SCHEMA,
        "classification": "bounded_post_proposal_complete_state_comparison",
        "declared_final_counts": list(serial_counts),
        "promotable": False,
        "universal_equivalence_claim": False,
        "equal": equal,
        "first_difference": first_difference,
        "comparisons": comparisons,
        "inputs": {
            "serial_stream": {
                "path": str(args.serial.absolute()),
                "sha256": _digest(serial_payload),
            },
            "m8_stream": {"path": str(args.m8.absolute()), "sha256": _digest(m8_payload)},
            **identities,
        },
    }
    result["comparison_sha256"] = _digest(_canonical(result))
    return result, equal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-full-state-diagnostic-compare")
    parser.add_argument("--serial", required=True, type=Path)
    parser.add_argument("--m8", required=True, type=Path)
    parser.add_argument("--serial-command", required=True, type=Path)
    parser.add_argument("--serial-command-sha256", required=True)
    parser.add_argument("--serial-manifest", required=True, type=Path)
    parser.add_argument("--serial-manifest-sha256", required=True)
    parser.add_argument("--m8-command", required=True, type=Path)
    parser.add_argument("--m8-command-sha256", required=True)
    parser.add_argument("--m8-manifest", required=True, type=Path)
    parser.add_argument("--m8-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        output = args.output.expanduser().absolute()
        if output.exists() or output.is_symlink():
            raise StateComparisonError("comparison output is create-only")
        result, equal = compare(args)
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, _canonical(result))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if equal else 1
    except (OSError, StateComparisonError, ValueError) as error:
        print(f"qwen-full-state-diagnostic-compare: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
