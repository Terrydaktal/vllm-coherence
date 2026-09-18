# QWEN_ASSURANCE_ONLY_BEGIN: coding-turbo-state-capture
"""Reduce commit-time assurance evidence into a Quest96 target oracle.

The rejection sampler observes token decisions before hybrid state postprocessing.
Those decisions are therefore joined only with a second, commit-time stream emitted
after target KV and all recurrent state have reached their authoritative state.
The reducer rejects missing, duplicated, reordered, or loosely correlated records.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from qwen_r9700_lab import coding_turbo_oracle as oracle

STATE_HEADER_SCHEMA = "urn:qwen-r9700:coding-turbo-state-capture-header:v1"
STATE_EVENT_SCHEMA = "urn:qwen-r9700:coding-turbo-state-commit:v1"
ROUND_SCHEMA = "qwen-r9700.dflash-lossless-round.v4"
TARGET_ONLY_ROUND_SCHEMA = "qwen-r9700.target-only-round.v2"
ROUND_HEADER_SCHEMA = "qwen-r9700.dflash-lossless-capture-header.v1"
PUBLIC_RESULT_SCHEMA = "qwen-r9700.dflash-lossless-public-result.v1"
MAX_STREAM_BYTES = 256 * 1024 * 1024

STATE_EVENT_KEYS = {
    "accepted_draft_count",
    "cache_mapping_sha256",
    "canonical_gdn_layer_sha256",
    "committed_token_count",
    "convolution_layer_sha256",
    "device_error_word",
    "draft_kv_scales_sha256",
    "draft_kv_sha256",
    "event_sha256",
    "logical_length",
    "nonfinite_count",
    "payload_producer_receipt_sha256",
    "request_id",
    "rollback_generation",
    "round_index",
    "schema",
    "target_kv_scales_sha256",
    "target_kv_sha256",
}


class CaptureSafetyError(RuntimeError):
    """The evidence streams cannot prove an authoritative committed state."""


def _canonical_line(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _event_digest(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("event_sha256", None)
    return _sha256(_canonical_line(unsigned))


def _require_integer(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise CaptureSafetyError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _require_sha256(value: object, label: str) -> str:
    try:
        result = oracle._require_sha256(value, label)
    except oracle.OracleSafetyError as error:
        raise CaptureSafetyError(str(error)) from error
    assert isinstance(result, str)
    return result


def _require_layer_hashes(value: object, label: str) -> list[str]:
    try:
        return oracle._layer_hashes(value, label)
    except oracle.OracleSafetyError as error:
        raise CaptureSafetyError(str(error)) from error


def normalize_state_event(value: object) -> dict[str, Any]:
    """Authenticate one state record produced after accepted-path commit."""

    if not isinstance(value, dict) or set(value) != STATE_EVENT_KEYS:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise CaptureSafetyError(f"state event keys are invalid: {observed}")
    if value["schema"] != STATE_EVENT_SCHEMA:
        raise CaptureSafetyError("state event schema mismatch")
    request_id = value["request_id"]
    if (
        not isinstance(request_id, str)
        or not request_id
        or any(char.isspace() for char in request_id)
    ):
        raise CaptureSafetyError("state event request_id is invalid")
    normalized = {
        **value,
        "accepted_draft_count": _require_integer(
            value["accepted_draft_count"], "accepted_draft_count", minimum=0, maximum=7
        ),
        "committed_token_count": _require_integer(
            value["committed_token_count"],
            "committed_token_count",
            minimum=1,
            maximum=1_000_000,
        ),
        "device_error_word": _require_integer(
            value["device_error_word"], "device_error_word", minimum=0, maximum=2**32 - 1
        ),
        "logical_length": _require_integer(
            value["logical_length"], "logical_length", minimum=1, maximum=1_000_000
        ),
        "nonfinite_count": _require_integer(
            value["nonfinite_count"], "nonfinite_count", minimum=0, maximum=2**31 - 1
        ),
        "rollback_generation": _require_integer(
            value["rollback_generation"],
            "rollback_generation",
            minimum=0,
            maximum=2**31 - 1,
        ),
        "round_index": _require_integer(
            value["round_index"], "round_index", minimum=0, maximum=1_000_000
        ),
        "canonical_gdn_layer_sha256": _require_layer_hashes(
            value["canonical_gdn_layer_sha256"], "canonical_gdn_layer_sha256"
        ),
        "convolution_layer_sha256": _require_layer_hashes(
            value["convolution_layer_sha256"], "convolution_layer_sha256"
        ),
    }
    for key in (
        "cache_mapping_sha256",
        "draft_kv_scales_sha256",
        "draft_kv_sha256",
        "event_sha256",
        "payload_producer_receipt_sha256",
        "target_kv_scales_sha256",
        "target_kv_sha256",
    ):
        normalized[key] = _require_sha256(value[key], key)
    if normalized["event_sha256"] != _event_digest(normalized):
        raise CaptureSafetyError("state event self-hash mismatch")
    if normalized["device_error_word"] != 0 or normalized["nonfinite_count"] != 0:
        raise CaptureSafetyError("state event reports a device error or nonfinite value")
    return normalized


def seal_state_event(value: dict[str, Any]) -> dict[str, Any]:
    """Add the canonical self-hash and validate a newly captured state event."""

    event = dict(value)
    event["event_sha256"] = _event_digest(event)
    return normalize_state_event(event)


def create_state_stream(path: Path, header: dict[str, Any]) -> int:
    """Create an owner-only append stream with the state-capture schema."""

    path = path.absolute()
    try:
        parent = path.parent.lstat()
    except FileNotFoundError as error:
        raise CaptureSafetyError(f"state stream parent does not exist: {path.parent}") from error
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) & 0o077
    ):
        raise CaptureSafetyError("state stream parent must be an owned private directory")
    if path.exists() or path.is_symlink():
        raise CaptureSafetyError(f"refusing to replace state stream: {path}")
    document = {"schema": STATE_HEADER_SCHEMA, **header}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, _canonical_line(document) + b"\n")
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def append_state_event(descriptor: int, value: dict[str, Any]) -> dict[str, Any]:
    """Authenticate and durably append one post-commit state event."""

    event = seal_state_event(value)
    os.write(descriptor, _canonical_line({"event": event}) + b"\n")
    os.fsync(descriptor)
    return event


def _require_private_regular(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise CaptureSafetyError(f"{label} does not exist: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise CaptureSafetyError(f"{label} must be an owned regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise CaptureSafetyError(f"{label} must be owner-only")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require_private_regular(path, label)
    payload = path.read_bytes()
    if len(payload) > MAX_STREAM_BYTES:
        raise CaptureSafetyError(f"{label} exceeds the capture size bound")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureSafetyError(f"invalid {label} JSON: {error}") from error
    if not isinstance(value, dict) or payload != _canonical_json(value):
        raise CaptureSafetyError(f"{label} must be a canonical JSON object")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require_private_regular(path, label)
    if path.stat().st_size > MAX_STREAM_BYTES:
        raise CaptureSafetyError(f"{label} exceeds the capture size bound")
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.endswith(b"\n"):
                raise CaptureSafetyError(f"{label}:{line_number} is not newline terminated")
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CaptureSafetyError(f"invalid {label}:{line_number}: {error}") from error
            if not isinstance(value, dict) or raw != _canonical_line(value) + b"\n":
                raise CaptureSafetyError(f"{label}:{line_number} is not canonical JSONL")
            rows.append(value)
    if len(rows) < 2:
        raise CaptureSafetyError(f"{label} contains no evidence records")
    return rows


def _normalize_round_record(
    value: object,
    expected_index: int,
    *,
    dflash_enabled: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"request_id", "round"}:
        raise CaptureSafetyError("token stream record wrapper is invalid")
    round_value = value["round"]
    expected_schema = ROUND_SCHEMA if dflash_enabled else TARGET_ONLY_ROUND_SCHEMA
    if not isinstance(round_value, dict) or round_value.get("schema") != expected_schema:
        raise CaptureSafetyError("token stream round schema mismatch")
    if round_value.get("request_id") != value["request_id"]:
        raise CaptureSafetyError("token stream request identity mismatch")
    if round_value.get("round_index") != expected_index:
        raise CaptureSafetyError("token stream round indices are not contiguous")
    supplied_digest = round_value.get("round_sha256")
    _require_sha256(supplied_digest, "round_sha256")
    unsigned = dict(round_value)
    unsigned.pop("round_sha256", None)
    if _sha256(_canonical_line(unsigned)) != supplied_digest:
        raise CaptureSafetyError("token stream round self-hash mismatch")
    if round_value.get("greedy_contract_valid") is not True:
        raise CaptureSafetyError("token stream contains a failed greedy contract")
    accepted = _require_integer(
        round_value.get("accepted_draft_count"),
        "round.accepted_draft_count",
        minimum=0,
        maximum=7,
    )
    committed = round_value.get("committed_token_ids")
    committed_target = round_value.get("committed_target_top1_token_ids")
    sampled = round_value.get("sampled_token_ids")
    proposal = round_value.get("proposal_token_ids")
    target = round_value.get("target_argmax_token_ids")
    if not all(
        isinstance(tokens, list)
        for tokens in (committed, committed_target, sampled, proposal, target)
    ):
        raise CaptureSafetyError("token stream round is missing token arrays")
    if sampled != target[: accepted + 1]:
        raise CaptureSafetyError("sampler outputs differ from current target top-1 rows")
    if committed_target != committed:
        raise CaptureSafetyError("canonical tokens differ from causal target top-1 tokens")
    if dflash_enabled:
        if committed[1:] != proposal[:accepted] or committed[1:] != sampled[:accepted]:
            raise CaptureSafetyError(
                "canonical draft inputs differ from the accepted target prefix"
            )
        if len(committed) != accepted + 1:
            raise CaptureSafetyError("canonical transition width differs from accepted plus anchor")
    elif (
        accepted != 0
        or proposal != []
        or len(target) != 1
        or len(sampled) != 1
        or len(committed) != 1
    ):
        raise CaptureSafetyError("target-only transition is not a single serial M1 step")
    return round_value


def _result_tokens(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise CaptureSafetyError(f"{label} must be a non-empty token-ID list")
    result: list[int] = []
    for index, token in enumerate(value):
        if type(token) is not int or not 0 <= token < 253_952:
            raise CaptureSafetyError(f"{label}[{index}] is outside the padded vocabulary")
        result.append(token)
    return result


def _validate_public_result(value: object, identity: dict[str, Any]) -> list[int]:
    if not isinstance(value, dict) or value.get("schema") != PUBLIC_RESULT_SCHEMA:
        raise CaptureSafetyError("public result schema mismatch")
    request = value.get("request")
    response = value.get("response")
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise CaptureSafetyError("public result is missing request or response evidence")
    if request.get("id") != identity["run_id"]:
        raise CaptureSafetyError("public result request ID differs from oracle identity")
    if request.get("prompt_tokens") != identity["context_tokens"]:
        raise CaptureSafetyError("public result prompt length differs from oracle identity")
    if request.get("sampling") != identity["sampling"]:
        raise CaptureSafetyError("public result sampling differs from oracle identity")
    max_tokens = _require_integer(
        request.get("max_tokens"),
        "public result max_tokens",
        minimum=1,
        maximum=2049,
    )
    completion = _result_tokens(response.get("completion_token_ids"), "completion_token_ids")
    expected_digest = _sha256(_canonical_line(completion))
    if response.get("completion_token_ids_sha256") != expected_digest:
        raise CaptureSafetyError("public result completion token hash mismatch")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise CaptureSafetyError("public result usage is missing")
    if usage.get("prompt_tokens") != identity["context_tokens"]:
        raise CaptureSafetyError("public result usage prompt length mismatch")
    if usage.get("completion_tokens") != len(completion):
        raise CaptureSafetyError("public result usage completion length mismatch")
    if usage.get("total_tokens") != identity["context_tokens"] + len(completion):
        raise CaptureSafetyError("public result usage total length mismatch")
    if response.get("finish_reason") != "length" or len(completion) != max_tokens:
        raise CaptureSafetyError("assurance request did not reach its exact max-token boundary")
    return completion


def reduce_capture(
    *,
    identity_path: Path,
    round_stream: Path,
    state_stream: Path,
    result_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Join token and post-commit state streams into an unsealed oracle source."""

    identity = _load_json(identity_path.absolute(), "oracle identity")
    try:
        identity = oracle._validate_identity(identity)
    except oracle.OracleSafetyError as error:
        raise CaptureSafetyError(str(error)) from error
    public_tokens = _validate_public_result(
        _load_json(result_path.absolute(), "public result"), identity
    )

    round_rows = _load_jsonl(round_stream.absolute(), "token stream")
    round_header = round_rows.pop(0)
    if round_header.get("schema") != ROUND_HEADER_SCHEMA:
        raise CaptureSafetyError("token stream header schema mismatch")
    dflash_enabled = bool(identity["dflash_enabled"])
    rounds = [
        _normalize_round_record(row, index, dflash_enabled=dflash_enabled)
        for index, row in enumerate(round_rows)
    ]

    state_rows = _load_jsonl(state_stream.absolute(), "state stream")
    state_header = state_rows.pop(0)
    if state_header.get("schema") != STATE_HEADER_SCHEMA:
        raise CaptureSafetyError("state stream header schema mismatch")
    committed_count_filter = state_header.get("committed_count_filter")
    if committed_count_filter is not None and (
        not isinstance(committed_count_filter, list)
        or not 1 <= len(committed_count_filter) <= 64
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1_000_000
            for value in committed_count_filter
        )
        or committed_count_filter != sorted(set(committed_count_filter))
    ):
        raise CaptureSafetyError("state committed-count filter is invalid")
    states: list[dict[str, Any]] = []
    prior_state_round = -1
    for wrapper in state_rows:
        if not isinstance(wrapper, dict) or set(wrapper) != {"event"}:
            raise CaptureSafetyError("state stream event wrapper is invalid")
        event = normalize_state_event(wrapper["event"])
        if event["round_index"] <= prior_state_round:
            raise CaptureSafetyError("state stream round indices are not strictly increasing")
        if event["round_index"] >= len(rounds):
            raise CaptureSafetyError("state stream references an absent token round")
        prior_state_round = event["round_index"]
        states.append(event)
    if committed_count_filter is None and len(rounds) != len(states):
        raise CaptureSafetyError("token and state streams have different round counts")
    if committed_count_filter is None and any(
        event["round_index"] != index for index, event in enumerate(states)
    ):
        raise CaptureSafetyError("state stream round indices are not contiguous")
    states_by_round = {state["round_index"]: state for state in states}

    committed: list[int] = []
    target_top1: list[int] = []
    checkpoints: list[dict[str, Any]] = []
    previous_next_anchor: int | None = None
    observed_state_counts: list[int] = []
    for round_value in rounds:
        state = states_by_round.get(round_value["round_index"])
        accepted = int(round_value["accepted_draft_count"])
        round_committed = [int(token) for token in round_value["committed_token_ids"]]
        round_target = [int(token) for token in round_value["committed_target_top1_token_ids"]]
        if previous_next_anchor is not None and round_committed[0] != previous_next_anchor:
            raise CaptureSafetyError(
                "canonical round anchor does not chain from prior target sample"
            )
        previous_next_anchor = int(round_value["sampled_token_ids"][-1])
        committed.extend(round_committed)
        target_top1.extend(round_target)
        if state is None:
            continue
        if round_value["request_id"] != state["request_id"]:
            raise CaptureSafetyError("token/state request identity mismatch")
        if accepted != state["accepted_draft_count"]:
            raise CaptureSafetyError("token/state accepted-draft count mismatch")
        if state["committed_token_count"] != len(committed):
            raise CaptureSafetyError("state committed count does not match token stream")
        observed_state_counts.append(len(committed))
        expected_logical = identity["context_tokens"] + len(committed)
        if state["logical_length"] != expected_logical:
            raise CaptureSafetyError("state logical length does not match context plus output")
        if state["payload_producer_receipt_sha256"] != identity["payload_producer_receipt_sha256"]:
            raise CaptureSafetyError("state producer receipt does not match oracle identity")
        prefix_digest = oracle._prefix_digest(committed, len(committed))
        top1_digest = oracle._prefix_digest(target_top1, len(target_top1))
        checkpoint = {
            "accepted_draft_count": accepted,
            "cache_mapping_sha256": state["cache_mapping_sha256"],
            "canonical_gdn_layer_sha256": state["canonical_gdn_layer_sha256"],
            "committed_token_count": len(committed),
            "committed_token_prefix_sha256": prefix_digest,
            "convolution_layer_sha256": state["convolution_layer_sha256"],
            "device_error_word": state["device_error_word"],
            "draft_kv_scales_sha256": state["draft_kv_scales_sha256"],
            "draft_kv_sha256": state["draft_kv_sha256"],
            "logical_length": state["logical_length"],
            "nonfinite_count": state["nonfinite_count"],
            "payload_producer_receipt_sha256": state["payload_producer_receipt_sha256"],
            "rollback_generation": state["rollback_generation"],
            "target_kv_scales_sha256": state["target_kv_scales_sha256"],
            "target_kv_sha256": state["target_kv_sha256"],
            "target_top1_prefix_sha256": top1_digest,
        }
        if committed_count_filter is not None:
            checkpoint["captured_round_committed_count"] = len(round_committed)
        checkpoints.append(checkpoint)

    if committed_count_filter is not None and observed_state_counts != committed_count_filter:
        raise CaptureSafetyError(
            "state stream does not contain every requested exact committed-count boundary"
        )

    if previous_next_anchor is None:
        raise CaptureSafetyError("canonical stream has no authenticated lookahead sample")
    # Both serial M1 and speculative M8 are autoregressive pipelines.  A model
    # invocation consumes a preceding sample into canonical KV/GDN state and
    # samples the following token.  A max_tokens cap may meet a verification
    # round in either of two exact states:
    #
    # * public == committed: the cap discarded the newly sampled lookahead;
    # * public == committed + lookahead: the final public sample is pending and
    #   has not yet been consumed into canonical state.
    #
    # No other skew is a legal autoregressive boundary.  In particular, never
    # invent a state transition for the pending sample and never expose the
    # discarded lookahead as public output.
    pending_public = [*committed, previous_next_anchor]
    if public_tokens not in (committed, pending_public):
        lane = "DFlash" if dflash_enabled else "target-only"
        raise CaptureSafetyError(
            f"{lane} canonical transitions/max-token lookahead differ from API completion"
        )
    if target_top1 != committed:
        lane = "DFlash" if dflash_enabled else "target-only"
        raise CaptureSafetyError(f"{lane} committed tokens differ from causal target top-1")

    source = {
        "checkpoints": checkpoints,
        "committed_target_top1_token_ids": target_top1,
        "committed_token_ids": committed,
        "identity": identity,
        "schema": oracle.ORACLE_SCHEMA,
    }
    try:
        source = oracle.validate_oracle(source, sealed=False)
        oracle._write_create_only(output.absolute(), source)
    except oracle.OracleSafetyError as error:
        raise CaptureSafetyError(str(error)) from error
    return source


# QWEN_ASSURANCE_ONLY_END: coding-turbo-state-capture
