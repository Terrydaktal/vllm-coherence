"""Produce authenticated provider/parser evidence for one assurance fragment.

This producer consumes a create-only raw chat-stream capture.  It independently
authenticates the exact prompt vector, request/response wire bytes, raw generated
token IDs, parser output, and structured tool ledger before it emits the single
``structured_outcome`` event owned by the provider controller.

The resulting event is bounded evidence for this exact request.  It is not a
universal arbitrary-prompt or whole-backend equivalence claim.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_r9700_lab import assurance_instrumentation as instrumentation

CHAT_STREAM_SCHEMA = "urn:qwen-r9700:failed-prompt-chat-stream:v1"
TOKEN_MAGIC = b"QWENSTG1"
TOKEN_DIGEST_DOMAIN = b"qwen-r9700-token-ids-u32be-v1\0"
QWEN_TOOL_CALL_START_TOKEN_ID = 248_058
QWEN_TOOL_CALL_END_TOKEN_ID = 248_059
MAX_CAPTURE_BYTES = 256 << 20
MAX_PROMPT_TOKENS = 253_792
PRODUCER_KIND = "provider_controller"
INSTRUMENTATION_SITE = "provider.raw-token-outcome"
EXPECTED_SCOPE = (
    "structured_outcome",
    None,
    None,
    None,
    instrumentation.UNIT_PHASES["structured_outcome"],
    None,
)


class ProviderProducerError(RuntimeError):
    """The raw provider evidence is incomplete, altered, or structurally invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
    except (TypeError, ValueError) as error:
        raise ProviderProducerError(f"value is not canonical JSON: {error}") from error


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProviderProducerError(f"{label} must be a lowercase SHA-256")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns


def _read_stable_owned_file(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
    maximum: int = MAX_CAPTURE_BYTES,
) -> bytes:
    path = path.expanduser()
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ProviderProducerError(f"{label} path must be normalized and absolute")
    try:
        before = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_mode & 0o022
            or not 0 < before.st_size <= maximum
        ):
            raise ProviderProducerError(f"{label} is not a safe owned regular file")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise ProviderProducerError(f"cannot read {label}: {error}") from error
    if _stat_identity(before) != _stat_identity(after):
        raise ProviderProducerError(f"{label} changed while being authenticated")
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise ProviderProducerError(
            f"{label} SHA-256 differs: expected {expected_sha256}, observed {observed}"
        )
    return payload


def _load_json(path: Path, label: str, *, expected_sha256: str | None = None) -> Any:
    payload = _read_stable_owned_file(path, label, expected_sha256=expected_sha256)
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProviderProducerError(f"{label} is not valid JSON: {error}") from error


def _token_digest(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256(TOKEN_DIGEST_DOMAIN)
    for token in tokens:
        if type(token) is not int or not 0 <= token <= 0xFFFFFFFF:
            raise ProviderProducerError("completion contains an invalid token ID")
        digest.update(struct.pack(">I", token))
    return digest.hexdigest()


def _read_token_vector(path: Path, expected_file_sha256: str) -> list[int]:
    payload = _read_stable_owned_file(
        path,
        "provider prompt token vector",
        expected_sha256=expected_file_sha256,
        maximum=16 + MAX_PROMPT_TOKENS * 4 + 32,
    )
    minimum = len(TOKEN_MAGIC) + 8 + 32
    if len(payload) < minimum or payload[: len(TOKEN_MAGIC)] != TOKEN_MAGIC:
        raise ProviderProducerError("provider prompt token vector is not QWENSTG1")
    body, trailer = payload[:-32], payload[-32:]
    if hashlib.sha256(body).digest() != trailer:
        raise ProviderProducerError("provider prompt token vector trailer differs")
    count = struct.unpack(">Q", body[8:16])[0]
    if not 1 <= count <= MAX_PROMPT_TOKENS or len(body) != 16 + count * 4:
        raise ProviderProducerError("provider prompt token-vector geometry differs")
    return [struct.unpack_from(">I", body, 16 + index * 4)[0] for index in range(count)]


def _decode_bound_bytes(
    document: Mapping[str, Any], encoded_field: str, digest_field: str, label: str
) -> bytes:
    encoded = document.get(encoded_field)
    expected = document.get(digest_field)
    if not isinstance(encoded, str) or not isinstance(expected, str):
        raise ProviderProducerError(f"capture {label} bytes are absent")
    _require_digest(expected, f"capture {label}")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ProviderProducerError(f"capture {label} is not canonical base64") from error
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ProviderProducerError(f"capture {label} bytes differ from their digest")
    return payload


def _normalize_tool_calls(
    parser: Mapping[str, Any], allowed_names: Sequence[str], final_tool_name: str
) -> tuple[list[dict[str, Any]], str]:
    raw_calls = parser.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        raise ProviderProducerError("provider parser did not emit a structured tool call")
    calls: list[dict[str, Any]] = []
    allowed = set(allowed_names)
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            raise ProviderProducerError("provider parser tool ledger is malformed")
        name = raw_call.get("name")
        arguments_text = raw_call.get("arguments")
        if (
            raw_call.get("index") != index
            or raw_call.get("type") != "function"
            or not isinstance(raw_call.get("id"), str)
            or not raw_call["id"]
            or not isinstance(name, str)
            or name not in allowed
            or raw_call.get("arguments_valid_json") is not True
            or not isinstance(arguments_text, str)
        ):
            raise ProviderProducerError("provider parser emitted an invalid tool ledger entry")
        try:
            arguments = json.loads(arguments_text)
        except json.JSONDecodeError as error:
            raise ProviderProducerError("provider parser emitted invalid JSON arguments") from error
        if not isinstance(arguments, dict):
            raise ProviderProducerError("provider tool arguments must be a JSON object")
        calls.append(
            {
                "arguments": arguments,
                "arguments_text_sha256": hashlib.sha256(arguments_text.encode()).hexdigest(),
                "index": index,
                "name": name,
                "type": "function",
            }
        )
    reserved = [call for call in calls if call["name"] == final_tool_name]
    if reserved:
        if len(calls) != 1:
            raise ProviderProducerError("final-answer outcome was mixed with another tool call")
        answer = reserved[0]["arguments"].get("answer")
        if (
            set(reserved[0]["arguments"]) != {"answer"}
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ProviderProducerError("final-answer outcome has invalid arguments")
        return calls, "final_answer"
    return calls, "tool_call"


def build_evidence(
    capture_value: object,
    *,
    header: Mapping[str, Any],
    expected_configuration_sha256: str,
    final_tool_name: str,
) -> dict[str, Any]:
    """Validate one raw capture and derive its comparable semantic witnesses."""

    if not isinstance(capture_value, dict):
        raise ProviderProducerError("provider capture is not an object")
    capture = json.loads(_canonical(capture_value))
    fingerprint = capture.pop("capture_sha256", None)
    if fingerprint != hashlib.sha256(_canonical(capture)).hexdigest():
        raise ProviderProducerError("provider capture self-hash differs")
    capture["capture_sha256"] = fingerprint
    if capture.get("schema") != CHAT_STREAM_SCHEMA:
        raise ProviderProducerError("provider capture schema differs")

    arm = capture.get("arm")
    if (
        not isinstance(arm, dict)
        or set(arm) != {"configuration_sha256", "id"}
        or arm.get("configuration_sha256") != expected_configuration_sha256
        or not isinstance(arm.get("id"), str)
        or not arm["id"]
    ):
        raise ProviderProducerError("provider capture arm/configuration binding differs")

    source = capture.get("source")
    if not isinstance(source, dict):
        raise ProviderProducerError("provider capture source is absent")
    source_prompt_digest = _require_digest(
        source.get("prompt_token_ids_sha256"), "provider prompt token IDs"
    )
    source_file_digest = _require_digest(
        source.get("token_file_sha256"), "provider prompt token file"
    )
    token_file = source.get("token_file")
    if not isinstance(token_file, str):
        raise ProviderProducerError("provider prompt token path is invalid")
    prompt_tokens = _read_token_vector(Path(token_file), source_file_digest)
    if (
        source.get("prompt_token_count") != len(prompt_tokens)
        or _token_digest(prompt_tokens) != source_prompt_digest
        or header.get("prompt_tokens") != len(prompt_tokens)
        or header.get("prompt_sha256") != source_prompt_digest
    ):
        raise ProviderProducerError("provider prompt identity differs from campaign header")

    preflight = capture.get("preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("exact_prompt_token_ids_verified") is not True
    ):
        raise ProviderProducerError("provider prompt preflight was not exact")
    tool_names = preflight.get("tool_names")
    if (
        not isinstance(tool_names, list)
        or not tool_names
        or any(not isinstance(name, str) or not name for name in tool_names)
        or len(tool_names) != len(set(tool_names))
    ):
        raise ProviderProducerError("provider tool-name inventory is invalid")
    tools_sha256 = _require_digest(preflight.get("tools_sha256"), "provider tool schema")

    request = capture.get("request")
    response = capture.get("response")
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise ProviderProducerError("provider request/response evidence is absent")
    request_body = _decode_bound_bytes(request, "body_base64", "body_sha256", "request body")
    response_body = _decode_bound_bytes(
        response, "body_base64", "body_sha256", "response body"
    )
    request_wire_path = request.get("wire_evidence_path")
    request_wire_sha256 = _require_digest(
        request.get("wire_evidence_sha256"), "provider request wire evidence"
    )
    if not isinstance(request_wire_path, str):
        raise ProviderProducerError("provider request wire evidence path is invalid")
    request_wire_payload = _read_stable_owned_file(
        Path(request_wire_path),
        "provider request wire evidence",
        expected_sha256=request_wire_sha256,
    )
    try:
        request_wire = json.loads(request_wire_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProviderProducerError("provider request wire journal is invalid") from error
    if not isinstance(request_wire, dict):
        raise ProviderProducerError("provider request wire journal is not an object")
    wire_body = _decode_bound_bytes(
        request_wire, "body_base64", "body_sha256", "request wire body"
    )
    if (
        wire_body != request_body
        or request_wire.get("configuration_sha256") != expected_configuration_sha256
        or request_wire.get("semantic_body_sha256") != request.get("semantic_body_sha256")
        or request_wire.get("source_token_file_sha256") != source_file_digest
        or request_wire.get("state_contract") != request.get("state_contract")
        or request_wire.get("transport") != request.get("transport")
        or request_wire.get("endpoint") != request.get("endpoint")
        or request_wire.get("request_id") != request.get("request_id")
    ):
        raise ProviderProducerError("provider request wire journal differs from capture")

    response_wire_path = response.get("wire_evidence_path")
    response_wire_sha256 = _require_digest(
        response.get("wire_evidence_sha256"), "provider response wire evidence"
    )
    if not isinstance(response_wire_path, str):
        raise ProviderProducerError("provider response wire evidence path is invalid")
    response_wire = _read_stable_owned_file(
        Path(response_wire_path),
        "provider response wire evidence",
        expected_sha256=response_wire_sha256,
    )
    if response_wire != response_body:
        raise ProviderProducerError("provider response wire evidence differs from embedded bytes")

    completion_ids = response.get("completion_token_ids")
    if not isinstance(completion_ids, list) or not completion_ids:
        raise ProviderProducerError("provider completion token vector is absent")
    raw_token_ids_sha256 = _token_digest(completion_ids)
    if (
        response.get("http_status") != 200
        or response.get("done_markers") != 1
        or response.get("completion_token_count") != len(completion_ids)
        or response.get("completion_token_ids_sha256") != raw_token_ids_sha256
    ):
        raise ProviderProducerError("provider stream/token accounting differs")
    usage = response.get("usage")
    if (
        not isinstance(usage, dict)
        or usage.get("prompt_tokens") != len(prompt_tokens)
        or usage.get("completion_tokens") != len(completion_ids)
    ):
        raise ProviderProducerError("provider usage accounting differs")

    parser = response.get("parser")
    if (
        not isinstance(parser, dict)
        or parser.get("classification") != "parsed_structured_tool_call"
        or parser.get("errors") != []
    ):
        raise ProviderProducerError("provider parser did not produce one valid structural outcome")
    content = _decode_bound_bytes(
        parser, "content_base64", "content_utf8_sha256", "parser content"
    )
    reasoning = _decode_bound_bytes(
        parser, "reasoning_base64", "reasoning_utf8_sha256", "parser reasoning"
    )
    raw_completion = response.get("raw_completion")
    if not isinstance(raw_completion, dict):
        raise ProviderProducerError("provider raw completion evidence is absent")
    raw_text = _decode_bound_bytes(
        raw_completion, "text_base64", "text_utf8_sha256", "raw completion text"
    )
    text_available = raw_completion.get("text_available")
    if not isinstance(text_available, bool):
        raise ProviderProducerError("provider raw completion availability flag is invalid")
    if not text_available and raw_text:
        raise ProviderProducerError("unavailable raw completion unexpectedly contains text")

    starts = [
        index
        for index, token in enumerate(completion_ids)
        if token == QWEN_TOOL_CALL_START_TOKEN_ID
    ]
    ends = [
        index for index, token in enumerate(completion_ids) if token == QWEN_TOOL_CALL_END_TOKEN_ID
    ]
    if (
        starts != raw_completion.get("tool_call_start_positions")
        or ends != raw_completion.get("tool_call_end_positions")
        or len(starts) != len(ends)
        or any(start >= end for start, end in zip(starts, ends, strict=True))
        or response.get("finish_reason") != "tool_calls"
        or response.get("stop_reason") is not None
    ):
        raise ProviderProducerError("provider raw/parser terminal structure differs")

    calls, outcome = _normalize_tool_calls(parser, tool_names, final_tool_name)
    if len(starts) != len(calls):
        raise ProviderProducerError("provider raw and parsed tool-call counts differ")
    normalized_parser = {
        "classification": parser["classification"],
        "content_utf8_sha256": hashlib.sha256(content).hexdigest(),
        "reasoning_utf8_sha256": hashlib.sha256(reasoning).hexdigest(),
        "tool_calls": calls,
    }
    parser_input = {
        "completion_token_ids": completion_ids,
        "tool_call_end_positions": ends,
        "tool_call_start_positions": starts,
    }
    return {
        "raw_token_ids_sha256": raw_token_ids_sha256,
        "parser_input_sha256": _sha(parser_input),
        "parser_output_sha256": _sha(normalized_parser),
        "finish_reason": response["finish_reason"],
        "tool_ledger_sha256": _sha(calls),
        "outcome": outcome,
        "capture_sha256": fingerprint,
        "configuration_sha256": expected_configuration_sha256,
        "prompt_token_ids_sha256": source_prompt_digest,
        "raw_completion_text_available": text_available,
        "raw_completion_utf8_sha256": hashlib.sha256(raw_text).hexdigest(),
        "tool_schema_sha256": tools_sha256,
    }


def _fragment_context() -> tuple[dict[str, Any], str]:
    if os.environ.get(instrumentation.FULL_ASSURANCE_ENABLE_ENV) != "1":
        raise ProviderProducerError("provider fragment was not explicitly enabled")
    missing = [
        name
        for name in instrumentation.FULL_ASSURANCE_ENVIRONMENT
        if not os.environ.get(name)
    ]
    if missing:
        raise ProviderProducerError(f"provider fragment environment is incomplete: {missing}")
    if os.environ[instrumentation.FULL_ASSURANCE_PRODUCER_KIND_ENV] != PRODUCER_KIND:
        raise ProviderProducerError("provider fragment producer kind differs")
    own_path = Path(__file__).resolve(strict=True)
    own_payload = _read_stable_owned_file(own_path, "provider producer source")
    own_sha256 = hashlib.sha256(own_payload).hexdigest()
    if own_sha256 != os.environ[instrumentation.FULL_ASSURANCE_PRODUCER_SHA_ENV]:
        raise ProviderProducerError("provider producer source binding differs")

    header = instrumentation.normalize_header(
        _load_json(
            Path(os.environ[instrumentation.FULL_ASSURANCE_HEADER_ENV]),
            "provider campaign header",
        )
    )
    scopes_document = _load_json(
        Path(os.environ[instrumentation.FULL_ASSURANCE_SCOPES_ENV]),
        "provider runtime scopes",
    )
    if not isinstance(scopes_document, dict) or not isinstance(
        scopes_document.get("scopes"), list
    ):
        raise ProviderProducerError("provider runtime scope document is malformed")
    expected_document = instrumentation.runtime_scopes_document(
        header, scopes_document["scopes"]
    )
    if scopes_document != expected_document or scopes_document["scopes"] != [list(EXPECTED_SCOPE)]:
        raise ProviderProducerError("provider runtime scope identity differs")

    return header, own_sha256


def produce(
    capture_path: Path,
    expected_capture_sha256: str,
    expected_configuration_sha256: str,
    *,
    final_tool_name: str = "qwen_final_answer",
) -> dict[str, Any]:
    expected_capture_sha256 = _require_digest(
        expected_capture_sha256, "provider capture file"
    )
    expected_configuration_sha256 = _require_digest(
        expected_configuration_sha256, "provider configuration"
    )
    if not final_tool_name or any(character.isspace() for character in final_tool_name):
        raise ProviderProducerError("final tool name is invalid")
    capture = _load_json(
        capture_path,
        "provider chat-stream capture",
        expected_sha256=expected_capture_sha256,
    )
    header, own_sha256 = _fragment_context()
    evidence = build_evidence(
        capture,
        header=header,
        expected_configuration_sha256=expected_configuration_sha256,
        final_tool_name=final_tool_name,
    )
    writer = instrumentation.FullAssuranceFragmentWriter(
        Path(os.environ[instrumentation.FULL_ASSURANCE_ROOT_ENV]),
        header,
        fragment_id=os.environ[instrumentation.FULL_ASSURANCE_FRAGMENT_ID_ENV],
        scopes=(EXPECTED_SCOPE,),
        producer_kind=PRODUCER_KIND,
        producer_sha256=own_sha256,
        sync_interval=1,
    )
    try:
        writer.append(
            unit="structured_outcome",
            phase=instrumentation.UNIT_PHASES["structured_outcome"],
            position=None,
            layer_index=None,
            row=None,
            evidence=evidence,
        )
        return writer.finalize(
            device_synchronize_count=0,
            installed_instrumentation_sites=(INSTRUMENTATION_SITE,),
        )
    except BaseException:
        writer.abort()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-assurance-provider-producer",
        description="Authenticate one raw structured outcome into an assurance fragment.",
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--expected-capture-sha256", required=True)
    parser.add_argument("--expected-configuration-sha256", required=True)
    parser.add_argument("--final-tool-name", default="qwen_final_answer")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = produce(
            args.capture,
            args.expected_capture_sha256,
            args.expected_configuration_sha256,
            final_tool_name=args.final_tool_name,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        instrumentation.InstrumentationError,
        ProviderProducerError,
    ) as error:
        print(f"qwen-assurance-provider-producer: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
