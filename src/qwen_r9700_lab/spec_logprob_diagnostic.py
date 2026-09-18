"""Capture and compare one exact speculative-decoding logprob diagnostic."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwen_r9700_lab.config import ConfigurationError, find_project_root

CAPTURE_TYPE = "qwen-r9700-speculative-logprob-diagnostic"
CASE_ID = "code-merge-intervals"
CASE_MESSAGES_SHA256 = "99443fa51e21fe2d5871da49506e4d4ef7f3b3377906bbc3dc1878a1fea2a78f"
FIXTURE_ID = "qwen3.8-r9700-true-c1-v1"
FIXTURE_SHA256 = "5678b2581bf0ca34db1746f72c25896c67506567a03ec3e1d52aa3ec1b759fcd"
MAX_TOKENS = 160
TOP_LOGPROBS = 20


class SpecLogprobDiagnosticError(RuntimeError):
    """A capture or comparison did not satisfy the diagnostic contract."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _capture_id(backend_id: str, request_body: bytes, response_body: bytes) -> str:
    return _sha256(
        _canonical_bytes(
            {
                "backend_id": backend_id,
                "fixture_sha256": FIXTURE_SHA256,
                "request_body_sha256": _sha256(request_body),
                "response_body_sha256": _sha256(response_body),
            }
        )
    )


def _absolute_without_resolving_final(path: Path) -> Path:
    return path.expanduser().absolute()


def _body_record(body: bytes) -> dict[str, Any]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SpecLogprobDiagnosticError("HTTP JSON body is not valid UTF-8") from error
    return {
        "base64": base64.b64encode(body).decode("ascii"),
        "sha256": _sha256(body),
        "size_bytes": len(body),
        "utf8": text,
    }


def _decode_body_record(record: object, label: str) -> bytes:
    if not isinstance(record, dict):
        raise SpecLogprobDiagnosticError(f"{label} raw_body must be an object")
    encoded = record.get("base64")
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    expected_text = record.get("utf8")
    if not isinstance(encoded, str):
        raise SpecLogprobDiagnosticError(f"{label} raw_body.base64 must be a string")
    try:
        body = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise SpecLogprobDiagnosticError(f"{label} raw_body.base64 is invalid") from error
    if expected_hash != _sha256(body):
        raise SpecLogprobDiagnosticError(f"{label} raw body SHA-256 mismatch")
    if expected_size != len(body):
        raise SpecLogprobDiagnosticError(f"{label} raw body size mismatch")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SpecLogprobDiagnosticError(f"{label} raw body is not UTF-8") from error
    if expected_text != text:
        raise SpecLogprobDiagnosticError(f"{label} raw body UTF-8/base64 mismatch")
    return body


def _json_object(body: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise SpecLogprobDiagnosticError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise SpecLogprobDiagnosticError(f"{label} must be a JSON object")
    return value


def _load_locked_code_fixture(path: Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    try:
        fixture_bytes = path.read_bytes()
    except OSError as error:
        raise SpecLogprobDiagnosticError(f"cannot read fixture {path}: {error}") from error
    actual_hash = _sha256(fixture_bytes)
    if actual_hash != FIXTURE_SHA256:
        raise SpecLogprobDiagnosticError(
            f"fixture SHA-256 mismatch: expected {FIXTURE_SHA256}, got {actual_hash}"
        )
    fixture = _json_object(fixture_bytes, f"fixture {path}")
    if fixture.get("schema_version") != 1 or fixture.get("id") != FIXTURE_ID:
        raise SpecLogprobDiagnosticError("fixture identity does not match the locked C1 v1 fixture")
    cases = fixture.get("cases")
    if not isinstance(cases, list):
        raise SpecLogprobDiagnosticError("fixture cases must be a list")
    matches = [case for case in cases if isinstance(case, dict) and case.get("id") == CASE_ID]
    if len(matches) != 1:
        raise SpecLogprobDiagnosticError(f"fixture must contain exactly one {CASE_ID!r} case")
    case = matches[0]
    messages = case.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SpecLogprobDiagnosticError(f"fixture case {CASE_ID!r} has no messages")
    if _sha256(_canonical_bytes(messages)) != CASE_MESSAGES_SHA256:
        raise SpecLogprobDiagnosticError(f"fixture case {CASE_ID!r} messages do not match the lock")
    return fixture, fixture_bytes, case


def _diagnostic_payload(fixture: Mapping[str, Any], case: Mapping[str, Any], model: str) -> dict:
    common = fixture.get("common_request")
    if not isinstance(common, dict):
        raise SpecLogprobDiagnosticError("fixture common_request must be an object")
    payload = dict(common)
    payload.pop("stream_options", None)
    payload.update(
        {
            "logprobs": True,
            "max_tokens": MAX_TOKENS,
            "messages": case["messages"],
            "model": model,
            "n": 1,
            "return_token_ids": True,
            "stream": False,
            "temperature": 0.0,
            "top_k": 1,
            "top_logprobs": TOP_LOGPROBS,
            "top_p": 1.0,
        }
    )
    return payload


def _chat_endpoint(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SpecLogprobDiagnosticError(f"expected an absolute HTTP(S) base URL, got {base_url!r}")
    if parsed.query or parsed.fragment:
        raise SpecLogprobDiagnosticError("base URL must not contain a query or fragment")
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def _response_choice(response: Mapping[str, Any]) -> tuple[dict, list[int], list[dict]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise SpecLogprobDiagnosticError("response must contain choices[0]")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise SpecLogprobDiagnosticError("response choices[0].message must be an object")
    token_ids = message.get("token_ids", choice.get("token_ids"))
    if not isinstance(token_ids, list) or not all(type(item) is int for item in token_ids):
        raise SpecLogprobDiagnosticError(
            "response must contain integer choices[0].token_ids or legacy message.token_ids"
        )
    logprobs = choice.get("logprobs")
    content_logprobs = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content_logprobs, list) or not all(
        isinstance(item, dict) for item in content_logprobs
    ):
        raise SpecLogprobDiagnosticError(
            "response must contain choices[0].logprobs.content records"
        )
    if len(content_logprobs) != len(token_ids):
        raise SpecLogprobDiagnosticError(
            "response token_ids/logprobs length mismatch: "
            f"{len(token_ids)} token IDs versus {len(content_logprobs)} logprob records"
        )
    return choice, token_ids, content_logprobs


def _prepare_destination(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise SpecLogprobDiagnosticError(f"refusing to replace existing capture: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SpecLogprobDiagnosticError(
            f"cannot create capture parent directory {path.parent}: {error}"
        ) from error


def _write_create_once(path: Path, document: Mapping[str, Any]) -> None:
    _prepare_destination(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.spec-logprob-",
            delete=False,
        ) as handle:
            output = (
                json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
            )
            handle.write(output)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
            temporary = Path(handle.name)
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as error:
        raise SpecLogprobDiagnosticError(f"refusing to replace existing capture: {path}") from error
    except OSError as error:
        raise SpecLogprobDiagnosticError(f"cannot create capture {path}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _headers_for_request(api_key: str | None) -> tuple[dict[str, str], list[dict[str, str]]]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    recorded = [
        {"name": name, "value": "<redacted>" if name == "Authorization" else value}
        for name, value in headers.items()
    ]
    return headers, recorded


def capture(args: argparse.Namespace) -> dict[str, Any]:
    """Send the single locked request and atomically publish its evidence."""

    if not args.model.strip() or not args.backend_id.strip():
        raise SpecLogprobDiagnosticError("model and backend-id must be non-empty")
    if args.timeout <= 0:
        raise SpecLogprobDiagnosticError("timeout must be positive")
    destination = _absolute_without_resolving_final(args.output)
    _prepare_destination(destination)

    if args.fixture is None:
        try:
            fixture_path = find_project_root() / "benchmarks" / "c1" / "fixture-v1.json"
        except ConfigurationError as error:
            raise SpecLogprobDiagnosticError(str(error)) from error
    else:
        fixture_path = args.fixture.expanduser().resolve()
    fixture, fixture_bytes, case = _load_locked_code_fixture(fixture_path)
    payload = _diagnostic_payload(fixture, case, args.model)
    request_body = _canonical_bytes(payload)
    endpoint = _chat_endpoint(args.base_url)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    headers, recorded_headers = _headers_for_request(api_key)
    request = urllib.request.Request(endpoint, data=request_body, headers=headers, method="POST")
    started_at = _utc_now()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as http_response:
            response_body = http_response.read()
            response_status = http_response.status
            response_reason = http_response.reason
            response_headers = [
                {"name": name, "value": value} for name, value in http_response.headers.raw_items()
            ]
    except urllib.error.HTTPError as error:
        error.read()
        raise SpecLogprobDiagnosticError(
            f"POST {endpoint} returned HTTP status {error.code}"
        ) from error
    except (OSError, urllib.error.URLError) as error:
        raise SpecLogprobDiagnosticError(f"POST {endpoint} failed: {error}") from error
    completed_at = _utc_now()
    if response_status != 200:
        raise SpecLogprobDiagnosticError(
            f"POST {endpoint} returned unexpected HTTP status {response_status}"
        )

    response_json = _json_object(response_body, "HTTP response body")
    choice, token_ids, content_logprobs = _response_choice(response_json)
    message = choice["message"]
    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    derived = {
        "choice_index": 0,
        "completion_token_ids": token_ids,
        "completion_token_ids_sha256": _sha256(_canonical_bytes(token_ids)),
        "finish_reason": choice.get("finish_reason"),
        "logprob_position_count": len(content_logprobs),
        "message_content_sha256": _sha256(str(content).encode()) if content is not None else None,
        "reasoning_content_sha256": (
            _sha256(str(reasoning_content).encode()) if reasoning_content is not None else None
        ),
    }
    request_record = _body_record(request_body)
    response_record = _body_record(response_body)
    capture_id = _capture_id(args.backend_id, request_body, response_body)
    document = {
        "schema_version": 1,
        "capture_type": CAPTURE_TYPE,
        "capture_id": capture_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "backend_id": args.backend_id,
        "model": args.model,
        "fixture": {
            "case_id": CASE_ID,
            "id": FIXTURE_ID,
            "messages_sha256": CASE_MESSAGES_SHA256,
            "path": str(fixture_path),
            "sha256": _sha256(fixture_bytes),
        },
        "request": {
            "headers": recorded_headers,
            "method": "POST",
            "raw_body": request_record,
            "url": endpoint,
        },
        "response": {
            "headers": response_headers,
            "http_reason": response_reason,
            "http_status": response_status,
            "raw_body": response_record,
        },
        "derived": derived,
    }
    _write_create_once(destination, document)
    return document


def _load_capture(
    path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], list[int], list[dict]]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise SpecLogprobDiagnosticError(f"cannot load capture {path}: {error}") from error
    if not isinstance(document, dict):
        raise SpecLogprobDiagnosticError(f"capture must be a JSON object: {path}")
    if document.get("schema_version") != 1 or document.get("capture_type") != CAPTURE_TYPE:
        raise SpecLogprobDiagnosticError(f"not a {CAPTURE_TYPE} v1 capture: {path}")
    fixture = document.get("fixture")
    if (
        not isinstance(fixture, dict)
        or fixture.get("sha256") != FIXTURE_SHA256
        or fixture.get("id") != FIXTURE_ID
        or fixture.get("case_id") != CASE_ID
        or fixture.get("messages_sha256") != CASE_MESSAGES_SHA256
    ):
        raise SpecLogprobDiagnosticError(f"capture does not use the locked C1 fixture: {path}")
    request = document.get("request")
    response = document.get("response")
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise SpecLogprobDiagnosticError(f"capture request/response records are invalid: {path}")
    request_body = _decode_body_record(request.get("raw_body"), f"{path} request")
    response_body = _decode_body_record(response.get("raw_body"), f"{path} response")
    request_json = _json_object(request_body, f"{path} request body")
    response_json = _json_object(response_body, f"{path} response body")
    required_request_fields = {
        "logprobs": True,
        "max_tokens": MAX_TOKENS,
        "n": 1,
        "return_token_ids": True,
        "stream": False,
        "temperature": 0.0,
        "top_k": 1,
        "top_logprobs": TOP_LOGPROBS,
        "top_p": 1.0,
    }
    if any(request_json.get(key) != value for key, value in required_request_fields.items()):
        raise SpecLogprobDiagnosticError(f"capture request contract is invalid: {path}")
    if "stream_options" in request_json:
        raise SpecLogprobDiagnosticError(f"capture request unexpectedly has stream_options: {path}")
    if _sha256(_canonical_bytes(request_json.get("messages"))) != CASE_MESSAGES_SHA256:
        raise SpecLogprobDiagnosticError(
            f"capture request messages do not match the locked case: {path}"
        )
    backend_id = document.get("backend_id")
    if not isinstance(backend_id, str) or not backend_id:
        raise SpecLogprobDiagnosticError(f"capture backend_id is invalid: {path}")
    if document.get("model") != request_json.get("model"):
        raise SpecLogprobDiagnosticError(f"capture model disagrees with raw request: {path}")
    if document.get("capture_id") != _capture_id(backend_id, request_body, response_body):
        raise SpecLogprobDiagnosticError(f"capture ID mismatch: {path}")
    _, token_ids, content_logprobs = _response_choice(response_json)
    derived = document.get("derived")
    if not isinstance(derived, dict):
        raise SpecLogprobDiagnosticError(f"capture derived record is invalid: {path}")
    if derived.get("completion_token_ids") != token_ids:
        raise SpecLogprobDiagnosticError(
            f"capture derived token IDs disagree with raw response: {path}"
        )
    if derived.get("completion_token_ids_sha256") != _sha256(_canonical_bytes(token_ids)):
        raise SpecLogprobDiagnosticError(f"capture token ID SHA-256 mismatch: {path}")
    return document, request_body, request_json, token_ids, content_logprobs


def _same_token(candidate: object, selected: object, token_id: int | None) -> bool:
    if not isinstance(candidate, dict) or not isinstance(selected, dict):
        return False
    if token_id is not None and candidate.get("token_id") == token_id:
        return True
    candidate_bytes = candidate.get("bytes")
    selected_bytes = selected.get("bytes")
    if isinstance(candidate_bytes, list) and isinstance(selected_bytes, list):
        return candidate_bytes == selected_bytes
    return candidate.get("token") == selected.get("token")


def _candidate_rank(top_logprobs: object, selected: object, token_id: int | None) -> int | None:
    if not isinstance(top_logprobs, list):
        return None
    return next(
        (
            index
            for index, candidate in enumerate(top_logprobs)
            if _same_token(candidate, selected, token_id)
        ),
        None,
    )


def _peer_candidate(
    top_logprobs: object, peer_selected: object, peer_token_id: int | None
) -> dict | None:
    if not isinstance(top_logprobs, list):
        return None
    return next(
        (
            candidate
            for candidate in top_logprobs
            if isinstance(candidate, dict) and _same_token(candidate, peer_selected, peer_token_id)
        ),
        None,
    )


def _side_at_divergence(
    token_ids: Sequence[int],
    logprobs: Sequence[dict],
    index: int,
    peer_token_ids: Sequence[int],
    peer_logprobs: Sequence[dict],
) -> dict[str, Any] | None:
    if index >= len(token_ids):
        return None
    selected = logprobs[index]
    selected_id = token_ids[index]
    top_logprobs = selected.get("top_logprobs")
    peer_selected = peer_logprobs[index] if index < len(peer_logprobs) else None
    peer_id = peer_token_ids[index] if index < len(peer_token_ids) else None
    peer_candidate = _peer_candidate(top_logprobs, peer_selected, peer_id)
    selected_logprob = selected.get("logprob")
    peer_logprob = peer_candidate.get("logprob") if peer_candidate is not None else None
    gap = (
        selected_logprob - peer_logprob
        if isinstance(selected_logprob, (int, float)) and isinstance(peer_logprob, (int, float))
        else None
    )
    return {
        "chosen": selected,
        "chosen_rank_in_top_logprobs": _candidate_rank(top_logprobs, selected, selected_id),
        "chosen_token_id": selected_id,
        "chosen_logprob_minus_peer_choice": gap,
        "peer_choice_in_top_logprobs": peer_candidate,
        "peer_token_id": peer_id,
    }


def compare(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Verify and compare two captures at their first generated-token divergence."""

    left_path = left_path.expanduser().resolve()
    right_path = right_path.expanduser().resolve()
    left, left_request, left_payload, left_ids, left_logprobs = _load_capture(left_path)
    right, right_request, right_payload, right_ids, right_logprobs = _load_capture(right_path)
    if left_request != right_request or left_payload != right_payload:
        raise SpecLogprobDiagnosticError(
            "capture request bodies differ; refusing a non-controlled comparison"
        )
    shared_length = min(len(left_ids), len(right_ids))
    divergence_index = next(
        (index for index in range(shared_length) if left_ids[index] != right_ids[index]),
        shared_length if len(left_ids) != len(right_ids) else None,
    )
    identical = divergence_index is None
    window_start = max(0, (divergence_index or 0) - 8)
    window_end = (
        min(max(len(left_ids), len(right_ids)), divergence_index + 9)
        if divergence_index is not None
        else min(len(left_ids), 16)
    )
    report = {
        "comparison_type": "qwen-r9700-speculative-logprob-first-divergence",
        "requests_byte_identical": True,
        "request_body_sha256": _sha256(left_request),
        "identical_completion_token_ids": identical,
        "common_prefix_token_count": divergence_index
        if divergence_index is not None
        else len(left_ids),
        "first_divergence_index": divergence_index,
        "left": {
            "backend_id": left.get("backend_id"),
            "capture_id": left.get("capture_id"),
            "path": str(left_path),
            "token_count": len(left_ids),
            "token_id_window": {
                "start_index": window_start,
                "values": left_ids[window_start:window_end],
            },
        },
        "right": {
            "backend_id": right.get("backend_id"),
            "capture_id": right.get("capture_id"),
            "path": str(right_path),
            "token_count": len(right_ids),
            "token_id_window": {
                "start_index": window_start,
                "values": right_ids[window_start:window_end],
            },
        },
        "divergence": None,
    }
    if divergence_index is not None:
        report["divergence"] = {
            "index": divergence_index,
            "left": _side_at_divergence(
                left_ids,
                left_logprobs,
                divergence_index,
                right_ids,
                right_logprobs,
            ),
            "right": _side_at_divergence(
                right_ids,
                right_logprobs,
                divergence_index,
                left_ids,
                left_logprobs,
            ),
        }
    return report


def _description(name: str, synopsis: str, description: str) -> str:
    return f"""NAME
  {name}

SYNOPSIS
  {synopsis}

DESCRIPTION
  {description}

OPTIONS
  The accepted options are listed below."""


def _epilog(operation: str, examples: str, files: str, paths: str, exit_status: str) -> str:
    return f"""OPERATION
  {operation}

EXAMPLES
{examples}

FILES
  {files}

PATHS
  {paths}

SECURITY NOTES
  API credentials are read only from the selected environment variable and are never persisted.
  Authorization is recorded only as <redacted>. Response bodies contain model output; treat captures
  as potentially sensitive and do not render untrusted text directly in a terminal.

EXIT STATUS
  {exit_status}

AUTHORS
  Qwen R9700 inference lab contributors."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-r9700-spec-logprob",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_description(
            "qwen-r9700-spec-logprob - capture and compare one speculative logprob probe",
            "qwen-r9700-spec-logprob COMMAND [OPTIONS]",
            "Capture the exact C1 code request with token IDs and top-20 logprobs, or compare two "
            "immutable captures at their first generated-token divergence.",
        ),
        epilog=_epilog(
            "Use capture once against each already-running target-only and MTP worker, then "
            "compare the two files offline. The command never starts, stops, or reconfigures a "
            "server.",
            "  qwen-r9700-spec-logprob capture --help\n  qwen-r9700-spec-logprob compare --help",
            "The checked-in benchmarks/c1/fixture-v1.json is SHA-256 locked. Capture JSON embeds "
            "exact request/response bodies and their hashes.",
            "Capture outputs use atomic create-without-replacement semantics and mode 0444. "
            "Compare is read-only and writes its report to standard output.",
            "0 on capture success or equal token sequences; 1 when compare finds a divergence; 2 "
            "for arguments, fixture, HTTP, integrity, or evidence errors.",
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    capture_parser = commands.add_parser(
        "capture",
        help="capture one exact non-streaming code request",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_description(
            "qwen-r9700-spec-logprob capture - preserve one token/logprob response",
            "qwen-r9700-spec-logprob capture --base-url URL --model MODEL --backend-id ID "
            "--output PATH [OPTIONS]",
            "Send exactly one greedy max_tokens=160 request for the locked C1 code fixture with "
            "logprobs=true, top_logprobs=20, and return_token_ids=true.",
        ),
        epilog=_epilog(
            "Refuse an existing output before dispatch, verify the response has aligned token IDs "
            "and logprob positions, then atomically publish raw-body evidence.",
            "  qwen-r9700-spec-logprob capture \\\n"
            "    --base-url http://127.0.0.1:8000 --model qwen3.8-27b-frozenlock \\\n"
            "    --backend-id stock-vllm-target-only --output results/c1/spec-target.json",
            "Reads benchmarks/c1/fixture-v1.json by default; --fixture is accepted only when its "
            "bytes match the locked fixture SHA-256.",
            "The output parent is created when needed. Existing files, directories, and symlinks "
            "are refused without sending a request.",
            "0 after one successful request and capture; 2 for arguments, fixture, HTTP, response, "
            "or output errors.",
        ),
    )
    capture_parser.add_argument("--base-url", required=True)
    capture_parser.add_argument("--model", required=True)
    capture_parser.add_argument("--backend-id", required=True)
    capture_parser.add_argument("--output", required=True, type=Path)
    capture_parser.add_argument("--fixture", type=Path)
    capture_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    capture_parser.add_argument("--timeout", type=float, default=600.0)

    compare_parser = commands.add_parser(
        "compare",
        help="compare two captures at the first token divergence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_description(
            "qwen-r9700-spec-logprob compare - inspect first token/logprob divergence",
            "qwen-r9700-spec-logprob compare LEFT.json RIGHT.json",
            "Verify both immutable captures, require byte-identical request bodies, and report the "
            "first differing token ID with both top-20 candidate distributions.",
        ),
        epilog=_epilog(
            "Decode and hash-check both raw bodies, verify derived token hashes, find the first "
            "different generated ID, and emit a JSON report without modifying either input.",
            "  qwen-r9700-spec-logprob compare \\\n"
            "    results/c1/spec-target.json results/c1/spec-mtp3.json",
            "LEFT.json and RIGHT.json must both be captures created by this command from the "
            "locked fixture and an identical request body.",
            "Inputs may be anywhere readable. The JSON report is written only to standard output.",
            "0 when token ID sequences are identical; 1 when a first divergence is reported; 2 for "
            "arguments, integrity, format, or comparability errors.",
        ),
    )
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capture":
            document = capture(args)
            print(f"created {args.output}: {document['capture_id']}")
            return 0
        report = compare(args.left, args.right)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["identical_completion_token_ids"] else 1
    except SpecLogprobDiagnosticError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
