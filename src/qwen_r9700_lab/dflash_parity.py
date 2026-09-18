"""Immutable greedy target-versus-DFlash parity captures and comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwen_r9700_lab.config import ConfigurationError, find_project_root
from qwen_r9700_lab.spec_logprob_diagnostic import (
    CASE_ID,
    CASE_MESSAGES_SHA256,
    FIXTURE_ID,
    FIXTURE_SHA256,
    _body_record,
    _canonical_bytes,
    _decode_body_record,
    _load_locked_code_fixture,
    _sha256,
    _write_create_once,
)

CAPTURE_TYPE = "qwen-r9700-dflash-greedy-parity"
COMPARISON_TYPE = "qwen-r9700-dflash-greedy-first-divergence"
DEFAULT_MAX_TOKENS = 128
MAX_CAPTURE_TOKENS = 2048


class DFlashParityError(RuntimeError):
    """The parity capture or comparison contract was violated."""


def _chat_endpoint(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DFlashParityError(f"expected an absolute HTTP(S) base URL, got {base_url!r}")
    if parsed.query or parsed.fragment:
        raise DFlashParityError("base URL must not contain a query or fragment")
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def _greedy_payload(
    fixture: Mapping[str, Any], case: Mapping[str, Any], model: str, max_tokens: int
) -> dict[str, Any]:
    common = fixture.get("common_request")
    if not isinstance(common, dict):
        raise DFlashParityError("fixture common_request must be an object")
    payload = dict(common)
    payload.pop("stream_options", None)
    payload.update(
        {
            "frequency_penalty": 0.0,
            "logprobs": True,
            "max_tokens": max_tokens,
            "messages": case["messages"],
            "model": model,
            "n": 1,
            "presence_penalty": 0.0,
            "return_token_ids": True,
            "seed": 42,
            "stream": False,
            "temperature": 0.0,
            "top_k": 1,
            "top_logprobs": 1,
            "top_p": 1.0,
        }
    )
    return payload


def _response_token_ids(response: Mapping[str, Any]) -> tuple[dict[str, Any], list[int]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise DFlashParityError("response must contain choices[0]")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise DFlashParityError("response choices[0].message must be an object")
    token_ids = choice.get("token_ids", message.get("token_ids"))
    if not isinstance(token_ids, list) or not all(type(item) is int for item in token_ids):
        raise DFlashParityError(
            "response must contain integer choices[0].token_ids or message.token_ids"
        )
    return choice, token_ids


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise DFlashParityError(f"expected LABEL=PATH, got {value!r}")
    if any(character.isspace() for character in label):
        raise DFlashParityError(f"evidence label must not contain whitespace: {label!r}")
    return label, Path(raw_path).expanduser().resolve()


def _hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise DFlashParityError(f"cannot hash evidence file {path}: {error}") from error
    return digest.hexdigest()


def _evidence_records(category: str, values: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    labels: set[str] = set()
    for value in values:
        label, path = _parse_labeled_path(value)
        if label in labels:
            raise DFlashParityError(f"duplicate {category} evidence label: {label}")
        labels.add(label)
        if not path.is_file():
            raise DFlashParityError(f"{category} evidence is not a regular file: {path}")
        stat = path.stat()
        records.append(
            {
                "category": category,
                "label": label,
                "path": str(path),
                "sha256": _hash_file(path),
                "size_bytes": stat.st_size,
            }
        )
    return records


def _integer_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise DFlashParityError(f"telemetry {label} must be an integer list")
    return value


def _normalize_round(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DFlashParityError("every telemetry round must be an object")
    required_integers = (
        "round_index",
        "accepted_draft_count",
        "rollback_token_count",
        "sequence_length_before",
        "sequence_length_after",
    )
    integers: dict[str, int] = {}
    for name in required_integers:
        item = value.get(name)
        if type(item) is not int or item < 0:
            raise DFlashParityError(f"telemetry {name} must be a non-negative integer")
        integers[name] = item
    proposal = _integer_list(value.get("proposal_token_ids"), "proposal_token_ids")
    target = _integer_list(value.get("target_argmax_token_ids"), "target_argmax_token_ids")
    emitted = _integer_list(value.get("emitted_token_ids"), "emitted_token_ids")
    positions = _integer_list(value.get("positions"), "positions")
    accepted = integers["accepted_draft_count"]
    violations: list[str] = []
    if accepted > len(proposal):
        violations.append("accepted_draft_count_exceeds_proposal_count")
    if len(target) <= accepted:
        violations.append("target_argmax_missing_rejection_or_bonus_position")
    if len(positions) != len(target):
        violations.append("positions_and_target_argmax_lengths_differ")
    if proposal[:accepted] != target[:accepted]:
        violations.append("accepted_proposal_prefix_differs_from_target_argmax")
    expected_emitted = target[: accepted + 1]
    if emitted != expected_emitted:
        violations.append("emitted_tokens_differ_from_greedy_target_argmax")
    if integers["sequence_length_after"] - integers["sequence_length_before"] != len(emitted):
        violations.append("sequence_length_delta_differs_from_emitted_count")
    rejected = max(0, len(proposal) - accepted)
    rollback_matches_rejected = integers["rollback_token_count"] == rejected
    return {
        **integers,
        "proposal_token_ids": proposal,
        "target_argmax_token_ids": target,
        "emitted_token_ids": emitted,
        "positions": positions,
        "greedy_contract_valid": not violations,
        "greedy_contract_violations": violations,
        "rollback_matches_rejected_proposal_count": rollback_matches_rejected,
    }


def _telemetry_payload_from_json(value: object, request_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        raise DFlashParityError("telemetry JSON must be an object")
    if value.get("request_id") != request_id:
        raise DFlashParityError("telemetry request_id does not match the capture request")
    rounds = value.get("rounds")
    if rounds is None and "round" in value:
        rounds = [value["round"]]
    if not isinstance(rounds, list):
        raise DFlashParityError("telemetry JSON must contain rounds or one round")
    return [_normalize_round(item) for item in rounds]


def _telemetry_payload_from_jsonl(text: str, request_id: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise DFlashParityError(
                f"telemetry JSONL line {line_number} is invalid: {error}"
            ) from error
        if not isinstance(value, dict):
            raise DFlashParityError(f"telemetry JSONL line {line_number} must be an object")
        if value.get("request_id") == request_id:
            matches.append(_normalize_round(value.get("round")))
    if not matches:
        raise DFlashParityError(f"telemetry JSONL has no rounds for request_id {request_id!r}")
    return matches


def _load_telemetry_file(path: Path, request_id: str) -> tuple[list[dict[str, Any]], dict]:
    path = path.expanduser().resolve()
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DFlashParityError(f"cannot read telemetry {path}: {error}") from error
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        rounds = _telemetry_payload_from_jsonl(text, request_id)
        encoding = "jsonl"
    else:
        rounds = _telemetry_payload_from_json(parsed, request_id)
        encoding = "json"
    return rounds, {
        "kind": "file",
        "encoding": encoding,
        "path": str(path),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _inline_telemetry(response: Mapping[str, Any], request_id: str) -> list[dict[str, Any]] | None:
    extensions = response.get("server_extensions")
    if not isinstance(extensions, dict):
        return None
    value = extensions.get("dflash_parity")
    if value is None:
        return None
    return _telemetry_payload_from_json(value, request_id)


def _telemetry_record(
    response: Mapping[str, Any], request_id: str, telemetry_path: Path | None
) -> dict[str, Any]:
    if telemetry_path is not None:
        rounds, source = _load_telemetry_file(telemetry_path, request_id)
    else:
        rounds = _inline_telemetry(response, request_id)
        source = {"kind": "response-extension"} if rounds is not None else None
    if rounds is None:
        return {
            "available": False,
            "reason": "no telemetry file and no server_extensions.dflash_parity response",
            "rounds": [],
            "source": None,
        }
    indices = [round_record["round_index"] for round_record in rounds]
    return {
        "available": True,
        "all_rounds_greedy_contract_valid": all(
            round_record["greedy_contract_valid"] for round_record in rounds
        ),
        "round_indices_contiguous_from_zero": indices == list(range(len(rounds))),
        "rounds": rounds,
        "source": source,
    }


def _capture_fingerprint(document: Mapping[str, Any]) -> str:
    fields = {
        "backend": document["backend"],
        "capture_type": document["capture_type"],
        "completed_at": document["completed_at"],
        "derived": document["derived"],
        "elapsed_seconds": document["elapsed_seconds"],
        "fixture": document["fixture"],
        "request": document["request"],
        "response": document["response"],
        "schema_version": document["schema_version"],
        "started_at": document["started_at"],
        "telemetry": document["telemetry"],
    }
    return _sha256(_canonical_bytes(fields))


def capture(args: argparse.Namespace) -> dict[str, Any]:
    """Capture one bounded request from an already-running backend."""

    if args.backend_kind not in {"target", "dflash"}:
        raise DFlashParityError("backend-kind must be target or dflash")
    if not args.backend_id.strip() or not args.build_id.strip() or not args.model.strip():
        raise DFlashParityError("backend-id, build-id, and model must be non-empty")
    if not 1 <= args.max_tokens <= MAX_CAPTURE_TOKENS:
        raise DFlashParityError(f"max-tokens must be between 1 and {MAX_CAPTURE_TOKENS}")
    if args.timeout <= 0:
        raise DFlashParityError("timeout must be positive")
    if not args.request_id.strip():
        raise DFlashParityError("request-id must be non-empty")
    if not args.build_file or not args.config_file:
        raise DFlashParityError(
            "at least one --build-file and one --config-file are required for provenance"
        )
    destination = args.output.expanduser().absolute()

    if args.fixture is None:
        try:
            fixture_path = find_project_root() / "benchmarks" / "c1" / "fixture-v1.json"
        except ConfigurationError as error:
            raise DFlashParityError(str(error)) from error
    else:
        fixture_path = args.fixture.expanduser().resolve()
    fixture, fixture_bytes, case = _load_locked_code_fixture(fixture_path)
    payload = _greedy_payload(fixture, case, args.model, args.max_tokens)
    request_body = _canonical_bytes(payload)
    endpoint = _chat_endpoint(args.base_url)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Request-ID": args.request_id,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=request_body, headers=headers, method="POST")
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as http_response:
            response_body = http_response.read()
            status = http_response.status
            response_headers = [
                {"name": name, "value": value} for name, value in http_response.headers.raw_items()
            ]
    except urllib.error.HTTPError as error:
        error.read()
        raise DFlashParityError(f"POST {endpoint} returned HTTP status {error.code}") from error
    except (OSError, urllib.error.URLError) as error:
        raise DFlashParityError(f"POST {endpoint} failed: {error}") from error
    if status != 200:
        raise DFlashParityError(f"POST {endpoint} returned unexpected HTTP status {status}")
    completed_at = datetime.now(UTC).isoformat()
    elapsed_seconds = time.perf_counter() - started
    try:
        response_json = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DFlashParityError("HTTP response is not a UTF-8 JSON document") from error
    if not isinstance(response_json, dict):
        raise DFlashParityError("HTTP response must be a JSON object")
    choice, token_ids = _response_token_ids(response_json)
    evidence = [
        *_evidence_records("build", args.build_file),
        *_evidence_records("config", args.config_file),
    ]
    telemetry = _telemetry_record(response_json, args.request_id, args.telemetry)
    recorded_headers = [
        {"name": name, "value": "<redacted>" if name == "Authorization" else value}
        for name, value in headers.items()
    ]
    message = choice["message"]
    document: dict[str, Any] = {
        "schema_version": 1,
        "capture_type": CAPTURE_TYPE,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "started_at": started_at,
        "backend": {
            "build_id": args.build_id,
            "evidence_files": evidence,
            "id": args.backend_id,
            "kind": args.backend_kind,
        },
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
            "raw_body": _body_record(request_body),
            "request_id": args.request_id,
            "url": endpoint,
        },
        "response": {
            "headers": response_headers,
            "http_status": status,
            "raw_body": _body_record(response_body),
        },
        "derived": {
            "completion_token_ids": token_ids,
            "completion_token_ids_sha256": _sha256(_canonical_bytes(token_ids)),
            "finish_reason": choice.get("finish_reason"),
            "message_content_sha256": (
                _sha256(str(message.get("content")).encode())
                if message.get("content") is not None
                else None
            ),
            "reasoning_content_sha256": (
                _sha256(str(message.get("reasoning_content")).encode())
                if message.get("reasoning_content") is not None
                else None
            ),
        },
        "telemetry": telemetry,
    }
    document["capture_id"] = _capture_fingerprint(document)
    _write_create_once(destination, document)
    return document


def _load_capture(path: Path) -> tuple[dict[str, Any], bytes, list[int]]:
    path = path.expanduser().resolve()
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise DFlashParityError(f"cannot load capture {path}: {error}") from error
    if not isinstance(document, dict):
        raise DFlashParityError(f"capture must be an object: {path}")
    if document.get("schema_version") != 1 or document.get("capture_type") != CAPTURE_TYPE:
        raise DFlashParityError(f"not a {CAPTURE_TYPE} v1 capture: {path}")
    if document.get("capture_id") != _capture_fingerprint(document):
        raise DFlashParityError(f"capture fingerprint mismatch: {path}")
    fixture = document.get("fixture")
    if not isinstance(fixture, dict) or fixture.get("sha256") != FIXTURE_SHA256:
        raise DFlashParityError(f"capture fixture mismatch: {path}")
    request = document.get("request")
    response = document.get("response")
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise DFlashParityError(f"capture request/response is invalid: {path}")
    request_body = _decode_body_record(request.get("raw_body"), f"{path} request")
    response_body = _decode_body_record(response.get("raw_body"), f"{path} response")
    try:
        payload = json.loads(request_body)
        response_json = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise DFlashParityError(f"capture body is invalid JSON: {path}") from error
    required = {
        "frequency_penalty": 0.0,
        "logprobs": True,
        "n": 1,
        "presence_penalty": 0.0,
        "return_token_ids": True,
        "seed": 42,
        "stream": False,
        "temperature": 0.0,
        "top_k": 1,
        "top_logprobs": 1,
        "top_p": 1.0,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in required.items()
    ):
        raise DFlashParityError(f"capture request is not deterministic greedy: {path}")
    if not 1 <= payload.get("max_tokens", 0) <= MAX_CAPTURE_TOKENS:
        raise DFlashParityError(f"capture request max_tokens is out of bounds: {path}")
    if _sha256(_canonical_bytes(payload.get("messages"))) != CASE_MESSAGES_SHA256:
        raise DFlashParityError(f"capture request messages do not match fixture: {path}")
    if not isinstance(response_json, dict):
        raise DFlashParityError(f"capture response is not an object: {path}")
    _, token_ids = _response_token_ids(response_json)
    derived = document.get("derived")
    if not isinstance(derived, dict) or derived.get("completion_token_ids") != token_ids:
        raise DFlashParityError(f"capture derived token IDs mismatch raw response: {path}")
    if derived.get("completion_token_ids_sha256") != _sha256(_canonical_bytes(token_ids)):
        raise DFlashParityError(f"capture token ID hash mismatch: {path}")
    return document, request_body, token_ids


def _round_for_output_index(telemetry: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    if not telemetry.get("available"):
        return None
    offset = 0
    rounds = telemetry.get("rounds")
    if not isinstance(rounds, list):
        return None
    for round_record in rounds:
        if not isinstance(round_record, dict):
            continue
        emitted = round_record.get("emitted_token_ids")
        if not isinstance(emitted, list):
            continue
        if index < offset + len(emitted):
            local_index = index - offset
            target = round_record.get("target_argmax_token_ids", [])
            return {
                "output_index_within_round": local_index,
                "round": round_record,
                "target_argmax_at_output": (
                    target[local_index] if local_index < len(target) else None
                ),
            }
        offset += len(emitted)
    return None


def _telemetry_summary(document: Mapping[str, Any], token_ids: Sequence[int]) -> dict[str, Any]:
    telemetry = document.get("telemetry")
    if not isinstance(telemetry, dict) or not telemetry.get("available"):
        return {"available": False}
    rounds = telemetry.get("rounds", [])
    flattened = [token for item in rounds for token in item["emitted_token_ids"]]
    return {
        "available": True,
        "all_rounds_greedy_contract_valid": telemetry.get(
            "all_rounds_greedy_contract_valid"
        ),
        "emitted_tokens_match_response": flattened == list(token_ids),
        "round_count": len(rounds),
        "round_indices_contiguous_from_zero": telemetry.get("round_indices_contiguous_from_zero"),
        "total_accepted_drafts": sum(item["accepted_draft_count"] for item in rounds),
        "total_proposed_drafts": sum(len(item["proposal_token_ids"]) for item in rounds),
    }


def _evidence_by_key(document: Mapping[str, Any]) -> dict[str, str]:
    backend = document.get("backend")
    records = backend.get("evidence_files", []) if isinstance(backend, dict) else []
    return {
        f"{record['category']}:{record['label']}": record["sha256"]
        for record in records
        if isinstance(record, dict)
    }


def compare(target_path: Path, dflash_path: Path) -> dict[str, Any]:
    """Compare immutable target and DFlash captures at their first token divergence."""

    target, target_request, target_ids = _load_capture(target_path)
    dflash, dflash_request, dflash_ids = _load_capture(dflash_path)
    if target.get("backend", {}).get("kind") != "target":
        raise DFlashParityError("first capture must have backend kind target")
    if dflash.get("backend", {}).get("kind") != "dflash":
        raise DFlashParityError("second capture must have backend kind dflash")
    if target_request != dflash_request:
        raise DFlashParityError("request bodies differ; refusing a non-controlled comparison")
    shared = min(len(target_ids), len(dflash_ids))
    divergence = next(
        (index for index in range(shared) if target_ids[index] != dflash_ids[index]),
        shared if len(target_ids) != len(dflash_ids) else None,
    )
    window_start = max(0, (divergence or 0) - 8)
    window_end = (
        min(max(len(target_ids), len(dflash_ids)), divergence + 9)
        if divergence is not None
        else min(len(target_ids), 16)
    )
    target_evidence = _evidence_by_key(target)
    dflash_evidence = _evidence_by_key(dflash)
    shared_evidence = sorted(target_evidence.keys() & dflash_evidence.keys())
    divergence_record = None
    if divergence is not None:
        target_round = _round_for_output_index(target["telemetry"], divergence)
        dflash_round = _round_for_output_index(dflash["telemetry"], divergence)
        dflash_target_argmax = (
            dflash_round.get("target_argmax_at_output") if dflash_round is not None else None
        )
        divergence_record = {
            "index": divergence,
            "target_token_id": target_ids[divergence] if divergence < len(target_ids) else None,
            "dflash_token_id": dflash_ids[divergence] if divergence < len(dflash_ids) else None,
            "target_round": target_round,
            "dflash_round": dflash_round,
            "dflash_emitted_matches_its_target_argmax": (
                dflash_target_argmax == dflash_ids[divergence]
                if divergence < len(dflash_ids) and dflash_target_argmax is not None
                else None
            ),
            "dflash_target_argmax_matches_target_reference": (
                dflash_target_argmax == target_ids[divergence]
                if divergence < len(target_ids) and dflash_target_argmax is not None
                else None
            ),
        }
    return {
        "schema_version": 1,
        "comparison_type": COMPARISON_TYPE,
        "requests_byte_identical": True,
        "request_body_sha256": _sha256(target_request),
        "identical_completion_token_ids": divergence is None,
        "common_prefix_token_count": divergence if divergence is not None else len(target_ids),
        "first_divergence_index": divergence,
        "target": {
            "backend": target["backend"],
            "capture_id": target["capture_id"],
            "path": str(target_path.expanduser().resolve()),
            "telemetry": _telemetry_summary(target, target_ids),
            "token_count": len(target_ids),
            "token_id_window": target_ids[window_start:window_end],
        },
        "dflash": {
            "backend": dflash["backend"],
            "capture_id": dflash["capture_id"],
            "path": str(dflash_path.expanduser().resolve()),
            "telemetry": _telemetry_summary(dflash, dflash_ids),
            "token_count": len(dflash_ids),
            "token_id_window": dflash_ids[window_start:window_end],
        },
        "shared_evidence_hashes_match": {
            key: target_evidence[key] == dflash_evidence[key] for key in shared_evidence
        },
        "divergence": divergence_record,
    }


def _description(name: str, synopsis: str, description: str) -> str:
    return f"""NAME
  {name}

SYNOPSIS
  {synopsis}

DESCRIPTION
  {description}

OPTIONS
  The accepted options are listed below."""


def _epilog(operation: str, examples: str) -> str:
    return f"""OPERATION
  {operation}

EXAMPLES
{examples}

FILES
  Capture output is immutable JSON. Optional telemetry input is JSON or request-filtered JSONL.

PATHS
  Evidence paths are resolved before hashing. Their contents are not copied into the capture.

SECURITY NOTES
  API credentials are read only from the selected environment variable and recorded as <redacted>.
  Captures contain raw model output and may contain sensitive prompt material.

EXIT STATUS
  0 indicates success. 2 indicates invalid arguments. 1 indicates capture or comparison failure.

AUTHORS
  Qwen R9700 inference lab contributors."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-r9700-dflash-parity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_description(
            "qwen-r9700-dflash-parity - preserve and compare deterministic DFlash evidence",
            "qwen-r9700-dflash-parity COMMAND [OPTIONS]",
            "Capture one bounded greedy request from an already-running target or DFlash backend, "
            "then compare immutable captures by raw completion token ID.",
        ),
        epilog=_epilog(
            "Run capture once per lane without changing either server. Use the same model and "
            "max-tokens. Optional per-round telemetry localizes whether a divergence came from the "
            "proposal, target argmax, emitted token, position, acceptance, or rollback state.",
            "  qwen-r9700-dflash-parity capture --help\n"
            "  qwen-r9700-dflash-parity compare --help",
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser(
        "capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_description(
            "qwen-r9700-dflash-parity capture - capture one lane",
            "qwen-r9700-dflash-parity capture [OPTIONS]",
            "Send the locked deterministic request and publish one create-once evidence file.",
        ),
        epilog=_epilog(
            "The command never starts, stops, or reconfigures a server. JSONL telemetry records "
            "must contain request_id and round fields; JSON telemetry contains request_id and "
            "rounds.",
            "  qwen-r9700-dflash-parity capture --backend-kind target --backend-id target-v1 "
            "--build-id d626108b --base-url http://127.0.0.1:8000 --model model "
            "--request-id parity-target-1 --build-file wheel=/tmp/vllm.whl "
            "--config-file launcher=/tmp/launcher --output target.json",
        ),
    )
    capture_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    capture_parser.add_argument("--backend-id", required=True)
    capture_parser.add_argument("--backend-kind", choices=("target", "dflash"), required=True)
    capture_parser.add_argument("--base-url", required=True)
    capture_parser.add_argument("--build-id", required=True)
    capture_parser.add_argument("--build-file", action="append", default=[], metavar="LABEL=PATH")
    capture_parser.add_argument("--config-file", action="append", default=[], metavar="LABEL=PATH")
    capture_parser.add_argument("--fixture", type=Path)
    capture_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    capture_parser.add_argument("--model", required=True)
    capture_parser.add_argument("--output", required=True, type=Path)
    capture_parser.add_argument("--request-id", required=True)
    capture_parser.add_argument("--telemetry", type=Path)
    capture_parser.add_argument("--timeout", type=float, default=600.0)
    capture_parser.set_defaults(handler=_capture_command)

    compare_parser = subparsers.add_parser(
        "compare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_description(
            "qwen-r9700-dflash-parity compare - find the first target/DFlash divergence",
            "qwen-r9700-dflash-parity compare [OPTIONS] TARGET DFLASH",
            "Validate two captures and report raw-token parity plus optional round telemetry.",
        ),
        epilog=_epilog(
            "Comparison refuses different request bodies and verifies the immutable capture "
            "hashes.",
            "  qwen-r9700-dflash-parity compare target.json dflash.json --output parity.json",
        ),
    )
    compare_parser.add_argument("target", type=Path)
    compare_parser.add_argument("dflash", type=Path)
    compare_parser.add_argument("--output", type=Path)
    compare_parser.set_defaults(handler=_compare_command)
    return parser


def _capture_command(args: argparse.Namespace) -> int:
    document = capture(args)
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    report = compare(args.target, args.dflash)
    if args.output is not None:
        _write_create_once(args.output.expanduser().absolute(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["identical_completion_token_ids"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except DFlashParityError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
