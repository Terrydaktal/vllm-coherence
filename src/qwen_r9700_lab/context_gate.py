"""Deterministic long-context retrieval and repeated-prefix equivalence gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import tokenizers
from tokenizers import Tokenizer


class ContextGateError(RuntimeError):
    """The context fixture, request protocol, or immutable evidence contract failed."""


class TokenCounter(Protocol):
    """Minimal interface used by deterministic prompt construction."""

    def count(self, text: str) -> int:
        """Return the number of model tokenizer tokens in ``text`` without special tokens."""


class JsonTokenizerCounter:
    """Count text with an exact Hugging Face ``tokenizer.json`` file."""

    def __init__(self, path: Path) -> None:
        try:
            self._tokenizer = Tokenizer.from_file(str(path))
        except Exception as error:  # tokenizers raises several implementation-specific exceptions
            raise ContextGateError(f"cannot load tokenizer JSON {path}: {error}") from error

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_fixture(path: Path) -> dict[str, Any]:
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContextGateError(f"cannot load fixture {path}: {error}") from error
    if not isinstance(fixture, dict) or fixture.get("schema_version") != 1:
        raise ContextGateError("context fixture must be a schema_version 1 JSON object")

    protocol = fixture.get("protocol")
    generation = fixture.get("generation")
    request = fixture.get("request")
    queries = fixture.get("queries")
    if not isinstance(protocol, dict):
        raise ContextGateError("fixture protocol must be an object")
    if protocol.get("client_max_in_flight") != 1:
        raise ContextGateError("fixture must require client_max_in_flight=1")
    if protocol.get("required_server_max_sequences") != 1:
        raise ContextGateError("fixture must require required_server_max_sequences=1")
    if protocol.get("required_prefix_cache") is not True:
        raise ContextGateError("fixture must require prefix caching")
    if not isinstance(generation, dict):
        raise ContextGateError("fixture generation must be an object")
    needles = generation.get("needles")
    words = generation.get("filler_words")
    if not isinstance(needles, list) or len(needles) < 3:
        raise ContextGateError("fixture must define at least three needles")
    if (
        not isinstance(words, list)
        or len(words) < 8
        or not all(isinstance(word, str) and word for word in words)
    ):
        raise ContextGateError("fixture must define at least eight non-empty filler words")
    needle_ids: list[str] = []
    depths: list[float] = []
    values: list[str] = []
    for needle in needles:
        if not isinstance(needle, dict):
            raise ContextGateError("each needle must be an object")
        needle_id = needle.get("id")
        value = needle.get("value")
        depth = needle.get("target_depth")
        if not isinstance(needle_id, str) or not needle_id:
            raise ContextGateError("each needle id must be non-empty")
        if not isinstance(value, str) or not value:
            raise ContextGateError(f"needle {needle_id} must have a non-empty value")
        if not isinstance(depth, (int, float)) or not 0 < float(depth) < 1:
            raise ContextGateError(f"needle {needle_id} depth must be between zero and one")
        needle_ids.append(needle_id)
        values.append(value)
        depths.append(float(depth))
    if len(set(needle_ids)) != len(needle_ids) or len(set(values)) != len(values):
        raise ContextGateError("needle ids and values must be unique")
    if depths != sorted(depths) or len(set(depths)) != len(depths):
        raise ContextGateError("needle depths must be unique and strictly increasing")

    if not isinstance(queries, list) or len(queries) != 2:
        raise ContextGateError("fixture must define exactly two shared-prefix queries")
    query_ids = []
    for query in queries:
        if not isinstance(query, dict):
            raise ContextGateError("each query must be an object")
        if not isinstance(query.get("id"), str) or not isinstance(query.get("instruction"), str):
            raise ContextGateError("each query needs string id and instruction fields")
        query_ids.append(query["id"])
    if len(set(query_ids)) != 2:
        raise ContextGateError("query ids must be unique")
    if not isinstance(request, dict) or request.get("stream") is not True:
        raise ContextGateError("fixture request must enable streaming")
    if request.get("n") != 1 or request.get("temperature") != 0.0:
        raise ContextGateError("fixture request must require one greedy completion")
    stream_options = request.get("stream_options")
    if not isinstance(stream_options, dict) or stream_options.get("include_usage") is not True:
        raise ContextGateError("fixture request must request streaming usage")
    if fixture.get("request_order") != [
        f"{queries[0]['id']}-initial",
        f"{queries[1]['id']}-shared-prefix",
        f"{queries[0]['id']}-repeat",
    ]:
        raise ContextGateError("fixture request_order does not match the two declared queries")
    for field in (
        "system_message",
        "calibration_user_message",
        "shared_prefix_header",
    ):
        if not isinstance(fixture.get(field), str) or not fixture[field]:
            raise ContextGateError(f"fixture {field} must be a non-empty string")
    return fixture


def _filler_line(seed: str, index: int, words: Sequence[str]) -> str:
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    selected = [words[value % len(words)] for value in digest[:12]]
    checksum = hashlib.sha256(f"record:{seed}:{index}".encode()).hexdigest()[:16]
    return f"Record {index:08d} | {' '.join(selected)} | checksum {checksum}.\n"


def _needle_marker(needle: Mapping[str, Any]) -> str:
    return f"AUTHORITATIVE_NEEDLE id={needle['id']} value={needle['value']}\n"


def _query_suffix(query: Mapping[str, Any]) -> str:
    return f"\nQUERY id={query['id']}\n{query['instruction']}\n"


def _render_shared_prefix(
    fixture: Mapping[str, Any],
    cache_namespace: str,
    sections: Sequence[Sequence[str]],
    padding: str,
) -> tuple[str, str]:
    needles = fixture["generation"]["needles"]
    corpus_parts: list[str] = []
    for index, needle in enumerate(needles):
        corpus_parts.extend(sections[index])
        corpus_parts.append(_needle_marker(needle))
    corpus_parts.extend(sections[-1])
    corpus_parts.append(padding)
    corpus = "".join(corpus_parts)
    shared = (
        f"CACHE_NAMESPACE {cache_namespace}\n"
        f"{fixture['shared_prefix_header']}\n"
        "BEGIN_CORPUS\n"
        f"{corpus}"
        "END_CORPUS\n"
    )
    return shared, corpus


def _fill_sections(
    *,
    fixture: Mapping[str, Any],
    counter: TokenCounter,
    target_filler_tokens: int,
) -> list[list[str]]:
    needles = fixture["generation"]["needles"]
    depths = [float(item["target_depth"]) for item in needles]
    proportions = [depths[0], *[right - left for left, right in pairwise(depths)]]
    proportions.append(1.0 - depths[-1])
    words = fixture["generation"]["filler_words"]
    seed = fixture["generation"]["seed"]
    sections: list[list[str]] = [[] for _ in proportions]
    line_index = 0
    for section_index, proportion in enumerate(proportions):
        budget = max(0, int(target_filler_tokens * proportion))
        used = 0
        while True:
            line = _filler_line(seed, line_index, words)
            line_tokens = counter.count(line)
            if used + line_tokens > budget:
                break
            sections[section_index].append(line)
            used += line_tokens
            line_index += 1
    return sections


def _padding_for_exact_count(
    *,
    fixture: Mapping[str, Any],
    cache_namespace: str,
    sections: Sequence[Sequence[str]],
    counter: TokenCounter,
    target_tokens: int,
) -> tuple[str, str, str]:
    shared, corpus = _render_shared_prefix(fixture, cache_namespace, sections, "")
    base_tokens = counter.count(shared)
    if base_tokens > target_tokens:
        raise ContextGateError("internal prompt construction exceeded the aligned prefix target")
    if base_tokens == target_tokens:
        return shared, corpus, ""

    gap = target_tokens - base_tokens
    candidates = (" lattice", " ledger", " amber", " context", " 0", "x")
    for candidate in candidates:
        low = 0
        high = max(32, gap * 4 + 32)
        while low <= high:
            repetitions = (low + high) // 2
            padding = f"PADDING_RECORD{candidate * repetitions}\n"
            candidate_shared, candidate_corpus = _render_shared_prefix(
                fixture, cache_namespace, sections, padding
            )
            count = counter.count(candidate_shared)
            if count == target_tokens:
                return candidate_shared, candidate_corpus, padding
            if count < target_tokens:
                low = repetitions + 1
            else:
                high = repetitions - 1
        for repetitions in range(max(0, high - 8), low + 9):
            padding = f"PADDING_RECORD{candidate * repetitions}\n"
            candidate_shared, candidate_corpus = _render_shared_prefix(
                fixture, cache_namespace, sections, padding
            )
            if counter.count(candidate_shared) == target_tokens:
                return candidate_shared, candidate_corpus, padding
    raise ContextGateError(
        f"could not construct an exact {target_tokens}-token shared prefix with stable padding"
    )


def _measure_needle_positions(
    fixture: Mapping[str, Any], counter: TokenCounter, shared: str, corpus: str
) -> list[dict[str, Any]]:
    corpus_tokens = counter.count(corpus)
    tolerance = float(fixture["protocol"]["needle_depth_tolerance"])
    positions = []
    for needle in fixture["generation"]["needles"]:
        marker = _needle_marker(needle)
        if corpus.count(marker) != 1 or shared.count(needle["value"]) != 1:
            raise ContextGateError(f"needle {needle['id']} is not unique in generated content")
        character_offset = corpus.index(marker)
        token_offset = counter.count(corpus[:character_offset])
        actual_depth = token_offset / corpus_tokens
        error = abs(actual_depth - float(needle["target_depth"]))
        positions.append(
            {
                "id": needle["id"],
                "value": needle["value"],
                "target_depth": float(needle["target_depth"]),
                "actual_depth": actual_depth,
                "absolute_depth_error": error,
                "within_tolerance": error <= tolerance,
                "corpus_token_offset": token_offset,
                "corpus_character_offset": character_offset,
            }
        )
    return positions


def build_prompt_set(
    *,
    fixture: Mapping[str, Any],
    counter: TokenCounter,
    cache_namespace: str,
    target_prompt_tokens: int,
    calibrated_chat_overhead_tokens: int,
    prefix_alignment_tokens: int,
) -> dict[str, Any]:
    """Build exact aligned shared content and measure every needle's realized depth."""

    if (
        not cache_namespace
        or len(cache_namespace) > 128
        or any(character in "\r\n\x00" for character in cache_namespace)
    ):
        raise ContextGateError("cache namespace must be 1-128 characters without controls/newlines")
    if prefix_alignment_tokens < 1 or prefix_alignment_tokens > 4096:
        raise ContextGateError("prefix alignment must be between 1 and 4096 tokens")
    system_tokens = counter.count(fixture["system_message"])
    suffixes = [_query_suffix(query) for query in fixture["queries"]]
    suffix_token_reserve = max(counter.count(suffix) for suffix in suffixes) + 4
    desired_user_tokens = target_prompt_tokens - calibrated_chat_overhead_tokens - system_tokens
    if desired_user_tokens <= suffix_token_reserve:
        raise ContextGateError("target prompt budget is too small after calibrated chat overhead")
    unaligned_shared_target = desired_user_tokens - suffix_token_reserve
    shared_target = (
        round(unaligned_shared_target / prefix_alignment_tokens) * prefix_alignment_tokens
    )

    empty_sections = [[] for _ in range(len(fixture["generation"]["needles"]) + 1)]
    empty_shared, _ = _render_shared_prefix(fixture, cache_namespace, empty_sections, "")
    marker_and_header_tokens = counter.count(empty_shared)
    target_filler = shared_target - marker_and_header_tokens - prefix_alignment_tokens
    if target_filler < 1:
        raise ContextGateError("target prompt budget is too small for the context fixture")
    sections = _fill_sections(
        fixture=fixture,
        counter=counter,
        target_filler_tokens=target_filler,
    )
    shared, corpus = _render_shared_prefix(fixture, cache_namespace, sections, "")
    while counter.count(shared) > shared_target:
        removable = next(
            (section for section in reversed(sections) if section),
            None,
        )
        if removable is None:
            raise ContextGateError("could not reduce generated prefix to the aligned target")
        removable.pop()
        shared, corpus = _render_shared_prefix(fixture, cache_namespace, sections, "")
    padding = ""
    needle_positions: list[dict[str, Any]] = []
    for _iteration in range(8):
        shared, corpus, padding = _padding_for_exact_count(
            fixture=fixture,
            cache_namespace=cache_namespace,
            sections=sections,
            counter=counter,
            target_tokens=shared_target,
        )
        needle_positions = _measure_needle_positions(fixture, counter, shared, corpus)
        if all(position["within_tolerance"] for position in needle_positions):
            break
        present_lines = [line for section in sections for line in section]
        if not present_lines:
            raise ContextGateError("not enough filler records to place needles at requested depths")
        average_line_tokens = max(
            1.0,
            sum(counter.count(line) for line in present_lines) / len(present_lines),
        )
        corpus_tokens = counter.count(corpus)
        for index, position in enumerate(needle_positions):
            signed_error_tokens = (
                position["target_depth"] - position["actual_depth"]
            ) * corpus_tokens
            lines_to_move = max(1, round(abs(signed_error_tokens) / average_line_tokens))
            if signed_error_tokens > 0:
                moved = sections[index + 1][:lines_to_move]
                del sections[index + 1][: len(moved)]
                sections[index].extend(moved)
            elif signed_error_tokens < 0:
                moved = sections[index][-lines_to_move:]
                if moved:
                    del sections[index][-len(moved) :]
                    sections[index + 1][0:0] = moved
        shared_without_padding, _ = _render_shared_prefix(fixture, cache_namespace, sections, "")
        while counter.count(shared_without_padding) > shared_target:
            removable = next((section for section in reversed(sections) if section), None)
            if removable is None:
                break
            removable.pop()
            shared_without_padding, _ = _render_shared_prefix(
                fixture, cache_namespace, sections, ""
            )
    if not all(position["within_tolerance"] for position in needle_positions):
        raise ContextGateError("generated needle depth exceeds fixture tolerance")

    corpus_tokens = counter.count(corpus)

    prompts = []
    for query, suffix in zip(fixture["queries"], suffixes, strict=True):
        content = shared + suffix
        local_user_tokens = counter.count(content)
        predicted_prompt_tokens = (
            calibrated_chat_overhead_tokens + system_tokens + local_user_tokens
        )
        prompts.append(
            {
                "query_id": query["id"],
                "content": content,
                "content_sha256": _sha256_bytes(content.encode()),
                "local_user_tokens": local_user_tokens,
                "predicted_server_prompt_tokens": predicted_prompt_tokens,
                "target_prompt_token_delta": predicted_prompt_tokens - target_prompt_tokens,
            }
        )

    return {
        "target_prompt_tokens": target_prompt_tokens,
        "calibrated_chat_overhead_tokens": calibrated_chat_overhead_tokens,
        "system_message_tokens": system_tokens,
        "prefix_alignment_tokens": prefix_alignment_tokens,
        "shared_prefix": shared,
        "shared_prefix_sha256": _sha256_bytes(shared.encode()),
        "shared_prefix_utf8_bytes": len(shared.encode()),
        "shared_prefix_local_tokens": counter.count(shared),
        "shared_prefix_alignment_remainder": counter.count(shared) % prefix_alignment_tokens,
        "corpus_sha256": _sha256_bytes(corpus.encode()),
        "corpus_utf8_bytes": len(corpus.encode()),
        "corpus_local_tokens": corpus_tokens,
        "padding_sha256": _sha256_bytes(padding.encode()),
        "padding_local_tokens": counter.count(padding),
        "needles": needle_positions,
        "prompts": prompts,
    }


def _authorized_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ContextGateError(f"expected an absolute HTTP(S) URL, got {url!r}")
    return url


def _request_headers(api_key: str | None, request_id: str) -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _reasoning_fragment(delta: Mapping[str, Any]) -> str:
    values = []
    for field in ("reasoning", "reasoning_content"):
        value = delta.get(field)
        if value is not None and not isinstance(value, str):
            raise ContextGateError(f"streamed {field} must be a string or null")
        if value:
            values.append(value)
    if len(set(values)) > 1:
        raise ContextGateError("streamed reasoning fields conflict")
    return values[0] if values else ""


def _stream_chat_request(
    *,
    url: str,
    payload: Mapping[str, Any],
    request_id: str,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    request_bytes = _canonical_bytes(payload)
    request = urllib.request.Request(
        _authorized_url(url),
        data=request_bytes,
        headers=_request_headers(api_key, request_id),
        method="POST",
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    first_output_ns: int | None = None
    wire_parts: list[bytes] = []
    chunk_count = 0
    started_at = _utc_now()
    started_ns = time.perf_counter_ns()
    response_status: int | None = None
    response_headers: dict[str, str] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_status = response.status
            for name in ("Content-Type", "Date", "Server", "X-Request-ID"):
                value = response.headers.get(name)
                if value is not None:
                    response_headers[name.lower()] = value
            for raw_line in response:
                wire_parts.append(raw_line)
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as error:
                    raise ContextGateError(f"invalid SSE JSON: {error}") from error
                if not isinstance(chunk, dict) or chunk.get("error"):
                    raise ContextGateError(f"server returned invalid/error SSE chunk: {chunk}")
                chunk_count += 1
                if isinstance(chunk.get("usage"), dict):
                    usage = copy.deepcopy(chunk["usage"])
                for choice in chunk.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict):
                        raise ContextGateError("streamed choice delta must be an object")
                    content = delta.get("content")
                    if content is not None and not isinstance(content, str):
                        raise ContextGateError("streamed content must be a string or null")
                    reasoning = _reasoning_fragment(delta)
                    if content:
                        content_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    if (content or reasoning) and first_output_ns is None:
                        first_output_ns = time.perf_counter_ns()
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise ContextGateError(f"POST {url} failed with HTTP {error.code}: {body}") from error
    except (OSError, UnicodeDecodeError, urllib.error.URLError) as error:
        raise ContextGateError(f"POST {url} failed: {error}") from error
    completed_ns = time.perf_counter_ns()
    completed_at = _utc_now()
    wire = b"".join(wire_parts)
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    if usage is None or not isinstance(usage.get("prompt_tokens"), int):
        raise ContextGateError("server did not return integer usage.prompt_tokens")
    total_seconds = (completed_ns - started_ns) / 1_000_000_000
    ttft_seconds = (
        (first_output_ns - started_ns) / 1_000_000_000 if first_output_ns is not None else None
    )
    return {
        "request_id": request_id,
        "request": copy.deepcopy(dict(payload)),
        "request_sha256": _sha256_bytes(request_bytes),
        "request_utf8_bytes": len(request_bytes),
        "request_headers": {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "x-request-id": request_id,
            "authorization": "Bearer <redacted>" if api_key else None,
        },
        "response": {
            "http_status": response_status,
            "headers": response_headers,
            "finish_reason": finish_reason,
            "usage": usage,
            "content": content,
            "reasoning_content": reasoning,
            "content_sha256": _sha256_bytes(content.encode()),
            "reasoning_sha256": _sha256_bytes(reasoning.encode()),
            "wire_sha256": _sha256_bytes(wire),
            "wire_bytes": len(wire),
            "sse_chunk_count": chunk_count,
        },
        "timing": {
            "started_at": started_at,
            "completed_at": completed_at,
            "total_seconds": total_seconds,
            "ttft_seconds": ttft_seconds,
        },
    }


def _expected_document(
    needles: Sequence[Mapping[str, Any]], *, reverse: bool = False
) -> dict[str, Any]:
    ordered = list(reversed(needles)) if reverse else list(needles)
    return {"needles": [{"id": item["id"], "value": item["value"]} for item in ordered]}


def evaluate_retrieval(response: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    content = response.get("content")
    reasoning = response.get("reasoning_content")
    finish_reason = response.get("finish_reason")
    parsed: Any = None
    parse_error: str | None = None
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            parse_error = str(error)
    checks = {
        "finish_reason_stop": finish_reason == "stop",
        "reasoning_empty": reasoning == "",
        "content_is_json": parse_error is None and parsed is not None,
        "exact_document": parsed == expected,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected": copy.deepcopy(dict(expected)),
        "parsed": parsed,
        "parse_error": parse_error,
    }


def _cached_tokens(response: Mapping[str, Any]) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return None
    value = details.get("cached_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def evaluate_prefix_equivalence(
    initial: Mapping[str, Any],
    repeated: Mapping[str, Any],
    alternate: Mapping[str, Any],
    *,
    shared_prefix_local_tokens: int,
    prefix_alignment_tokens: int,
    require_reported_cache_hit: bool,
) -> dict[str, Any]:
    initial_response = initial["response"]
    repeated_response = repeated["response"]
    alternate_response = alternate["response"]
    cached = {
        "initial": _cached_tokens(initial_response),
        "alternate": _cached_tokens(alternate_response),
        "repeat": _cached_tokens(repeated_response),
    }
    reported_hits = [
        value
        for name, value in cached.items()
        if name in {"alternate", "repeat"} and isinstance(value, int)
    ]
    minimum_reported_cached_tokens = max(
        1, shared_prefix_local_tokens - 2 * prefix_alignment_tokens
    )
    cache_observation_passed = (
        max(reported_hits, default=0) >= minimum_reported_cached_tokens
        if require_reported_cache_hit
        else None
    )
    checks = {
        "identical_primary_request_bytes": initial["request_sha256"] == repeated["request_sha256"],
        "identical_primary_content_bytes": initial_response["content_sha256"]
        == repeated_response["content_sha256"],
        "identical_primary_reasoning_bytes": initial_response["reasoning_sha256"]
        == repeated_response["reasoning_sha256"],
        "identical_primary_finish_reason": initial_response["finish_reason"]
        == repeated_response["finish_reason"],
        "identical_primary_prompt_usage": initial_response["usage"].get("prompt_tokens")
        == repeated_response["usage"].get("prompt_tokens"),
    }
    if require_reported_cache_hit:
        checks["reported_cache_hit"] = cache_observation_passed is True
    initial_total = initial["timing"]["total_seconds"]
    repeated_total = repeated["timing"]["total_seconds"]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "shared_prefix_local_tokens": shared_prefix_local_tokens,
        "minimum_reported_cached_tokens": minimum_reported_cached_tokens,
        "reported_cached_tokens": cached,
        "cache_observation_required": require_reported_cache_hit,
        "cache_observation_passed": cache_observation_passed,
        "initial_total_seconds": initial_total,
        "alternate_total_seconds": alternate["timing"]["total_seconds"],
        "repeat_total_seconds": repeated_total,
        "repeat_to_initial_total_time_ratio": (
            repeated_total / initial_total if initial_total > 0 else None
        ),
    }


def _write_create_once(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ContextGateError(f"refusing to replace existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".qwen-context-",
            delete=False,
        ) as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise ContextGateError(f"refusing to replace existing result: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_evidence_hash(document: Mapping[str, Any]) -> bool:
    expected = document.get("evidence_sha256")
    unsigned = {key: value for key, value in document.items() if key != "evidence_sha256"}
    return isinstance(expected, str) and expected == _sha256_bytes(_canonical_bytes(unsigned))


def _payload(
    fixture: Mapping[str, Any], model: str, user_content: str, *, calibration: bool = False
) -> dict[str, Any]:
    payload = copy.deepcopy(fixture["request"])
    if calibration:
        payload["max_tokens"] = 8
    payload.update(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": fixture["system_message"]},
                {"role": "user", "content": user_content},
            ],
        }
    )
    return payload


def run_context_gate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ContextGateError(f"refusing to replace existing result: {output}")
    fixture_path = args.fixture.expanduser().resolve()
    fixture = _load_fixture(fixture_path)
    fixture_bytes = fixture_path.read_bytes()
    tokenizer_path = args.tokenizer_json.expanduser().resolve()
    try:
        tokenizer_bytes = tokenizer_path.read_bytes()
    except OSError as error:
        raise ContextGateError(f"cannot read tokenizer JSON {tokenizer_path}: {error}") from error
    counter = JsonTokenizerCounter(tokenizer_path)
    command_path = args.server_command_file.expanduser().resolve()
    try:
        command_bytes = command_path.read_bytes()
        command = command_bytes.decode("utf-8").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise ContextGateError(
            f"cannot read server command evidence {command_path}: {error}"
        ) from error
    if not command:
        raise ContextGateError("server command evidence is empty")

    minimum = int(fixture["protocol"]["minimum_target_prompt_tokens"])
    if args.target_prompt_tokens < minimum:
        raise ContextGateError(f"target prompt tokens must be at least {minimum}")
    max_completion_tokens = int(fixture["request"]["max_tokens"])
    prompt_tolerance = int(fixture["protocol"]["prompt_token_tolerance"])
    if (
        args.target_prompt_tokens + prompt_tolerance + max_completion_tokens
        > args.server_max_model_len
    ):
        raise ContextGateError(
            "target prompt plus tolerance and completion allowance exceeds server max model length"
        )
    alignment = args.prefix_alignment_tokens or int(
        fixture["protocol"]["default_prefix_alignment_tokens"]
    )
    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    _authorized_url(endpoint)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None

    started_at = _utc_now()
    calibration_payload = _payload(
        fixture,
        args.model,
        fixture["calibration_user_message"],
        calibration=True,
    )
    calibration = _stream_chat_request(
        url=endpoint,
        payload=calibration_payload,
        request_id="context-calibration",
        timeout=args.timeout,
        api_key=api_key,
    )
    calibration_prompt_tokens = calibration["response"]["usage"]["prompt_tokens"]
    local_calibration_tokens = counter.count(fixture["system_message"]) + counter.count(
        fixture["calibration_user_message"]
    )
    calibrated_overhead = calibration_prompt_tokens - local_calibration_tokens
    if calibrated_overhead < 0:
        raise ContextGateError("calibrated chat-template overhead is negative")

    generated = build_prompt_set(
        fixture=fixture,
        counter=counter,
        cache_namespace=args.cache_namespace,
        target_prompt_tokens=args.target_prompt_tokens,
        calibrated_chat_overhead_tokens=calibrated_overhead,
        prefix_alignment_tokens=alignment,
    )
    prompt_by_id = {prompt["query_id"]: prompt for prompt in generated["prompts"]}
    primary_id = fixture["queries"][0]["id"]
    alternate_id = fixture["queries"][1]["id"]
    request_plan = (
        (f"{primary_id}-initial", primary_id),
        (f"{alternate_id}-shared-prefix", alternate_id),
        (f"{primary_id}-repeat", primary_id),
    )
    measurements = []
    expected_primary = _expected_document(fixture["generation"]["needles"])
    expected_alternate = _expected_document(fixture["generation"]["needles"], reverse=True)
    for phase, query_id in request_plan:
        request_payload = _payload(
            fixture,
            args.model,
            prompt_by_id[query_id]["content"],
        )
        measurement = _stream_chat_request(
            url=endpoint,
            payload=request_payload,
            request_id=f"context-{phase}",
            timeout=args.timeout,
            api_key=api_key,
        )
        expected = expected_primary if query_id == primary_id else expected_alternate
        measurement.update(
            {
                "phase": phase,
                "query_id": query_id,
                "retrieval": evaluate_retrieval(measurement["response"], expected),
            }
        )
        measurements.append(measurement)

    tolerance = prompt_tolerance
    token_budget_rows = []
    for measurement in measurements:
        query_prompt = prompt_by_id[measurement["query_id"]]
        actual = measurement["response"]["usage"]["prompt_tokens"]
        predicted = query_prompt["predicted_server_prompt_tokens"]
        token_budget_rows.append(
            {
                "phase": measurement["phase"],
                "target_prompt_tokens": args.target_prompt_tokens,
                "predicted_prompt_tokens": predicted,
                "actual_prompt_tokens": actual,
                "actual_target_delta": actual - args.target_prompt_tokens,
                "actual_prediction_delta": actual - predicted,
                "within_target_tolerance": abs(actual - args.target_prompt_tokens) <= tolerance,
                "within_calibration_tolerance": abs(actual - predicted) <= tolerance,
                "within_server_max_model_len": actual + max_completion_tokens
                <= args.server_max_model_len,
            }
        )
    token_budget_gate = {
        "passed": all(
            row["within_target_tolerance"]
            and row["within_calibration_tolerance"]
            and row["within_server_max_model_len"]
            for row in token_budget_rows
        ),
        "tolerance_tokens": tolerance,
        "rows": token_budget_rows,
    }
    retrieval_gate = {
        "passed": all(item["retrieval"]["passed"] for item in measurements),
        "results": [{"phase": item["phase"], **item["retrieval"]} for item in measurements],
    }
    prefix_gate = evaluate_prefix_equivalence(
        measurements[0],
        measurements[2],
        measurements[1],
        shared_prefix_local_tokens=generated["shared_prefix_local_tokens"],
        prefix_alignment_tokens=alignment,
        require_reported_cache_hit=args.require_reported_cache_hit,
    )
    gates = {
        "token_budget": token_budget_gate,
        "needle_retrieval": retrieval_gate,
        "prefix_equivalence": prefix_gate,
    }
    gates["overall_passed"] = all(gate["passed"] for gate in gates.values())
    document: dict[str, Any] = {
        "schema_version": 1,
        "result_type": "long-context-retrieval-prefix-equivalence",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "backend_id": args.backend_id,
        "model": args.model,
        "endpoint": endpoint,
        "fixture": {
            "path": str(fixture_path),
            "sha256": _sha256_bytes(fixture_bytes),
            "id": fixture["id"],
            "document": fixture,
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": _sha256_bytes(tokenizer_bytes),
            "implementation": "tokenizers.Tokenizer.from_file",
            "tokenizers_version": tokenizers.__version__,
            "add_special_tokens": False,
        },
        "server_command_evidence": {
            "path": str(command_path),
            "sha256": _sha256_bytes(command_bytes),
            "command": command,
        },
        "protocol": {
            "client_max_in_flight": 1,
            "single_threaded_request_loop": True,
            "request_order": [phase for phase, _query_id in request_plan],
            "server_max_sequences_attested": args.server_max_sequences,
            "server_max_model_len_attested": args.server_max_model_len,
            "prefix_cache_enabled_attested": args.prefix_cache_enabled,
            "cache_namespace": args.cache_namespace,
        },
        "calibration": {
            "method": "server-usage-minus-local-content-v1",
            "assumption": (
                "For the same system/user role layout, chat-template framing is a constant token "
                "overhead. Every measured request verifies the prediction against server usage."
            ),
            "local_system_plus_user_tokens": local_calibration_tokens,
            "server_prompt_tokens": calibration_prompt_tokens,
            "calibrated_chat_overhead_tokens": calibrated_overhead,
            "measurement": calibration,
        },
        "generated": generated,
        "measurements": measurements,
        "gates": gates,
    }
    document["evidence_sha256"] = _sha256_bytes(_canonical_bytes(document))
    _write_create_once(output, document)
    return document


def build_parser() -> argparse.ArgumentParser:
    epilog = """OPERATION
  Refuses an existing output before network access, calibrates chat-template overhead from one small
  request, constructs one tokenizer-counted corpus, then sends exactly three measured requests in a
  synchronous loop: primary, alternate shared-prefix query, and byte-identical primary repeat.

EXAMPLES
  qwen-r9700-context --base-url http://127.0.0.1:8000 --model qwen3.8-27b \\
    --backend-id stock-vllm-bf16kv-64k --tokenizer-json /models/qwen/tokenizer.json \\
    --server-command-file artifacts/commands/stock-64k.txt --server-max-sequences 1 \\
    --server-max-model-len 65536 --prefix-cache-enabled --target-prompt-tokens 60000 \\
    --cache-namespace stock-bf16kv-64k-run1 --output results/context/stock-64k.json

FILES
  benchmarks/context/fixture-v1.json fixes filler generation, needles, request sampling, and
  scoring.
  The tokenizer JSON and redacted server-command evidence are hashed into each result.

PATHS
  The output is atomically created once, made mode 0444, and never replaced. The tokenizer and
  command paths are resolved before the calibration request.

SECURITY NOTES
  API credentials are read only from the named environment variable and recorded as redacted. Model
  output is untrusted JSON data and is parsed but never executed. Bind local benchmark servers to
  loopback and ensure command evidence contains no secret.

EXIT STATUS
  Returns 0 when all gates pass, 2 for argument, fixture, tokenizer, HTTP, or evidence errors, and 3
  after preserving a result whose token-budget, retrieval, or prefix-equivalence gate failed.

AUTHORS
  Qwen R9700 inference lab contributors.
"""
    parser = argparse.ArgumentParser(
        prog="qwen-r9700-context",
        description=(
            "NAME\n"
            "  qwen-r9700-context - gate long-context retrieval and prefix equivalence\n\n"
            "SYNOPSIS\n"
            "  qwen-r9700-context [OPTIONS]\n\n"
            "DESCRIPTION\n"
            "  Build deterministic token-budgeted prompts and test an already-running "
            "OpenAI-compatible server. The gate never starts, stops, or reconfigures a backend.\n\n"
            "OPTIONS\n"
            "  The options below identify the endpoint, tokenizer, context budget, and evidence."
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend-id", required=True)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--server-command-file", required=True, type=Path)
    parser.add_argument("--server-max-sequences", required=True, type=int, choices=[1])
    parser.add_argument("--server-max-model-len", required=True, type=int)
    parser.add_argument("--prefix-cache-enabled", required=True, action="store_true")
    parser.add_argument("--target-prompt-tokens", required=True, type=int)
    parser.add_argument("--prefix-alignment-tokens", type=int)
    parser.add_argument("--cache-namespace", required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("benchmarks/context/fixture-v1.json"),
    )
    parser.add_argument("--require-reported-cache-hit", action="store_true")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = run_context_gate(args)
    except ContextGateError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"created {args.output}: {len(document['measurements'])} serial context requests")
    if not document["gates"]["overall_passed"]:
        print("error: context gate failed; result was preserved", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
