"""Backend-neutral, strictly serial OpenAI chat-completions benchmark client."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from qwen_r9700_lab.functional_quality import SUPPORTED_QUALITY_KINDS, evaluate_response

PROMETHEUS_KEYWORDS = (
    "cache_usage",
    "decode",
    "e2e_request",
    "generation_token",
    "num_requests",
    "prefill",
    "prompt_token",
    "spec_decode",
    "time_per_output_token",
    "time_to_first_token",
)


class BenchmarkError(RuntimeError):
    """The benchmark contract or an HTTP request failed."""


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
        raise BenchmarkError(f"cannot load fixture {path}: {error}") from error
    if not isinstance(fixture, dict):
        raise BenchmarkError(f"fixture must be a JSON object: {path}")

    protocol = fixture.get("protocol")
    common = fixture.get("common_request")
    cases = fixture.get("cases")
    if fixture.get("schema_version") != 1:
        raise BenchmarkError("fixture schema_version must be 1")
    if not isinstance(protocol, dict) or protocol.get("client_max_in_flight") != 1:
        raise BenchmarkError("fixture must require client_max_in_flight=1")
    if protocol.get("required_server_max_sequences") != 1:
        raise BenchmarkError("fixture must require required_server_max_sequences=1")
    if not isinstance(common, dict) or common.get("stream") is not True:
        raise BenchmarkError("fixture common_request must enable streaming")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("fixture must contain at least one case")
    if common.get("n") != 1:
        raise BenchmarkError("fixture common_request must require n=1")
    is_greedy = common.get("temperature") == 0.0 and common.get("top_k") == 1
    is_quality_fixture = all(
        isinstance(case, dict) and case.get("evaluation") is not None for case in cases
    )
    if not is_greedy and not is_quality_fixture:
        raise BenchmarkError(
            "non-greedy sampling is allowed only for fully evaluated quality fixtures"
        )

    case_ids: list[str] = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise BenchmarkError("every fixture case must have a string id")
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages:
            raise BenchmarkError(f"fixture case {case['id']} must have messages")
        if not isinstance(case.get("workload_class"), str):
            raise BenchmarkError(f"fixture case {case['id']} must have a workload_class")
        overrides = case.get("request_overrides", {})
        if not isinstance(overrides, dict):
            raise BenchmarkError(f"fixture case {case['id']} request_overrides must be an object")
        forbidden = sorted({"messages", "model", "n", "stream"}.intersection(overrides))
        if forbidden:
            raise BenchmarkError(
                f"fixture case {case['id']} cannot override C1 fields: {', '.join(forbidden)}"
            )
        chat_kwargs = overrides.get("chat_template_kwargs")
        if chat_kwargs is not None and not isinstance(chat_kwargs, dict):
            raise BenchmarkError(
                f"fixture case {case['id']} chat_template_kwargs must be an object"
            )
        evaluation = case.get("evaluation")
        if evaluation is not None:
            if not isinstance(evaluation, dict):
                raise BenchmarkError(f"fixture case {case['id']} evaluation must be an object")
            if evaluation.get("kind") not in SUPPORTED_QUALITY_KINDS:
                raise BenchmarkError(
                    f"fixture case {case['id']} has unsupported quality evaluation kind"
                )
            if evaluation.get("kind") == "single-tool-call-v1":
                if not isinstance(evaluation.get("expected_tool_name"), str) or not evaluation.get(
                    "expected_tool_name"
                ):
                    raise BenchmarkError(
                        f"fixture case {case['id']} must define a non-empty expected_tool_name"
                    )
                if not isinstance(evaluation.get("expected_arguments"), dict):
                    raise BenchmarkError(
                        f"fixture case {case['id']} expected_arguments must be an object"
                    )
        case_ids.append(case["id"])
    if len(case_ids) != len(set(case_ids)):
        raise BenchmarkError("fixture case ids must be unique")
    return fixture


def _merge_request_objects(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_request_objects(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _case_payload(
    fixture: Mapping[str, Any], case: Mapping[str, Any], model: str
) -> dict[str, Any]:
    payload = _merge_request_objects(
        fixture["common_request"],
        case.get("request_overrides", {}),
    )
    payload.update({"model": model, "messages": copy.deepcopy(case["messages"])})
    return payload


def _authorized_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BenchmarkError(f"expected an absolute HTTP(S) URL, got {url!r}")
    return url


def _request_headers(
    api_key: str | None,
    request_id: str | None = None,
    accept: str = "text/event-stream",
) -> dict[str, str]:
    headers = {"Accept": accept, "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if request_id:
        headers["X-Request-ID"] = request_id
    return headers


def _http_get_text(url: str, timeout: float, api_key: str | None) -> str:
    request = urllib.request.Request(
        _authorized_url(url),
        headers=_request_headers(api_key, accept="text/plain"),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.HTTPError, urllib.error.URLError) as error:
        raise BenchmarkError(f"GET {url} failed: {error}") from error


def _relevant_prometheus_lines(text: str) -> list[str]:
    lines = [
        line
        for line in text.splitlines()
        if line and not line.startswith("#") and any(word in line for word in PROMETHEUS_KEYWORDS)
    ]
    return sorted(lines)


def _prometheus_counter_deltas(before: Iterable[str], after: Iterable[str]) -> dict[str, float]:
    def counters(lines: Iterable[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for line in lines:
            try:
                signature, raw_value = line.rsplit(maxsplit=1)
                name = signature.split("{", maxsplit=1)[0]
                value = float(raw_value)
            except (ValueError, IndexError):
                continue
            if name.endswith(("_total", "_sum", "_count")):
                result[signature] = value
        return result

    earlier = counters(before)
    later = counters(after)
    return {
        signature: value - earlier.get(signature, 0.0)
        for signature, value in later.items()
        if value - earlier.get(signature, 0.0) != 0.0
    }


def _counter_total(counter_deltas: Mapping[str, float], metric_name: str) -> float | None:
    values = [
        value
        for signature, value in counter_deltas.items()
        if signature == metric_name or signature.startswith(f"{metric_name}{{")
    ]
    return sum(values) if values else None


def _discover_amd_gpu(pci_bdf: str | None) -> Path | None:
    pci_root = Path("/sys/bus/pci/devices")
    if pci_bdf:
        candidate = pci_root / pci_bdf
        if not candidate.is_dir():
            raise BenchmarkError(f"AMD PCI device does not exist: {candidate}")
        return candidate

    candidates = []
    for candidate in pci_root.glob("*"):
        try:
            vendor = (candidate / "vendor").read_text(encoding="ascii").strip().lower()
            class_code = (candidate / "class").read_text(encoding="ascii").strip().lower()
        except OSError:
            continue
        if vendor == "0x1002" and class_code.startswith("0x03"):
            candidates.append(candidate)
    if not candidates:
        return None
    if len(candidates) != 1:
        bdfs = ", ".join(sorted(path.name for path in candidates))
        raise BenchmarkError(f"multiple AMD display devices found ({bdfs}); pass --amd-pci-bdf")
    return candidates[0]


def _first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _read_number(path: Path | None, divisor: float = 1.0) -> float | None:
    if path is None:
        return None
    try:
        return int(path.read_text(encoding="ascii").strip()) / divisor
    except (OSError, ValueError):
        return None


class GpuSampler:
    """Sample global AMD sysfs telemetry while exactly one HTTP request is active."""

    def __init__(self, device: Path | None, interval_seconds: float) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float | str | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._vram = device / "mem_info_vram_used" if device else None
        self._busy = device / "gpu_busy_percent" if device else None
        hwmons = sorted((device / "hwmon").glob("hwmon*")) if device else []
        self._power = _first_existing(
            path for hwmon in hwmons for path in (hwmon / "power1_average", hwmon / "power1_input")
        )
        self._temperature = _first_existing(hwmon / "temp1_input" for hwmon in hwmons)

    def _sample(self) -> None:
        self.samples.append(
            {
                "monotonic_ns": time.perf_counter_ns(),
                "observed_at": _utc_now(),
                "vram_used_bytes": _read_number(self._vram),
                "power_watts": _read_number(self._power, 1_000_000.0),
                "gpu_busy_percent": _read_number(self._busy),
                "temperature_celsius": _read_number(self._temperature, 1_000.0),
            }
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self.device is None:
            return
        self._thread = threading.Thread(target=self._run, name="amd-sysfs-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._sample()


def _parse_sse(response: BinaryIO) -> Iterable[dict[str, Any]]:
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="strict").strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as error:
            raise BenchmarkError(f"invalid SSE JSON: {error}") from error
        if not isinstance(chunk, dict):
            raise BenchmarkError("SSE data must decode to an object")
        if chunk.get("error"):
            raise BenchmarkError(f"server returned an SSE error: {chunk['error']}")
        yield chunk


def _normalize_reasoning_delta(delta: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Normalize canonical and legacy reasoning fields without duplicating a fragment."""

    observed = tuple(field for field in ("reasoning", "reasoning_content") if field in delta)
    fragments: list[tuple[str, str]] = []
    for field in observed:
        value = delta[field]
        if value is None:
            continue
        if not isinstance(value, str):
            raise BenchmarkError(f"chat delta {field} must be a string or null")
        if value:
            fragments.append((field, value))
    distinct = {value for _field, value in fragments}
    if len(distinct) > 1:
        fields = ", ".join(field for field, _value in fragments)
        raise BenchmarkError(f"chat delta has conflicting simultaneous reasoning fields: {fields}")
    return (fragments[0][1] if fragments else ""), observed


def _assemble_tool_calls(raw_deltas: Iterable[Any]) -> list[dict[str, Any]]:
    """Assemble OpenAI streaming tool-call fragments without dispatching them."""

    assembled: dict[int, dict[str, Any]] = {}
    for raw_delta in raw_deltas:
        if not isinstance(raw_delta, list):
            continue
        for position, fragment in enumerate(raw_delta):
            if not isinstance(fragment, Mapping):
                continue
            index = fragment.get("index", position)
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                continue
            call = assembled.setdefault(
                index,
                {
                    "index": index,
                    "id": "",
                    "type": "",
                    "function": {"name": "", "arguments": ""},
                },
            )
            for field in ("id", "type"):
                value = fragment.get(field)
                if isinstance(value, str) and value and not call[field]:
                    call[field] = value
            function = fragment.get("function")
            if not isinstance(function, Mapping):
                continue
            name = function.get("name")
            arguments = function.get("arguments")
            if isinstance(name, str):
                call["function"]["name"] += name
            if isinstance(arguments, str):
                call["function"]["arguments"] += arguments
    return [assembled[index] for index in sorted(assembled)]


def _run_request(
    *,
    url: str,
    payload: Mapping[str, Any],
    request_id: str,
    timeout: float,
    api_key: str | None,
    gpu_device: Path | None,
    telemetry_interval: float,
    metrics_url: str | None,
) -> dict[str, Any]:
    request_bytes = _canonical_bytes(payload)
    before_metrics = (
        _relevant_prometheus_lines(_http_get_text(metrics_url, timeout, api_key))
        if metrics_url
        else []
    )
    sampler = GpuSampler(gpu_device, telemetry_interval)
    request = urllib.request.Request(
        _authorized_url(url),
        data=request_bytes,
        headers=_request_headers(api_key, request_id),
        method="POST",
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_wire_fields: set[str] = set()
    reasoning_wire_nonempty_fragment_counts = {
        "reasoning": 0,
        "reasoning_content": 0,
    }
    raw_tool_call_deltas: list[Any] = []
    completion_token_ids: list[int] = []
    completion_token_ids_observed = False
    first_output_ns: int | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    server_extensions: dict[str, Any] = {}
    chunk_count = 0
    started_ns = time.perf_counter_ns()
    sampler.start()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for chunk in _parse_sse(response):
                chunk_count += 1
                choices = chunk.get("choices") or []
                for choice in choices:
                    token_ids = choice.get("token_ids")
                    if token_ids is not None:
                        if not isinstance(token_ids, list) or not all(
                            type(token_id) is int and token_id >= 0 for token_id in token_ids
                        ):
                            raise BenchmarkError(
                                "streaming choice token_ids must be non-negative integers"
                            )
                        completion_token_ids_observed = True
                        completion_token_ids.extend(token_ids)
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    reasoning, observed_reasoning_fields = _normalize_reasoning_delta(delta)
                    reasoning_wire_fields.update(observed_reasoning_fields)
                    for field in observed_reasoning_fields:
                        if isinstance(delta[field], str) and delta[field]:
                            reasoning_wire_nonempty_fragment_counts[field] += 1
                    produced_output = False
                    if isinstance(content, str) and content:
                        content_parts.append(content)
                        produced_output = True
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_parts.append(reasoning)
                        produced_output = True
                    tool_call_delta = delta.get("tool_calls")
                    if tool_call_delta:
                        raw_tool_call_deltas.append(copy.deepcopy(tool_call_delta))
                        produced_output = True
                    if produced_output and first_output_ns is None:
                        first_output_ns = time.perf_counter_ns()
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                server_extensions.update(
                    {
                        key: value
                        for key, value in chunk.items()
                        if key not in {"choices", "created", "id", "model", "object", "usage"}
                    }
                )
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"POST {url} failed with HTTP {error.code}: {body}") from error
    except (OSError, UnicodeDecodeError, urllib.error.URLError) as error:
        raise BenchmarkError(f"POST {url} failed: {error}") from error
    finally:
        sampler.stop()
    finished_ns = time.perf_counter_ns()
    after_metrics = (
        _relevant_prometheus_lines(_http_get_text(metrics_url, timeout, api_key))
        if metrics_url
        else []
    )
    counter_deltas = _prometheus_counter_deltas(before_metrics, after_metrics)

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    tool_calls = _assemble_tool_calls(raw_tool_call_deltas)
    output_value: list[Any] = [reasoning, content]
    if tool_calls:
        output_value.append(tool_calls)
    total_seconds = (finished_ns - started_ns) / 1_000_000_000
    ttft_seconds = (
        (first_output_ns - started_ns) / 1_000_000_000 if first_output_ns is not None else None
    )
    completion_tokens = usage.get("completion_tokens") if usage else None
    if payload.get("return_token_ids") is True:
        if not completion_token_ids_observed:
            raise BenchmarkError(
                "return_token_ids was requested but no streaming token IDs arrived"
            )
        if completion_tokens != len(completion_token_ids):
            raise BenchmarkError(
                "streaming token-ID count differs from response completion-token usage"
            )
    post_first_token_tps = None
    e2e_output_tps = None
    if isinstance(completion_tokens, int) and completion_tokens > 0:
        e2e_output_tps = completion_tokens / total_seconds
        post_first_seconds = total_seconds - (ttft_seconds or 0.0)
        if completion_tokens > 1 and post_first_seconds > 0:
            post_first_token_tps = (completion_tokens - 1) / post_first_seconds

    timing = server_extensions.get("timings")
    if not isinstance(timing, dict):
        timing = {}
    draft_tokens = timing.get("draft_n")
    accepted_tokens = timing.get("draft_n_accepted")
    if not isinstance(draft_tokens, (int, float)):
        draft_tokens = _counter_total(counter_deltas, "vllm:spec_decode_num_draft_tokens_total")
    if not isinstance(accepted_tokens, (int, float)):
        accepted_tokens = _counter_total(
            counter_deltas, "vllm:spec_decode_num_accepted_tokens_total"
        )
    acceptance = None
    if (
        isinstance(draft_tokens, (int, float))
        and draft_tokens > 0
        and isinstance(accepted_tokens, (int, float))
    ):
        acceptance = accepted_tokens / draft_tokens

    return {
        "request_id": request_id,
        "request_sha256": _sha256_bytes(request_bytes),
        "response": {
            "finish_reason": finish_reason,
            "usage": usage,
            "content": content,
            "reasoning_content": reasoning,
            "reasoning_wire_fields": sorted(reasoning_wire_fields),
            "reasoning_wire_nonempty_fragment_counts": {
                field: reasoning_wire_nonempty_fragment_counts[field]
                for field in sorted(reasoning_wire_fields)
            },
            "tool_calls": tool_calls,
            "raw_tool_call_deltas": raw_tool_call_deltas,
            "content_sha256": _sha256_bytes(content.encode()),
            "tool_calls_sha256": _sha256_bytes(_canonical_bytes(tool_calls)),
            "raw_tool_call_deltas_sha256": _sha256_bytes(_canonical_bytes(raw_tool_call_deltas)),
            "completion_token_ids": completion_token_ids
            if completion_token_ids_observed
            else None,
            "completion_token_ids_sha256": (
                _sha256_bytes(_canonical_bytes(completion_token_ids))
                if completion_token_ids_observed
                else None
            ),
            "output_sha256": _sha256_bytes(_canonical_bytes(output_value)),
            "sse_chunk_count": chunk_count,
            "server_extensions": server_extensions,
        },
        "client_timing": {
            "total_seconds": total_seconds,
            "ttft_seconds": ttft_seconds,
            "post_first_token_tps": post_first_token_tps,
            "e2e_output_tps": e2e_output_tps,
        },
        "speculation": {
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted_tokens,
            "acceptance_rate": acceptance,
        },
        "gpu": {
            "pci_bdf": gpu_device.name if gpu_device else None,
            "telemetry_interval_seconds": telemetry_interval,
            "samples": sampler.samples,
        },
        "prometheus": {
            "before": before_metrics,
            "after": after_metrics,
            "counter_deltas": counter_deltas,
        },
    }


def _percentile(values: Iterable[float | None], fraction: float) -> float | None:
    present = sorted(value for value in values if value is not None and math.isfinite(value))
    if not present:
        return None
    index = max(0, math.ceil(fraction * len(present)) - 1)
    return present[index]


def _median(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.median(present) if present else None


def _summarize_case(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    hashes = sorted({item["response"]["output_sha256"] for item in measurements})
    token_id_hashes = sorted(
        {
            digest
            for item in measurements
            if (digest := item["response"].get("completion_token_ids_sha256")) is not None
        }
    )
    timings = [item["response"]["server_extensions"].get("timings", {}) for item in measurements]
    draft_total = sum(item["speculation"]["draft_tokens"] or 0 for item in measurements)
    accepted_total = sum(item["speculation"]["accepted_tokens"] or 0 for item in measurements)
    cached_tokens = []
    prompt_tokens = []
    completion_tokens = []
    finish_reasons = []
    for item in measurements:
        usage = item["response"]["usage"] or {}
        if isinstance(usage.get("prompt_tokens"), int):
            prompt_tokens.append(usage["prompt_tokens"])
        if isinstance(usage.get("completion_tokens"), int):
            completion_tokens.append(usage["completion_tokens"])
        if item["response"].get("finish_reason") is not None:
            finish_reasons.append(item["response"]["finish_reason"])
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details.get("cached_tokens"), int):
            cached_tokens.append(details["cached_tokens"])
        else:
            timings_for_item = item["response"]["server_extensions"].get("timings") or {}
            if isinstance(timings_for_item.get("cache_n"), int):
                cached_tokens.append(timings_for_item["cache_n"])

    telemetry = [sample for item in measurements for sample in item["gpu"]["samples"]]
    power = [sample["power_watts"] for sample in telemetry if sample["power_watts"] is not None]
    vram = [
        sample["vram_used_bytes"] for sample in telemetry if sample["vram_used_bytes"] is not None
    ]
    ttft_values = [item["client_timing"]["ttft_seconds"] for item in measurements]
    post_first_tps_values = [item["client_timing"]["post_first_token_tps"] for item in measurements]
    e2e_tps_values = [item["client_timing"]["e2e_output_tps"] for item in measurements]
    summary = {
        "measured_repeats": len(measurements),
        "output_sha256_values": hashes,
        "output_is_byte_deterministic": len(hashes) == 1,
        "completion_token_ids_sha256_values": token_id_hashes,
        "completion_token_ids_are_deterministic": len(token_id_hashes) == 1
        if token_id_hashes
        else None,
        "finish_reasons": finish_reasons,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "client_ttft_seconds_min": _percentile(ttft_values, 0.0),
        "client_ttft_seconds_p50": _median(ttft_values),
        "client_ttft_seconds_p95": _percentile(ttft_values, 0.95),
        "client_ttft_seconds_max": _percentile(ttft_values, 1.0),
        "client_post_first_token_tps_min": _percentile(post_first_tps_values, 0.0),
        "client_post_first_token_tps_p50": _median(post_first_tps_values),
        "client_post_first_token_tps_max": _percentile(post_first_tps_values, 1.0),
        "client_e2e_output_tps_min": _percentile(e2e_tps_values, 0.0),
        "client_e2e_output_tps_p50": _median(e2e_tps_values),
        "client_e2e_output_tps_max": _percentile(e2e_tps_values, 1.0),
        "server_prefill_tps_p50": _median(
            timing.get("prompt_per_second") if isinstance(timing, dict) else None
            for timing in timings
        ),
        "server_decode_tps_p50": _median(
            timing.get("predicted_per_second") if isinstance(timing, dict) else None
            for timing in timings
        ),
        "speculation_draft_tokens": draft_total or None,
        "speculation_accepted_tokens": accepted_total or None,
        "speculation_weighted_acceptance_rate": (
            accepted_total / draft_total if draft_total else None
        ),
        "cached_prompt_tokens_observed": cached_tokens,
        "prefix_cache_unused": all(value == 0 for value in cached_tokens)
        if cached_tokens
        else None,
        "gpu_vram_used_bytes_max": max(vram) if vram else None,
        "gpu_power_watts_mean": statistics.fmean(power) if power else None,
        "gpu_power_watts_max": max(power) if power else None,
    }
    quality_results = [item["quality"] for item in measurements if item.get("quality") is not None]
    if quality_results:
        summary["quality"] = {
            "applicable": True,
            "passed": all(result["passed"] for result in quality_results),
            "results": quality_results,
        }
    return summary


def _write_create_once(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise BenchmarkError(f"refusing to replace existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".qwen-c1-",
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
        raise BenchmarkError(f"refusing to replace existing result: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    fixture_path = args.fixture.expanduser().resolve()
    fixture = _load_fixture(fixture_path)
    fixture_bytes = fixture_path.read_bytes()
    protocol = fixture["protocol"]
    warmups = args.warmups if args.warmups is not None else protocol["warmups_per_case"]
    repeats = args.repeats if args.repeats is not None else protocol["measured_repeats_per_case"]
    if warmups < 0 or repeats < 1:
        raise BenchmarkError("warmups must be non-negative and repeats must be positive")

    command_path = args.server_command_file.expanduser().resolve()
    try:
        server_command_bytes = command_path.read_bytes()
        server_command = server_command_bytes.decode("utf-8").strip()
    except OSError as error:
        raise BenchmarkError(
            f"cannot read server command evidence {command_path}: {error}"
        ) from error
    if not server_command:
        raise BenchmarkError("server command evidence is empty")

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    gpu_device = None if args.no_gpu_telemetry else _discover_amd_gpu(args.amd_pci_bdf)
    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    metrics_url = args.metrics_url
    if metrics_url:
        _authorized_url(metrics_url)

    warmup_records: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    started_at = _utc_now()
    for case in fixture["cases"]:
        payload = _case_payload(fixture, case, args.model)
        warmup_records.extend(
            {
                "case_id": case["id"],
                "warmup_index": warmup_index,
                **_run_request(
                    url=endpoint,
                    payload=payload,
                    request_id=f"c1-{case['id']}-warmup-{warmup_index}",
                    timeout=args.timeout,
                    api_key=api_key,
                    gpu_device=gpu_device,
                    telemetry_interval=args.telemetry_interval,
                    metrics_url=metrics_url,
                ),
            }
            for warmup_index in range(warmups)
        )
        measurements.extend(
            {
                "case_id": case["id"],
                "workload_class": case["workload_class"],
                "repeat_index": repeat_index,
                **_run_request(
                    url=endpoint,
                    payload=payload,
                    request_id=f"c1-{case['id']}-repeat-{repeat_index}",
                    timeout=args.timeout,
                    api_key=api_key,
                    gpu_device=gpu_device,
                    telemetry_interval=args.telemetry_interval,
                    metrics_url=metrics_url,
                ),
            }
            for repeat_index in range(repeats)
        )
        for measurement in measurements[-repeats:]:
            measurement["quality"] = evaluate_response(
                case.get("evaluation"),
                measurement["response"],
            )

    summaries = {
        case["id"]: _summarize_case(
            [item for item in measurements if item["case_id"] == case["id"]]
        )
        for case in fixture["cases"]
    }
    quality_results = [
        {
            "case_id": item["case_id"],
            "repeat_index": item["repeat_index"],
            "evaluation": item["quality"],
        }
        for item in measurements
        if item.get("quality") is not None
    ]
    quality_gate = {
        "applicable": bool(quality_results),
        "passed": all(item["evaluation"]["passed"] for item in quality_results)
        if quality_results
        else None,
        "results": quality_results,
    }
    document = {
        "schema_version": 1,
        "result_type": "true-c1-openai-chat-completions",
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
        "c1_contract": {
            "client_max_in_flight": 1,
            "single_threaded_request_loop": True,
            "server_max_sequences_attested": args.server_max_sequences,
            "exclusive_gpu_attested": args.exclusive_gpu,
            "prefix_cache_required_disabled": protocol["required_prefix_cache"] is False,
            "warmups_per_case": warmups,
            "measured_repeats_per_case": repeats,
        },
        "server_command_evidence": {
            "path": str(command_path),
            "file_sha256": _sha256_bytes(server_command_bytes),
            "command": server_command,
        },
        "gpu": {"pci_bdf": gpu_device.name if gpu_device else None},
        "warmups": warmup_records,
        "measurements": measurements,
        "summary_by_case": summaries,
        "quality_gate": quality_gate,
    }
    _write_create_once(args.output.expanduser().resolve(), document)
    return document


def build_parser() -> argparse.ArgumentParser:
    epilog = """OPERATION
  Sends one request at a time. For every case it discards the configured warmups, records measured
  streaming repeats, samples AMD sysfs telemetry, and snapshots relevant Prometheus metrics.

EXAMPLES
  qwen-r9700-c1 --base-url http://127.0.0.1:8000 --model qwen3.8-27b \\
    --backend-id radiance-target-only --server-command-file artifacts/radiance-command.txt \\
    --server-max-sequences 1 --exclusive-gpu --metrics-url http://127.0.0.1:8000/metrics \\
    --output results/c1/radiance-target-only.json

FILES
  The throughput fixture is benchmarks/c1/fixture-v1.json. The independent complete-output fixture
  family is benchmarks/c1/quality-fixture-v*.json; the bounded thinking fixture family is
  benchmarks/c1/reasoning-quality-fixture-v*.json; and static tool-call fixtures are
  benchmarks/c1/tool-quality-fixture-v*.json and tool-reasoning-quality-fixture-v*.json. Results
  embed the selected fixture, hashes, command evidence, outputs, timings, metrics snapshots,
  telemetry, and functional-quality checks.

PATHS
  Result destinations are created once and made read-only. Existing files and symlinks are refused.

SECURITY NOTES
  API credentials are read only from the named environment variable and are never written. Ensure
  the server command evidence file is already redacted and bind benchmark servers to localhost.

EXIT STATUS
  Returns 0 when the run and any quality gate pass, 2 for fixture, HTTP, evidence, or output errors,
  and 3 after preserving a result whose functional-quality gate failed.

AUTHORS
  Qwen R9700 inference lab contributors.
"""
    parser = argparse.ArgumentParser(
        prog="qwen-r9700-c1",
        description=(
            "NAME\n"
            "  qwen-r9700-c1 - run a backend-neutral, true-concurrency-one benchmark\n\n"
            "SYNOPSIS\n"
            "  qwen-r9700-c1 [OPTIONS]\n\n"
            "DESCRIPTION\n"
            "  Replays content-locked chat requests with fixture-defined sampling against an "
            "already-running OpenAI-compatible server. It never starts, stops, or configures a "
            "model server.\n\n"
            "OPTIONS\n"
            "  The options below select the endpoint, exact server evidence, fixture, and output."
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend-id", required=True)
    parser.add_argument("--server-command-file", required=True, type=Path)
    parser.add_argument("--server-max-sequences", required=True, type=int, choices=[1])
    parser.add_argument("--exclusive-gpu", required=True, action="store_true")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("benchmarks/c1/fixture-v1.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--amd-pci-bdf")
    parser.add_argument("--no-gpu-telemetry", action="store_true")
    parser.add_argument("--telemetry-interval", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repeats", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        document = run_benchmark(args)
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"created {args.output}: {len(document['measurements'])} measured C1 requests")
    if document["quality_gate"]["applicable"] and not document["quality_gate"]["passed"]:
        print("error: functional-quality gate failed; result was preserved", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
