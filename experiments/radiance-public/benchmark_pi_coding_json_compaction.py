"""Measure a long 60K-context coding turn, prose, JSON, thinking and checkpoint generation.

The benchmark sends one append-only conversation to the pinned Radiance endpoint.
It starts from the existing private 60K token fixture, asks for a deliberately
substantial coding response, asks for a short prose explanation of code
measurement, then asks for a JSON document, a longer engineering-prose turn,
and finally submits the same conversation to the checkpoint prompt.  EOS remains
enabled throughout; the maximum token values are safety ceilings, not fixed-length
generation.

Only hashes, token counts, timings and validation results are written.  Prompt,
response and token arrays remain in memory and are never saved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"
COMPACTION_MARKER = "COMPACTION_SUMMARY_COMPLETE"
CODING_MIN_TOKENS = 5_000
PROSE_CODE_MIN_TOKENS = 500
JSON_MIN_TOKENS = 1_000
THINKING_MIN_TOKENS = 1_000

CODING_PROMPT = """Work as a senior engineer on one finite coding task. Design and implement a small, coherent Python package for a durable, content-addressed chat snapshot cache. It must support immutable prefix blocks, a mutable tail, compressed disk snapshots, atomic publication, verification before replacement, crash recovery, RAM eviction, and a cache-inspection command.

Think through the design before the visible answer, then write one short architecture explanation followed by exactly three complete fenced file edits in this order: cache.py, cli.py, and tests/test_cache.py. Treat every fenced file edit as code. Do not repeat a file, add another module, use placeholder ellipses, or call tools. Make the three files substantial and internally coherent, with a combined response of about 5,000–6,000 generated tokens. After the third file and one brief closing sentence, finish the assistant turn at the normal turn boundary; do not continue expanding the package."""

JSON_PROMPT = """Now produce a machine-readable project manifest for the snapshot-cache package just designed. Return one valid JSON document and no markdown or prose outside the JSON. Include a schema version, package name, invariants, commands, storage tiers, failure modes, and exactly eight detailed test-case objects. Each test case should have an id, category, setup, expected result, and evidence field, with enough detail for the complete document to exceed 1,000 generated tokens. Keep every count field correct and stop immediately after the closing brace."""

PROSE_CODE_PROMPT = """Explain, in five to eight concise paragraphs, how to measure code-generation performance in an inference backend. Cover output tokens per second, a three-second sliding peak, mean round latency, speculative acceptance rate, time to first data, and how code versus prose output should be classified. Do not write code, JSON, markdown fences, or call tools. Aim for roughly 500 to 800 generated tokens and finish naturally."""

THINKING_PROMPT = """Turn thinking on for this turn. Analyse a realistic engineering decision involving the snapshot-cache package: reason through a slow restore, compare at least three possible causes and remedies, identify what measurements would distinguish them, discuss correctness and performance trade-offs, and finish with a concrete recommendation and next-step plan. Use detailed ordinary prose, no code, no JSON, and no tool calls. Produce at least 1,000 generated tokens, then finish naturally at the turn boundary."""

COMPACTION_PROMPT = f"""This is a checkpoint compaction request, not a request to continue the coding, prose or JSON tasks. Summarize the complete preceding conversation into a durable checkpoint for resuming the work. Preserve the architecture, invariants, interfaces, implementation decisions, test plan, generated artifact structure, unresolved risks, and next actions. Use these exact level-three Markdown headings, each once on its own line:
### Goal
### Current Authoritative State
### Constraints & Invariants
### Progress
### Measurements & Evidence
### Key Decisions
### Rejected / Failed Approaches
### Unresolved Questions & Hypotheses
### Next Steps
### Critical Context
Do not call tools or add a reasoning preamble. Aim for 2,500–3,500 generated tokens. Finish with this exact standalone line: {COMPACTION_MARKER}"""


def _load_previous() -> Any:
    path = Path(__file__).with_name("benchmark_pi_task_workloads.py")
    spec = importlib.util.spec_from_file_location("pi_task_workloads", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load benchmark helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_previous()
_read_sse = _HELPERS._read_sse
_metrics = _HELPERS._metrics
_delta = _HELPERS._delta
_summarize_metrics = _HELPERS._summarize_metrics
_task_token_ids = _HELPERS._task_token_ids
_turn_suffix = _HELPERS._turn_suffix
_wait_for_metrics = _HELPERS._wait_for_metrics


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _post_json(opener, url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise TypeError("backend returned a non-object JSON response")
    return result


def _render_user_turn(
    opener,
    base_url: str,
    prompt: str,
    *,
    thinking: bool,
    timeout: float,
) -> list[int]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {
            "enable_thinking": thinking,
            "preserve_thinking": thinking,
            **({"reasoning_effort": "xhigh"} if thinking else {}),
        },
    }
    result = _post_json(opener, f"{base_url}/tokenize", body, timeout)
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or not tokens or not all(
        type(token) is int and token >= 0 for token in tokens
    ):
        raise RuntimeError("tokenize returned an invalid rendered turn")
    return tokens


def _phase_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(_task_token_ids(tokenizer, text)) if text else 0


def _coding_phases(tokenizer: Tokenizer, content: str, reasoning: str) -> dict[str, Any]:
    """Classify visible code fences as file edits and other text as prose."""
    spans: list[tuple[int, int]] = []
    code_tokens = 0
    code_blocks = 0
    for match in re.finditer(r"```[^\n]*\n(.*?)```", content, flags=re.DOTALL):
        spans.append(match.span(1))
        code_tokens += _phase_tokens(tokenizer, match.group(1))
        code_blocks += 1
    prose_parts: list[str] = []
    cursor = 0
    for start, end in spans:
        prose_parts.append(content[cursor:start])
        cursor = end
    prose_parts.append(content[cursor:])
    prose = "".join(prose_parts)
    return {
        "reasoning_tokens": _phase_tokens(tokenizer, reasoning),
        "prose_tokens": _phase_tokens(tokenizer, prose),
        "file_edit_code_tokens": code_tokens,
        "reasoning_blocks": 1 if reasoning.strip() else 0,
        "prose_blocks": sum(bool(part.strip()) for part in prose_parts),
        "file_edit_code_blocks": code_blocks,
        "classification": "reasoning channel; non-fenced prose; all fenced file edits treated as code",
    }


def _json_phases(tokenizer: Tokenizer, content: str, reasoning: str) -> dict[str, Any]:
    parsed: Any = None
    valid = False
    failure = None
    try:
        parsed = json.loads(content.strip())
        valid = isinstance(parsed, (dict, list))
        if not valid:
            failure = "JSON root must be an object or array"
    except (json.JSONDecodeError, TypeError) as error:
        failure = str(error)
    return {
        "reasoning_tokens": _phase_tokens(tokenizer, reasoning),
        "prose_tokens": 0,
        "file_edit_json_tokens": _phase_tokens(tokenizer, content),
        "reasoning_blocks": 1 if reasoning.strip() else 0,
        "prose_blocks": 0,
        "file_edit_json_blocks": 1 if content.strip() else 0,
        "json_valid": valid,
        "json_root_type": type(parsed).__name__ if valid else None,
        "failure": failure,
        "classification": "all visible file-edit content treated as JSON",
    }


def _thinking_phases(tokenizer: Tokenizer, content: str, reasoning: str) -> dict[str, Any]:
    return {
        "reasoning_tokens": _phase_tokens(tokenizer, reasoning),
        "prose_tokens": _phase_tokens(tokenizer, content),
        "file_edit_code_tokens": 0,
        "reasoning_blocks": 1 if reasoning.strip() else 0,
        "prose_blocks": 1 if content.strip() else 0,
        "file_edit_code_blocks": 0,
        "classification": "reasoning channel and ordinary non-reasoning prose; no file edits requested",
    }


def _checkpoint_validation(text: str, finish_reason: str | None) -> dict[str, Any]:
    lines = text.strip().splitlines()
    matches = [line.strip() for line in lines if line.strip() == COMPACTION_MARKER]
    marker_ok = bool(lines) and lines[-1].strip() == COMPACTION_MARKER and len(matches) == 1
    headings = [
        "Goal",
        "Current Authoritative State",
        "Constraints & Invariants",
        "Progress",
        "Measurements & Evidence",
        "Key Decisions",
        "Rejected / Failed Approaches",
        "Unresolved Questions & Hypotheses",
        "Next Steps",
        "Critical Context",
    ]
    heading_count = {heading: text.count(f"### {heading}") for heading in headings}
    return {
        "finish_reason": finish_reason,
        "marker_valid": marker_ok,
        "headings_valid": all(count == 1 for count in heading_count.values()),
        "heading_counts": heading_count,
        "passed": finish_reason == "stop" and marker_ok and all(
            count == 1 for count in heading_count.values()
        ),
    }


def _request(
    *,
    opener,
    args,
    identity: dict[str, str],
    tokenizer: Tokenizer,
    prompt_tokens: list[int],
    suffix_tokens: list[int],
    stage: str,
    max_tokens: int,
    thinking: bool,
) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    before = _metrics(opener, args.base_url)
    if before["vllm:num_requests_running"] or before["vllm:num_requests_waiting"]:
        raise RuntimeError("backend busy; benchmark did not submit concurrent work")
    body = {
        "model": MODEL,
        "prompt": prompt_tokens,
        "max_tokens": max_tokens,
        "ignore_eos": False,
        "seed": args.seed,
        "temperature": args.temperature if stage != "compaction" else 0.3,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "return_token_ids": True,
        "add_special_tokens": False,
        "cache_salt": f"qwen-chat-cache-v1:{identity['id']}:{identity['generation']}",
        "kv_transfer_params": {
            "qwen_chat": identity,
            "qwen_snapshot_abi": args.abi,
            **({"qwen_snapshot_force_flush": True} if stage == "compaction" else {}),
        },
    }
    if stage in {"coding", "prose_code", "json", "thinking"}:
        body["stop"] = ["<|im_end|>"]
    request = urllib.request.Request(
        f"{args.base_url}/v1/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with opener.open(request, timeout=args.request_timeout) as response:
        completion = _read_sse(
            response, prompt_tokens, tokenizer=tokenizer, started=started
        )
    elapsed = time.monotonic() - started
    _, metrics_delta = _wait_for_metrics(
        opener,
        args.base_url,
        before,
        len(completion["token_ids"]),
        args.metrics_timeout,
    )
    generated = len(completion["token_ids"])
    if metrics_delta["vllm:generation_tokens_total"] != generated:
        raise RuntimeError("backend generation counter disagrees with streamed token IDs")
    if metrics_delta["vllm:request_success_total"] != 1:
        raise RuntimeError("backend did not record exactly one successful request")
    if metrics_delta["vllm:num_preemptions_total"]:
        raise RuntimeError("benchmark request was preempted")
    if completion["finish_reason"] != "stop":
        raise RuntimeError(
            f"{stage} did not finish naturally (finish_reason={completion['finish_reason']!r})"
        )
    usage = completion["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if usage.get("prompt_tokens") != len(prompt_tokens) or cached is None:
        raise RuntimeError("backend prompt/cache usage does not match submitted prompt")
    summary = _summarize_metrics(metrics_delta)
    result = {
        "stage": stage,
        "thinking_enabled": thinking,
        "sampling": {
            name: body[name] for name in ("temperature", "top_p", "top_k", "seed")
        },
        "prompt_tokens": len(prompt_tokens),
        "suffix_tokens": len(suffix_tokens),
        "generated_tokens": generated,
        "elapsed_seconds": elapsed,
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": len(prompt_tokens) - cached,
        "finish_reason": completion["finish_reason"],
        "first_token_seconds": completion["first_token_seconds"],
        "stream_seconds": completion["stream_seconds"],
        "peak_3s_tokens_per_second": completion["peak_3s_tokens_per_second"],
        "peak_rate_window_seconds": completion["peak_rate_window_seconds"],
        "rate_sample_count": completion["rate_sample_count"],
        "post_first_tokens_per_second": completion["post_first_tokens_per_second"]
        if "post_first_tokens_per_second" in completion
        else (
            (generated - completion["first_stream_token_count"])
            / completion["stream_seconds"]
            if completion["stream_seconds"] > 0
            else None
        ),
        **summary,
        "output_sha256": completion["output_sha256"],
        "content_sha256": completion["content_sha256"],
        "reasoning_sha256": completion["reasoning_sha256"],
        "reasoning_channel_observed": bool(completion["reasoning"]),
        "phase_token_counts": completion["phase_token_counts"],
        "phase_token_counts_cover_output": completion["phase_token_counts_cover_output"],
    }
    if stage == "coding":
        result["phase_blocks"] = _coding_phases(
            tokenizer, completion["content"], completion["reasoning"]
        )
        result["minimum_output_tokens"] = args.coding_min_tokens
        result["minimum_output_met"] = generated >= args.coding_min_tokens
    elif stage == "json":
        result["phase_blocks"] = _json_phases(
            tokenizer, completion["content"], completion["reasoning"]
        )
        result["minimum_output_tokens"] = args.json_min_tokens
        result["minimum_output_met"] = generated >= args.json_min_tokens
    elif stage in {"prose_code", "thinking"}:
        result["phase_blocks"] = _thinking_phases(
            tokenizer, completion["content"], completion["reasoning"]
        )
        minimum_tokens = (
            args.prose_code_min_tokens
            if stage == "prose_code"
            else args.thinking_min_tokens
        )
        result["minimum_output_tokens"] = minimum_tokens
        result["minimum_output_met"] = generated >= minimum_tokens
    else:
        result["phase_blocks"] = {"checkpoint_tokens": generated}
        result["checkpoint_validation"] = _checkpoint_validation(
            completion["content"] + completion["reasoning"], completion["finish_reason"]
        )
    return result, completion["token_ids"], completion


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    sequence = fixture.get("prefix")
    if not isinstance(sequence, list) or len(sequence) != 60000 or any(
        type(token) is not int or token < 0 for token in sequence
    ):
        raise ValueError("expected the existing private 60K Pi token fixture")
    identity = {
        "id": hashlib.sha256(args.identity.encode()).hexdigest(),
        "generation": hashlib.sha256(f"{args.identity}:initial".encode()).hexdigest(),
        "title": "60K coding prose JSON thinking compaction benchmark",
        "cwd": "/qualification/coding-json-compaction",
        "session_file": "",
    }
    report: dict[str, Any] = {
        "schema": "urn:coherence:pi-coding-json-compaction:v1",
        "status": "running",
        "started_at": time.time(),
        "model": MODEL,
        "fixture_sha256": _sha256_bytes(fixture_bytes),
        "benchmark_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "tokenizer_sha256": _sha256_bytes(args.tokenizer_json.read_bytes()),
        "runtime": json.loads(args.runtime_manifest.read_text())
        if args.runtime_manifest
        else None,
        "prefix_tokens": len(sequence),
        "privacy": "No prompt, response or token arrays are saved; only hashes and metrics are retained.",
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "coding_min_tokens": args.coding_min_tokens,
        "prose_code_min_tokens": args.prose_code_min_tokens,
        "json_min_tokens": args.json_min_tokens,
        "thinking_min_tokens": args.thinking_min_tokens,
        "coding_max_tokens": args.coding_max_tokens,
        "prose_code_max_tokens": args.prose_code_max_tokens,
        "json_max_tokens": args.json_max_tokens,
        "thinking_max_tokens": args.thinking_max_tokens,
        "compaction_max_tokens": args.compaction_max_tokens,
        "stages": [],
    }
    _write(args.output, report)

    rendered_coding = _render_user_turn(
        opener, args.base_url, CODING_PROMPT, thinking=False, timeout=args.request_timeout
    )
    coding_suffix = _turn_suffix(tokenizer, sequence, rendered_coding, first=True)
    coding_result, coding_ids, _ = _request(
        opener=opener,
        args=args,
        identity=identity,
        tokenizer=tokenizer,
        prompt_tokens=sequence + coding_suffix,
        suffix_tokens=coding_suffix,
        stage="coding",
        max_tokens=args.coding_max_tokens,
        thinking=False,
    )
    report["stages"].append(coding_result)
    _write(args.output, report)
    print(json.dumps(coding_result), flush=True)
    sequence.extend(coding_suffix)
    sequence.extend(coding_ids)

    rendered_prose_code = _render_user_turn(
        opener,
        args.base_url,
        PROSE_CODE_PROMPT,
        thinking=True,
        timeout=args.request_timeout,
    )
    prose_code_suffix = _turn_suffix(tokenizer, sequence, rendered_prose_code, first=False)
    prose_code_result, prose_code_ids, _ = _request(
        opener=opener,
        args=args,
        identity=identity,
        tokenizer=tokenizer,
        prompt_tokens=sequence + prose_code_suffix,
        suffix_tokens=prose_code_suffix,
        stage="prose_code",
        max_tokens=args.prose_code_max_tokens,
        thinking=True,
    )
    report["stages"].append(prose_code_result)
    _write(args.output, report)
    print(json.dumps(prose_code_result), flush=True)
    sequence.extend(prose_code_suffix)
    sequence.extend(prose_code_ids)

    rendered_json = _render_user_turn(
        opener, args.base_url, JSON_PROMPT, thinking=False, timeout=args.request_timeout
    )
    json_suffix = _turn_suffix(tokenizer, sequence, rendered_json, first=False)
    json_result, json_ids, _ = _request(
        opener=opener,
        args=args,
        identity=identity,
        tokenizer=tokenizer,
        prompt_tokens=sequence + json_suffix,
        suffix_tokens=json_suffix,
        stage="json",
        max_tokens=args.json_max_tokens,
        thinking=False,
    )
    report["stages"].append(json_result)
    _write(args.output, report)
    print(json.dumps(json_result), flush=True)
    sequence.extend(json_suffix)
    sequence.extend(json_ids)

    rendered_thinking = _render_user_turn(
        opener,
        args.base_url,
        THINKING_PROMPT,
        thinking=True,
        timeout=args.request_timeout,
    )
    thinking_suffix = _turn_suffix(tokenizer, sequence, rendered_thinking, first=False)
    thinking_result, thinking_ids, _ = _request(
        opener=opener,
        args=args,
        identity=identity,
        tokenizer=tokenizer,
        prompt_tokens=sequence + thinking_suffix,
        suffix_tokens=thinking_suffix,
        stage="thinking",
        max_tokens=args.thinking_max_tokens,
        thinking=True,
    )
    report["stages"].append(thinking_result)
    _write(args.output, report)
    print(json.dumps(thinking_result), flush=True)
    sequence.extend(thinking_suffix)
    sequence.extend(thinking_ids)

    rendered_compaction = _render_user_turn(
        opener,
        args.base_url,
        COMPACTION_PROMPT,
        thinking=False,
        timeout=args.request_timeout,
    )
    compaction_suffix = _turn_suffix(tokenizer, sequence, rendered_compaction, first=False)
    compaction_result, _, _ = _request(
        opener=opener,
        args=args,
        identity=identity,
        tokenizer=tokenizer,
        prompt_tokens=sequence + compaction_suffix,
        suffix_tokens=compaction_suffix,
        stage="compaction",
        max_tokens=args.compaction_max_tokens,
        thinking=False,
    )
    report["stages"].append(compaction_result)
    report["status"] = (
        "complete"
        if compaction_result["checkpoint_validation"]["passed"]
        else "complete_with_validation_failure"
    )
    report["validation_failures"] = (
        []
        if compaction_result["checkpoint_validation"]["passed"]
        else ["checkpoint headings or completion contract did not validate"]
    )
    report["completed_at"] = time.time()
    _write(args.output, report)
    print(json.dumps(compaction_result), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abi", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--coding-min-tokens", type=int, default=CODING_MIN_TOKENS)
    parser.add_argument(
        "--prose-code-min-tokens", type=int, default=PROSE_CODE_MIN_TOKENS
    )
    parser.add_argument("--json-min-tokens", type=int, default=JSON_MIN_TOKENS)
    parser.add_argument("--thinking-min-tokens", type=int, default=THINKING_MIN_TOKENS)
    parser.add_argument("--coding-max-tokens", type=int, default=10_000)
    parser.add_argument("--prose-code-max-tokens", type=int, default=8_192)
    parser.add_argument("--json-max-tokens", type=int, default=4_096)
    parser.add_argument("--thinking-max-tokens", type=int, default=8_192)
    parser.add_argument("--compaction-max-tokens", type=int, default=8_192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-timeout", type=float, default=12.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--identity", default="coherence-coding-json-compaction-20260920")
    args = parser.parse_args()
    if (
        args.coding_min_tokens < 1
        or args.prose_code_min_tokens < 1
        or args.json_min_tokens < 1
        or args.thinking_min_tokens < 1
        or args.coding_max_tokens < args.coding_min_tokens
        or args.prose_code_max_tokens < args.prose_code_min_tokens
        or args.json_max_tokens < args.json_min_tokens
        or args.thinking_max_tokens < args.thinking_min_tokens
        or args.top_k < 1
        or not 0 < args.top_p <= 1
        or args.temperature < 0
    ):
        parser.error("maximum tokens must cover positive minimums; sampling values are invalid")
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            report = json.loads(args.output.read_text())
            report["status"] = "failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            _write(args.output, report)
        raise


if __name__ == "__main__":
    main()
