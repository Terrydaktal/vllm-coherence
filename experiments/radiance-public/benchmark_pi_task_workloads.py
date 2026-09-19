"""Measure eight natural-stop workloads back-to-back on one cached context.

The benchmark deliberately uses one request per workload and chains the exact
output token IDs into the next request.  Each arm therefore loads its initial
context once: the 60K arm starts from the supplied private Pi prefix and the
0K arm starts empty.  EOS is always enabled; ``--tokens`` is only a safety
budget, so a workload that reaches the budget is rejected instead of being
treated as a natural completion.  Only hashes, phase counts and measurements
are written to the result; prompt, output and token arrays never leave process
memory.
"""

from __future__ import annotations

import argparse
import ast
import codecs
import hashlib
import json
import re
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from tokenizers import Tokenizer

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"


class TaskSpec(NamedTuple):
    name: str
    prompt: str
    output_kind: str
    enable_thinking: bool


# These are natural-stop probes, rather than requests to emit a fixed number of
# tokens.  Each prompt asks for a substantial completion (at least roughly one
# thousand generated tokens); ``--tokens`` remains a safety ceiling and EOS is
# still respected.  The protocol mode is part of the workload contract: code,
# diffs and JSON must not silently include a reasoning channel, while the
# reasoning task deliberately records both the reasoning and final-content
# phases.
TASKS = (
    TaskSpec(
        "Chat",
        "Write a detailed practical guide to planning a busy week. Cover prioritisation, time blocks, interruptions, energy management, communication, and a worked example schedule. Use clear headings and concrete examples. Produce about 1,100–1,300 generated tokens and do not stop after a short summary; finish only when the guide is complete.",
        "free_text",
        True,
    ),
    TaskSpec(
        "Code",
        "Output only valid Python code: no markdown, comments outside code, or explanation. Build a self-contained, well-tested module for processing a stream of records: include typed data structures, validation, grouping, stable sorting, error handling, a small command-line entry point, and several helper functions. Include docstrings and executable test data in the module. Produce at least about 1,000 generated tokens of code before ending.",
        "python_code",
        False,
    ),
    TaskSpec(
        "File edit",
        "Output only a valid unified diff, with no prose or markdown fences. Apply a substantial change to a small Python command-line project: add a --verbose flag, configuration loading, input validation, structured logging, and tests. Include realistic file headers and several complete hunks touching the implementation, helper module, and tests. Make the patch roughly 1,000 or more generated tokens so it represents a real edit, then end with the diff only.",
        "unified_diff",
        False,
    ),
    TaskSpec(
        "JSON",
        'Return only one valid JSON object with exactly the keys "name", "count", and "items"; do not use markdown fences or explanation. Set name to a descriptive string, count to the number of entries, and items to an array of at least 140 distinct, descriptive project-planning strings. Keep every item a string and make the object large enough to contain at least about 1,000 generated tokens before ending.',
        "json_object",
        False,
    ),
    TaskSpec(
        "Math",
        "Solve a challenging but self-contained planning and optimisation problem involving several equations, constraints, and a final numerical decision. Give a rigorous, step-by-step derivation, check the result independently, discuss edge cases, and explain why the answer follows. Produce about 1,100–1,300 generated tokens rather than a short answer; finish only after the verification.",
        "free_text",
        True,
    ),
    TaskSpec(
        "Prose",
        "Write a complete short story about a person solving an unexpected problem during a storm in a city. Develop the setting, characters, cause and effect, turning point, and ending with vivid but controlled prose. Produce about 1,100–1,300 generated tokens, with no heading, and do not stop after a single paragraph.",
        "free_text",
        True,
    ),
    TaskSpec(
        "Reasoning",
        "Analyse a complex engineering decision in depth: decide whether to optimise a slow service, replace a dependency, or redesign its data flow. State assumptions, compare alternatives, reason through failure modes and measurements, and reach a justified recommendation. Produce about 1,100–1,300 generated tokens of explicit reasoning and conclusion; do not stop after a brief answer.",
        "reasoning_prompt",
        True,
    ),
    TaskSpec(
        "Summarisation",
        "Write a detailed, self-contained summary of how a small engineering team investigates and fixes a slow reporting system. Cover the original symptoms, measurement plan, database findings, code changes, validation, rollout, risks, and lessons learned. Organise the summary into coherent paragraphs and produce about 1,100–1,300 generated tokens rather than two sentences.",
        "free_text",
        True,
    ),
)

MIN_NATURAL_OUTPUT_TOKENS = 1000
ROLLING_RATE_WINDOW_SECONDS = 3.0

METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:request_success_total",
)
COUNTER_SUFFIXES = ("total", "sum", "count")
DEFAULT_OUTPUT = Path("benchmarks/results/pi-task-workloads.json")


def _metric_values(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in METRICS:
        matches = re.findall(
            r"^" + re.escape(name) + r"(?:\{[^\n]*\})? ([\d.eE+\-]+)$",
            text,
            re.MULTILINE,
        )
        if not matches:
            raise RuntimeError(f"required backend metric is absent: {name}")
        result[name] = sum(map(float, matches))
    return result


def _task_token_ids(tokenizer: Tokenizer, prompt: str) -> list[int]:
    return list(tokenizer.encode(prompt, add_special_tokens=False).ids)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _peak_rolling_tokens_per_second(
    samples: list[tuple[float, int]],
    *,
    window_seconds: float = ROLLING_RATE_WINDOW_SECONDS,
) -> float | None:
    """Return the peak completed rolling-window output rate.

    ``samples`` contains monotonic ``(timestamp, cumulative_output_tokens)``
    observations from the streamed token-ID frames.  The boundary count is
    linearly interpolated when the three-second window starts between frames,
    matching Pi's three-second rolling-rate window.  Incomplete startup windows
    are excluded, so the result is never an instantaneous or half-second burst.
    """

    if len(samples) < 2 or window_seconds <= 0:
        return None
    first_at = samples[0][0]
    boundary = 0
    peak: float | None = None
    for index, (end_at, end_tokens) in enumerate(samples):
        start_at = max(first_at, end_at - window_seconds)
        elapsed = end_at - start_at
        if elapsed < window_seconds:
            continue
        while (
            boundary + 1 < len(samples)
            and samples[boundary + 1][0] <= start_at
        ):
            boundary += 1
        before_at, before_tokens = samples[boundary]
        after = samples[boundary + 1] if boundary + 1 <= index else None
        if after is not None and after[0] > before_at and start_at > before_at:
            after_at, after_tokens = after
            start_tokens = before_tokens + (after_tokens - before_tokens) * (
                (start_at - before_at) / (after_at - before_at)
            )
        else:
            start_tokens = before_tokens
        rate = max(0.0, end_tokens - start_tokens) / elapsed
        peak = rate if peak is None else max(peak, rate)
    return peak


def _python_source(value: str) -> str:
    """Return a single optional fenced Python block without accepting prose."""

    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3 or not lines[0].startswith("```") or lines[-1] != "```":
            raise ValueError("malformed code fence")
        language = lines[0][3:].strip().lower()
        if language not in {"", "python", "py"}:
            raise ValueError("code fence is not Python")
        stripped = "\n".join(lines[1:-1]).strip()
    if "```" in stripped:
        raise ValueError("code contains an unexpected markdown fence")
    return stripped


def _validate_task_output(
    task: TaskSpec, *, content: str, reasoning: str, unclassified: str
) -> dict[str, Any]:
    """Validate only the user-visible content; never persist the content itself."""

    phase_labels = []
    if reasoning:
        phase_labels.append("reasoning")
    if content:
        phase_labels.append("content")
    if unclassified:
        phase_labels.append("unclassified")

    validation: dict[str, Any] = {
        "kind": task.output_kind,
        "phase_labels": phase_labels,
        "phase_purity": "not_applicable",
        "content_valid": None,
        "failure": None,
    }
    if unclassified:
        validation["failure"] = (
            "one or more output tokens had no reasoning/content phase label"
        )
    if task.output_kind in {"python_code", "unified_diff", "json_object"}:
        validation["phase_purity"] = (
            "pass" if not reasoning and not unclassified else "fail"
        )

    try:
        if task.output_kind == "python_code":
            source = _python_source(content)
            tree = ast.parse(source, mode="exec")
            if not any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                for node in ast.walk(tree)
            ):
                raise ValueError("Python output contains no function definition")
            validation["content_valid"] = True
        elif task.output_kind == "unified_diff":
            lines = content.strip().splitlines()
            if (
                "```" in content
                or not any(line.startswith("--- ") for line in lines)
                or not any(line.startswith("+++ ") for line in lines)
                or not any(line.startswith("@@") for line in lines)
            ):
                raise ValueError("output is not a unified diff")
            validation["content_valid"] = True
        elif task.output_kind == "json_object":
            value = json.loads(content.strip())
            if not isinstance(value, dict) or set(value) != {"name", "count", "items"}:
                raise ValueError("JSON object does not match the required schema")
            if not isinstance(value["name"], str) or isinstance(value["count"], bool):
                raise ValueError("JSON name/count types are invalid")
            if not isinstance(value["count"], int) or not isinstance(value["items"], list):
                raise ValueError("JSON count/items types are invalid")
            if (
                len(value["items"]) < 3
                or value["count"] != len(value["items"])
                or not all(isinstance(item, str) for item in value["items"])
            ):
                raise ValueError(
                    "JSON items must contain at least three strings and count must match"
                )
            validation["content_valid"] = True
    except (SyntaxError, TypeError, ValueError) as error:
        validation["content_valid"] = False
        validation["failure"] = str(error)

    if task.output_kind in {"python_code", "unified_diff", "json_object"}:
        if validation["phase_purity"] == "fail":
            validation["failure"] = validation["failure"] or (
                "reasoning or unclassified output appeared in a content-only task"
            )
        validation["passed"] = bool(
            validation["phase_purity"] == "pass"
            and validation["content_valid"] is True
        )
    elif task.output_kind == "reasoning_prompt":
        validation["phase_purity"] = (
            "observed" if reasoning else "not_observed"
        )
        validation["reasoning_channel_observed"] = bool(reasoning)
        validation["passed"] = bool((content or reasoning) and not unclassified)
    else:
        validation["phase_purity"] = "observed" if phase_labels else "fail"
        validation["passed"] = bool(phase_labels and not unclassified)
    return validation


def _read_sse(
    response,
    prompt_tokens: list[int],
    tokenizer: Tokenizer | None = None,
    started: float | None = None,
) -> dict[str, Any]:
    """Read one natural SSE completion and retain phase text only in memory."""

    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    pending = ""
    data_lines: list[str] = []
    done = False
    ids: list[int] = []
    text_bytes = hashlib.sha256()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    unclassified_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    started = time.monotonic() if started is None else started
    first_data: float | None = None
    last_data: float | None = None
    first_count = 0
    rate_samples: list[tuple[float, int]] = []

    def consume(frame: str) -> None:
        nonlocal done, finish_reason, usage, first_data, last_data, first_count
        data = "\n".join(data_lines)
        data_lines.clear()
        if not data:
            return
        if done:
            raise RuntimeError("SSE emitted data after its DONE marker")
        if data == "[DONE]":
            done = True
            return
        event = json.loads(data)
        if event.get("error"):
            raise RuntimeError("backend returned an inference error (body withheld)")
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise RuntimeError("benchmark expects one completion choice")
            chunk_ids = choice.get("token_ids") or []
            if not isinstance(chunk_ids, list) or any(
                type(token) is not int or token < 0 for token in chunk_ids
            ):
                raise RuntimeError("backend returned invalid output token IDs")
            ids.extend(chunk_ids)
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            text = choice.get("text")
            if not isinstance(text, str):
                text = choice.get("content")
            if not isinstance(text, str):
                text = delta.get("content")
            if not isinstance(text, str):
                text = message.get("content")
            text = text if isinstance(text, str) else ""
            reasoning = choice.get("reasoning")
            if not isinstance(reasoning, str):
                reasoning = choice.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = delta.get("reasoning")
            if not isinstance(reasoning, str):
                reasoning = delta.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = delta.get("reasoning_text")
            if not isinstance(reasoning, str):
                reasoning = message.get("reasoning")
            if not isinstance(reasoning, str):
                reasoning = message.get("reasoning_content")
            reasoning = reasoning if isinstance(reasoning, str) else ""
            if text:
                content_parts.append(text)
            if reasoning:
                reasoning_parts.append(reasoning)
            if not text and not reasoning and chunk_ids:
                unclassified_parts.append("<token-id-only-frame>")
            text_bytes.update((text + reasoning).encode())
            # Some vLLM builds prepend prompt IDs to the first token-ID frame.
            # Do not count that frame as generated output.
            has_output = (
                len(ids) > len(prompt_tokens)
                if ids[: len(prompt_tokens)] == prompt_tokens
                else bool(chunk_ids) or bool(text) or bool(reasoning)
            )
            if has_output:
                now = time.monotonic()
                prompt_count = (
                    len(prompt_tokens)
                    if ids[: len(prompt_tokens)] == prompt_tokens
                    else 0
                )
                output_count = max(0, len(ids) - prompt_count)
                if first_data is None:
                    first_count = output_count
                first_data = first_data or now
                last_data = now
                if rate_samples and rate_samples[-1][0] == now:
                    rate_samples[-1] = (now, output_count)
                else:
                    rate_samples.append((now, output_count))
            if choice.get("finish_reason") is not None:
                if (
                    finish_reason is not None
                    and finish_reason != choice["finish_reason"]
                ):
                    raise RuntimeError(
                        "backend changed finish_reason during the stream"
                    )
                finish_reason = choice["finish_reason"]

    while True:
        raw = response.readline()
        if not raw:
            pending += decoder.decode(b"", final=True)
            break
        pending += decoder.decode(raw, final=False)
        pending = pending.replace("\r\n", "\n")
        while "\n\n" in pending:
            frame, pending = pending.split("\n\n", 1)
            for line in frame.split("\n"):
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif line.startswith((":", "event:", "id:", "retry:")) or not line:
                    continue
                else:
                    raise RuntimeError("unsupported SSE field from backend")
            consume(frame)
    if pending.strip():
        for line in pending.split("\n"):
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        consume(pending)
    if data_lines:
        consume("")
    if not done or finish_reason is None:
        raise RuntimeError("completion stream ended without finish_reason and DONE")

    output_ids = ids
    if output_ids[: len(prompt_tokens)] == prompt_tokens:
        output_ids = output_ids[len(prompt_tokens) :]
    expected = usage.get("completion_tokens")
    if expected is not None and len(output_ids) != expected:
        raise RuntimeError(
            f"return_token_ids count {len(output_ids)} differs from completion usage {expected}"
        )
    if not output_ids:
        raise RuntimeError("completion returned no output token IDs")
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    unclassified = "".join(unclassified_parts)
    phase_token_counts: dict[str, int] = {}
    if tokenizer is not None:
        phase_token_counts = {
            "content": len(_task_token_ids(tokenizer, content)) if content else 0,
            "reasoning": len(_task_token_ids(tokenizer, reasoning)) if reasoning else 0,
            "unclassified": len(unclassified_parts),
        }
    phase_token_counts_cover_output = bool(
        phase_token_counts
        and not unclassified
        and sum(phase_token_counts.values()) == len(output_ids)
    )
    return {
        "token_ids": output_ids,
        "usage": usage,
        "finish_reason": finish_reason,
        "first_token_seconds": (first_data or started) - started,
        "first_stream_token_count": first_count,
        "stream_seconds": (last_data or started) - (first_data or started),
        "peak_3s_tokens_per_second": _peak_rolling_tokens_per_second(rate_samples),
        "peak_rate_window_seconds": ROLLING_RATE_WINDOW_SECONDS,
        "rate_sample_count": len(rate_samples),
        "output_sha256": text_bytes.hexdigest(),
        "content": content,
        "reasoning": reasoning,
        "unclassified": unclassified,
        "content_sha256": _sha256_text(content),
        "reasoning_sha256": _sha256_text(reasoning),
        "phase_token_counts": phase_token_counts,
        "phase_token_counts_method": (
            "tokenize_streamed_channels; coverage flag is conservative and not a phase-boundary proof"
            if tokenizer is not None
            else "not_collected"
        ),
        "phase_token_counts_cover_output": phase_token_counts_cover_output,
    }


def _delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        name: after[name] - before[name]
        for name in METRICS
        if name.endswith(COUNTER_SUFFIXES)
    }


def _summarize_metrics(delta: dict[str, float]) -> dict[str, Any]:
    rounds = delta["vllm:spec_decode_num_drafts_total"]
    draft_tokens = delta["vllm:spec_decode_num_draft_tokens_total"]
    interval_count = delta["vllm:inter_token_latency_seconds_count"]
    generated = delta["vllm:generation_tokens_total"]
    return {
        "generation_rounds": int(rounds),
        "generated_tokens": int(generated),
        "mean_generation_round_ms": (
            1000 * delta["vllm:inter_token_latency_seconds_sum"] / interval_count
            if interval_count > 0
            else None
        ),
        "acceptance_rate": (
            delta["vllm:spec_decode_num_accepted_tokens_total"] / draft_tokens
            if draft_tokens > 0
            else None
        ),
        "draft_tokens": int(draft_tokens),
        "accepted_tokens": int(delta["vllm:spec_decode_num_accepted_tokens_total"]),
        "inter_token_samples": int(interval_count),
    }


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _metrics(opener, base_url: str) -> dict[str, float]:
    with opener.open(base_url + "/metrics", timeout=10) as response:
        return _metric_values(response.read().decode())


def _framed_tasks(
    opener, base_url: str, tokenizer: Tokenizer
) -> list[tuple[TaskSpec, list[int], list[int]]]:
    """Use the live template for synthetic tasks; never decode the private prefix."""
    user_start = _task_token_ids(tokenizer, "<|im_start|>user\n")
    tasks = []
    for task in TASKS:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": task.prompt}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {
                "enable_thinking": task.enable_thinking,
                "preserve_thinking": task.enable_thinking,
                **({"reasoning_effort": "xhigh"} if task.enable_thinking else {}),
            },
        }
        request = urllib.request.Request(
            base_url + "/tokenize",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=30) as response:
            rendered = json.load(response)["tokens"]
        positions = [
            i
            for i in range(len(rendered))
            if rendered[i : i + len(user_start)] == user_start
        ]
        if len(positions) != 1:
            raise RuntimeError(
                "live chat template has an unexpected user-turn boundary"
            )
        tasks.append((task, rendered, rendered[positions[0] :]))
    return tasks


def _turn_suffix(
    tokenizer: Tokenizer, sequence: list[int], rendered: list[int], *, first: bool
) -> list[int]:
    if not sequence:
        return list(rendered)
    end = tokenizer.token_to_id("<|im_end|>")
    if end is None:
        raise RuntimeError("tokenizer has no Qwen turn boundary")
    # The frozen prefix is cut at an exact token budget; close its unfinished turn.
    boundary = "\n</think>\n<|im_end|>\n" if first else "\n"
    if not first and sequence[-1] != end:
        boundary = "<|im_end|>\n"
    return _task_token_ids(tokenizer, boundary) + rendered


def _wait_for_metrics(
    opener,
    base_url: str,
    before: dict[str, float],
    expected_output: int,
    timeout_seconds: float,
) -> tuple[dict[str, float], dict[str, float]]:
    deadline = time.monotonic() + timeout_seconds
    after = before
    while True:
        after = _metrics(opener, base_url)
        delta = _delta(before, after)
        if (
            delta["vllm:generation_tokens_total"] >= expected_output
            and delta["vllm:request_success_total"] >= 1
        ):
            return after, delta
        if time.monotonic() >= deadline:
            return after, delta
        time.sleep(0.25)


def _run_request(
    *,
    opener,
    args,
    identity: dict[str, str],
    tokenizer: Tokenizer,
    prompt_tokens: list[int],
    task: TaskSpec,
    task_prompt_tokens: list[int],
    context_tokens_before: int,
    continuation: bool,
) -> tuple[dict[str, Any], list[int]]:
    before = _metrics(opener, args.base_url)
    if before["vllm:num_requests_running"] or before["vllm:num_requests_waiting"]:
        raise RuntimeError("backend busy; benchmark did not submit concurrent work")
    body = {
        "model": MODEL,
        "prompt": prompt_tokens,
        "max_tokens": args.tokens,
        "ignore_eos": False,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "return_token_ids": True,
        "cache_salt": f"qwen-chat-cache-v1:{identity['id']}:{identity['generation']}",
        "kv_transfer_params": {"qwen_chat": identity, "qwen_snapshot_abi": args.abi},
    }
    request = urllib.request.Request(
        args.base_url + "/v1/completions",
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
    if (
        metrics_delta["vllm:generation_tokens_total"] != len(completion["token_ids"])
        or metrics_delta["vllm:request_success_total"] != 1
    ):
        raise RuntimeError(
            "backend counters do not match one isolated completion; concurrent work or unsettled metrics"
        )
    if metrics_delta["vllm:num_preemptions_total"]:
        raise RuntimeError("benchmark request was preempted")
    if completion["finish_reason"] != "stop":
        raise RuntimeError(
            "workload did not finish naturally; increase the safety token budget"
        )
    usage = completion["usage"]
    if usage.get("prompt_tokens") != len(prompt_tokens):
        raise RuntimeError("backend prompt token count differs from submitted context")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        raise RuntimeError("backend did not report prompt cache coverage")
    # Hybrid recurrent/KV checkpoints can leave a suffix to recompute. The live
    # profile retains an 8192-token tail; reject a fresh-context reload rather
    # than incorrectly requiring the 2048-token prefill chunk to be this bound.
    if continuation and cached < context_tokens_before - 8192:
        raise RuntimeError(
            f"chained workload reused only {cached} of {context_tokens_before} prior tokens"
        )
    content_validation = _validate_task_output(
        task,
        content=completion["content"],
        reasoning=completion["reasoning"],
        unclassified=completion["unclassified"],
    )
    generated_tokens = len(completion["token_ids"])
    content_validation["minimum_output_tokens"] = args.min_output_tokens
    content_validation["generated_tokens"] = generated_tokens
    content_validation["length_valid"] = (
        generated_tokens >= args.min_output_tokens
    )
    if not content_validation["length_valid"]:
        content_validation["passed"] = False
        content_validation["failure"] = (
            content_validation["failure"]
            or f"natural completion produced {generated_tokens} tokens; "
            f"minimum requested is {args.min_output_tokens}"
        )
    result = {
        "workload": task.name,
        "output_kind": task.output_kind,
        "thinking_enabled": task.enable_thinking,
        "context_tokens_before": context_tokens_before,
        "prompt_tokens": len(prompt_tokens),
        "task_prompt_tokens": len(task_prompt_tokens),
        "context_tokens_after": context_tokens_before
        + len(task_prompt_tokens)
        + generated_tokens,
        "elapsed_seconds": elapsed,
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": len(prompt_tokens) - cached,
        "metrics_wait_seconds": time.monotonic() - started - elapsed,
        "output_sha256": completion["output_sha256"],
        "content_sha256": completion["content_sha256"],
        "reasoning_sha256": completion["reasoning_sha256"],
        "reasoning_channel_observed": bool(completion["reasoning"]),
        "finish_reason": completion["finish_reason"],
        "first_token_seconds": completion["first_token_seconds"],
        "stream_seconds": completion["stream_seconds"],
        "peak_3s_tokens_per_second": completion["peak_3s_tokens_per_second"],
        "peak_rate_window_seconds": completion["peak_rate_window_seconds"],
        "rate_sample_count": completion["rate_sample_count"],
        "post_first_tokens_per_second": (
            (len(completion["token_ids"]) - completion["first_stream_token_count"])
            / completion["stream_seconds"]
            if completion["stream_seconds"] > 0
            else None
        ),
        **_summarize_metrics(metrics_delta),
        "phase_token_counts": completion["phase_token_counts"],
        "phase_token_counts_method": completion["phase_token_counts_method"],
        "phase_token_counts_cover_output": completion["phase_token_counts_cover_output"],
        "content_validation": content_validation,
        "metrics_delta": metrics_delta,
        "metrics_settled": True,
    }
    return result, completion["token_ids"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    framed_tasks = _framed_tasks(opener, args.base_url, tokenizer)
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    prefix = fixture.get("prefix")
    if (
        not isinstance(prefix, list)
        or len(prefix) != 60000
        or any(type(t) is not int or t < 0 for t in prefix)
    ):
        raise ValueError("expected the existing private 60K Pi token fixture")
    prefix_digest = hashlib.sha256(
        json.dumps(prefix, separators=(",", ":")).encode()
    ).hexdigest()
    report: dict[str, Any] = {
        "schema": "urn:coherence:pi-task-workloads:v2",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            args.tokenizer_json.read_bytes()
        ).hexdigest(),
        "runtime": json.loads(args.runtime_manifest.read_text())
        if args.runtime_manifest
        else None,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "prefix_token_ids_sha256": prefix_digest,
        "privacy": "No prompt, output or token arrays are saved; only hashes and metrics are retained.",
        "task_order": [task.name for task in TASKS],
        "task_contracts": {
            task.name: {
                "output_kind": task.output_kind,
                "thinking_enabled": task.enable_thinking,
                "natural_stop_required": True,
            }
            for task in TASKS
        },
        "prompt_count_per_arm": len(TASKS),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "output_token_cap": args.tokens,
        "minimum_natural_output_tokens": args.min_output_tokens,
        "ignore_eos": False,
        "completion_policy": (
            "natural stop only; fixed-length EOS suppression is not supported; "
            "each workload must naturally produce at least the configured minimum"
        ),
        "arms_requested": args.arms,
        "reasoning_effort": "xhigh",
        "framing": "live Qwen chat template; append-only token IDs; thinking retained",
        "validation_failures": [],
        "arms": {},
    }
    _write(args.output, report)
    arms = (("60K", prefix), ("0K", []))
    if args.arms != "both":
        arms = tuple(arm for arm in arms if arm[0] == args.arms)
    for arm_name, initial_prefix in arms:
        identity_seed = f"{args.identity}:{arm_name}:{report['fixture_sha256']}"
        identity = {
            "id": hashlib.sha256(identity_seed.encode()).hexdigest(),
            "generation": hashlib.sha256(b"initial").hexdigest(),
            "title": f"task workload {arm_name}",
            "cwd": "/qualification/task-workloads",
            "session_file": "",
        }
        sequence = list(initial_prefix)
        arm: dict[str, Any] = {
            "initial_context_tokens": len(sequence),
            "context_id": identity["id"],
            "requests": len(TASKS),
            "results": [],
        }
        report["arms"][arm_name] = arm
        _write(args.output, report)
        for index, (task, first_turn, later_turn) in enumerate(framed_tasks):
            task_prompt_tokens = _turn_suffix(
                tokenizer,
                sequence,
                first_turn if index == 0 else later_turn,
                first=index == 0,
            )
            prompt_tokens = sequence + task_prompt_tokens
            result, output_ids = _run_request(
                opener=opener,
                args=args,
                identity=identity,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                task=task,
                task_prompt_tokens=task_prompt_tokens,
                context_tokens_before=len(sequence),
                continuation=index > 0,
            )
            arm["results"].append(result)
            if not result["content_validation"]["passed"]:
                report["validation_failures"].append(
                    {
                        "arm": arm_name,
                        "workload": task.name,
                        "output_kind": task.output_kind,
                        "failure": result["content_validation"]["failure"]
                        or "validation failed",
                    }
                )
            sequence.extend(task_prompt_tokens)
            sequence.extend(output_ids)
            _write(args.output, report)
            print(json.dumps({"arm": arm_name, **result}), flush=True)
    report["status"] = "complete"
    report["validation_summary"] = {
        "total": len(TASKS) * len(arms),
        "passed": len(TASKS) * len(arms) - len(report["validation_failures"]),
        "failed": len(report["validation_failures"]),
    }
    report["completed_at"] = datetime.now(UTC).isoformat()
    _write(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        required=True,
        help="private JSON fixture containing a 60K-token 'prefix'",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        required=True,
        help="exact tokenizer.json used by the backend",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--runtime-manifest", type=Path, help="content-free deployment identity receipt"
    )
    parser.add_argument(
        "--abi", required=True, help="snapshot ABI advertised by the backend"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--tokens",
        type=int,
        default=8192,
        help="natural-output safety budget per workload; EOS is always respected",
    )
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=MIN_NATURAL_OUTPUT_TOKENS,
        help="minimum natural output required for a valid workload result",
    )
    parser.add_argument(
        "--arms",
        choices=("both", "60K", "0K"),
        default="both",
        help="which context arm to run (default: both)",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-timeout", type=float, default=12.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--identity", default="coherence-task-workloads-v1")
    args = parser.parse_args()
    if (
        args.tokens < 2
        or args.min_output_tokens < 1
        or args.tokens < args.min_output_tokens
        or args.top_k < 1
        or not 0 < args.top_p <= 1
        or args.temperature < 0
    ):
        parser.error(
            "tokens must be >= minimum output >=1, top-k >=1, "
            "0 < top-p <=1 and temperature >=0"
        )
    try:
        run(args)
    except Exception as error:
        # Keep the result visibly incomplete without retaining request text.
        if args.output.exists():
            report = json.loads(args.output.read_text())
            report["status"] = "failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            _write(args.output, report)
        raise


if __name__ == "__main__":
    main()
