"""Measure eight bounded workloads back-to-back on one cached context.

The benchmark deliberately uses one request per workload and chains the exact
output token IDs into the next request.  Each arm therefore loads its initial
context once: the 60K arm starts from the supplied private Pi prefix and the
0K arm starts empty.  Only hashes and measurements are written to the result;
prompt, output and token arrays never leave process memory.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import re
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"
TASKS = (
    (
        "Chat",
        "Answer this everyday question in three concise sentences: what is a practical way to plan a busy week?",
    ),
    (
        "Code",
        "Write a short Python function that returns the first repeated item in a list, with one type hint and no explanation.",
    ),
    (
        "File edit",
        "Describe a minimal unified diff that adds a --verbose flag to a small command-line program; show only the diff.",
    ),
    (
        "JSON",
        'Return one valid JSON object with keys "name", "count", and "items"; use a string, an integer, and an array of three strings.',
    ),
    (
        "Math",
        "Solve 37*48-19 exactly and show two short arithmetic steps.",
    ),
    (
        "Prose",
        "Write a compact descriptive paragraph about rain on a city street, with no heading.",
    ),
    (
        "Reasoning",
        "Explain briefly why checking an assumption before optimizing a program prevents wasted work; give one concrete example.",
    ),
    (
        "Summarisation",
        "Summarise this in two sentences: a small team reduced a slow report by measuring its database queries before changing code.",
    ),
)

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


def _read_sse(
    response, prompt_tokens: list[int], started: float | None = None
) -> dict[str, Any]:
    """Read one strict SSE completion and return output IDs plus timing metadata."""

    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    pending = ""
    data_lines: list[str] = []
    done = False
    ids: list[int] = []
    text_bytes = hashlib.sha256()
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    started = time.monotonic() if started is None else started
    first_data: float | None = None
    last_data: float | None = None
    first_count = 0

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
            text = choice.get("text") or ""
            reasoning = "".join(
                value
                for value in (
                    choice.get("reasoning"),
                    choice.get("reasoning_content"),
                    (choice.get("delta") or {}).get("reasoning"),
                    (choice.get("delta") or {}).get("reasoning_content"),
                )
                if isinstance(value, str)
            )
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
                if first_data is None:
                    first_count = len(ids) - (
                        len(prompt_tokens)
                        if ids[: len(prompt_tokens)] == prompt_tokens
                        else 0
                    )
                first_data = first_data or now
                last_data = now
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
    return {
        "token_ids": output_ids,
        "usage": usage,
        "finish_reason": finish_reason,
        "first_token_seconds": (first_data or started) - started,
        "first_stream_token_count": first_count,
        "stream_seconds": (last_data or started) - (first_data or started),
        "output_sha256": text_bytes.hexdigest(),
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
) -> list[tuple[str, list[int], list[int]]]:
    """Use the live template for synthetic tasks; never decode the private prefix."""
    user_start = _task_token_ids(tokenizer, "<|im_start|>user\n")
    tasks = []
    for name, prompt in TASKS:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "xhigh",
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
        tasks.append((name, rendered, rendered[positions[0] :]))
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
    prompt_tokens: list[int],
    task_name: str,
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
        "ignore_eos": args.ignore_eos,
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
        completion = _read_sse(response, prompt_tokens, started)
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
    if args.ignore_eos:
        if completion["finish_reason"] not in {"length", "stop"}:
            raise RuntimeError("fixed-length workload ended with an unsupported finish reason")
        if len(completion["token_ids"]) < args.tokens:
            raise RuntimeError(
                f"fixed-length workload produced only {len(completion['token_ids'])} "
                f"of {args.tokens} requested tokens"
            )
    elif completion["finish_reason"] != "stop":
        raise RuntimeError(
            "workload did not finish naturally; increase the benchmark output budget"
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
    result = {
        "workload": task_name,
        "context_tokens_before": context_tokens_before,
        "prompt_tokens": len(prompt_tokens),
        "task_prompt_tokens": len(task_prompt_tokens),
        "context_tokens_after": context_tokens_before
        + len(task_prompt_tokens)
        + len(completion["token_ids"]),
        "elapsed_seconds": elapsed,
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": len(prompt_tokens) - cached,
        "metrics_wait_seconds": time.monotonic() - started - elapsed,
        "output_sha256": completion["output_sha256"],
        "finish_reason": completion["finish_reason"],
        "first_token_seconds": completion["first_token_seconds"],
        "stream_seconds": completion["stream_seconds"],
        "post_first_tokens_per_second": (
            (len(completion["token_ids"]) - completion["first_stream_token_count"])
            / completion["stream_seconds"]
            if completion["stream_seconds"] > 0
            else None
        ),
        **_summarize_metrics(metrics_delta),
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
        "schema": "urn:coherence:pi-task-workloads:v1",
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
        "task_order": [name for name, _ in TASKS],
        "prompt_count_per_arm": len(TASKS),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "output_token_cap": args.tokens,
        "ignore_eos": args.ignore_eos,
        "arms_requested": args.arms,
        "reasoning_effort": "xhigh",
        "framing": "live Qwen chat template; append-only token IDs; thinking retained",
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
        for index, (task_name, first_turn, later_turn) in enumerate(framed_tasks):
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
                prompt_tokens=prompt_tokens,
                task_name=task_name,
                task_prompt_tokens=task_prompt_tokens,
                context_tokens_before=len(sequence),
                continuation=index > 0,
            )
            arm["results"].append(result)
            sequence.extend(task_prompt_tokens)
            sequence.extend(output_ids)
            _write(args.output, report)
            print(json.dumps({"arm": arm_name, **result}), flush=True)
    report["status"] = "complete"
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
        help="safety budget per workload; a truncated answer fails the run",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="run a fixed-length workload and require the requested token budget",
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
        or args.top_k < 1
        or not 0 < args.top_p <= 1
        or args.temperature < 0
    ):
        parser.error("tokens must be >=2, top-k >=1, 0 < top-p <=1 and temperature >=0")
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
