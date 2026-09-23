"""Measure one natural-stop coding task at 0K, 60K and 200K context.

Each context arm submits the same coding prompt as an independent request with
EOS enabled.  The 60K and 200K arms take private token-prefix fixtures supplied
by the operator; the fixture contents are never decoded, written to the
report, or printed.  The report contains only fixture/prompt/output hashes,
token counts, timings, cache coverage, backend counter deltas and the complete
content-free round event record for each request.  A missing or incomplete
round feed is a validation failure; it is never silently represented by a
partial histogram.

The task is intentionally the same substantial coding task used by
``benchmark_pi_coding_json_compaction.py``.  A maximum is a safety ceiling,
not a request to emit a fixed number of tokens.  A short natural completion is
retained as a visible validation failure rather than silently padded or
classified as a passing throughput result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"
MODEL_CONTEXT_TOKENS = 253_792
CODING_MIN_TOKENS = 5_000
CODING_MAX_TOKENS = 10_000
DEFAULT_OUTPUT = Path("benchmarks/results/pi-coding-contexts.json")
DEFAULT_ROUND_LOG = Path("/dev/shm/qwen-radiance-fair-public-rounds.jsonl")
ROUND_LOG_SCHEMA = "urn:qwen-r9700:decode-rounds:v1"

# The order is part of the published table and report contract.
CONTEXTS = ("0K", "60K", "200K")
CONTEXT_TOKEN_COUNTS = {"0K": 0, "60K": 60_000, "200K": 200_000}

# Fixed, exhaustive bins make separate context arms directly comparable.  The
# first and last bins are open-ended; every finite measured round belongs to
# exactly one bin.
ROUND_HISTOGRAM_BINS = (
    ("<35", None, 35.0),
    ("35–37", 35.0, 37.0),
    ("37–39", 37.0, 39.0),
    ("39–40", 39.0, 40.0),
    ("40–40.5", 40.0, 40.5),
    ("40.5–41", 40.5, 41.0),
    ("41–41.5", 41.0, 41.5),
    ("41.5–42", 41.5, 42.0),
    ("42–43", 42.0, 43.0),
    ("43–45", 43.0, 45.0),
    ("45–46", 45.0, 46.0),
    ("46–47", 46.0, 47.0),
    ("47–48", 47.0, 48.0),
    ("48–48.5", 48.0, 48.5),
    ("48.5–49", 48.5, 49.0),
    ("49–51", 49.0, 51.0),
    ("51–52.5", 51.0, 52.5),
    ("52.5–53", 52.5, 53.0),
    ("53–53.5", 53.0, 53.5),
    ("53.5–54", 53.5, 54.0),
    ("54–55", 54.0, 55.0),
    ("55–56", 55.0, 56.0),
    ("56–60", 56.0, 60.0),
    ("60–62.5", 60.0, 62.5),
    ("62.5–63", 62.5, 63.0),
    ("63–63.5", 63.0, 63.5),
    ("63.5–64", 63.5, 64.0),
    ("64–65", 64.0, 65.0),
    ("65–70", 65.0, 70.0),
    ("70–100", 70.0, 100.0),
    ("100–250", 100.0, 250.0),
    ("250–500", 250.0, 500.0),
    ("≥500", 500.0, None),
)


def _load_coding_benchmark():
    path = Path(__file__).with_name("benchmark_pi_coding_json_compaction.py")
    spec = importlib.util.spec_from_file_location("pi_coding_json_compaction", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load coding benchmark helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CODING = _load_coding_benchmark()
CODING_PROMPT = _CODING.CODING_PROMPT
_request = _CODING._request
_render_user_turn = _CODING._render_user_turn
_turn_suffix = _CODING._turn_suffix
_write = _CODING._write


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prefix_digest(prefix: list[int]) -> str:
    return _sha256_bytes(json.dumps(prefix, separators=(",", ":")).encode())


def _round_event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    """Return a stable key for one content-free scheduler event."""

    return (
        event.get("pid"),
        event.get("chat_id"),
        event.get("generation"),
        event.get("request_id"),
        event.get("round"),
        event.get("observed_at_ms"),
    )


def _read_round_log(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Read the current and rotated public round feeds without payload data.

    The live feed is intentionally bounded and may rotate while a request is
    running.  Reading both names and deduplicating by the event identity keeps
    the benchmark from silently losing rows at a rotation boundary.
    """

    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    found = False
    for candidate in (Path(f"{path}.1"), path):
        try:
            payload = candidate.read_bytes()
        except FileNotFoundError:
            continue
        except OSError as error:
            return [], f"unable to read {candidate}: {type(error).__name__}"
        found = True
        for line in payload.splitlines():
            try:
                event = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(event, dict) or event.get("schema") != ROUND_LOG_SCHEMA:
                continue
            key = _round_event_key(event)
            if key in seen:
                continue
            seen.add(key)
            records.append(event)
    records.sort(
        key=lambda event: (event.get("observed_at_ms", 0), event.get("round", 0))
    )
    return records, None if found else "round log is not present"


def _public_round_record(event: dict[str, Any]) -> dict[str, Any]:
    """Keep only numeric, content-free fields in the benchmark artifact."""

    round_number = event.get("round")
    observed_at_ms = event.get("observed_at_ms")
    round_ms = event.get("round_ms")
    draft_tokens = event.get("draft_tokens")
    accepted_tokens = event.get("accepted_tokens")
    acceptance_rate = event.get("acceptance_rate")
    return {
        "round": round_number
        if type(round_number) is int and round_number > 0
        else None,
        "observed_at_ms": (
            observed_at_ms
            if type(observed_at_ms) is int and observed_at_ms > 0
            else None
        ),
        "round_ms": (
            float(round_ms)
            if type(round_ms) in (int, float)
            and math.isfinite(round_ms)
            and round_ms >= 0
            else None
        ),
        "draft_tokens": (
            draft_tokens if type(draft_tokens) is int and draft_tokens >= 0 else None
        ),
        "accepted_tokens": (
            accepted_tokens
            if type(accepted_tokens) is int and accepted_tokens >= 0
            else None
        ),
        "acceptance_rate": (
            float(acceptance_rate)
            if type(acceptance_rate) in (int, float)
            and math.isfinite(acceptance_rate)
            and 0 <= acceptance_rate <= 1
            else None
        ),
    }


def _round_histogram(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive an exhaustive latency histogram from the retained round rows."""

    values = [
        record["round_ms"]
        for record in records
        if type(record.get("round_ms")) in (int, float)
        and math.isfinite(record["round_ms"])
        and record["round_ms"] >= 0
    ]
    counts = []
    for _label, lower, upper in ROUND_HISTOGRAM_BINS:
        count = sum(
            (lower is None or value >= lower) and (upper is None or value < upper)
            for value in values
        )
        counts.append(count)
    if sum(counts) != len(values):
        raise ValueError("round histogram bins do not cover every measured round")
    denominator = len(values)
    return {
        "schema": "urn:coherence:round-latency-histogram:v1",
        "measured_round_count": denominator,
        "unmeasured_round_count": len(records) - denominator,
        "bins": [
            {
                "label": label,
                "count": count,
                "percentage": (100 * count / denominator if denominator else None),
            }
            for (label, _lower, _upper), count in zip(ROUND_HISTOGRAM_BINS, counts)
        ],
        "mean_ms": statistics.mean(values) if values else None,
        "median_ms": statistics.median(values) if values else None,
        "minimum_ms": min(values) if values else None,
        "maximum_ms": max(values) if values else None,
    }


def _round_capture(
    *,
    path: Path,
    identity: dict[str, str],
    baseline_keys: set[tuple[Any, ...]],
    started_at_ms: int,
    ended_at_ms: int,
    expected_rounds: int | None,
) -> dict[str, Any]:
    """Capture every new event for one request and report coverage explicitly."""

    events, read_error = _read_round_log(path)
    selected = []
    for event in events:
        if _round_event_key(event) in baseline_keys:
            continue
        if event.get("chat_id") != identity["id"]:
            continue
        if event.get("generation") != identity["generation"]:
            continue
        observed_at_ms = event.get("observed_at_ms")
        if (
            type(observed_at_ms) is not int
            or not started_at_ms <= observed_at_ms <= ended_at_ms
        ):
            continue
        selected.append(_public_round_record(event))

    round_numbers = [
        record["round"] for record in selected if record["round"] is not None
    ]
    unique_rounds = sorted(set(round_numbers))
    missing_rounds = (
        [
            number
            for number in range(1, unique_rounds[-1] + 1)
            if number not in unique_rounds
        ]
        if unique_rounds
        else []
    )
    duplicate_rounds = sorted(
        number for number in set(round_numbers) if round_numbers.count(number) > 1
    )
    measured = sum(record["round_ms"] is not None for record in selected)
    unmeasured = len(selected) - measured
    # generation_rounds comes from spec_decode_num_drafts_total. The first
    # prefill/first-token record has no draft and is intentionally retained,
    # but must not be charged against that speculative-round counter.
    speculative = sum((record.get("draft_tokens") or 0) > 0 for record in selected)
    count_matches = expected_rounds is not None and speculative == expected_rounds
    status = "unavailable" if read_error and not selected else "captured"
    if read_error and selected:
        status = "partial_read"
    if not read_error and (
        (expected_rounds is not None and not count_matches)
        or missing_rounds
        or duplicate_rounds
    ):
        status = "incomplete"
    return {
        "schema": "urn:coherence:decode-round-capture:v1",
        "source": str(path),
        "status": status,
        "read_error": read_error,
        "started_at_ms": started_at_ms,
        "ended_at_ms": ended_at_ms,
        "expected_rounds": expected_rounds,
        "expected_rounds_metric": "vllm:spec_decode_num_drafts_total",
        "speculative_round_count": speculative,
        "record_count": len(selected),
        "measured_round_count": measured,
        "unmeasured_round_count": unmeasured,
        "round_numbers": unique_rounds,
        "missing_round_numbers": missing_rounds,
        "duplicate_round_numbers": duplicate_rounds,
        "records": selected,
    }


def _capture_rounds_until_complete(
    *,
    path: Path,
    identity: dict[str, str],
    baseline_keys: set[tuple[Any, ...]],
    started_at_ms: int,
    expected_rounds: int | None,
    settle_timeout: float,
) -> dict[str, Any]:
    """Wait for the scheduler's queued events to reach the public feed."""

    deadline = time.monotonic() + max(0.0, settle_timeout)
    capture = _round_capture(
        path=path,
        identity=identity,
        baseline_keys=baseline_keys,
        started_at_ms=started_at_ms,
        ended_at_ms=int(time.time() * 1000),
        expected_rounds=expected_rounds,
    )
    while capture["status"] != "captured" and time.monotonic() < deadline:
        time.sleep(0.05)
        capture = _round_capture(
            path=path,
            identity=identity,
            baseline_keys=baseline_keys,
            started_at_ms=started_at_ms,
            ended_at_ms=int(time.time() * 1000),
            expected_rounds=expected_rounds,
        )
    return capture


def _round_log_snapshot(path: Path, identity: dict[str, str]) -> set[tuple[Any, ...]]:
    events, _ = _read_round_log(path)
    return {
        _round_event_key(event)
        for event in events
        if event.get("chat_id") == identity["id"]
        and event.get("generation") == identity["generation"]
    }


def _load_prefix(
    path: Path | None, expected_tokens: int
) -> tuple[list[int], str | None]:
    """Load and validate a token fixture without decoding or exposing its content."""

    if expected_tokens == 0:
        if path is not None:
            raise ValueError("the 0K arm must not be given a fixture")
        return [], None
    if path is None:
        raise ValueError(f"the {expected_tokens // 1000}K arm requires a fixture")
    fixture_bytes = path.read_bytes()
    try:
        fixture = json.loads(fixture_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"fixture is not valid JSON: {path}") from error
    prefix = fixture.get("prefix") if isinstance(fixture, dict) else None
    if (
        not isinstance(prefix, list)
        or len(prefix) != expected_tokens
        or any(type(token) is not int or token < 0 for token in prefix)
    ):
        raise ValueError(
            f"{path} must contain exactly {expected_tokens:,} non-negative integer prefix tokens"
        )
    return list(prefix), _sha256_bytes(fixture_bytes)


def _fixture_path(args: argparse.Namespace, context: str) -> Path | None:
    return {
        "0K": None,
        "60K": args.fixture_60k,
        "200K": args.fixture_200k,
    }[context]


def _requested_contexts(args: argparse.Namespace) -> tuple[str, ...]:
    if args.contexts == "all":
        return CONTEXTS
    return (args.contexts,)


def _report_row(
    result: dict[str, Any], context: str, prefix: list[int]
) -> dict[str, Any]:
    """Keep the reusable request result while making the context explicit."""

    return {
        "context": context,
        "prefix_tokens": len(prefix),
        **result,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested = _requested_contexts(args)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    report: dict[str, Any] = {
        "schema": "urn:coherence:pi-coding-contexts:v2",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "benchmark_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "coding_prompt_sha256": _sha256_bytes(CODING_PROMPT.encode()),
        "tokenizer_sha256": _sha256_bytes(args.tokenizer_json.read_bytes()),
        "runtime": json.loads(args.runtime_manifest.read_text())
        if args.runtime_manifest
        else None,
        "privacy": (
            "No prompt, response, decoded fixture text or token arrays are saved; "
            "only hashes and numeric measurements are retained."
        ),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "natural_stop_required": True,
        "ignore_eos": False,
        "coding_min_tokens": args.coding_min_tokens,
        "coding_max_tokens": args.coding_max_tokens,
        "model_context_tokens": args.model_context_tokens,
        "contexts_requested": list(requested),
        "round_telemetry": {
            "source": str(args.round_log),
            "retains_every_event": True,
            "privacy": "Only numeric scheduler round metadata is retained; no prompt or response data is copied.",
        },
        "fixture_sha256": {},
        "prefix_token_ids_sha256": {},
        "contexts": {},
        "validation_failures": [],
    }
    _write(args.output, report)

    # Render once with the live chat template.  The same suffix is reused for
    # each arm; only the supplied prefix and cache identity differ.
    rendered_coding = _render_user_turn(
        opener,
        args.base_url,
        CODING_PROMPT,
        thinking=False,
        timeout=args.request_timeout,
    )

    for context in requested:
        expected_tokens = CONTEXT_TOKEN_COUNTS[context]
        prefix, fixture_sha256 = _load_prefix(
            _fixture_path(args, context), expected_tokens
        )
        report["fixture_sha256"][context] = fixture_sha256
        report["prefix_token_ids_sha256"][context] = _prefix_digest(prefix)
        suffix = _turn_suffix(tokenizer, prefix, rendered_coding, first=True)
        prompt_tokens = prefix + suffix
        if len(prompt_tokens) + args.coding_max_tokens > args.model_context_tokens:
            raise ValueError(
                f"{context} prompt plus coding safety ceiling exceeds the model context"
            )
        identity_seed = f"{args.identity}:{context}:{_prefix_digest(prefix)}"
        identity = {
            "id": hashlib.sha256(identity_seed.encode()).hexdigest(),
            "generation": hashlib.sha256(
                f"{identity_seed}:initial".encode()
            ).hexdigest(),
            "title": f"coding context benchmark {context}",
            "cwd": "/qualification/coding-contexts",
            "session_file": "",
        }
        request_args = argparse.Namespace(
            base_url=args.base_url,
            abi=args.abi,
            identity=args.identity,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            request_timeout=args.request_timeout,
            metrics_timeout=args.metrics_timeout,
            coding_min_tokens=args.coding_min_tokens,
        )
        round_started_at_ms = int(time.time() * 1000)
        round_baseline_keys = _round_log_snapshot(args.round_log, identity)
        result, _output_ids, _completion = _request(
            opener=opener,
            args=request_args,
            identity=identity,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            suffix_tokens=suffix,
            stage="coding",
            max_tokens=args.coding_max_tokens,
            thinking=False,
        )
        result["round_capture"] = _capture_rounds_until_complete(
            path=args.round_log,
            identity=identity,
            baseline_keys=round_baseline_keys,
            started_at_ms=round_started_at_ms,
            expected_rounds=result.get("generation_rounds"),
            settle_timeout=args.round_log_settle_timeout,
        )
        result["round_capture"]["histogram"] = _round_histogram(
            result["round_capture"]["records"]
        )
        row = _report_row(result, context, prefix)
        report["contexts"][context] = row
        if not result["minimum_output_met"]:
            report["validation_failures"].append(
                {
                    "context": context,
                    "failure": (
                        f"natural completion produced {result['generated_tokens']:,} tokens; "
                        f"minimum is {args.coding_min_tokens:,}"
                    ),
                }
            )
        if result["round_capture"]["status"] != "captured":
            report["validation_failures"].append(
                {
                    "context": context,
                    "failure": (
                        "round telemetry did not provide one retained event for every "
                        f"generation round (status={result['round_capture']['status']})"
                    ),
                }
            )
        _write(args.output, report)
        print(
            json.dumps(
                {
                    "context": context,
                    "generated_tokens": result["generated_tokens"],
                    "mean_generation_round_ms": result["mean_generation_round_ms"],
                    "post_first_tokens_per_second": result[
                        "post_first_tokens_per_second"
                    ],
                    "peak_3s_tokens_per_second": result["peak_3s_tokens_per_second"],
                    "acceptance_rate": result["acceptance_rate"],
                    "round_capture_status": result["round_capture"]["status"],
                    "round_records": result["round_capture"]["record_count"],
                    "timed_round_records": result["round_capture"][
                        "measured_round_count"
                    ],
                    "untimed_round_records": result["round_capture"][
                        "unmeasured_round_count"
                    ],
                    "minimum_output_met": result["minimum_output_met"],
                }
            ),
            flush=True,
        )

    report["status"] = (
        "complete_with_validation_failure"
        if report["validation_failures"]
        else "complete"
    )
    report["completed_at"] = datetime.now(UTC).isoformat()
    _write(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-60k", type=Path)
    parser.add_argument("--fixture-200k", type=Path)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--abi", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--contexts",
        choices=("all", *CONTEXTS),
        default="all",
        help="run all three arms or one selected context (default: all)",
    )
    parser.add_argument("--coding-min-tokens", type=int, default=CODING_MIN_TOKENS)
    parser.add_argument("--coding-max-tokens", type=int, default=CODING_MAX_TOKENS)
    parser.add_argument(
        "--model-context-tokens", type=int, default=MODEL_CONTEXT_TOKENS
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-timeout", type=float, default=12.0)
    parser.add_argument(
        "--round-log",
        type=Path,
        default=DEFAULT_ROUND_LOG,
        help="content-free scheduler round feed to retain in the report",
    )
    parser.add_argument(
        "--round-log-settle-timeout",
        type=float,
        default=2.0,
        help="seconds to wait for queued round events after each request",
    )
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--identity", default="coherence-coding-contexts-v1")
    args = parser.parse_args()
    if (
        args.coding_min_tokens < 1
        or args.coding_max_tokens < args.coding_min_tokens
        or args.model_context_tokens < 1
        or args.top_k < 1
        or not 0 < args.top_p <= 1
        or args.temperature < 0
        or args.round_log_settle_timeout < 0
    ):
        parser.error(
            "coding maximum must cover a positive minimum; sampling and telemetry values are invalid"
        )
    for context in _requested_contexts(args):
        if context == "60K" and args.fixture_60k is None:
            parser.error("--fixture-60k is required when running the 60K arm")
        if context == "200K" and args.fixture_200k is None:
            parser.error("--fixture-200k is required when running the 200K arm")
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
