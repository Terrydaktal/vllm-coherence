"""Build isolated, public-data-only Pi sessions for the 250K qualification lane."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import tokenizers
from tokenizers import Tokenizer

TARGET_PROMPT_TOKENS = 249_957
MODEL_CONTEXT_TOKENS = 253_792
COMPLETION_CAP_TOKENS = 2_048
TARGET_OUTPUT_RATE_TPS = 70.0
DESTINATION_OUTPUT_RATE_TPS = 100.0
MAX_FIRST_DATA_SECONDS = 5.0
MIN_RATE_WINDOW_SECONDS = 3.0
MIN_OUTPUT_TOKENS = 512
DEFAULT_CLONE_COUNT = 3

BASE_SYSTEM_PROMPT = (
    "You are a deterministic throughput qualification assistant. Do not use tools. "
    "Follow the user's output-format instruction exactly."
)
BENCHMARK_PROMPT = (
    "Do not use tools. Output only the comma-separated integers from 1 through 512, "
    "with no explanation."
)
PUBLIC_PREFIX = (
    "PUBLIC SYNTHETIC PI 250K QUALIFICATION FIXTURE.\n"
    "This generated repetition contains no conversation, repository, or user transcript data.\n"
    "PUBLIC TOKENS:"
)
PUBLIC_SUFFIX = "\nEND PUBLIC SYNTHETIC QUALIFICATION FIXTURE.\n"
PADDING_UNITS = (" one", " red", " sun", " two", " map", " oak")
FIXED_TIMESTAMP = "2026-01-01T00:00:00.000Z"


class Pi250kFixtureError(RuntimeError):
    """The public fixture or its evidence contract is invalid."""


class TokenCounter(Protocol):
    """Minimal exact tokenizer interface used by the fixture builder."""

    def count(self, text: str) -> int:
        """Count raw text without tokenizer-added special tokens."""


class JsonTokenizerCounter:
    """Count with the exact Hugging Face ``tokenizer.json`` supplied by the operator."""

    def __init__(self, path: Path) -> None:
        try:
            self._tokenizer = Tokenizer.from_file(str(path))
        except Exception as error:  # tokenizers has multiple implementation exceptions
            raise Pi250kFixtureError(f"cannot load tokenizer JSON {path}: {error}") from error

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


def _canonical_value(value: object) -> object:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical_value(item) for key, item in value.items()}
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_value(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_system_prompt(base_system_prompt: str, cwd: Path) -> str:
    """Mirror Pi 0.84.2's custom-system-prompt working-directory suffix exactly."""

    return f"{base_system_prompt}\nCurrent working directory: {cwd.as_posix()}\n"


def _count_public_repetitions(counter: TokenCounter, unit: str, repetitions: int) -> int:
    return counter.count(f"{PUBLIC_PREFIX}{unit * repetitions}{PUBLIC_SUFFIX}")


def build_public_filler(counter: TokenCounter, target_tokens: int) -> tuple[str, str]:
    """Return public synthetic text with exactly ``target_tokens`` raw tokenizer tokens."""

    if target_tokens < 1:
        raise Pi250kFixtureError("raw filler target must be positive")

    empty = f"{PUBLIC_PREFIX}{PUBLIC_SUFFIX}"
    empty_tokens = counter.count(empty)
    if empty_tokens > target_tokens:
        raise Pi250kFixtureError(
            f"raw filler target {target_tokens} is smaller than the public envelope "
            f"({empty_tokens} tokens)"
        )

    for unit in PADDING_UNITS:
        count_for = functools.cache(functools.partial(_count_public_repetitions, counter, unit))

        low = 0
        high = max(32, (target_tokens - empty_tokens) * 2 + 32)
        while count_for(high) < target_tokens:
            high *= 2
            if high > target_tokens * 16 + 16_384:
                break

        while low <= high:
            middle = (low + high) // 2
            measured = count_for(middle)
            if measured < target_tokens:
                low = middle + 1
            elif measured > target_tokens:
                high = middle - 1
            else:
                filler = f"{PUBLIC_PREFIX}{unit * middle}{PUBLIC_SUFFIX}"
                pi_estimate = math.ceil(len(filler) / 4)
                tolerance = max(128, math.ceil(target_tokens * 0.01))
                if abs(pi_estimate - target_tokens) <= tolerance:
                    return filler, unit
                break

        # BPE counts are normally monotonic here, but inspect the boundary explicitly.
        for repetitions in range(max(0, high - 64), low + 65):
            if count_for(repetitions) != target_tokens:
                continue
            filler = f"{PUBLIC_PREFIX}{unit * repetitions}{PUBLIC_SUFFIX}"
            pi_estimate = math.ceil(len(filler) / 4)
            tolerance = max(128, math.ceil(target_tokens * 0.01))
            if abs(pi_estimate - target_tokens) <= tolerance:
                return filler, unit

    raise Pi250kFixtureError(
        f"could not construct exactly {target_tokens} raw tokens with a Pi-visible "
        "four-characters-per-token public unit"
    )


def _validate_build_arguments(
    *,
    target_prompt_tokens: int,
    model_context_tokens: int,
    completion_cap_tokens: int,
    calibrated_chat_overhead_tokens: int,
    clone_count: int,
    target_output_rate_tps: float,
    max_first_data_seconds: float,
    min_rate_window_seconds: float,
    min_output_tokens: int,
) -> None:
    if target_prompt_tokens < 1:
        raise Pi250kFixtureError("target prompt tokens must be positive")
    if model_context_tokens < 1 or completion_cap_tokens < 1:
        raise Pi250kFixtureError("model context and completion cap must be positive")
    if target_prompt_tokens + completion_cap_tokens > model_context_tokens:
        raise Pi250kFixtureError(
            "target prompt plus completion cap exceeds the qualified model context"
        )
    if calibrated_chat_overhead_tokens < 1:
        raise Pi250kFixtureError("calibrated chat overhead must be positive")
    if calibrated_chat_overhead_tokens >= target_prompt_tokens:
        raise Pi250kFixtureError("calibrated chat overhead leaves no raw filler budget")
    if not 3 <= clone_count <= 16:
        raise Pi250kFixtureError(
            "clone count must be between 3 and 16 (one cold prime plus two warm passes)"
        )
    if target_output_rate_tps < TARGET_OUTPUT_RATE_TPS:
        raise Pi250kFixtureError(
            f"output-rate gate cannot be below the {TARGET_OUTPUT_RATE_TPS:g} tok/s interim floor"
        )
    if not 0 < max_first_data_seconds <= 30:
        raise Pi250kFixtureError(
            "first-data qualification threshold must be above zero and at most 30s"
        )
    if min_rate_window_seconds <= 0 or min_output_tokens < 2:
        raise Pi250kFixtureError(
            "minimum rate window must be positive and output at least two tokens"
        )
    if min_output_tokens > completion_cap_tokens:
        raise Pi250kFixtureError("minimum output tokens cannot exceed the completion cap")


def _write_new_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    path.chmod(0o600)


def build_fixture(
    *,
    tokenizer_json: Path,
    output_dir: Path,
    cwd: Path,
    calibrated_chat_overhead_tokens: int,
    target_prompt_tokens: int = TARGET_PROMPT_TOKENS,
    model_context_tokens: int = MODEL_CONTEXT_TOKENS,
    completion_cap_tokens: int = COMPLETION_CAP_TOKENS,
    clone_count: int = DEFAULT_CLONE_COUNT,
    target_output_rate_tps: float = TARGET_OUTPUT_RATE_TPS,
    max_first_data_seconds: float = MAX_FIRST_DATA_SECONDS,
    min_rate_window_seconds: float = MIN_RATE_WINDOW_SECONDS,
    min_output_tokens: int = MIN_OUTPUT_TOKENS,
    base_system_prompt: str = BASE_SYSTEM_PROMPT,
    benchmark_prompt: str = BENCHMARK_PROMPT,
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """Create pristine Pi v3 session clones and their fail-closed request manifest."""

    _validate_build_arguments(
        target_prompt_tokens=target_prompt_tokens,
        model_context_tokens=model_context_tokens,
        completion_cap_tokens=completion_cap_tokens,
        calibrated_chat_overhead_tokens=calibrated_chat_overhead_tokens,
        clone_count=clone_count,
        target_output_rate_tps=target_output_rate_tps,
        max_first_data_seconds=max_first_data_seconds,
        min_rate_window_seconds=min_rate_window_seconds,
        min_output_tokens=min_output_tokens,
    )
    try:
        tokenizer_path = tokenizer_json.resolve(strict=True)
    except OSError as error:
        raise Pi250kFixtureError(f"tokenizer JSON does not exist: {tokenizer_json}") from error
    if not tokenizer_path.is_file():
        raise Pi250kFixtureError(f"tokenizer JSON is not a regular file: {tokenizer_path}")
    try:
        resolved_cwd = cwd.resolve(strict=True)
    except OSError as error:
        raise Pi250kFixtureError(f"session cwd does not exist: {cwd}") from error
    if not resolved_cwd.is_dir():
        raise Pi250kFixtureError(f"session cwd is not a directory: {resolved_cwd}")
    if any(character in resolved_cwd.as_posix() for character in "\r\n\x00"):
        raise Pi250kFixtureError("session cwd contains a control character")
    if not base_system_prompt or any(character in base_system_prompt for character in "\x00"):
        raise Pi250kFixtureError("base system prompt must be non-empty and contain no NUL")
    if not benchmark_prompt or any(character in benchmark_prompt for character in "\x00"):
        raise Pi250kFixtureError("benchmark prompt must be non-empty and contain no NUL")
    if output_dir.exists():
        raise Pi250kFixtureError(f"refusing to overwrite existing output path: {output_dir}")

    token_counter = counter if counter is not None else JsonTokenizerCounter(tokenizer_path)
    raw_filler_tokens = target_prompt_tokens - calibrated_chat_overhead_tokens
    filler, padding_unit = build_public_filler(token_counter, raw_filler_tokens)
    measured_raw_tokens = token_counter.count(filler)
    if measured_raw_tokens != raw_filler_tokens:
        raise Pi250kFixtureError("internal error: final public filler token count changed")

    effective_system = effective_system_prompt(base_system_prompt, resolved_cwd)
    filler_sha256 = _sha256_text(filler)
    effective_system_sha256 = _sha256_text(effective_system)
    benchmark_sha256 = _sha256_text(benchmark_prompt)
    tokenizer_sha256 = _file_sha256(tokenizer_path)
    fixture_seed = _canonical_bytes(
        {
            "benchmark_sha256": benchmark_sha256,
            "effective_system_sha256": effective_system_sha256,
            "filler_sha256": filler_sha256,
            "target_prompt_tokens": target_prompt_tokens,
        }
    )
    fixture_id = f"pi-250k-public-{_sha256_bytes(fixture_seed)[:20]}"

    output_dir.mkdir(parents=True, mode=0o700)
    output_dir.chmod(0o700)
    system_path = output_dir / "system-prompt.txt"
    benchmark_path = output_dir / "benchmark-prompt.txt"
    _write_new_text(system_path, base_system_prompt)
    _write_new_text(benchmark_path, benchmark_prompt)

    sessions: list[dict[str, Any]] = []
    for clone_index in range(1, clone_count + 1):
        session_id = f"{fixture_id}-{clone_index:02d}"
        entry_id = _sha256_text(f"{session_id}:public-filler")[:8]
        header = {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": FIXED_TIMESTAMP,
            "cwd": resolved_cwd.as_posix(),
        }
        entry = {
            "type": "custom_message",
            "id": entry_id,
            "parentId": None,
            "timestamp": FIXED_TIMESTAMP,
            "customType": "qwen-r9700-public-250k-fixture",
            "content": filler,
            "display": False,
            "details": {
                "fixture_id": fixture_id,
                "public_data_only": True,
                "raw_tokens": raw_filler_tokens,
                "sha256": filler_sha256,
            },
        }
        session_text = f"{json.dumps(header, separators=(',', ':'))}\n"
        session_text += f"{json.dumps(entry, separators=(',', ':'))}\n"
        session_path = output_dir / f"session-{clone_index:02d}.jsonl"
        _write_new_text(session_path, session_text)
        sessions.append(
            {
                "clone": clone_index,
                "purpose": "cold-prefix-prime" if clone_index == 1 else "warm-qualification",
                "path": session_path.name,
                "session_id": session_id,
                "sha256": _file_sha256(session_path),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "fixture_id": fixture_id,
        "created_at": datetime.now(UTC).isoformat(),
        "public_data_only": True,
        "source_session": None,
        "session_format_version": 3,
        "tokenizer": {
            "path": tokenizer_path.as_posix(),
            "sha256": tokenizer_sha256,
            "tokenizers_version": tokenizers.__version__,
            "add_special_tokens": False,
        },
        "token_accounting": {
            "target_prompt_tokens": target_prompt_tokens,
            "calibrated_chat_overhead_tokens": calibrated_chat_overhead_tokens,
            "raw_filler_tokens": raw_filler_tokens,
            "measured_raw_filler_tokens": measured_raw_tokens,
            "predicted_prompt_tokens": measured_raw_tokens + calibrated_chat_overhead_tokens,
            "model_context_tokens": model_context_tokens,
            "completion_cap_tokens": completion_cap_tokens,
            "reserved_context_tokens": model_context_tokens
            - target_prompt_tokens
            - completion_cap_tokens,
            "overhead_definition": (
                "Exact model chat-template token count for the effective system message, "
                "both user-message wrappers, benchmark message, assistant generation prefix, "
                "and boundary effects; excludes only raw filler tokens."
            ),
            "authoritative_runtime_gate": "usage.input + usage.cacheRead + usage.cacheWrite",
        },
        "public_filler": {
            "sha256": filler_sha256,
            "utf8_bytes": len(filler.encode()),
            "characters": len(filler),
            "pi_char_estimate_tokens": math.ceil(len(filler) / 4),
            "padding_unit": padding_unit,
            "description": "Deterministic repetition of public English number/colour words.",
        },
        "request_contract": {
            "provider": "qwen-r9700",
            "api": "openai-completions",
            "model": "qwen3.8-27b-frozenlock",
            "message_roles": ["system", "user", "user"],
            "message_count": 3,
            "base_system_prompt_sha256": _sha256_text(base_system_prompt),
            "effective_system_prompt_sha256": effective_system_sha256,
            "filler_message_sha256": filler_sha256,
            "benchmark_prompt_sha256": benchmark_sha256,
            "tools": "absent-or-empty",
            "stream": True,
            "sampling": {"temperature": 0, "top_p": 1, "top_k": 1},
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "xhigh",
            },
        },
        "qualification": {
            "target_output_rate_tps": target_output_rate_tps,
            "interim_output_rate_tps": TARGET_OUTPUT_RATE_TPS,
            "destination_output_rate_tps": DESTINATION_OUTPUT_RATE_TPS,
            "configured_milestone": (
                "destination"
                if target_output_rate_tps >= DESTINATION_OUTPUT_RATE_TPS
                else "interim"
            ),
            "rate_definition": (
                "(authoritative output tokens - output tokens reported with first data) / "
                "seconds after first data"
            ),
            "max_first_data_seconds": max_first_data_seconds,
            "min_rate_window_seconds": min_rate_window_seconds,
            "min_output_tokens": min_output_tokens,
            "required_passes": 2,
            "cold_prime_clone": 1,
            "warm_qualification_clones": list(range(2, clone_count + 1)),
        },
        "files": {
            "system_prompt": {
                "path": system_path.name,
                "sha256": _file_sha256(system_path),
            },
            "benchmark_prompt": {
                "path": benchmark_path.name,
                "sha256": _file_sha256(benchmark_path),
            },
            "sessions": sessions,
        },
        "safety": {
            "original_session_read": False,
            "original_session_modified": False,
            "fixture_displayed_in_tui": False,
            "guard_mismatch_action": "replace-with-fixed-public-one-token-request",
            "compaction": "cancelled-by-qualification-extension",
        },
    }
    manifest["manifest_payload_sha256"] = _sha256_bytes(_canonical_bytes(manifest))
    manifest_path = output_dir / "manifest.json"
    _write_new_text(manifest_path, f"{json.dumps(manifest, indent=2, sort_keys=True)}\n")

    return {
        "fixture_id": fixture_id,
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": _file_sha256(manifest_path),
        "sessions": sessions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pristine public-only Pi sessions for exact 249,957-token qualification."
    )
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--calibrated-chat-overhead-tokens", type=int, required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=TARGET_PROMPT_TOKENS)
    parser.add_argument("--model-context-tokens", type=int, default=MODEL_CONTEXT_TOKENS)
    parser.add_argument("--completion-cap-tokens", type=int, default=COMPLETION_CAP_TOKENS)
    parser.add_argument("--clone-count", type=int, default=DEFAULT_CLONE_COUNT)
    parser.add_argument("--target-output-rate-tps", type=float, default=TARGET_OUTPUT_RATE_TPS)
    parser.add_argument("--max-first-data-seconds", type=float, default=MAX_FIRST_DATA_SECONDS)
    parser.add_argument("--min-rate-window-seconds", type=float, default=MIN_RATE_WINDOW_SECONDS)
    parser.add_argument("--min-output-tokens", type=int, default=MIN_OUTPUT_TOKENS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_fixture(
            tokenizer_json=args.tokenizer_json,
            output_dir=args.output_dir,
            cwd=args.cwd,
            calibrated_chat_overhead_tokens=args.calibrated_chat_overhead_tokens,
            target_prompt_tokens=args.target_prompt_tokens,
            model_context_tokens=args.model_context_tokens,
            completion_cap_tokens=args.completion_cap_tokens,
            clone_count=args.clone_count,
            target_output_rate_tps=args.target_output_rate_tps,
            max_first_data_seconds=args.max_first_data_seconds,
            min_rate_window_seconds=args.min_rate_window_seconds,
            min_output_tokens=args.min_output_tokens,
        )
    except (OSError, Pi250kFixtureError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
