#!/usr/bin/env python3
"""Measure one streaming vLLM endpoint at exact occupied prompt depths."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


SOURCE_SUFFIXES = {".c", ".cpp", ".h", ".hip", ".js", ".json", ".md", ".py", ".sh", ".ts"}
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "node_modules",
    "__pycache__",
    "results",
}


def post_json(url: str, payload: dict[str, object], timeout: int) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def build_corpus(root: Path, minimum_chars: int) -> str:
    chunks: list[str] = []
    size = 0
    for path in sorted(root.rglob("*")):
        if size >= minimum_chars:
            break
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            content = path.read_text(errors="replace")
        except OSError:
            continue
        if not content.strip():
            continue
        chunk = f"\n\n===== {path.relative_to(root)} =====\n{content}"
        chunks.append(chunk)
        size += len(chunk)
    if not chunks:
        raise RuntimeError(f"no source corpus found below {root}")
    corpus = "".join(chunks)
    repetitions = max(1, (minimum_chars + len(corpus) - 1) // len(corpus))
    return (corpus * repetitions)[:minimum_chars]


def tokenize(endpoint: str, model: str, prompt: str, timeout: int) -> list[int]:
    result = post_json(
        f"{endpoint}/tokenize",
        {"model": model, "prompt": prompt},
        timeout,
    )
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError("tokenization endpoint did not return integer token IDs")
    return tokens


def stream_completion(
    endpoint: str,
    model: str,
    prompt_tokens: list[int],
    output_tokens: int,
    mode: str,
    timeout: int,
    cache_salt: str | None = None,
) -> dict[str, object]:
    if mode == "greedy":
        sampling = {"temperature": 0.0, "top_p": 1.0, "top_k": 1}
    else:
        sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt_tokens,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
        **sampling,
    }
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    request = urllib.request.Request(
        f"{endpoint}/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_data: float | None = None
    last_data: float | None = None
    usage: dict[str, object] = {}
    finish_reason: str | None = None
    completion_digest = hashlib.sha256()
    completion_chunks: list[str] = []
    completion_characters = 0
    raw_token_ids: list[int] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            event_usage = event.get("usage")
            if isinstance(event_usage, dict):
                usage = event_usage
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            text = choice.get("text")
            if text:
                now = time.monotonic()
                first_data = first_data or now
                last_data = now
                encoded = text.encode()
                completion_digest.update(encoded)
                completion_chunks.append(text)
                completion_characters += len(text)
            event_token_ids = choice.get("token_ids")
            if isinstance(event_token_ids, list) and all(
                isinstance(token_id, int) for token_id in event_token_ids
            ):
                raw_token_ids.extend(event_token_ids)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    ended = time.monotonic()
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        raise RuntimeError("stream ended without exact completion token usage")
    if first_data is None or last_data is None:
        raise RuntimeError("stream ended without generated data")
    decode_seconds = max(last_data - first_data, 1e-9)
    return {
        "mode": mode,
        "prompt_tokens": usage.get("prompt_tokens", len(prompt_tokens)),
        "completion_tokens": completion_tokens,
        "ttft_seconds": first_data - started,
        "decode_seconds_post_first": decode_seconds,
        "tokens_per_second_post_first": max(completion_tokens - 1, 0) / decode_seconds,
        "wall_seconds": ended - started,
        "finish_reason": finish_reason,
        "completion_characters": completion_characters,
        "completion_text_sha256": completion_digest.hexdigest(),
        "raw_token_ids": raw_token_ids or None,
        "completion_text": "".join(completion_chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus-root", type=Path, default=Path.cwd())
    parser.add_argument("--contexts", default="60000,117300")
    parser.add_argument("--modes", default="stochastic,greedy")
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write completion text and metadata for first-divergence analysis",
    )
    parser.add_argument(
        "--prompt-fixture-dir",
        type=Path,
        help="reuse or create exact prompt-token fixtures for differential replay",
    )
    parser.add_argument(
        "--master-prompt-fixture",
        type=Path,
        help="derive every requested context from prefixes of one integer-token fixture",
    )
    parser.add_argument(
        "--terminal-token",
        type=int,
        help="replace the final token of each master-fixture prefix with this token",
    )
    parser.add_argument(
        "--cache-salt-prefix",
        help="give every request an isolated prefix-cache namespace",
    )
    args = parser.parse_args()

    contexts = [int(value) for value in args.contexts.split(",")]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    unsupported = set(modes) - {"greedy", "stochastic"}
    if unsupported:
        raise SystemExit(f"unsupported modes: {sorted(unsupported)}")

    master_prompt_tokens: list[int] | None = None
    if args.master_prompt_fixture is not None:
        master_prompt_tokens = json.loads(args.master_prompt_fixture.read_text())
        if not isinstance(master_prompt_tokens, list) or not all(
            isinstance(token, int) for token in master_prompt_tokens
        ):
            raise RuntimeError(
                f"invalid master prompt-token fixture: {args.master_prompt_fixture}"
            )
        required = max(contexts) - (1 if args.terminal_token is not None else 0)
        if len(master_prompt_tokens) < required:
            raise RuntimeError(
                f"master fixture has {len(master_prompt_tokens)} tokens; need {required}"
            )
        suffix_tokens: list[int] = []
        corpus_tokens: list[int] = []
    else:
        suffix = (
            "\n\nAnalyze the preceding implementation as a senior systems engineer. "
            "Produce a detailed technical review covering architecture, correctness, "
            "failure atomicity, performance, testing, and concrete improvements."
        )
        suffix_tokens = tokenize(args.endpoint, args.model, suffix, args.timeout)
        maximum_context = max(contexts)
        corpus = build_corpus(args.corpus_root, maximum_context * 6)
        corpus_tokens = tokenize(args.endpoint, args.model, corpus, args.timeout)
        if len(corpus_tokens) < maximum_context:
            raise RuntimeError(
                f"corpus has only {len(corpus_tokens)} tokens; need {maximum_context}"
            )

    for context in contexts:
        fixture_path = None
        if master_prompt_tokens is not None:
            prefix_length = context - (1 if args.terminal_token is not None else 0)
            prompt_tokens = master_prompt_tokens[:prefix_length]
            if args.terminal_token is not None:
                prompt_tokens = prompt_tokens + [args.terminal_token]
        elif args.prompt_fixture_dir is not None:
            fixture_path = args.prompt_fixture_dir / f"context-{context}.json"
        if master_prompt_tokens is not None:
            pass
        elif fixture_path is not None and fixture_path.exists():
            prompt_tokens = json.loads(fixture_path.read_text())
            if not isinstance(prompt_tokens, list) or not all(
                isinstance(token, int) for token in prompt_tokens
            ):
                raise RuntimeError(f"invalid prompt-token fixture: {fixture_path}")
            if len(prompt_tokens) != context:
                raise RuntimeError(
                    f"prompt-token fixture has {len(prompt_tokens)} tokens; "
                    f"expected {context}: {fixture_path}"
                )
        else:
            prefix_count = context - len(suffix_tokens)
            if prefix_count <= 0:
                raise ValueError(f"context {context} is too small for benchmark suffix")
            prompt_tokens = corpus_tokens[:prefix_count] + suffix_tokens
            if fixture_path is not None:
                fixture_path.parent.mkdir(parents=True, exist_ok=True)
                fixture_path.write_text(json.dumps(prompt_tokens, separators=(",", ":")) + "\n")
        for mode in modes:
            result = stream_completion(
                args.endpoint,
                args.model,
                prompt_tokens,
                args.output_tokens,
                mode,
                args.timeout,
                (
                    f"{args.cache_salt_prefix}-{context}-{mode}"
                    if args.cache_salt_prefix is not None
                    else None
                ),
            )
            completion_text = result.pop("completion_text")
            if args.output_dir is not None:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                stem = f"context-{context}-{mode}"
                (args.output_dir / f"{stem}.txt").write_text(completion_text)
                (args.output_dir / f"{stem}.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n"
                )
            console_result = dict(result)
            token_ids = console_result.pop("raw_token_ids")
            if token_ids is not None:
                packed_ids = ",".join(str(token_id) for token_id in token_ids).encode()
                console_result["raw_token_ids_count"] = len(token_ids)
                console_result["raw_token_ids_sha256"] = hashlib.sha256(packed_ids).hexdigest()
            print(json.dumps(console_result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
