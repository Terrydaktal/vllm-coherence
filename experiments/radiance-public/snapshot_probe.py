#!/usr/bin/env python3
"""Capture and replay an exact greedy continuation around a server restart."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from benchmark_server import build_corpus, post_json, tokenize


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus-root", type=Path, default=Path.cwd())
    parser.add_argument("--context", type=int, default=60000)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--max-offload-tokens",
        type=int,
        help="Cap this request's snapshot stores while retaining normal lookup",
    )
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    if args.prompt_file.exists():
        prompt_record = json.loads(args.prompt_file.read_text())
        prompt_tokens = prompt_record["token_ids"]
        if len(prompt_tokens) != args.context:
            raise RuntimeError(
                f"preserved prompt has {len(prompt_tokens)} tokens, expected {args.context}"
            )
    else:
        suffix = (
            "\n\nContinue with a rigorous systems analysis of the preceding source. "
            "Cover correctness, transactional state, performance, failure modes and tests."
        )
        suffix_tokens = tokenize(args.endpoint, args.model, suffix, args.timeout)
        corpus = build_corpus(args.corpus_root, args.context * 6)
        corpus_tokens = tokenize(args.endpoint, args.model, corpus, args.timeout)
        prefix_count = args.context - len(suffix_tokens)
        if prefix_count <= 0 or len(corpus_tokens) < prefix_count:
            raise RuntimeError("source corpus could not supply the requested prompt depth")
        prompt_tokens = corpus_tokens[:prefix_count] + suffix_tokens
        prompt_record = {
            "schema": "qwen-radiance-snapshot-probe-prompt-v1",
            "model": args.model,
            "token_count": len(prompt_tokens),
            "token_ids_sha256": sha256_json(prompt_tokens),
            "token_ids": prompt_tokens,
        }
        write_json(args.prompt_file, prompt_record)

    payload: dict[str, object] = {
        "model": args.model,
        "prompt": prompt_tokens,
        "max_tokens": args.output_tokens,
        "min_tokens": args.output_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "return_token_ids": True,
    }
    if args.max_offload_tokens is not None:
        if args.max_offload_tokens < 0:
            raise RuntimeError("--max-offload-tokens must be non-negative")
        payload["kv_transfer_params"] = {
            "max_offload_tokens": args.max_offload_tokens
        }
    started = time.monotonic()
    response = post_json(f"{args.endpoint}/v1/completions", payload, args.timeout)
    wall_seconds = time.monotonic() - started
    choice = response["choices"][0]
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list) or not all(isinstance(token, int) for token in token_ids):
        raise RuntimeError("completion did not return exact output token IDs")
    result = {
        "schema": "qwen-radiance-snapshot-probe-result-v1",
        "prompt_token_count": len(prompt_tokens),
        "prompt_token_ids_sha256": sha256_json(prompt_tokens),
        "completion_token_count": len(token_ids),
        "completion_token_ids_sha256": sha256_json(token_ids),
        "completion_text_sha256": hashlib.sha256(choice["text"].encode()).hexdigest(),
        "finish_reason": choice.get("finish_reason"),
        "wall_seconds": wall_seconds,
        "usage": response.get("usage"),
        "token_ids": token_ids,
    }

    if args.reference:
        reference = json.loads(args.reference.read_text())
        result["reference_file"] = str(args.reference)
        result["exact_token_match"] = token_ids == reference["token_ids"]
        result["exact_text_hash_match"] = (
            result["completion_text_sha256"] == reference["completion_text_sha256"]
        )
        if not result["exact_token_match"] or not result["exact_text_hash_match"]:
            write_json(args.result_file, result)
            raise SystemExit("restored continuation differs from the pre-restart reference")

    write_json(args.result_file, result)
    print(json.dumps({key: value for key, value in result.items() if key != "token_ids"}))


if __name__ == "__main__":
    main()
