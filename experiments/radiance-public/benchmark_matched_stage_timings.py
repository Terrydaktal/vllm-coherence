#!/usr/bin/env python3
"""Bracket kernel traces with natural serving controls on identical requests."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import benchmark_pi_coding_contexts as coding
from benchmark_pi_coding_json_compaction import (
    CODING_PROMPT, _render_user_turn, _request, _turn_suffix,
)


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def run(args):
    from tokenizers import Tokenizer

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    args.output.mkdir(parents=True, exist_ok=True)

    def rpc(method, *values):
        request = urllib.request.Request(
            args.base_url + "/collective_rpc",
            json.dumps({"method": method, "args": values, "timeout": 300}).encode(),
            {"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=360) as response:
            payload = json.load(response)
        results = payload.get("results") if isinstance(payload, dict) else payload
        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError("collective RPC must return exactly one worker result")
        return results[0]

    metadata = rpc("qwen_optimized_metadata")
    if metadata["enforce_eager"] or not metadata["compilation_mode"]:
        raise RuntimeError("compiled execution required before benchmark warmup")
    rendered = _render_user_turn(opener, args.base_url, CODING_PROMPT,
                                 thinking=False, timeout=args.request_timeout)
    report = {"schema": "urn:coherence:matched-stage-serving:v1", "status": "running",
              "started_at": time.time(), "sampling": {"temperature": args.temperature,
              "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed},
              "natural_stop": True, "contexts": {}, "privacy": "Numeric telemetry and hashes only."}
    write(args.output / "report.json", report)
    for context in args.contexts.split(","):
        path = {"0K": None, "60K": args.fixture_60k, "200K": args.fixture_200k}[context]
        prefix, fixture_hash = coding._load_prefix(path, coding.CONTEXT_TOKEN_COUNTS[context])
        suffix = _turn_suffix(tokenizer, prefix, rendered, first=True)
        prompt = prefix + suffix
        seed = f"{args.identity}:{context}"
        identity = {"id": hashlib.sha256(seed.encode()).hexdigest(),
                    "generation": hashlib.sha256((seed + ':initial').encode()).hexdigest(),
                    "title": "Matched stage timing benchmark", "cwd": "/qualification/stage-timing",
                    "session_file": ""}
        section = {"fixture_sha256": fixture_hash,
                   "prefix_sha256": coding._prefix_digest(prefix),
                   "prompt_sha256": coding._prefix_digest(prompt), "arms": {}}
        report["contexts"][context] = section
        for arm in ("warmup", "control_before", "profile", "control_after"):
            arm_root = args.output / (context + "-" + arm)
            report["current"] = {"context": context, "arm": arm, "started_at": time.time()}
            write(args.output / "report.json", report)
            print(json.dumps(report["current"]), flush=True)
            if arm != "warmup":
                armed = rpc("qwen_timing_arm", str(arm_root),
                            "profile" if arm == "profile" else "control",
                            args.warmup_rounds, args.chunk_rounds, args.profile_rounds)
                metadata = armed["metadata"]
                if metadata["enforce_eager"] or not metadata["compilation_mode"]:
                    raise RuntimeError("compiled execution required")
            baseline = coding._round_log_snapshot(args.round_log, identity)
            started_ms = int(time.time() * 1000)
            try:
                result, ids, completion = _request(
                    opener=opener, args=args, identity=identity, tokenizer=tokenizer,
                    prompt_tokens=prompt, suffix_tokens=suffix, stage="coding",
                    max_tokens=10000, thinking=False,
                )
                # The existing streaming consumer keeps text in memory only;
                # immediately discard it. Never serialize prompt/output arrays.
                del ids, completion
            finally:
                if arm != "warmup":
                    worker = rpc("qwen_timing_finish")
            result["round_capture"] = coding._capture_rounds_until_complete(
                path=args.round_log, identity=identity, baseline_keys=baseline,
                started_at_ms=started_ms, expected_rounds=result.get("generation_rounds"),
                settle_timeout=10,
            )
            if arm != "warmup":
                result["worker_file"] = str(arm_root / "worker.json")
                result["worker_decode_rounds"] = len(worker["rows"])
                result["trace_chunks"] = len(worker["chunks"])
            section["arms"][arm] = result
            write(args.output / "report.json", report)
            print(json.dumps({"context": context, "arm": arm,
                              "tokens": result["generated_tokens"],
                              "round_ms": result["mean_generation_round_ms"],
                              "output_sha256": result["output_sha256"]}), flush=True)
        outputs = {row["output_sha256"] for row in section["arms"].values()}
        section["identical_outputs"] = len(outputs) == 1
        write(args.output / "report.json", report)
    report.update(status="complete", finished_at=time.time())
    write(args.output / "report.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--fixture-60k", required=True, type=Path)
    p.add_argument("--fixture-200k", required=True, type=Path)
    p.add_argument("--tokenizer-json", required=True, type=Path)
    p.add_argument("--abi", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8081")
    p.add_argument("--contexts", default="0K,60K,200K")
    p.add_argument("--identity", default="matched-stage-20260923")
    p.add_argument("--round-log", type=Path, default=Path("/dev/shm/qwen-stage-timing-rounds.jsonl"))
    p.add_argument("--warmup-rounds", type=int, default=64)
    p.add_argument("--chunk-rounds", type=int, default=128)
    p.add_argument("--profile-rounds", type=int, default=1152)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=.95)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--request-timeout", type=float, default=1800)
    p.add_argument("--metrics-timeout", type=float, default=30)
    p.add_argument("--coding-min-tokens", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
