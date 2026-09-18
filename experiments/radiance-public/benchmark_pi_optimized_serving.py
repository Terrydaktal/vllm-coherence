"""Bounded serving comparison using an existing private token fixture, metrics only."""

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

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
)


def run(args):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def metrics():
        with opener.open(args.base_url + "/metrics", timeout=10) as response:
            text = response.read().decode()
        result = {}
        for name in METRICS:
            matches = re.findall(
                r"^" + re.escape(name) + r"(?:\{[^\n]*\})? ([\d.eE+\-]+)$", text, re.M
            )
            result[name] = sum(map(float, matches))
        return result

    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    tokens = fixture["prefix"]
    if len(tokens) != 60000 or not all(type(t) is int and t >= 0 for t in tokens):
        raise ValueError("expected the existing 60K Pi token fixture")
    identity = {
        "id": hashlib.sha256(args.identity.encode()).hexdigest(),
        "generation": hashlib.sha256(b"initial").hexdigest(),
        "title": "60K optimized serving benchmark",
        "cwd": "/qualification/optimized-serving-benchmark",
        "session_file": "",
    }
    report = {
        "variant": args.variant,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "prefix_tokens": len(tokens),
        "max_output_tokens": args.tokens,
        "privacy": "No text/token arrays printed or saved; existing fixture sent to local backend",
        "sampling": {"temperature": args.temperature, "top_p": 0.95, "top_k": 20},
        "runs": [],
    }
    for seed in (0, 17, 42):
        before = metrics()
        if before["vllm:num_requests_running"] or before["vllm:num_requests_waiting"]:
            raise RuntimeError("backend busy; benchmark did not submit")
        body = {
            "model": "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate",
            "prompt": tokens,
            "max_tokens": args.tokens,
            "seed": seed,
            **report["sampling"],
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "cache_salt": f"qwen-chat-cache-v1:{identity['id']}:{identity['generation']}",
            "kv_transfer_params": {"qwen_chat": identity, "qwen_snapshot_abi": args.abi},
        }
        request = urllib.request.Request(
            args.base_url + "/v1/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        started = time.monotonic()
        first_at = last_at = None
        first_count = 0
        usage = {}
        reason = None
        output_hash = hashlib.sha256()
        with opener.open(request, timeout=600) as response:
            for raw in response:
                if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                    continue
                event = json.loads(raw[6:])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    text = choice.get("text") or ""
                    if text:
                        last_at = time.monotonic()
                        if first_at is None:
                            first_at, first_count = last_at, usage.get("completion_tokens", 0)
                        output_hash.update(text.encode())
                    reason = choice.get("finish_reason") or reason
        if first_at is None or last_at <= first_at:
            raise RuntimeError("no measurable generation interval")
        # Prometheus consumes the engine's periodic stats, so let the last batch arrive.
        time.sleep(6)
        after = metrics()
        delta = {k: after[k] - before[k] for k in METRICS if k.endswith(("total", "sum", "count"))}
        rounds = delta["vllm:spec_decode_num_drafts_total"]
        drafts = delta["vllm:spec_decode_num_draft_tokens_total"]
        intervals = delta["vllm:inter_token_latency_seconds_count"]
        result = {
            "seed": seed,
            "usage": usage,
            "finish_reason": reason,
            "first_token_seconds": first_at - started,
            "after_first_seconds": last_at - first_at,
            "after_first_tps": (usage["completion_tokens"] - first_count) / (last_at - first_at),
            "first_stream_token_count": first_count,
            "mean_round_interval_ms": 1000
            * delta["vllm:inter_token_latency_seconds_sum"]
            / intervals
            if intervals
            else None,
            "draft_acceptance": delta["vllm:spec_decode_num_accepted_tokens_total"] / drafts
            if drafts
            else None,
            "output_tokens_per_round": delta["vllm:generation_tokens_total"] / rounds
            if rounds
            else None,
            "metrics_delta": delta,
            "output_sha256": output_hash.hexdigest(),
        }
        report["runs"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--abi", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0, help="Pi defaults to 1.0")
    parser.add_argument("--identity", default="optimized-pi-60k-head-20260918")
    run(parser.parse_args())
