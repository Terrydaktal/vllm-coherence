"""Qualify uninterrupted two-chat GPU responses with synthetic token prompts only.

The script never opens Pi sessions or prints generated text. It compares hashes of
greedy token IDs, records scheduler state transitions, and removes its own cache
namespaces after the result has been written.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path

from radiance_cache import FORMAT, cache_salt, request_tail_flush

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"


def post(endpoint: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(
        endpoint + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.load(response)


def identity(run_id: str, label: str) -> dict[str, str]:
    def digest(value: str) -> str:
        return hashlib.sha256(f"{run_id}:{value}".encode()).hexdigest()

    return {
        "id": digest(label),
        "generation": digest(label + ":generation"),
        "title": f"Synthetic fair scheduler qualification {label}",
    }


def generate(endpoint: str, chat: dict, prompt: list[int], count: int) -> dict:
    started = time.monotonic()
    result = post(
        endpoint,
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": count,
            "ignore_eos": True,
            "temperature": 0,
            "top_k": 1,
            "seed": 0,
            "return_token_ids": True,
            "cache_salt": cache_salt(chat),
            "kv_transfer_params": {
                "qwen_chat": chat,
                "qwen_snapshot_abi": os.environ.get("QWEN_RADIANCE_CACHE_ABI"),
            },
        },
    )
    token_ids = result["choices"][0]["token_ids"]
    assert len(token_ids) == count
    return {
        "seconds": time.monotonic() - started,
        "tokens": len(token_ids),
        "sha256": hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest(),
        "usage": result["usage"],
    }


def load_status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--status-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=3000)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("qualification output must be a new file")
    if args.tokens < 2048:
        raise ValueError("qualification must be long enough to exercise a handover")

    run_id = hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()
    prompts = [
        post(
            args.endpoint,
            "/tokenize",
            {
                "model": MODEL,
                "prompt": "\n".join(
                    f"Synthetic scheduler record {side} {index}: deterministic cache state."
                    for index in range(64)
                ),
            },
        )["tokens"]
        for side in ("alpha", "beta")
    ]
    baseline_chats = [identity(run_id, f"baseline-{side}") for side in ("alpha", "beta")]
    concurrent_chats = [identity(run_id, f"concurrent-{side}") for side in ("alpha", "beta")]
    all_chats = baseline_chats + concurrent_chats
    scheduler_path = args.status_prefix.with_name(args.status_prefix.name + "-scheduler.json")
    worker_path = args.status_prefix.with_name(args.status_prefix.name + "-worker.json")

    try:
        baseline = [
            generate(args.endpoint, chat, prompt, args.tokens)
            for chat, prompt in zip(baseline_chats, prompts, strict=True)
        ]
        before = load_status(scheduler_path) or {}
        transitions: list[dict] = []
        stop = threading.Event()

        def observe() -> None:
            previous = None
            while not stop.wait(0.1):
                value = load_status(scheduler_path)
                if value is None:
                    continue
                states = sorted(
                    (row["chat_id"], row["state"], row["computed_tokens"])
                    for row in value.get("requests", [])
                    if row["chat_id"] in {chat["id"] for chat in concurrent_chats}
                )
                signature = (value.get("switches"), states)
                if states and signature != previous:
                    transitions.append(
                        {
                            "elapsed": time.monotonic(),
                            "switches": value.get("switches"),
                            "states": states,
                        }
                    )
                    previous = signature

        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        concurrent_started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(generate, args.endpoint, chat, prompt, args.tokens)
                for chat, prompt in zip(concurrent_chats, prompts, strict=True)
            ]
            concurrent_results = [future.result() for future in futures]
        stop.set()
        observer.join()
        concurrent_seconds = time.monotonic() - concurrent_started
        after = load_status(scheduler_path) or {}
        worker = load_status(worker_path) or {}
        switch_delta = int(after.get("switches", 0)) - int(before.get("switches", 0))
        assert switch_delta == 2, "expected one admission and one response-boundary handover"
        assert int(worker.get("switches", 0)) >= switch_delta
        assert int(worker.get("allocated_bytes", 0)) > 0
        for expected, observed in zip(baseline, concurrent_results, strict=True):
            assert expected["sha256"] == observed["sha256"], "handover changed greedy output"
        first_time = transitions[0]["elapsed"] if transitions else concurrent_started
        for row in transitions:
            row["elapsed"] = round(row["elapsed"] - first_time, 3)
            row["states"] = [
                {"chat_id": chat_id, "state": state, "computed_tokens": tokens}
                for chat_id, state, tokens in row["states"]
            ]
        assert any(
            {state["state"] for state in row["states"]} >= {"running", "queued"}
            for row in transitions
        ), "no concurrent running/queued state was observed"
        assert not any(
            state["state"] == "paused" for row in transitions for state in row["states"]
        ), "a response was paused mid-generation"
        running_order = []
        for row in transitions:
            running = [state["chat_id"] for state in row["states"] if state["state"] == "running"]
            assert len(running) <= 1
            if running and (not running_order or running_order[-1] != running[0]):
                running_order.append(running[0])
        assert len(running_order) == 2 and len(set(running_order)) == 2, (
            "the GPU must serve each concurrent response once without switching back"
        )
        assert after.get("quantum_seconds") == 0, "time slicing is still enabled"
        report = {
            "synthetic": True,
            "passed": True,
            "policy": "response_boundary",
            "generated_tokens_per_request": args.tokens,
            "prompt_tokens": [len(value) for value in prompts],
            "baseline": baseline,
            "concurrent": concurrent_results,
            "concurrent_wall_seconds": concurrent_seconds,
            "scheduler_switches": switch_delta,
            "worker": {
                key: worker.get(key)
                for key in (
                    "allocated_bytes",
                    "reserved_capacity_bytes",
                    "last_transfer_bytes",
                    "last_transfer_seconds",
                    "transferred_bytes",
                    "transfer_seconds",
                    "last_allocation_bytes",
                    "last_allocation_seconds",
                    "allocation_events",
                    "allocation_seconds",
                    "generation_replacements",
                    "last_handover",
                )
            },
            "transitions": transitions,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "transitions"}))
    finally:
        managed = args.data_root / FORMAT
        for chat in all_chats:
            directory = managed / chat["id"]
            if directory.is_dir() and not directory.is_symlink():
                # The backend can still hold the just-completed request's tail
                # after HTTP completion. Drain it before deleting our namespace
                # so later RAM eviction cannot retry writes into a deleted chat.
                request_tail_flush(chat)
                shutil.rmtree(directory)


if __name__ == "__main__":
    main()
