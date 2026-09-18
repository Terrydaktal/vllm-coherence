"""Synthetic release checks; never opens private sessions or logs generated text.

Run prepare against the temporary endpoint, restart, then run restore against
the promoted endpoint. State contains only this script's synthetic token IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from radiance_cache import ChatStore, cache_salt, request_tail_flush

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"


def post(endpoint, path, body):
    request = urllib.request.Request(
        endpoint + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.load(response)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def generate(endpoint, chat, prompt, *, logprobs=None):
    started = time.monotonic()
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 256,
        "ignore_eos": True,
        "temperature": 0,
        "top_k": 1,
        "seed": 0,
        "return_token_ids": True,
        "cache_salt": cache_salt(chat),
        "kv_transfer_params": {
            "qwen_chat": chat,
            "qwen_snapshot_abi": os.environ["QWEN_RADIANCE_CACHE_ABI"],
        },
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
    result = post(endpoint, "/v1/completions", payload)
    tokens = result["choices"][0]["token_ids"]
    assert len(tokens) == 256
    return {
        "seconds": round(time.monotonic() - started, 3),
        "tokens": len(tokens),
        "sha256": digest(tokens),
        "usage": result["usage"],
    }


def flush(chat):
    result = request_tail_flush(chat, timeout=120)
    assert result["status"] in ("flushed", "already_durable"), result["status"]
    return {key: result[key] for key in ("status", "tokens") if key in result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "seed", "ram", "restore"))
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    if args.mode in ("prepare", "seed"):
        assert not args.state.exists()
        chat = {
            "id": digest([str(args.state), "release"]),
            "generation": digest([str(args.state), "initial"]),
            "title": "Synthetic Radiance release qualification",
        }
        lines = "\n".join(
            f"Synthetic release record {index}: the test value is {index % 17}."
            for index in range(4096)
        )
        prompt = post(args.endpoint, "/tokenize", {"model": MODEL, "prompt": lines})["tokens"]
        print(json.dumps({"phase": "cold_prefix", "prompt_tokens": len(prompt)}), flush=True)
        cold = generate(args.endpoint, chat, prompt)
        print(json.dumps({"phase": "cold_complete", **cold}), flush=True)
        warm = generate(args.endpoint, chat, prompt)
        assert cold["sha256"] == warm["sha256"], "warm GPU reuse changed greedy output"
        # Requesting logprobs deliberately selects the full BF16 verify head.
        exact = generate(args.endpoint, chat, prompt, logprobs=1)
        assert cold["sha256"] == exact["sha256"], "verify head differs from full-head greedy output"
        flushed = flush(chat)
        tools = []
        for seed in range(8 if args.mode == "prepare" else 0):
            result = post(
                args.endpoint,
                "/v1/chat/completions",
                {
                    "model": MODEL,
                    "messages": [
                        {"role": "user", "content": "Call record with number 37. Use the tool."}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "record",
                                "description": "Record a number",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"number": {"type": "integer"}},
                                    "required": ["number"],
                                },
                            },
                        }
                    ],
                    "tool_choice": "auto",
                    "max_tokens": 512,
                    "temperature": 1,
                    "top_p": 0.95,
                    "top_k": 20,
                    "seed": seed,
                    "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "off"},
                },
            )
            choice = result["choices"][0]
            calls = choice["message"].get("tool_calls") or []
            assert choice["finish_reason"] == "tool_calls" and len(calls) == 1
            assert calls[0]["function"]["name"] == "record"
            assert json.loads(calls[0]["function"]["arguments"]) == {"number": 37}
            tools.append(
                {"seed": seed, "finish_reason": choice["finish_reason"], "usage": result["usage"]}
            )
        state = {"chat": chat, "prompt": prompt, "baseline": cold}
        args.state.write_text(json.dumps(state) + "\n")
        report = {
            "mode": args.mode,
            "prompt_tokens": len(prompt),
            "cold": cold,
            "warm": warm,
            "full_verify_head": exact,
            "tail_flush": flushed,
            "tool_cases": tools,
        }
    elif args.mode == "ram":
        state = json.loads(args.state.read_text())
        chat, prompt = state["chat"], state["prompt"]
        other = {**chat, "id": digest([chat["id"], "RAM handover"])}
        generate(args.endpoint, other, prompt[:8192])
        resumed = generate(args.endpoint, chat, prompt)
        assert resumed["sha256"] == state["baseline"]["sha256"], "RAM restore changed output"
        cached = resumed["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        assert cached >= len(prompt) - 2 * 1648, "RAM handover lost the live prefix"
        flush(other)
        flush(chat)
        worker = json.loads(Path("/dev/shm/qwen-radiance-fair-public-worker.json").read_text())
        assert worker["last_handover"] == "swap" and worker["last_transfer_bytes"] > 0
        report = {
            "mode": args.mode,
            "resumed": resumed,
            "transfer_bytes": worker["last_transfer_bytes"],
            "transfer_seconds": worker["last_transfer_seconds"],
        }
    else:
        state = json.loads(args.state.read_text())
        chat, prompt = state["chat"], state["prompt"]
        restored = generate(args.endpoint, chat, prompt)
        print(json.dumps({"phase": "disk_restored", **restored}), flush=True)
        assert restored["sha256"] == state["baseline"]["sha256"], (
            "disk restore changed greedy output"
        )
        cached = restored["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        # The settled draft tail excludes its volatile block; lookup keeps a
        # further rollback block, followed by the prompt's unaligned suffix.
        assert cached >= len(prompt) - 3 * 1648, "restart lost the stable DFlash prefix"
        flush(chat)
        successor = {**chat, "generation": digest([chat["generation"], "compacted"])}
        store = ChatStore(args.data_root, successor)
        store.activate()
        checkpoint_prompt = prompt[:8192]
        compacted = generate(args.endpoint, successor, checkpoint_prompt)
        flushed = flush(successor)
        metadata = store.metadata()
        assert metadata["generation"] == successor["generation"]
        assert metadata["gc"]["status"] == "complete"
        assert not (store.generations / chat["generation"]).exists(), (
            "old generation was not retired"
        )
        report = {
            "mode": args.mode,
            "restored": restored,
            "compacted": compacted,
            "tail_flush": flushed,
            "generation_advanced": True,
            "old_generation_collected": True,
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "mode": args.mode, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
