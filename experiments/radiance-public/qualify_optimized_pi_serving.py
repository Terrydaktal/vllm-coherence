"""Public-input serving check through the VM relay; never opens Pi transcripts."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"


def tool_check(args):
    """Exercise short, graph-padded decode through Pi's streaming tool protocol."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Call report_ready with ready set to true."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "report_ready",
                    "parameters": {
                        "type": "object",
                        "properties": {"ready": {"type": "boolean"}},
                        "required": ["ready"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "report_ready"}},
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 64,
        "temperature": 0,
        "top_k": 1,
        "stream": True,
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        args.base_url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    started = time.monotonic()
    calls, finish_reason = {}, None
    with opener.open(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError("streamed backend error during public tool check")
            for choice in event.get("choices", []):
                finish_reason = choice.get("finish_reason") or finish_reason
                for call in choice.get("delta", {}).get("tool_calls", []):
                    target = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                    for key in target:
                        target[key] += call.get("function", {}).get(key, "")
    passed = len(calls) == 1 and calls[0]["name"] == "report_ready"
    passed = passed and json.loads(calls[0]["arguments"]) == {"ready": True}
    report = {
        "public_tool_call_passed": passed,
        "finish_reason": finish_reason,
        "seconds": time.monotonic() - started,
        "tool_executed": False,
        "private_chat_read": False,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError("public tool emission failed")


def run(args):
    if args.phase == "tool":
        return tool_check(args)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(path, body):
        request = urllib.request.Request(
            args.base_url + path,
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=180) as response:
            return json.loads(response.read())

    config = post("/qwen-radiance/control", {"operation": "config"})
    if config["abi"] != args.abi:
        raise RuntimeError("VM relay has not adopted the new snapshot identity")

    def identity(name):
        return {
            "id": hashlib.sha256(
                ("optimized-pi-serving-20260918-10k-" + name).encode()
            ).hexdigest(),
            "generation": hashlib.sha256(b"initial").hexdigest(),
            "title": "Public optimized serving check " + name,
            "cwd": "/workspace/optimized-serving-check",
            "session_file": "/workspace/optimized-serving-check/.pi/sessions/public-check.jsonl",
        }

    text = "\n".join(
        f"Record {i}: the public test value is {i * 7}; retain each record in order."
        for i in range(1100)
    )
    tokens = post("/tokenize", {"model": MODEL, "prompt": text})["tokens"]
    if len(tokens) < 14000:
        raise RuntimeError("public fixture is unexpectedly short")

    def generate(name, prompt):
        info = identity(name)
        started = time.monotonic()
        result = post(
            "/v1/completions",
            {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 16,
                "temperature": 0,
                "top_k": 1,
                "seed": 0,
                "cache_salt": f"qwen-chat-cache-v1:{info['id']}:{info['generation']}",
                "kv_transfer_params": {
                    "qwen_chat": info,
                    "qwen_snapshot_abi": args.abi,
                    "qwen_snapshot_force_flush": True,
                },
            },
        )
        return {
            "seconds": time.monotonic() - started,
            "usage": result["usage"],
            "output_sha256": hashlib.sha256(result["choices"][0]["text"].encode()).hexdigest(),
            "finish_reason": result["choices"][0]["finish_reason"],
        }

    if args.phase == "seed":
        # Use the established durable-cache qualification geometry. Short
        # prefixes can hit the GPU prefix cache without a restorable disk tail.
        first = generate("A", tokens[:10000])
        second = generate("B", tokens[10000:14000])
        reused = generate("A", tokens[:10000])
        report = {"abi": args.abi, "first": first, "second": second, "handover_reuse": reused}
        report["checks"] = {
            "handover_output_equal": first["output_sha256"] == reused["output_sha256"],
            "handover_reused_context": reused["usage"]["prompt_tokens_details"]["cached_tokens"]
            >= 6592,
        }
    else:
        report = json.loads(args.report.read_text())
        if report["abi"] != args.abi:
            raise RuntimeError("restart check belongs to a different runtime")
        restored = generate("A", tokens[:10000])
        report["disk_restore"] = restored
        report["checks"].update(
            disk_restore_output_equal=restored["output_sha256"] == report["first"]["output_sha256"],
            disk_restore_reused_context=restored["usage"]["prompt_tokens_details"]["cached_tokens"]
            >= 6592,
        )
    report["private_chat_read"] = False
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not all(report["checks"].values()):
        raise RuntimeError("serving check failed; evidence retained in report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--abi", required=True)
    parser.add_argument("--phase", choices=("seed", "resume", "tool"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    run(parser.parse_args())
