"""Small synthetic before/after check; never loads a saved chat."""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"


def post(path, payload):
    request = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        data=json.dumps({"model": MODEL, **payload}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    records = []
    for index in range(3):
        start = time.monotonic()
        result = post(
            "/v1/completions",
            {
                "prompt": (
                    "# Python example: implement a binary search with a docstring.\n"
                    "def binary_search(values, target):\n"
                ),
                "temperature": 0,
                "top_k": 1,
                "seed": 0,
                "max_tokens": 128,
                "ignore_eos": True,
                "return_token_ids": True,
            },
        )
        seconds = time.monotonic() - start
        tokens = result["choices"][0]["token_ids"]
        assert len(tokens) == 128
        records.append(
            {
                "index": index,
                "seconds": seconds,
                "output_tokens": len(tokens),
                "tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                "usage": result["usage"],
            }
        )
    result = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "Call add with a=2 and b=3."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two integers.",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "add"}},
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "top_k": 1,
            "max_tokens": 128,
        },
    )
    calls = result["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "add"
    assert json.loads(calls[0]["function"]["arguments"]) == {"a": 2, "b": 3}
    report = {"generations": records, "tool_call_passed": True}
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
