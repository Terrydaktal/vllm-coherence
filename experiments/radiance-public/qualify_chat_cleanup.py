"""Synthetic cold-prefill/cleanup and disk-restore check; never reads Pi sessions.

Run seed before updating the backend, check afterward, then restart once more
and run resume. State contains synthetic token IDs only. Use a unique state file.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from radiance_cache import ChatStore, cache_salt

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"


def post(path, body):
    request = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.load(response)


def snapshot(root, chat):
    store = ChatStore(root, chat)
    with store.lock():
        metadata = store.metadata()
        objects = {p.name: p.stat().st_size for p in store.generation.glob("*.qkv")}
    return metadata, objects


def generate(root, chat, tokens, count=8):
    started = time.monotonic()
    result = post(
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": tokens,
            "max_tokens": count,
            "ignore_eos": True,
            "temperature": 0,
            "top_k": 1,
            "return_token_ids": True,
            "cache_salt": cache_salt(chat),
            "kv_transfer_params": {
                "qwen_chat": chat,
                "qwen_snapshot_force_flush": True,
            },
        },
    )
    expected = len(tokens) + result["usage"]["completion_tokens"]
    deadline = time.monotonic() + 45
    while True:
        metadata, objects = snapshot(root, chat)
        if metadata["status"] == "ready" and metadata["tokens"] == expected:
            assert set(objects) == set(metadata["head"]), "unreferenced blocks remain"
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                json.dumps(
                    {
                        "status": metadata["status"],
                        "published_tokens": metadata["tokens"],
                        "expected_tokens": expected,
                        "publication": metadata.get("publication"),
                    }
                )
            )
        time.sleep(0.1)
    print(
        json.dumps(
            {
                "chat_id": chat["id"],
                "seconds": round(time.monotonic() - started, 2),
                "usage": result["usage"],
                "blocks": len(objects),
                "bytes": sum(objects.values()),
                "status": metadata["status"],
                "gc": metadata.get("gc"),
            }
        ),
        flush=True,
    )
    return result["choices"][0]["token_ids"], result["usage"], objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("seed", "check", "resume"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    if args.phase == "seed":
        if args.state_file.exists():
            raise ValueError("seed requires a new state file")
        seed = post(
            "/tokenize",
            {
                "model": MODEL,
                "prompt": "\n".join(
                    f"Synthetic record {i}: retain exact cached state." for i in range(25000)
                ),
            },
        )["tokens"]
        assert len(seed) >= 208633
        chats = [
            {
                "id": hashlib.sha256(f"{args.state_file.resolve()}:{name}".encode()).hexdigest(),
                "generation": hashlib.sha256(b"initial").hexdigest(),
                "title": f"Synthetic cleanup qualification {name}",
            }
            for name in ("prefill", "other")
        ]
        original, _, _ = generate(root, chats[0], seed[:10000])
        _, _, other = generate(root, chats[1], seed[10000:14000])
        state = {"chats": chats, "seed": seed, "original": original, "other": other}
        args.state_file.write_text(json.dumps(state))
        print("SEED_READY_FOR_UPDATED_BACKEND", flush=True)
        return

    state = json.loads(args.state_file.read_text())
    one, two = state["chats"]
    if args.phase == "check":
        restored, usage, _ = generate(root, one, state["seed"][:10000])
        assert usage["prompt_tokens_details"]["cached_tokens"] >= 6592
        assert restored == state["original"], "old-runtime snapshot output changed"
        # Change the very first token to force a full prefill in the same chat.
        tokens = [state["seed"][10], *state["seed"][1:208633]]
        assert tokens[0] != state["seed"][0]
        output, usage, _ = generate(root, one, tokens, 2519)
        assert usage["prompt_tokens_details"]["cached_tokens"] == 0
        state["continuation"] = tokens + output
        expected, warm_usage, _ = generate(root, one, state["continuation"])
        assert warm_usage["prompt_tokens_details"]["cached_tokens"] > 200000
        state["expected_continuation"] = expected
        args.state_file.write_text(json.dumps(state))
        assert snapshot(root, two)[1] == state["other"], "other chat changed"
        print("FULL_PREFILL_CLEANUP_PASSED; READY_FOR_DISK_RESTART", flush=True)
        return

    restored, usage, _ = generate(root, one, state["continuation"])
    assert usage["prompt_tokens_details"]["cached_tokens"] > 200000, "long snapshot did not restore"
    assert restored == state["expected_continuation"], "disk restore changed greedy output"
    assert snapshot(root, two)[1] == state["other"], "other chat changed"
    print("DISK_RESTORE_AND_ISOLATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
