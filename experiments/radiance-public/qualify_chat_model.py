"""Small model-level checks for compressed cache reuse and isolated compaction."""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

from radiance_cache import ChatStore, cache_salt, report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--phase", choices=("seed", "resume"), required=True)
parser.add_argument("--state-file", type=Path, required=True)
args = parser.parse_args()
model = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"


def post(path, body=None):
    request = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        value = response.read()
        return json.loads(value) if value else None


def chat(name, generation="initial"):
    return {
        "id": hashlib.sha256(f"storage-qualification-20260907-{name}".encode()).hexdigest(),
        "generation": hashlib.sha256(generation.encode()).hexdigest(),
        "title": f"Storage qualification {name}",
        "cwd": "/qualification",
        "session_file": "",
    }


def row(info):
    return next(item for item in report(args.data_root)["chats"] if item["id"] == info["id"])


def generate(info, tokens):
    started = time.monotonic()
    result = post(
        "/v1/completions",
        {
            "model": model,
            "prompt": tokens,
            "max_tokens": 8,
            "temperature": 0,
            "top_k": 1,
            "cache_salt": cache_salt(info),
            "kv_transfer_params": {
                "qwen_chat": info,
                "qwen_snapshot_force_flush": True,
            },
        },
    )
    print(
        json.dumps(
            {
                "chat": info["title"],
                "generation": info["generation"],
                "seconds": time.monotonic() - started,
                "usage": result.get("usage"),
            }
        ),
        flush=True,
    )
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        current = row(info)
        if current["status"] == "ready":
            return result, current
        time.sleep(0.1)
    raise RuntimeError(f"snapshot head did not publish: {row(info)}")


tokens = post(
    "/tokenize",
    {
        "model": model,
        "prompt": "\n".join(
            f"Record {i}: preserve the current chat and its exact cached attention state."
            for i in range(1300)
        ),
    },
)["tokens"]
assert len(tokens) > 14000
one, two = chat("A"), chat("B")
if args.phase == "seed":
    first, first_size = generate(one, tokens[:10000])
    second, second_size = generate(two, tokens[10000:14000])
    args.state_file.write_text(
        json.dumps({"first": first, "first_size": first_size, "second_size": second_size})
    )
    print("SEED_READY_FOR_BACKEND_RESTART", flush=True)
    sys.exit(0)
# The caller must restart the backend between these two phases, dropping both
# GPU prefix caching and the CPU offload arena before the replay below.
seed = json.loads(args.state_file.read_text())
first, first_size, second_size = seed["first"], seed["first_size"], seed["second_size"]
restored, restored_size = generate(one, tokens[:10000])
assert restored["usage"]["prompt_tokens_details"]["cached_tokens"] >= 6592, restored
assert restored["choices"][0]["text"] == first["choices"][0]["text"]
results = {
    "first": first_size,
    "other_chat": second_size,
    "disk_restore_usage": restored["usage"],
    "cycles": [],
}
for cycle in range(3):
    compacted = chat("A", f"compaction-{cycle}")
    retirement = ChatStore(args.data_root, compacted).activate()
    assert retirement["removed_file_bytes"] > 0
    assert row(two)["file_bytes"] == second_size["file_bytes"]
    assert row(compacted)["file_bytes"] == 0
    _, compacted_size = generate(compacted, tokens[:4000])
    assert len(list(ChatStore(args.data_root, compacted).generations.iterdir())) == 1
    results["cycles"].append({"retirement": retirement, "snapshot": compacted_size})
assert max(item["snapshot"]["file_bytes"] for item in results["cycles"]) < first_size["file_bytes"]
print(json.dumps({"result": "pass", **results}), flush=True)
