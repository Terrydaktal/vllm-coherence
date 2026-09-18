"""Qualify the live RAM tail journal with synthetic tokens and content-free output."""

import hashlib
import json
import shutil
import time
import urllib.request
import uuid
from pathlib import Path

from radiance_cache import FORMAT, ChatStore, cache_salt, request_tail_flush, sync_directory

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"
DATA_ROOT = Path(
    "/cache/snapshots/74aef30706ffab186ee2fb89d3827c9895496918bdfb3d61188196daeced5b94/data"
)
TAIL_STATUS = Path("/dev/shm/qwen-radiance-snapshot-tail.json")
CHAT = {
    "id": hashlib.sha256(
        f"radiance-tail-journal-live-qualification-v1:{uuid.uuid4().hex}".encode()
    ).hexdigest(),
    "generation": hashlib.sha256(b"initial").hexdigest(),
    "title": "Synthetic tail-journal qualification",
    "cwd": "/qualification",
    "session_file": "",
}


def post(path, body):
    request = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def generate(tokens, chat=CHAT):
    result = post(
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": tokens,
            "max_tokens": 4,
            "ignore_eos": True,
            "temperature": 0,
            "top_k": 1,
            "return_token_ids": True,
            "cache_salt": cache_salt(chat),
            "kv_transfer_params": {"qwen_chat": chat},
        },
    )
    return result["choices"][0]["token_ids"], result["usage"]


def wait_for_tail(tokens):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        status = json.loads(TAIL_STATUS.read_text())
        row = next((item for item in status["chats"] if item["chat_id"] == CHAT["id"]), None)
        if row is not None and row["tokens"] == tokens:
            return row
        time.sleep(0.1)
    raise TimeoutError("synthetic RAM tail did not appear")


def cleanup():
    directory = DATA_ROOT / FORMAT / CHAT["id"]
    if not directory.exists():
        return
    # Do not retire a synthetic generation unless the live manager first proves
    # that no acknowledged RAM tail or active request remains.
    request_tail_flush(CHAT, timeout=30)
    successor_chat = {
        **CHAT,
        "generation": hashlib.sha256(b"qualification-cleanup").hexdigest(),
    }
    successor = ChatStore(DATA_ROOT, successor_chat)
    successor.activate()
    if not successor.publish([], 0, 1):
        raise RuntimeError("could not publish empty synthetic cleanup generation")
    shutil.rmtree(successor.directory)
    sync_directory(successor.managed)


def qualify_stale_generation(token):
    stale = {
        **CHAT,
        "id": hashlib.sha256(f"stale:{uuid.uuid4().hex}".encode()).hexdigest(),
        "generation": hashlib.sha256(b"retired").hexdigest(),
        "title": "Synthetic stale-generation qualification",
    }
    ChatStore(DATA_ROOT, stale).activate()
    successor_chat = {
        **stale,
        "generation": hashlib.sha256(b"current").hexdigest(),
    }
    successor = ChatStore(DATA_ROOT, successor_chat)
    successor.activate()
    try:
        _, usage = generate([token], stale)
        return usage["completion_tokens"]
    finally:
        if not successor.publish([], 0, 1):
            raise RuntimeError("could not publish stale-request cleanup generation")
        shutil.rmtree(successor.directory)
        sync_directory(successor.managed)


def main():
    existing = DATA_ROOT / FORMAT / CHAT["id"]
    if existing.exists():
        raise RuntimeError("synthetic qualification identity already exists")
    try:
        source = post(
            "/tokenize",
            {
                "model": MODEL,
                "prompt": "\n".join(
                    f"Synthetic immutable cache record {index}." for index in range(1200)
                ),
            },
        )["tokens"]
        assert len(source) > 7000
        stale_completion_tokens = qualify_stale_generation(source[0])
        prompt = source[:7000]
        first_output, first_usage = generate(prompt)
        first_tokens = len(prompt) + first_usage["completion_tokens"]
        first_tail = wait_for_tail(first_tokens)
        store = ChatStore(DATA_ROOT, CHAT)
        first_io = store.io_totals()
        first_metadata = store.metadata()
        assert first_metadata["tokens"] == 0
        assert 0 < first_tail["blocks"] <= 15 and first_tail["bytes"] > 0

        continuation = [*prompt, *first_output]
        _, second_usage = generate(continuation)
        second_tokens = len(continuation) + second_usage["completion_tokens"]
        second_tail = wait_for_tail(second_tokens)
        second_io = store.io_totals()
        assert second_tail["durable_tokens"] == 0
        assert second_io["written_file_bytes"] == first_io["written_file_bytes"]

        flushed = request_tail_flush(CHAT, timeout=45)
        metadata = store.metadata()
        final_io = store.io_totals()
        assert flushed["status"] == "flushed"
        assert metadata["status"] == "ready" and metadata["tokens"] == second_tokens
        assert final_io["written_file_bytes"] > second_io["written_file_bytes"]
        print(
            json.dumps(
                {
                    "result": "pass",
                    "stale_generation_completion_tokens": stale_completion_tokens,
                    "prompt_tokens": len(prompt),
                    "first_cached_tokens": first_usage["prompt_tokens_details"]["cached_tokens"],
                    "second_cached_tokens": second_usage["prompt_tokens_details"]["cached_tokens"],
                    "tail_blocks": second_tail["blocks"],
                    "tail_raw_bytes": second_tail["bytes"],
                    "immutable_bytes_after_first": first_io["written_file_bytes"],
                    "immutable_bytes_after_second": second_io["written_file_bytes"],
                    "bytes_after_flush": final_io["written_file_bytes"],
                },
                sort_keys=True,
            )
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
