"""Verify RAM-to-disk snapshot repair with synthetic token prompts only.

The script never opens Pi sessions or prints model text. It removes one object from
its own synthetic snapshot, evicts that chat's GPU bank, and proves the next RAM
cache hit recreates and republishes only the missing durable object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

from radiance_cache import FORMAT, ChatStore, cache_salt

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
        "title": f"Synthetic snapshot repair qualification {label}",
    }


def wait_ready(store: ChatStore, expected_tokens: int, required: tuple[Path, ...] = ()) -> dict:
    deadline = time.monotonic() + 60
    while True:
        with store.lock():
            metadata = store.metadata()
            present = all(path.is_file() for path in required)
        if (
            metadata.get("status") == "ready"
            and metadata.get("tokens") == expected_tokens
            and metadata.get("publication", {}).get("result") == "committed"
            and present
        ):
            return metadata
        if time.monotonic() >= deadline:
            raise RuntimeError(
                json.dumps(
                    {
                        "status": metadata.get("status"),
                        "tokens": metadata.get("tokens"),
                        "publication": metadata.get("publication"),
                        "required_present": present,
                    }
                )
            )
        time.sleep(0.1)


def generate(endpoint: str, root: Path, chat: dict, prompt: list[int], count: int = 8):
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
                "qwen_snapshot_force_flush": True,
            },
        },
    )
    output = result["choices"][0]["token_ids"]
    assert len(output) == count
    store = ChatStore(root, chat)
    metadata = wait_ready(store, len(prompt) + count)
    return result, output, store, metadata


def evict_gpu_bank(endpoint: str, prompt: list[int]) -> None:
    result = post(
        endpoint,
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": prompt[:32],
            "max_tokens": 8,
            "ignore_eos": True,
            "temperature": 0,
            "top_k": 1,
        },
    )
    assert result["usage"]["completion_tokens"] == 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("qualification output must be a new file")

    run_id = hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()
    chats = [identity(run_id, "repair")]
    prompt = post(
        args.endpoint,
        "/tokenize",
        {
            "model": MODEL,
            "prompt": "\n".join(
                f"Synthetic durable repair record {index}: preserve exact cache state."
                for index in range(360)
            ),
        },
    )["tokens"]
    assert len(prompt) > 3296

    try:
        first, first_ids, store, metadata = generate(
            args.endpoint, args.data_root, chats[0], prompt
        )
        head = list(metadata["head"])
        assert head
        paths = tuple(store.path(key) for key in head)
        before_sizes = {path.name: path.stat().st_size for path in paths}
        before_io = store.io_totals()
        for path in paths:
            path.unlink()
        assert not any(path.exists() for path in paths)

        # Two tiny unlabelled requests force the repair chat out of the two-bank
        # GPU cache without writing another durable snapshot or displacing the
        # just-written blocks from the 18 GiB primary RAM tier.
        evict_gpu_bank(args.endpoint, prompt)
        evict_gpu_bank(args.endpoint, prompt[64:])

        second, second_ids, _, repaired = generate(args.endpoint, args.data_root, chats[0], prompt)
        wait_ready(store, len(prompt) + 8, paths)
        after_full_repair = store.io_totals()
        assert first_ids == second_ids, "RAM repair changed greedy output"
        cached_tokens = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert cached_tokens > 0, "the full-head repair did not use the RAM tier"
        assert set(repaired["head"]) == set(head)
        assert {path.name: path.stat().st_size for path in paths} == before_sizes
        assert after_full_repair["written_blocks"] == before_io["written_blocks"] + len(head)

        # With the RAM source now proven, repeat the damage with one object and
        # require the immutable writer to reuse every intact object.
        missing = paths[0]
        missing.unlink()
        evict_gpu_bank(args.endpoint, prompt[128:])
        evict_gpu_bank(args.endpoint, prompt[192:])
        third, third_ids, _, _ = generate(args.endpoint, args.data_root, chats[0], prompt)
        wait_ready(store, len(prompt) + 8, (missing,))
        after_single_repair = store.io_totals()
        assert first_ids == third_ids, "incremental repair changed greedy output"
        third_cached = third["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert third_cached > 0
        assert missing.stat().st_size == before_sizes[missing.name]
        assert after_single_repair["written_blocks"] == after_full_repair["written_blocks"] + 1
        assert after_single_repair["reused_blocks"] > after_full_repair.get("reused_blocks", 0)
        report = {
            "synthetic": True,
            "passed": True,
            "prompt_tokens": len(prompt),
            "cached_tokens_after_damage": cached_tokens,
            "cached_tokens_after_single_damage": third_cached,
            "head_blocks": len(head),
            "full_repair_blocks": after_full_repair["written_blocks"] - before_io["written_blocks"],
            "incremental_repair_blocks": after_single_repair["written_blocks"]
            - after_full_repair["written_blocks"],
            "incremental_reused_blocks": after_single_repair["reused_blocks"]
            - after_full_repair.get("reused_blocks", 0),
            "repaired_file_bytes": after_single_repair["written_file_bytes"]
            - before_io["written_file_bytes"],
            "output_token_sha256": hashlib.sha256(
                json.dumps(first_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "seed_cached_tokens": first["usage"]["prompt_tokens_details"]["cached_tokens"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        managed = args.data_root / FORMAT
        for chat in chats:
            directory = managed / chat["id"]
            if directory.is_dir() and not directory.is_symlink():
                shutil.rmtree(directory)


if __name__ == "__main__":
    main()
