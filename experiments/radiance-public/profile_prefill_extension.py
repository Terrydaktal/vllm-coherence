"""Attribute one real prefill extension to native kernels in an isolated server.

Synthetic data only. This trace is for locating work, not measuring production
throughput. The independent unprofiled paired driver supplies wall-clock timing.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from qualify_speed_lifecycle import Suite


async def run(args):
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise ValueError("a new capture path is required")
    fixture = json.loads(args.fixture.read_text())
    if fixture.get("synthetic") is not True:
        raise ValueError("explicitly synthetic history required")
    seed = fixture["prefix"]
    prefix = (seed * ((args.context + len(seed) - 1) // len(seed)))[: args.context]
    cfg = SimpleNamespace(
        abi=args.abi,
        run_id=f"prefill-attribution-{time.time_ns()}",
        output=args.output.with_suffix(".json"),
        observe=False,
        case_group="prefill-attribution",
        status_prefix=Path("/dev/shm/qwen-prefill-diagnosis"),
    )
    async with httpx.AsyncClient(base_url=args.base_url, timeout=1200) as client:
        suite = Suite(cfg, client)
        suite.sampling = {"temperature": 0, "top_k": 129, "seed": 0}
        await suite.rpc("qwen_prefill_probe_reload")
        chat = suite.chat("extension")
        if args.resume_chat:
            metadata = json.loads(args.resume_chat.read_text())
            if metadata.get("cwd") != "/qualification" or not metadata.get(
                "title", ""
            ).startswith("Synthetic lifecycle "):
                raise ValueError(
                    "only a disposable synthetic benchmark bank may resume"
                )
            chat = {
                "id": args.resume_chat.parent.name,
                "generation": metadata["generation"],
                "title": metadata["title"],
                "cwd": "/qualification",
                "session_file": "",
            }
            if args.warm_extensions < 1:
                raise ValueError("resuming requires an unprofiled reuse check")
            # Extend the processed endpoint. Re-requesting exactly the same
            # prefix needs its final logits again and may replay a cache block.
            prompt = prefix + seed[: args.rows]
        else:
            initial = suite.stream("warm-prefix", prefix, 1, chat)
            await asyncio.wait_for(asyncio.shield(initial.task), 1200)
            await initial.result()
            await suite.absent(chat)
            prompt = prefix + initial.ids + seed[: args.rows]
        for index in range(args.warm_extensions):
            warm = suite.stream(f"warm-extension-{index}", prompt, 1, chat)
            warm_result = await warm.result()
            if (
                args.resume_chat
                and index == 0
                and (warm_result["usage"].get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                )
                < args.context - 8
            ):
                raise AssertionError("the requested warm prefix was not resident")
            await suite.absent(chat)
            prompt += warm.ids + seed[: args.rows]
        await suite.rpc("qwen_prefill_kernel_profile", [str(args.output)])
        try:
            job = suite.stream("extension", prompt, 1, chat)
            result = await job.result()
            await suite.absent(chat)
        finally:
            saved = await suite.rpc("qwen_prefill_kernel_profile")
        record = {
            "status": "CAPTURED",
            "scope": "Profiled kernel attribution, not throughput",
            "context": args.context,
            "requested_extension_rows": args.rows,
            "warm_extensions": args.warm_extensions,
            "resumed_synthetic_bank": bool(args.resume_chat),
            "usage": result["usage"],
            "trace": saved,
        }
        args.output.with_suffix(".summary.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--abi", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=60000)
    parser.add_argument("--rows", type=int, default=1648)
    parser.add_argument("--warm-extensions", type=int, default=1)
    parser.add_argument(
        "--resume-chat",
        type=Path,
        help="Explicit chat.json of an already warm synthetic benchmark bank",
    )
    asyncio.run(run(parser.parse_args()))
