"""Measure prefill versus forced generation on the same synthetic token history."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from qualify_speed_lifecycle import Suite


async def run(args):
    root = args.output
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    original = json.loads(args.fixture.read_text())
    seed_prefix = original["prompt"][:1651]
    if not seed_prefix or args.prefix_rows < 1 or args.rows < 1:
        raise ValueError("positive prefix and comparison lengths required")
    prefix = (
        seed_prefix * ((args.prefix_rows + len(seed_prefix) - 1) // len(seed_prefix))
    )[: args.prefix_rows]
    cfg = SimpleNamespace(
        base_url=args.base_url,
        abi=args.abi,
        run_id=f"prefill-difference-{time.time_ns()}",
        output=root / "suite.json",
        status_prefix=Path("/dev/shm/qwen-prefill-diagnosis"),
        observe=False,
        case_group="prefill-difference",
    )
    async with httpx.AsyncClient(base_url=args.base_url, timeout=1200) as client:
        suite = Suite(cfg, client)
        suite.sampling = {"temperature": 0, "top_k": 129, "seed": 0}

        async def complete(job):
            # A near-limit cold prefill can exceed the lifecycle suite's short
            # request timeout. Keep its completion validation once it finishes.
            await asyncio.wait_for(asyncio.shield(job.task), 1200)
            return await job.result()

        corpus = root / "synthetic-corpus.json"
        if corpus.exists():
            data = json.loads(corpus.read_text())
            assert data["synthetic"] and data["prefix"] == prefix
            output = data["output"]
            assert len(output) == args.rows + 1
        else:
            natural = suite.stream("synthetic-corpus", prefix, args.rows + 1)
            natural_result = await complete(natural)
            await suite.absent(natural.chat)
            output = natural.ids
            corpus.write_text(
                json.dumps({"synthetic": True, "prefix": prefix, "output": output})
                + "\n"
            )
            print(json.dumps({"phase": "corpus_frozen", **natural_result}), flush=True)
        await suite.rpc("qwen_prefill_probe_reload")
        first = len(prefix) - 3 if args.stages or args.gdn else len(prefix)
        if args.first is not None:
            first = args.first
        stages = (
            {"window": args.stage_window, "limit": args.stage_limit}
            if args.stages
            else False
        )
        await suite.rpc(
            "qwen_prefill_capture_begin",
            [str(root / "prefill"), first, None, stages, args.attention, args.gdn],
        )
        try:
            cold = suite.stream("full-prefill", prefix + output[:-1], 1)
            start = time.monotonic()
            result = await complete(cold)
            print(
                json.dumps(
                    {
                        "phase": "full_prefill_request",
                        "wall_seconds": time.monotonic() - start,
                        **result,
                    }
                ),
                flush=True,
            )
            await suite.absent(cold.chat)
        finally:
            print(
                json.dumps(
                    {
                        "phase": "prefill_capture",
                        **await suite.rpc("qwen_prefill_capture_finish"),
                    }
                ),
                flush=True,
            )
        await suite.rpc(
            "qwen_prefill_capture_begin",
            [
                str(root / "decode"),
                first,
                str(corpus),
                stages,
                args.attention,
                args.gdn,
            ],
        )
        try:
            decode = suite.stream("forced-decode", prefix, len(output))
            result = await complete(decode)
            print(json.dumps({"phase": "forced_decode_request", **result}), flush=True)
            await suite.absent(decode.chat)
        finally:
            record = await suite.rpc("qwen_prefill_capture_finish")
            print(
                json.dumps(
                    {
                        "phase": "decode_capture",
                        "batches": len(record["batches"]),
                        "forced_complete": record["forced_complete"],
                    }
                ),
                flush=True,
            )
        comparison = await suite.rpc(
            "qwen_prefill_compare", [str(root / "prefill"), str(root / "decode")]
        )
        print(json.dumps({"phase": "comparison", **comparison}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--abi", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--prefix-rows", type=int, default=1651)
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--gdn", action="store_true")
    parser.add_argument("--first", type=int)
    parser.add_argument("--stage-window", type=int, default=8)
    parser.add_argument("--stage-limit", type=int, default=10000)
    asyncio.run(run(parser.parse_args()))
