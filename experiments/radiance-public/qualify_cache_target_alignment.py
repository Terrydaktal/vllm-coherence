"""Find the first target-state difference on cold versus reused synthetic prompts."""

import argparse
import asyncio
import json
import os
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import httpx
from qualify_combined_continuity import sha, write
from qualify_speed_lifecycle import Suite


def prompt_ids(base_url, shared_control):
    import benchmark_pi_coding_contexts as coding
    import benchmark_pi_coding_json_compaction as chain
    from tokenizers import Tokenizer

    section = json.loads((shared_control / "runs/60K/report.json").read_text())[
        "contexts"
    ]["60K"]
    prefix, digest = coding._load_prefix(Path("/fixtures/60k.json"), 60_000)
    tokenizer = Tokenizer.from_file(
        "/models/Qwen3.8-27B-Uncensored-MXFP4-awq/tokenizer.json"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    turn = chain._render_user_turn(
        opener, base_url, chain.CODING_PROMPT, thinking=False, timeout=60
    )
    prompt = prefix + chain._turn_suffix(tokenizer, prefix, turn, first=True)
    if (
        digest != section["fixture_sha256"]
        or coding._prefix_digest(prompt) != section["prompt_sha256"]
    ):
        raise ValueError("synthetic prefix differs from 60K benchmark")
    return prompt


def prefill_rows(path, first, prompt):
    import torch

    intervals, rows = [], {}
    for item in json.loads((path / "capture.json").read_text())["batches"]:
        batch = torch.load(path / item["file"], weights_only=True, map_location="cpu")
        pp = batch["positions"]
        if not pp or pp[0] >= len(prompt):
            continue
        if pp != list(range(pp[0], pp[-1] + 1)) or pp[-1] >= len(prompt):
            raise ValueError("expected a single contiguous in-prompt target batch")
        if batch["input_ids"] != prompt[pp[0] : pp[-1] + 1]:
            raise ValueError("target input IDs differ from the verified common prompt")
        intervals.append([pp[0], pp[-1] + 1])
        for i in range(max(0, first - pp[0]), len(pp)):
            if pp[i] in rows:
                raise ValueError("multiple valid prefill rows at one position")
            rows[pp[i]] = batch["hidden"][i].clone()
    if not intervals or set(rows) != set(range(first, len(prompt))):
        raise ValueError("target capture misses common-prefix rows")
    return intervals, torch.stack([rows[p] for p in range(first, len(prompt))])


async def run(args):
    import torch

    os.umask(0o077)
    root = args.root
    root.mkdir(mode=0o700, exist_ok=False)
    prompt = prompt_ids(args.base_url, args.shared_control)
    if not root.resolve().is_relative_to("/prefill-diagnosis/refresh"):
        raise ValueError("expected isolated qualification root")
    binding = json.loads(args.build_manifest.read_text())["sha256"]
    report = {
        "binding": binding,
        "prompt_sha256": sha(prompt),
        "prompt_tokens": len(prompt),
        "cases": [],
    }
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:
        suite = Suite(
            SimpleNamespace(
                run_id=f"target-prefix-alignment-{time.time_ns()}",
                abi=args.abi,
                observe=False,
                case_group="target-prefix-alignment",
                status_prefix=Path("/dev/shm/qwen-prefill-diagnosis"),
            ),
            client,
        )
        owner = suite.chat("one-identical-prompt")
        for arm in ("cold", "warm"):
            path = root / arm
            stages = (
                {"window": args.stage_window, "limit": 2500}
                if args.stage_at is not None
                else False
            )
            await suite.rpc(
                "qwen_prefill_capture_begin",
                [str(path), args.stage_at or 0, None, stages],
            )
            armed = False
            try:
                if args.state_snapshot or args.disable_indexed_880:
                    await suite.rpc(
                        "qwen_target_cache_boundary",
                        [
                            "begin",
                            str(path / "recurrent-state.json"),
                            args.disable_indexed_880,
                        ],
                    )
                    armed = True
                body = suite.body(owner, prompt, 128)
                body.update(
                    temperature=1.0,
                    top_k=40,
                    top_p=0.95,
                    seed=0,
                    ignore_eos=False,
                    add_special_tokens=False,
                    stop=["<|im_end|>"],
                )
                r = await client.post("/v1/completions", json=body)
                r.raise_for_status()
                v = r.json()
                choice = v["choices"][0]
                usage = v["usage"]
                cached = usage["prompt_tokens_details"]["cached_tokens"]
                ids = choice["token_ids"]
                if (
                    not len(ids) == usage["completion_tokens"] == 128
                    or usage["prompt_tokens"] != len(prompt)
                    or (cached > 0) != (arm == "warm")
                ):
                    raise ValueError("unexpected response in paired target alignment")
                report["cases"].append(
                    {
                        "arm": arm,
                        "cached_tokens": cached,
                        "output_sha256": sha(ids),
                        "output_count": len(ids),
                    }
                )
            finally:
                try:
                    if armed:
                        await suite.rpc("qwen_target_cache_boundary", ["end"])
                finally:
                    await suite.rpc("qwen_prefill_capture_finish")
            await suite.absent(owner)
    first = max(report["cases"][1]["cached_tokens"], args.stage_at or 0)
    x, y = [prefill_rows(root / arm, first, prompt) for arm in ("cold", "warm")]
    a, b = x[1], y[1]
    counts = a.view(torch.int16).ne(b.view(torch.int16)).sum(1)
    difference = torch.where(counts > 0)[0]
    first_diff = int(difference[0]) if len(difference) else None
    for case, xs in zip(report["cases"], (x, y), strict=True):
        case["actual_prefill_intervals_half_open"] = xs[0]
    report.update(
        first_common_position=first,
        common_rows=len(a),
        exact_hidden_bit_rows=int((counts == 0).sum()),
        first_hidden_difference=first + first_diff if first_diff is not None else None,
        first_differing_elements=int(counts[first_diff])
        if first_diff is not None
        else None,
        per_64_rows=[
            {
                "start_position": first + i,
                "different_rows": int((counts[i : i + 64] > 0).sum()),
                "different_elements": int(counts[i : i + 64].sum()),
            }
            for i in range(0, len(counts), 64)
        ],
    )
    write(root / "result.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "binding",
                    "common_rows",
                    "first_common_position",
                    "exact_hidden_bit_rows",
                    "first_hidden_difference",
                    "first_differing_elements",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--abi", required=True)
    p.add_argument("--build-manifest", type=Path, default=Path("/retained/manifest.json"))
    p.add_argument("--stage-at", type=int)
    p.add_argument("--stage-window", type=int, default=8)
    p.add_argument("--state-snapshot", action="store_true")
    p.add_argument("--disable-indexed-880", action="store_true")
    p.add_argument("--base-url", default="http://127.0.0.1:8081")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--shared-control", type=Path, default=Path("/bench/shared"))
    asyncio.run(run(p.parse_args()))
