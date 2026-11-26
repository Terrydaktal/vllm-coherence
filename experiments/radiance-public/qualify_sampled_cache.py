"""Locate the first cold/warm difference in the synthetic shared coding fixture."""

import json
import time
import urllib.request
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace

from qualify_combined_continuity import compare_hidden_bits, sha, write
from qualify_speed_lifecycle import Suite


def first_difference(left, right):
    for i, (x, y) in enumerate(zip(left, right, strict=False)):
        if x != y:
            return i
    return None if len(left) == len(right) else min(len(left), len(right))


def select_accepted_rows(source, destination, prefix, outputs, count):
    """Select only rows whose entire speculative prefix matches committed history."""
    import torch

    first = len(prefix) - 1
    expected = prefix[-1:] + outputs[:count]
    chosen = {}
    for item in json.loads((source / "capture.json").read_text())["batches"]:
        batch = torch.load(source / item["file"], weights_only=True, map_location="cpu")
        for i, (p, token) in enumerate(
            zip(batch["positions"], batch["input_ids"], strict=True)
        ):
            offset = p - first
            if offset < 0:
                continue
            if offset >= len(expected) or token != expected[offset]:
                break
            chosen[p] = (token, batch["hidden"][i])
    positions = list(range(first, first + count + 1))
    if not set(positions).issubset(chosen):
        raise ValueError(
            "missing target hidden rows on the shared generated trajectory"
        )
    destination.mkdir(mode=0o700)
    torch.save(
        {
            "positions": positions,
            "input_ids": [chosen[p][0] for p in positions],
            "hidden": torch.stack([chosen[p][1] for p in positions]),
        },
        destination / "selected.pt",
    )
    write(destination / "capture.json", {"batches": [{"file": "selected.pt"}]})


def compare_logical_cache_bytes(root, kind, meta):
    import torch

    if "chunks_of_32" not in meta:
        return {}
    buffers = [
        torch.load(
            root / f"{kind}-{arm}-draft" / f"layer-{meta['layer']}.pt",
            weights_only=True,
            map_location="cpu",
        )
        for arm in ("cold", "warm")
    ]
    cold, warm = [t.reshape(t.shape[0], -1) for t in buffers]
    changed = cold.ne(warm)
    count = changed.sum(dim=1)
    indices = torch.where(count > 0)[0]
    cold_zero = cold.eq(0).all(dim=1)
    warm_zero = warm.eq(0).all(dim=1)
    return {
        "logical_slots": cold.shape[0],
        "bytes": cold.numel(),
        "differing_bytes": int(count.sum()),
        "differing_slots": len(indices),
        "first_different_position": meta["first"] + int(indices[0])
        if len(indices)
        else None,
        "last_different_position": meta["first"] + int(indices[-1])
        if len(indices)
        else None,
        "cold_zero_warm_nonzero_slots": int((cold_zero & ~warm_zero).sum()),
        "warm_zero_cold_nonzero_slots": int((warm_zero & ~cold_zero).sum()),
    }


async def sampled_replay(args, client):
    import benchmark_pi_coding_contexts as coding
    import benchmark_pi_coding_json_compaction as chain
    from tokenizers import Tokenizer

    root = args.root
    suite = Suite(
        SimpleNamespace(
            run_id=f"sampled-replay-{time.time_ns()}",
            abi=args.abi,
            observe=False,
            case_group="synthetic-cache-replay",
            status_prefix=Path("/dev/shm/qwen-prefill-diagnosis"),
        ),
        client,
    )
    shared = getattr(args, "shared_control", Path("/bench/shared"))
    section = json.loads((shared / "runs/60K/report.json").read_text())["contexts"][
        "60K"
    ]
    reference = section["arms"]["control_after"]
    archive = json.loads(
        (shared / "continuations" / (reference["capture_id"] + ".json")).read_text()
    )
    if sha(archive) != reference["output_token_ids_sha256"]:
        raise ValueError("saved synthetic continuation differs from benchmark digest")
    prefix, fixture_digest = coding._load_prefix(Path("/fixtures/60k.json"), 60000)
    tokenizer = Tokenizer.from_file(
        "/models/Qwen3.8-27B-Uncensored-MXFP4-awq/tokenizer.json"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    turn = chain._render_user_turn(
        opener, args.base_url, chain.CODING_PROMPT, thinking=False, timeout=60
    )
    prompt = prefix + chain._turn_suffix(tokenizer, prefix, turn, first=True)
    if (
        fixture_digest != section["fixture_sha256"]
        or coding._prefix_digest(prompt) != section["prompt_sha256"]
    ):
        raise ValueError("sampled replay prompt differs from shared suite")
    report = {
        "prompt_tokens": len(prompt),
        "prompt_sha256": section["prompt_sha256"],
        "cases": [],
    }
    for kind, temperature in (("sampled", 1.0), ("greedy", 0.0)):
        owner = suite.chat(kind)
        body = suite.body(owner, prompt, 128)
        body.update(
            temperature=temperature,
            top_k=40,
            top_p=0.95,
            seed=0,
            ignore_eos=False,
            add_special_tokens=False,
            stop=["<|im_end|>"],
            stream=True,
            stream_options={"include_usage": True, "continuous_usage_stats": True},
        )
        ids_by_arm, proposal_by_arm, case = (
            [],
            [],
            {"sampling_temperature": temperature, "requests": {}},
        )
        for arm in ("cold", "warm"):
            trace = root / (kind + "-" + arm)
            draft = root / (kind + "-" + arm + "-draft")
            await suite.rpc("qwen_prefill_capture_begin", [str(trace), len(prompt) - 1])
            try:
                await suite.rpc("qwen_sampled_cache_begin", [str(draft), len(prompt)])
                try:
                    ids, usage, finish, done = [], None, None, False
                    async with client.stream("POST", "/v1/completions", json=body) as r:
                        r.raise_for_status()
                        async for line in r.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            value = line[5:].strip()
                            if value == "[DONE]":
                                done = True
                                continue
                            event = json.loads(value)
                            if "error" in event:
                                raise RuntimeError(
                                    "inference failed during sampled replay"
                                )
                            usage = event.get("usage") or usage
                            for choice in event.get("choices", []):
                                ids.extend(choice.get("token_ids") or [])
                                finish = choice.get("finish_reason") or finish
                    if (
                        not usage
                        or not done
                        or finish not in ("stop", "length")
                        or len(ids) != usage["completion_tokens"]
                    ):
                        raise ValueError("incomplete samples in synthetic replay")
                    cached = usage["prompt_tokens_details"]["cached_tokens"]
                    if usage["prompt_tokens"] != len(prompt) or (arm == "warm") != (
                        cached > 0
                    ):
                        raise ValueError(
                            "cold/warm replay did not exercise expected prefix-cache path"
                        )
                    write(trace / "natural-token-ids.json", ids)
                    ids_by_arm.append(ids)
                    case["requests"][arm] = {
                        "output_tokens": len(ids),
                        "output_sha256": sha(ids),
                        "cached_tokens": cached,
                        "first_difference_from_archived_warm": first_difference(
                            ids, archive[: len(ids)]
                        )
                        if kind == "sampled"
                        else None,
                    }
                finally:
                    await suite.rpc("qwen_sampled_cache_finish")
            finally:
                await suite.rpc("qwen_prefill_capture_finish")
            proposal_by_arm.append(json.loads((draft / "proposals.json").read_text()))
            if not proposal_by_arm[-1]:
                raise RuntimeError("no speculative proposals captured")
            await suite.absent(owner)
        first = first_difference(*ids_by_arm)
        case["first_emitted_difference_zero_based"] = first
        # Final generated token may have been emitted but not forwarded yet.
        count = min(len(x) - 1 for x in ids_by_arm)
        if first is not None:
            count = min(count, first)
        selected = []
        for arm, ids in zip(("cold", "warm"), ids_by_arm, strict=True):
            dest = root / (kind + "-" + arm + "-accepted")
            select_accepted_rows(root / (kind + "-" + arm), dest, prompt, ids, count)
            selected.append(dest)
        compare = await suite.rpc("qwen_prefill_compare", [str(s) for s in selected])
        compare.update(compare_hidden_bits(*selected, len(prompt) - 1, count + 1))
        case["common_trajectory_targets"] = compare
        first_cold, first_warm = [arm[0] for arm in proposal_by_arm]
        case["first_draft_round"] = {
            key: {
                "observed": first_cold[key] is not None and first_warm[key] is not None,
                "equal": first_cold[key] == first_warm[key]
                if first_cold[key] is not None and first_warm[key] is not None
                else None,
                "first_different_index": first_difference(
                    first_cold[key], first_warm[key]
                )
                if isinstance(first_cold[key], list)
                and isinstance(first_warm[key], list)
                else None,
            }
            for key in first_cold
            if key != "logical_draft_cache"
        }
        cache_a, cache_b = (
            item["logical_draft_cache"] for item in (first_cold, first_warm)
        )
        case["first_draft_round"]["logical_context_kv"] = [
            {
                "layer": a["layer"],
                "observed_layout": "chunks_of_32" in a and "chunks_of_32" in b,
                "first_different_chunk": first_difference(
                    a.get("chunks_of_32", []), b.get("chunks_of_32", [])
                ),
                "differing_chunks": sum(
                    x != y
                    for x, y in zip_longest(
                        a.get("chunks_of_32", []), b.get("chunks_of_32", [])
                    )
                ),
                **(
                    compare_logical_cache_bytes(root, kind, a)
                    if "chunks_of_32" in b
                    else {}
                ),
                "cold_null_slots": a.get("null_pages"),
                "warm_null_slots": b.get("null_pages"),
            }
            for a, b in zip(cache_a, cache_b, strict=True)
        ]
        report["cases"].append(case)
        write(root / "sampled-replay-results.json", report)
    return report
