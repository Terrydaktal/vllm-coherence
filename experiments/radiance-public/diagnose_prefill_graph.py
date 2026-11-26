"""Compare native or forced target rows with complete cold prefill, observing existing graphs."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace


def compare(root):
    import torch

    def read(name):
        directory = root / name
        receipt = json.loads((directory / "observer.json").read_bytes())
        rows = {}
        for batch in receipt["batches"]:
            raw = torch.load(
                directory / batch["file"], weights_only=True, map_location="cpu"
            )
            for i, position in enumerate(raw["positions"]):
                if position in rows:
                    raise ValueError("duplicate committed position")
                rows[position] = (raw["input_ids"][i], [f[i] for f in raw["features"]])
        return receipt, rows

    before, left = read("full-prefill")
    after, right = read("decode-outputs")
    if before["layers"] != after["layers"] or set(left) != set(right) or not left:
        raise ValueError("incomplete paired auxiliary records")
    if not any(b["route"] == "full" for b in after["batches"]):
        raise ValueError("no real FULL decode graph was observed")
    boundaries = [f"aux_layer_{layer}" for layer in before["layers"]] + ["final_hidden"]
    first = None
    counts = {name: 0 for name in boundaries}
    changed = []
    for p in sorted(left):
        l_id, a = left[p]
        r_id, b = right[p]
        if int(l_id) != int(r_id):
            raise ValueError(f"committed input differs at position {p}")
        entry = {}
        for label, x, y in zip(boundaries, a, b, strict=True):
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError("mismatched observable boundary shape")
            different = int(
                (x.view(torch.uint16) != y.view(torch.uint16)).count_nonzero()
            )
            entry[label] = different
            counts[label] += int(different == 0)
            if different and first is None:
                first = {
                    "position": p,
                    "boundary": label,
                    "differing_elements": different,
                }
        changed.append({"position": p, "differing_elements": entry})
    summary = {
        "mode": after["mode"],
        "first": first,
        "total_positions": len(left),
        "exact_positions": counts,
        "first_observed_position": min(left),
        "rows": changed,
        "actual_decode_batches": after["batches"],
    }
    kv_paths = [
        root / directory / "target-kv.json"
        for directory in ("full-prefill", "decode-outputs")
    ]
    kv_status = [before.get("target_kv_status"), after.get("target_kv_status")]
    if any(s and s.get("status") != "complete" for s in kv_status):
        summary["target_kv_unavailable"] = kv_status
    elif all(path.exists() for path in kv_paths):
        a, b = (json.loads(path.read_bytes()) for path in kv_paths)
        if a.keys() != b.keys():
            raise ValueError("different forward target KV layer coverage")
        kv = {}
        for layer, stored in a.items():
            if stored.keys() != b[layer].keys():
                raise ValueError("different target KV position coverage")
            differences = sorted(
                int(p) for p, digest in stored.items() if b[layer][p] != digest
            )
            kv[layer] = {
                "positions": len(stored),
                "different": len(differences),
                "first": differences[0] if differences else None,
                "different_in_prior_prompt": sum(p < min(left) for p in differences),
            }
        summary["target_kv"] = kv
    (root / "graph-output-comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("mode", "first", "total_positions", "exact_positions")
            }
        ),
        flush=True,
    )


async def run(args):
    import httpx
    from qualify_speed_lifecycle import Suite

    original = json.loads(args.corpus.read_bytes())
    if original.get("synthetic") is not True or not 0 < args.rows < 128:
        raise ValueError("only bounded synthetic fixtures admitted")
    prefix = original["prefix"]
    expected = original["output"][: args.rows + 1]
    if len(expected) != args.rows + 1:
        raise ValueError("fixture lacks continuation rows")
    root = args.output
    root.mkdir(mode=0o700)
    cfg = SimpleNamespace(
        base_url=args.base_url,
        abi=args.abi,
        run_id=f"graph-output-{time.time_ns()}",
        output=root / "suite.json",
        status_prefix=Path("/dev/shm/qwen-prefill-diagnosis"),
        observe=False,
        case_group="graph-output-observer",
    )
    async with httpx.AsyncClient(base_url=args.base_url, timeout=1200) as client:
        suite = Suite(cfg, client)
        suite.sampling = {"temperature": 0, "top_k": 129, "seed": 0}
        first, last = len(prefix) - 1, len(prefix) + args.rows

        async def request(label, tokens, count, mode, output):
            probe = ["begin", str(output), first, last, mode]
            if args.kv:
                probe.append(True)
            await suite.rpc("qwen_prefill_graph_observer", probe)
            try:
                job = suite.stream(label, tokens, count)
                await asyncio.wait_for(asyncio.shield(job.task), 1200)
                result = await job.result()
                await suite.absent(job.chat)
                if len(job.ids) != count:
                    raise ValueError(
                        f"{label} emitted {len(job.ids)}, expected {count}"
                    )
                print(json.dumps({"phase": label, **result}), flush=True)
                return list(job.ids)
            finally:
                await suite.rpc("qwen_prefill_graph_observer", ["finish"])

        if args.mode == "forced":
            await suite.rpc(
                "qwen_prefill_capture_begin",
                [str(root / "forced-raw"), first, str(args.corpus)],
            )
        try:
            emitted = await request(
                "decode", prefix, len(expected), args.mode, root / "decode-outputs"
            )
        finally:
            if args.mode == "forced":
                await suite.rpc("qwen_prefill_capture_finish")
        if args.mode == "forced" and emitted != expected:
            raise ValueError("forced inputs differ from frozen fixture")
        await request(
            "cold", prefix + emitted[:-1], 1, "prefill", root / "full-prefill"
        )
    compare(root)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--abi", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--mode", choices=("native", "forced"), required=True)
    parser.add_argument(
        "--kv",
        action="store_true",
        help="Compare numeric target KV fingerprints at all attention layers through position 2,047",
    )
    asyncio.run(run(parser.parse_args()))
