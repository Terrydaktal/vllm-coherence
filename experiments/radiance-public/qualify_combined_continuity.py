"""Synthetic full hidden/logit and cached greedy continuation qualification."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from qualify_speed_lifecycle import MODEL, Suite

try:
    import radiance_cache
except ImportError:
    from qwen_r9700_lab import radiance_cache

    sys.modules["radiance_cache"] = radiance_cache


SNAPSHOT_ROOT = Path("/cache/snapshots")


def sha(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def compare_hidden_bits(left, right, first, count):
    import torch

    positions = range(first, first + count)

    def hashes(root):
        records = json.loads((root / "capture.json").read_text())
        found = {}
        for item in records["batches"]:
            payload = torch.load(root / item["file"], weights_only=True, map_location="cpu")
            for i, position in enumerate(payload["positions"]):
                if position not in positions:
                    continue
                if position in found:
                    raise ValueError("multiple hidden rows at a selected position")
                row = payload["hidden"][i].contiguous()
                digest = hashlib.sha256(row.view(torch.uint8).numpy().tobytes()).digest()
                found[position] = (str(row.dtype), tuple(row.shape), digest)
        if len(found) != count:
            raise ValueError("missing hidden rows in selected range")
        return found

    a, b = hashes(left), hashes(right)
    different = [p for p in positions if a[p] != b[p]]
    mismatch = None
    if different:
        p = different[0]
        mismatch = {
            "position": p,
            "left_sha256": a[p][-1].hex(),
            "right_sha256": b[p][-1].hex(),
        }
    return {
        "hidden_bit_exact_rows": count - len(different),
        "first_hidden_bit_mismatch": mismatch,
    }


def reused_tokens(response):
    return (response["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)


async def check_scheduler(suite):
    await suite.wait(lambda: "requests" in suite.status(), "scheduler sample missing")


async def seed(api):
    text = "Write Python unit tests for numeric comparison of cached states.\n" + "".join(
        f"# case {i}: compare saved rows with the fresh reference and check rejected prefixes.\n"
        for i in range(160)
    )
    result = await api.post("/tokenize", json={"model": MODEL, "prompt": text})
    result.raise_for_status()
    tokens = result.json()["tokens"]
    if len(tokens) < 1651:
        raise RuntimeError("synthetic seed too short")
    return tokens


async def vectors(args, tokens):
    from diagnose_prefill_decode import run

    fixture = args.root / "synthetic-fixture.json"
    write(fixture, {"synthetic": True, "prompt": tokens})
    results = []
    # The first grid checks long forced walks after the boundary; the second
    # checks the final eight vectors of a single cold prompt *at* each width.
    cases = [("prefix", length, args.rows) for length in args.contexts]
    edges = (127, 128, 129, 255, 256, 1023, 1024, 1025, 1647, 1648, 1649, 3296)
    cases += [("cold-width", width - 8, 8) for width in edges]
    for group, length, rows in cases:
        out = args.root / f"vectors-{group}-{length}"
        cfg = SimpleNamespace(
            output=out,
            fixture=fixture,
            prefix_rows=length,
            installed=True,
            rows=rows,
            base_url=args.base_url,
            abi=args.abi,
            first=None,
            stages=False,
            gdn=False,
            attention=False,
        )
        await run(cfg)
        result = json.loads((out / "decode/comparison.json").read_text())
        result.update(compare_hidden_bits(out / "prefill", out / "decode", length, rows))
        ok = (
            result["hidden_bit_exact_rows"] == rows
            and result["positions"] == rows
            and result["first_position"] == length
            and result["last_position"] == length + rows - 1
            and result["hidden_exact_rows"] == rows
            and result["logits"]["full_logits_exact"] == rows
        )
        results.append(
            {
                "case": group,
                "context_tokens": length,
                "cold_prompt_rows": length + rows,
                "pass": ok,
                **result,
            }
        )
        write(args.root / "vector-results.json", results)
    failing = [(r["case"], r["context_tokens"]) for r in results if not r["pass"]]
    if failing:
        raise RuntimeError(f"prefill versus forced decode hidden/logit discrepancies: {failing}")
    return results


async def sealed(args, client):
    """Compare actual installed cold prefill to separately captured full M8 rows."""
    cfg = SimpleNamespace(
        run_id=f"direct-sealed-{time.time_ns()}",
        abi=args.abi,
        observe=False,
        case_group="current-prefill-versus-archive",
        status_prefix=Path("/dev/shm/qwen-prefill-diagnosis"),
    )
    suite = Suite(cfg, client)
    results = []

    async def comparison(reference, candidate, first, count):
        measured = await suite.rpc("qwen_prefill_compare", [str(reference), str(candidate)])
        measured.update(compare_hidden_bits(reference, candidate, first, count))
        return {
            "pass": (
                measured["hidden_bit_exact_rows"] == count
                and measured["positions"] == count
                and measured["hidden_exact_rows"] == count
                and measured["logits"]["full_logits_exact"] == count
                and measured["first_position"] == first
                and measured["last_position"] == first + count - 1
            ),
            **measured,
        }

    for ref in args.references:
        path = ref / "synthetic-corpus.json"
        corpus = json.loads(path.read_text())
        if not corpus.get("synthetic") or not corpus["prefix"] or len(corpus["output"]) <= 1:
            raise ValueError("independent comparison requires a frozen synthetic corpus")
        length = len(corpus["prefix"])
        rows = len(corpus["output"]) - 1
        dest = args.root / f"new-{length}"
        await suite.rpc("qwen_prefill_capture_begin", [str(dest), length])
        try:
            job = suite.stream(f"new-{length}", corpus["prefix"] + corpus["output"][:-1], 1)
            await asyncio.wait_for(asyncio.shield(job.task), 1200)
            result = await job.result()
            await suite.absent(job.chat)
        finally:
            await suite.rpc("qwen_prefill_capture_finish")
        cached = reused_tokens(result)
        if cached:
            raise ValueError("sealed cold prefill unexpectedly reused a prefix")
        case = {
            "context_tokens": length,
            "cached_tokens": cached,
            "synthetic_fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **await comparison(ref / "decode", dest, length, rows),
        }
        results.append(case)
        write(args.root / "sealed-results.json", results)
        if not case["pass"] and args.replay_changed_references:
            decode = args.root / f"current-m8-{length}"
            await suite.rpc("qwen_prefill_capture_begin", [str(decode), length, str(path)])
            try:
                job = suite.stream(f"forced-{length}", corpus["prefix"], rows + 1)
                await asyncio.wait_for(asyncio.shield(job.task), 1200)
                response = await job.result()
                await suite.absent(job.chat)
            finally:
                capture = await suite.rpc("qwen_prefill_capture_finish")
            if (
                reused_tokens(response)
                or job.ids != corpus["output"]
                or capture["forced_complete"] is not True
            ):
                raise ValueError("current M8 did not complete the exact frozen trajectory")
            case["current_prefill_vs_current_m8"] = await comparison(decode, dest, length, rows)
            case["current_m8_vs_archived_m8"] = await comparison(
                ref / "decode", decode, length, rows
            )
            write(args.root / "sealed-results.json", results)
    changed = [case["context_tokens"] for case in results if not case["pass"]]
    if changed:
        raise RuntimeError(f"current prefill differs from archived M8 at contexts {changed}")


async def cache(args, tokens, client):
    results = []
    suite = Suite(
        SimpleNamespace(
            run_id=f"direct-cache-{time.time_ns()}",
            observe=False,
            case_group="prefill-cache",
            abi=args.abi,
            status_prefix=Path(args.status_prefix),
        ),
        client,
    )
    await check_scheduler(suite)
    # A genuine tool suffix changes the continuation; its counterfactual is a
    # separate never-before-used identity that cold-prefills the identical text.
    suffix = (
        await suite.post(
            "/tokenize",
            {
                "model": MODEL,
                "prompt": (
                    "\nTool output: 7 tests passed, 1 failed. Correct the index and try again.\n"
                ),
            },
        )
    )["tokens"]
    for length in args.contexts:
        prompt = (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]
        for handover in (False, True):
            label = f"{length}-{handover}"
            owner = suite.chat("producer-" + label)
            first = suite.stream("produce", prompt, 11, owner)
            await first.result()
            await suite.absent(owner)
            next_prompt = prompt + first.ids + suffix
            if handover:
                await suite.generate("other-" + label, tokens[:256], 8)
            # Run the warm successor before the independent cold reference; this
            # keeps the two cases distinct (same active bank vs parked owner).
            warm = suite.stream("resume", next_prompt, args.rows, owner)
            warm_info = await warm.result()
            await suite.absent(owner)
            cold = suite.stream("fresh-" + label, next_prompt, args.rows)
            cold_info = await cold.result()
            await suite.absent(cold.chat)

            observed = reused_tokens(warm_info)
            fresh = reused_tokens(cold_info)
            ok = warm.ids == cold.ids and observed >= length + len(first.ids) - 1 and fresh == 0
            results.append(
                {
                    "context_tokens": length,
                    "handover": handover,
                    "pass": ok,
                    "successor_tokens": args.rows,
                    "cached_tokens": observed,
                    "cold_control_cached": fresh,
                    "warm_hash": sha(warm.ids),
                    "cold_hash": sha(cold.ids),
                    "first_difference": next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(warm.ids, cold.ids, strict=False))
                            if a != b
                        ),
                        None,
                    ),
                }
            )
            write(args.root / "cache-results.json", results)
    failing = [(r["context_tokens"], r["handover"]) for r in results if not r["pass"]]
    if failing:
        raise RuntimeError(f"cache continuation or reuse differed in {failing}")
    return results


def instance_id():
    # PID namespace init process identifies the actual server container.
    return Path("/proc/1/stat").read_text().rsplit(")", 1)[1].split()[19]


def prepare_checkpoint_encoding(owner, abi, run_id, mode):
    if mode not in ("raw", "compressed", "damage"):
        raise ValueError("unsupported snapshot qualification encoding")
    if not radiance_cache.ID.fullmatch(abi) or not run_id.startswith("direct-disk-"):
        raise ValueError("unexpected snapshot ABI or fixture identity")
    label = owner["title"].removeprefix("Synthetic lifecycle ")
    if owner["id"] != sha([run_id, label]):
        raise ValueError("snapshot control does not belong to this exact preparation")

    if owner["cwd"] != "/qualification" or not owner["title"].startswith(
        f"Synthetic lifecycle durable-{mode}-"
    ):
        raise ValueError("encoding tests require their own synthetic fixture")
    root = SNAPSHOT_ROOT / abi / "data"
    store = radiance_cache.ChatStore(root, owner)
    with store.lock(exclusive=True):
        metadata = store.metadata()
        if any(metadata.get(k) != value for k, value in radiance_cache.identity(owner).items()):
            raise ValueError("published cache identity differs from the prepared synthetic fixture")
        if metadata["publication"]["result"] != "committed" or not metadata["head"]:
            raise ValueError("encoding control needs a published head")
        first_group = [key for key in metadata["head"] if key.startswith("g0-")]
        if len(first_group) != 1:
            raise ValueError("encoding fixture must contain exactly one attention block")
        (key,) = first_group
        path = store.path(key)
        original = path.read_bytes()
        size = metadata["verified_block_size"]
        raw = radiance_cache.decode_block(original, size)
        digest = hashlib.sha256(raw).digest()
        if mode == "damage":
            updated = original[:-1] + bytes([original[-1] ^ 1])
        elif mode == "raw":
            updated = radiance_cache.HEADER.pack(radiance_cache.RAW, size, digest) + raw
        else:
            import zstandard

            frame = zstandard.ZstdCompressor(level=1, write_checksum=True).compress(raw)
            if len(frame) >= size:
                raise ValueError("fixture would exceed the compressed-object size limit")
            updated = radiance_cache.HEADER.pack(radiance_cache.COMPRESSED, size, digest) + frame
        if mode != "damage" and radiance_cache.decode_block(updated, size) != raw:
            raise ValueError("forced storage encoding changed the checkpoint")
        # Replacing this fixture's link preserves other immutable objects even
        # if the underlying inode happens to be shared between generations.
        radiance_cache.atomic_write(path, updated)
    if mode != "damage" and not store.publish(
        metadata["head"],
        metadata["tokens"],
        size,
        response_end=metadata["response_end"],
        immutable_keys=metadata.get("immutable_head"),
    ):
        raise ValueError("failed to verify and publish the alternate encoding")
    return {
        "mode": mode,
        "payload_bytes": size,
        "before_bytes": len(original),
        "after_bytes": len(updated),
        "encoded_sha256": hashlib.sha256(updated).hexdigest(),
        "original_payload_sha256": digest.hex(),
    }


async def disk(args, tokens, client, restore):
    fixture = args.root / "disk-fixture.json"
    cfg = SimpleNamespace(
        run_id=f"direct-disk-{time.time_ns()}",
        observe=False,
        case_group="disk",
        abi=args.abi,
        status_prefix=Path(args.status_prefix),
    )
    suite = Suite(cfg, client)
    await check_scheduler(suite)
    body = suite.body

    def force_flush(owner, prompt, count):
        request = body(owner, prompt, count)
        request["kv_transfer_params"]["qwen_snapshot_force_flush"] = True
        return request

    results = []
    if restore:
        saved = json.loads(fixture.read_text())
        if not saved.get("synthetic") or saved["abi"] != args.abi:
            raise ValueError("unexpected durable replay fixture")
        if saved["container_start_ticks"] == instance_id():
            raise ValueError("disk replay requires a new server container")
        for case in saved["cases"]:
            mode = case["mode"]
            negative = mode == "damage"
            changed = (
                prepare_checkpoint_encoding(case["owner"], args.abi, saved["run_id"], mode)
                if mode != "natural"
                else None
            )
            warm = suite.stream("disk", case["prompt"], case["length"], case["owner"])
            try:
                response = await warm.result()
            except Exception as error:
                results.append(
                    {
                        "context": len(case["prompt"]),
                        "storage_case": mode,
                        "encoding": changed,
                        "first_attempt_error": type(error).__name__,
                        "emitted_tokens_before_error": len(warm.ids),
                        "pass": False,
                    }
                )
                write(args.root / "disk-results.json", results)
                raise
            await suite.absent(warm.chat)
            reused = reused_tokens(response)
            ok = sha(warm.ids) == case["expected"] and (
                reused == 0 if negative else reused >= case["minimum"]
            )
            results.append(
                {
                    "context": len(case["prompt"]),
                    "ids_exact": sha(warm.ids) == case["expected"],
                    "cached_tokens": reused,
                    "expected_min": 0 if negative else case["minimum"],
                    "storage_case": mode,
                    "encoding": changed,
                    "output_tokens": len(warm.ids),
                    "pass": ok,
                }
            )
            write(args.root / "disk-results.json", results)
        failing = [r["storage_case"] for r in results if not r["pass"]]
        if failing:
            raise RuntimeError(f"disk replay or checkpoint admission differed for {failing}")
    else:
        suite.body = force_flush
        cases = [(length, "natural") for length in args.contexts]
        cases += [(min(128, args.contexts[0]), mode) for mode in ("raw", "compressed", "damage")]
        for length, mode in cases:
            prefix = (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]
            label = f"durable-{mode}-{length}"
            owner = suite.chat(label)
            producer = suite.stream("producer", prefix, 19, owner)
            await producer.result()
            await suite.absent(owner)
            prompt = prefix + producer.ids + tokens[-17:]
            cold = suite.stream("cold-" + label, prompt, args.rows)
            cold_result = await cold.result()
            await suite.absent(cold.chat)
            if reused_tokens(cold_result) != 0:
                raise ValueError("disk replay's independent cold control reused a prefix")
            results.append(
                {
                    "owner": owner,
                    "mode": mode,
                    "prompt": prompt,
                    "minimum": length + len(producer.ids) - 1,
                    "expected": sha(cold.ids),
                    "length": len(cold.ids),
                }
            )
        write(
            fixture,
            {
                "synthetic": True,
                "container_start_ticks": instance_id(),
                "run_id": cfg.run_id,
                "abi": args.abi,
                "cases": results,
            },
        )


def prepare_run_directory(args):
    binding = json.loads(args.build_manifest.read_text())["sha256"]
    schema = "urn:coherence:combined-continuity:v1"
    marker = args.root / "qualification-run.json"
    if args.phase == "disk-restore":
        previous = json.loads(marker.read_text())
        if (previous["schema"], previous["phase"], previous["binding"]) != (
            schema,
            "disk-prepare",
            binding,
        ):
            raise ValueError("disk control was prepared on a different build")
        return
    if args.root.exists() and any(args.root.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()).get("schema") != schema:
            raise ValueError("qualification output already contains unrelated files")
        args.root.rename(args.root.with_name(f"{args.root.name}.previous-{time.time_ns()}"))
    args.root.mkdir(parents=True, exist_ok=True)
    write(
        marker,
        {
            "schema": schema,
            "phase": args.phase,
            "binding": binding,
            "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "started_at": time.time(),
            "request_abi": args.abi,
        },
    )


async def main(args):
    os.umask(0o077)
    prepare_run_directory(args)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=1200) as client:
        tokens = await seed(client)
        if args.phase == "sampled-replay":
            from qualify_sampled_cache import sampled_replay

            await sampled_replay(args, client)
        elif args.phase == "sealed":
            await sealed(args, client)
        elif args.phase == "vectors":
            await vectors(args, tokens)
        elif args.phase == "cache":
            await cache(args, tokens, client)
        else:
            await disk(args, tokens, client, args.phase == "disk-restore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "sampled-replay",
            "vectors",
            "sealed",
            "cache",
            "disk-prepare",
            "disk-restore",
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--abi", required=True)
    parser.add_argument("--build-manifest", type=Path, default=Path("/retained/manifest.json"))
    parser.add_argument(
        "--references",
        type=Path,
        nargs="+",
        default=[
            Path("/prefill-diagnosis/release-short"),
            Path("/prefill-diagnosis/release-60k-final"),
            Path("/prefill-diagnosis/release-200k-final"),
        ],
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=(256, 1648))
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument(
        "--replay-changed-references",
        action="store_true",
        help="Also compare current forced M8 on each synthetic trajectory where sealed M8 differs",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shared-control", type=Path, default=Path("/bench/shared"))
    parser.add_argument("--status-prefix", default="/dev/shm/qwen-stage-timing")
    asyncio.run(main(parser.parse_args()))
