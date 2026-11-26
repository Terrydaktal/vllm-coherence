"""Run or independently verify the 17 short-graph and cache-boundary regression checks.

Run inside an isolated qualification server started with --worker-extension-cls
sampled_cache_probe.SampledCacheProbe, using its IPC and capture mounts.
The 60K shared controller output and its synthetic fixture must be mounted.
Verify needs only the JSON measurement reports, and uses no GPU or API.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WIDTHS = (256, 1024)
MODES = ("native", "forced")
ROWS = 64
FEATURES = {"aux_layer_6", "aux_layer_20", "aux_layer_34", "aux_layer_48", "aux_layer_62", "final_hidden"}
KV_LAYERS = {str(i) for i in range(3, 64, 4)}
REUSE = {"cold": 0, "warm": 57_680}
PROMPT = 60_208


def read(path):
    return json.loads(path.read_bytes())


def full_graph(row, width, mode):
    return (
        row["mode"] == mode and row["first"] is None
        and row["total_positions"] == ROWS + 1
        and [r["position"] for r in row["rows"]] == list(range(width - 1, width + ROWS))
        and row["exact_positions"] == dict.fromkeys(FEATURES, ROWS + 1)
        and any(b["route"] == "full" for b in row["actual_decode_batches"])
    )


def target_kv(row, width, mode):
    layers = row.get("target_kv", {})
    return (full_graph(row, width, mode) and layers.keys() == KV_LAYERS
            and all(v["positions"] == width + ROWS and v["different"] == 0
                    and v["first"] is None for v in layers.values()))


def replay_matches(observation, count):
    a, b = observation["control"], observation["observed"]
    return (observation["first_differing_output_index"] is None
            and a["tokens"] == b["tokens"] == count and a["sha256"] == b["sha256"])


def summarize(root):
    checks = {}
    for width in WIDTHS:
        row = read(root / f"short-{width}/decode/comparison.json")
        checks[f"exact_{width}"] = (
            (row["first_position"], row["last_position"]) == (width, width + ROWS - 1)
            and row["positions"] == row["hidden_exact_rows"] == row["logits"]["full_logits_exact"] == ROWS
        )
        for mode in MODES:
            graph = read(root / f"graph-{width}-{mode}/graph-output-comparison.json")
            kv = read(root / f"kv-{width}-{mode}/graph-output-comparison.json")
            checks[f"graph_{width}_{mode}"] = full_graph(graph, width, mode)
            checks[f"kv_{width}_{mode}"] = target_kv(kv, width, mode)

    target = read(root / "target/result.json")
    cases = {r["arm"]: r for r in target["cases"]}
    checks["cache_admitted"] = (
        len(target["cases"]) == 2 and {k: r["cached_tokens"] for k, r in cases.items()} == REUSE
        and cases["cold"]["actual_prefill_intervals_half_open"][-1:] == [[57_680, PROMPT]]
        and cases["warm"]["actual_prefill_intervals_half_open"] == [[57_680, 59_328], [59_328, PROMPT]]
    )
    checks["all_2528_target_rows"] = (target["prompt_tokens"] == PROMPT and target["first_hidden_difference"] is None
                                      and target["common_rows"] == target["exact_hidden_bit_rows"] == 2528)
    natural_run = read(root / "natural/qualification-run.json")
    if natural_run["phase"] != "sampled-replay" or natural_run["binding"] != target["binding"]:
        raise ValueError("target and sampled captures must identify the same build")
    natural = read(root / "natural/sampled-replay-results.json")
    by_temp = {c["sampling_temperature"]: c for c in natural["cases"]}
    for name, temperature in (("sampled", 1.0), ("greedy", 0.0)):
        case = by_temp.get(temperature, {})
        requests = case.get("requests", {})
        t = case.get("common_trajectory_targets", {})
        coverage = (len(natural["cases"]) == 2 and set(by_temp) == {0.0, 1.0}
                    and natural["prompt_sha256"] == target["prompt_sha256"]
                    and {arm: r["cached_tokens"] for arm, r in requests.items()} == REUSE)
        checks[f"{name}_128_ids"] = (
            coverage and case["first_emitted_difference_zero_based"] is None
            and all(r["output_tokens"] == 128 for r in requests.values())
            and requests["cold"]["output_sha256"] == requests["warm"]["output_sha256"]
        )
        checks[f"{name}_targets"] = (
            coverage and t["positions"] == t["hidden_bit_exact_rows"] == t["logits"]["full_logits_exact"] == 128
            and (t["first_position"], t["last_position"]) == (PROMPT - 1, PROMPT + 126)
        )

    lifecycle = read(root / "sampled-cancel.json")
    replay = lifecycle["sampled_replay_comparisons"]
    counts = {"cancelled_replay": 512, "waiter": 384}
    checks["sampled_cancel_and_waiter"] = (
        lifecycle["status"] == "PASS" and replay.keys() == counts.keys()
        and all(replay_matches(replay[key], count) for key, count in counts.items())
    )
    return checks


def run(args):
    import asyncio

    import httpx
    from qualify_combined_continuity import seed, write

    root = args.root.resolve()
    if not root.is_relative_to("/prefill-diagnosis/refresh"):
        raise ValueError("run output must be under the server's /prefill-diagnosis/refresh capture mount")
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    manifest = args.build_manifest.resolve()
    clients = {}

    def client(name, *options):
        source = Path(__file__).with_name(name + ".py")
        clients[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        subprocess.run([sys.executable, str(source), *map(str, options),
                        "--abi", args.abi, "--base-url", args.base_url], check=True)

    if args.references:
        fixture = args.references / "synthetic-fixture.json"
        if read(fixture).get("synthetic") is not True:
            raise ValueError("references must use a synthetic fixture")
    else:
        async def make_fixture():
            async with httpx.AsyncClient(base_url=args.base_url) as api:
                return await seed(api)
        fixture = root / "synthetic-fixture.json"
        write(fixture, {"synthetic": True, "prompt": asyncio.run(make_fixture())})

    for width in WIDTHS:
        short = root / f"short-{width}"
        short.mkdir(mode=0o700)
        corpus = short / "synthetic-corpus.json"
        if args.references:
            shutil.copyfile(args.references / f"corpus-{width}.json", corpus)
        client("diagnose_prefill_decode", "--output", short, "--prefix-rows", width, "--fixture", fixture,
               "--installed", "--rows", ROWS)
        for mode in MODES:
            for label, flags in (("graph", []), ("kv", ["--kv"])):
                client("diagnose_prefill_graph", "--output", root / f"{label}-{width}-{mode}", "--corpus", corpus,
                       "--rows", ROWS, "--mode", mode, *flags)

    common = ["--shared-control", args.shared_control, "--build-manifest", manifest]
    client("qualify_cache_target_alignment", "--root", root / "target", *common)
    client("qualify_combined_continuity", "sampled-replay", "--root", root / "natural", *common)
    client("qualify_speed_lifecycle", "--case-group", "sampled-replay", "--status-prefix", args.status_prefix,
           "--run-id", f"sampled-cancel-{time.time_ns()}", "--output", root / "sampled-cancel.json")
    checks = summarize(root)
    write(root / "result.json", {"abi": args.abi, "binding": read(manifest)["sha256"], "clients": clients,
                                 "checks": checks, "passed": all(checks.values())})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    collect = modes.add_parser("run", help="capture synthetic evidence and verify it")
    collect.add_argument("--root", type=Path, required=True, help="fresh directory on the capture mount")
    collect.add_argument("--references", type=Path, help="directory with frozen synthetic fixture and both corpora")
    collect.add_argument("--shared-control", type=Path, required=True, help="60K coding control and continuation archive")
    collect.add_argument("--build-manifest", type=Path, default=Path("/retained/manifest.json"))
    collect.add_argument("--abi", required=True)
    collect.add_argument("--base-url", default="http://127.0.0.1:8081")
    collect.add_argument("--status-prefix", type=Path, default=Path("/dev/shm/qwen-prefill-diagnosis"),
                         help="lifecycle scheduler telemetry prefix for the cancellation comparison")
    verify = modes.add_parser("verify", help="recompute gates from existing JSON reports offline")
    verify.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    checks = run(args) if args.mode == "run" else summarize(args.root)
    print(json.dumps({"checks": checks, "passed": all(checks.values())}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
