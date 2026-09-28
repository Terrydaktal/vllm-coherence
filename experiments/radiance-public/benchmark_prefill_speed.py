"""Paired cold-prefill timing on one frozen synthetic history.

Run inside the isolated diagnostic server container. Normal compilation,
arithmetic, offloading and scheduling remain active; no stage profiler runs.
One generated token measures first-data latency, not decode throughput.
"""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from qualify_speed_lifecycle import Suite

from qwen_r9700_lab.diagnostic_contract import seal, write_private


async def run(args):
    if args.output.exists():
        raise ValueError("refusing to replace an existing result")
    original = json.loads(args.fixture.read_text())
    if original.get("synthetic") is not True:
        raise ValueError("an explicitly synthetic fixture is required")
    seed = original.get("prefix", original.get("prompt"))
    if not seed:
        raise ValueError("empty synthetic prefix")
    cfg = SimpleNamespace(
        base_url=args.base_url,
        abi=args.abi,
        run_id=f"prefill-speed-{time.time_ns()}",
        output=args.output,
        status_prefix=Path("/dev/shm/qwen-prefill-diagnosis"),
        observe=False,
        case_group="prefill-speed",
    )
    report = {
        "schema": "urn:coherence:paired-prefill-speed:v1",
        "status": "RUNNING",
        "scope": "Unprofiled cold synthetic prefill plus one first token; same process, fresh chat generations",
        "cases": [],
        "candidate_only": args.candidate_only,
        "baseline_speed": getattr(args, "baseline_speed", False),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    async with httpx.AsyncClient(base_url=args.base_url, timeout=1200) as client:
        suite = Suite(cfg, client)
        suite.sampling = {"temperature": 0, "top_k": 129, "seed": 0, "logprobs": 1}
        await suite.rpc("qwen_prefill_probe_reload")
        report["sources"] = await suite.rpc(
            "qwen_prefill_load_candidate_sources", [str(args.sources)]
        )
        variants = {
            "baseline": (args.baseline_attention, args.baseline_projection),
            "candidate": (args.attention, args.projection),
        }
        warm = suite.stream(
            "claim-bank", seed[:256], 1, suite.chat("paired-prefill", "warmup")
        )
        await warm.result()
        await suite.absent(warm.chat)
        for context in args.contexts:
            prompt = (seed * ((context + len(seed) - 1) // len(seed)))[:context]
            # Counterbalance clock/temperature/order effects. Each new generation
            # supersedes the last bank, avoiding unrelated-chat RAM handovers.
            for repeat in range(args.repeats):
                order = (
                    ["candidate"] if args.candidate_only else ["baseline", "candidate"]
                )
                if args.legacy:
                    order.insert(0, "legacy")
                for name in order if repeat % 2 == 0 else reversed(order):
                    if name == "legacy":
                        installed = await suite.rpc("qwen_prefill_legacy_timing")
                    else:
                        # The frozen server may itself contain the speed hooks.
                        # Remove those too: otherwise a "baseline" can retain
                        # the candidate scan/convolution below its new wrapper.
                        await suite.rpc("qwen_prefill_legacy_timing")
                        attention, projection = variants[name]
                        installed = await suite.rpc(
                            "qwen_prefill_install_runtime",
                            [
                                str(attention),
                                str(projection),
                                (name == "candidate" and args.speed)
                                or (
                                    name == "baseline"
                                    and getattr(args, "baseline_speed", False)
                                ),
                            ],
                        )
                    chat = suite.chat("paired-prefill", f"{context}/{repeat}/{name}")
                    job = suite.stream(f"{context}-{repeat}-{name}", prompt, 1, chat)
                    await asyncio.wait_for(asyncio.shield(job.task), 1200)
                    result = await job.result()
                    await suite.absent(chat)
                    cached = (result["usage"].get("prompt_tokens_details") or {}).get(
                        "cached_tokens", 0
                    )
                    if cached != 0:
                        raise AssertionError("cold benchmark reused prompt tokens")
                    row = {
                        "context": context,
                        "repeat": repeat,
                        "variant": name,
                        "prompt_sha256": hashlib.sha256(
                            json.dumps(prompt, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "first_data_seconds": job.first - job.started,
                        "tokens_per_second": context / (job.first - job.started),
                        "installed": installed,
                        **result,
                    }
                    phase_path = cfg.status_prefix.with_name(
                        cfg.status_prefix.name + "-phases.json"
                    )
                    if phase_path.exists():
                        phases = json.loads(phase_path.read_text())
                        phase = next(
                            (
                                p
                                for p in reversed(phases.get("recent", []))
                                if p.get("chat_id") == chat["id"]
                                and p.get("generation") == chat["generation"]
                            ),
                            None,
                        )
                        if phase is not None:
                            row["backend_timings_ms"] = phase.get("timings_ms", {})
                    report["cases"].append(row)
                    checkpoint = args.output.with_name(
                        args.output.stem + f".step-{len(report['cases']):03d}.json"
                    )
                    write_private(checkpoint, seal(report))
                    print(json.dumps(row), flush=True)
            outcomes = {
                r["sha256"]
                for r in report["cases"]
                if r["context"] == context and r["variant"] != "legacy"
            }
            if len(outcomes) != 1:
                raise AssertionError(
                    "first prediction differs; full-vector qualification also required"
                )
    report["status"] = "MEASURED"
    write_private(args.output, seal(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--abi", required=True)
    for name in (
        "fixture",
        "sources",
        "baseline-attention",
        "baseline-projection",
        "attention",
        "projection",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[60000, 200000])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Measure only the candidate, for an isolated scheduler-capacity experiment",
    )
    parser.add_argument(
        "--speed",
        action="store_true",
        help="Also test prepared GDN and expanded input tiling",
    )
    parser.add_argument(
        "--baseline-speed",
        action="store_true",
        help="Retain prepared GDN, input tiling and dynamic convolution in the baseline too",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Include uncorrected pre-alignment attention/projection as an isolated timing control",
    )
    asyncio.run(run(parser.parse_args()))
