"""Isolate cache-copy preservation from prefill/decode numerical differences.

Requires response_end_probe.ResponseEndProbe in an isolated DEV_MODE server.
The input must be the synthetic failure fixture from qualify_response_end.py;
never substitute a private chat. No output text or token IDs enter the report.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from qualify_speed_lifecycle import Suite, digest


async def run(args):
    fixture = json.loads(args.fixture.read_text())
    report = {
        "schema": "urn:coherence:response-end-state-check:v1",
        "status": "RUNNING",
        "cases": [],
        "scope": "Byte preservation; legacy-rule controls are not a numerical reference proof",
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:
        suite = Suite(args, client)
        report["begin"] = await suite.rpc("qwen_response_probe_begin")
        save()
        try:
            for mode in ("exact", "aligned", "legacy"):
                await suite.rpc("qwen_response_probe_mode", [mode])
                for context in (1651, 3301):
                    initial = (fixture["prompt"][:1651] * 3)[:context]
                    chat = suite.chat(f"{mode}-{context}")
                    producer = suite.stream("producer", initial, 19, chat)
                    await producer.result()
                    await suite.absent(chat)
                    prompt = initial + producer.ids + fixture["prompt"][1670:]
                    fresh = suite.stream(f"{mode}-{context}-cold", prompt, 64)
                    await fresh.result()
                    await suite.absent(fresh.chat)
                    resume = suite.stream("resume", prompt, 64, chat)
                    observed = await resume.result()
                    await suite.absent(chat)
                    report["cases"].append(
                        {
                            "mode": mode,
                            "context": context,
                            "equal_to_fresh": resume.ids == fresh.ids,
                            "first_difference": next(
                                (
                                    i
                                    for i, (a, b) in enumerate(
                                        zip(resume.ids, fresh.ids)
                                    )
                                    if a != b
                                ),
                                None,
                            ),
                            "cached": (
                                observed["usage"].get("prompt_tokens_details") or {}
                            ).get("cached_tokens", 0),
                            "expected_sha256": digest(fresh.ids),
                            "observed_sha256": digest(resume.ids),
                        }
                    )
                    save()
        finally:
            report["state_checks"] = await suite.rpc("qwen_response_probe_finish")
            save()
        checks = report["state_checks"]
        assert checks["copies"] and checks["cow"], "no real copies observed"
        assert all(r["same_bytes"] for r in checks["copies"] + checks["cow"])
        report["status"] = "PASS_FOR_BYTE_PRESERVATION_ONLY"
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--abi", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--status-prefix", type=Path, default=Path("/dev/shm/qwen-response-end")
    )
    args = parser.parse_args()
    args.run_id = f"response-state-{time.time_ns()}"
    args.observe = False
    args.case_group = "response-state"
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
