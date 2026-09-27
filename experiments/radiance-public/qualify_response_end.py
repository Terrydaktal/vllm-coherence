"""Synthetic native DFlash response-end continuation and cache qualification.

Run inside the isolated test container. No private chat files are read. Reports
contain counts, timings and hashes. Restart fixtures contain synthetic token
arrays and must not be substituted with private conversations.
"""

import argparse
import asyncio
import hashlib
import json
import sysconfig
import time
from pathlib import Path

import httpx
from qualify_speed_lifecycle import Suite


def digest(tokens):
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


async def run(args):
    package = Path(sysconfig.get_paths()["purelib"])
    report = {
        "schema": "urn:coherence:response-end-qualification:v1",
        "status": "RUNNING",
        "cases": [],
        "mode": args.mode,
        "expect_rebuild": args.expect_rebuild,
        "sampling": "T=1,p=0.95,k=40,seed=0" if args.sampled else "greedy",
        "target_head_policy": "full-bf16" if getattr(args, "full_bf16_head", False) else "production",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "installed_sources": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in (
                "qwen_radiance_response_end.py",
                "qwen_radiance_response_offload.py",
                "qwen_radiance_fair_scheduler.py",
                "qwen_radiance_chat_tier.py",
                "qwen_radiance_cache.py",
                "speed_candidate_worker.py",
                "vllm/v1/core/kv_cache_utils.py",
                "vllm/v1/core/sched/scheduler.py",
                "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:
        suite = Suite(args, client)
        if args.sampled:
            suite.sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "seed": 0}
        elif getattr(args, "full_bf16_head", False):
            suite.sampling = {"temperature": 0, "top_k": 129, "seed": 0}

        async def consume(chat, next_prompt, expected, minimum, label):
            resumed = suite.stream(label, next_prompt, len(expected), chat)
            observed = await resumed.result()
            await suite.absent(chat)
            cached = (
                (observed.get("usage") or {})
                .get("prompt_tokens_details", {})
                .get("cached_tokens", 0)
            )
            row = {
                "case": label,
                "continuation_tokens": len(resumed.ids),
                "same_tokens": resumed.ids == expected,
                "first_difference": next(
                    (
                        i
                        for i, pair in enumerate(zip(expected, resumed.ids))
                        if pair[0] != pair[1]
                    ),
                    None,
                ),
                "expected_sha256": digest(expected),
                "observed_sha256": digest(resumed.ids),
                "cached_tokens": cached,
                "prompt_tokens": len(next_prompt),
                "uncached_tokens": len(next_prompt) - cached,
                "first_data_seconds": resumed.first - resumed.started,
                "seconds": observed["seconds"],
            }
            phases_path = args.status_prefix.with_name(
                args.status_prefix.name + "-phases.json"
            )
            if phases_path.exists():
                phases = json.loads(phases_path.read_text())
                phase = next(
                    (
                        r
                        for r in phases.get("recent", [])
                        if r.get("chat_id") == chat["id"]
                        and r.get("generation") == chat["generation"]
                        and r.get("input_tokens") == len(next_prompt)
                    ),
                    None,
                )
                if phase is not None:
                    row["backend"] = {
                        k: phase.get(k)
                        for k in (
                            "input_tokens",
                            "computed_tokens",
                            "cached_tokens",
                            "timings_ms",
                            "response_end_tokens",
                            "local_response_end_tokens",
                        )
                    }
                    if (
                        phase.get("local_response_end_tokens", 0)
                        and phase["cached_tokens"] != cached
                    ):
                        raise AssertionError(
                            "spinner telemetry rounded down an exact GPU hit"
                        )
            report["cases"].append(row)
            save()
            if not row["same_tokens"]:
                args.output.with_suffix(".failure-synthetic.json").write_text(
                    json.dumps(
                        {
                            "chat": chat,
                            "prompt": next_prompt,
                            "expected": expected,
                            "observed": resumed.ids,
                            "minimum": minimum,
                        }
                    )
                    + "\n"
                )
                raise AssertionError(f"continuation differs: {label}")
            if cached < minimum:
                raise AssertionError(f"response endpoint was not reused: {row}")
            return row

        if args.mode == "restore":
            for fixture in json.loads(args.fixture.read_text()):
                row = await consume(
                    fixture["chat"],
                    fixture["prompt"],
                    fixture["expected"],
                    0 if args.expect_rebuild else fixture["minimum"],
                    "disk-restart",
                )
                if args.expect_rebuild and row["cached_tokens"] >= fixture["minimum"]:
                    raise AssertionError("damaged endpoint was not rejected")
            report["status"] = "PASS_FOR_DECLARED_SCOPE"
            save()
            return

        fixtures = []
        text = (
            "Complete this Python module with numbered, distinct test cases. "
            "Each case should include assertions and explanatory comments.\n"
            + "# Documentation of expected integer arithmetic.\n" * 400
            + "\ndef test_case_001():\n"
        )
        tokenized = await suite.post(
            "/tokenize",
            {
                "model": "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate",
                "prompt": text,
            },
        )
        raw = tokenized["tokens"]
        if args.mode == "pressure":
            # Fixed synthetic dimensions reproduce the September 27 near-limit
            # compaction incident. The negative control changes ownership only,
            # never numerical kernels. No user transcript is used.
            chat = suite.chat("capacity")
            initial = (raw * ((233162 + len(raw) - 1) // len(raw)))[:233162]
            producer = suite.stream("capacity-producer", initial, 20, chat)
            produced = await producer.result()
            await suite.absent(chat)
            prefix = initial + producer.ids
            suffix = (raw * ((239720 + len(raw) - 1) // len(raw)))[:239720 - len(prefix)]
            prompt = prefix + suffix
            report["producer"] = produced
            save()
            await suite.rpc("qwen_response_probe_pressure_mode", [False])
            negative = suite.stream("old-policy", prompt, 1024, chat)
            samples = []
            deadline = time.monotonic() + 60
            try:
                while time.monotonic() < deadline and not negative.task.done():
                    rows = await suite.rpc("qwen_response_probe_allocator")
                    samples.extend(rows)
                    if any(row["preemptions"] >= 2 for row in rows):
                        break
                    await asyncio.sleep(0.5)
                report["negative_control"] = {
                    "samples": samples,
                    "received_tokens": len(negative.ids),
                    "repeated_preemption": any(r["preemptions"] >= 2 for r in samples),
                }
                save()
            finally:
                if not negative.task.done():
                    await negative.cancel()
                await suite.absent(chat)
                await suite.rpc("qwen_response_probe_pressure_mode", [True])
            assert report["negative_control"]["repeated_preemption"], "old policy did not reproduce"
            for label, current, count in (("repaired-compaction-size", prompt, 1024),):
                job = suite.stream(label, current, count, chat)
                samples = []
                deadline = time.monotonic() + 120
                while not job.task.done():
                    samples.extend(await suite.rpc("qwen_response_probe_allocator"))
                    if time.monotonic() >= deadline:
                        report["stalled_case"] = {"case": label, "samples": samples}
                        save()
                        await job.cancel()
                        raise TimeoutError("repaired continuation did not progress")
                    await asyncio.sleep(0.5)
                result = await job.result()
                await suite.absent(chat)
                report["cases"].append({
                    "case": label, **result, "samples": samples,
                    "first_data_seconds": job.first - job.started,
                })
                save()
                assert not any(row["preemptions"] for row in samples)
            prefix = prompt + job.ids
            near_limit = prefix + (raw * 100)[:252000 - len(prefix)]
            assert len(near_limit) == 252000
            job = suite.stream("near-model-limit", near_limit, 512, chat)
            samples = []
            deadline = time.monotonic() + 120
            while not job.task.done():
                samples.extend(await suite.rpc("qwen_response_probe_allocator"))
                if time.monotonic() >= deadline:
                    report["stalled_case"] = {"case": "near-model-limit", "samples": samples}
                    save()
                    await job.cancel()
                    raise TimeoutError("near-limit admission or generation stalled")
                await asyncio.sleep(0.5)
            result = await job.result()
            await suite.absent(chat)
            report["cases"].append({
                "case": "near-model-limit", **result, "samples": samples,
                "first_data_seconds": job.first - job.started,
            })
            assert not any(row["preemptions"] for row in samples)
            pressure_file = args.status_prefix.with_name(args.status_prefix.name + "-cache-pressure.json")
            report["last_reclaim"] = json.loads(pressure_file.read_text())
            report["status"] = "PASS_FOR_CAPACITY_AND_FORWARD_PROGRESS"
            report["scope"] = "Allocator liveness, not an independent numerical oracle"
            save()
            return
        tool_suffix = (
            (
                await suite.post(
                    "/tokenize",
                    {
                        "model": "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate",
                        "prompt": "\nTool result: tests finished. 7 passed, 1 failed: expected 42, got 41.\nContinue by fixing the off-by-one error and adding a regression test.\n",
                    },
                )
            )["tokens"]
            if args.mode in {"tool", "tool-repeat"}
            else None
        )
        for context in args.contexts:
            prompt = (raw * ((context + len(raw) - 1) // len(raw)))[:context]
            reference = suite.stream(f"reference-{context}", prompt, args.output_tokens)
            await reference.result()
            await suite.absent(reference.chat)
            for split in args.splits:
                chat = suite.chat(f"split-{context}-{split}")
                if args.mode == "stop":
                    candidates = [
                        i
                        for i in range(
                            split - 1,
                            len(reference.ids)
                            - args.append_tokens
                            - args.continue_tokens,
                        )
                        if reference.ids[i] not in reference.ids[:i]
                    ]
                    if not candidates:
                        raise AssertionError(
                            "fixture has no suitable distinct stop token"
                        )
                    sampling = suite.sampling
                    suite.sampling = {
                        **sampling,
                        "stop_token_ids": [reference.ids[candidates[0]]],
                    }
                    first = suite.stream(
                        "producer-stop", prompt, args.output_tokens, chat
                    )
                    await asyncio.wait_for(first.task, 300)
                    suite.sampling = sampling
                    if first.finish != "stop" or not first.ids:
                        raise AssertionError(
                            "stop-token control did not terminate normally"
                        )
                    split = len(first.ids)
                else:
                    first = suite.stream("producer", prompt, split, chat)
                    await first.result()
                await suite.absent(chat)
                if first.ids != reference.ids[:split]:
                    raise AssertionError(
                        "fresh producer differs from uninterrupted reference"
                    )
                extension = (
                    tool_suffix
                    if tool_suffix is not None
                    else reference.ids[split : split + args.append_tokens]
                )
                next_prompt = prompt + first.ids + extension
                expected = reference.ids[
                    split + len(extension) : split
                    + len(extension)
                    + args.continue_tokens
                ]
                if args.mode in {"tool", "tool-repeat"}:
                    control_chat = suite.chat(f"control-tool-{context}-{split}")
                    if args.mode == "tool-repeat":
                        # Keep the executed prefill/decode partition identical.
                        # A monolithic fresh prefill is a separate numerical
                        # check (--mode tool), not an equivalent cache oracle.
                        control = suite.stream(
                            "control-producer", prompt, split, control_chat
                        )
                        await control.result()
                        await suite.absent(control_chat)
                        if control.ids != first.ids:
                            raise AssertionError("independent producer differs")
                    fresh = suite.stream(
                        f"fresh-tool-{context}-{split}",
                        next_prompt,
                        args.continue_tokens,
                        control_chat,
                    )
                    await fresh.result()
                    await suite.absent(fresh.chat)
                    expected = fresh.ids
                minimum = context + split - 1
                if args.mode == "prepare":
                    fixtures.append(
                        {
                            "chat": chat,
                            "prompt": next_prompt,
                            "expected": expected,
                            "minimum": minimum,
                        }
                    )
                    args.fixture.write_text(json.dumps(fixtures) + "\n")
                    continue
                if args.mode in {"handover", "eviction"}:
                    for index in range(1 if args.mode == "handover" else 3):
                        await suite.generate(
                            f"interloper-{context}-{split}-{index}", prompt, 16
                        )
                if args.mode == "cancel":
                    aborted = suite.stream("aborted-successor", next_prompt, 2048, chat)
                    await suite.wait(
                        lambda job=aborted: len(job.ids) >= 32, "decode not observed"
                    )
                    await aborted.cancel()
                row = await consume(chat, next_prompt, expected, minimum, args.mode)
                row.update(
                    context_tokens=context,
                    split_output_tokens=split,
                    appended_tokens=len(extension),
                    producer_finish=first.finish,
                )
                save()
                if args.fixture is not None:
                    # Preserve a future continuation of the completed consumer
                    # for a separate process/container restart. All IDs here
                    # came from this synthetic uninterrupted control.
                    offset = split + len(extension) + len(expected)
                    future = reference.ids[offset : offset + args.append_tokens]
                    future_expected = reference.ids[
                        offset + len(future) : offset
                        + len(future)
                        + args.continue_tokens
                    ]
                    if len(future_expected) != args.continue_tokens:
                        raise AssertionError(
                            "reference too short for restart continuation"
                        )
                    fixtures.append(
                        {
                            "chat": chat,
                            "prompt": next_prompt + expected + future,
                            "expected": future_expected,
                            "minimum": len(next_prompt) + len(expected) - 1,
                        }
                    )
                    args.fixture.write_text(json.dumps(fixtures) + "\n")
    report["status"] = "PASS_FOR_DECLARED_SCOPE"
    save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8081")
    p.add_argument("--abi", required=True)
    p.add_argument("--run-id", default=f"response-end-{time.time_ns()}")
    p.add_argument(
        "--status-prefix", type=Path, default=Path("/dev/shm/qwen-response-end")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=(
            "warm",
            "handover",
            "eviction",
            "cancel",
            "prepare",
            "restore",
            "tool",
            "tool-repeat",
            "stop",
            "pressure",
        ),
        default="warm",
    )
    p.add_argument("--fixture", type=Path)
    p.add_argument("--sampled", action="store_true")
    p.add_argument("--full-bf16-head", action="store_true", help="Greedy diagnostics: request top_k=129 to select the qualified full-vocabulary head")
    p.add_argument(
        "--expect-rebuild",
        action="store_true",
        help="Negative control: require rejection of a damaged restart endpoint",
    )
    p.add_argument("--contexts", type=int, nargs="+", default=[1641, 1651, 3301])
    p.add_argument("--splits", type=int, nargs="+", default=[1, 7, 19, 63])
    p.add_argument("--append-tokens", type=int, default=5)
    p.add_argument("--continue-tokens", type=int, default=64)
    p.add_argument("--output-tokens", type=int, default=192)
    args = p.parse_args()
    if args.sampled and args.full_bf16_head:
        p.error("--full-bf16-head is restricted to greedy comparisons")
    args.observe = False
    args.case_group = "response-end"
    if args.mode in {"prepare", "restore"} and args.fixture is None:
        p.error("restart qualification requires --fixture")
    if args.fixture is not None and args.mode not in {"warm", "prepare", "restore"}:
        p.error("restart fixtures require warm, prepare or restore mode")
    if args.expect_rebuild and args.mode != "restore":
        p.error("--expect-rebuild is only for an intentionally damaged restart fixture")
    if (
        min(args.contexts) < 1
        or min(args.splits) < 1
        or max(args.splits) + args.append_tokens + args.continue_tokens
        > args.output_tokens
    ):
        p.error("nonempty cases must fit inside the uninterrupted reference")
    try:
        asyncio.run(run(args))
    except Exception as exc:
        report = json.loads(args.output.read_text()) if args.output.exists() else {}
        report.update(status="FAIL", failure=type(exc).__name__ + ": " + str(exc))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
