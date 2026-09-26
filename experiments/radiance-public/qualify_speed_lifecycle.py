"""Bounded cancellation/scheduling qualification using synthetic token prompts.

Run inside an isolated server container (same IPC namespace), never against a
user's snapshot root. Output contains hashes, counts and timings, not text or
token arrays. Length-limited/ignore-EOS responses deliberately test transaction
boundaries; this is not a throughput or natural-completion benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"
logger = logging.getLogger(__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_dispatch(report):
    require(bool(report["observed"]), "no real dispatch observations")
    require(any(r["mode"] == "FULL" for r in report["observed"]), "FULL never ran")
    for row in report["observed"]:
        require(row["requests"] == 1, "fair scheduler batched different live requests")
        if row["mode"] == "FULL":
            require(
                row["tokens"] == 8
                and row["uniform"] == 8
                and row["graph_requests"] == 1
                and row["graph_tokens"] == 8,
                "unqualified FULL graph shape was executed",
            )
    require(len(report["dispatch_grid"]) == 18, "incomplete dispatch metadata grid")
    for row in report["dispatch_grid"]:
        full = row["selected"]["cg_mode"] == "FULL"
        require(
            full == (row["requests"] == 1 and row["width"] == 8),
            "dispatcher admitted an unsupported FULL shape or lost M8",
        )


def check_equal(reference, observed):
    require(
        reference["tokens"] > 0 and reference["tokens"] == observed["tokens"],
        "output token count differs or is empty",
    )
    require(reference["sha256"] == observed["sha256"], "output token sequence differs")
    require(reference["finish"] == observed["finish"], "finish reason differs")


class Stream:
    def __init__(self, suite, label, chat, prompt, count):
        self.suite, self.label, self.chat = suite, label, chat
        self.prompt, self.count = prompt, count
        self.ids = []
        self.finish = None
        self.usage = None
        self.cancelled = False
        self.error = None
        self.started = time.monotonic()
        self.first = None
        self.ended = None
        self.task = asyncio.create_task(self.run())

    async def run(self):
        body = self.suite.body(self.chat, self.prompt, self.count)
        body.update(stream=True, stream_options={"include_usage": True})
        try:
            async with self.suite.client.stream(
                "POST", "/v1/completions", json=body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    require("error" not in event, "stream returned an error")
                    for choice in event.get("choices", []):
                        ids = choice.get("token_ids") or []
                        if ids and self.first is None:
                            self.first = time.monotonic()
                        self.ids.extend(ids)
                        if choice.get("finish_reason"):
                            self.finish = choice["finish_reason"]
                    if event.get("usage"):
                        self.usage = event["usage"]
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        except Exception as exc:
            self.error = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            self.ended = time.monotonic()

    async def cancel(self):
        require(not self.task.done(), "cancellation missed the live request")
        start = time.monotonic()
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        await self.suite.absent(self.chat)
        require(self.cancelled, "stream was not cancelled")
        return {
            "received_tokens": len(self.ids),
            "release_seconds": time.monotonic() - start,
        }

    async def result(self):
        await asyncio.wait_for(asyncio.shield(self.task), 300)
        require(not self.cancelled and self.error is None, "response did not complete")
        require(
            len(self.ids) == self.count, "stream missing token IDs or stopped early"
        )
        require(self.finish == "length", "unexpected boundary finish reason")
        return {
            "tokens": len(self.ids),
            "sha256": digest(self.ids),
            "finish": self.finish,
            "seconds": self.ended - self.started,
            "usage": self.usage,
        }


class Suite:
    def __init__(self, args, client):
        self.args, self.client = args, client
        self.jobs = []
        self.controls = {}
        self.sampling = {"temperature": 0, "top_k": 1, "seed": 0}
        self.report = {
            "schema": "urn:coherence:speed-lifecycle:v1",
            "status": "RUNNING",
            "run_id": args.run_id,
            "cases": [],
            "failures": [],
            "observer_enabled": args.observe,
            "case_group": args.case_group,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Single R9700, D7, serial fair scheduling, two resident banks",
            "sampling": {"temperature": 0, "top_k": 1, "ignore_eos": True},
        }

    def save(self):
        temporary = self.args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2) + "\n")
        temporary.replace(self.args.output)

    def chat(self, label, generation="initial"):
        return {
            "id": digest([self.args.run_id, label]),
            "generation": digest(generation),
            "title": "Synthetic lifecycle " + label,
            "cwd": "/qualification",
            "session_file": "",
        }

    def body(self, chat, prompt, count):
        from radiance_cache import cache_salt

        return {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": count,
            **self.sampling,
            "ignore_eos": True,
            "return_token_ids": True,
            "cache_salt": cache_salt(chat),
            "kv_transfer_params": {
                "qwen_chat": chat,
                "qwen_snapshot_abi": self.args.abi,
            },
        }

    async def post(self, path, body):
        result = await self.client.post(path, json=body)
        result.raise_for_status()
        return result.json()

    async def rpc(self, method, args=None):
        return (
            await self.post(
                "/collective_rpc", {"method": method, "args": args or [], "timeout": 90}
            )
        )["results"][0]

    def status(self):
        try:
            return json.loads(
                self.args.status_prefix.with_name(
                    self.args.status_prefix.name + "-scheduler.json"
                ).read_text()
            )
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def rows(self, chat):
        return [
            r for r in self.status().get("requests", []) if r["chat_id"] == chat["id"]
        ]

    async def wait(self, predicate, label, timeout=90):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            for job in self.jobs:
                if job.error:
                    await job.task
            result = predicate()
            if result:
                return result
            await asyncio.sleep(0.025)
        raise TimeoutError(label)

    async def absent(self, chat):
        await self.wait(
            lambda: not self.rows(chat), "cancelled/completed request still scheduled"
        )

    def stream(self, label, prompt, count, chat=None):
        job = Stream(self, label, chat or self.chat(label), prompt, count)
        self.jobs.append(job)
        return job

    async def generate(self, label, prompt, count, chat=None):
        job = self.stream(label, prompt, count, chat)
        result = await job.result()
        await self.absent(job.chat)
        return result

    async def priority(self, chat, level, active):
        value = self.controls.setdefault(
            chat["id"],
            {
                "chat_id": chat["id"],
                "client": digest([self.args.run_id, "client"])[:32],
                "answer": digest([chat["id"], "answer"])[:32],
                "sequence": 0,
            },
        )
        value.update(sequence=value["sequence"] + 1, priority=level, active=active)
        result = await self.post(
            "/qwen-radiance/priority", {**value, "abi": self.args.abi}
        )
        require(result.get("applied") is True, "priority was not applied")

    async def case(self, name, operation):
        self.report["current_case"] = name
        self.save()
        print(json.dumps({"case": name, "status": "RUNNING"}), flush=True)
        start = time.monotonic()
        try:
            details = await operation()
        except Exception as exc:
            self.report["failures"].append(
                {"case": name, "error": type(exc).__name__ + ": " + str(exc)}
            )
            self.report["status"] = "FAIL"
            self.save()
            raise
        self.report["cases"].append(
            {
                "name": name,
                "status": "PASS",
                "seconds": time.monotonic() - start,
                **(details or {}),
            }
        )
        self.save()
        print(json.dumps({"case": name, "status": "PASS"}), flush=True)

    async def run(self):
        self.report["candidate"] = await self.rpc("qwen_speed_metadata")
        require(
            self.report["candidate"].get("target_gemm_candidate"),
            "qualified GEMM absent",
        )
        require(
            self.report["candidate"].get("draft_attention_candidate"),
            "qualified drafter absent",
        )
        text = "\n".join(
            f"Record {i}: x={i % 71}; y={i % 137}; result=x+y." for i in range(10000)
        )
        ids = (await self.post("/tokenize", {"model": MODEL, "prompt": text}))["tokens"]
        suffix = (
            await self.post(
                "/tokenize",
                {
                    "model": MODEL,
                    "prompt": "\nContinue with a detailed Python implementation of a bounded queue and its tests.\n",
                },
            )
        )["tokens"]
        require(len(ids) >= 60000, "synthetic prompt too short")

        def prompt(size):
            return (
                ids[: size - len(suffix)] + suffix
                if size >= len(suffix)
                else ids[:size]
            )

        pa, pb, pl = prompt(2049), prompt(4097), prompt(60000)
        baseline = {}

        async def boundaries():
            rows = []
            for width in range(1, 9):
                p = prompt(width)
                count = (1, 2, 7, 8, 9, 15, 16, 17)[width - 1]
                await self.rpc("qwen_speed_set_combined", [False])
                a = await self.generate(f"boundary-control-{width}", p, count)
                await self.rpc("qwen_speed_set_combined", [True])
                b = await self.generate(f"boundary-candidate-{width}", p, count)
                self.report.setdefault("boundary_comparisons", []).append(
                    {"prompt_tokens": width, "control": a, "candidate": b}
                )
                self.save()
                check_equal(a, b)
                rows.append(
                    {"prompt_tokens": width, "output_tokens": count, "equal": True}
                )
            for size in (2047, 2048, 2049):
                await self.rpc("qwen_speed_set_combined", [False])
                a = await self.generate(f"chunk-control-{size}", prompt(size), 33)
                await self.rpc("qwen_speed_set_combined", [True])
                b = await self.generate(f"chunk-candidate-{size}", prompt(size), 33)
                self.report.setdefault("boundary_comparisons", []).append(
                    {"prompt_tokens": size, "control": a, "candidate": b}
                )
                self.save()
                check_equal(a, b)
                rows.append({"prompt_tokens": size, "output_tokens": 33, "equal": True})
            return {"comparisons": rows}

        if self.args.case_group == "all":
            await self.case("capture_and_chunk_boundaries", boundaries)

        async def controls():
            await self.rpc("qwen_speed_set_combined", [False])
            for label, p, n in (("a", pa, 768), ("b", pb, 384), ("long", pl, 128)):
                baseline[label] = await self.generate("baseline-" + label, p, n)
            await self.rpc("qwen_speed_set_combined", [True])
            for label, p, n in (("a", pa, 768), ("b", pb, 384)):
                observed = await self.generate("candidate-" + label, p, n)
                check_equal(baseline[label], observed)
            return {"references": baseline}

        await self.case("uninterrupted_controls", controls)

        async def cancel_decode():
            rows = []
            for count in (1, 64, 257):
                job = self.stream("cancel-decode-" + str(count), pa, 768)
                await self.wait(
                    lambda job=job, count=count: len(job.ids) >= count,
                    "decode did not begin",
                )
                cancelled = await job.cancel()
                resumed = await self.generate("resume-decode", pa, 768, job.chat)
                check_equal(baseline["a"], resumed)
                rows.append({**cancelled, "replay_equal": True})
            return {"interruptions": rows}

        await self.case("cancel_decode_and_replay", cancel_decode)

        async def cancel_prefill():
            job = self.stream("cancel-prefill", pl, 128)
            seen = await self.wait(
                lambda: next(
                    (
                        r
                        for r in self.rows(job.chat)
                        if 2048 <= r["computed_tokens"] < len(pl)
                    ),
                    None,
                ),
                "cold prefill not observed",
            )
            require(not job.ids, "prefill cancellation happened after output")
            result = await job.cancel()
            resumed = await self.generate("resume-prefill", pl, 128, job.chat)
            check_equal(baseline["long"], resumed)
            return {
                **result,
                "computed_tokens_when_cancelled": seen["computed_tokens"],
                "replay_equal": True,
            }

        await self.case("cancel_cold_prefill_and_replay", cancel_prefill)

        async def cancel_queued():
            a = self.stream("queue-owner", pa, 768)
            await self.wait(lambda: len(a.ids) >= 16, "owner did not start")
            b = self.stream("queue-cancel", pb, 384)
            await self.wait(
                lambda: any(r["state"] == "queued" for r in self.rows(b.chat)),
                "second request was not queued",
            )
            require(not b.ids, "queued request generated prematurely")
            result = await b.cancel()
            check_equal(baseline["a"], await a.result())
            check_equal(
                baseline["b"], await self.generate("resume-queued", pb, 384, b.chat)
            )
            return {**result, "owner_and_cancelled_replay_equal": True}

        await self.case("cancel_queued_request", cancel_queued)

        async def concurrent_equal():
            a = self.stream("equal-a", pa, 768)
            await self.wait(lambda: len(a.ids) >= 16, "owner did not start")
            b = self.stream("equal-b", pb, 384)
            await self.wait(lambda: bool(self.rows(b.chat)), "second chat not admitted")
            ar, br = await asyncio.gather(a.result(), b.result())
            check_equal(baseline["a"], ar)
            check_equal(baseline["b"], br)
            # A final packet and B's first packet may overlap in transport.
            # A 100 ms allowance cannot hide interleaving of these long outputs.
            require(
                b.first is not None and b.first >= a.ended - 0.1,
                "equal-priority chat interrupted response generation",
            )
            return {"a": ar, "b": br, "both_equal": True}

        await self.case("equal_priority_response_handover", concurrent_equal)

        async def urgent(cancel_side=None):
            label = "priority-" + (cancel_side or "complete")
            a = self.stream(label + "-a", pa, 768)
            await self.wait(lambda: len(a.ids) >= 32, "owner did not start")
            bc = self.chat(label + "-b")
            await self.priority(bc, 2, True)
            b = self.stream(label + "-b", pb, 384, bc)
            await self.wait(
                lambda: (
                    len(b.ids) >= 16
                    and any(r["state"] == "paused" for r in self.rows(a.chat))
                ),
                "priority preemption not observed",
            )
            paused_at = len(a.ids)
            if cancel_side == "parked":
                cancelled = await a.cancel()
                check_equal(baseline["b"], await b.result())
                await self.priority(bc, 2, False)
                check_equal(
                    baseline["a"],
                    await self.generate(label + "-resume", pa, 768, a.chat),
                )
            elif cancel_side == "urgent":
                cancelled = await b.cancel()
                await self.priority(bc, 2, False)
                check_equal(baseline["a"], await a.result())
                check_equal(
                    baseline["b"], await self.generate(label + "-resume", pb, 384, bc)
                )
            else:
                cancelled = {}
                check_equal(baseline["b"], await b.result())
                await self.priority(bc, 2, False)
                check_equal(baseline["a"], await a.result())
            return {"paused_after_tokens": paused_at, "both_equal": True, **cancelled}

        for mode in (None, "parked", "urgent"):
            await self.case(
                "priority2_" + (mode or "complete"), lambda mode=mode: urgent(mode)
            )

        async def priority_hold():
            ac = self.chat("hold-a")
            await self.priority(ac, 1, True)
            a = self.stream("hold-a", pa, 768, ac)
            await self.wait(lambda: len(a.ids) >= 16, "owner did not start")
            b = self.stream("hold-b", pb, 384)
            check_equal(baseline["a"], await a.result())
            await self.wait(
                lambda: bool(self.status().get("priority_hold")),
                "tool-time priority hold absent",
            )
            require(not b.ids, "priority 1 released GPU during the active answer")
            # The next model request represents a tool result arriving while
            # the same answer is reserved. Replay is deterministic by design.
            a2 = self.stream("hold-tool-continuation", pa, 768, ac)
            check_equal(baseline["a"], await a2.result())
            require(not b.ids, "lower-priority request bypassed answer reservation")
            await self.priority(ac, 1, False)
            check_equal(baseline["b"], await b.result())
            return {
                "tool_continuation_equal": True,
                "waiter_progress_after_release": True,
            }

        await self.case("priority1_tool_boundary_hold", priority_hold)

        async def sampled_cancel():
            # Exercise Pi's stochastic path as well as greedy state equality.
            # Each request has its own fixed seed, so cancelling/replaying A
            # must neither advance B's RNG nor alter A's fresh replay.
            self.sampling = {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
                "seed": 113,
            }
            references = {}
            for label, p, n in (("a", pa, 512), ("b", pb, 384)):
                references[label] = await self.generate(
                    "sampled-control-" + label, p, n
                )
            a = self.stream("sampled-cancel-a", pa, 512)
            await self.wait(lambda: len(a.ids) >= 16, "sampled owner did not start")
            b = self.stream("sampled-waiter-b", pb, 384)
            await self.wait(
                lambda: bool(self.rows(b.chat)), "sampled waiter not admitted"
            )
            cancellation = await a.cancel()
            check_equal(references["b"], await b.result())
            check_equal(
                references["a"],
                await self.generate("sampled-replay-a", pa, 512, a.chat),
            )
            return {
                "sampling": self.sampling,
                **cancellation,
                "waiter_and_cancelled_replay_equal": True,
            }

        await self.case("sampled_cancel_and_waiter_replay", sampled_cancel)

        if self.args.observe:
            self.report["graph_dispatch"] = await self.rpc("qwen_lifecycle_report")
            check_dispatch(self.report["graph_dispatch"])
        require(not self.status().get("requests"), "scheduler not idle after suite")
        self.report["status"] = "PASS"
        self.report.pop("current_case", None)
        self.save()

    async def cleanup(self):
        for job in self.jobs:
            if not job.task.done():
                job.task.cancel()
        await asyncio.gather(*(j.task for j in self.jobs), return_exceptions=True)
        for chat_id, value in list(self.controls.items()):
            if value.get("active"):
                try:
                    await self.priority({"id": chat_id}, value["priority"], False)
                except Exception as exc:
                    logger.exception("Failed to release qualification priority lease")
                    self.report.setdefault("cleanup_failures", []).append(str(exc))
                    self.report["status"] = "FAIL"
                    self.save()


async def execute(args):
    import httpx

    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=600, trust_env=False
    ) as client:
        suite = Suite(args, client)
        try:
            await suite.run()
        finally:
            await suite.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--abi", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--status-prefix", type=Path, default=Path("/dev/shm/qwen-stage-timing")
    )
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--case-group", choices=("all", "lifecycle"), default="all")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
