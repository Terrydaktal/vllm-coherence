"""Collect >=60K natural output tokens per head variant on private Pi prefixes.

No tools execute. Prompts, output IDs and hidden vectors stay in private tmpfs.
Timing runs exclude capture; a separate reference run captures every head call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from benchmark_private_head import digest, prepare_environment, request, write

MODES = ("full", "block80", "global128", "global256")


def totals(records, modes=MODES):
    return {mode: sum(r["output_tokens"] for r in records if r["mode"] == mode) for mode in modes}


def complete(records, target, modes=MODES):
    return bool(records) and min(totals(records, modes).values()) >= target


def plan_pair(index, fixture_count):
    cycle, fixture = divmod(index, fixture_count)
    offset = index % len(MODES)
    return fixture, 731 + cycle * 1009 + fixture, MODES[offset:] + MODES[:offset]


def tokenize(server, payload):
    body = {
        k: payload[k]
        for k in ("messages", "tools", "model", "chat_template_kwargs")
        if k in payload
    }
    body["add_generation_prompt"] = True
    req = urllib.request.Request(
        server.client.base + "/tokenize",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + server.client.nonce,
        },
    )
    with server.client.opener.open(req, timeout=180) as response:
        result = json.load(response)["tokens"]
    if not 55000 <= len(result) <= 67000:
        raise ValueError("natural workload outside admitted approximately-60K input window")
    return result


def serve(spec, root, reports, target):
    from qwen_r9700_lab.conformance_runtime import NativeServer

    records, captures = [], []
    prepare_environment(spec, root, reports)
    spec["case_timeout_seconds"] = 21600
    write(root / "control.json", {"mode": "full", "capture": False})
    with NativeServer(spec, root / "server", allow_gpu=True, head=True, observe=False) as server:
        paths = sorted(root.glob("payload-*.json"))
        if not paths:
            raise ValueError("no private Pi requests supplied")
        fixtures = []
        description = []
        for path in paths:
            payload = json.loads(path.read_text())
            tokens = tokenize(server, payload)
            fixtures.append(tokens)
            description.append(
                {
                    "index": len(fixtures) - 1,
                    "input_tokens": len(tokens),
                    "token_sha256": digest(tokens),
                    "messages": len(payload["messages"]),
                    "tools": len(payload.get("tools", [])),
                }
            )
        write(reports / "fixtures.json", description)
        common = {
            "chat_id": digest(str(root)),
            "progress": reports / "current-request.json",
        }
        warm = request(server, root, fixtures[0], "full", 0, limit=32, **common)
        write(reports / "warmup.json", warm)
        if not list(root.glob("hook-*.json")):
            raise ValueError("target hook absent; comparison refused")
        pair = 0
        while not complete(records, target):
            fixture, seed, order = plan_pair(pair, len(fixtures))
            tokens = fixtures[fixture]
            # Admit all remaining model context; retain natural EOS/tool boundaries.
            limit = server.settings["config"]["max_model_len"] - len(tokens) - 1
            for mode in order:
                result = request(server, root, tokens, mode, seed, limit=limit, **common)
                result.update(fixture=fixture, pair=pair)
                records.append(result)
                write(reports / "generation.json", records)
                status = {"phase": "generation", "completed": totals(records), "last": result}
                write(reports / "checkpoint.json", status)
                print(json.dumps(status), flush=True)
            pair += 1
        # Separate capture costs from timing. Capture all calls without the old 256-call ceiling.
        pair = 0
        while not complete(captures, target, ("full",)):
            fixture, seed, _ = plan_pair(pair, len(fixtures))
            tokens = fixtures[fixture]
            limit = server.settings["config"]["max_model_len"] - len(tokens) - 1
            result = request(
                server,
                root,
                tokens,
                "full",
                seed,
                limit=limit,
                capture=True,
                capture_calls=1000000,
                compact_capture=True,
                **common,
            )
            result.update(fixture=fixture, pair=pair)
            matches = [r for r in records if r["pair"] == pair and r["mode"] == "full"]
            if len(matches) != 1:
                raise ValueError("capture has no unique paired reference timing run")
            result["matches_timed_reference"] = (
                matches[0]["output_sha256"] == result["output_sha256"]
            )
            captures.append(result)
            write(reports / "capture-generation.json", captures)
            status = {"phase": "capture", "completed": totals(captures, ("full",)), "last": result}
            write(reports / "checkpoint.json", status)
            print(json.dumps(status), flush=True)
            pair += 1
        request(server, root, fixtures[0], "full", 0, limit=1, **common)
        write(
            reports / "head-calls.json",
            [json.loads(p.read_text()) for p in sorted(root.glob("trial-*.json"))],
        )
        write(
            reports / "capture-counts.json",
            {
                "files": len(list(root.glob("head-*.pt"))),
                "output_tokens": totals(captures, ("full",))["full"],
            },
        )


def main(args):
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    root, reports = args.private_root, args.report_root
    if not str(root).startswith("/dev/shm/qwen-private-head-") or root.stat().st_mode & 0o077:
        raise ValueError("private workload must be private tmpfs")
    if args.output_tokens < 60000:
        raise ValueError("long benchmark requires at least 60,000 output tokens per variant")
    reports.mkdir(mode=0o700)
    sources = [
        Path(__file__),
        *(
            Path(__file__).parent / n
            for n in (
                "benchmark_private_head.py",
                "private_head_probe.py",
                "benchmark_private_head_stage.py",
            )
        ),
    ]
    write(
        reports / "methodology.json",
        {
            "minimum_output_tokens_per_variant": args.output_tokens,
            "minimum_captured_reference_output_tokens": args.output_tokens,
            "ignore_eos": False,
            "tools_executed": 0,
            "sampling": {"temperature": 1, "top_p": 0.95, "top_k": 20},
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "started_ns": time.time_ns(),
        },
    )
    spec = json.loads(args.spec.read_text())
    with gpu_lease(reports / "gpu-lease"):
        print(json.dumps({"phase": "gpu_acquired"}), flush=True)
        serve(spec, root, reports, args.output_tokens)
        from benchmark_private_head_stage import run

        run(spec, root, reports)
    write(reports / "completed.json", {"status": "MEASURED", "completed_ns": time.time_ns()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=60000)
    main(parser.parse_args())
