"""Isolated eight-combination kernel experiment, with Pi sampling and correctness checks."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import benchmark_runtime_flags as bench

SAMPLING = {"temperature": 1, "top_p": 0.95, "top_k": 20, "seed": 0}
TOOL_CASES = 64


def variants():
    result = {}
    for attention, residual, tiled in itertools.product((0, 1), repeat=3):
        result[f"K{attention}{residual}{tiled}"] = {
            "R4D_ATTN_FP8": "3" if attention else "0",
            "RADIANCE_NORMQUANT_FUSION": str(residual),
            "RADIANCE_FP8_STREAM": str(residual),
            "RADIANCE_MXFP4_A_TILED_MIN_M": "513" if tiled else "0",
            "RADIANCE_GDN_NORM_QUANT": str(tiled),
        }
    result["K000R"] = dict(result["K000"])
    return result


def launcher(original, root, variant):
    settings = variants()[variant]
    text = bench.make_launcher(original, root, variant, f"benchmarks/{root.name}", flags={})
    # Match the environment source, not another use of the same profile path
    # in the optimized-payload integrity check.
    profile = "done < <(jq -r '.kernel_environment | to_entries[] | [.key, .value] | @tsv' radiance-vllm-mxfp4/runtime-radiance-1.0.16.json)"
    if text.count(profile) != 1 or text.count("-e R4D_ATTN_FP8=0 ") != 1:
        raise ValueError("kernel environment launcher anchors changed")
    text = text.replace(profile, profile.rsplit(" ", 1)[0] + " " + shlex.quote(str(root / f"profile-{variant}.json")) + ")")
    text = text.replace("-e R4D_ATTN_FP8=0 ", f"-e R4D_ATTN_FP8={settings['R4D_ATTN_FP8']} ")
    compilation = re.search(r" \\\n[ \t]*--compilation-config ('[^']+')\n?$", text)
    if not compilation:
        raise ValueError("expected final compilation-config argument is missing")
    config = json.loads(shlex.split(compilation[1])[0])
    text = text[: compilation.start()] + "\n"
    anchor = '\t"${release_environment[@]}" \\\n'
    if text.count(anchor) != 1:
        raise ValueError("release environment anchor changed")
    extra = (
        "\t-e RADIANCE_COMPILATION_CONFIG="
        + shlex.quote(json.dumps(config, separators=(",", ":")))
        + " \\\n"
    )
    private_directory = bench.read_json(root / "manifest.json").get("private_fixture_directory")
    if private_directory:
        if not re.fullmatch(r"/dev/shm/qwen-private-replay-[a-z0-9_]+", private_directory):
            raise ValueError(
                "private fixtures must be in their isolated temporary memory directory"
            )
        extra += f"\t-v {shlex.quote(private_directory)}:/private-fixtures:ro \\\n"
        extra += "\t-e QWEN_BENCHMARK_PRIVATE_FIXTURES=/private-fixtures \\\n"
    return text.replace(anchor, anchor + extra)


def tool_case(index):
    """Small tasks with independently known arguments, including escapes and nested JSON."""
    kind = index % 4
    values = (
        ({"number": 37 + index}, {"number": {"type": "integer"}}),
        (
            {"path": f"src/module_{index}.py", "text": 'line one\nline "two"\\end'},
            {"path": {"type": "string"}, "text": {"type": "string"}},
        ),
        (
            {"items": [index, index + 1, index + 2], "enabled": False},
            {
                "items": {"type": "array", "items": {"type": "integer"}},
                "enabled": {"type": "boolean"},
            },
        ),
        (
            {"record": {"id": index, "label": "alpha"}},
            {
                "record": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "label": {"type": "string"}},
                    "required": ["id", "label"],
                }
            },
        ),
    )
    arguments, properties = values[kind]
    function = {
        "name": "record",
        "description": "Record the supplied values exactly.",
        "parameters": {"type": "object", "properties": properties, "required": list(properties)},
    }
    return arguments, function


def check_tool(choice, arguments):
    calls = choice.get("message", {}).get("tool_calls") or []
    if choice.get("finish_reason") != "tool_calls" or len(calls) != 1:
        return False
    try:
        function = calls[0]["function"]
        actual = json.loads(function["arguments"])
        return function["name"] == "record" and json.dumps(actual, sort_keys=True) == json.dumps(
            arguments, sort_keys=True
        )
    except (KeyError, TypeError, json.JSONDecodeError):
        return False


def tool_checks(args):
    from qwen_radiance_cache import request_tail_flush

    chat = {
        "id": bench.digest([args.variant, "kernel-tools"]),
        "generation": bench.digest([args.variant, "kernel-tools-generation"]),
    }
    rows = []
    for index in range(TOOL_CASES):
        expected, function = tool_case(index)
        payload = {
            "model": bench.MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Call record exactly once with these arguments. "
                    "Do not provide a textual answer: " + json.dumps(expected),
                }
            ],
            "tools": [{"type": "function", "function": function}],
            "tool_choice": "auto",
            "max_tokens": 512,
            **SAMPLING,
            "seed": index,
            "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "off"},
            "cache_salt": f"qwen-chat-cache-v1:{chat['id']}:{chat['generation']}",
            "kv_transfer_params": {
                "qwen_chat": chat,
                "qwen_snapshot_abi": os.environ["QWEN_RADIANCE_CACHE_ABI"],
            },
        }
        result = json.loads(bench.http(args.endpoint, "/v1/chat/completions", payload, 120))
        choice = result["choices"][0]
        row = {
            "case": index,
            "kind": index % 4,
            "passed": check_tool(choice, expected),
            "finish_reason": choice.get("finish_reason"),
            "tool_call_count": len(choice.get("message", {}).get("tool_calls") or []),
            "usage": result["usage"],
        }
        rows.append(row)
        bench.write_json(args.root / f"{args.variant}-tool-tests.json", rows)
        if index % 8 == 7:
            bench.emit(
                event="tool_progress",
                variant=args.variant,
                completed=len(rows),
                passed=sum(row["passed"] for row in rows),
            )
    request_tail_flush(chat, timeout=120)
    return rows


def worker(args):
    from qwen_radiance_cache import request_tail_flush

    checks = []
    for context in (60000, 200000):
        prompt = bench.fixture(args.endpoint, args.root, context)
        chat = {
            "id": bench.digest([args.variant, context, "kernel-speed"]),
            "generation": bench.digest([args.variant, context, "kernel-speed-generation"]),
        }
        bench.write_json(args.root / f"{args.variant}-{context}-identity.json", chat)

        def trial(label, count, chat=chat, prompt=prompt, **kwargs):
            return bench.trial(
                args.endpoint, args.root, args.variant, chat, prompt, label, count, **kwargs
            )

        cold = trial("cold", 256)
        warm = trial("greedy-warm", 256, cool=False)
        exact = trial("full-head", 256, sampling={"logprobs": 1}, cool=False)
        sampled = [trial("warmup", 1024, sampling=SAMPLING)]
        sampled += [trial(f"measured-{repeat}", 1024, sampling=SAMPLING) for repeat in range(1, 4)]
        checks.append(
            {
                "context": context,
                "actual_context": len(prompt),
                "cold_warm_greedy_match": cold["output_sha256"] == warm["output_sha256"],
                "full_verify_head_match": warm["output_sha256"] == exact["output_sha256"],
                "sampled_repeats_identical": len({row["output_sha256"] for row in sampled}) == 1,
                "minimum_distinct_sampled_tokens": min(
                    row["distinct_output_tokens"] for row in sampled
                ),
                "greedy_sha256": warm["output_sha256"],
            }
        )
        bench.write_json(args.root / f"{args.variant}-correctness.json", {"contexts": checks})
        flushed = request_tail_flush(chat, timeout=120)
        bench.emit(
            event="benchmark_cache_flushed",
            variant=args.variant,
            context=context,
            status=flushed["status"],
        )
    tools = tool_checks(args)
    bench.write_json(
        args.root / f"{args.variant}-correctness.json",
        {
            "contexts": checks,
            "tool_cases": len(tools),
            "tool_passed": sum(row["passed"] for row in tools),
            "scope": "Cold/warm reuse, verify-head agreement, repeated seeded sampled output "
            "and short tool correctness. Full disk/RAM/compaction qualification remains "
            "necessary before production changes.",
        },
    )
    bench.emit(
        event="variant_checks_complete",
        variant=args.variant,
        passed=sum(row["passed"] for row in tools),
        total=len(tools),
    )
    return 0


def prepare(root, after):
    original = (root / "production-launcher.sh").read_text()
    profile = bench.read_json(root / "production-profile.json")
    manifest = bench.read_json(root / "manifest.json")
    manifest.update(
        {
            "after_run": str(after),
            "worker_script": Path(__file__).name,
            "continue_on_variant_failure": True,
            "sampling": SAMPLING,
            "tool_cases": TOOL_CASES,
            "variants": variants(),
            "held_runtime_flags": {"GPU_MAX_HW_QUEUES": None, "HSA_ENABLE_MWAITX": None},
        }
    )
    bench.write_json(root / "manifest.json", manifest)
    validation = {}
    for variant, settings in variants().items():
        updated = dict(profile)
        updated["kernel_environment"] = dict(
            profile["kernel_environment"],
            **{key: value for key, value in settings.items() if key != "R4D_ATTN_FP8"},
        )
        bench.write_json(root / f"profile-{variant}.json", updated)
        path = root / f"launch-{variant}.sh"
        path.write_text(launcher(original, root, variant))
        subprocess.run(["shfmt", "-w", str(path)], check=True)
        subprocess.run(["shellcheck", str(path)], check=True)
        subprocess.run(["shfmt", "-d", str(path)], check=True)
        validation[variant] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shellcheck": True,
            "shfmt": True,
        }
    bench.write_json(root / "launcher-validation.json", validation)


FLUSH_PRODUCTION = """
import json, time
from pathlib import Path
from qwen_radiance_cache import request_tail_flush
phases = Path("/dev/shm/qwen-radiance-fair-public-phases.json")
tail = Path("/dev/shm/qwen-radiance-snapshot-tail.json")
current = json.loads(phases.read_text())
if any(row.get("phase") != "complete" for row in current.get("requests", [])):
    raise SystemExit(75)
flushed = []
for row in json.loads(tail.read_text()).get("chats", []):
    chat = {"id": row["chat_id"], "generation": row["generation"]}
    result = request_tail_flush(chat, timeout=120)
    if result["status"] not in ("flushed", "already_durable"):
        raise RuntimeError("pending chat tail was not flushed")
    flushed.append({key: result[key] for key in ("status", "tokens") if key in result})
for attempt in range(20):
    pending = json.loads(tail.read_text()).get("chats", [])
    if not pending:
        break
    time.sleep(0.25)
if pending:
    raise RuntimeError("chat tails are still pending")
active = json.loads(phases.read_text()).get("requests", [])
if any(row.get("phase") != "complete" for row in active):
    raise SystemExit(75)
print(json.dumps({"pending_chats": [], "flushed": flushed, "at": time.time()}))
"""


def queue(args):
    manifest = bench.read_json(args.root / "manifest.json")
    after = Path(manifest["after_run"])
    bench.write_json(
        args.root / "status.json", {"stage": "waiting_for_runtime_matrix", "at": time.time()}
    )
    while True:
        previous = bench.read_json(after / "status.json")
        if previous.get("production_restored"):
            if previous.get("stage") != "complete" and not (
                manifest.get("allow_preceding_variant_failures")
                and previous.get("stage") == "complete_with_failures"
            ):
                raise RuntimeError(
                    "the preceding runtime matrix failed; inspect it before continuing"
                )
            break
        time.sleep(2)
    cleanup_cache(after)
    stopped = False
    try:
        while True:
            result = subprocess.run(
                [
                    "podman",
                    "exec",
                    bench.PRODUCTION,
                    "/opt/vllm/bin/python",
                    "-c",
                    FLUSH_PRODUCTION,
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode == 75:
                bench.emit(event="waiting_for_production_idle")
                time.sleep(2)
                continue
            if result.returncode:
                raise RuntimeError("production flush failed: " + result.stderr[-1200:])
            bench.write_json(args.root / "flush.json", json.loads(result.stdout))
            break
        bench.command(["podman", "stop", "--time", "120", bench.PRODUCTION], timeout=135)
        stopped = True
        bench.emit(event="production_flushed_and_stopped")
        result = bench.run(args)
        cleanup_cache(args.root)
        return result
    except Exception:
        if (
            stopped
            and subprocess.run(["podman", "container", "exists", bench.PRODUCTION]).returncode != 0
        ):
            if subprocess.run(["podman", "container", "exists", args.root.name]).returncode == 0:
                bench.command(["podman", "stop", "--time", "120", args.root.name], timeout=135)
            with (args.root / "emergency-production-restore.log").open("w") as log:
                process = subprocess.Popen(
                    ["bash", manifest["production_launcher"]],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            bench.wait_ready("http://127.0.0.1:8080", process)
        raise


def cleanup_cache(root):
    if not re.fullmatch(r"qwen-runtime-ab-[a-z0-9]+", root.name):
        raise ValueError("unexpected experiment cache identity")
    cache = bench.CACHE_ROOT / "benchmarks" / root.name
    if cache.is_symlink():
        raise ValueError("experiment cache root is a symlink")
    if not cache.exists():
        return
    size = int(
        bench.command(["du", "-s", "--block-size=1", str(cache)], timeout=60).stdout.split()[0]
    )
    shutil.rmtree(cache)
    bench.write_json(
        root / "cache-cleanup.json", {"path": str(cache), "freed_bytes": size, "at": time.time()}
    )
    bench.emit(event="synthetic_cache_removed", root=str(root), freed_bytes=size)


def comparison(root):
    """Build a content-free comparison only after the isolated experiment has ended."""
    manifest = bench.read_json(root / "manifest.json")
    status = bench.read_json(root / "status.json")
    if status.get("stage") not in ("complete", "complete_with_failures") or not status.get(
        "production_restored"
    ):
        raise ValueError("kernel matrix has not completed and restored production")
    fixtures = manifest["private_replay_metadata"]
    failures = bench.read_json(root / "variant-failures.json")
    omissions = bench.read_json(root / "scope-change.json").get("omitted_variants", {})
    rows, images = [], set()
    for variant, flags in variants().items():
        if variant in omissions:
            rows.append(
                {"variant": variant, "status": "omitted_by_user", "reason": omissions[variant]}
            )
            continue
        if variant in failures:
            rows.append(
                {
                    "variant": variant,
                    "flags": flags,
                    "status": "startup_rejected"
                    if failures[variant].get("stage") == "startup"
                    else "worker_failed",
                    "failure": failures[variant],
                    "known_dependency": manifest.get("expected_startup_rejections", {}).get(
                        variant
                    ),
                }
            )
            continue
        configuration = bench.read_json(root / f"{variant}-configuration.json")
        if configuration.get("flags") != flags:
            raise ValueError(f"missing or mismatched configuration for {variant}")
        images.add(configuration["image"])
        checks = bench.read_json(root / f"{variant}-correctness.json")
        tools = bench.read_json(root / f"{variant}-tool-tests.json")
        if (
            checks.get("tool_cases") != TOOL_CASES
            or not isinstance(tools, list)
            or len(tools) != TOOL_CASES
            or len(checks.get("contexts", [])) != len(fixtures)
        ):
            raise ValueError(f"incomplete correctness checks for {variant}")
        for fixture in fixtures:
            context = fixture["actual_context"]
            records = [
                bench.read_json(root / f"{variant}-{context}-measured-{repeat}.json")
                for repeat in range(1, 4)
            ]
            for repeat, record in enumerate(records, 1):
                if (
                    record.get("variant") != variant
                    or record.get("context") != context
                    or record.get("trial") != f"measured-{repeat}"
                    or record.get("output_tokens") != 1024
                    or record.get("prompt_sha256") != fixture["sha256"]
                    or record.get("sampling") != SAMPLING
                ):
                    raise ValueError(f"mismatched private trial {variant}/{context}/{repeat}")
            cold = bench.read_json(root / f"{variant}-{context}-cold.json")
            rows.append(
                {
                    "variant": variant,
                    "flags": flags,
                    "status": "measured",
                    "context": context,
                    "nominal_context": fixture["nominal_context"],
                    "median": {
                        key: statistics.median(record["steady"][key] for record in records)
                        for key in records[0]["steady"]
                    },
                    "range": {
                        key: [
                            min(r["steady"][key] for r in records),
                            max(r["steady"][key] for r in records),
                        ]
                        for key in ("round_ms", "tokens_per_second")
                    },
                    "cold_prefill_seconds": cold["phase_timings_ms"]["prefill"] / 1000,
                    "output_hashes": sorted({r["output_sha256"] for r in records}),
                    "correctness": next(
                        c for c in checks["contexts"] if c["actual_context"] == context
                    ),
                    "tool_passed": sum(bool(t["passed"]) for t in tools),
                    "tool_cases": TOOL_CASES,
                    "allocator_errors": max(
                        r["allocator"]["allocation_retries"]
                        + r["allocator"]["out_of_memory_events"]
                        for r in records
                    ),
                }
            )
    if len(images) != 1:
        raise ValueError("kernel benchmark images differ")
    for row in rows:
        if row["status"] != "measured":
            continue
        baseline = next(
            r for r in rows if r["variant"] == "K000" and r.get("context") == row["context"]
        )
        row["round_time_change_percent"] = (
            row["median"]["round_ms"] / baseline["median"]["round_ms"] - 1
        ) * 100
        row["token_rate_change_percent"] = (
            row["median"]["tokens_per_second"] / baseline["median"]["tokens_per_second"] - 1
        ) * 100
        row["greedy_matches_baseline"] = (
            row["correctness"]["greedy_sha256"] == baseline["correctness"]["greedy_sha256"]
        )
    return {
        "image": images.pop(),
        "fixtures": fixtures,
        "sampling": SAMPLING,
        "rows": rows,
        "measurement": "Medians of three warmed 1,024-token trials using private historical Pi "
        "requests. Native speculative counters exclude prefill from generation timing.",
        "scope": "Two historical requests, output identity checks and short synthetic tool cases. "
        "No private responses were reviewed. This is not an old/new stack comparison or "
        "comprehensive model, snapshot or compaction correctness qualification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "queue", "run", "worker", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--variant", choices=tuple(variants()), default="K000")
    args = parser.parse_args()
    os.umask(0o077)
    bench.FLAGS = variants()
    if args.mode == "summarize":
        report = comparison(args.root)
        bench.write_json(args.root / "comparison.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.mode == "prepare":
        prepare(args.root, args.after)
        return 0
    if args.mode == "worker":
        return worker(args)
    if args.mode == "queue":
        try:
            return queue(args)
        except Exception as error:
            failure = {
                "stage": "queue_failed",
                "type": type(error).__name__,
                "message": str(error),
                "at": time.time(),
            }
            bench.write_json(args.root / "queue-failure.json", failure)
            bench.write_json(args.root / "status.json", failure)
            bench.emit(event="queue_failed", **failure)
            return 1
    return bench.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
