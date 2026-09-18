"""Compare HIP queue/MWAITX settings using isolated synthetic long-context work.

The host controller requires production to have been flushed and stopped. It
keeps Pi's normal API in maintenance, runs a four-arm comparison plus a repeated
baseline, and restores the unmodified production launcher even after failure.
Only numeric metrics, synthetic output hashes and experiment logs are retained.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import http.server as http_server
import json
import os
import re
import shlex
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"
PRODUCTION = "qwen38-27b-uncensored-mxfp4-public-snapshot-candidate"
FLAGS = {
    "A": {},
    "B": {"GPU_MAX_HW_QUEUES": "1"},
    "C": {"HSA_ENABLE_MWAITX": "1"},
    "D": {"GPU_MAX_HW_QUEUES": "1", "HSA_ENABLE_MWAITX": "1"},
    "A2": {},
}
METRICS = {
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
}
PHASES = Path("/dev/shm/qwen-radiance-fair-public-phases.json")
MEMORY = Path("/dev/shm/qwen-radiance-memory-v1/report.json")
CACHE_ROOT = Path("/home/lewis/.cache/qwen-radiance-public-clean-snapshot-v1")


def emit(**value):
    print(json.dumps({"at": time.time(), **value}, sort_keys=True), flush=True)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def http(endpoint, path, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        endpoint + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_metrics(body):
    result = {}
    for line in body.splitlines():
        match = re.fullmatch(r"([^ {]+)(?:\{.*\})? ([0-9.eE+\-]+)", line)
        if match and match[1] in METRICS:
            result[match[1]] = result.get(match[1], 0) + float(match[2])
    return result


def thermal():
    device = Path("/sys/class/drm/card1/device")
    monitors = list((device / "hwmon").glob("hwmon*"))
    if not monitors:
        return {}
    result = {}
    for name in (
        "temp1_input",
        "temp2_input",
        "temp3_input",
        "power1_average",
        "power1_cap",
        "fan1_input",
        "pwm1",
        "pwm1_max",
    ):
        with contextlib.suppress(OSError, ValueError):
            result[name] = int((monitors[0] / name).read_text())
    for name in ("gpu_busy_percent", "mem_info_vram_used", "mem_info_vram_total"):
        with contextlib.suppress(OSError, ValueError):
            result[name] = int((device / name).read_text())
    with contextlib.suppress(OSError, StopIteration):
        result["active_sclk"] = next(
            line.strip()
            for line in (device / "pp_dpm_sclk").read_text().splitlines()
            if "*" in line
        )
    return result


def cooldown():
    started = time.monotonic()
    while time.monotonic() - started < 180:
        values = thermal()
        if (
            values.get("temp2_input", 999999) <= 55000
            and values.get("temp1_input", 999999) <= 45000
        ):
            return {"seconds": time.monotonic() - started, "thermal": values}
        time.sleep(1)
    raise RuntimeError("GPU did not return to the comparison temperature envelope")


def phase_for(chat):
    report = read_json(PHASES)
    matches = [
        r
        for r in report.get("requests", []) + report.get("recent", [])
        if r.get("chat_id") == chat["id"] and r.get("generation") == chat["generation"]
    ]
    # One outstanding request in this isolated backend. A current row wins over recent rows.
    return next(
        (r for r in matches if r.get("phase") != "complete"), matches[-1] if matches else {}
    )


def steady_summary(samples):
    groups = {}
    for sample in samples:
        phase = sample["phase"]
        if phase.get("phase") == "generate":
            groups.setdefault(phase.get("request_id"), []).append(sample)
    if not groups:
        raise ValueError("no steady generation samples")
    window = max(groups.values(), key=len)
    first, last = window[0], window[-1]
    seconds = last["monotonic"] - first["monotonic"]
    delta = {key: last["metrics"][key] - first["metrics"][key] for key in METRICS}
    rounds = delta["vllm:spec_decode_num_drafts_total"]
    drafted = delta["vllm:spec_decode_num_draft_tokens_total"]
    if seconds < 4 or rounds < 25 or drafted <= 0 or any(v < 0 for v in delta.values()):
        raise ValueError("insufficient or reset steady generation counters")
    if delta["vllm:num_preemptions_total"] or any(
        s["metrics"]["vllm:num_requests_waiting"] or s["metrics"]["vllm:num_requests_running"] != 1
        for s in window
    ):
        raise ValueError("GPU contention/preemption invalidated the isolated trial")
    tokens = delta["vllm:generation_tokens_total"]
    return {
        "seconds": seconds,
        "rounds": rounds,
        "output_tokens": tokens,
        "round_ms": seconds * 1000 / rounds,
        "tokens_per_second": tokens / seconds,
        "tokens_per_round": tokens / rounds,
        "acceptance": delta["vllm:spec_decode_num_accepted_tokens_total"] / drafted,
        "hotspot_c_median": statistics.median(s["thermal"]["temp2_input"] / 1000 for s in window),
        "hotspot_c_max": max(s["thermal"]["temp2_input"] / 1000 for s in window),
        "power_w_median": statistics.median(s["thermal"]["power1_average"] / 1e6 for s in window),
    }


def counter_generation_window(samples):
    """Use native draft progress to locate decode on releases without phase telemetry."""
    if not samples or samples[0]["phase"].get("phase") == "generate":
        raise ValueError("a pre-generation counter sample is required")
    initial = samples[0]["metrics"]["vllm:spec_decode_num_drafts_total"]
    result = []
    for sample in samples:
        counters = sample["metrics"]
        generating = (
            counters["vllm:spec_decode_num_drafts_total"] > initial
            and counters["vllm:num_requests_running"] == 1
        )
        result.append(
            {
                **sample,
                "phase": {
                    "phase": "generate" if generating else "outside_generation",
                    "request_id": "isolated-counter-window",
                },
            }
        )
    return result


def fixture(endpoint, root, context):
    private_directory = os.environ.get("QWEN_BENCHMARK_PRIVATE_FIXTURES")
    if private_directory:
        value = read_json(Path(private_directory) / f"fixture-{context}.json")
        tokens = value.get("tokens", [])
        if (
            value.get("private_replay") is not True
            or value.get("nominal_context") != context
            or value.get("actual_context") != len(tokens)
            or not 0 < len(tokens) < 252000
            or digest(tokens) != value.get("sha256")
        ):
            raise ValueError("private replay fixture failed its identity/count/hash checks")
        return tokens
    path = root / f"fixture-{context}.json"
    if path.exists():
        value = read_json(path)
        assert len(value["tokens"]) == context
        return value["tokens"]
    prefix = (
        "<|im_start|>system\nYou are a careful software engineer. Explain your reasoning "
        "and produce complete, useful code.<|im_end|>\n<|im_start|>user\n"
        "The following synthetic repository records provide background for a design exercise.\n"
    )
    records = "\n".join(
        f"Case {i}: worker {i % 29} processed {i * 17 % 997} records; "
        f"queue capacity {16 + i % 113}; retry delay {1 + i % 7}; "
        f"expected checksum {(i * 7919) % 65521}. Preserve ordering and report failures."
        for i in range(6000)
    )
    suffix = (
        "\nEnd of reference records. Design a Python task queue with ordered results, "
        "bounded concurrency, cancellation and retry handling. Explain the design, then "
        "write a complete implementation and several examples. Work through edge cases "
        "carefully. This is a design exercise; do not call tools."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )

    def tokenize(text):
        return json.loads(http(endpoint, "/tokenize", {"model": MODEL, "prompt": text}, 60))[
            "tokens"
        ]

    pool, ending = tokenize(prefix + records), tokenize(suffix)
    assert len(pool) > context and len(ending) < context
    tokens = pool[: context - len(ending)] + ending
    write_json(path, {"synthetic": True, "tokens": tokens, "sha256": digest(tokens)})
    emit(event="fixture_ready", context=context, sha256=digest(tokens))
    return tokens


def trial(endpoint, root, variant, chat, prompt, label, count, sampling=None, cool=True):
    legacy = os.environ.get("QWEN_BENCHMARK_LEGACY_METRICS") == "1"
    before = cooldown() if cool else {"seconds": 0, "thermal": thermal()}
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": count,
        "ignore_eos": True,
        "temperature": 0,
        "top_k": 1,
        "seed": 0,
        "return_token_ids": True,
        "cache_salt": f"qwen-chat-cache-v1:{chat['id']}:{chat['generation']}",
        "kv_transfer_params": {
            "qwen_chat": chat,
            "qwen_snapshot_abi": os.environ["QWEN_RADIANCE_CACHE_ABI"],
        },
    }
    payload.update(sampling or {})
    if os.environ.get("QWEN_BENCHMARK_PRIVATE_FIXTURES") and payload["temperature"] > 0:
        payload["ignore_eos"] = False
    started = time.monotonic()
    samples = []
    if legacy:
        samples.append(
            {
                "monotonic": time.monotonic(),
                "unix_time": time.time(),
                "phase": {"phase": "before_request"},
                "metrics": parse_metrics(http(endpoint, "/metrics").decode()),
                "thermal": thermal(),
            }
        )
    emit(
        event="trial_start",
        variant=variant,
        context=len(prompt),
        trial=label,
        starting_hotspot_c=before["thermal"]["temp2_input"] / 1000,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(http, endpoint, "/v1/completions", payload, 900)
        last_progress = 0
        while not future.done():
            phase = {} if legacy else phase_for(chat)
            counters = parse_metrics(http(endpoint, "/metrics").decode())
            sample = {
                "monotonic": time.monotonic(),
                "unix_time": time.time(),
                "phase": {
                    k: phase[k]
                    for k in (
                        "request_id",
                        "phase",
                        "input_tokens",
                        "computed_tokens",
                        "cached_tokens",
                        "phase_elapsed_ms",
                    )
                    if k in phase
                },
                "metrics": counters,
                "thermal": thermal(),
            }
            samples.append(sample)
            if time.monotonic() - last_progress >= 10:
                emit(
                    event="trial_progress",
                    variant=variant,
                    context=len(prompt),
                    trial=label,
                    phase=phase.get("phase"),
                    computed_tokens=phase.get("computed_tokens"),
                    seconds=round(time.monotonic() - started, 1),
                    hotspot_c=sample["thermal"].get("temp2_input", 0) / 1000,
                )
                last_progress = time.monotonic()
            time.sleep(1)
        result = json.loads(future.result())
    elapsed = time.monotonic() - started
    ids = result["choices"][0]["token_ids"]
    assert len(ids) == count, (
        f"speed request ended after {len(ids)} tokens; expected {count} "
        f"(finish_reason={result['choices'][0].get('finish_reason')})"
    )
    final = phase_for(chat) if not legacy else {"phase": "complete"}
    timings = final.get("timings_ms", {})
    if legacy:
        timing_metrics = result.get("metrics") or {}
        timings = {
            "first_token_wait": timing_metrics.get("time_to_first_token_ms"),
            "generate": timing_metrics.get("generation_time_ms"),
            "queue": timing_metrics.get("queue_time_ms"),
        }
        if timings["first_token_wait"] is None or timings["generate"] is None:
            raise ValueError("old backend did not return per-request timing metrics")
        samples = counter_generation_window(samples)
    assert final.get("phase") == "complete", "request phase completion missing"
    allocator = read_json(MEMORY).get("allocator", {})
    assert allocator.get("out_of_memory_events") == 0, "allocator recorded an OOM"
    assert allocator.get("allocation_retries") == 0, "allocator recorded an allocation retry"
    record = {
        "variant": variant,
        "context": len(prompt),
        "trial": label,
        "output_tokens": len(ids),
        "output_sha256": digest(ids),
        "first_256_sha256": digest(ids[:256]),
        "distinct_output_tokens": len(set(ids)),
        "sampling": {
            key: payload[key] for key in ("temperature", "top_k", "seed", "top_p") if key in payload
        },
        "prompt_sha256": digest(prompt),
        "elapsed_seconds": elapsed,
        "usage": result["usage"],
        "phase_timings_ms": timings,
        "timing_source": "native counters and response metrics" if legacy else "native phases",
        "cooldown": before,
        "samples": samples,
        "allocator": allocator,
    }
    if label.startswith("measured"):
        record["steady"] = steady_summary(samples)
    write_json(root / f"{variant}-{len(prompt)}-{label}.json", record)
    emit(
        event="trial_complete",
        variant=variant,
        context=len(prompt),
        trial=label,
        output_sha256=record["output_sha256"],
        steady=record.get("steady"),
        timings_ms=timings,
    )
    return record


def worker(args):
    from qwen_radiance_cache import request_tail_flush

    records = []
    for context in (60000, 200000):
        prompt = fixture(args.endpoint, args.root, context)
        chat = {
            "id": digest([args.root.name, args.variant, context]),
            "generation": digest([args.root.name, args.variant, context, "generation"]),
            "title": f"Synthetic runtime benchmark {args.variant} {context}",
        }
        write_json(args.root / f"{args.variant}-{context}-identity.json", chat)
        trial(args.endpoint, args.root, args.variant, chat, prompt, "cold", 256)
        trial(args.endpoint, args.root, args.variant, chat, prompt, "warmup", 1024)
        records.extend(
            trial(
                args.endpoint,
                args.root,
                args.variant,
                chat,
                prompt,
                f"measured-{repeat + 1}",
                1024,
            )
            for repeat in range(3)
        )
        flushed = request_tail_flush(chat, timeout=120)
        emit(
            event="synthetic_cache_flushed",
            variant=args.variant,
            context=context,
            status=flushed["status"],
            tokens=flushed.get("tokens"),
        )
    summary = [
        {
            k: r[k]
            for k in (
                "variant",
                "context",
                "trial",
                "output_sha256",
                "steady",
                "phase_timings_ms",
                "usage",
            )
        }
        for r in records
    ]
    write_json(args.root / f"{args.variant}-summary.json", summary)
    tool_chat = {
        "id": digest([args.variant, "tools"]),
        "generation": digest([args.variant, "tools-generation"]),
    }
    tool_results = []
    for seed in range(8):
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Call record once with number 37. Do not provide a textual answer.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "record",
                        "description": "Record a number.",
                        "parameters": {
                            "type": "object",
                            "properties": {"number": {"type": "integer"}},
                            "required": ["number"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 512,
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 20,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "off"},
            "cache_salt": f"qwen-chat-cache-v1:{tool_chat['id']}:{tool_chat['generation']}",
            "kv_transfer_params": {
                "qwen_chat": tool_chat,
                "qwen_snapshot_abi": os.environ["QWEN_RADIANCE_CACHE_ABI"],
            },
        }
        value = json.loads(http(args.endpoint, "/v1/chat/completions", payload, 120))
        choice = value["choices"][0]
        calls = choice["message"].get("tool_calls", [])
        passed = (
            choice["finish_reason"] == "tool_calls"
            and len(calls) == 1
            and calls[0]["function"]["name"] == "record"
            and json.loads(calls[0]["function"]["arguments"]) == {"number": 37}
        )
        tool_results.append(
            {
                "seed": seed,
                "passed": passed,
                "finish_reason": choice["finish_reason"],
                "usage": value["usage"],
            }
        )
    request_tail_flush(tool_chat, timeout=120)
    write_json(args.root / f"{args.variant}-tool-tests.json", tool_results)
    assert all(row["passed"] for row in tool_results), "synthetic tool correctness check failed"
    emit(event="tool_tests_complete", variant=args.variant, passed=len(tool_results))
    return 0


class Maintenance(http_server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(
            {
                "error": {
                    "message": "Radiance performance testing is in progress. "
                    "Chat caches have been flushed. Retry after testing finishes.",
                    "type": "maintenance",
                    "code": "runtime_benchmark",
                }
            }
        ).encode()
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    # Deliberately do not read, retain or log request bodies from private Pi chats.
    do_POST = do_GET  # noqa: N815 - HTTP handler API

    def log_message(self, *args):
        pass


def command(argv, timeout=30):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=timeout)


def wait_ready(endpoint, process, timeout=600):
    deadline = time.monotonic() + timeout
    last_report = 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"backend launcher exited with code {process.returncode}")
        try:
            http(endpoint, "/health", timeout=2)
            return
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() - last_report > 20:
            emit(event="backend_starting", endpoint=endpoint)
            last_report = time.monotonic()
        time.sleep(1)
    raise TimeoutError("backend startup deadline exceeded")


def make_launcher(original, root, variant, cache_relative, flags=None):
    text = original
    # The source uses one leading tab; keep matching independent of indentation.
    old = '--arg root_dir "/cache/snapshots/${data_abi}/data"'
    assert text.count(old) == 1
    text = text.replace(old, f'--arg root_dir "/cache/{cache_relative}/data"')
    anchor = '\t-e QWEN_RADIANCE_CACHE_ABI="$data_abi" \\\n'
    assert text.count(anchor) == 1
    settings = FLAGS[variant] if flags is None else flags
    flags = "".join(f"\t-e {name}={value} \\\n" for name, value in settings.items())
    text = text.replace(
        anchor, anchor + flags + f"\t-v {shlex.quote(str(root))}:/benchmark:rw \\\n"
    )
    return text


def run(args):
    root = args.root.resolve()
    assert re.fullmatch(r"qwen-runtime-ab-[a-z0-9]+", root.name)
    original = (root / "production-launcher.sh").read_text()
    manifest = read_json(root / "manifest.json")
    assert hashlib.sha256(original.encode()).hexdigest() == manifest["launcher_sha256"]
    assert not read_json(root / "flush.json")["pending_chats"]
    assert subprocess.run(["podman", "container", "exists", PRODUCTION]).returncode != 0
    container = root.name
    cache_relative = f"benchmarks/{root.name}"
    cache_root = CACHE_ROOT / cache_relative
    (cache_root / "data").mkdir(mode=0o700, parents=True, exist_ok=False)
    server = http_server.ThreadingHTTPServer(("127.0.0.1", 8080), Maintenance)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    production_launcher = Path(manifest["production_launcher"])
    process = None
    failure = None
    variant_failures = {}
    flag_names = set().union(*(flags.keys() for flags in FLAGS.values()))
    try:
        for variant in FLAGS:
            path = root / f"launch-{variant}.sh"
            validation = read_json(root / "launcher-validation.json")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == validation[variant]["sha256"]
            assert validation[variant]["shellcheck"] and validation[variant]["shfmt"]
            command(["bash", "-n", str(path)])
            env = dict(
                os.environ,
                QWEN_QUALIFICATION_CONTAINER=container,
                QWEN_QUALIFICATION_PORT=str(args.port),
            )
            write_json(
                root / "status.json",
                {
                    "stage": "startup",
                    "variant": variant,
                    "flags": FLAGS[variant],
                    "at": time.time(),
                },
            )
            emit(event="variant_start", variant=variant, flags=FLAGS[variant])
            with (root / f"backend-{variant}.log").open("w") as log:
                process = subprocess.Popen(
                    ["bash", str(path)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            endpoint = f"http://127.0.0.1:{args.port}"
            try:
                wait_ready(endpoint, process)
            except Exception as error:
                if not manifest.get("continue_on_variant_failure") or (
                    manifest.get("baseline_failures_are_fatal") and variant == next(iter(FLAGS))
                ):
                    raise
                variant_failures[variant] = {
                    "stage": "startup",
                    "type": type(error).__name__,
                    "message": str(error),
                }
                write_json(root / "variant-failures.json", variant_failures)
                emit(event="variant_failed", variant=variant, **variant_failures[variant])
                if subprocess.run(["podman", "container", "exists", container]).returncode == 0:
                    command(["podman", "stop", "--time", "120", container], timeout=135)
                process.wait(timeout=15)
                process = None
                continue
            info = json.loads(command(["podman", "inspect", container]).stdout)[0]
            selected = {
                line.split("=", 1)[0]: line.split("=", 1)[1]
                for line in info["Config"]["Env"]
                if line.split("=", 1)[0] in flag_names
            }
            assert selected == FLAGS[variant], "container did not receive the requested flags"
            write_json(
                root / f"{variant}-configuration.json",
                {
                    "flags": selected,
                    "image": info["Image"],
                    "started": info["State"]["StartedAt"],
                    "launcher_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            )
            emit(event="backend_ready", variant=variant, flags=selected)
            write_json(
                root / "status.json",
                {"stage": "benchmark", "variant": variant, "flags": selected, "at": time.time()},
            )
            argv = [
                "podman",
                "exec",
                container,
                "/opt/vllm/bin/python",
                "/benchmark/" + manifest.get("worker_script", "benchmark_runtime_flags.py"),
                "worker",
                "--root",
                "/benchmark",
                "--endpoint",
                endpoint,
                "--variant",
                variant,
            ]
            with (root / f"worker-{variant}.log").open("w") as log:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
            if result.returncode:
                if not manifest.get("continue_on_variant_failure") or (
                    manifest.get("baseline_failures_are_fatal") and variant == next(iter(FLAGS))
                ):
                    raise RuntimeError(f"benchmark worker {variant} exited {result.returncode}")
                variant_failures[variant] = {"stage": "worker", "returncode": result.returncode}
                write_json(root / "variant-failures.json", variant_failures)
                emit(event="variant_failed", variant=variant, **variant_failures[variant])
            if subprocess.run(["podman", "container", "exists", container]).returncode == 0:
                command(["podman", "stop", "--time", "120", container], timeout=135)
            process.wait(timeout=10)
            process = None
            if variant not in variant_failures:
                emit(event="variant_complete", variant=variant)
    except KeyboardInterrupt:
        failure = {"type": "KeyboardInterrupt", "message": "experiment interrupted"}
        write_json(root / "failure.json", failure)
        emit(event="benchmark_cancelled")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        write_json(root / "failure.json", failure)
        emit(event="benchmark_failed", **failure)
    finally:
        if subprocess.run(["podman", "container", "exists", container]).returncode == 0:
            command(["podman", "stop", "--time", "120", container], timeout=135)
        if process is not None:
            process.wait(timeout=15)
        server.shutdown()
        server.server_close()
        assert (
            hashlib.sha256(production_launcher.read_bytes()).hexdigest()
            == manifest["launcher_sha256"]
        )
        with (root / "production-restored.log").open("w") as log:
            restored = subprocess.Popen(
                ["bash", str(production_launcher)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        wait_ready("http://127.0.0.1:8080", restored)
        write_json(
            root / "restored.json",
            {"at": time.time(), "launcher_pid": restored.pid, "original_configuration": True},
        )
        write_json(
            root / "status.json",
            {
                "stage": "cancelled"
                if failure and failure["type"] == "KeyboardInterrupt"
                else "failed"
                if failure
                else "complete_with_failures"
                if variant_failures
                else "complete",
                "variant_failures": variant_failures,
                "production_restored": True,
                "at": time.time(),
            },
        )
        emit(event="production_restored", failed=bool(failure), root=str(root))
    return int(failure is not None)


def comparison(root):
    """Summarize the complete matrix, rejecting mismatched inputs or configurations."""
    rows = []
    prompt_hashes = {context: set() for context in (60000, 200000)}
    output_hashes = {context: set() for context in prompt_hashes}
    images = set()
    tools = {}
    for variant, flags in FLAGS.items():
        configuration = read_json(root / f"{variant}-configuration.json")
        if configuration.get("flags") != flags:
            raise ValueError(f"missing or mismatched configuration for {variant}")
        images.add(configuration["image"])
        tools[variant] = read_json(root / f"{variant}-tool-tests.json")
        if not isinstance(tools[variant], list) or len(tools[variant]) != 8:
            raise ValueError(f"incomplete tool smoke checks for {variant}")
        for context in prompt_hashes:
            records = [
                read_json(root / f"{variant}-{context}-measured-{repeat}.json")
                for repeat in range(1, 4)
            ]
            for repeat, record in enumerate(records, 1):
                if (
                    record.get("variant") != variant
                    or record.get("context") != context
                    or record.get("trial") != f"measured-{repeat}"
                    or record.get("output_tokens") != 1024
                ):
                    raise ValueError(f"missing or mismatched trial {variant}/{context}/{repeat}")
                prompt_hashes[context].add(record["prompt_sha256"])
                output_hashes[context].add(record["output_sha256"])
            cold = read_json(root / f"{variant}-{context}-cold.json")
            rows.append(
                {
                    "variant": variant,
                    "flags": flags,
                    "context": context,
                    "trials": len(records),
                    "median": {
                        key: statistics.median(record["steady"][key] for record in records)
                        for key in records[0]["steady"]
                    },
                    "range": {
                        key: [
                            min(record["steady"][key] for record in records),
                            max(record["steady"][key] for record in records),
                        ]
                        for key in ("round_ms", "tokens_per_second")
                    },
                    "cold_prefill_seconds": cold["phase_timings_ms"]["prefill"] / 1000,
                    "warm_prefill_seconds_median": statistics.median(
                        record["phase_timings_ms"]["prefill"] / 1000 for record in records
                    ),
                    "cached_tokens": sorted(
                        {
                            record["usage"]["prompt_tokens_details"]["cached_tokens"]
                            for record in records
                        }
                    ),
                    "output_hashes": sorted({record["output_sha256"] for record in records}),
                    "allocator_errors": max(
                        record["allocator"]["allocation_retries"]
                        + record["allocator"]["out_of_memory_events"]
                        for record in records
                    ),
                }
            )
    if len(images) != 1 or any(len(hashes) != 1 for hashes in prompt_hashes.values()):
        raise ValueError("benchmark images or prompt tokens differ between variants")
    for row in rows:
        first = next(r for r in rows if r["variant"] == "A" and r["context"] == row["context"])
        last = next(r for r in rows if r["variant"] == "A2" and r["context"] == row["context"])
        row["round_time_change_percent_vs_A"] = (
            row["median"]["round_ms"] / first["median"]["round_ms"] - 1
        ) * 100
        row["baseline_drift_percent"] = (
            last["median"]["round_ms"] / first["median"]["round_ms"] - 1
        ) * 100
    return {
        "image": images.pop(),
        "prompt_hashes": {str(k): sorted(v) for k, v in prompt_hashes.items()},
        "identical_greedy_outputs": all(len(hashes) == 1 for hashes in output_hashes.values()),
        "all_tool_smoke_checks_passed": all(
            row["passed"] for rows in tools.values() for row in rows
        ),
        "tool_tests": tools,
        "rows": rows,
        "measurement": "Medians of three isolated warmed 1,024-token trials; round timing uses "
        "native speculative counters during generation, excluding prefill and completed time.",
        "scope": "Synthetic greedy generation and short sampled tool smoke checks. This is not "
        "an old/new stack comparison or proof of general model correctness.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--variant", choices=tuple(FLAGS), default="A")
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == "summarize":
        result = comparison(args.root)
        write_json(args.root / "comparison.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return worker(args) if args.mode == "worker" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
