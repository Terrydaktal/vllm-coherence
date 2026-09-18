"""Measure original/candidate GEMM in identical compiled models; no text output."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def worker(args):
    from benchmark_optimized_d7 import make_config
    from vllm import LLM, SamplingParams

    spec = json.loads(Path("/qualification/spec.json").read_text())
    fixture = json.loads(
        Path("/dev/shm/qwen-d7-restored-fixture-20260917/fixture.json").read_text()
    )
    config = make_config(spec, "fixed-bf16")
    config["worker_cls"] = "gemm_round_audit_worker.GemmAuditWorker"
    llm = LLM(**config)
    report = {
        "variant": args.variant,
        "target_head": args.target_head,
        "prefix_tokens": len(fixture["prefix"]),
        "fixture": fixture["sha256"],
        "config": config,
        "runs": [],
    }
    engine = llm.llm_engine
    try:
        for index, seed in enumerate((0, 17, 42, 117)):
            profiling = index == 3
            if profiling:
                llm.collective_rpc(
                    "qwen_optimized_begin", args=(str(args.output / "trace"), True, None)
                )
            params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                seed=seed,
                max_tokens=256 if profiling else 1024,
                detokenize=False,
            )
            engine.add_request(f"gemm-{index}", {"prompt_token_ids": fixture["prefix"]}, params)
            first_at = None
            first_count = received = 0
            steps = []
            while engine.has_unfinished_requests():
                before = received
                start = time.perf_counter()
                outputs = engine.step()
                elapsed = time.perf_counter() - start
                for output in outputs:
                    if output.outputs:
                        last = output.outputs[0]
                        received = len(last.token_ids)
                if first_at is None and received:
                    first_at, first_count = time.perf_counter(), received
                elif first_at is not None:
                    steps.append({"seconds": elapsed, "tokens": received - before})
                if profiling and len(steps) in (8, 16):
                    llm.collective_rpc("qwen_optimized_profile", args=(len(steps) == 8,))
            elapsed = time.perf_counter() - first_at
            if profiling:
                observed = llm.collective_rpc("qwen_optimized_finish")[0]
                if len(steps) < 16:
                    raise RuntimeError("profile request ended before the measurement window")
                report["profile"] = observed
            result = {
                "seed": seed,
                "profiled": profiling,
                "tokens": received,
                "finish_reason": last.finish_reason,
                "after_first_seconds": elapsed,
                "after_first_tokens": received - first_count,
                "after_first_tps": (received - first_count) / elapsed,
                "median_round_ms": 1000 * statistics.median(x["seconds"] for x in steps[8:]),
                "mean_round_ms": 1000 * statistics.mean(x["seconds"] for x in steps[8:]),
                "mean_tokens_per_round": statistics.mean(x["tokens"] for x in steps[8:]),
                "output_sha256": hashlib.sha256(
                    json.dumps(list(last.token_ids)).encode()
                ).hexdigest(),
            }
            report["runs"].append(result)
            report["dispatch"] = llm.collective_rpc("qwen_gemm_audit")[0]
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        engine.engine_core.shutdown()


def run(args):
    spec = json.loads(Path("/qualification/spec.json").read_text())
    bundle = Path("/qualification/preflight/tp1-lazy-backport-v1/bundle-existing-v1")
    launch = json.loads((bundle / "launch.json").read_text())
    env = os.environ.copy()
    env.update(spec["environment"])
    env.update(launch["environment"])
    env.update(
        RADIANCE_VERIFY_HEAD="1" if args.target_head == "global256" else "0",
        RADIANCE_VERIFY_HEAD_GLOBAL_TOPK="256",
        QWEN_GEMM_AUDIT_VARIANT=args.variant,
        QWEN_OPTIMIZED_REPAIR="/qualification/preflight/d7-stock-repair-bundle-003.json",
        QWEN_OPTIMIZED_STARTUP_RECEIPT=str(args.output / "startup.json"),
    )
    env.pop("QWEN_CONFORMANCE_NATIVE_EXPERIMENT", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).parent),
            str(bundle / "runtime"),
            "/qualification/preflight/d7-rotary-intervention-v1/src",
            "/qualification/preflight/d7-rotary-intervention-v1/experiments/radiance-public",
            "/qualification/preflight/stock-gdn-model-v22/runtime",
            "/qualification/preflight/stock-gdn-model-v22/experiments/radiance-public",
        ]
    )
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    with (args.output / "process.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                "--variant",
                args.variant,
                "--target-head",
                args.target_head,
                "--output",
                str(args.output),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1200,
        )
    print(json.dumps({"variant": args.variant, "returncode": result.returncode}), flush=True)
    if result.returncode:
        raise RuntimeError("diagnostic failed; private log retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=("original", "candidate"), required=True)
    parser.add_argument("--target-head", choices=("full-bf16", "global256"), default="global256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (worker if args.worker else run)(args)
