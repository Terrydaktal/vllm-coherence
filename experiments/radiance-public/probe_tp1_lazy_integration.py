"""Short real compiled-runtime integration check using public synthetic input.

This checks wiring, graph capture, state-window allocation and clean shutdown;
exact arithmetic is covered by the separate operator probes. It is not a new
10K-token model-equivalence or Pi snapshot qualification.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def worker(args):
    from benchmark_optimized_d7 import make_config
    from vllm import LLM, SamplingParams

    spec = json.loads(args.spec.read_text())
    config = make_config(spec, "fixed-bf16")
    llm = LLM(**config)
    try:
        prompt = "Write a Python function that returns primes below n, with an explanation."
        results = []
        for i in range(2):
            outputs = llm.generate(
                [prompt + f" Use example n={97 + i}."],
                SamplingParams(temperature=0, max_tokens=64, detokenize=False),
                use_tqdm=False,
            )
            results.append(
                {
                    "generated_tokens": len(outputs[0].outputs[0].token_ids),
                    "finish_reason": outputs[0].outputs[0].finish_reason,
                    "output_sha256": hashlib.sha256(
                        json.dumps(list(outputs[0].outputs[0].token_ids)).encode()
                    ).hexdigest(),
                }
            )
        metadata = llm.collective_rpc("qwen_optimized_metadata")[0]
        performance = metadata["performance"]
        lazy = performance["lazy_gdn"]
        lazy_expected = os.environ.get("QWEN_STOCK_GDN_LAZY") == "1"
        if (
            metadata["enforce_eager"]
            or not metadata["compilation_mode"]
            or bool(lazy) != lazy_expected
            or (lazy_expected and not lazy["decode"])
            or not performance["tp1_fp8"]["norm"]
        ):
            raise RuntimeError("compiled backport paths did not execute")
        (args.output / "result.json").write_text(
            json.dumps(
                {
                    "status": "SMOKE_CHECKED",
                    "private_chat_read": False,
                    "gdn_state_layout": "lazy" if lazy_expected else "existing",
                    "responses": results,
                    "runtime": metadata,
                    "scope": "Public input; no model-equivalence claim",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        llm.llm_engine.engine_core.shutdown()


def run(args):
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
    from qwen_r9700_lab.conformance_runtime import worker_environment
    from qwen_r9700_lab.conformance_transport import OwnedProcess

    spec = json.loads(args.spec.read_text())
    launch = json.loads((args.bundle / "launch.json").read_text())
    args.output.mkdir(mode=0o700)
    env = worker_environment(spec, args.output)
    env.update(launch["environment"])
    env["QWEN_OPTIMIZED_REPAIR"] = str(args.repair)
    env["QWEN_OPTIMIZED_STARTUP_RECEIPT"] = str(args.output / "startup.json")
    env["RADIANCE_VERIFY_HEAD"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(
        launch["pythonpath_prepend"] + os.environ["PYTHONPATH"].split(os.pathsep)
    )
    env.pop("QWEN_CONFORMANCE_NATIVE_EXPERIMENT", None)
    argv = [
        sys.executable,
        "-P",
        str(Path(__file__).resolve()),
        "worker",
        "--spec",
        str(args.spec),
        "--output",
        str(args.output),
    ]
    with (
        gpu_lease(args.output / "gpu-lease"),
        OwnedProcess(argv, args.output / "process", env=env, timeout=900) as process,
    ):
        code = process.wait()
    print(json.dumps({"returncode": code, "result": str(args.output / "result.json")}), flush=True)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    for name in ("spec", "output", "bundle", "repair"):
        parser.add_argument("--" + name, type=Path, required=name in ("spec", "output"))
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "worker":
        worker(args)
    else:
        if not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
            raise RuntimeError("shared GPU admission required")
        raise SystemExit(run(args))
