"""Matched natural-completion A/B in one disposable compiled worker.

The sealed fixture is consumed as token IDs. Reports retain numerical results,
source identities and output hashes, never prompt or response contents. This
experiment requires SpeedCandidateWorker's paired drafter graph captures; it
does not restart or modify production. ABBA order controls slow clock drift.
"""

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from benchmark_pi_coding_json_compaction import (
    CODING_PROMPT,
    _post_json,
    _render_user_turn,
    _request,
    _turn_suffix,
)
from tokenizers import Tokenizer


def hardware():
    result = {}
    for device in Path("/sys/class/drm").glob("card*/device"):
        if not (device / "device").is_file():
            continue
        if (device / "device").read_text().strip() != "0x7551":
            continue
        for name in (
            "power_dpm_force_performance_level",
            "pp_dpm_sclk",
            "pp_od_clk_voltage",
        ):
            try:
                result[name] = (device / name).read_text().strip()
            except OSError:
                pass
        for sensor in device.glob("hwmon/hwmon*"):
            for name in (
                "temp1_input",
                "temp2_input",
                "power1_average",
                "power1_cap",
                "freq1_input",
            ):
                try:
                    result[name] = int((sensor / name).read_text().strip())
                except OSError:
                    pass
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError("refusing to overwrite benchmark evidence")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def rpc(method, arguments=None):
        return _post_json(
            opener,
            args.base_url + "/collective_rpc",
            {"method": method, "args": arguments or [], "timeout": 60},
            90,
        )["results"][0]

    metadata = rpc("qwen_speed_metadata")
    if (
        metadata["execution"]["enforce_eager"]
        or not metadata["execution"]["compilation_mode"]
    ):
        raise RuntimeError("paired benchmark requires compiled execution")
    if args.component == "drafter" and not metadata["draft_attention_candidate"]:
        raise RuntimeError("qualified drafter candidate is missing")
    if args.component == "sampler" and not metadata.get("sampler_graph"):
        raise RuntimeError("experimental sampler graph is missing")
    if args.component == "gemm" and not metadata.get("target_gemm_candidate"):
        raise RuntimeError("qualified GEMM candidate is missing")
    prefix = json.loads(args.fixture.read_bytes())["prefix"]
    if len(prefix) not in (1024, 60000, 200000):
        raise ValueError("this matched experiment requires an admitted sealed fixture")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    request_args = SimpleNamespace(
        base_url=args.base_url,
        seed=113,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        abi=args.abi,
        request_timeout=1200,
        metrics_timeout=20,
        coding_min_tokens=1,
    )
    report = {
        "status": "running",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "metadata": metadata,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "seed": 113},
        "order": args.order.split(","),
        "component": args.component,
        "prefix_tokens": len(prefix),
        "arms": [],
        "privacy": "token IDs consumed privately; only numerical results and hashes retained",
    }
    if any(arm not in ("control", "candidate") for arm in report["order"]):
        raise ValueError("unknown paired arm")

    def save():
        temporary = args.output.with_suffix(".tmp")
        with temporary.open("w") as f:
            json.dump(report, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, args.output)

    def request(prompt, cap):
        rendered = _render_user_turn(
            opener, args.base_url, prompt, thinking=False, timeout=60
        )
        suffix = _turn_suffix(tokenizer, prefix, rendered, first=True)
        result, _, _ = _request(
            opener=opener,
            args=request_args,
            identity={
                "id": hashlib.sha256(str(args.output).encode()).hexdigest(),
                "generation": hashlib.sha256(b"coherence-speed-pair-generation-v1").hexdigest(),
            },
            tokenizer=tokenizer,
            prompt_tokens=prefix + suffix,
            suffix_tokens=suffix,
            stage="coding",
            max_tokens=cap,
            thinking=False,
        )
        return result

    save()
    try:
        report["warmup"] = request(
            "Write one complete Python function that implements binary search, then briefly explain its boundary conditions and complexity. Finish naturally.",
            1024,
        )
        save()
        expected = None
        for index, arm in enumerate(report["order"]):
            selector = {
                "drafter": "qwen_speed_set_draft_attention",
                "sampler": "qwen_speed_set_sampler_graph",
                "combined": "qwen_speed_set_combined",
                "gemm": "qwen_speed_set_gemm",
            }[args.component]
            rpc(selector, [arm == "candidate"])
            before = hardware()
            result = request(CODING_PROMPT, 10000)
            after = hardware()
            signature = tuple(
                result[key]
                for key in (
                    "output_sha256",
                    "generated_tokens",
                    "accepted_tokens",
                    "draft_tokens",
                )
            )
            if expected is None:
                expected = signature
            row = {
                "index": index,
                "arm": arm,
                "hardware_before": before,
                "hardware_after": after,
                "result": result,
                "same_output_and_acceptance": signature == expected,
            }
            report["arms"].append(row)
            save()
            print(
                json.dumps(
                    {
                        "arm": arm,
                        **{
                            key: result.get(key)
                            for key in (
                                "generated_tokens",
                                "mean_generation_round_ms",
                                "post_first_tokens_per_second",
                                "acceptance_rate",
                                "output_sha256",
                            )
                        },
                    }
                ),
                flush=True,
            )
            if signature != expected:
                raise AssertionError("paired arm changed output or proposal acceptance")
        report["status"] = "complete"
        save()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--abi", default=hashlib.sha256(b"isolated-speed-probe").hexdigest())
    parser.add_argument("--order", default="control,candidate,candidate,control")
    parser.add_argument(
        "--component",
        choices=("drafter", "sampler", "combined", "gemm"),
        default="drafter",
    )
    run(parser.parse_args())
