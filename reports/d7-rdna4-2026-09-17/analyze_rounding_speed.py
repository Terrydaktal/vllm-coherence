"""Audit the four compiled ABBA timing controls; export no transcript content."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def seal(data):
    require("sha256" not in data, "document already sealed")
    encoded = json.dumps(data, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return {**data, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def authenticate(data):
    expected = seal({k: v for k, v in data.items() if k != "sha256"})["sha256"]
    require(expected == data["sha256"], "receipt hash mismatch")


def read(path, *, sealed=True):
    data = json.loads(path.read_text())
    if sealed:
        authenticate(data)
    return data


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), "metric mismatch")


def aggregate(passes):
    tokens = sum(p["after_first_tokens"] for p in passes)
    seconds = sum(p["after_first_seconds"] for p in passes)
    return {
        "responses": len(passes),
        "output_tokens": sum(p["output_tokens"] for p in passes),
        "after_first_tokens": tokens,
        "after_first_seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "median_round_ms": statistics.median(p["steady_median_step_ms"] for p in passes),
        "round_ms_range": [
            min(p["steady_median_step_ms"] for p in passes),
            max(p["steady_median_step_ms"] for p in passes),
        ],
        "mean_tokens_per_round": statistics.mean(p["steady_tokens_per_step"] for p in passes),
        "mean_prefill_to_first_seconds": statistics.mean(
            p["prefill_to_first_seconds"] for p in passes
        ),
    }


def summarize(root):
    cases = sorted(root.glob("d7-rounding-speed-20260917-001-*-casts*"))
    require(len(cases) == 4, "four completed ABBA controls required")
    controls = []
    baseline = None
    previous_release_ns = 0
    for index, (case, casts) in enumerate(zip(cases, (0, 1, 1, 0), strict=True)):
        require(case.name.endswith(f"-{index:02d}-casts{casts}"), "ABBA order mismatch")
        lane = case / "fixed-bf16"
        require(
            read(lane / "process-result.json", sealed=False)["returncode"] == 0, "worker failed"
        )
        lease = {
            name: read(case / f"gpu-lease/{name}.json")
            for name in ("requested", "acquired", "released")
        }
        require(len({v["pid"] for v in lease.values()}) == 1, "lease owner changed")
        require(
            previous_release_ns
            < lease["requested"]["requested_ns"]
            < lease["released"]["released_ns"],
            "leases overlap or run order differs",
        )
        previous_release_ns = lease["released"]["released_ns"]
        measurement = read(case / "measurement.json")
        require(
            measurement["repeats"] == 3 and measurement["prefix_tokens"] == 60000, "wrong fixture"
        )
        require(
            not measurement["correctness"] and not measurement["isolated_capture"], "timed capture"
        )
        require(not measurement["m1"] and measurement["lanes"] == ["fixed-bf16"], "wrong lane")
        config = read(lane / "requested-config.json")
        before = read(lane / "before-compile.json")
        require(before["installed_before_compile_and_capture"], "late repair installation")
        actual = read(lane / "actual-runtime.json")
        require(not actual["enforce_eager"] and actual["compilation_mode"] == 3, "not compiled")
        require(actual["graph_mode"] == "PIECEWISE", "graph mode differs")
        runtime = actual["runtime"]
        authenticate(runtime)
        require(
            runtime["compiler_settings"]["emulate_precision_casts"] is bool(casts),
            "wrong compiler setting",
        )
        flags = dict(runtime["flags"])
        require(
            flags.pop("TORCHINDUCTOR_EMULATE_PRECISION_CASTS") == str(casts), "wrong env setting"
        )
        require(flags["RADIANCE_VERIFY_HEAD"] == "0", "approximate head active")
        identity = {
            "measurement": measurement,
            "requested_config": config,
            "diagnostic_sources": actual["diagnostic_sources"],
            "capacity": actual["effective_capacity"],
            "repair": actual["repair"]["bundle"],
            "performance": actual["performance"]["manifest"],
            "packages": runtime["packages"],
            "kernel": runtime["kernel"],
            "flags_except_precision_casts": flags,
            "capture_sizes": actual["capture_sizes"],
        }
        if baseline is None:
            baseline = identity
        require(identity == baseline, "unintended configuration or source difference")
        summary = read(lane / "summary.json")
        require(
            summary["metadata"] == {k: v for k, v in actual.items() if k != "sha256"},
            "runtime receipt differs from summary",
        )
        require(
            summary["status"] == "MEASURED" and summary["replay_result"] is None, "wrong run mode"
        )
        receipts = [read(lane / f"pass-{i:02d}.json") for i in range(4)]
        require(
            sorted(lane.glob("pass-*.json")) == [lane / f"pass-{i:02d}.json" for i in range(4)],
            "extra passes",
        )
        require(summary["passes"] == [r["sha256"] for r in receipts], "incomplete pass inventory")
        require(receipts[0]["mode"] == "warmup", "missing warmup")
        passes = []
        for i, record in enumerate(receipts[1:], 1):
            require(record["mode"] == "clean" and record["index"] == i, "wrong timing pass")
            require(record["execution_mode"] == "compiled", "eager timing pass")
            require(record["fixture"] == measurement["fixture"], "fixture mismatch")
            require(record["finish_reason"] == "stop", "completion hit output ceiling")
            require(record["observation"]["forced"] is None, "forced tokens in natural run")
            counts = record["observation"]["observation"]["counts"]
            require(counts["target_graph_replays"] > 0, "graph replay unobserved")
            rounds = record["steps_after_first"]
            require(len(rounds) > 8, "insufficient steady rounds")
            require(
                all(t["seconds"] > 0 and 0 <= t["tokens"] <= 8 for t in rounds), "invalid round"
            )
            require(
                sum(t["tokens"] for t in rounds) == record["after_first_tokens"], "token accounting"
            )
            close(
                record["steady_median_step_ms"],
                1000 * statistics.median(t["seconds"] for t in rounds[8:]),
            )
            close(
                record["steady_tokens_per_step"], statistics.mean(t["tokens"] for t in rounds[8:])
            )
            close(
                record["tokens_per_second"],
                record["after_first_tokens"] / record["after_first_seconds"],
            )
            fields = (
                "index",
                "output_tokens",
                "finish_reason",
                "after_first_tokens",
                "after_first_seconds",
                "tokens_per_second",
                "prefill_to_first_seconds",
                "steady_median_step_ms",
                "steady_tokens_per_step",
                "output_sha256",
                "steps_after_first",
            )
            passes.append(
                {
                    **{k: record[k] for k in fields},
                    "source_receipt": record["sha256"],
                    "target_graph_replays": counts["target_graph_replays"],
                }
            )
        metrics = aggregate(passes)
        close(summary["tokens_per_second"], metrics["tokens_per_second"])
        close(summary["median_step_ms"], metrics["median_round_ms"])
        close(summary["tokens_per_step"], metrics["mean_tokens_per_round"])
        controls.append(
            {
                "case": case.name,
                "precision_casts": casts,
                "passes": passes,
                "summary_receipt": summary["sha256"],
                "runtime_receipt": actual["sha256"],
                "lease_seconds": (
                    lease["released"]["released_ns"] - lease["requested"]["requested_ns"]
                )
                / 1e9,
                "metrics": metrics,
            }
        )
    arms = {}
    for casts, name in ((0, "before"), (1, "aligned")):
        selected = [c for c in controls if c["precision_casts"] == casts]
        arms[name] = aggregate([p for c in selected for p in c["passes"]])
        arms[name]["same_output_hashes_across_restarts"] = [
            a["output_sha256"] == b["output_sha256"]
            for a, b in zip(selected[0]["passes"], selected[1]["passes"], strict=True)
        ]
    deltas = {
        key: 100 * (arms["aligned"][key] / arms["before"][key] - 1)
        for key in ("tokens_per_second", "median_round_ms", "mean_tokens_per_round")
    }
    return seal(
        {
            "schema": "urn:qwen:d7-rounding-speed:1",
            "status": "MEASURED",
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fixture": baseline["measurement"]["fixture"],
            "prefix_tokens": 60000,
            "driver_sha256": baseline["measurement"]["driver_sha256"],
            "binding": baseline["measurement"]["binding"],
            "repair": baseline["repair"],
            "performance": baseline["performance"],
            "diagnostic_sources": baseline["diagnostic_sources"],
            "capacity": baseline["capacity"],
            "packages": baseline["packages"],
            "order": [0, 1, 1, 0],
            "sampling": {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 20,
                "seeds_each_case": [118, 119, 120],
                "max_tokens": 4096,
                "ignore_eos": False,
            },
            "execution": {
                "compiled": True,
                "graph_mode": "PIECEWISE",
                "head": "full BF16",
                "gpu_profile": False,
                "tensor_capture": False,
                "forced_tokens": False,
            },
            "checks": {
                "all_receipt_hashes_valid": True,
                "config_and_source_binding_match": True,
                "only_compiler_precision_cast_flag_differs": True,
                "all_natural_completions_included": True,
                "all_metrics_recomputed": True,
                "graph_replays_observed_each_pass": True,
                "abba_lease_order_verified": True,
            },
            "method": "Pool post-first tokens/time. Median of per-response steady round medians; "
            "exclude first eight post-first rounds from that median. Startup, warmup, "
            "cold prefill, HTTP, Pi, tools and snapshots excluded. Lightweight dispatch "
            "counters remain active. Natural trajectories may differ; token rate also "
            "depends on tokens accepted per round.",
            "controls": controls,
            "arms": arms,
            "percent_change": deltas,
            "scope": "Brief compiled-engine performance controls; no new correctness "
            "qualification. "
            "The native RoPE rounding intervention affects eager execution and is not timed here.",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"arms": report["arms"], "percent_change": report["percent_change"]}, indent=2)
    )


if __name__ == "__main__":
    main()
