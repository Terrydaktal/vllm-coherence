"""Bind fresh original/final stage traces to observed compiled runtime receipts."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from analyze_rounding_speed import authenticate, close, read, require, seal


def analyze(root):
    raw = root / "profile-raw/qualification/diagnostics"
    arms = {}
    common = None
    for arm, lane, casts in (("original", "old-bf16", 0), ("final", "fixed-bf16", 1)):
        run = raw / ("d7-two-fix-profile-20260917-001-" + arm)
        directory = run / lane
        measurement = read(run / "measurement.json")
        actual = read(directory / "actual-runtime.json")
        summary = read(directory / "summary.json")
        runtime = actual["runtime"]
        authenticate(runtime)
        require(measurement["prefix_tokens"] == 60000, "wrong prefix length")
        require(not measurement["correctness"] and not measurement["m1"], "wrong benchmark mode")
        require(measurement["repeats"] == 3, "three natural controls required")
        require(not actual["enforce_eager"] and actual["compilation_mode"] == 3, "not compiled")
        require(actual["graph_mode"] == "PIECEWISE", "not release graph replay")
        require(actual["capture_sizes"] == [1, 2, 4, 8], "capture sizes changed")
        require(
            runtime["compiler_settings"]["emulate_precision_casts"] is bool(casts), "wrong casts"
        )
        require(runtime["flags"]["RADIANCE_VERIFY_HEAD"] == "0", "approximate target head")
        flags = dict(runtime["flags"])
        require(
            flags.pop("TORCHINDUCTOR_EMULATE_PRECISION_CASTS") == str(casts),
            "declared cast flag differs from the observed compiler setting",
        )
        identity = {
            "fixture": measurement["fixture"],
            "binding": measurement["binding"],
            "driver": measurement["driver_sha256"],
            "capacity": actual["effective_capacity"],
            "packages": runtime["packages"],
            "kernel": runtime["kernel"],
            "diagnostic_sources": actual["diagnostic_sources"],
            "flags_except_casts": flags,
        }
        if common is None:
            common = identity
        require(identity == common, "unintended identity difference")
        require((actual["repair"] is None) == (arm == "original"), "wrong repair arm")
        if arm == "final":
            require(actual["performance"] is not None, "performance repairs missing")
        require(
            summary["metadata"] == {k: v for k, v in actual.items() if k != "sha256"},
            "runtime changed",
        )
        passes = [read(directory / f"pass-{i:02d}.json") for i in range(5)]
        require(summary["passes"] == [p["sha256"] for p in passes], "pass inventory changed")
        require(
            [p["mode"] for p in passes] == ["warmup", "clean", "clean", "clean", "profile"],
            "wrong passes",
        )
        clean = passes[1:4]
        for p in clean:
            require(p["finish_reason"] == "stop", "natural control was truncated")
            require(
                p["fixture"] == common["fixture"] and p["execution_mode"] == "compiled",
                "wrong control",
            )
            require(p["observation"]["forced"] is None, "forced tokens in speed control")
            require(
                p["observation"]["observation"]["counts"]["target_graph_replays"] > 0,
                "no graph replay",
            )
            close(
                p["steady_median_step_ms"],
                1000 * statistics.median(s["seconds"] for s in p["steps_after_first"][8:]),
            )
        close(
            summary["median_step_ms"], statistics.median(p["steady_median_step_ms"] for p in clean)
        )
        close(
            summary["tokens_per_second"],
            sum(p["after_first_tokens"] for p in clean)
            / sum(p["after_first_seconds"] for p in clean),
        )
        profile_path = root / (arm + "-compiled-profile.json")
        profile = read(profile_path)
        dispatch_path = root / (arm + "-compiled-dispatches.json")
        dispatch = json.loads(dispatch_path.read_text())
        require(
            dispatch["profile_file_sha256"]
            == hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            "profile binding changed",
        )
        require(dispatch["trace_sha256"] == profile["trace_sha256"], "wrong trace")
        require(profile["attribution"]["unlinked_kernels"] == 0, "unlinked GPU events")
        require(
            dispatch["profiled_target_graph_launches_per_round"] == [65] * 8, "missing graph replay"
        )
        arms[arm] = {
            "runtime_receipt": actual["sha256"],
            "summary_receipt": summary["sha256"],
            "profile_pass": passes[4]["sha256"],
            "trace_sha256": profile["trace_sha256"],
            "dispatch_file_sha256": hashlib.sha256(dispatch_path.read_bytes()).hexdigest(),
            "precision_casts": bool(casts),
            "repair": actual["repair"]["bundle"] if actual["repair"] else None,
            "performance": actual["performance"]["manifest"] if actual["performance"] else None,
            "natural_controls": [
                {
                    k: p[k]
                    for k in (
                        "sha256",
                        "output_tokens",
                        "finish_reason",
                        "after_first_tokens",
                        "after_first_seconds",
                        "steady_median_step_ms",
                        "steady_tokens_per_step",
                        "tokens_per_second",
                    )
                }
                for p in clean
            ],
            "median_round_ms": summary["median_step_ms"],
            "pooled_tokens_per_second": summary["tokens_per_second"],
            "complete_profile_rounds": dispatch["complete_target_inventory_rounds"],
            "inventory_gaps": dispatch["inventory_gaps"],
            "target_graph_launches": dispatch["profiled_target_graph_launches_per_round"],
        }
    paired = sorted(
        set(arms["original"]["complete_profile_rounds"])
        & set(arms["final"]["complete_profile_rounds"])
    )
    require(len(paired) >= 6, "insufficient complete paired rounds")
    return seal(
        {
            "schema": "qwen.two-fix-compiled-profile-audit.v1",
            "status": "MEASURED",
            "common": common,
            "arms": arms,
            "paired_profile_rounds": paired,
            "selection": (
                "Complete target inventories in both traces, selected without examining timings."
            ),
            "scope": (
                "Original compiled M8 versus final compiled M8 after both major fixes. "
                "Natural speed controls and GPU profiles are separate passes. "
                "Not a 60K-output speed run."
            ),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": result["status"], "paired_profile_rounds": result["paired_profile_rounds"]}
        )
    )
