"""Reattribute preserved native HIP traces without loading a model or using a GPU.

Outputs contain timings, kernel names and evidence hashes, never chat contents.
The source trace and original (possibly incomplete) reports remain unchanged.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from d7_stage_attribution import attribute_trace

from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def analyze(run, private, baseline):
    measurement = private_json(run / "measurement.json")
    authenticate(measurement)
    original = private_json(baseline / "measurement.json")
    authenticate(original)
    for key in ("fixture", "binding", "timed_steps", "warmup_steps", "clean_repeats"):
        if original[key] != measurement[key]:
            raise ValueError("reused baseline configuration changed")
    summaries = {arm: private_json(run / f"{arm}-summary.json") for arm in ("old", "fixed")}
    for summary in summaries.values():
        authenticate(summary)
        if summary["fixture"] != measurement["fixture"]:
            raise ValueError("profile fixture identity changed")
    source_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (Path(__file__).name, "d7_stage_attribution.py")
    }
    arms = {}
    for arm, summary in summaries.items():
        clean_root = baseline if arm == "old" else run
        clean = [
            private_json(p)
            for p in sorted(clean_root.glob(f"{arm}-pass-*.json"))
            if private_json(p)["mode"] == "clean"
        ]
        profile = [
            private_json(p)
            for p in sorted(run.glob(f"{arm}-pass-*.json"))
            if private_json(p)["mode"] == "profile"
        ]
        if len(clean) != measurement["clean_repeats"] or len(profile) != 1:
            raise ValueError("incomplete clean or profile passes")
        hashes = set()
        for receipt in [*clean, *profile]:
            authenticate(receipt)
            authenticate(receipt["worker"])
            if receipt["sha256"] not in summary["passes"]:
                raise ValueError("pass was not part of the authenticated arm summary")
            hashes.add(receipt["worker"]["final_logits_sha256"])
        if len(hashes) != 1 or not summary["same_final_logits_all_passes"]:
            raise ValueError("clean/profile results changed")
        paths = list(private.glob(f"{arm}-*/profile-trace.json"))
        if len(paths) != 1:
            raise ValueError("expected exactly one retained trace per arm")
        raw = paths[0].read_bytes()
        attributed = attribute_trace(json.loads(raw)["traceEvents"])
        if attributed["unlinked_kernels"]:
            raise ValueError("GPU launches could not all be linked")
        required = {
            "gdn_convolution",
            "gdn_recurrence_gates_state",
            "gdn_output_norm",
            "kv_cache_and_attention",
            "target_vocabulary_head",
            "drafter",
        }
        if any(not attributed["stages"].get(stage, {}).get("kernels") for stage in required):
            raise ValueError("required stage has no GPU dispatches")
        steps = measurement["profile_steps"]
        if profile[0]["worker"]["profile"]["steps"] != steps:
            raise ValueError("profile step count changed")
        for row in attributed["stages"].values():
            row["ms_per_step"] = row["kernel_us"] / (1000 * steps)
        medians = [receipt["median_step_ms"] for receipt in clean]
        if statistics.median(medians) != summary["clean_median_step_ms"]:
            raise ValueError("clean summary disagrees with original measurements")
        arms[arm] = {
            "clean_step_median_ms": statistics.median(medians),
            "clean_repeat_medians_ms": medians,
            "clean_steps": sum(len(receipt["step_seconds"]) for receipt in clean),
            "clean_prefill_seconds": [receipt["prefill_seconds"] for receipt in clean],
            "profiled_step_median_ms": statistics.median(profile[0]["step_seconds"][:steps]) * 1000,
            "same_final_logits": True,
            "final_logits_sha256": hashes.pop(),
            "clean_receipts": [receipt["sha256"] for receipt in clean],
            "profile_receipt": profile[0]["sha256"],
            "summary_receipt": summary["sha256"],
            "trace_sha256": hashlib.sha256(raw).hexdigest(),
            "profile_steps": steps,
            "profile": attributed,
        }
    old, fixed = (arms[arm]["clean_step_median_ms"] for arm in ("old", "fixed"))
    return seal(
        {
            "status": "MEASURED",
            "measurement": measurement["sha256"],
            "fixture": measurement["fixture"],
            "source_sha256": source_hashes,
            "whole_step_increase_ms": fixed - old,
            "whole_step_increase_percent": 100 * (fixed / old - 1),
            "fixed_work_throughput_change_percent": 100 * (old / fixed - 1),
            "arms": arms,
            "scope": (
                "Eager TP1, 60K private Pi prefix, eight forced accepted positions per step. "
                "Wall times use separate unprofiled passes. Stage times sum actual HIP kernel "
                "durations across all layers in eight profiled steps. Host elapsed times include "
                "waits and overlap GPU work; do not add them to GPU times. Not a production "
                "graph-mode or natural-acceptance throughput result."
            ),
            "attribution_correction": (
                "Original PyTorch CPU-event kernel associations used stale external IDs for "
                "native HIP calls and could double count GPU work. Unique runtime correlations "
                "plus host launch scopes recover the actual stage from unchanged raw traces. "
                "The original PROFILE_INCOMPLETE report is retained."
            ),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "private", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.private, args.baseline)
    write_private(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "whole_step_increase_ms",
                    "whole_step_increase_percent",
                    "sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
