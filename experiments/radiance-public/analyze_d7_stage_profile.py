"""Analyze preserved low-overhead HIP stage evidence without loading a model.

New runs contain numeric event intervals and no heavyweight profiler trace;
legacy Chrome traces remain supported for historical evidence.  Outputs contain
timings and evidence hashes, never chat contents.  Preserved source evidence
and original reports remain unchanged.
"""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from d7_stage_attribution import attribute_trace
from low_overhead_stage_events import LOW_OVERHEAD_PROFILE_SCHEMA

from qwen_r9700_lab.diagnostic_contract import (
    authenticate,
    private_json,
    seal,
    write_private,
)


def _gap_diagnostics(profile):
    """Aggregate per-round event gaps without discarding raw round evidence."""
    boundary_totals_us = defaultdict(float)
    boundary_counts = defaultdict(int)
    host_boundary_totals_us = defaultdict(float)
    host_boundary_counts = defaultdict(int)
    segment_totals_us = defaultdict(float)
    segment_counts = defaultdict(int)
    gap_rounds = 0
    max_gap_ms = 0.0
    max_overlap_ms = 0.0
    for round_record in profile.get("rounds", []):
        gap_ms = float(round_record.get("unattributed_gap_ms", 0.0))
        overlap_ms = float(round_record.get("overlap_ms", 0.0))
        max_gap_ms = max(max_gap_ms, gap_ms)
        max_overlap_ms = max(max_overlap_ms, overlap_ms)
        if gap_ms > 0:
            gap_rounds += 1
        for boundary in round_record.get("boundary_gaps", []):
            key = f"{boundary.get('from', '?')} -> {boundary.get('to', '?')}"
            value_ms = max(0.0, float(boundary.get("gap_ms", 0.0)))
            boundary_totals_us[key] += value_ms * 1_000.0
            boundary_counts[key] += 1
        for boundary in round_record.get("host_boundary_gaps", []):
            key = f"{boundary.get('from', '?')} -> {boundary.get('to', '?')}"
            value_ms = max(0.0, float(boundary.get("gap_ms", 0.0)))
            host_boundary_totals_us[key] += value_ms * 1_000.0
            host_boundary_counts[key] += 1
        for segment in round_record.get("gap_segments", []):
            key = str(segment.get("kind", "unattributed"))
            value_ms = max(0.0, float(segment.get("gap_ms", 0.0)))
            segment_totals_us[key] += value_ms * 1_000.0
            segment_counts[key] += 1
    return {
        "accounting": profile.get("gap_accounting", "legacy-stage-sum-remainder"),
        "rounds_with_positive_gap": gap_rounds,
        "max_gap_ms": max_gap_ms,
        "max_overlap_ms": max_overlap_ms,
        "boundary_gap_us": {
            key: boundary_totals_us[key] for key in sorted(boundary_totals_us)
        },
        "boundary_observations": {
            key: boundary_counts[key] for key in sorted(boundary_counts)
        },
        "host_boundary_gap_us": {
            key: host_boundary_totals_us[key]
            for key in sorted(host_boundary_totals_us)
        },
        "host_boundary_observations": {
            key: host_boundary_counts[key]
            for key in sorted(host_boundary_counts)
        },
        "segment_gap_us": {
            key: segment_totals_us[key] for key in sorted(segment_totals_us)
        },
        "segment_observations": {
            key: segment_counts[key] for key in sorted(segment_counts)
        },
    }


def _low_overhead_attribution(profile, steps):
    """Normalize the event-only profile to the historical report shape."""
    if profile.get("schema") != LOW_OVERHEAD_PROFILE_SCHEMA:
        return None
    if profile.get("steps") != steps:
        raise ValueError("low-overhead profile step count changed")
    if profile.get("pool_exhaustions"):
        raise ValueError("low-overhead event pool exhausted")
    if profile.get("coverage") != "COMPLETE":
        raise ValueError("low-overhead stage coverage is incomplete")
    stages = {}
    linked_us = 0.0
    linked_intervals = 0
    for name, row in profile.get("stages", {}).items():
        gpu_us = float(row["gpu_us"])
        intervals = int(row["intervals"])
        if gpu_us < 0 or intervals < 0:
            raise ValueError("invalid low-overhead stage evidence")
        linked_us += gpu_us
        linked_intervals += intervals
        stages[name] = {
            "kernel_us": gpu_us,
            "kernels": intervals,
            "scope_intervals": intervals,
            "scope_calls": int(row.get("scope_calls", intervals)),
            "ms_per_step": gpu_us / (1000.0 * steps),
        }
    return {
        "stages": stages,
        "linked_kernel_us": linked_us,
        "linked_kernels": linked_intervals,
        "evidence_kind": "gpu_scope_intervals",
        "unlinked_kernels": 0,
        "accounting": profile["scope_accounting"],
        "observer": profile["observer"],
        "finish_sync_ms": profile["finish_sync_ms"],
        "max_pending_rounds": profile["max_pending_rounds"],
        "round_span_us": float(profile["round_span_us"]),
        "unattributed_gap_us": float(profile["unattributed_gap_us"]),
        "overlap_us": float(profile["overlap_us"]),
        "host_round_us": float(profile.get("host_round_us", 0.0)),
        "host_observed_wait_us": float(
            profile.get("host_observed_wait_us", 0.0)
        ),
        "rounds": profile.get("rounds", []),
        "gap_diagnostics": _gap_diagnostics(profile),
    }


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
        for name in (
            Path(__file__).name,
            "d7_stage_attribution.py",
            "low_overhead_stage_events.py",
        )
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
        worker_profile = profile[0]["worker"].get("profile")
        if not isinstance(worker_profile, dict):
            raise TypeError("profile pass did not return stage evidence")
        steps = measurement["profile_steps"]
        attributed = _low_overhead_attribution(worker_profile, steps)
        if attributed is None:
            paths = list(private.glob(f"{arm}-*/profile-trace.json"))
            if len(paths) != 1:
                raise ValueError("expected exactly one retained trace or event profile per arm")
            raw = paths[0].read_bytes()
            attributed = attribute_trace(json.loads(raw)["traceEvents"])
            evidence_sha256 = hashlib.sha256(raw).hexdigest()
        else:
            evidence_sha256 = hashlib.sha256(
                json.dumps(worker_profile, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
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
        if any(
            attributed["stages"].get(stage, {}).get("kernels", 0) < steps
            for stage in required
        ):
            raise ValueError("required stage has fewer intervals than profile steps")
        if worker_profile["steps"] != steps:
            raise ValueError("profile step count changed")
        for row in attributed["stages"].values():
            row["ms_per_step"] = row["kernel_us"] / (1000 * steps)
        medians = [receipt["median_step_ms"] for receipt in clean]
        if statistics.median(medians) != summary["clean_median_step_ms"]:
            raise ValueError("clean summary disagrees with original measurements")
        clean_step_median_ms = statistics.median(medians)
        named_stage_sum_ms = attributed["linked_kernel_us"] / (1000 * steps)
        profiled_step_median_ms = statistics.median(profile[0]["step_seconds"][:steps]) * 1000
        arms[arm] = {
            "clean_step_median_ms": clean_step_median_ms,
            "clean_repeat_medians_ms": medians,
            "clean_steps": sum(len(receipt["step_seconds"]) for receipt in clean),
            "clean_prefill_seconds": [receipt["prefill_seconds"] for receipt in clean],
            "profiled_step_median_ms": profiled_step_median_ms,
            "named_stage_sum_ms": named_stage_sum_ms,
            # This mixes a control median with means from a different profile
            # window. Scope events can include dispatch gaps and recorder work.
            # Preserve the arithmetic as a diagnostic; it is not real idle time.
            "control_median_minus_stage_means_ms": clean_step_median_ms - named_stage_sum_ms,
            "production_gap_ms": None,
            "production_timing_eligible": False,
            "residual_limit": (
                "Different windows and statistics; scope intervals may include observer effects. "
                "The subtraction cannot certify production gaps or localize profiling overhead."
            ),
            "observer_delta_ms": profiled_step_median_ms - clean_step_median_ms,
            "same_final_logits": True,
            "final_logits_sha256": hashes.pop(),
            "clean_receipts": [receipt["sha256"] for receipt in clean],
            "profile_receipt": profile[0]["sha256"],
            "summary_receipt": summary["sha256"],
            "trace_sha256": evidence_sha256,
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
                "Wall times use separate unprofiled passes. Low-overhead stage times use "
                "preallocated HIP events around semantic leaf scopes and asynchronous "
                "resolution; the final drain wait is reported separately. Uncovered container "
                "work is not assigned to a child stage. Legacy reports may instead contain "
                "kernel-dispatch attribution. This is not a production graph-mode or "
                "natural-acceptance throughput result."
            ),
            "attribution_correction": (
                "New profiles use preallocated HIP events with asynchronous resolution and no "
                "per-round device synchronization; setup and final drain waits are reported "
                "separately. Legacy PyTorch CPU-event kernel associations used stale external "
                "IDs for native HIP calls and could double count GPU work, so historical traces "
                "remain on the old correlation path. The original PROFILE_INCOMPLETE report is "
                "retained."
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
