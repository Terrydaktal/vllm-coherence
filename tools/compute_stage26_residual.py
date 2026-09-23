#!/usr/bin/env python3
"""Compute the separate uninstrumented-round residual for stage 26.

An arithmetic remainder is not, by itself, a measurement of production gaps.
Qualification requires a matched natural workload, kernel activity timestamps,
overlap accounting and an independently observed profiler-on/off comparison.
Archived or incompletely bound inputs remain diagnostic and cannot supply a
published runtime-gap value.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "urn:qwen:stage26-uninstrumented-residual-v1"
DEFAULT_CONTEXT_ORDER = ("0K", "60K", "200K")


def qualification_issues(profile: dict[str, Any], control: dict[str, Any]) -> list[str]:
    """Separate evidence eligibility from the much weaker subtraction identity.

    These requirements exclude known measurement contamination. Passing them
    does not prove that tracing has zero indirect effect on every kernel.
    """
    issues = []
    if profile.get("production_timing_eligible") is not True:
        issues.append("stage profile is not qualified for production timing")
    if profile.get("observer_effect", {}).get("first_use_triton_jit_observed") is not False:
        issues.append("a compilation-free measurement window has not been established")
    stage_contract = profile.get("timing_contract", {})
    if stage_contract.get("stage_metric") != "gpu_activity_duration":
        issues.append("stage values are not bound to GPU activity start/end timestamps")
    for key in ("cpu_scope_time_included", "per_stage_event_probes", "added_synchronization"):
        if stage_contract.get(key) is not False:
            issues.append(f"stage timing must exclude {key}")
    control_contract = control.get("timing_contract", {})
    for key in ("stage_profiler_enabled", "forced_replay_hooks", "added_synchronization"):
        if control_contract.get(key) is not False:
            issues.append(f"control timing must exclude {key}")
    for key in ("round_boundary", "workload_schedule_sha256", "runtime_artifact_sha256"):
        expected = stage_contract.get(key)
        if not expected or expected != control_contract.get(key):
            issues.append(f"independently recorded {key} does not match")
    comparison = profile.get("observer_comparison", {})
    # A positive measured delta must not be silently subtracted from individual
    # stages: a total delta does not locate the cost or rule out cancellation.
    if comparison.get("status") != "measured":
        issues.append("matched profiler-on/off observer effect has not been measured")
    else:
        for context in profile.get("context_order", DEFAULT_CONTEXT_ORDER):
            evidence = comparison.get("contexts", {}).get(context, {})
            for name in ("profiled_round_ms", "control_before_round_ms", "control_after_round_ms"):
                samples = evidence.get(name)
                if (
                    not isinstance(samples, list)
                    or not samples
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0
                        for value in samples
                    )
                ):
                    issues.append(f"{context}: observer comparison lacks valid {name} samples")
    return issues


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def compute(profile: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """Return validated stage-26 residuals for matching profile/control data."""

    benchmark = profile.get("stage26_benchmark", {})
    if benchmark.get("schema") not in (None, SCHEMA):
        raise ValueError("stage-26 benchmark schema is not supported")
    if control.get("schema") != SCHEMA:
        raise ValueError("uninstrumented control has the wrong schema")
    # The profile must carry its own identity.  Falling back to the control
    # artifact here would make a copied control self-authenticate and would
    # no longer prove that the two runs used the same build and fixture.
    profile_identity = profile.get("stage26_execution")
    if not isinstance(profile_identity, dict):
        # Keep the compact unit-test/public schema useful while refusing the
        # old nested-control-only shape in real evidence files.
        profile_identity = {
            key: profile.get(key)
            for key in ("measurement_commit", "execution_identity", "fixture_identity")
        }
    for key in ("measurement_commit", "execution_identity", "fixture_identity"):
        expected = profile_identity.get(key)
        actual = control.get(key)
        if expected is None or actual is None:
            raise ValueError(f"stage-26 {key} is required for an execution match")
        if actual != expected:
            raise ValueError(f"stage-26 {key} does not match")
    order = tuple(profile.get("context_order", DEFAULT_CONTEXT_ORDER))
    if tuple(control.get("context_order", order)) != order:
        raise ValueError("stage-26 context order does not match")
    profile_contexts = profile.get("contexts")
    control_contexts = control.get("contexts")
    if not isinstance(profile_contexts, dict) or not isinstance(control_contexts, dict):
        raise ValueError("both profile and control must contain context records")

    issues = qualification_issues(profile, control)
    result_contexts: dict[str, dict[str, Any]] = {}
    for context in order:
        profiled = profile_contexts.get(context)
        uninstrumented = control_contexts.get(context)
        if not isinstance(profiled, dict) or not isinstance(uninstrumented, dict):
            raise ValueError(f"missing stage-26 context: {context}")
        # The published historical profile uses ``stages_ms`` while the
        # small unit-test fixture uses the shorter ``stages`` spelling.  Both
        # describe the same 25 named stage means; accepting the canonical
        # published spelling keeps the validation tool on the real artifact
        # rather than requiring a lossy hand-written wrapper.
        stages = profiled.get("stages")
        if stages is None:
            stages = profiled.get("stages_ms")
        if not isinstance(stages, dict) or len(stages) != 25:
            raise ValueError(f"{context} must contain exactly 25 named stages")
        stage_sum = sum(_finite_number(value, f"{context} stage") for value in stages.values())
        timing = profiled.get("round_timing_ms", {})
        busy = timing.get("gpu_busy_ms")
        overlap = timing.get("gpu_overlap_ms")
        if busy is None or overlap is None:
            issues.append(f"{context}: GPU interval union/overlap is not recorded")
            busy = None
        else:
            busy = _finite_number(busy, f"{context} GPU busy union")
            overlap = _finite_number(overlap, f"{context} GPU overlap")
            if not math.isclose(stage_sum - overlap, busy, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{context} GPU interval union does not reconcile with stage durations")
        full_round = _finite_number(
            uninstrumented.get("full_uninstrumented_round_ms"),
            f"{context} full uninstrumented round",
        )
        residual = full_round - stage_sum
        occupied = stage_sum if busy is None else busy
        if full_round < occupied - 1e-6:
            raise ValueError(
                f"{context} control is incompatible with the instrumented profile: "
                f"{full_round:.6f} ms < {occupied:.6f} ms occupied GPU time"
            )
        result_contexts[context] = {
            "full_uninstrumented_round_ms": full_round,
            "instrumented_stage_sum_ms": stage_sum,
            "residual_ms": residual,
            "gpu_busy_union_ms": busy,
            "union_corrected_difference_ms": None if busy is None else full_round - busy,
            "round_samples": uninstrumented.get("round_samples", []),
        }
    return {
        "schema": SCHEMA,
        "status": "diagnostic_only" if issues else "matched_estimate",
        "qualification_issues": issues,
        "real_gap_ms": None,
        "zero_observer_effect_proven": False,
        "context_order": list(order),
        "contexts": result_contexts,
        "definition": (
            "Historical arithmetic difference: control round minus stage-duration sum. "
            "Only union_corrected_difference_ms accounts for concurrent GPU work. Neither "
            "difference proves profiler-free production gaps; matched_estimate is an empirical estimate."
        ),
        "private_chat_text_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--diagnostic-only", action="store_true",
        help="Allow arithmetic on unqualified historical evidence; never labels it as real gaps.",
    )
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if "stage_profile_2k" in profile:
        profile = profile["stage_profile_2k"]
    control = json.loads(args.control.read_text())
    result = compute(profile, control)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    if result["status"] == "diagnostic_only" and not args.diagnostic_only:
        raise SystemExit("Unqualified stage/control evidence; no production gap measurement was produced")


if __name__ == "__main__":
    main()
