#!/usr/bin/env python3
"""Compute the separate uninstrumented-round residual for stage 26.

The named stage profile and the full-round control must describe the same
execution identity.  A profiled wall-time remainder is deliberately not an
acceptable substitute: profiler hooks and synchronisation change that wall
time and make the result context-dependent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "urn:qwen:stage26-uninstrumented-residual-v1"
DEFAULT_CONTEXT_ORDER = ("0K", "60K", "200K")


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
    for key in ("measurement_commit", "execution_identity", "fixture_identity"):
        expected = profile.get(key) or benchmark.get(key)
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

    result_contexts: dict[str, dict[str, Any]] = {}
    for context in order:
        profiled = profile_contexts.get(context)
        uninstrumented = control_contexts.get(context)
        if not isinstance(profiled, dict) or not isinstance(uninstrumented, dict):
            raise ValueError(f"missing stage-26 context: {context}")
        stages = profiled.get("stages")
        if not isinstance(stages, dict) or len(stages) != 25:
            raise ValueError(f"{context} must contain exactly 25 named stages")
        stage_sum = sum(_finite_number(value, f"{context} stage") for value in stages.values())
        full_round = _finite_number(
            uninstrumented.get("full_uninstrumented_round_ms"),
            f"{context} full uninstrumented round",
        )
        residual = full_round - stage_sum
        if residual < -1e-6:
            raise ValueError(
                f"{context} control is incompatible with the instrumented profile: "
                f"{full_round:.6f} ms < {stage_sum:.6f} ms"
            )
        result_contexts[context] = {
            "full_uninstrumented_round_ms": full_round,
            "instrumented_stage_sum_ms": stage_sum,
            "residual_ms": max(0.0, residual),
            "round_samples": uninstrumented.get("round_samples", []),
        }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "context_order": list(order),
        "contexts": result_contexts,
        "definition": (
            "full uninstrumented round minus the sum of the 25 named instrumented stages"
        ),
        "private_chat_text_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--output", type=Path)
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


if __name__ == "__main__":
    main()
