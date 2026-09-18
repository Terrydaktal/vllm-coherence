"""Admit exactly one declared compiler custom-op intervention, preserving receipts."""

from copy import deepcopy

from qwen_r9700_lab.conformance_execution_modes import admit_pair, compare_pair
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, seal


def normalized_candidate(left, right, *, base_driver, experiment_driver):
    for side in (left, right):
        for value in side.values():
            authenticate(value)
    if (
        left["measurement"]["driver_sha256"] != base_driver
        or right["measurement"]["driver_sha256"] != experiment_driver
    ):
        raise DiagnosticError("SiLU experiment driver identity differs from its reviewed source")
    if right["measurement"]["execution_mode"] not in {"compiled", "compiled-no-graphs"}:
        raise DiagnosticError("SiLU intervention must execute the compiled candidate")
    if "custom_ops" in left["config"]["compilation_config"]:
        raise DiagnosticError("SiLU control has an additional explicit custom-op selection")
    if right["config"]["compilation_config"].get("custom_ops") != ["none", "+silu_and_mul"]:
        raise DiagnosticError("SiLU experiment changed more than its declared custom-op selection")

    # Compare configurations modulo ONLY the declared intervention. These are
    # normalized views, never claimed to be the original execution receipts.
    normalized = deepcopy(right)
    normalized["measurement"].pop("sha256")
    normalized["measurement"]["driver_sha256"] = base_driver
    normalized["measurement"] = seal(normalized["measurement"])
    normalized["config"].pop("sha256")
    normalized["config"]["compilation_config"].pop("custom_ops")
    normalized["config"] = seal(normalized["config"])
    return normalized


def admit_silu_intervention(left, right, *, base_driver, experiment_driver):
    normalized = normalized_candidate(
        left, right, base_driver=base_driver, experiment_driver=experiment_driver
    )
    checked = admit_pair(left, normalized)
    checked.pop("sha256")
    checked["normalized_receipts"] = checked.pop("receipts")
    checked["original_receipts"] = [
        [side[k]["sha256"] for k in ("measurement", "config", "runtime", "pass")]
        for side in (left, right)
    ]
    checked["schema"] = "qwen.silu-intervention-admission.v1"
    checked["declared_change"] = {
        "custom_ops": ["none", "+silu_and_mul"],
        "base_driver": base_driver,
        "experiment_driver": experiment_driver,
    }
    checked["scope"] = (
        "Controlled metadata comparison modulo the explicitly declared SiLU implementation change"
    )
    return seal(checked)


def compare_silu_intervention(
    left, right, left_rows, right_rows, *, base_driver, experiment_driver
):
    normalized = normalized_candidate(
        left, right, base_driver=base_driver, experiment_driver=experiment_driver
    )
    checked = compare_pair(left, normalized, left_rows, right_rows)
    return seal(
        {
            "schema": "qwen.silu-intervention-comparison.v1",
            "status": "COMPARED_DECLARED_INTERVENTION",
            "declared_change": {
                "custom_ops": ["none", "+silu_and_mul"],
                "base_driver": base_driver,
                "experiment_driver": experiment_driver,
            },
            "original_receipts": [
                [side[k]["sha256"] for k in ("measurement", "config", "runtime", "pass")]
                for side in (left, right)
            ],
            "normalized_admission": checked["admission"],
            "row_sources": checked["row_sources"],
            "decode": checked["decode"],
            "prefill": checked["prefill"],
            "scope": (
                "Controlled output comparison after one explicit SiLU implementation change. "
                "Normalized admission checks all other recorded settings; "
                "no universal or isolated-state claim."
            ),
        }
    )
