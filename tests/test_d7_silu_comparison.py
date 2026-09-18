"""Only the named numerical intervention may be normalized out of admission."""

from copy import deepcopy

import pytest
from test_conformance_execution_modes import reseal, saved_rows, side

from qwen_r9700_lab.conformance_silu_intervention import (
    admit_silu_intervention,
    compare_silu_intervention,
)
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def experiment():
    rows = saved_rows()
    a, b = side("eager", rows), side("compiled", rows)
    b["measurement"]["driver_sha256"] = "e" * 64
    b["measurement"] = reseal(b["measurement"])
    b["config"]["compilation_config"]["custom_ops"] = ["none", "+silu_and_mul"]
    b["config"] = reseal(b["config"])
    return a, b, rows


def run(a, b, rows):
    return compare_silu_intervention(
        a, b, rows, rows, base_driver="d" * 64, experiment_driver="e" * 64
    )


def test_preserves_original_receipts_and_names_normalized_admission():
    a, b, rows = experiment()
    saved = deepcopy(b)
    result = run(a, b, rows)
    assert result["decode"]["full_logits_exact"] == 320
    assert result["original_receipts"][1][1] == b["config"]["sha256"]
    assert result["normalized_admission"]["receipts"][1][1] != b["config"]["sha256"]
    assert saved == b
    admission = admit_silu_intervention(a, b, base_driver="d" * 64, experiment_driver="e" * 64)
    assert admission["schema"] == "qwen.silu-intervention-admission.v1"
    assert admission["original_receipts"][1][1] == b["config"]["sha256"]
    assert "normalized_receipts" in admission and "receipts" not in admission


@pytest.mark.parametrize("change", ["other_custom_op", "capacity", "driver", "seed"])
def test_rejects_other_changes_even_with_matching_outputs(change):
    a, b, rows = experiment()
    if change == "other_custom_op":
        b["config"]["compilation_config"]["custom_ops"].append("+rms_norm")
    elif change == "capacity":
        b["config"]["max_num_seqs"] = 1
    elif change == "seed":
        b["config"]["seed"] = 38
    elif change == "driver":
        b["measurement"]["driver_sha256"] = "x" * 64
        b["measurement"] = reseal(b["measurement"])
    b["config"] = reseal(b["config"])
    with pytest.raises(DiagnosticError):
        run(a, b, rows)
