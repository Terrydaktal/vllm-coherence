"""The audit must refuse incomplete evidence and the wrong target inventory."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "experiments/radiance-public"


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "independent_stage_audit_test", ROOT / "probe_m1_stage_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_missing_or_empty_stages_never_pass(audit):
    report = {"errors": [], "checks": [{"passed": True}], "stage_counts": {"norms": 1}}
    assert audit.result_status(report, ["norms"]) == "SAMPLE_CHECKED"
    assert audit.result_status(report, ["norms", "attention"]) == "INCOMPLETE"
    assert audit.result_status(report, ["norms", "norms"]) == "INCOMPLETE"
    report["stage_counts"]["attention"] = 0
    assert audit.result_status(report, ["norms", "attention"]) == "INCOMPLETE"
    assert (
        audit.result_status({"errors": [], "checks": [], "stage_counts": {}}, [])
        == "INCOMPLETE"
    )


def test_mismatch_exception_and_dropped_records_are_not_passes(audit):
    report = {"errors": [], "checks": [{"passed": False}], "stage_counts": {"norms": 1}}
    assert audit.result_status(report, ["norms"]) == "FAILURES_OBSERVED"
    report["errors"].append({"stage": "norms", "message": "injected launch error"})
    assert audit.result_status(report, ["norms"]) == "INCOMPLETE"
    report["errors"].clear()
    report["checks"] = []
    assert audit.result_status(report, ["norms"]) == "INCOMPLETE"


def test_mtp_weights_cannot_replace_missing_target_projections(audit):
    # An equal-size but unrelated inventory must not count as all target layers.
    fake = {
        f"model.language_model.layers.0.fake{i}.weight_scale": "unused"
        for i in range(496)
    }
    with pytest.raises(ValueError, match="architecture"):
        audit.target_projection_names(fake)
    mtp = {f"mtp.layers.0.fake{i}.weight_scale": "unused" for i in range(496)}
    with pytest.raises(ValueError, match="found 0"):
        audit.target_projection_names(mtp)


def test_oracle_does_not_import_gpu_framework(audit):
    # Loading the audit and checking its report must not initialize a GPU.
    assert "torch" not in audit.__dict__
