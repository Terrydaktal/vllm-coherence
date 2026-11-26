"""The audit must refuse incomplete evidence and the wrong target inventory."""

import importlib.util
import json
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


def test_retained_bias_binds_the_selected_adapter_sources(audit, tmp_path):
    from qwen_r9700_lab.diagnostic_contract import seal

    manifest_path = tmp_path / "manifest.json"
    sources = {}
    for name in ("gdn_prefill_direct", "prefill_bank_norm"):
        source = tmp_path / f"{name}.py"
        source.write_text(f"# {name}\n")
        sources[name] = source
    data = {
        "schema": "coherence-source-bound-combined-experiment-v1",
        "selection": ["gdn_bias"],
        "files": {source.name: audit.sha(source) for source in sources.values()},
    }
    bound = seal(data)
    manifest_path.write_text(json.dumps(bound))
    assert audit.retained_bias_sources(manifest_path) == (bound["sha256"], sources)

    manifest_path.write_text(json.dumps({**bound, "selection": []}))
    with pytest.raises(ValueError, match="changed after publication"):
        audit.retained_bias_sources(manifest_path)

    manifest_path.write_text(json.dumps(seal({**data, "selection": []})))
    with pytest.raises(ValueError, match="does not select"):
        audit.retained_bias_sources(manifest_path)

    manifest_path.write_text(json.dumps(bound))
    sources["prefill_bank_norm"].write_text("# changed\n")
    with pytest.raises(ValueError, match="source changed"):
        audit.retained_bias_sources(manifest_path)


def test_wrong_release_is_rejected_before_gpu_initialization(
    audit, monkeypatch, tmp_path, capsys
):
    (tmp_path / "optimized-release.json").write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--release",
            str(tmp_path),
            "--output",
            str(tmp_path / "output.json"),
            "--allow-gpu",
            "--expected-release-sha256",
            "0" * 64,
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        audit.main()
    assert "release manifest differs" in capsys.readouterr().err
