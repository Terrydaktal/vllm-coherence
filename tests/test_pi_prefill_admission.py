"""Incomplete or changed native evidence must not activate a performance path."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "experiments/radiance-public")
    )
    return importlib.import_module("pi_prefill_admission")


def write_report(module, tmp_path, report, names):
    path = tmp_path / "native-evidence.json"
    path.write_text(json.dumps(report))
    return {
        "qualification": str(path),
        "qualification_sha256": module.digest(path),
        "sources": {
            name: module.digest(Path(module.__file__).with_name(name)) for name in names
        },
    }


def gdn_report(module):
    return {
        "status": "SAMPLE_CHECKED",
        "sites": 48,
        "input_rows_per_site": 1000,
        "source_sha256": module.digest(
            Path(module.__file__).with_name("stock_gdn_norm_quant.py")
        ),
        "negative_control_detected": True,
        "checks": [
            {"site": site, "width": width, "matching_rows": 1000}
            for site in range(48)
            for width in (1, 8)
        ]
        + [
            {"prefill_width": width, "equal": True}
            for width in (9, 320, 1000, 1648, 2048)
        ],
    }


@pytest.mark.parametrize(
    "fault",
    [
        "missing-site",
        "missing-width",
        "short-sample",
        "failed-prefill",
        "negative-control",
        "failed-status",
    ],
)
def test_gdn_requires_every_site_width_and_fault_control(admission, tmp_path, fault):
    report = gdn_report(admission)
    names = ("stock_gdn_norm_quant.py", "probe_stock_gdn_norm_quant.py")
    entry = write_report(admission, tmp_path, report, names)
    admission.gdn_evidence(entry)
    if fault == "missing-site":
        report["checks"] = report["checks"][2:]
    elif fault == "missing-width":
        report["checks"].pop(1)
    elif fault == "short-sample":
        report["checks"][0]["matching_rows"] = 320
    elif fault == "failed-prefill":
        report["checks"][-1]["equal"] = False
    elif fault == "negative-control":
        report["negative_control_detected"] = False
    else:
        report["status"] = "FAILED"
    with pytest.raises(ValueError):
        admission.gdn_evidence(write_report(admission, tmp_path, report, names))


def test_changed_report_or_source_cannot_reuse_qualification(admission, tmp_path):
    entry = write_report(
        admission, tmp_path, gdn_report(admission), ("stock_gdn_norm_quant.py",)
    )
    bad = dict(entry, sources={"stock_gdn_norm_quant.py": "0" * 64})
    with pytest.raises(ValueError, match="implementation changed"):
        admission.gdn_evidence(bad)
    Path(entry["qualification"]).write_text("{}")
    with pytest.raises(ValueError, match="qualification changed"):
        admission.gdn_evidence(entry)


@pytest.mark.parametrize("row_invariant", [False, True])
def test_norm_evidence_must_use_the_selected_arithmetic(
    admission, tmp_path, row_invariant
):
    report = gdn_report(admission)
    report["row_invariant"] = row_invariant
    entry = write_report(admission, tmp_path, report, ("stock_gdn_norm_quant.py",))
    entry["row_invariant"] = row_invariant
    admission.gdn_evidence(entry)
    entry["row_invariant"] = not row_invariant
    with pytest.raises(ValueError, match="different normalization contract"):
        admission.gdn_evidence(entry)


def test_scan_requires_actual_adapter_not_just_alternative_tiles(admission, tmp_path):
    names = ("optimized_prefill_scan.py", "probe_prefill_tiles.py")
    report = {
        "status": "SAMPLE_CHECKED",
        "negative_control_detected": True,
        "sources": {
            n: admission.digest(Path(admission.__file__).with_name(n)) for n in names
        },
        "checks": [
            {"tokens": rows, "tile": tile, "exact": True}
            for rows in (1, 8, 64, 320, 1000, 1648, 2048)
            for tile in (8, 16, 32, "selected")
        ],
    }
    admission.scan_evidence(write_report(admission, tmp_path, report, names))
    report["checks"] = [c for c in report["checks"] if c["tile"] != "selected"]
    with pytest.raises(ValueError, match="incomplete"):
        admission.scan_evidence(write_report(admission, tmp_path, report, names))
