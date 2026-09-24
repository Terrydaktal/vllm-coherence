"""The tiled prefill fast path must reject partial or changed evidence."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "fault",
    [None, "row", "layout", "adapter", "binary", "source", "control", "wrapper"],
)
def test_tile_evidence_fails_closed(monkeypatch, tmp_path, fault):
    root = Path(__file__).parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(root))
    module = importlib.import_module("prefill_tiles_admission")
    names = ("prefill_activation_tiles.py", "probe_prefill_activation_tiles.py")
    sources = {n: module.digest(root / n) for n in names}
    report = {
        "status": "SAMPLE_CHECKED",
        "negative_control_detected": True,
        "pack_sha256": sources[names[0]],
        "probe_sha256": sources[names[1]],
        "binary_sha256": "a" * 64,
        "wrapper_sha256": "c" * 64,
        "cases": [
            {
                "M": m,
                "N": n,
                "K": k,
                "layout_bytes_equal": True,
                "unequal_elements": 0,
                "equal_rows": m,
                "canaries": True,
                "finite": True,
                "adapter_exact": True,
            }
            for m in module.ROWS
            for n, k in module.SHAPES
        ],
    }
    if fault == "row":
        report["cases"].pop()
    elif fault == "layout":
        report["cases"][0]["layout_bytes_equal"] = False
    elif fault == "adapter":
        report["cases"][0]["adapter_exact"] = False
    elif fault == "binary":
        report["binary_sha256"] = "b" * 64
    elif fault == "source":
        report["pack_sha256"] = "b" * 64
    elif fault == "control":
        report["negative_control_detected"] = False
    elif fault == "wrapper":
        report["wrapper_sha256"] = "old wrapper"
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(report))
    entry = {
        "qualification": str(path),
        "qualification_sha256": module.digest(path),
        "sources": sources,
        "binary_sha256": "a" * 64,
        "wrapper_sha256": "c" * 64,
    }
    if fault:
        with pytest.raises(ValueError):
            module.evidence(entry)
    else:
        assert module.evidence(entry) == report
