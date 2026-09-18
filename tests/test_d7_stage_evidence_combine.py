"""Publication must not combine equal-sized sweeps from different references."""

import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "reports/d7-rdna4-2026-09-17/combine_native_stage_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("d7_stage_evidence_combine", SOURCE)
combine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(combine)


def receipt(tmp_path, name, *, reference="same-reference", fixture="same-fixture", matches=320):
    counts = {"positions": 320, "top20_set_exact": matches, "top20_order_exact": matches}
    data = {
        "status": "SAMPLE_CHECKED",
        "positions": 320,
        "release_full_vector_bridge": 320,
        "prefill_bridge_exact": True,
        "fixture": fixture,
        "receipts": {"release_rows": reference},
        "stages": {"stage": {"old": counts}},
        "per_instance": {"stage": {"layer0": {"old": counts}}},
    }
    data["sha256"] = combine.digest(data)
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_identical_overlap_retains_both_receipts(tmp_path):
    paths = [receipt(tmp_path, name) for name in ("first.json", "second.json")]
    merged = combine.combine(paths)
    assert set(merged["receipts"]) == {"first.json", "second.json"}
    assert merged["release_rows"] == "same-reference"


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        ({"reference": "different-reference"}, "different release reference"),
        ({"fixture": "different-fixture"}, "different fixtures"),
        ({"matches": 319}, "overlapping stage results disagree"),
    ],
)
def test_incompatible_evidence_is_rejected(tmp_path, changed, reason):
    paths = [receipt(tmp_path, "first.json"), receipt(tmp_path, "second.json", **changed)]
    with pytest.raises(AssertionError, match=reason):
        combine.combine(paths)


def test_tampered_receipt_is_rejected(tmp_path):
    path = receipt(tmp_path, "first.json")
    document = json.loads(path.read_text())
    document["stages"]["stage"]["old"]["top20_set_exact"] = 0
    path.write_text(json.dumps(document))
    with pytest.raises(AssertionError):
        combine.combine([path])
