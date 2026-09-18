"""CPU controls for the preserved-transition beta-precision ablation."""

import ast
import gzip
import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "experiments/radiance-public"


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    spec = importlib.util.spec_from_file_location(
        "probe_gdn_beta_precision", DIRECTORY / "probe_gdn_beta_precision.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def source():
    return gzip.decompress(
        (ROOT / "tests/fixtures/radiance_stock_gdn_packed_decode.py.gz").read_bytes()
    )


def test_ablation_changes_only_the_reviewed_beta_cast(probe, source):
    modified = probe.ablation_source(source)
    before, after = ast.parse(source), ast.parse(modified)
    changed = []
    for left, right in zip(before.body, after.body, strict=True):
        if ast.dump(left) != ast.dump(right):
            changed.append(left.name)
    assert changed == [probe.KERNEL]
    assert source.replace(probe.OLD.encode(), probe.NEW.encode(), 1) == modified
    assert modified.startswith(b"# SPDX-License-Identifier: Apache-2.0")


def test_unknown_or_already_modified_source_is_refused(probe, source):
    for modified in (source + b"\n", probe.ablation_source(source), b"# unsupported source"):
        with pytest.raises(ValueError, match="reviewed preimage"):
            probe.ablation_source(modified)


def test_exact_control_detects_signed_zero_difference(probe):
    positive = np.array([0.0, 1.0], dtype=np.float32)
    negative = np.array([-0.0, 1.0], dtype=np.float32)
    comparison = probe.exact_comparison(positive, negative)
    assert comparison["equal"]
    assert not comparison["bit_equal"]


def test_nonfinite_or_changed_representation_is_not_a_valid_control(probe):
    value = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="representations"):
        probe.exact_comparison(value, value.astype(np.float64))
    with pytest.raises(ValueError, match="finite"):
        probe.exact_comparison(value, np.array([np.nan], dtype=np.float32))


@pytest.fixture
def rows():
    return {
        mode: {
            "guards_unchanged": True,
            "inputs_unchanged": True,
            "state_to_captured_stock": {"bit_equal": mode != "beta_fp32_only"},
            "output_to_captured_stock": {"bit_equal": mode != "beta_fp32_only"},
        }
        for mode in ("stock", "stock_repeat", "beta_fp32_only")
    }


def test_measured_ablation_is_never_promoted_to_qualified_repair(probe, rows):
    assert probe.classify(rows) == "DIAGNOSTIC_MEASURED"
    assert probe.classify({}) == "INCOMPLETE"


@pytest.mark.parametrize("mode", ["stock", "stock_repeat"])
@pytest.mark.parametrize("boundary", ["state_to_captured_stock", "output_to_captured_stock"])
def test_unreproduced_original_cannot_validate_precision_ablation(probe, rows, mode, boundary):
    rows[mode][boundary]["bit_equal"] = False
    assert probe.classify(rows) == "INVALID_CONTROL"


@pytest.mark.parametrize("boundary", ["guards_unchanged", "inputs_unchanged"])
def test_candidate_cannot_change_memory_outside_its_selected_state(probe, rows, boundary):
    modified = deepcopy(rows)
    modified["beta_fp32_only"][boundary] = False
    assert probe.classify(modified) == "INVALID_CONTROL"


def test_gpu_authorization_is_checked_before_reading_sources(probe):
    with pytest.raises(ValueError, match="explicit --allow-gpu"):
        probe.main(SimpleNamespace(allow_gpu=False))
