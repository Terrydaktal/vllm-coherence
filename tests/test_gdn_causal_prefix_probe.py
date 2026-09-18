"""The causal regression must not accept broken controls or missing cases."""

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "experiments/radiance-public/probe_gdn_causal_prefix.py"
)
spec = importlib.util.spec_from_file_location("gdn_causal_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def controls():
    cases = [
        {"name": name, "finite": True, "guards_intact": True, "initial_state_unchanged": True}
        for name in probe.CASES
    ]
    comparisons = [
        {
            "other": name,
            "identical_input_prefix": True,
            "identical_initial_state": True,
            "oracle_prefix_exact": True,
            "kkt_prefix_lower_exact": True,
            "output_prefix_exact_bytes": True,
        }
        for name in probe.COMPARISONS
    ]
    return cases, comparisons


def test_exact_controls_pass():
    assert probe.outcome(*controls()) == "TESTED"


@pytest.mark.parametrize("name", ["prefix32", "prefix48", "future_gates"])
def test_detects_causal_or_partition_discrepancy(name):
    cases, comparisons = controls()
    next(row for row in comparisons if row["other"] == name)["output_prefix_exact_bytes"] = False
    assert probe.outcome(cases, comparisons) == "CAUSAL_PREFIX_DISCREPANCY"


@pytest.mark.parametrize("name", ["repeat", "future_values"])
def test_control_failure_is_not_attributed_to_gate_rescaling(name):
    cases, comparisons = controls()
    next(row for row in comparisons if row["other"] == name)["output_prefix_exact_bytes"] = False
    assert probe.outcome(cases, comparisons) == "CONTROL_DISCREPANCY"


@pytest.mark.parametrize(
    "key", ["identical_input_prefix", "identical_initial_state", "oracle_prefix_exact"]
)
def test_invalid_input_or_oracle_blocks_claim(key):
    cases, comparisons = controls()
    comparisons[1][key] = False
    assert probe.outcome(cases, comparisons) == "INVALID_CONTROL"


@pytest.mark.parametrize("key", ["finite", "guards_intact", "initial_state_unchanged"])
def test_memory_or_finite_failure_blocks_claim(key):
    cases, comparisons = controls()
    cases[2][key] = False
    assert probe.outcome(cases, comparisons) == "INVALID_CONTROL"


def test_empty_or_duplicate_domain_is_not_a_pass():
    cases, comparisons = controls()
    assert probe.outcome([], []) == "INVALID_CONTROL"
    assert probe.outcome(cases[:-1], comparisons) == "INVALID_CONTROL"
    assert probe.outcome(cases, [*comparisons, comparisons[0]]) == "INVALID_CONTROL"


def test_kkt_difference_remains_failure_without_misattributing_scan():
    cases, comparisons = controls()
    comparisons[1]["kkt_prefix_lower_exact"] = False
    comparisons[1]["output_prefix_exact_bytes"] = False
    assert probe.outcome(cases, comparisons) == "CAUSAL_PREFIX_DISCREPANCY"


def test_torch_runtime_is_loaded_before_native_rocm_extension(monkeypatch, tmp_path):
    events = []
    original_import = builtins.__import__

    def intercept(name, *args, **kwargs):
        if name == "torch":
            events.append("torch")
            return SimpleNamespace()
        if name == "safetensors.torch":
            return SimpleNamespace(save_file=None)
        return original_import(name, *args, **kwargs)

    class StopBeforeNativeLoadError(Exception):
        pass

    def native_import(name):
        assert name == "radiance_gdn"
        assert events == ["torch"]
        raise StopBeforeNativeLoadError

    monkeypatch.setattr(builtins, "__import__", intercept)
    monkeypatch.setattr(probe.importlib, "import_module", native_import)
    with pytest.raises(StopBeforeNativeLoadError):
        probe.probe(tmp_path)
