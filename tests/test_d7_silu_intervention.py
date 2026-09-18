"""The causal experiment must change only its declared activation implementation."""

import importlib.util
from pathlib import Path

import pytest

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


@pytest.fixture
def driver(monkeypatch):
    root = Path(__file__).parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        "silu_experiment", root / "benchmark_silu_contract_d7.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["compiled", "compiled-no-graphs"])
def test_only_enables_native_silu_under_the_existing_compiler_defaults(driver, mode):
    spec = {"native_config": {"max_num_seqs": 2, "seed": 19, "max_num_batched_tokens": 2048}}
    kwargs = {"execution_mode": mode}
    original = driver.BASE_CONFIG(spec, "fixed-bf16", **kwargs)
    actual = driver.make_config(spec, "fixed-bf16", **kwargs)
    assert actual["compilation_config"].pop("custom_ops") == ["none", "+silu_and_mul"]
    assert actual == original
    assert spec["native_config"]["seed"] == 19


def test_rejects_eager_and_other_lanes(driver):
    spec = {"native_config": {}}
    with pytest.raises(DiagnosticError):
        driver.make_config(spec, "fixed-bf16", execution_mode="eager")
    with pytest.raises(DiagnosticError):
        driver.make_config(spec, "old-bf16")


def test_rejects_an_additional_custom_op_change(driver, monkeypatch):
    monkeypatch.setattr(
        driver, "BASE_CONFIG", lambda *a, **k: {"compilation_config": {"custom_ops": ["all"]}}
    )
    with pytest.raises(DiagnosticError):
        driver.make_config({}, "fixed-bf16")


def test_binds_the_real_base_driver(driver):
    assert (
        driver.hashlib.sha256(Path(driver.benchmark.__file__).read_bytes()).hexdigest()
        == driver.BASE_DRIVER_SHA256
    )
