import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def driver(monkeypatch):
    path = Path(__file__).parents[1] / "experiments/radiance-public/benchmark_optimized_d7.py"
    spec = importlib.util.spec_from_file_location("optimized_d7_config_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compiled_profile_keeps_production_capture_sizes_and_does_not_mutate_spec(driver):
    source = {
        "native_config": {
            "enforce_eager": True,
            "async_scheduling": False,
            "kv_transfer_config": {"private": "store"},
            "max_num_seqs": 2,
            "speculative_config": {"method": "dflash", "num_speculative_tokens": 7},
        }
    }
    config = driver.make_config(source, "old-bf16")
    assert config["enforce_eager"] is False
    assert config["compilation_config"] == {
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }
    assert "async_scheduling" not in config
    assert "kv_transfer_config" not in config
    assert config["max_num_seqs"] == 2
    assert config["speculative_config"]["num_speculative_tokens"] == 7
    assert source["native_config"]["enforce_eager"] is True
    assert "kv_transfer_config" in source["native_config"]


def test_old_and_fixed_use_identical_compilation_configuration(driver):
    spec = {"native_config": {"max_num_seqs": 2}}
    assert driver.make_config(spec, "old-bf16") == driver.make_config(spec, "fixed-bf16")


def test_m1_control_only_removes_speculation_and_keeps_graphs(driver):
    config = driver.make_config(
        {"native_config": {"speculative_config": {"method": "dflash"}}},
        "old-bf16",
        speculation=False,
    )
    assert "speculative_config" not in config
    assert config["async_scheduling"] is False
    assert config["enforce_eager"] is False
    assert config["compilation_config"]["cudagraph_mode"] == "PIECEWISE"


def test_isolated_capture_stays_compiled_but_is_not_a_release_timing_profile(driver):
    source = {"native_config": {"speculative_config": {"method": "dflash"}}}
    diagnostic = driver.make_config(source, "fixed-bf16", isolated_capture=True)
    release = driver.make_config(source, "fixed-bf16")
    assert diagnostic["enforce_eager"] is False
    assert diagnostic["compilation_config"]["cudagraph_mode"] == "NONE"
    assert diagnostic["worker_cls"].endswith("IsolatedCaptureWorker")
    assert release["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
    assert release["worker_cls"].endswith("OptimizedWorker")
    assert diagnostic["speculative_config"] == release["speculative_config"]


def test_capture_cannot_be_used_to_report_a_clean_speed_result(driver, monkeypatch, tmp_path):
    import sys

    from qwen_r9700_lab.diagnostic_contract import DiagnosticError

    argv = ["benchmark", "run", "--isolated-capture", "--allow-gpu"]
    for key in ("spec", "fixture", "private", "output"):
        argv += ["--" + key, str(tmp_path / key)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(DiagnosticError, match="untimed correctness diagnostic"):
        driver.main()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("capture", [False, True])
def test_execution_controls_keep_the_same_nonexecution_configuration(driver, capture):
    from qwen_r9700_lab.conformance_execution_modes import numerical_config

    spec = {
        "native_config": {
            "max_num_seqs": 2,
            "max_num_batched_tokens": 2048,
            "max_model_len": 253792,
            "kv_cache_memory_bytes": 10_000_000_000,
            "kv_cache_dtype": "fp8",
            "speculative_config": {"method": "dflash", "num_speculative_tokens": 7},
        }
    }
    configs = {
        m: driver.make_config(spec, "fixed-bf16", isolated_capture=capture, execution_mode=m)
        for m in ("compiled", "compiled-no-graphs", "eager")
    }
    assert len({str(numerical_config(c)) for c in configs.values()}) == 1
    assert configs["eager"]["enforce_eager"]
    assert configs["eager"]["compilation_config"] == {"mode": 0, "cudagraph_mode": "NONE"}
    assert configs["compiled-no-graphs"]["enforce_eager"] is False
    assert configs["eager"]["worker_cls"] == configs["compiled-no-graphs"]["worker_cls"]


@pytest.mark.parametrize("mode", ["compiled-no-graphs", "eager"])
def test_nonrelease_mode_cannot_silently_become_a_speed_benchmark(
    driver, monkeypatch, tmp_path, mode
):
    import sys

    from qwen_r9700_lab.diagnostic_contract import DiagnosticError

    argv = ["benchmark", "run", "--execution-mode", mode, "--allow-gpu"]
    for key in ("spec", "fixture", "private", "output"):
        argv += ["--" + key, str(tmp_path / key)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(DiagnosticError, match="untimed forced correctness"):
        driver.main()
    assert not (tmp_path / "output").exists()


def test_wrong_actual_compiler_or_graph_mode_is_rejected(driver):
    from qwen_r9700_lab.diagnostic_contract import DiagnosticError

    eager = {"enforce_eager": True, "compilation_mode": 0, "graph_mode": "NONE"}
    driver.validate_execution_metadata(eager, execution_mode="eager", isolated_capture=False)
    for corrupt in ({"enforce_eager": False}, {"compilation_mode": 3}, {"graph_mode": "PIECEWISE"}):
        with pytest.raises(DiagnosticError):
            driver.validate_execution_metadata(
                {**eager, **corrupt}, execution_mode="eager", isolated_capture=False
            )
