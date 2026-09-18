"""CPU wiring checks: reuse never replaces the fresh candidate or either oracle."""

import os
import shutil
from pathlib import Path

import pytest
from conformance_fixture import tiny_plan
from test_conformance_campaign import spec as spec
from test_conformance_reflink import simulated_clone

from qwen_r9700_lab import conformance_scenarios as scenarios
from qwen_r9700_lab.conformance_campaign import validate_spec
from qwen_r9700_lab.conformance_replay import run_reference
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, private_json, seal, write_private


def reseal(value, **changes):
    return seal({**{k: v for k, v in value.items() if k != "sha256"}, **changes})


@pytest.mark.parametrize("value", [0, 1, "true", None, []])
def test_reference_reuse_requires_an_explicit_boolean(spec, value):
    with pytest.raises(DiagnosticError, match="explicitly boolean"):
        validate_spec({**spec, "reuse_serial_reference": value})


def test_reference_reuse_is_not_enabled_implicitly(spec):
    assert not validate_spec(spec).get("reuse_serial_reference", False)
    assert validate_spec({**spec, "reuse_serial_reference": True})["reuse_serial_reference"] is True


def test_effective_configuration_preserves_baseline_overrides():
    original = {
        "model": "checkpoint",
        "enforce_eager": False,
        "async_scheduling": True,
        "max_num_seqs": 9,
        "speculative_config": {"method": "dflash"},
        **dict.fromkeys(
            ["kv_transfer_config", "scheduler_cls", "additional_config", "compilation_config"],
            "omit",
        ),
    }
    spec = {"native_config": original}
    serial = scenarios.native_configuration(spec, speculation=False)
    assert serial == {
        "model": "checkpoint",
        "enforce_eager": True,
        "async_scheduling": False,
        "max_num_seqs": 1,
    }
    assert "speculative_config" in scenarios.native_configuration(spec)
    assert original["enforce_eager"] is False


def case_roots(tmp_path, spec):
    cases = [{"id": f"accept{w}", "axes": {"accepted": w}, "seed": 0} for w in [0, 7]]
    campaign = seal({"spec": spec, "cases": cases})
    roots = []
    for i, case in enumerate(cases):
        root = tmp_path / f"case-{i:05d}"
        root.mkdir(mode=0o700)
        write_private(root / "input.json", {"campaign": campaign, "case": case})
        roots.append(root)
    return roots, cases


def test_environment_identity_keeps_unknown_flags_and_redacts_values(tmp_path, monkeypatch):
    spec = {"native_config": {"model": "checkpoint"}, "native_call_mode": "metadata"}
    roots, _ = case_roots(tmp_path, spec)
    monkeypatch.setenv("QWEN_EXAMPLE_SECRET", "not for receipts")
    identities = []
    for root in roots:
        monkeypatch.setenv("TRITON_CACHE_DIR", str(root / "runtime/triton"))
        identities.append(scenarios.serial_reference_identity(spec, root))
    assert identities[0] == identities[1]
    assert "not for receipts" not in str(identities)
    monkeypatch.setenv("QWEN_UNKNOWN_KERNEL_FLAG", "different")
    assert scenarios.serial_reference_identity(spec, roots[1]) != identities[1]
    monkeypatch.delenv("QWEN_UNKNOWN_KERNEL_FLAG")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/another/compiled-cache")
    assert scenarios.serial_reference_identity(spec, roots[1]) != identities[1]


def test_wrong_case_is_refused_before_model_or_store_work(tmp_path):
    spec = {"native_config": {"model": "checkpoint"}, "native_call_mode": "metadata"}
    roots, cases = case_roots(tmp_path, spec)
    with pytest.raises(DiagnosticError, match="case changed"):
        scenarios.serial_reference_for_d7(spec, cases[1], {}, roots[0])
    assert not (tmp_path / "serial-reference-store").exists()


def test_copied_aiter_identity_binds_contents_instead_of_case_path(tmp_path, monkeypatch):
    from qwen_r9700_lab.conformance_runtime import worker_environment

    source = tmp_path / "aiter-source"
    source.mkdir()
    (source / "kernel.py").write_text("original kernel")
    spec = {
        "native_config": {"model": "checkpoint"},
        "native_call_mode": "metadata",
        "binding": {"live_data_abi": "test"},
        "environment": {"AITER_ROOT_DIR": str(source)},
    }
    roots, _ = case_roots(tmp_path, spec)
    identities = []
    for root in roots:
        with monkeypatch.context() as current:
            for key, value in worker_environment(spec, root).items():
                current.setenv(key, value)
            identities.append(scenarios.serial_reference_identity(spec, root))
    assert identities[0] == identities[1]
    altered = roots[1] / "runtime/aiter/kernel.py"
    altered.write_text("modified kernel")
    with monkeypatch.context() as current:
        for key, value in worker_environment(spec, roots[1]).items():
            current.setenv(key, value)
        assert scenarios.serial_reference_identity(spec, roots[1]) != identities[1]
        current.setenv("AITER_ROOT_DIR", str(source))
        assert scenarios.serial_reference_identity(spec, roots[1]) != identities[1]


@pytest.mark.parametrize(
    "fault", ["bytes", "name", "mode", "symlink", "fifo", "empty", "during_read"]
)
def test_copied_tree_identity_detects_changes_and_unsupported_entries(tmp_path, monkeypatch, fault):
    from qwen_r9700_lab import conformance_artifacts
    from qwen_r9700_lab.conformance_reference_store import copied_tree_identity

    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.py").write_text("kernel bytes")
    expected = copied_tree_identity(source)
    target = tmp_path / "copy"
    shutil.copytree(source, target)
    assert copied_tree_identity(target) == expected
    kernel = target / "kernel.py"
    if fault == "bytes":
        kernel.write_text("changed bytes")
    elif fault == "name":
        kernel.rename(target / "other.py")
    elif fault == "mode":
        kernel.chmod(0o700)
    elif fault == "symlink":
        (target / "link").symlink_to(kernel)
    elif fault == "fifo":
        os.mkfifo(target / "fifo")
    elif fault == "empty":
        kernel.unlink()
    elif fault == "during_read":
        original = conformance_artifacts.file_identity

        def changed(path):
            result = original(path)
            (target / "new.py").write_text("new kernel")
            return result

        monkeypatch.setattr(conformance_artifacts, "file_identity", changed)
    if fault in {"bytes", "name", "mode"}:
        assert copied_tree_identity(target) != expected
    else:
        with pytest.raises(DiagnosticError):
            copied_tree_identity(target)


def test_two_widths_share_only_the_serial_execution(tmp_path, monkeypatch):
    base = tiny_plan(tmp_path / "checkpoint")
    spec = {
        "native_config": {"model": "CPU fixture"},
        "native_call_mode": "metadata",
        "binding": seal({"fixture": "CPU storage wiring"}),
        "reuse_serial_reference": True,
    }
    roots, cases = case_roots(tmp_path, spec)
    monkeypatch.setattr(scenarios, "tokens", lambda spec, count, seed: list(range(1, count + 1)))

    def plan_for(spec, case, forced, *, accepted=None):
        return reseal(
            base,
            forced_tokens=forced,
            **({"accepted_widths": accepted} if accepted is not None else {}),
        )

    monkeypatch.setattr(scenarios, "plan_for", plan_for)
    if os.environ.get("QWEN_REQUIRE_REFLINK") != "1":
        monkeypatch.setattr("qwen_r9700_lab.conformance_state.fcntl.ioctl", simulated_clone)
    executions = []

    def native(spec, plan, root, *, speculation=True):
        executions.append((root.name, speculation, len(plan["forced_tokens"])))
        root.mkdir(mode=0o700)
        capture = root / "run/capture"
        capture.parent.mkdir(mode=0o700)
        # Only model execution is substituted. Real plans, store, tensor bytes,
        # prefix projection and both comparison gates execute without mocking.
        run_reference(plan, capture)
        return capture

    monkeypatch.setattr(scenarios, "native", native)
    for root, case in zip(roots, cases, strict=True):
        checks = scenarios.forced_d7(spec, case, root)
        assert "same_consumed_pending_and_all_logical_state" in checks
        assert "selected_causal_rows" in checks
        assert (root / "aligned-comparison.json").is_file()
        assert private_json(root / "boundary-comparison/report.json")["equal"]
    assert executions == [("serial-population", False, 11), ("d7", True, 4), ("d7", True, 11)]
    assert private_json(roots[0] / "serial/reuse.json")["hit"] is False
    assert private_json(roots[1] / "serial/reuse.json")["hit"] is True
    assert (roots[0] / "serial-population/run/capture/schedule.json").is_file()
    assert Path(roots[1] / "serial/run/capture/store-origin.json").is_file()
