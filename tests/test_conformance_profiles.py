from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from conformance_fixture import tiny_checkpoint

from qwen_r9700_lab.conformance_cli import make_plan
from qwen_r9700_lab.conformance_model import Checkpoint, QuantizedQwenReference
from qwen_r9700_lab.conformance_reference import REFERENCE_PROFILES, bf16, reference_precision
from qwen_r9700_lab.conformance_replay import run_reference, validate_plan
from qwen_r9700_lab.conformance_state import compare_frames, read_frame
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal


def spec(root, profile="weight-only-bf16"):
    return {
        "checkpoint": str(root),
        "checkpoint_files": tiny_checkpoint(root),
        "kv_scales": {"3": [1.0, 1.0]},
        "prefix": [1, 4, 8],
        "forced_tokens": [7, 9, 13, 3],
        "reference_profile": profile,
    }


def model(plan):
    return QuantizedQwenReference(
        Checkpoint(Path(plan["checkpoint"]), plan["checkpoint_files"]),
        kv_scales=plan["kv_scales"],
        contract=plan["contract"],
        execution=plan["execution"],
        adapter=plan["adapter"],
        reference_profile=plan["reference_profile"],
    )


def test_new_plan_defaults_to_weight_only_and_separately_names_extra_quantization(tmp_path):
    specification = spec(tmp_path / "checkpoint")
    del specification["reference_profile"]
    plan, semantics, _ = make_plan(specification)
    assert plan["reference_profile"] == "weight-only-bf16"
    assert semantics["activation_quantization"]["extra_quantizer"] is False
    assert semantics["kv_representation"]["format"] == "bf16"
    contracts = set()
    for profile in REFERENCE_PROFILES:
        p, _, _ = make_plan({**specification, "reference_profile": profile})
        contracts.add(p["contract"])
    assert len(contracts) == 4


@pytest.mark.parametrize("profile", REFERENCE_PROFILES)
def test_all_precision_profiles_preserve_snapshot_suffix_and_encoding(tmp_path, profile):
    plan, _, _ = make_plan(spec(tmp_path / "checkpoint", profile))
    a, b = model(plan), model(plan)
    try:
        for token in plan["prefix"]:
            a.step(token)
        a.frame(tmp_path / "saved", phase="commit", pending=7)
        encoding = read_frame(tmp_path / "saved")["components"]["layer.003.keys"]["dtype"]
        assert encoding == reference_precision(profile)["kv_encoding"]
        b.restore(tmp_path / "saved")
        for token in plan["forced_tokens"]:
            np.testing.assert_array_equal(a.step(token), b.step(token))
        a.frame(tmp_path / "a", phase="step")
        b.frame(tmp_path / "b", phase="step")
        assert compare_frames(tmp_path / "a", tmp_path / "b")["equal"]
    finally:
        a.close()
        b.close()


def test_weight_only_path_cannot_silently_quantize_activations_or_kv(tmp_path, monkeypatch):
    from qwen_r9700_lab import conformance_reference as reference

    plan, _, _ = make_plan(spec(tmp_path / "checkpoint"))

    def forbidden(*_):
        pytest.fail("FP8 quantization entered weight-only reference")

    monkeypatch.setattr(reference, "fp8_encode", forbidden)
    monkeypatch.setattr(reference, "activation_quantize", forbidden)
    run_reference(plan, tmp_path / "reference")


def test_fp8_is_an_observable_change_to_weight_only_arithmetic(tmp_path):
    specification = spec(tmp_path / "checkpoint")
    rows = {}
    for profile in REFERENCE_PROFILES:
        plan, _, _ = make_plan({**specification, "reference_profile": profile})
        engine = model(plan)
        try:
            rows[profile] = np.stack([engine.step(t) for t in specification["prefix"]])
        finally:
            engine.close()
    for profile in set(REFERENCE_PROFILES) - {"weight-only-bf16"}:
        assert not np.array_equal(rows[profile], rows["weight-only-bf16"])


@pytest.mark.parametrize("change", ["profile", "scales", "weights", "config", "arithmetic"])
def test_resealed_plan_cannot_change_its_mathematical_target(tmp_path, change):
    plan, _, _ = make_plan(spec(tmp_path / "checkpoint"))
    altered = deepcopy(plan)
    if change == "profile":
        altered["reference_profile"] = "radiance-fp8"
    elif change == "scales":
        altered["kv_scales"]["3"][0] = 2
    elif change == "weights":
        altered["checkpoint_files"]["model-00001.safetensors"] = "0" * 64
    elif change == "config":
        altered["reference_semantics"]["weights"]["config"]["hidden_size"] = 64
    else:
        altered["reference_semantics"]["numerical_contract"]["reductions"] = "anything"
    altered = seal({k: v for k, v in altered.items() if k != "sha256"})
    with pytest.raises(DiagnosticError):
        validate_plan(altered)


@pytest.mark.parametrize("profile,code", [("radiance-fp8", 127), ("weight-only-bf16", 0x7FC0)])
def test_encoded_nan_cache_cannot_be_restored(tmp_path, profile, code):
    plan, _, _ = make_plan(spec(tmp_path / "checkpoint", profile))
    engine = model(plan)
    try:
        engine.step(1)
        engine.state[3]["keys"].flat[0] = code
        engine.frame(tmp_path / "corrupt", phase="commit")
        with pytest.raises(DiagnosticError, match="nonfinite KV"):
            engine.restore(tmp_path / "corrupt")
    finally:
        engine.close()


def test_bf16_rounding_all_representable_finite_patterns_and_midpoints():
    codes = np.arange(65536, dtype=np.uint32)
    bits = codes << 16
    finite = (bits & 0x7F800000) != 0x7F800000
    np.testing.assert_array_equal(bf16(bits[finite].view(np.float32)).view(np.uint32), bits[finite])
    # Independent midpoint oracle: odd retained codes round up, even codes stay.
    midpoint = bits[finite] | 0x8000
    expected = (codes[finite] + (codes[finite] % 2)) << 16
    np.testing.assert_array_equal(bf16(midpoint.view(np.float32)).view(np.uint32), expected)


def test_cpu_reference_runtime_drift_is_refused_before_execution(tmp_path, monkeypatch):
    from qwen_r9700_lab import conformance_replay

    plan, _, _ = make_plan(spec(tmp_path / "checkpoint"))
    monkeypatch.setattr(
        conformance_replay, "reference_runtime_identity", lambda: {"sha256": "0" * 64}
    )
    with pytest.raises(DiagnosticError, match="runtime changed"):
        run_reference(plan, tmp_path / "should-not-exist")
    assert not (tmp_path / "should-not-exist").exists()
