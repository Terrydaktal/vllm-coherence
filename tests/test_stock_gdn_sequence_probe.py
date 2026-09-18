"""Admission and evidence-domain checks; this file never requests a GPU."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments/radiance-public"))
    return importlib.import_module("probe_stock_gdn_sequence")


@pytest.fixture
def domain():
    names = ["captured-stock-state", "captured-stock-output"]
    for prefix, count in (("accept", 8), ("suffix", 7), ("chunks", 3)):
        names.extend(
            f"{prefix}-{index}-{kind}" for index in range(count) for kind in ("state", "output")
        )
    return [{"case": name, "equal": True} for name in names]


@pytest.fixture
def controls():
    return {"wrong-prefix-state": True, "one-bit-state-corruption": True}


def test_no_gpu_access_without_explicit_admission(probe):
    with pytest.raises(DiagnosticError, match="explicit --allow-gpu"):
        probe.main(SimpleNamespace(allow_gpu=False))


def test_gpu_lease_is_required_even_with_admission(probe, monkeypatch):
    monkeypatch.delenv("QWEN_CONFORMANCE_GPU_LOCK", raising=False)
    with pytest.raises(DiagnosticError, match="shared GPU lease"):
        probe.main(SimpleNamespace(allow_gpu=True))


def test_only_complete_matching_observations_are_tested(probe, domain, controls):
    assert probe.classify(domain, controls) == "TESTED"
    assert probe.classify([], controls) == "INCOMPLETE"
    assert probe.classify(domain[:-1], controls) == "INCOMPLETE"
    assert probe.classify([*domain[:-1], domain[0]], controls) == "INCOMPLETE"
    assert probe.classify([*domain, domain[0]], controls) == "INCOMPLETE"
    domain[0]["equal"] = False
    assert probe.classify(domain, controls) == "FAILED"


@pytest.mark.parametrize("name", ["wrong-prefix-state", "one-bit-state-corruption"])
def test_checker_must_detect_both_structural_and_bit_corruption(probe, domain, controls, name):
    assert (
        probe.classify(domain, {key: value for key, value in controls.items() if key != name})
        == "INCOMPLETE"
    )
    controls[name] = False
    assert probe.classify(domain, controls) == "INVALID_CHECKER"


def test_comparator_preserves_one_bit_difference_and_its_actual_bytes(probe, tmp_path):
    torch = pytest.importorskip("torch")
    left = torch.tensor([1.0, -0.0], dtype=torch.float32)
    right = left.clone()
    right.view(torch.uint8)[0] ^= 1
    row = probe.compare_tensors("bit-fault", left, right, tmp_path)
    assert row["equal"] is False
    assert row["different_bytes"] == 1
    assert row["first_differing_byte"] == 0
    evidence = tmp_path / "bit-fault"
    saved = json.loads((evidence / "comparison.json").read_text())
    authenticate(saved)
    assert saved["payload_sha256"] == row["payload_sha256"]
    for name, value, digest in zip(
        ("actual.bin", "expected.bin"), (left, right), row["payload_sha256"], strict=True
    ):
        data = (evidence / name).read_bytes()
        assert data == value.view(torch.uint8).numpy().tobytes()
        assert hashlib.sha256(data).hexdigest() == digest


def test_signed_zero_is_a_difference_and_equal_tensors_need_no_failure_dump(probe, tmp_path):
    torch = pytest.importorskip("torch")
    positive = torch.tensor([0.0], dtype=torch.bfloat16)
    negative = torch.tensor([-0.0], dtype=torch.bfloat16)
    assert probe.compare_tensors("zero", positive, negative, tmp_path)["equal"] is False
    assert probe.compare_tensors("equal", positive, positive.clone(), tmp_path)["equal"] is True
    assert not (tmp_path / "equal").exists()


def test_singleton_strided_tensor_is_compared_in_logical_order(probe, tmp_path):
    torch = pytest.importorskip("torch")
    left = torch.tensor([[7.0, 99.0]])[:, 0]
    assert left.shape == (1,) and left.stride() == (2,)
    expected = torch.tensor([7.0])
    assert probe.compare_tensors("strided-equal", left, expected, tmp_path)["equal"]
    expected.view(torch.uint8)[0] ^= 1
    assert probe.compare_tensors("strided-fault", left, expected, tmp_path)["different_bytes"] == 1


def test_different_representations_cannot_pass_on_matching_bytes(probe, tmp_path):
    torch = pytest.importorskip("torch")
    value = torch.tensor([1.0], dtype=torch.float32)
    for other in (value.view(torch.uint32), value.reshape(1, 1)):
        with pytest.raises(DiagnosticError, match="identical representations"):
            probe.compare_tensors("invalid", value, other, tmp_path)
    assert list(tmp_path.iterdir()) == []
