import json

from test_conformance_cli import cli

from qwen_r9700_lab.conformance_obligations import proof_obligations
from qwen_r9700_lab.diagnostic_contract import authenticate


def test_every_declared_component_has_an_obligation_implementation_and_failure_test():
    report = proof_obligations()
    authenticate(report)
    rows = report["obligations"]
    assert len({row["id"] for row in rows}) == len(rows) == 25
    assert {
        "prompt",
        "gdn",
        "prefill",
        "snapshot",
        "scheduler",
        "gate",
        "dispatch",
        "compiler",
        "protocol",
    } <= {row["id"] for row in rows}
    for row in rows:
        assert row["formula"] and row["remaining_obligation"]
        assert row["backend_status"] == "UNPROVED"
        assert len(row["available_checks"]) == 2
        assert all(
            isinstance(binding, dict) and len(binding["sha256"]) == 64
            for binding in row["available_checks"].values()
        )
    assert report["undischarged"] == [row["id"] for row in rows]
    assert report["universal_equivalence"] == "UNPROVED"
    assert report["instrumentation_complete"] is False


def test_real_cli_does_not_turn_inventory_into_a_proof(tmp_path):
    result = cli("proof-obligations", "--output", tmp_path / "coverage.json", "--require-proved")
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["universal_equivalence"] == "UNPROVED"
    assert report["gpu_used"] is False
    assert report["production_gate_installed"] is False
    assert report["reference"]["precision"]["additional_quantization"] == []
    assert json.loads((tmp_path / "coverage.json").read_text()) == report


def test_profile_cannot_hide_additional_fp8_approximations():
    report = proof_obligations("radiance-fp8")
    assert report["reference"]["precision"]["additional_quantization"] == ["activations", "KV"]
    assert report["stock_quantized_equivalence"] == "UNPROVED"
