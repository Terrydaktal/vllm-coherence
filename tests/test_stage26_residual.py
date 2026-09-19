import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compute_stage26_residual import compute


CONTEXTS = ("0K", "60K", "200K")


def profile():
    return {
        "measurement_commit": "a" * 40,
        "execution_identity": "eager-test-build",
        "fixture_identity": "fixture-test",
        "context_order": list(CONTEXTS),
        "contexts": {
            context: {"stages": {f"stage-{i}": float(i + 1) for i in range(25)}}
            for context in CONTEXTS
        },
    }


def control(values):
    return {
        "schema": "urn:qwen:stage26-uninstrumented-residual-v1",
        "measurement_commit": "a" * 40,
        "execution_identity": "eager-test-build",
        "fixture_identity": "fixture-test",
        "context_order": list(CONTEXTS),
        "contexts": {
            context: {"full_uninstrumented_round_ms": value}
            for context, value in zip(CONTEXTS, values)
        },
    }


def test_stage26_subtracts_instrumented_sum_from_separate_full_round():
    result = compute(profile(), control((330, 331, 332)))
    assert result["status"] == "complete"
    assert [result["contexts"][context]["instrumented_stage_sum_ms"] for context in CONTEXTS] == [
        325.0,
        325.0,
        325.0,
    ]
    assert [result["contexts"][context]["residual_ms"] for context in CONTEXTS] == [
        5.0,
        6.0,
        7.0,
    ]


def test_stage26_rejects_a_control_from_a_different_execution():
    different = control((330, 331, 332))
    different["execution_identity"] = "compiled-production"
    with pytest.raises(ValueError, match="execution_identity"):
        compute(profile(), different)


def test_stage26_rejects_a_negative_residual_instead_of_hiding_incompatibility():
    with pytest.raises(ValueError, match="incompatible"):
        compute(profile(), control((324, 324, 324)))
