import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
path = (
    Path(__file__).parents[1]
    / "experiments/radiance-public/benchmark_head_candidate_depth.py"
)
spec = importlib.util.spec_from_file_location("head_candidate_depth", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reference():
    return torch.linspace(-4, 4, 128).reshape(1, -1)


def test_identical_full_head_has_zero_error():
    value = reference()
    row = module.compare(value, value.clone())
    assert row["argmax_equal"] == row["top40_complete_including_ties"] == 1
    assert row["top40_same_order"] == row["top40_same_set"] == 1
    assert row["diagnostic_tv_sum"] == row["retained_value_mismatches"] == 0


def test_missing_sampled_token_is_detected_even_when_winner_matches():
    value = reference()
    candidate = value.clone()
    candidate[0, -2] = -float("inf")
    row = module.compare(value, candidate)
    assert row["argmax_equal"] == row["argmax_retained"] == 1
    assert row["top20_complete_including_ties"] == row["top40_same_set"] == 0
    assert row["diagnostic_tv_sum"] > 0
    assert row["excluded_reference_probability_mass_sum"] > 0


def test_rounding_error_is_separate_from_candidate_retention():
    value = reference()
    candidate = value.clone()
    candidate[0, -2] += 1
    row = module.compare(value, candidate)
    assert row["top40_complete_including_ties"] == row["argmax_retained"] == 1
    assert row["argmax_equal"] == 0
    assert row["retained_value_mismatches"] == 1
    assert row["diagnostic_tv_sum"] > 0
    assert row["excluded_reference_probability_mass_sum"] == 0


def test_boundary_tie_cannot_be_silently_discarded():
    value = reference()
    value[0, -41] = value[0, -40]
    candidate = value.clone()
    candidate[0, -41] = -float("inf")
    assert module.compare(value, candidate)["top40_complete_including_ties"] == 0


def test_accumulator_preserves_maxima_and_counts():
    total = {}
    module.accumulate(
        total,
        {
            "rows": 8,
            "diagnostic_tv_sum": 2,
            "diagnostic_tv_max": 0.5,
            "max_retained_logit_difference": 2,
        },
    )
    module.accumulate(
        total,
        {
            "rows": 8,
            "diagnostic_tv_sum": 1,
            "diagnostic_tv_max": 0.4,
            "max_retained_logit_difference": 1,
        },
    )
    assert total == {
        "rows": 16,
        "diagnostic_tv_sum": 3,
        "diagnostic_tv_max": 0.5,
        "max_retained_logit_difference": 2,
    }


def test_insufficient_support_is_rejected():
    value = reference()
    candidate = value.clone()
    candidate[0, :100] = -float("inf")
    with pytest.raises(ValueError, match="insufficient"):
        module.compare(value, candidate)
