"""Negative controls for the lifecycle qualification's actual pass predicates."""

import copy
import importlib.util
from pathlib import Path

import pytest

PATH = (
    Path(__file__).parents[1] / "experiments/radiance-public/qualify_speed_lifecycle.py"
)
spec = importlib.util.spec_from_file_location("speed_lifecycle_checks", PATH)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def dispatch_report():
    return {
        "observed": [
            {
                "requests": 1,
                "tokens": 8,
                "uniform": 8,
                "mode": "FULL",
                "graph_requests": 1,
                "graph_tokens": 8,
                "count": 10,
            }
        ],
        "dispatch_grid": [
            {
                "requests": n,
                "width": w,
                "selected": {"cg_mode": "FULL" if n == 1 and w == 8 else "NONE"},
            }
            for n in (1, 2)
            for w in range(1, 10)
        ],
    }


def test_real_dispatch_domain_passes():
    checks.check_dispatch(dispatch_report())


@pytest.mark.parametrize(
    "field,value",
    [
        ("requests", 2),
        ("tokens", 7),
        ("uniform", 1),
        ("graph_requests", 2),
        ("graph_tokens", 16),
    ],
)
def test_invalid_full_replay_is_detected(field, value):
    report = dispatch_report()
    report["observed"][0][field] = value
    with pytest.raises(AssertionError):
        checks.check_dispatch(report)


def test_unsafe_multi_request_dispatch_is_detected():
    report = dispatch_report()
    report["dispatch_grid"][-2]["selected"]["cg_mode"] = "FULL"
    with pytest.raises(AssertionError):
        checks.check_dispatch(report)


def test_vacuous_dispatch_coverage_rejected():
    report = dispatch_report()
    report["observed"] = []
    with pytest.raises(AssertionError):
        checks.check_dispatch(report)


@pytest.mark.parametrize("mutation", ["tokens", "sha256", "finish"])
def test_output_oracle_detects_corruption(mutation):
    reference = {
        "tokens": 8,
        "sha256": checks.digest(list(range(8))),
        "finish": "length",
    }
    observed = copy.deepcopy(reference)
    observed[mutation] = 7 if mutation == "tokens" else "bad"
    with pytest.raises(AssertionError):
        checks.check_equal(reference, observed)


def test_equal_outputs_pass():
    reference = {
        "tokens": 8,
        "sha256": checks.digest(list(range(8))),
        "finish": "length",
    }
    checks.check_equal(reference, dict(reference))
