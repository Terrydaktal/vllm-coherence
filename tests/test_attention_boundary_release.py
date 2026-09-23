"""A deployment must not silently lose boundary or full-Pi sample coverage."""

import importlib.util
from pathlib import Path

import pytest

from qwen_r9700_lab.diagnostic_contract import seal


PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/radiance-public/prepare_attention_boundary_release.py"
)
SPEC = importlib.util.spec_from_file_location("attention_boundary_release", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def evidence():
    build = seal(
        {"status": "BUILT_UNTESTED", "kernel_abi": "qwen-stock-m1-shared-attention-v1"}
    )
    prefixes = {base + i for base in (0, 1024, 60000, 200000) for i in range(16)}
    prefixes.update(base + i for base in (512, 59904, 200192) for i in range(-7, 1))
    checks = [
        {
            "dtype": dtype,
            "prefix": prefix,
            "rows": 8,
            "candidate_mismatches": 0,
            "baseline_mismatches": 0,
        }
        for dtype in ("torch.uint8", "torch.bfloat16", "torch.float8_e4m3fn")
        for prefix in sorted(prefixes)
    ]
    operator = {
        "status": "SAMPLE_CHECKED",
        "build": build["sha256"],
        "graph_checks": True,
        "negative_control_detected": True,
        "checks": checks,
    }
    comparison = {
        "status": "SAMPLE_CHECKED",
        "candidate_build": build["sha256"],
        "parent_manifest_sha256": "parent",
        "contexts": {
            context: {
                "output_exact": True,
                "generated_tokens": 1500,
                "round_capture_complete": True,
            }
            for context in ("0K", "60K", "200K")
        },
    }
    return build, operator, comparison


def test_complete_sample_evidence_is_admitted():
    build, operator, comparison = evidence()
    MODULE.validate_evidence(build, seal(operator), seal(comparison), "parent")


@pytest.mark.parametrize(
    "fault", ["page_offset", "split_boundary", "mismatch", "graph", "negative_control"]
)
def test_operator_coverage_cannot_be_replaced_by_one_aligned_timing(fault):
    build, operator, comparison = evidence()
    if fault in ("page_offset", "split_boundary"):
        missing = 60009 if fault == "page_offset" else 59903
        operator["checks"] = [c for c in operator["checks"] if c["prefix"] != missing]
    elif fault == "mismatch":
        operator["checks"][0]["candidate_mismatches"] = 1
    elif fault == "graph":
        operator["graph_checks"] = False
    else:
        operator["negative_control_detected"] = False
    with pytest.raises(ValueError, match="qualification"):
        MODULE.validate_evidence(build, seal(operator), seal(comparison), "parent")


@pytest.mark.parametrize(
    "fault", ["wrong_parent", "missing_context", "changed_output", "missing_rounds"]
)
def test_natural_pi_comparison_must_cover_this_parent_and_all_contexts(fault):
    build, operator, comparison = evidence()
    if fault == "wrong_parent":
        comparison["parent_manifest_sha256"] = "another build"
    elif fault == "missing_context":
        del comparison["contexts"]["200K"]
    elif fault == "changed_output":
        comparison["contexts"]["60K"]["output_exact"] = False
    else:
        comparison["contexts"]["0K"]["round_capture_complete"] = False
    with pytest.raises(ValueError, match="matching natural Pi outputs"):
        MODULE.validate_evidence(build, seal(operator), seal(comparison), "parent")
