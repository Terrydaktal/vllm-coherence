"""Keep the published current tables accountable to their retained evidence."""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from analyze_release_timings import analyze, phase
from render_current_tables import update


def test_readme_is_current_only_and_matches_committed_measurements():
    data = json.loads((ROOT / "benchmarks/results/coherence-current.json").read_text())
    readme = (ROOT / "README.md").read_text()
    assert update(readme, data) == readme
    assert len(data["stages"]) == 29 and len(data["layers"]) == 64
    assert [row["layer"] for row in data["layers"]] == list(range(64))
    assert math.isclose(
        sum(r["ms"] or 0 for r in data["stages"]), data["gpu_ms"], abs_tol=1e-8
    )
    assert math.isclose(
        sum(r["ms"] for r in data["kernels"]), data["gpu_ms"], abs_tol=1e-8
    )
    assert "Before: same set" not in readme and "Old ms per round" not in readme
    for layer in data["layers"]:
        assert math.isclose(
            sum(layer[k] for k in ("input", "output", "gate_up", "down", "other")),
            layer["total"],
            abs_tol=1e-8,
        )


def test_latest_profile_selects_complete_inventory_not_fast_rounds():
    evidence = json.loads(
        (ROOT / "benchmarks/results/20260918-ggz14-stage-profile.json").read_text()
    )
    assert evidence["profile_rounds"] == 8
    assert evidence["included_rounds"] == [0, 1, 2, 3, 4, 6, 7]
    assert evidence["omitted_inventory_rounds"] == [5]
    assert evidence["graph_launches_per_round"] == [65] * 8
    assert len(evidence["layers_ms"]) == 64
    assert all(
        math.isfinite(value) and value > 0 for value in evidence["stages_ms"].values()
    )
    assert math.isclose(
        sum(evidence["stages_ms"].values()), evidence["all_gpu_ms"], abs_tol=1e-8
    )
    assert (
        max(evidence["per_round_gpu_ms"].values()) > 43
    )  # Retains the slower complete rounds.


def test_fused_launches_have_one_duration_boundary():
    assert (
        phase(0, "input", "void norm_quant<false, 512>")
        == "Embedding + first input normalization"
    )
    assert (
        phase(1, "post", "void norm_quant<true, 512>")
        == "Post-attention/GDN residual/normalization"
    )
    assert (
        phase(1, "mix", "gdn_norm_quant_kernel.kd") == "GDN output gated normalization"
    )
    assert (
        phase(
            1,
            "activation",
            "triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_0.kd",
        )
        == "MLP SiLU and gating"
    )
    assert (
        phase(1, "activation", "dynamic_per_token_scaled_fp8_quant_kernel")
        == "MLP down input FP8 quantization"
    )
    with pytest.raises(ValueError):
        phase(3, "mix", "gdn_norm_quant_kernel.kd")
    with pytest.raises(ValueError):
        phase(1, "input", "gdn_norm_quant_kernel.kd")
    with pytest.raises(ValueError):
        phase(1, "mix", "void norm_quant<true, 512>")


def test_unprofiled_or_missing_graph_trace_is_not_accepted_as_measurement():
    with pytest.raises(ValueError, match="compiled rounds"):
        analyze(json.dumps({"traceEvents": []}).encode(), "global256")
