"""Keep the published current tables accountable to their retained evidence."""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from analyze_release_timings import analyze, measure_round_windows, phase
from render_current_tables import current_stage_profile, update


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
    measured_stages = {row["stage"] for row in data["stages"] if row["ms"] is not None}
    assert set(data["stage_provenance"]) == measured_stages
    for entry in data["stage_provenance"].values():
        assert len(entry["commit"]) == 40
        assert entry["commit"] in readme
        assert entry["change"]
    evidence = json.loads((ROOT / data["sources"]["profile"]).read_text())
    assert data["round_timing"] == evidence["round_timing"]
    assert data["gpu_ms"] == evidence["all_gpu_ms"]
    assert "26. Estimated runtime overhead" in readme
    assert "GPU execution subtotal" not in readme
    from collections import Counter

    profile_commit = current_stage_profile(data)["measurement_commit"]
    expected_links = Counter({data["measurement_commit"]: 1})
    parent = None
    for row in data["stages"]:
        if row["ms"] is not None:
            parent = row["stage"]
        provenance_stage = row["stage"] if row["ms"] is not None else parent
        expected_links.update(
            data.get("evidence_commits", {}).get(
                provenance_stage, [data["measurement_commit"]]
            )
        )
        expected_links[data["stage_provenance"][provenance_stage]["commit"]] += 1
    expected_links[data["measurement_commit"]] += 1  # overhead evidence
    for commit in expected_links:
        commit_url = f"https://github.com/Terrydaktal/vllm-coherence/commit/{commit}"
        assert readme.count(commit_url) >= 1
    profile_url = (
        f"https://github.com/Terrydaktal/vllm-coherence/commit/{profile_commit}"
    )
    assert readme.count(profile_url) >= 1
    assert "The timing-table header and each correctness result link to the commit" in readme
    assert (
        f"Current timing per retained compiled profile cycle (0K / 60K / 200K; milliseconds unless explicitly marked; evidence run [`{profile_commit[:7]}`]"
        in readme
    )
    compiled_table = readme.split("## Compiled backend stages\n\n", 1)[1].split(
        "\n\nThe table restores", 1
    )[0]
    row_names = [
        line.split("|")[1].strip()
        for line in compiled_table.splitlines()
        if line.startswith("| **") or line.startswith("| ↳")
    ]
    assert row_names == [
        "**1. Drafter**",
        "**2. Embedding + first input normalization + FP8 production**",
        "**3. Layer input residual/normalization + FP8 production**",
        "↳ GDN input activation FP8 quantization",
        "↳ Attention input activation FP8 quantization",
        "**4. GDN input projection**",
        "**5. GDN layout/copies and buffer initialization**",
        "**6. GDN convolution**",
        "**7. GDN recurrence and gates**",
        "**8. GDN output gated normalization + FP8 production**",
        "↳ GDN output activation FP8 quantization",
        "**9. GDN output projection**",
        "**10. Attention input projection**",
        "**11. Attention Q/K normalization, RoPE and layout**",
        "**12. Attention KV write**",
        "**13. Attention decode**",
        "**14. Attention split-KV merge**",
        "**15. Attention output gating**",
        "**16. Attention output activation FP8 quantization**",
        "**17. Attention output projection**",
        "**18. Post-attention/GDN residual/normalization + FP8 production**",
        "↳ MLP gate/up input FP8 quantization",
        "**19. MLP gate/up projection**",
        "**20. MLP SiLU and gating**",
        "**21. MLP down input FP8 quantization**",
        "**22. MLP down projection**",
        "**23. Final normalization/layout**",
        "**24. Global-256 target head**",
        "**25. Other GPU bookkeeping**",
        "**26. Estimated runtime overhead**",
        "**Total profile cycle (stages 1–26)**",
    ]
    assert "| **3. Input preparation and cache metadata** |" not in readme
    assert "| **12. RoPE and layout** |" not in readme
    assert "| **22. Target sampling and acceptance bookkeeping** |" not in readme
    assert "| **24. Forced replay control** |" not in readme
    assert "the ↳ rows are detail-only inclusion records and add no timing" in readme
    assert "zero rows have no separately emitted scope" not in readme
    assert "Last relevant code commit / change" in readme
    assert "The provenance column links the last relevant implementation commit" in readme
    assert "2,183 / 1,191 / 1,191 requested rounds" in readme
    assert "2,062 / 1,132 / 1,133 complete cycles" in readme
    assert "0K / 60K / 200K" in readme
    assert "separate uninstrumented full-round mean minus the sum of the 25 named instrumented-stage means" in readme
    assert "profile-cycle wall time minus the 25 named stage kernel totals" not in readme
    assert data["stage_profile_2k"]["stage26_benchmark"]["status"] == "pending"
    for context in data["stage_profile_2k"]["contexts"].values():
        assert "profile_residual_ms" not in context
        assert "profile_stage26_ms" not in context
    assert "Calls in 6 retained cycles" in readme
    assert readme.index("## Global-256 target-head") < readme.index("## Benchmarks")
    assert readme.count("## Global-256 target-head") == 1
    assert readme.count("## Benchmarks") == 1
    assert "## Global-256 target-head benchmark" not in readme
    assert "## 60K coding, prose, JSON, thinking and compaction benchmark" not in readme
    assert "**≈2.8**" not in readme
    assert "estimated from separate runs" not in readme
    assert "## Task workload performance" not in readme
    assert "60K live-chat cache-state diagnosis" not in readme
    assert "Coding task" in readme
    assert "Prose about code measurement" in readme
    assert "JSON task" in readme
    assert "Thinking/prose task" in readme
    assert "Compaction checkpoint" in readme
    assert "## Coding task by context length" not in readme
    assert "This is the same natural-stop coding task run independently" in readme
    assert "benchmark_pi_coding_contexts.py" in readme
    assert "| 0K |" in readme and "| 60K |" in readme and "| 200K |" in readme
    assert "Peak 3s" in readme
    assert "completion marker valid, required headings missing" in readme
    assert "phase_token_counts_cover_output=false" in readme
    assert "peak_3s_tokens_per_second" in readme
    assert "### Known remaining symptoms and likely causes" in readme
    assert readme.index("### Known remaining symptoms and likely causes") > readme.index(
        "The earlier retained round log gives this historical partial latency histogram"
    )
    assert "#### Changes since `cbbf495`" not in readme
    assert "### Remaining symptoms and likely causes" not in readme
    assert "Status: pending fix" not in readme
    assert "HIP/ROCr stream or queue dependency" in readme
    assert "After all 64 layers: finish target verification" not in readme
    assert "Draft proposals" not in readme
    assert "Total elapsed GPU cycle" not in readme
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
    assert evidence["included_rounds"] == [0, 1, 2, 3, 4, 6]
    assert evidence["omitted_inventory_rounds"] == [5]
    assert evidence["omitted_unbounded_rounds"] == [7]
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
    timing = evidence["round_timing"]
    assert [row["round"] for row in timing["rounds"]] == evidence["included_rounds"]
    for row in timing["rounds"]:
        assert row["kernel_sum_ms"] == pytest.approx(
            evidence["per_round_gpu_ms"][str(row["round"])]
        )
        assert row["elapsed_ms"] == pytest.approx(
            row["kernel_sum_ms"] - row["gpu_overlap_ms"] + row["overhead_ms"]
        )
    for key, value in timing["mean"].items():
        assert value == pytest.approx(
            sum(row[key] for row in timing["rounds"]) / len(timing["rounds"])
        )


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


def kernel(start, duration, stream=1, device=0):
    return {
        "ph": "X",
        "cat": "kernel",
        "ts": start,
        "dur": duration,
        "args": {"stream": stream, "device": device},
    }


def test_overhead_counts_gaps_once_and_subtracts_concurrent_gpu_work():
    events = [
        kernel(0, 900),  # Before the first round: must not become bookkeeping.
        kernel(1000, 4000),
        kernel(3000, 3000, stream=2),
        kernel(8000, 1000),
        kernel(11000, 1000),  # Next round: must not enter this round's totals.
        {"ph": "X", "cat": "cpu_op", "ts": 1000, "dur": 10000},
    ]
    timing = measure_round_windows(events, [1000, 11000], [0])
    assert timing["mean"] == {
        "elapsed_ms": 10,
        "kernel_sum_ms": 8,
        "gpu_busy_ms": 6,
        "gpu_overlap_ms": 2,
        "overhead_ms": 4,
    }
    assert timing["rounds"][0]["kernel_count"] == 3
    assert timing["rounds"][0]["streams"] == [1, 2]


@pytest.mark.parametrize(
    "starts, selected",
    [
        ([0, 10], [1]),
        ([0, 10], []),
        ([0, 10], [0, 0]),
        ([10, 0], [0]),
        ([0, float("nan")], [0]),
    ],
)
def test_unobserved_or_ambiguous_round_boundaries_are_rejected(starts, selected):
    with pytest.raises(ValueError, match="bounded round windows"):
        measure_round_windows([kernel(0, 1)], starts, selected)


@pytest.mark.parametrize(
    "events, error",
    [
        ([kernel(-1, 2)], "crosses a round boundary"),
        ([kernel(9, 2)], "crosses a round boundary"),
        ([kernel(0, float("nan"))], "invalid GPU activity"),
        ([kernel(0, -1)], "invalid GPU activity"),
        ([kernel(0, 1), kernel(2, 1, device=1)], "exactly one GPU"),
        ([{**kernel(0, 1), "cat": "gpu_memcpy"}], "explicit stage accounting"),
    ],
)
def test_unaccounted_gpu_activity_does_not_become_reported_overhead(events, error):
    with pytest.raises(ValueError, match=error):
        measure_round_windows(events, [0, 10], [0])


def test_renderer_rejects_overhead_from_a_different_kernel_population():
    data = json.loads((ROOT / "benchmarks/results/coherence-current.json").read_text())
    data["round_timing"]["mean"]["kernel_sum_ms"] += 1
    with pytest.raises(ValueError, match="same rounds"):
        update((ROOT / "README.md").read_text(), data)


def test_profiled_gaps_are_not_published_as_serving_overhead():
    data = json.loads((ROOT / "benchmarks/results/coherence-current.json").read_text())
    text = (ROOT / "README.md").read_text()
    expected = update(text, data)
    data["round_timing"]["mean"]["overhead_ms"] += 100
    data["round_timing"]["mean"]["elapsed_ms"] += 100
    assert update(text, data) == expected
    data["serving_overhead"]["ms"] = data["round_timing"]["mean"]["overhead_ms"]
    with pytest.raises(ValueError, match="does not match its source timings"):
        update(text, data)


def test_runtime_overhead_estimate_reconciles_to_unprofiled_round_time():
    data = json.loads((ROOT / "benchmarks/results/coherence-current.json").read_text())
    estimate = data["serving_overhead"]
    assert estimate["ms"] + data["gpu_ms"] == pytest.approx(
        estimate["unprofiled_round_ms"]
    )
    assert "performance" not in data
    assert "performance_scope" not in data
    assert "performance_limits" not in data
    estimate["status"] = "measured"
    with pytest.raises(ValueError, match="labelled as an estimate"):
        update((ROOT / "README.md").read_text(), data)
