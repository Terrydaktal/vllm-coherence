"""Keep the published current tables accountable to their retained evidence."""

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from analyze_release_timings import analyze, measure_round_windows, measure_worker_windows, phase, target_inventory_issue
from render_current_tables import (
    commit_marker,
    current_stage_profile,
    render_chained_workload_results,
    workload_target_head,
    update,
)


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
    matched = json.loads((ROOT / data["matched_stage_profile"]).read_text())
    for stage, entry in data["stage_provenance"].items():
        if entry.get("document"):
            assert entry["document"] == data["current_qualification_document"]
            assert f"[Qualified speed changes]({entry['document']})" in readme
            assert entry["change"]
            continue
        assert len(entry["commit"]) == 40
        if stage == "Global-256 target head" and matched.get("target_head") == "global512":
            assert commit_marker(data["target_head_change_commit"]) in readme
        else:
            assert entry["commit"] in readme
        assert entry["change"]
    evidence = json.loads((ROOT / data["sources"]["profile"]).read_text())
    diagnostic_profile = json.loads(
        (ROOT / "benchmarks/results/compiled-global256-stage-profile-1200.json").read_text()
    )
    assert data["stage_profile_2k"]["measurement_kind"] == (
        "archived_diagnostic_stage_attribution_only"
    )
    assert data["stage_profile_2k"]["status"] == "archived_not_production_metric"
    assert diagnostic_profile["measurement_kind"] == "archived_diagnostic_stage_attribution_only"
    assert diagnostic_profile["status"] == "archived_not_production_metric"
    assert diagnostic_profile["production_timing_eligible"] is False
    assert diagnostic_profile["observer_effect"]["first_use_triton_jit_observed"] is True
    assert diagnostic_profile["observer_effect"]["diagnosis_artifact"].endswith(
        "round-jit-diagnosis-20260921.json"
    )
    assert matched["status"] == "matched_estimate"
    assert matched["observer_effect"]["zero_observer_effect_proven"] is False
    assert "No per-stage event probes or forced-token replay" in readme
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
        provenance = data["stage_provenance"][provenance_stage]
        if "commit" in provenance:
            expected_links[provenance["commit"]] += 1
    expected_links[data["measurement_commit"]] += 1  # overhead evidence
    for commit in expected_links:
        commit_url = f"https://github.com/Terrydaktal/vllm-coherence/commit/{commit}"
        assert readme.count(commit_url) >= 1
    assert data.get("current_qualification_commit", data.get("capture_base_commit")) == profile_commit
    assert "pending commit" not in readme
    assert "Current stage confirmations identify M1/M8 and eager/compiled M8 separately" in readme
    confirmations = json.loads((ROOT / data["current_confirmations"]).read_text())
    assert confirmations["optimized_manifest_sha256"] == matched["binding"]["optimized_manifest_sha256"]
    assert confirmations["decode_tokens_per_arm"] == 320
    assert len(confirmations["comparisons"]) == 4
    stages = json.loads((ROOT / data["current_stage_confirmations"]).read_text())
    assert stages["optimized_manifest_sha256"] == confirmations["optimized_manifest_sha256"]
    assert stages["fixture_sha256"] == confirmations["fixture_sha256"]
    assert len(stages["inventory"]) == 22
    assert sum(len(s["instances"]) for s in stages["inventory"].values()) == 770
    assert stages["compiled_graph_bridge"]["decode"]["full_logits_exact"] == 320
    assert stages["negative_controls_passed_groups"] == stages["cache_restored_groups"] == 40
    assert all(v["positions"] == v["full_logits_exact"] == v["top20_set_exact"] == v["top20_order_exact"] == 320
               for s in stages["stages"].values() for v in s.values())
    run_label = (
        f"; run [qualified speed refresh]({data['current_qualification_document']})"
        if data.get("current_qualification_document") else
        f"; base {commit_marker(profile_commit)} + recorded working-tree changes"
        if data.get("uncommitted_qualification") else f"; run {commit_marker(profile_commit)}"
    )
    assert (
        f"Current GPU activity per retained compiled M8 cycle (0K / 60K / 200K; milliseconds unless explicitly marked; {matched['measurement_date']}"
        f"{run_label})"
        in readme
    )
    compiled_table = readme.split("## Compiled backend stages\n\n", 1)[1].split(
        "\n\nThe table restores", 1
    )[0]
    assert "exact source hashes in capture" not in compiled_table.splitlines()[0]
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
        f"**24. Global-{matched.get('target_head', 'global256').removeprefix('global')} target head**",
        "**25. Other GPU bookkeeping**",
        "**26. Estimated runtime overhead**",
        "**Total reconstructed round (stages 1–26)**",
    ]
    assert "| **3. Input preparation and cache metadata** |" not in readme
    assert "| **12. RoPE and layout** |" not in readme
    assert "| **22. Target sampling and acceptance bookkeeping** |" not in readme
    assert "| **24. Forced replay control** |" not in readme
    assert "the ↳ rows are detail-only inclusion records and add no timing" in readme
    assert "zero rows have no separately emitted scope" not in readme
    assert "Last relevant code commit / change" in readme
    assert "The provenance column identifies the last relevant code change" in readme
    counts = " / ".join(str(matched["contexts"][c]["included_rounds"]) for c in matched["context_order"])
    assert counts + " complete M8 cycles" in readme
    assert "0K / 60K / 200K" in readme
    from compute_stage26_residual import compute
    control = json.loads((ROOT / data["matched_stage_control"]).read_text())
    audit = compute(matched, control)
    residuals = " / ".join(f"{audit['contexts'][c]['union_corrected_difference_ms']:.3f}" for c in matched["context_order"])
    totals = " / ".join(f"{audit['contexts'][c]['full_uninstrumented_round_ms']:.3f}" for c in matched["context_order"])
    assert f"| **26. Estimated runtime overhead** | {residuals} (estimate)" in readme
    assert f"| **Total reconstructed round (stages 1–26)** | **{totals}**" in readme
    assert "old forced-replay subtraction is superseded" in readme
    assert "profile-cycle wall time minus the 25 named stage kernel totals" not in readme
    assert data["stage_profile_2k"]["stage26_benchmark"]["status"] == "unqualified_as_runtime_gap"
    assert data["stage_profile_2k"]["stage26_benchmark"]["artifact"].endswith(
        "stage26-control-20260921.json"
    )
    if "kernel_groups" in matched["contexts"]["60K"]:
        count = matched["contexts"]["60K"]["included_rounds"]
        assert f"Activity records in {count:,} retained cycles" in readme
        assert "same " + f"{count:,} retained 60K cycles" in readme
    else:
        assert "Calls in 6 retained cycles" in readme
    assert readme.index("## Global-512 target-head") < readme.index("## Benchmarks")
    assert readme.count("## Global-512 target-head") == 1
    assert readme.count("## Benchmarks") == 1
    assert "## Global-512 target-head benchmark" not in readme
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
    chained = json.loads(
        (ROOT / "benchmarks/results/pi-coding-json-compaction.json").read_text()
    )
    assert f"rerun status: `{chained['status']}`" in readme
    checkpoint = chained["stages"][-1]["checkpoint_validation"]
    assert ("The checkpoint format failed validation" in readme) is not checkpoint["passed"]
    assert "phase_token_counts_cover_output=false" in readme
    assert "peak_3s_tokens_per_second" in readme
    assert "### Known remaining symptoms and likely causes" in readme
    assert readme.index("### Known remaining symptoms and likely causes") > readme.index(
        "The histogram below is generated from the complete per-round records"
    )
    assert "#### Changes since `cbbf495`" not in readme
    assert "### Remaining symptoms and likely causes" not in readme
    assert "Status: pending fix" not in readme
    remaining = readme.split("### Known remaining symptoms and likely causes", 1)[1].split("\n## ", 1)[0]
    if data.get("full_graph_repair"):
        assert "The missing full-graph preparation hook is repaired" in remaining
        assert data["full_graph_repair"] in remaining
        assert "does not prove that all HIP/ROCr queue" in remaining
    else:
        assert "Occasional long-context pauses remain" in remaining
        assert "trace limitation" in remaining
    assert "lacks a separate reasoning channel" in remaining
    assert data["matched_stage_control"] in remaining
    for obsolete in ("cbbf495", "abb7668", "b8d6810", "September 20 chained run", "September 21"):
        assert obsolete not in remaining
    assert "Native-runtime validation of the repaired collector is still required" not in remaining
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


def test_workload_table_uses_result_values_and_validation_status():
    report = json.loads(
        (ROOT / "benchmarks/results/pi-coding-json-compaction.json").read_text()
    )
    report["sampling"]["top_k"] = 41
    report["status"] = "complete"
    for row in report["stages"]:
        row["phase_token_counts_cover_output"] = True
        if row["thinking_enabled"]:
            row["reasoning_channel_observed"] = True
    coding = report["stages"][0]
    coding.update(
        generated_tokens=12345,
        mean_generation_round_ms=12.345,
        post_first_tokens_per_second=321.234,
        peak_3s_tokens_per_second=456.789,
        acceptance_rate=0.45678,
    )
    checkpoint = report["stages"][-1]
    checkpoint["sampling"]["temperature"] = 0.25
    checkpoint["checkpoint_validation"].update(
        marker_valid=True, headings_valid=True, passed=True
    )

    rendered = "\n".join(render_chained_workload_results(report))
    coding_line = next(line for line in rendered.splitlines() if line.startswith("| Coding task |"))
    assert "12,345" in coding_line
    assert "12.35 ms | 321.23 tok/s | 456.79 tok/s | 45.68%" in coding_line
    assert "top-k 41" in rendered
    assert "compaction uses temperature 0.25" in rendered
    assert "rerun status: `complete`" in rendered
    assert "The checkpoint format passed" in rendered
    assert "format failed" not in rendered
    assert "No separate reasoning channel" not in rendered
    assert "phase_token_counts_cover_output=false" not in rendered


def test_workload_head_has_no_silent_global256_default():
    with pytest.raises(ValueError, match="target head is not recorded"):
        workload_target_head({})
    assert workload_target_head({"runtime": {"target_head": "global512"}}) == "global512"


def test_shared_workload_cannot_borrow_another_capture_head():
    report = json.loads((ROOT / "benchmarks/results/pi-coding-json-compaction.json").read_text())
    report["suite_capture_id"] = "unrelated-run"
    with pytest.raises(ValueError, match="identity mismatch: suite_capture_id"):
        workload_target_head(report)


def test_current_context_results_account_for_every_round():
    report = json.loads(
        (ROOT / "benchmarks/results/pi-coding-contexts.json").read_text()
    )
    for context in ("0K", "60K", "200K"):
        capture = report["contexts"][context]["round_capture"]
        records = capture["records"]
        assert capture["status"] == "captured"
        assert capture["missing_round_numbers"] == []
        assert capture["duplicate_round_numbers"] == []
        assert len(records) == capture["record_count"]
        assert [row["round"] for row in records] == list(range(1, len(records) + 1))
        assert sum((row["draft_tokens"] or 0) > 0 for row in records) == capture["expected_rounds"]
        values = [row["round_ms"] for row in records if row["round_ms"] is not None]
        histogram = capture["histogram"]
        assert len(values) == histogram["measured_round_count"] == capture["measured_round_count"]
        assert sum(row["count"] for row in histogram["bins"]) == len(values)
        assert math.isclose(histogram["mean_ms"], sum(values) / len(values), abs_tol=1e-8)


def test_hugepage_repair_evidence_matches_the_full_rerun():
    directory = ROOT / "benchmarks/results"
    evidence = json.loads((directory / "huge-page-promotion-20260923.json").read_text())
    for name, expected in evidence["rerun"]["source_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    results = json.loads((directory / evidence["rerun"]["context_result"]).read_text())
    for context, row in evidence["rerun"]["contexts"].items():
        observed = results["contexts"][context]
        values = [r["round_ms"] for r in observed["round_capture"]["records"] if r["round_ms"] is not None]
        assert row["timed_rounds"] == len(values)
        assert row["max_ms"] == max(values)
        assert row["spikes_over100ms"] == sum(value > 100 for value in values)
        assert row["generated_tokens"] == observed["generated_tokens"]
        assert row["output_sha256"] == observed["output_sha256"]
        assert row["same_output_as_previous"] is True
    scheduler = ROOT / "experiments/radiance-public/radiance_fair_scheduler.py"
    assert hashlib.sha256(scheduler.read_bytes()).hexdigest() == evidence["runtime"]["scheduler_sha256"]
    assert evidence["repair"]["global_thp_changed"] is False
    assert evidence["repair"]["per_round_syscalls_added"] == 0
    for comparison in evidence["intervention_comparisons"]:
        assert comparison["output_exact"] is True
        assert comparison["before"]["output_sha256"] == comparison["after"]["output_sha256"]
        assert comparison["before"]["generated_tokens"] == comparison["after"]["generated_tokens"]
    failures = evidence["rerun"]["validation_failures"]
    assert failures["contexts"] == results["validation_failures"]
    assert failures["chained"] == json.loads((directory / evidence["rerun"]["chained_result"]).read_text())["validation_failures"]


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


def test_worker_details_use_the_same_clipped_activity_as_stage_totals():
    tail = {**kernel(950, 100), "name": "draft"}
    projection = {**kernel(1100, 300), "name": "projection"}
    head = {**kernel(1500, 200), "name": "head"}
    cpu = {"ph": "X", "cat": "cpu_op", "ts": 1100, "dur": 800, "name": "observer"}
    markers = [{"ph": "X", "cat": "user_annotation", "name": f"qwen_timing_round/{i}",
                "ts": t, "dur": 900} for i, t in enumerate((1000, 2000))]
    stages = {id(tail): "Drafter", id(projection): "MLP down projection", id(head): "Target head (global512)"}
    targets = [(('target_body', 0), []), (('target_body', 0), []), (('target_body', 1100), [])]
    result = measure_worker_windows([tail, projection, head, cpu, *markers], targets, [2],
                                    stages, {}, "global512", {id(projection): 7})
    row = result['rounds'][0]
    assert row['kernel_sum_ms'] == pytest.approx(.55)
    assert row['clipped_boundary_activity_ms'] == pytest.approx(.05)
    assert result['layers_ms'] == {7: {'MLP down projection': .3}}
    assert sum(k['activity_records'] for k in result['kernel_groups']) == 3
    assert sum(k['ms_per_round'] for k in result['kernel_groups']) == pytest.approx(.55)


@pytest.mark.parametrize("duration", [1, 1000000])
def test_trace_stage_order_rejection_is_independent_of_round_duration(duration):
    events = []
    for layer in range(64):
        mix = (["shared_decode", "shared_merge"] * 2 if layer % 4 == 3 else
               ["causal_conv_update", "stock_gdn_scan", "gdn_norm_quant_kernel"])
        names = ["norm_quant<true, 512>", "radiance_mxfp4_fp8_gemm_decode<8, 128, 1>",
                 *mix, "radiance_mxfp4_fp8_gemm_decode<8, 128, 4>", "norm_quant<true, 512>",
                 "radiance_mxfp4_fp8_gemm_decode<8, 128, 1>", "silu_kernel",
                 "radiance_mxfp4_fp8_gemm_decode<8, 128, 4>"]
        events.extend({"name": name, "dur": duration} for name in names)
    assert target_inventory_issue(events) is None
    # Reproduce the observed inverted input-normalization/projection timestamp order.
    events[0], events[1] = events[1], events[0]
    assert "inconsistent stage order (layer 0, mix)" in target_inventory_issue(events)


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


def test_independent_m1_evidence_survives_timing_regeneration():
    from render_current_tables import render_stage_profile_table

    data = json.loads((ROOT / 'benchmarks/results/coherence-current.json').read_text())
    evidence = json.loads((ROOT / 'benchmarks/results/eager-m1-readme-evidence.json').read_text())
    rendered = '\n'.join(render_stage_profile_table(data))
    assert len(evidence['rows']) == 31
    for cell in evidence['rows'].values():
        assert cell in rendered
    assert 'not probabilities of being bug-free' in rendered
    assert 'arbitrary inputs' in rendered
