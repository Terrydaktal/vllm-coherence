import hashlib
import json
import subprocess
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
    assert result["status"] == "diagnostic_only"
    assert result["real_gap_ms"] is None
    assert result["zero_observer_effect_proven"] is False
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


def test_stage26_accepts_published_stages_ms_spelling():
    published = profile()
    for context in CONTEXTS:
        published["contexts"][context] = {
            "stages_ms": published["contexts"][context].pop("stages")
        }
    result = compute(published, control((330, 331, 332)))
    assert result["contexts"]["200K"]["residual_ms"] == 7.0


def test_stage26_rejects_profile_with_only_nested_control_identity():
    published = profile()
    identity = {
        "measurement_commit": published.pop("measurement_commit"),
        "execution_identity": published.pop("execution_identity"),
        "fixture_identity": published.pop("fixture_identity"),
    }
    published["stage26_benchmark"] = identity
    with pytest.raises(ValueError, match="measurement_commit"):
        compute(published, control((330, 331, 332)))


def test_published_compiled_profile_and_control_are_not_qualified_runtime_gaps():
    root = Path(__file__).resolve().parents[1]
    profile = json.loads(
        (root / "benchmarks/results/compiled-global256-stage-profile-1200.json").read_text()
    )
    control_artifact = json.loads(
        (root / "benchmarks/results/stage26-control-20260921.json").read_text()
    )
    result = compute(profile, control_artifact)
    assert result["status"] == "diagnostic_only"
    assert result["real_gap_ms"] is None
    assert "stage profile is not qualified for production timing" in result["qualification_issues"]
    assert "control timing must exclude forced_replay_hooks" in result["qualification_issues"]
    assert [
        round(result["contexts"][context]["residual_ms"], 3)
        for context in CONTEXTS
    ] == [8.060, 17.393, 42.013]


def test_stage26_uses_union_instead_of_double_counting_overlapping_kernels():
    source = profile()
    for row in source["contexts"].values():
        row["round_timing_ms"] = {"gpu_busy_ms": 315.0, "gpu_overlap_ms": 10.0}
    result = compute(source, control((320, 320, 320)))
    for row in result["contexts"].values():
        assert row["instrumented_stage_sum_ms"] == 325.0
        assert row["residual_ms"] == -5.0
        assert row["union_corrected_difference_ms"] == 5.0


def test_stage26_rejects_a_fabricated_union():
    source = profile()
    for row in source["contexts"].values():
        row["round_timing_ms"] = {"gpu_busy_ms": 314.0, "gpu_overlap_ms": 10.0}
    with pytest.raises(ValueError, match="union does not reconcile"):
        compute(source, control((330, 331, 332)))


def test_scope_events_cannot_be_declared_profiler_free_kernel_time():
    source = profile()
    source["production_timing_eligible"] = True
    source["timing_contract"] = {"stage_metric": "hip_scope_interval"}
    result = compute(source, control((330, 331, 332)))
    assert result["status"] == "diagnostic_only"
    assert "stage values are not bound to GPU activity start/end timestamps" in result["qualification_issues"]


def test_matching_labels_alone_cannot_certify_no_observer_effect():
    source = profile()
    source["production_timing_eligible"] = True
    source["observer_effect"] = {"first_use_triton_jit_observed": False}
    source["timing_contract"] = {
        "stage_metric": "gpu_activity_duration",
        "cpu_scope_time_included": False,
        "per_stage_event_probes": False,
        "added_synchronization": False,
        "round_boundary": "sample_complete_to_sample_complete",
        "workload_schedule_sha256": "s" * 64,
        "runtime_artifact_sha256": "r" * 64,
    }
    other = control((330, 331, 332))
    other["timing_contract"] = {
        **source["timing_contract"],
        "stage_profiler_enabled": False,
        "forced_replay_hooks": False,
    }
    result = compute(source, other)
    assert result["status"] == "diagnostic_only"
    assert "matched profiler-on/off observer effect has not been measured" in result["qualification_issues"]
    assert result["real_gap_ms"] is None


def test_measured_label_without_observer_samples_is_rejected():
    source = profile()
    source["observer_comparison"] = {"status": "measured"}
    result = compute(source, control((330, 331, 332)))
    assert result["status"] == "diagnostic_only"
    assert "60K: observer comparison lacks valid profiled_round_ms samples" in result["qualification_issues"]


def test_archived_audit_is_bound_to_its_measured_sources_and_full_histogram():
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / "benchmarks/results/stage-timing-audit-20260923.json").read_text())
    for path, expected in audit["sources"].items():
        assert hashlib.sha256((root / audit.get("source_artifacts", {}).get(path, path)).read_bytes()).hexdigest() == expected
    histogram = audit["histogram"]
    source = root / histogram["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == histogram["source_sha256"]
    contexts = json.loads(source.read_text())["contexts"]
    for name, counts in histogram["contexts"].items():
        capture = contexts[name]["round_capture"]
        assert counts["measured_round_count"] == sum(row["count"] for row in capture["histogram"]["bins"])
        assert counts["missing_round_numbers"] == []
        assert counts["duplicate_round_numbers"] == []


def test_native_matched_evidence_keeps_tracing_slowdown_out_of_runtime_gaps():
    root = Path(__file__).resolve().parents[1]
    current = json.loads((root / "benchmarks/results/coherence-current.json").read_text())
    profile_data = json.loads((root / current["matched_stage_profile"]).read_text())
    control_data = json.loads((root / current["matched_stage_control"]).read_text())
    result = compute(profile_data, control_data)
    assert result["status"] == "matched_estimate"
    assert result["zero_observer_effect_proven"] is False
    for path, expected in profile_data["binding"]["source_sha256"].items():
        # Historical evidence qualifies its recorded source, not later runner
        # refactors. Still verify every byte against the published source hash.
        source = subprocess.check_output([
            "git", "-C", str(root), "show",
            f"{current['current_qualification_commit']}:{path}",
        ])
        assert hashlib.sha256(source).hexdigest() == expected
    host = profile_data["binding"]["host_runtime"]
    for name, expected in host["source_sha256"].items():
        assert hashlib.sha256((root / "experiments/radiance-public" / name).read_bytes()).hexdigest() == expected
    observed = host["worker_page_policy"]
    if observed["allocated_bytes"]:
        assert observed["host_page_policy"] == "no_hugepage_promotion"
        assert observed["host_page_policy_bytes"] >= observed["allocated_bytes"]
    else:
        assert observed["host_page_policy"] == "unallocated"
        assert observed["host_page_policy_bytes"] == 0
    for context, observation in profile_data["binding"]["host_page_policy_observations"].items():
        if observation["allocated_bytes"]:
            assert observation["host_page_policy"] == "no_hugepage_promotion", context
            assert observation["host_page_policy_bytes"] >= observation["allocated_bytes"]
    for context in CONTEXTS:
        profiled = profile_data["contexts"][context]
        control_row = control_data["contexts"][context]
        observer = profile_data["observer_comparison"]["contexts"][context]
        count = profiled["included_rounds"]
        assert len(control_row["round_samples"]) == count * 2
        assert len(profiled["decode_indices"]) == count
        assert len(set(control_row["output_sha256"].values())) == 1
        assert len(set(control_row["schedule_sha256"].values())) == 1
        clean_mean = sum(control_row["round_samples"]) / (2 * count)
        traced_mean = sum(observer["profiled_round_ms"]) / count
        assert observer["mean_delta_ms"] == pytest.approx(traced_mean - clean_mean)
        remainder = result["contexts"][context]["union_corrected_difference_ms"]
        assert remainder + profiled["round_timing_ms"]["gpu_busy_ms"] == pytest.approx(clean_mean)
        assert remainder != pytest.approx(profiled["round_timing_ms"]["overhead_ms"])
        assert set(profiled["layers_ms"]) == {str(i) for i in range(64)}
        assert sum(k["ms_per_round"] for k in profiled["kernel_groups"]) == pytest.approx(profiled["stage_sum_ms"])
        for stage, expected in profiled["stages_ms"].items():
            assert sum(k["ms_per_round"] for k in profiled["kernel_groups"] if k["stage"] == stage) == pytest.approx(expected)
