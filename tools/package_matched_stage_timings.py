#!/usr/bin/env python3
"""Package numeric matched-timing evidence; private traces/fixtures stay local."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from compute_stage26_residual import compute

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("control_before", "profile", "control_after")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def runtime(metadata):
    return {k: metadata[k] for k in ("enforce_eager", "compilation_mode", "graph_mode", "effective_capacity", "diagnostic_sources")} | {
        "repair_bundle": metadata["repair"]["bundle"],
        "performance_manifest": metadata["performance"]["manifest"],
        "gemm_binary_sha256": metadata["performance"]["gemm_dispatch"]["binary_sha256"],
    }


def package(args):
    private = args.captures
    analyses = read(private / "analysis.json")["contexts"]
    analyses.update(read(private / "analysis-200K.json")["contexts"])
    if list(analyses) != ["0K", "60K", "200K"]:
        raise ValueError("all three contexts are required")
    original = read(private / "production-inspect.json")
    env = dict(item.split("=", 1) for item in original["Config"]["Env"])
    workers = {c: {arm: read(private / "runs" / c / f"{c}-{arm}" / "worker.json") for arm in ARMS} for c in analyses}
    runtimes = {arm: {c: runtime(workers[c][arm]["metadata"]) for c in analyses} for arm in ARMS}
    if len({digest(value) for value in runtimes.values()}) != 1:
        raise ValueError("numerical runtime differs across the paired arms")
    sources = (
        "experiments/radiance-public/matched_stage_profile_worker.py",
        "experiments/radiance-public/benchmark_matched_stage_timings.py",
        "tools/analyze_release_timings.py", "tools/analyze_matched_stage_timings.py",
        "tools/compute_stage26_residual.py", "tools/package_matched_stage_timings.py",
        "reports/d7-rdna4-2026-09-17/analyze_compiled_trace.py",
    )
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    for name in sources[:2]:
        if hashes[name] != hashlib.sha256((private / Path(name).name).read_bytes()).hexdigest():
            raise ValueError("the benchmark source differs from the executed copy")
    binding = {
        "image_id": original["Image"], "optimized_manifest_sha256": args.manifest_sha256,
        "worker": "matched_stage_profile_worker.MatchedStageWorker",
        "recorded_runtime": runtimes["profile"]["0K"],
        "environment": {k: env[k] for k in ("RADIANCE_VERIFY_HEAD", "RADIANCE_VERIFY_HEAD_GLOBAL_TOPK", "RADIANCE_DRAFT_RERANK")},
        "source_sha256": hashes,
        "measurement_source_state": f"{args.source_commit} plus recorded working-tree source; pinned September 23 numerical backend",
        "privacy": "No chat text or token arrays included.",
    }
    # A host-memory policy can change round stalls without changing the
    # numerical payload. Bind it separately so a rerun cannot silently claim
    # to qualify a different scheduler or snapshot-runtime release.
    runtime_binding = private / "runtime-binding.json"
    if runtime_binding.exists():
        host_runtime = read(runtime_binding)
        sources = host_runtime["source_sha256"]
        for name in ("radiance_fair_scheduler.py", "snapshot-abi-chat-cache-v1.json",
                     "runtime-radiance-1.0.16.json", "optimized-release.json"):
            local = ROOT / "experiments/radiance-public" / name
            if hashlib.sha256(local.read_bytes()).hexdigest() != sources[name]:
                raise ValueError(f"measured host runtime differs from source: {name}")
        if sources["optimized-release.json"] != args.manifest_sha256:
            raise ValueError("host runtime and numerical release identities differ")
        binding["host_runtime"] = host_runtime
        binding["host_runtime_binding_sha256"] = hashlib.sha256(runtime_binding.read_bytes()).hexdigest()
        binding["host_page_policy_observations"] = {}
        for context in analyses:
            observation = read(private / f"{context}-host-page-policy.json")
            if observation["allocated_bytes"]:
                if observation["host_page_policy"] != "no_hugepage_promotion":
                    raise ValueError(f"{context}: pinned RAM was not protected during measurement")
                if observation["host_page_policy_bytes"] < observation["allocated_bytes"]:
                    raise ValueError(f"{context}: page policy does not cover the pinned allocation")
            binding["host_page_policy_observations"][context] = observation
    if binding["environment"]["RADIANCE_VERIFY_HEAD_GLOBAL_TOPK"] != "256":
        raise ValueError("Global-256 is required")
    identity = {"measurement_commit": args.source_commit, "execution_identity": digest(binding),
                "fixture_identity": digest({c: v["prompt_sha256"] for c, v in analyses.items()})}
    common = {"round_boundary": "worker_execute_entry_to_next_worker_execute_entry",
              "workload_schedule_sha256": digest({c: v["schedule_sha256"]["profile"] for c, v in analyses.items()}),
              "runtime_artifact_sha256": digest(runtimes["profile"]), "added_synchronization": False}
    profile = {
        "schema": "urn:coherence:matched-compiled-stage-profile:v1", "status": "matched_estimate",
        "measurement_date": "2026-09-23", "context_order": list(analyses), "stage_count": 25,
        "stage_order": read(ROOT / "benchmarks/results/compiled-global256-stage-profile-1200.json")["stage_order"],
        "production_timing_eligible": True, "binding": binding, "stage26_execution": identity,
        "observer_effect": {"first_use_triton_jit_observed": False, "zero_observer_effect_proven": False,
                            "note": "No recorded compilation/module-load events. CPU recording/export cost excluded; indirect clock/scheduling effects are not proved absent."},
        "timing_contract": common | {"stage_metric": "gpu_activity_duration", "cpu_scope_time_included": False, "per_stage_event_probes": False},
        "observer_comparison": {"status": "measured", "contexts": {}}, "contexts": {},
        "sampling": {"temperature": 1., "top_p": .95, "top_k": 40, "seed": 0},
        "selection": "Natural sampling/EOS. Complete M8 inventories with a next worker-entry marker; exclude two trace-activation cycles per chunk and structurally incomplete records, never by duration. Controls use the identical indices. All native rounds remain in the capture, including outliers and partial-width steps. Requested 1,152 traced rounds per context; natural EOS can end earlier.",
    }
    control = {"schema": "urn:qwen:stage26-uninstrumented-residual-v1", "status": "complete", **identity,
               "context_order": list(analyses), "contexts": {}, "timing_contract": common | {
                   "stage_profiler_enabled": False, "forced_replay_hooks": False,
                   "scope": "Production telemetry plus one host entry clock/record per round; no stage events, tensor reads, forced sampling or added synchronization."}}
    for context, result in analyses.items():
        native = read(private / "runs" / context / "report.json")["contexts"][context]["arms"]
        sources, incomplete, rounds = [], {}, 0
        for path in sorted((private / "runs" / context / f"{context}-profile").glob("chunk-*/timings.json")):
            trace = read(path)
            if trace["compilation_events"] or trace["analyzer_sha256"] != hashes["tools/analyze_release_timings.py"]:
                raise ValueError("compiled window or analyzer identity does not match")
            rounds += trace["profile_rounds"]
            incomplete[path.parent.name] = trace.get("incomplete_inventory_reasons", {})
            sources.append({"trace_sha256": trace["trace_sha256"], "analyzer_sha256": trace["analyzer_sha256"],
                            "rounds": trace["profile_rounds"], "retained": len(trace["worker_timing"]["rounds"])})
        before, after = [statistics.fmean(result["samples"][arm]) for arm in ("control_before", "control_after")]
        busy = result["round_timing_ms"]["gpu_busy_ms"]
        profile["contexts"][context] = {
            "stages_ms": result["stages_ms"], "stage_sum_ms": sum(result["stages_ms"].values()),
            "round_timing_ms": result["round_timing_ms"], "profile_rounds": rounds,
            "included_rounds": result["retained_rounds"], "trace_chunks": result["trace_chunks"],
            "source_traces": sources, "incomplete_inventories": incomplete, "decode_indices": result["decode_indices"],
            "generated_tokens_per_arm": result["generated_tokens_per_arm"],
            **{k: result[k] for k in ("fixture_sha256", "prompt_sha256", "schedule_sha256")},
            "same_output_and_accepted_schedule": True,
            "marker_clock_mean_difference_ms": result["marker_clock_difference_mean_ms"],
            "marker_clock_max_difference_ms": result["marker_clock_difference_max_abs_ms"],
        }
        profile["observer_comparison"]["contexts"][context] = {
            "profiled_round_ms": result["samples"]["profile"],
            "control_before_round_ms": result["samples"]["control_before"],
            "control_after_round_ms": result["samples"]["control_after"], "mean_delta_ms": result["observer_delta_ms"],
            "whole_control_before_mean_ms": native["control_before"]["mean_generation_round_ms"],
            "whole_control_after_mean_ms": native["control_after"]["mean_generation_round_ms"],
            "whole_native_round_counts": {arm: native[arm]["round_capture"]["record_count"] for arm in ARMS},
            "remainder_before_after_range_ms": [min(before, after) - busy, max(before, after) - busy],
        }
        control["contexts"][context] = {
            "full_uninstrumented_round_ms": result["control_mean_ms"],
            "round_samples": result["samples"]["control_before"] + result["samples"]["control_after"],
            "matched_rounds_per_arm": result["retained_rounds"], "control_before_mean_ms": before, "control_after_mean_ms": after,
            "output_sha256": {arm: native[arm]["output_sha256"] for arm in ARMS}, "schedule_sha256": result["schedule_sha256"],
        }
    audit = compute(profile, control)
    if audit["status"] != "matched_estimate":
        raise ValueError(audit)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("compiled-global256-stage-profile-20260923.json", profile),
                       ("stage26-control-20260923.json", control), ("matched-stage-residual-20260923.json", audit)):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    package(parser.parse_args())
