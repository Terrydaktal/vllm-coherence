#!/usr/bin/env python3
"""Reduce natural serving controls and kernel traces without exposing text."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_release_timings import analyze

ANALYZER_SHA256 = hashlib.sha256(Path(__file__).with_name("analyze_release_timings.py").read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def schedule(arm, worker):
    return {
        "output_sha256": arm["output_sha256"],
        "acceptance": [[r["draft_tokens"], r["accepted_tokens"]]
                       for r in arm["round_capture"]["records"]],
        "worker_shapes": [r["scheduled_tokens"] for r in worker["rows"]],
    }


def reduce_context(root, context, head="global256"):
    report = json.loads((root / context / "report.json").read_text())
    section = report["contexts"][context]
    workers = {arm: json.loads((root / context / f"{context}-{arm}" / "worker.json").read_text())
               for arm in ("control_before", "profile", "control_after")}
    schedules = {arm: digest(schedule(section["arms"][arm], worker))
                 for arm, worker in workers.items()}
    if len(set(schedules.values())) != 1:
        raise ValueError(f"{context}: output/acceptance/shape schedule differs between arms")
    for arm in workers:
        if section["arms"][arm]["round_capture"]["status"] != "captured":
            raise ValueError(f"{context}: incomplete native round log in {arm}")
        m = workers[arm]["metadata"]
        if m["enforce_eager"] or not m["compilation_mode"] or "PIECEWISE" not in m["graph_mode"]:
            raise ValueError("not the compiled piecewise path")
    rows, sources, chunks = [], [], []
    layers, kernels = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: [0, 0.0])
    for number, chunk in enumerate(workers["profile"]["chunks"]):
        # The pinned release's GraphObservation predates its explicit file list;
        # it emits exactly one profile-trace.json in each isolated chunk root.
        files = chunk["observation"].get("profile_files") or [
            f"chunk-{number:03d}/profile-trace.json"
        ]
        for filename in files:
            path = root / context / f"{context}-profile" / Path(filename).parent.name / Path(filename).name
            if not path.exists():
                path = Path(str(path) + ".gz")
            raw = path.read_bytes()
            if path.suffix == ".gz":
                raw = gzip.decompress(raw)
            result_path = path.parent / "timings.json"
            signature = hashlib.sha256(raw).hexdigest()
            analyzer_hash = ANALYZER_SHA256
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result.get("trace_sha256") != signature or result.get("analyzer_sha256") != analyzer_hash or result.get("target_head") != head:
                    result = None
            else:
                result = None
            if result is None:
                result = analyze(raw, head, worker_boundaries=True,
                                 admitted_decode_indices={r["decode_index"] for r in workers["profile"]["rows"]
                                                          if r["scheduled_tokens"] == 8})
                result["analyzer_sha256"] = analyzer_hash
                result_path.write_text(json.dumps(result, indent=2) + "\n")
            detail = result["worker_timing"]
            count = len(detail["rounds"])
            rows.extend(detail["rounds"])
            for layer, stages in detail["layers_ms"].items():
                for stage, value in stages.items():
                    layers[layer][stage] += value * count
            for kernel in detail["kernel_groups"]:
                total = kernels[(kernel["stage"], kernel["kernel"])]
                total[0] += kernel["activity_records"]
                total[1] += kernel["ms_per_round"] * count
            if result.get("compilation_events"):
                raise ValueError(f"compilation/module load inside profile: {result['compilation_events']}")
            sources.append({"trace_sha256": signature, "file": str(path.relative_to(root))})
            chunks.append({"profile_rounds": result["profile_rounds"],
                           "retained_rounds": len(result["worker_timing"]["rounds"]),
                           "graph_launches": sorted(set(result["graph_launches_per_round"]))})
            print(json.dumps({"context": context, "chunk": path.parent.name,
                              "retained": len(result["worker_timing"]["rounds"])}), flush=True)
    rows.sort(key=lambda r: r["decode_index"])
    if not rows or len({r["decode_index"] for r in rows}) != len(rows):
        raise ValueError("empty/duplicate matched round selection")
    samples = {arm: [] for arm in workers}
    marker_differences = []
    for row in rows:
        i = row["decode_index"]
        for arm, worker in workers.items():
            current, following = worker["rows"][i:i+2]
            if current["decode_index"] != i or following["decode_index"] != i + 1:
                raise ValueError("missing host entry boundary")
            elapsed = (following["entry_ns"] - current["entry_ns"]) / 1e6
            samples[arm].append(elapsed)
            if arm == "profile":
                marker_differences.append(row["elapsed_ms"] - elapsed)
    names = json.loads((Path(__file__).parents[1] / "benchmarks/results/compiled-global256-stage-profile-1200.json").read_text())["stage_order"]
    names = [name.replace("Target head (global256)", f"Target head ({head})") for name in names]
    if isinstance(names[0], dict):
        raise TypeError("unexpected stage schema")
    found = set().union(*(r["stages_ms"] for r in rows))
    if found - set(names):
        raise ValueError(f"unknown stages: {found-set(names)}")
    means = {k: statistics.fmean(r[k] for r in rows)
             for k in ("elapsed_ms", "gpu_busy_ms", "kernel_sum_ms", "gpu_overlap_ms", "overhead_ms")}
    stages = {name: statistics.fmean(r["stages_ms"].get(name, 0.) for r in rows) for name in names}
    if not math.isclose(sum(value[1] for value in kernels.values()) / len(rows),
                        sum(stages.values()), abs_tol=1e-8):
        raise ValueError("granular kernel detail differs from stage accounting")
    control = samples["control_before"] + samples["control_after"]
    return {
        "context": context, "fixture_sha256": section["fixture_sha256"],
        "prompt_sha256": section["prompt_sha256"], "schedule_sha256": schedules,
        "identical_outputs_and_schedules": True, "generated_tokens_per_arm": section["arms"]["profile"]["generated_tokens"],
        "sampling": report["sampling"], "retained_rounds": len(rows),
        "decode_indices": [r["decode_index"] for r in rows], "trace_chunks": chunks,
        "sources": sources, "stages_ms": stages, "round_timing_ms": means,
        "layers_ms": {layer: {stage: value / len(rows) for stage, value in values.items()}
                      for layer, values in sorted(layers.items(), key=lambda item: int(item[0]))},
        "kernel_groups": [{"stage": stage, "kernel": name, "activity_records": value[0],
                           "ms_per_round": value[1] / len(rows)}
                          for (stage, name), value in sorted(kernels.items())],
        "samples": samples, "control_mean_ms": statistics.fmean(control),
        "profile_mean_ms": statistics.fmean(samples["profile"]),
        "observer_delta_ms": statistics.fmean(samples["profile"]) - statistics.fmean(control),
        "union_corrected_remainder_ms": statistics.fmean(control) - means["gpu_busy_ms"],
        "marker_clock_difference_mean_ms": statistics.fmean(marker_differences),
        "marker_clock_difference_max_abs_ms": max(abs(x) for x in marker_differences),
        "zero_observer_effect_proven": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--contexts", default="0K,60K,200K")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--head", choices=["global256", "global512"], default="global256")
    args = p.parse_args()
    result = {"schema": "urn:coherence:matched-stage-analysis:v1", "contexts": {}}
    for context in args.contexts.split(","):
        result["contexts"][context] = reduce_context(args.root, context, args.head)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: {k: row[k] for k in ("retained_rounds", "control_mean_ms", "profile_mean_ms", "observer_delta_ms", "union_corrected_remainder_ms")}
                      for key, row in result["contexts"].items()}))


if __name__ == "__main__":
    main()
