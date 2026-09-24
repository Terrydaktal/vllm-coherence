#!/usr/bin/env python3
"""Measure stages and cycle overhead in a saved R9700 trace without GPU work."""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports/d7-rdna4-2026-09-17"))
from analyze_compiled_trace import launch_groups, projection, semantic_phase


def phase(layer, region, name):
    # A fused launch has one measured duration; its constituents cannot each be
    # assigned that duration without double counting.
    if "norm_quant<" in name:
        if region == "input":
            return (
                "Embedding + first input normalization"
                if layer == 0
                else "Layer input residual/normalization"
            )
        if region == "post":
            return "Post-attention/GDN residual/normalization"
        raise ValueError("norm/quant kernel outside admitted normalization boundary")
    if "gdn_norm_quant_kernel" in name:
        if region != "mix" or layer % 4 == 3:
            raise ValueError("GDN norm/quant in another layer or region")
        return "GDN output gated normalization"
    if "silu" in name:
        if region != "activation":
            raise ValueError("SiLU outside MLP activation boundary")
        return "MLP SiLU and gating"
    if "sigmoid" in name:
        if region != "mix" or layer % 4 != 3:
            raise ValueError("attention gate outside attention boundary")
        return "Attention output gating"
    resolved = semantic_phase(layer, region, name)
    # Keep the original 25-row profiler schema. New traces expose the GDN
    # output FP8 producer as a separate kernel, but the original profiler
    # attributed it to the combined GDN output gated-normalization stage.
    if resolved == "GDN output activation FP8 quantization":
        return "GDN output gated normalization"
    return resolved


def measure_round_windows(events, starts, selected, *, allow_transfers=False):
    """Measure complete GPU start-to-next-start cycles on one device.

    Kernel durations are summed for stage accounting; interval union measures
    occupied time without double counting concurrent streams. CPU annotations
    are not added to GPU time because their execution can overlap.
    """
    if (
        len(starts) < 2
        or not all(math.isfinite(value) for value in starts)
        or any(b <= a for a, b in pairwise(starts))
        or not selected
        or selected != sorted(set(selected))
        or any(i < 0 or i + 1 >= len(starts) for i in selected)
    ):
        raise ValueError("round timing requires ordered, bounded round windows")
    work = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        }:
            continue
        begin, duration = float(event["ts"]), float(event["dur"])
        if not math.isfinite(begin) or not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid GPU activity interval")
        work.append((begin, duration, event))
    rows = []
    for index in selected:
        start, stop = starts[index : index + 2]
        elapsed = stop - start
        intervals, devices, streams, durations = [], set(), set(), []
        for begin, duration, event in work:
            relative = begin - start
            end = relative + duration
            if relative >= elapsed or end <= 0:
                continue
            if relative < 0 or end > elapsed:
                raise ValueError("GPU activity crosses a round boundary")
            if event["cat"] != "kernel" and not allow_transfers:
                raise ValueError("GPU copy activity needs explicit stage accounting")
            devices.add(event["args"]["device"])
            streams.add(event["args"]["stream"])
            intervals.append((relative, end))
            durations.append(duration)
        if len(devices) != 1 or not intervals:
            raise ValueError("round timing requires work on exactly one GPU")
        occupied, end = [], 0.0
        for begin, finish in sorted(intervals):
            occupied.append(max(0.0, finish - max(begin, end)))
            end = max(end, finish)
        busy = math.fsum(occupied)
        kernel_sum = math.fsum(durations)
        rows.append(
            {
                "round": index,
                "start_offset_ms": (start - starts[0]) / 1000,
                "elapsed_ms": elapsed / 1000,
                "kernel_sum_ms": kernel_sum / 1000,
                "gpu_busy_ms": busy / 1000,
                "gpu_overlap_ms": max(0.0, kernel_sum - busy) / 1000,
                "overhead_ms": max(0.0, elapsed - busy) / 1000,
                "kernel_count": len(intervals),
                "streams": sorted(streams),
            }
        )
    fields = (
        "elapsed_ms",
        "kernel_sum_ms",
        "gpu_busy_ms",
        "gpu_overlap_ms",
        "overhead_ms",
    )
    return {
        "boundary": "First target GPU kernel to the first target GPU kernel of the next round.",
        "scope": "Profiled GPU cycle; gaps include launch/host/wait and possible profiler overhead. Individual causes are not attributed. Excludes request admission, prefill and Pi/transport latency.",
        "equation": "elapsed_ms = kernel_sum_ms - gpu_overlap_ms + overhead_ms",
        "mean": {
            key: math.fsum(row[key] for row in rows) / len(rows) for key in fields
        },
        "rounds": rows,
    }


def measure_worker_windows(events, targets, selected, stages_by_event, groups, head="global256", layers_by_event=None):
    """Use CPU entry markers also timed by the clean control, clipping GPU work.

    Markers are only boundaries. Their CPU duration is never added to a stage.
    In-flight work straddling a boundary is split there rather than discarded.
    """
    markers = sorted(
        (float(e["ts"]), float(e["dur"]), int(e["name"].split("/")[-1]))
        for e in events
        if e.get("ph") == "X" and e.get("cat") == "user_annotation"
        and e.get("name", "").startswith("qwen_timing_round/")
    )
    if not markers:
        raise ValueError("worker boundary markers missing")
    marker_starts = [m[0] for m in markers]
    # Include the previous round's asynchronous draft/bookkeeping tail when it
    # extends beyond the next worker entry. Target stages require full mapping.
    for (scope, _), group in groups.items():
        if scope != "target_body":
            stage = {"drafter": "Drafter", "target_vocabulary_head": f"Target head ({head})"}.get(scope, "Other GPU bookkeeping")
            for event in group:
                stages_by_event.setdefault(id(event), stage)
    activity = [e for e in events if e.get("ph") == "X"
                and e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}]
    rows, omissions = [], []
    layers_by_event = layers_by_event or {}
    layer_totals = defaultdict(lambda: defaultdict(float))
    kernel_totals = defaultdict(lambda: [0, 0.0])
    for index in selected:
        # Omit the first two cycles after trace activation, independently of
        # their duration. They can reflect profiler startup / clock recovery.
        if index < 2:
            omissions.append({"target_index": index, "reason": "trace activation boundary"})
            continue
        cpu_start = float(targets[index][0][1])
        marker_index = bisect.bisect_right(marker_starts, cpu_start) - 1
        if marker_index < 0 or marker_index + 1 >= len(markers):
            raise ValueError("target scope has no matching worker-entry interval")
        start, duration, decode_index = markers[marker_index]
        stop, _, next_index = markers[marker_index + 1]
        if not start <= cpu_start < start + duration or next_index != decode_index + 1:
            raise ValueError("worker marker does not contain target scope or misses a round")
        intervals, stages, count, clipped, devices = [], defaultdict(float), 0, 0.0, set()
        for event in activity:
            begin, end = float(event["ts"]), float(event["ts"]) + float(event["dur"])
            lo, hi = max(start, begin), min(stop, end)
            if hi <= lo:
                continue
            stage = stages_by_event.get(id(event))
            if event["cat"] != "kernel":
                stage = "Other GPU bookkeeping"
            if stage is None:
                raise ValueError("unclassified GPU work crosses a worker-entry window")
            stages[stage] += (hi - lo) / 1000
            milliseconds = (hi - lo) / 1000
            layer = layers_by_event.get(id(event))
            if layer is not None:
                layer_totals[layer][stage] += milliseconds
            kernel = kernel_totals[(stage, event["name"])]
            kernel[0] += 1
            kernel[1] += milliseconds
            intervals.append((lo - start, hi - start))
            devices.add(event["args"]["device"])
            clipped += max(0, hi - lo) if lo != begin or hi != end else 0
            count += 1
        if len(devices) != 1:
            raise ValueError("worker interval must contain exactly one GPU")
        occupied, last = 0.0, 0.0
        for lo, hi in sorted(intervals):
            occupied += max(0, hi - max(lo, last))
            last = max(last, hi)
        total = math.fsum(stages.values())
        rows.append({"decode_index": decode_index, "target_index": index,
                     "elapsed_ms": (stop - start) / 1000,
                     "gpu_busy_ms": occupied / 1000, "kernel_sum_ms": total,
                     "gpu_overlap_ms": max(0, total - occupied / 1000),
                     "overhead_ms": (stop - start - occupied) / 1000,
                     "clipped_boundary_activity_ms": clipped / 1000,
                     "kernel_count": count, "stages_ms": dict(stages)})
    if not rows:
        raise ValueError("no matched worker intervals retained")
    count = len(rows)
    kernels = [{"stage": stage, "kernel": name, "activity_records": values[0],
                "ms_per_round": values[1] / count}
               for (stage, name), values in sorted(kernel_totals.items())]
    if not math.isclose(sum(k["ms_per_round"] for k in kernels),
                        sum(r["kernel_sum_ms"] for r in rows) / count, abs_tol=1e-8):
        raise ValueError("worker kernel detail and stage totals disagree")
    return {"boundary": "worker_execute_entry_to_next_worker_execute_entry",
            "rounds": rows, "omissions": omissions,
            "layers_ms": {layer: {stage: value / count for stage, value in stages.items()}
                          for layer, stages in layer_totals.items()},
            "kernel_groups": kernels}


def target_inventory_issue(events):
    """Admit complete M8 inventories, including both short-context page groups."""
    points = [(i, projection(e["name"])) for i, e in enumerate(events) if projection(e["name"])]
    if [p for _, p in points] != ["split1", "split4", "split1", "split4"] * 64:
        return f"incomplete projection inventory ({len(points)}/256)"
    for layer in range(64):
        a, b, c, d = [points[layer * 4 + j][0] for j in range(4)]
        names = [e["name"] for e in events[a + 1:b]]
        if layer % 4 != 3:
            if (sum("conv" in n for n in names) != 1 or
                    sum("recurrent" in n or "stock_gdn_scan" in n for n in names) != 1):
                return f"incomplete GDN inventory (layer {layer})"
        else:
            decodes = sum("attn_decode" in n or "shared_decode" in n for n in names)
            merges = sum("splitkv_combine" in n or "shared_merge" in n for n in names)
            if decodes not in {1, 2} or merges != decodes:
                return f"incomplete attention inventory (layer {layer})"
        if sum("silu" in e["name"] for e in events[c + 1:d]) != 1:
            return f"incomplete activation inventory (layer {layer})"
        start = 0 if layer == 0 else points[layer * 4 - 1][0] + 1
        for begin, end, region in ((start, a, "input"), (a + 1, b, "mix"),
                                   (b + 1, c, "post"), (c + 1, d, "activation")):
            for event in events[begin:end]:
                try:
                    phase(layer, region, event["name"])
                except ValueError as error:
                    return f"inconsistent stage order (layer {layer}, {region}): {error}"
    return None


def analyze(raw: bytes, head: str, expected_rounds: int | None = None, *,
            worker_boundaries=False, admitted_decode_indices=None):
    trace = json.loads(raw)
    groups, graph_launches = launch_groups(trace["traceEvents"])
    targets = sorted((k, v) for k, v in groups.items() if k[0] == "target_body")
    if expected_rounds is not None and len(targets) != expected_rounds:
        raise ValueError(
            f"expected {expected_rounds} compiled rounds, found {len(targets)}"
        )
    if not targets or any(graph_launches[k] != 65 for k, _ in targets):
        raise ValueError("compiled rounds must contain 65 target graphs each")
    inventories = [
        tuple(sorted(Counter(e["name"] for e in v).items())) for _, v in targets
    ]
    inventory = Counter(inventories).most_common(1)[0][0]
    # A natural 0K completion legitimately switches from the short-context R4D
    # attention path to shared attention. A modal-inventory filter would discard
    # one real execution path and bias its mean. Both still have to pass the
    # per-layer projection, attention and recurrence checks below.
    complete = (list(range(len(targets))) if worker_boundaries else
                [i for i, value in enumerate(inventories) if value == inventory])
    incomplete = {}
    if worker_boundaries:
        incomplete = {i: issue for i, (_, events) in enumerate(targets)
                      if (issue := target_inventory_issue(events)) is not None}
        complete = [i for i in complete if i not in incomplete]
    if worker_boundaries and admitted_decode_indices is not None:
        markers = sorted((float(e["ts"]), int(e["name"].split("/")[-1]))
                         for e in trace["traceEvents"]
                         if e.get("ph") == "X" and e.get("cat") == "user_annotation"
                         and e.get("name", "").startswith("qwen_timing_round/"))
        marker_starts = [m[0] for m in markers]
        complete = [i for i in complete
                    if markers[bisect.bisect_right(marker_starts, float(targets[i][0][1])) - 1][1]
                    in admitted_decode_indices]
    # The last target has no following start. Keep it out of *both* the kernel
    # averages and the elapsed averages; never invent its closing boundary.
    selected = [i for i in complete if i + 1 < len(targets)]
    starts = [min(e["ts"] for e in events) for _, events in targets]
    # A profiler restart can leave one asynchronous GPU event straddling the
    # first/last boundary of a chunk.  Exclude only that contaminated round;
    # complete rounds in the same trace remain valid for stage attribution.
    work = [
        (float(event["ts"]), float(event["dur"]))
        for event in trace["traceEvents"]
        if event.get("ph") == "X"
        and event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]
    selected = [
        index
        for index in selected
        if not any(
            begin < starts[index]
            or begin + duration > starts[index + 1]
            for begin, duration in work
            if begin < starts[index + 1] and begin + duration > starts[index]
        )
    ]
    if len(selected) < min(6, len(targets) - 1):
        raise ValueError("insufficient complete target inventories")
    timing = measure_round_windows(trace["traceEvents"], starts, selected,
                                   allow_transfers=worker_boundaries)
    records, assigned, stages_by_event, layers_by_event = [], set(), {}, {}

    def append(e, scope, index, layer, stage):
        relative = e["ts"] - starts[index]
        if relative < 0 or relative + e["dur"] > starts[index + 1] - starts[index]:
            raise ValueError("stage kernel is outside its measured round")
        if id(e) in assigned:
            raise ValueError("duplicate kernel accounting")
        assigned.add(id(e))
        stages_by_event[id(e)] = stage
        layers_by_event[id(e)] = layer
        records.append((scope, index, layer, stage, e["name"], float(e["dur"])))

    for index in selected:
        _, events = targets[index]
        if len({(e["args"]["device"], e["args"]["stream"]) for e in events}) != 1:
            raise ValueError("non-serial target stream")
        points = [
            (i, projection(e["name"]))
            for i, e in enumerate(events)
            if projection(e["name"])
        ]
        if [p for _, p in points] != ["split1", "split4", "split1", "split4"] * 64:
            raise ValueError(f"backported four-projection layer inventory differs: round={index}, count={len(points)}, kinds={dict(Counter(p for _,p in points))}")
        for layer in range(64):
            a, b, c, d = [points[layer * 4 + j][0] for j in range(4)]
            kind = "Attention" if layer % 4 == 3 else "GDN"
            middle = [e["name"] for e in events[a + 1 : b]]
            if kind == "GDN":
                if (
                    sum("conv" in n for n in middle) != 1
                    or sum("recurrent" in n or "stock_gdn_scan" in n for n in middle)
                    != 1
                ):
                    raise ValueError("incomplete GDN layer inventory")
            else:
                decodes = sum("attn_decode" in n or "shared_decode" in n for n in middle)
                merges = sum("splitkv_combine" in n or "shared_merge" in n for n in middle)
                # The native short-context fallback may process two query-page
                # groups. Both real decode/merge pairs belong in the measured
                # stage; dropping these rounds would hide the slow branch.
                allowed = {1, 2} if worker_boundaries else {1}
                if decodes not in allowed or merges != decodes:
                    raise ValueError(f"incomplete attention layer inventory: round={index}, layer={layer}, decode={decodes}, merge={merges}")
            if sum("silu" in e["name"] for e in events[c + 1 : d]) != 1:
                raise ValueError("MLP activation inventory differs")
            start = 0 if layer == 0 else points[layer * 4 - 1][0] + 1
            for begin, end, region in (
                (start, a, "input"),
                (a + 1, b, "mix"),
                (b + 1, c, "post"),
                (c + 1, d, "activation"),
            ):
                for event in events[begin:end]:
                    append(
                        event,
                        "target_body",
                        index,
                        layer,
                        phase(layer, region, event["name"]),
                    )
            for point, stage in (
                (a, f"{kind} input projection"),
                (b, f"{kind} output projection"),
                (c, "MLP gate/up projection"),
                (d, "MLP down projection"),
            ):
                append(events[point], "target_body", index, layer, stage)
        for event in events[points[-1][0] + 1 :]:
            append(event, "target_body", index, None, "Final normalization/layout")
    ordinal = {
        key: i
        for scope in ("drafter", "target_vocabulary_head")
        for i, key in enumerate(sorted(k for k in groups if k[0] == scope))
    }
    for scope in ("drafter", "target_vocabulary_head"):
        if sum(k[0] == scope for k in groups) != len(targets):
            raise ValueError("incomplete head/drafter scopes")
    for key, events in groups.items():
        scope, _ = key
        if scope == "target_body":
            continue
        for event in events:
            index = bisect.bisect_right(starts, event["ts"]) - 1
            if key in ordinal and index != ordinal[key]:
                raise ValueError("non-serial head/drafter ordering")
            if index in selected:
                stage = {
                    "drafter": "Drafter",
                    "target_vocabulary_head": f"Target head ({head})",
                }.get(scope, "Other GPU bookkeeping")
                append(event, scope, index, None, stage)
    if worker_boundaries:
        for event in trace["traceEvents"]:
            if event.get("ph") != "X" or event.get("cat") not in {"gpu_memcpy", "gpu_memset"}:
                continue
            index = bisect.bisect_right(starts, event["ts"]) - 1
            if index in selected:
                append(event, "transfer", index, None, "Other GPU bookkeeping")
    stages, layers, kernels, per_round = (
        defaultdict(float),
        defaultdict(lambda: defaultdict(float)),
        defaultdict(lambda: [0, 0.0]),
        defaultdict(float),
    )
    divisor = 1000 * len(selected)
    for scope, index, layer, stage, name, duration in records:
        stages[stage] += duration / divisor
        per_round[index] += duration / 1000
        if layer is not None:
            layers[layer][stage] += duration / divisor
        row = kernels[(scope, stage, name)]
        row[0] += 1
        row[1] += duration / divisor
    if abs(sum(stages.values()) - sum(per_round.values()) / len(selected)) > 1e-8:
        raise ValueError("duration accounting failure")
    counts = Counter(record[1] for record in records)
    for row in timing["rounds"]:
        if counts[row["round"]] != row["kernel_count"] or not math.isclose(
            per_round[row["round"]], row["kernel_sum_ms"], rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("stage and elapsed timing cover different GPU work")
    result = {
        "schema": "urn:coherence:compiled-stage-timings:v2",
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "production_timing_eligible": False,
        "timing_contract": {
            "stage_metric": "gpu_activity_duration",
            "cpu_scope_time_included": False,
            "round_boundary": "first_target_gpu_kernel_start_to_next_first_target_gpu_kernel_start",
            "zero_observer_effect_proven": False,
            "limitation": (
                "GPU activity durations exclude CPU annotation and export time. "
                "The trace alone does not bound indirect profiler effects or establish production gaps."
            ),
        },
        "scope": f"Saved compiled {head} Pi decode trace. GPU dispatch sums and elapsed GPU cycles from the same bounded rounds; profiled, not uninstrumented serving latency.",
        "profile_rounds": len(targets),
        "included_rounds": selected,
        "selection": "Complete modal kernel inventory and an observed next-target start; no duration-based exclusions. Work before the first target is excluded.",
        "graph_launches_per_round": [graph_launches[k] for k, _ in targets],
        "target_head": head,
        "compilation_events": dict(Counter(
            e.get("name", "") for e in trace["traceEvents"]
            if e.get("ph") == "X" and e.get("cat") != "kernel"
            and any(marker in e.get("name", "").lower() for marker in (
                "compile_inner", "compile_fx", "graphlowering", "triton.compile", "hipmoduleload"
            ))
        )),
        "stages_ms": dict(stages),
        "layers_ms": dict(layers),
        "all_gpu_ms": sum(stages.values()),
        "per_round_gpu_ms": dict(per_round),
        "round_timing": timing,
        "kernel_groups": [
            {
                "scope": scope,
                "stage": stage,
                "kernel": name,
                "calls": values[0],
                "ms_per_round": values[1],
            }
            for (scope, stage, name), values in sorted(kernels.items())
        ],
        "omitted_inventory_rounds": [i for i in range(len(targets)) if i not in complete],
        "incomplete_inventory_reasons": incomplete,
        "omitted_unbounded_rounds": [i for i in complete if i + 1 == len(targets)],
    }
    if worker_boundaries:
        result["worker_timing"] = measure_worker_windows(
            trace["traceEvents"], targets, selected, stages_by_event, groups, head, layers_by_event
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--head", required=True, choices=["global256", "global512", "full-bf16"])
    p.add_argument("--profile-rounds", type=int)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    raw = a.trace.read_bytes()
    if a.trace.suffix == ".gz":
        raw = gzip.decompress(raw)
    value = analyze(raw, a.head, a.profile_rounds)
    value["trace_file_sha256"] = hashlib.sha256(a.trace.read_bytes()).hexdigest()
    value["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                **{
                    k: value[k]
                    for k in ("included_rounds", "all_gpu_ms", "per_round_gpu_ms")
                },
                "round_timing_ms": value["round_timing"]["mean"],
            }
        )
    )


if __name__ == "__main__":
    main()
